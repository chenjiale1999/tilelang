"""Parallel-softmax FlashAttention forward for the P1/P2/P3 closed loop.

This is the *parallel* form, not the manual ``T.simd`` form in ``core.py``:

- QK^T and PV are TileOp GEMMs;
- row max/sum are TileOp reductions;
- the softmax elementwise pass and the direct NZ probability store are written
  as ``T.Parallel`` inside ``T.SimdVF``;
- ``P_nz_ub[i, j] = P_ub[i, j]`` is the layout-aware store that P1 must carry
  to P2/P3 for ``vsstb``.

The online-softmax state (m, l, alpha, O) is kept in UB and updated once per KV
block. This is the intended FA closed-loop case for P2/P3; it is not part of the
default pipeline yet.

Known existing issue (not this file's logic): the legacy
``AscendSimdVFLowerParallel`` still matches ``tir.exp`` after the TIR->TIRX
migration, while TIRX emits ``tirx.exp``. Therefore the legacy fallback for this
parallel FA reports ``unsupported expr: T.exp`` until that pass is migrated.
"""

import math

import tilelang
import tilelang.ascend.language as T
from tilelang.layout import make_ascend_compact_nz_layout

PARALLEL_FA_BR = 128
PARALLEL_FA_BC = 128
PARALLEL_FA_D = 64


def parallel_flash_attention_fwd(
    q_len=PARALLEL_FA_BR,
    kv_len=4 * PARALLEL_FA_BC,
    head_dim=PARALLEL_FA_D,
    *,
    dtype="bfloat16",
    accum_dtype="float32",
    softmax_scale=None,
):
    """Build a parallel-softmax FlashAttention forward kernel.

    Shapes: Q[BR,D], K/V[kv_len,D], O[BR,D], with q_len == BR and
    kv_len a multiple of BC_TILE. One block owns one query tile and iterates
    all KV blocks with online softmax.
    """
    BR = q_len
    D = head_dim
    BC_TILE = PARALLEL_FA_BC
    assert BR % 16 == 0
    assert kv_len % BC_TILE == 0
    assert D % 16 == 0
    NUM_KV = kv_len // BC_TILE
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)

    @T.prim_func
    def main(
        Q: T.Tensor((BR, D), dtype),
        K: T.Tensor((kv_len, D), dtype),
        V: T.Tensor((kv_len, D), dtype),
        O: T.Tensor((BR, D), accum_dtype),
    ):
        with T.Kernel(1):
            Q_shared = T.alloc_l1((BR, D), dtype)
            K_shared = T.alloc_l1((BC_TILE, D), dtype)
            V_shared = T.alloc_l1((D, BC_TILE), dtype)
            P_shared = T.alloc_l1((BR, BC_TILE), dtype)

            qk_a_l0 = T.alloc_l0a((BR, D), dtype)
            qk_b_l0 = T.alloc_l0b((BC_TILE, D), dtype)
            pv_a_l0 = T.alloc_l0a((BR, BC_TILE), dtype)
            pv_b_l0 = T.alloc_l0b((D, BC_TILE), dtype)
            qk_acc_l0c = T.alloc_l0c((BR, BC_TILE), accum_dtype)
            pv_acc_l0c = T.alloc_l0c((BR, D), accum_dtype)

            S_ub = T.alloc_shared((BR, BC_TILE), accum_dtype)
            P_ub = T.alloc_shared((BR, BC_TILE), accum_dtype)
            P_nz_ub = T.alloc_shared((BR + 1, BC_TILE), dtype)
            T.annotate_layout({P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)})
            O_tmp_ub = T.alloc_shared((BR, D), accum_dtype)
            O_ub = T.alloc_shared((BR, D), accum_dtype)

            m_old = T.alloc_shared((BR,), accum_dtype)
            m_new = T.alloc_shared((BR,), accum_dtype)
            l_old = T.alloc_shared((BR,), accum_dtype)
            l_cur = T.alloc_shared((BR,), accum_dtype)
            alpha = T.alloc_shared((BR,), accum_dtype)

            T.copy(Q, Q_shared)

            # Online-softmax state.
            with T.SimdVF():
                for i in T.Parallel(BR):
                    m_old[i] = -T.float32(3.0e38)
                    l_old[i] = T.float32(0.0)
            with T.SimdVF():
                for i, j in T.Parallel(BR, D):
                    O_ub[i, j] = T.float32(0.0)

            for kv in T.serial(NUM_KV):
                T.copy(K[kv * BC_TILE : (kv + 1) * BC_TILE, 0:D], K_shared)
                T.copy(V[kv * BC_TILE : (kv + 1) * BC_TILE, 0:D], V_shared, transpose=True)

                # QK^T
                T.copy(Q_shared, qk_a_l0)
                T.copy(K_shared, qk_b_l0)
                T.gemm(qk_a_l0, qk_b_l0, qk_acc_l0c, transpose_B=True, clear_accum=True)
                T.copy(qk_acc_l0c, S_ub)

                # Scale and row max for this KV block.
                with T.SimdVF():
                    for i, j in T.Parallel(BR, BC_TILE):
                        S_ub[i, j] = S_ub[i, j] * scale
                # Merge running max: m_new = max(m_old, rowmax(S)).
                T.copy(m_old, m_new)
                with T.SimdVF():
                    T.reduce_max(S_ub, m_new, dim=1, clear=False)

                # Rescale factor for the online-softmax update.
                with T.SimdVF():
                    for i in T.Parallel(BR):
                        alpha[i] = T.exp(m_old[i] - m_new[i])

                # Parallel softmax and the layout-aware direct NZ store.
                with T.SimdVF():
                    for i, j in T.Parallel(BR, BC_TILE):
                        P_ub[i, j] = T.exp(S_ub[i, j] - m_new[i])
                        P_nz_ub[i, j] = T.cast(P_ub[i, j], dtype)
                with T.SimdVF():
                    T.reduce_sum(P_ub, l_cur, dim=1, clear=True)

                # l_new = l_old * alpha + rowsum(P)
                with T.SimdVF():
                    for i in T.Parallel(BR):
                        l_old[i] = l_old[i] * alpha[i] + l_cur[i]
                T.copy(m_new, m_old)

                # Feed the NZ probabilities to the second GEMM.
                T.copy(P_nz_ub[0:BR, 0:BC_TILE], P_shared[0:BR, 0:BC_TILE])
                T.copy(P_shared, pv_a_l0)
                T.copy(V_shared, pv_b_l0)
                T.gemm(pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_B=True, clear_accum=True)
                T.copy(pv_acc_l0c, O_tmp_ub)

                # O_new = O_old * alpha + P @ V
                with T.SimdVF():
                    for i, j in T.Parallel(BR, D):
                        O_ub[i, j] = O_ub[i, j] * alpha[i] + O_tmp_ub[i, j]

            # Normalize by the final row sum.
            with T.SimdVF():
                for i, j in T.Parallel(BR, D):
                    O_ub[i, j] = O_ub[i, j] / l_old[i]
            T.copy(O_ub, O)

    return main


@tilelang.jit
def parallel_fa_compile_only():
    """Compile-only entry for ``run_lower_trace.sh`` (first JIT kernel is used)."""
    return parallel_flash_attention_fwd()


def ref_parallel_flash_attention(q, k, v, scale=None):
    """Reference for a single query tile."""
    import torch

    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    p = torch.softmax(s, dim=-1)
    return p.to(v.dtype) @ v.float()
