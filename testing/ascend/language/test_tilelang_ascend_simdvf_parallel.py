"""Self-contained device regression for the P1->P2->P3 ``T.Parallel`` lowering.

Every case is a ``T.SimdVF`` kernel whose ``T.Parallel`` nest is lifted by P1,
planned by P2 and emitted by P3.  The test first asserts that the generated CCE
really contains the expected vector intrinsic (so a silent fallback to the
legacy scalar path cannot pass), then runs the kernel on an Ascend NPU and
compares it with a torch golden computed from the *scalar* semantics of the
source program.

The file is ordered from simple to complex, and the sections follow that ramp:

1. **1-D elementwise** — whole chunks, in-place read-modify-write, two VFs in
   one kernel, ``T.max``/``T.min`` NaN and ``+-0`` contract, tail chunks, 8-bit
   whole chunks / byte tails / ``i8 <-> u8`` reinterpret;
2. **N-D nests** — two/three-axis collapse, outer-axis broadcast, flattened
   ``i // N, i % N`` indexing;
3. **index transformations** — stride-2 loads *and* stores, reversal
   (``PERMUTE``), indirect runtime gather, transposed scatter;
4. **guarded stores** — single-arm predication (also combined with a tail),
   lane-uniform ``update_mask``, same-address ``vsel``, nested ``else if``,
   divergent two-arm stores, ``and``/``or``/``not`` predicates, mask
   ``==``/``!=``, and two arms writing disjoint addresses of the *same* buffer
   (guard-aware legality);
5. **reductions and serial shells** — row ``max``/``min``, a row reduction
   whose tail chunk is masked, shell reductions, unrolled / kept /
   runtime-bound shells, a serial shell *around* the nest (VST->VLD barrier),
   sibling reduction shells (loop fission);
6. **cast stores** — narrowing (bf16 / i32 / i16 / fp16->fp8), same-width
   signedness reinterpret, widening (``UNPK_*``, including an 8-bit widening
   tail) and the full ``vcvt`` pair matrix;
7. **NZ layout and pack2** — annotated compact-NZ stores, a legal
   elementwise-NZ + row-reduction pair of nests, and the paired ``vld_x2`` +
   ``vor`` + ``vsstb`` form.
"""

import pytest
import torch
import torch_npu

import tilelang
import tilelang.testing
from tilelang.ascend import language as T
from tilelang.layout import make_ascend_compact_nz_layout

F32 = "float32"
NAN = float("nan")
INF = float("inf")

M, N = 8, 128


def _compile(func, out_idx=-1):
    """Compile a kernel and return it together with its generated CCE source.

    ``out_idx=None`` keeps every parameter as a caller-provided buffer, which is
    how an in-place update (and the read-back of an untouched lane) is expressed.
    """
    kernel = tilelang.compile(func, target="ascend", out_idx=out_idx)
    return kernel, kernel.get_kernel_source()


def _assert_close(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected, equal_nan=True, rtol=0, atol=0)


def _cycle(values, n, dtype=torch.float32):
    """Repeat ``values`` to exactly ``n`` elements (no length divisibility assumption)."""
    return torch.tensor([values[i % len(values)] for i in range(n)], dtype=dtype)


def _nz_physical(values, c0):
    """Logical ``(rows, cols)`` -> compact-NZ physical order ``[cols/C0, rows, C0]``.

    A UB buffer annotated with ``make_ascend_compact_nz_layout`` stores its
    elements in that order, and the UB->GM copy is a raw contiguous DMA, so the
    device output holds the physical permutation of the logical result.
    """
    rows, cols = values.shape
    return values.reshape(rows, cols // c0, c0).permute(1, 0, 2).reshape(rows, cols)


# ---------------------------------------------------------------------------
# 1-D elementwise: whole chunks, tails, multiple VFs and 8-bit stores
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_multiple_independent_stores():
    @T.prim_func
    def main(A: T.Tensor((N,), F32), C: T.Tensor((N,), F32), D: T.Tensor((N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            c_ub = T.alloc_shared((N,), F32)
            d_ub = T.alloc_shared((N,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    c_ub[i] = a_ub[i] + 1.0
                    d_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    kernel, source = _compile(main, out_idx=[-2, -1])
    assert "simd_inst::vadds(" in source and "simd_inst::vmuls(" in source
    assert source.count("simd_inst::vsts_norm(") == 2

    a = (torch.arange(N, dtype=torch.float32) % 17) - 8.0
    c_npu, d_npu = kernel(a.npu())
    _assert_close(c_npu, a + 1.0)
    _assert_close(d_npu, a * 2.0)


@tilelang.testing.requires_ascend
def test_parallel_in_place_update_without_a_shell():
    """``c[i] = c[i] + a[i]``: one vector pass over the buffer it reads.

    The chunk program is not wrapped in a scalar shell, so the emitter must not
    add a VST->VLD barrier: the chunks are independent and each ``vsts`` closes
    its own value chain.
    """
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), CIN: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(CIN, c_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = c_ub[i] + a_ub[i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vlds_norm<" in source
    assert "simd_inst::vadd(" in source
    assert "simd_inst::vsts_norm(" in source
    assert "simd_inst::mem_bar(" not in source, "no scalar shell, no barrier"

    a = (torch.arange(n, dtype=torch.float32) % 13) - 6.0
    c_in = (torch.arange(n, dtype=torch.float32) % 5) - 2.0
    _assert_close(kernel(a.npu(), c_in.npu()), c_in + a)


# Every row is one operand pair; the mixed NaN/+-0 rows are the contract.
_MAXMIN_A = [
    NAN, 1.0, NAN, 1.0,          # NaN left / right / both
    0.0, -0.0, 0.0, -0.0,        # +-0 in both orders
    INF, -INF, 1.0, -1.0,        # infinities and ordinary values
    0.0, 1.0, NAN, NAN,
]
_MAXMIN_B = [
    1.0, NAN, NAN, 1.0,
    -0.0, 0.0, 0.0, -0.0,
    -INF, INF, 1.0, -1.0,
    0.0, 0.0, 2.0, NAN,
]


def _maxmin_kernel(n, kind):
    @T.prim_func
    def main(A: T.Tensor((n,), F32), B: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            b_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if kind == "max":
                        c_ub[i] = T.max(a_ub[i], b_ub[i])
                    else:
                        c_ub[i] = T.min(a_ub[i], b_ub[i])
            T.copy(c_ub, C)

    return main


@pytest.mark.parametrize("kind", ["max", "min"])
@tilelang.testing.requires_ascend
def test_parallel_max_min_nan_and_signed_zero(kind):
    n = 128
    a = torch.tensor(_MAXMIN_A * (n // len(_MAXMIN_A)), dtype=torch.float32)
    b = torch.tensor(_MAXMIN_B * (n // len(_MAXMIN_B)), dtype=torch.float32)

    kernel, source = _compile(_maxmin_kernel(n, kind))
    assert f"simd_inst::v{kind}(" in source

    actual = kernel(a.npu(), b.npu()).cpu()
    # Pinned contract: the CANN ``asc_max`` / ``asc_min`` instructions, not the
    # scalar ``a > b ? a : b`` ternary.  A NaN operand propagates; the sign of a
    # zero result only survives when *both* operands carry it (``max`` keeps
    # ``-0`` only for two ``-0``; ``min`` keeps ``+0`` only for two ``+0``).
    nan = torch.isnan(a) | torch.isnan(b)
    both_zero = (a == 0) & (b == 0)
    expected = torch.full_like(a, float("nan"))
    if kind == "max":
        expected[~nan] = torch.maximum(a[~nan], b[~nan])
        keep_neg = both_zero & torch.signbit(a) & torch.signbit(b)
        expected[both_zero & ~keep_neg] = 0.0
        expected[keep_neg] = -0.0
    else:
        expected[~nan] = torch.minimum(a[~nan], b[~nan])
        keep_pos = both_zero & ~torch.signbit(a) & ~torch.signbit(b)
        expected[both_zero & ~keep_pos] = -0.0
        expected[keep_pos] = 0.0

    _assert_close(actual, expected)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros])), (
        "the sign of +-0 must follow the CANN max/min contract"
    )


@tilelang.testing.requires_ascend
def test_parallel_elementwise_tail():
    n = 1000  # 15 full float32 chunks (64 lanes) + a 40-lane tail

    @T.prim_func
    def main(A: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "update_mask_b32" in source, "the tail chunk must be predicated"

    a = torch.arange(n, dtype=torch.float32)
    _assert_close(kernel(a.npu()), a * 2.0)


@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@tilelang.testing.requires_ascend
def test_parallel_eight_bit_full_chunk(dtype):
    n = 256
    torch_dtype = getattr(torch, dtype)

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), dtype)
            c_ub = T.alloc_shared((n,), dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vlds_norm<" in source
    assert "simd_inst::vsts_norm(" in source
    assert "simd_inst::vgather2(" not in source, "a whole byte register uses the NORM path"

    if dtype == "uint8":
        a = (torch.arange(n, dtype=torch.int32) % 256).to(torch_dtype)
    else:
        a = (torch.arange(n, dtype=torch.int32) % 251 - 125).to(torch_dtype)
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.uint8), a.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@tilelang.testing.requires_ascend
def test_parallel_eight_bit_elementwise_tail(dtype):
    """An 8-bit tail is gathered into 16-bit lanes and written with ``PK_B16``.

    A ``NORM_B8`` store would reinterpret those 16-bit lanes as interleaved
    bytes and overrun the buffer (the row is shorter than one byte register).
    """
    n = 100  # one 256-byte chunk, so the whole vector is a 100-lane tail
    torch_dtype = getattr(torch, dtype)

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), dtype)
            c_ub = T.alloc_shared((n,), dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2(" in source, "the byte tail load is a widening gather"
    assert "vsts_pack_b16" in source, "the widened value is narrowed back by the pack store"

    if dtype == "uint8":
        a = (torch.arange(n, dtype=torch.int32) % 256).to(torch_dtype)
    else:
        a = (torch.arange(n, dtype=torch.int32) % 251 - 125).to(torch_dtype)
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.uint8), a.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("src_dtype", ["uint8", "int8"])
@tilelang.testing.requires_ascend
def test_parallel_eight_bit_same_width_reinterpret_tail(src_dtype):
    """``i8 <-> u8`` keeps the byte gather: bit identity through the pack store."""
    n = 100
    dst_dtype = "int8" if src_dtype == "uint8" else "uint8"
    src_torch = getattr(torch, src_dtype)

    @T.prim_func
    def main(A: T.Tensor((n,), src_dtype), C: T.Tensor((n,), dst_dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), src_dtype)
            c_ub = T.alloc_shared((n,), dst_dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast(dst_dtype, a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2(" in source
    assert "vsts_pack_b16" in source

    a = (torch.arange(n, dtype=torch.int32) % 256).to(src_torch)
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.uint8), a.view(torch.uint8), rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_parallel_two_vfs_in_one_kernel():
    """Two ``T.SimdVF`` blocks in one kernel are planned independently.

    The first VF rewrites ``a_ub`` in place, the second consumes it, so the test
    also pins that the two emitted programs run in source order.
    """
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), B: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            b_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    a_ub[i] = a_ub[i] * 2.0
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + b_ub[i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vmuls(" in source, "the first VF scales in place"
    assert "simd_inst::vadd(" in source, "the second VF adds the scaled value"
    assert source.count("simd_inst::vsts_norm(") == 2, "one store per VF (its 2-chunk loop)"

    a = (torch.arange(n, dtype=torch.float32) % 9) - 4.0
    b = (torch.arange(n, dtype=torch.float32) % 5) - 2.0
    _assert_close(kernel(a.npu(), b.npu()), a * 2.0 + b)


# ---------------------------------------------------------------------------
# N-D nests: collapse, broadcast and flattened indexing
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_two_axis_elementwise():
    @T.prim_func
    def main(A: T.Tensor((M, N), F32), C: T.Tensor((M, N), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            c_ub = T.alloc_shared((M, N), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[i, j] = a_ub[i, j] * 2.0 + 1.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vmuls(" in source and "simd_inst::vadds(" in source
    assert "simd_inst::vgather2(" not in source, "both axes are affine and contiguous"

    a = (torch.arange(M * N, dtype=torch.float32) % 11) - 5.0
    _assert_close(kernel(a.reshape(M, N).npu()), (a * 2.0 + 1.0).reshape(M, N))


@tilelang.testing.requires_ascend
def test_parallel_three_axis_collapse():
    """The P1 "collapse" is a region collapse, not a full flattening.

    ``T.Parallel(2, 4, 64)`` becomes one region with ``axes = [i, j, k]``; the
    vector axis is the innermost one, so the emitted program keeps ``i`` and
    ``j`` as serial loops and chunks only the 64 lanes of ``k``. Flattening all
    512 elements into 8 chunks is not what the name promises, and not what the
    planner does.
    """
    shape = (2, 4, 64)

    @T.prim_func
    def main(A: T.Tensor(shape, F32), C: T.Tensor(shape, F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared(shape, F32)
            c_ub = T.alloc_shared(shape, F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j, k in T.Parallel(2, 4, 64):
                    c_ub[i, j, k] = a_ub[i, j, k] + 1.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vadds(" in source
    assert "update_mask" not in source, "2 * 4 * 64 is a whole number of chunks"

    a = torch.arange(2 * 4 * 64, dtype=torch.float32).reshape(shape)
    _assert_close(kernel(a.npu()), a + 1.0)


@tilelang.testing.requires_ascend
def test_parallel_outer_axis_broadcast():
    @T.prim_func
    def main(A: T.Tensor((M, N), F32), B: T.Tensor((M,), F32), C: T.Tensor((M, N), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            b_ub = T.alloc_shared((M,), F32)
            c_ub = T.alloc_shared((M, N), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[i, j] = a_ub[i, j] + b_ub[i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vadds(" in source, "the outer operand is uniform inside a chunk"

    a = (torch.arange(M * N, dtype=torch.float32) % 13) - 6.0
    b = torch.arange(M, dtype=torch.float32) * 0.5
    _assert_close(kernel(a.reshape(M, N).npu(), b.npu()), a.reshape(M, N) + b[:, None])


@tilelang.testing.requires_ascend
def test_parallel_flattened_divmod_index():
    @T.prim_func
    def main(A: T.Tensor((M, N), F32), C: T.Tensor((M * N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            c_ub = T.alloc_shared((M * N,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(M * N):
                    c_ub[i] = a_ub[i // N, i % N] + 1.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vadds(" in source

    a = (torch.arange(M * N, dtype=torch.float32) % 7) - 3.0
    _assert_close(kernel(a.reshape(M, N).npu()), a + 1.0)


# ---------------------------------------------------------------------------
# index transformations: stride, permute, gather and scatter
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_strided_load():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((2 * n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2 * n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[2 * i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2(" in source, "a stride-2 load uses the per-lane index vector"

    a = torch.arange(2 * n, dtype=torch.float32)
    _assert_close(kernel(a.npu()), a[0::2])


@tilelang.testing.requires_ascend
def test_parallel_strided_store():
    """``c[2i] = a[i] * 2`` scatters with a per-lane index vector.

    Only the even destinations are written, so every odd lane must keep the
    value it had before the kernel.
    """
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), CIN: T.Tensor((2 * n,), F32), C: T.Tensor((2 * n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((2 * n,), F32)
            T.copy(A, a_ub)
            T.copy(CIN, c_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[2 * i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vscatter(" in source, "a stride-2 store is a per-lane scatter"
    assert "simd_inst::vsts_norm(" not in source

    a = (torch.arange(n, dtype=torch.float32) % 11) - 5.0
    c_in = torch.full((2 * n,), -9.0, dtype=torch.float32)
    expected = c_in.clone()
    expected[0::2] = a * 2.0
    _assert_close(kernel(a.npu(), c_in.npu()), expected)


@tilelang.testing.requires_ascend
def test_parallel_reversal_is_a_permute_gather():
    @T.prim_func
    def main(A: T.Tensor((N,), F32), C: T.Tensor((N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            c_ub = T.alloc_shared((N,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    c_ub[i] = a_ub[N - 1 - i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2(" in source, "the reversed index is a permutation gather"

    a = torch.arange(N, dtype=torch.float32)
    _assert_close(kernel(a.npu()), a.flip(0))


@tilelang.testing.requires_ascend
def test_parallel_indirect_gather():
    """A runtime index declares the whole owner; the values must still be exact."""
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), B: T.Tensor((n,), "int32"), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            b_ub = T.alloc_shared((n,), "int32")
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[b_ub[i]]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2" in source

    a = torch.arange(n, dtype=torch.float32)
    b = torch.randperm(n).to(torch.int32)
    _assert_close(kernel(a.npu(), b.npu()), a[b.long()])


@tilelang.testing.requires_ascend
def test_parallel_transposed_store():
    @T.prim_func
    def main(A: T.Tensor((M, N), F32), C: T.Tensor((N, M), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            c_ub = T.alloc_shared((N, M), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[j, i] = a_ub[i, j]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vscatter(" in source, "the transposed store is a per-lane scatter"

    a = (torch.arange(M * N, dtype=torch.float32) % 9) - 4.0
    _assert_close(kernel(a.reshape(M, N).npu()), a.reshape(M, N).t())


# ---------------------------------------------------------------------------
# guarded stores: single arm, select, divergent arms and masks
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_predicated_single_arm_store():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(C, c_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if a_ub[i] > 0.0:
                        c_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main, out_idx=None)

    # NaN / -0 / +0 / infinities around the boundary (8 values, so it divides 128).
    a = torch.tensor(
        [NAN, -1.0, -0.0, 0.0, 1.0, -2.0, INF, -INF] * (n // 8),
        dtype=torch.float32,
    )
    c_in = torch.arange(n, dtype=torch.float32) * 0.5
    c_npu = c_in.npu()
    kernel(a.npu(), c_npu)
    assert "vcmps_gt" in source, "the guard must be a vector compare"
    # The store is masked, so inactive lanes keep the old destination value.
    expected = torch.where(a > 0.0, a * 2.0, c_in)
    _assert_close(c_npu, expected)


@tilelang.testing.requires_ascend
def test_parallel_tail_with_a_statement_predicate():
    """The tail mask and a statement predicate are AND-ed into one store mask.

    The whole chunks use ``vcmps`` as the store mask directly; the tail chunk
    loads through a masked gather and combines the two masks with ``pand``.
    """
    n = 300  # 4 whole float32 chunks (64 lanes) + a 44-lane tail

    @T.prim_func
    def main(A: T.Tensor((n,), F32), CIN: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(CIN, c_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if a_ub[i] > 0.0:
                        c_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcmps_gt" in source
    assert "update_mask" in source
    assert "simd_inst::vgather2(" in source, "the tail chunk is a masked gather load"
    assert "simd_inst::pand(" in source, "statement predicate AND-ed with the tail mask"

    a = _cycle([-3.0, -0.0, 0.0, 1.0, 2.5, -1.5, 4.0, -2.0], n)
    c_in = (torch.arange(n, dtype=torch.float32) % 7) - 3.0
    expected = torch.where(a > 0.0, a * 2.0, c_in)
    _assert_close(kernel(a.npu(), c_in.npu()), expected)


@tilelang.testing.requires_ascend
def test_parallel_lane_uniform_condition_broadcasts_the_mask():
    @T.prim_func
    def main(A: T.Tensor((N,), F32), B: T.Tensor((1,), F32), C: T.Tensor((N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            b_ub = T.alloc_shared((1,), F32)
            c_ub = T.alloc_shared((N,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(C, c_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    if b_ub[0] > 0.0:
                        c_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main, out_idx=None)
    # A lane-uniform guard must not become a scalar compare/select inside
    # ``__simd_vf__`` (bisheng: "Unsupported scalar instruction in AIV loop").
    # It is a vector predicate instead: broadcast one operand and compare it.
    # The comparison is canonicalised (literal first), so ``b[0] > 0.0`` arrives
    # as ``0.0 < b[0]`` and lowers to ``vcmps_lt(vdup(0.0f), b[0])``: assert the
    # instruction family, not the authored direction.
    assert "simd_inst::vdup<" in source, "the scalar guard is broadcast, not scalarized"
    assert "simd_inst::vcmps_" in source, "the guard becomes a vector mask"
    assert "update_mask" not in source, "no scalar-select mask inside the VF"

    a = (torch.arange(N, dtype=torch.float32) % 5) - 2.0
    c_in = torch.full((N,), -7.0, dtype=torch.float32)
    positive = c_in.clone().npu()
    kernel(a.npu(), torch.ones(1, dtype=torch.float32).npu(), positive)
    _assert_close(positive, a * 2.0)

    negative = c_in.clone().npu()
    kernel(a.npu(), torch.full((1,), -1.0, dtype=torch.float32).npu(), negative)
    _assert_close(negative, c_in)


@tilelang.testing.requires_ascend
def test_parallel_two_arm_same_address_becomes_a_select():
    @T.prim_func
    def main(A: T.Tensor((N,), F32), C: T.Tensor((N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            c_ub = T.alloc_shared((N,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    if a_ub[i] > 0.0:
                        c_ub[i] = a_ub[i] * 2.0
                    else:
                        c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vsel(" in source, "both arms share an address: one register select"
    assert "simd_inst::vcmps_gt" in source

    a = _cycle([-3.0, -0.0, 0.0, 1.0, 2.5, -1.5], N)
    _assert_close(kernel(a.npu()), torch.where(a > 0.0, a * 2.0, a + 1.0))


@tilelang.testing.requires_ascend
def test_parallel_nested_else_if():
    @T.prim_func
    def main(A: T.Tensor((N,), F32), C: T.Tensor((N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            c_ub = T.alloc_shared((N,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    if a_ub[i] > 0.0:
                        c_ub[i] = 1.0
                    else:
                        if a_ub[i] < -1.0:
                            c_ub[i] = 2.0
                        else:
                            c_ub[i] = 3.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert source.count("simd_inst::vsel(") == 2, "the nested else-if is two selects"

    a = _cycle([-3.0, -1.5, -1.0, 0.0, 1.0], N)
    expected = torch.where(a > 0.0, 1.0, torch.where(a < -1.0, 2.0, 3.0))
    _assert_close(kernel(a.npu()), expected)


@tilelang.testing.requires_ascend
def test_parallel_divergent_two_arm_store():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), C: T.Tensor((n,), F32), D: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            d_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(C, c_ub)
            T.copy(D, d_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if a_ub[i] > 0.0:
                        c_ub[i] = a_ub[i]
                    else:
                        d_ub[i] = 1.0
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    kernel, source = _compile(main, out_idx=None)
    assert "simd_inst::pnot(" in source, "each arm keeps its own complementary mask"

    a = torch.tensor([NAN, -1.0, 0.0, 1.0, 2.0, -3.0, 4.0, -4.0] * (n // 8), dtype=torch.float32)
    c_in = torch.full((n,), -7.0, dtype=torch.float32)
    d_in = torch.full((n,), -9.0, dtype=torch.float32)
    c_npu, d_npu = c_in.npu(), d_in.npu()
    kernel(a.npu(), c_npu, d_npu)

    branch = a > 0.0
    _assert_close(c_npu, torch.where(branch, a, c_in))
    _assert_close(d_npu, torch.where(branch, d_in, torch.ones_like(d_in)))


@tilelang.testing.requires_ascend
def test_parallel_condition_and_or_not():
    @T.prim_func
    def main(
        A: T.Tensor((N,), F32),
        B: T.Tensor((N,), F32),
        C1: T.Tensor((N,), F32),
        C2: T.Tensor((N,), F32),
        C3: T.Tensor((N,), F32),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            b_ub = T.alloc_shared((N,), F32)
            c1_ub = T.alloc_shared((N,), F32)
            c2_ub = T.alloc_shared((N,), F32)
            c3_ub = T.alloc_shared((N,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.fill(c1_ub, 0.0)
            T.fill(c2_ub, 0.0)
            T.fill(c3_ub, 0.0)
            with T.SimdVF():
                for i in T.Parallel(N):
                    if (a_ub[i] > 0.0) and (b_ub[i] > 0.0):
                        c1_ub[i] = 1.0
                    if (a_ub[i] > 0.0) or (b_ub[i] > 0.0):
                        c2_ub[i] = 2.0
                    if not (a_ub[i] > 0.0):
                        c3_ub[i] = 3.0
            T.copy(c1_ub, C1)
            T.copy(c2_ub, C2)
            T.copy(c3_ub, C3)

    kernel, source = _compile(main, out_idx=[-3, -2, -1])
    assert "simd_inst::pand(" in source
    assert "simd_inst::por(" in source
    # ``not (a > 0)`` is folded into ``a <= 0`` by ``Simplify`` before lowering,
    # so the complementary mask shows up as a comparison instead of a ``pnot``.
    assert "simd_inst::pnot(" in source or "simd_inst::vcmps_le(" in source

    a = _cycle([-1.0, -0.0, 0.0, 1.0, 2.0, -3.0], N)
    b = _cycle([1.0, 2.0, -1.0, -2.0, 3.0, 0.0], N)
    c1, c2, c3 = kernel(a.npu(), b.npu())
    pa, pb = a > 0.0, b > 0.0
    _assert_close(c1, torch.where(pa & pb, 1.0, torch.zeros_like(a)))
    _assert_close(c2, torch.where(pa | pb, 2.0, torch.zeros_like(a)))
    _assert_close(c3, torch.where(~pa, 3.0, torch.zeros_like(a)))


def _mask_eq_kernel(n, op):
    @T.prim_func
    def main(A: T.Tensor((n,), F32), B: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            b_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(C, c_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if op == "eq":
                        if (a_ub[i] > 0.0) == (b_ub[i] > 0.0):
                            c_ub[i] = 1.0
                    else:
                        if (a_ub[i] > 0.0) != (b_ub[i] > 0.0):
                            c_ub[i] = 1.0
            T.copy(c_ub, C)

    return main


@pytest.mark.parametrize("op", ["eq", "ne"])
@tilelang.testing.requires_ascend
def test_parallel_mask_equality(op):
    n = 128
    kernel, source = _compile(_mask_eq_kernel(n, op), out_idx=None)
    assert "simd_inst::pxor(" in source, "mask ==/!= is emitted through the predicate truth table"
    assert "simd_inst::vcmp(" not in source

    a = torch.tensor([NAN, 1.0, -1.0, 0.0, 2.0, -0.0, NAN, 3.0] * (n // 8), dtype=torch.float32)
    b = torch.tensor([1.0, NAN, 1.0, -0.0, -2.0, 0.0, NAN, 0.0] * (n // 8), dtype=torch.float32)
    c_in = torch.full((n,), -5.0, dtype=torch.float32)
    c_npu = c_in.npu()
    kernel(a.npu(), b.npu(), c_npu)

    same = (a > 0.0) == (b > 0.0)
    if op == "ne":
        same = ~same
    _assert_close(c_npu, torch.where(same, torch.ones_like(c_in), c_in))


@tilelang.testing.requires_ascend
def test_parallel_same_buffer_disjoint_arm_addresses():
    """``c[2i]`` / ``c[2i+1]`` never collide, so the guard-aware prover lifts it."""
    @T.prim_func
    def main(A: T.Tensor((N,), F32), C: T.Tensor((2 * N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            c_ub = T.alloc_shared((2 * N,), F32)
            T.copy(A, a_ub)
            T.copy(C, c_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    if a_ub[i] > 0.0:
                        c_ub[2 * i] = 1.0
                    else:
                        c_ub[2 * i + 1] = 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main, out_idx=None)
    assert "simd_inst::vscatter(" in source
    assert "simd_inst::pnot(" in source, "the else arm uses the complementary mask"

    a = _cycle([-1.0, 0.0, 1.0, -2.0, 3.0, -0.0], N)
    c_in = torch.full((2 * N,), -9.0, dtype=torch.float32)
    c_npu = c_in.clone().npu()
    kernel(a.npu(), c_npu)

    expected = c_in.clone()
    positive = a > 0.0
    expected[0::2][positive] = 1.0
    expected[1::2][~positive] = 2.0
    _assert_close(c_npu, expected)


# ---------------------------------------------------------------------------
# reductions and serial shells
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["max", "min"])
@tilelang.testing.requires_ascend
def test_parallel_row_reduction(kind):
    rows, k = 8, 128
    identity = -3.4028234663852886e38 if kind == "max" else 3.4028234663852886e38

    @T.prim_func
    def main(A: T.Tensor((rows, k), F32), D: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, k), F32)
            d_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            T.fill(d_ub, identity)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for j in T.serial(k):
                        if kind == "max":
                            d_ub[i] = T.max(d_ub[i], a_ub[i, j])
                        else:
                            d_ub[i] = T.min(d_ub[i], a_ub[i, j])
            T.copy(d_ub, D)

    kernel, source = _compile(main)
    intrinsic = "simd_inst::vcmax(" if kind == "max" else "simd_inst::vcmin("
    assert intrinsic in source, "the row accumulator is a cross-lane reduction"

    a = (torch.arange(rows * k, dtype=torch.float32) % 37) - 18.0
    expected = a.reshape(rows, k).max(dim=1).values if kind == "max" else a.reshape(rows, k).min(dim=1).values
    _assert_close(kernel(a.reshape(rows, k).npu()), expected)


@tilelang.testing.requires_ascend
def test_parallel_row_reduction_with_a_tail():
    """A row that is not a whole number of chunks still reduces in registers.

    The whole chunk is a ``vcadd``; the tail chunk is a masked gather followed
    by a ``vcadd`` under the tail mask, and both partial sums are combined on
    lane 0 before the single ``ONEPT`` store.

    The row stride is kept 32B-aligned (104 f32 = 416B = 13 blocks): block
    ``vlds_norm``/``vsts_norm`` instructions need a 32B-aligned base, and a
    row stride that is not a multiple of 32B (e.g. 100 f32 = 400B) is rejected
    by P2 (``plan.alignment``).
    """
    rows, k = 8, 104  # 64 + 40 lanes, 416B per row

    @T.prim_func
    def main(A: T.Tensor((rows, k), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, k), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            T.fill(c_ub, 0.0)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for j in T.serial(k):
                        c_ub[i] = c_ub[i] + a_ub[i, j]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert source.count("simd_inst::vcadd(") == 2, "one whole-chunk and one tail reduction"
    assert "update_mask" in source, "the tail chunk reduces under a partial mask"
    assert "simd_inst::vgather2(" in source, "the tail load is a masked gather"

    a = (torch.arange(rows * k, dtype=torch.int32) % 7).to(torch.float32)
    _assert_close(kernel(a.reshape(rows, k).npu()), a.reshape(rows, k).sum(dim=1))


@tilelang.testing.requires_ascend
def test_parallel_serial_shell_reduction():
    rows, k = 8, 64

    @T.prim_func
    def main(A: T.Tensor((rows, k), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, k), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            T.fill(c_ub, 0.0)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for j in T.serial(k):
                        c_ub[i] = c_ub[i] + a_ub[i, j]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcadd(" in source, "the shell accumulator is a cross-lane reduction"

    # Small integers keep the sequential and vector sums bit-identical.
    a = (torch.arange(rows * k, dtype=torch.int32).reshape(rows, k) % 7).to(torch.float32)
    _assert_close(kernel(a.npu()), a.sum(dim=1))


@tilelang.testing.requires_ascend
def test_parallel_serial_shell_last_write_wins():
    rows, steps = 64, 4

    @T.prim_func
    def main(A: T.Tensor((rows, steps), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, steps), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for k in T.serial(steps):
                        c_ub[i] = a_ub[i, k]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert source.count("simd_inst::vsts_norm(") == steps, "the shell is unrolled iteration by iteration"

    a = torch.arange(rows * steps, dtype=torch.float32).reshape(rows, steps)
    _assert_close(kernel(a.npu()), a[:, -1])


@tilelang.testing.requires_ascend
def test_parallel_serial_shell_kept_as_a_scalar_loop():
    """A shell that cannot be absorbed stays a real loop around the vector work.

    ``c[i] = a[i, k]`` writes one address per lane, so ``k`` is neither a vector
    axis nor a reduction; the plan keeps it as a scalar shell instead of falling
    back (and does not unroll it, unlike the constant short shell above).
    """
    rows, steps = 64, 16

    @T.prim_func
    def main(A: T.Tensor((rows, steps), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, steps), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for k in T.serial(steps):
                        c_ub[i] = a_ub[i, k]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    # One store per chunk inside the shell loop: 16 unrolled stores would mean
    # the shell was replicated instead of kept as a loop.
    assert source.count("simd_inst::vsts_norm(") == 1, source
    assert "simd_inst::vgather2(" in source

    a = torch.arange(rows * steps, dtype=torch.float32).reshape(rows, steps)
    _assert_close(kernel(a.npu()), a[:, -1])


@tilelang.testing.requires_ascend
def test_parallel_runtime_shell_is_kept_as_a_scalar_loop():
    rows, steps = 8, 64

    @T.prim_func
    def main(A: T.Tensor((rows, steps), F32), C: T.Tensor((rows,), F32), t: T.int32):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, steps), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for k in T.serial(t):
                        c_ub[i] = a_ub[i, k]
            T.copy(c_ub, C)

    kernel, source = _compile(main, out_idx=None)
    assert "simd_inst::vgather2(" in source, "the lane address stays a gather"
    # One store per chunk inside the shell loop: replicating the shell would
    # multiply the store count by the trip count.
    assert source.count("simd_inst::vsts") == 1

    a = torch.arange(rows * steps, dtype=torch.float32).reshape(rows, steps)
    c_npu = torch.full((rows,), -1.0, dtype=torch.float32).npu()
    kernel(a.npu(), c_npu, steps)
    _assert_close(c_npu, a[:, -1])


@tilelang.testing.requires_ascend
def test_parallel_under_a_serial_shell():
    """A ``T.Parallel`` nested under a serial loop inside the VF is vectorized.

    The serial loop stays scalar and keeps its counter; the vector code replaces
    only the Parallel body, so the address may reference ``k``.
    """
    rows, steps = 128, 4

    @T.prim_func
    def main(A: T.Tensor((steps, rows), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((steps, rows), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            T.fill(c_ub, 0.0)
            with T.SimdVF():
                for k in T.serial(steps):
                    for i in T.Parallel(rows):
                        c_ub[i] = c_ub[i] + a_ub[k, i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vadd" in source
    assert "simd_inst::vlds_norm<" in source

    a = (torch.arange(steps * rows, dtype=torch.int32).reshape(steps, rows) % 5).to(torch.float32)
    _assert_close(kernel(a.npu()), a.sum(dim=0))


@tilelang.testing.requires_ascend
def test_parallel_sibling_reduction_shells_are_fissioned():
    rows, k = 8, 64

    @T.prim_func
    def main(
        A: T.Tensor((rows, k), F32),
        B: T.Tensor((rows, k), F32),
        C: T.Tensor((rows,), F32),
        D: T.Tensor((rows,), F32),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, k), F32)
            b_ub = T.alloc_shared((rows, k), F32)
            c_ub = T.alloc_shared((rows,), F32)
            d_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.fill(c_ub, 0.0)
            T.fill(d_ub, 0.0)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for j in T.serial(k):
                        c_ub[i] = c_ub[i] + a_ub[i, j]
                    for j in T.serial(k):
                        d_ub[i] = d_ub[i] + b_ub[i, j]
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    kernel, source = _compile(main, out_idx=[-2, -1])
    assert source.count("simd_inst::vcadd(") == 2, "one reduction per fissioned nest"

    a = (torch.arange(rows * k, dtype=torch.int32).reshape(rows, k) % 5).to(torch.float32)
    b = (torch.arange(rows * k, dtype=torch.int32).reshape(rows, k) % 3).to(torch.float32)
    c_npu, d_npu = kernel(a.npu(), b.npu())
    _assert_close(c_npu, a.sum(dim=1))
    _assert_close(d_npu, b.sum(dim=1))


# ---------------------------------------------------------------------------
# cast stores: narrowing, reinterpret, widening and the pair matrix
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_narrowing_cast_store():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), C: T.Tensor((n,), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), "bfloat16")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast("bfloat16", a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcvt" in source and "PART_EVEN" in source
    assert "vsts_pack_b32" in source, "64 bf16 results are written through the packed form"

    # Exact in both float32 and bfloat16: no rounding-mode dependence.
    a = (torch.arange(n, dtype=torch.float32) % 8) - 4.0
    _assert_close(kernel(a.npu()), a.to(torch.bfloat16))


@pytest.mark.parametrize("src_dtype,pack", [("int32", "vsts_pack_quarter"), ("int16", "vsts_pack_b16")])
@tilelang.testing.requires_ascend
def test_parallel_integer_narrowing_cast_store(src_dtype, pack):
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), src_dtype), C: T.Tensor((n,), "uint8")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), src_dtype)
            c_ub = T.alloc_shared((n,), "uint8")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast("uint8", a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcvt" in source and pack in source

    # Every value is inside the uint8 range, so the expected result is the same
    # whether the narrowing saturates or wraps; only the conversion is checked.
    a = (torch.arange(n, dtype=torch.int32) % 256).to(getattr(torch, src_dtype))
    golden = a.to(torch.uint8)
    _assert_close(kernel(a.npu()), golden)


@tilelang.testing.requires_ascend
def test_parallel_fp16_to_fp8_store():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), "float16"), C: T.Tensor((n,), "float8_e4m3fn")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float16")
            c_ub = T.alloc_shared((n,), "float8_e4m3fn")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast("float8_e4m3fn", a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcvt" in source and "vsts_pack_quarter" in source

    # All values are exactly representable in e4m3 (3 mantissa bits).
    values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 8.0]
    a = torch.tensor(values * (n // len(values)), dtype=torch.float16)
    golden = a.to(torch.float8_e4m3fn)
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.uint8), golden.view(torch.uint8), rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_parallel_same_width_signedness_reinterpret():
    """``i32 -> u32`` is bit-identical: a reinterpret, never a ``vcvt``."""
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), "int32"), C: T.Tensor((n,), "uint32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "int32")
            c_ub = T.alloc_shared((n,), "uint32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast("uint32", a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcvt" not in source, "a same-width sign change costs no instruction"
    assert "simd_inst::vld" in source and "simd_inst::vsts" in source

    # Every bit pattern must survive, including the sign bit and INT32_MIN.
    a = torch.tensor(
        [-2147483648, -1, -12345, 0, 1, 12345, 2147483647, -7] * (n // 8), dtype=torch.int32
    )
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.int32), a, rtol=0, atol=0)


_WIDENING_CASES = (
    ("float16", "float32", "vlds_unpack", "PART_EVEN"),
    ("int16", "int32", "vlds_unpack", "PART_EVEN"),
    ("int8", "int32", "vlds_unpack4", "PART_P0"),
    ("int8", "int16", "vlds_unpack", "PART_EVEN"),
    ("uint8", "uint16", "vlds_unpack", "PART_EVEN"),
    ("uint8", "float16", "vlds_unpack", "PART_EVEN"),
    ("int8", "float16", "vlds_unpack", "PART_EVEN"),
    ("float8_e4m3fn", "float32", "vlds_unpack4", "PART_P0"),
)

_FP8_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 8.0]


def _cast_kernel(n, src_dtype, dst_dtype):
    @T.prim_func
    def main(A: T.Tensor((n,), src_dtype), C: T.Tensor((n,), dst_dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), src_dtype)
            c_ub = T.alloc_shared((n,), dst_dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast(dst_dtype, a_ub[i])
            T.copy(c_ub, C)

    return main


@pytest.mark.parametrize(
    "src_dtype,dst_dtype,load,part",
    _WIDENING_CASES,
    ids=[case[0] + "->" + case[1] for case in _WIDENING_CASES],
)
@tilelang.testing.requires_ascend
def test_parallel_widening_cast_store(src_dtype, dst_dtype, load, part):
    """A widening store reads its source through the ``UNPK_*`` load layout."""
    n = 128
    kernel, source = _compile(_cast_kernel(n, src_dtype, dst_dtype))
    assert f"simd_inst::{load}" in source, "the widening load must spread the source elements"
    assert part in source, "the conversion reads the spread sub-slots"

    if src_dtype == "float8_e4m3fn":
        a = torch.tensor(_FP8_VALUES * (n // len(_FP8_VALUES)), dtype=torch.float32).npu()
        a = a.to(torch.float8_e4m3fn)
        golden = a.cpu().to(torch.float32)
        _assert_close(kernel(a), golden)
        return

    if src_dtype == "float16":
        # 0..127 are exact in float16, so the widening result is bit-exact too.
        a = torch.arange(n, dtype=torch.float16)
        golden = a.to(torch.float32)
    elif src_dtype == "int16":
        a = (torch.arange(n, dtype=torch.int32) % 401 - 200).to(torch.int16)
        golden = a.to(getattr(torch, dst_dtype))
    elif src_dtype == "uint8":
        a = (torch.arange(n, dtype=torch.int32) % 256).to(torch.uint8)
        golden = a.to(getattr(torch, dst_dtype))
    else:
        a = (torch.arange(n, dtype=torch.int32) % 251 - 125).to(torch.int8)
        golden = a.to(getattr(torch, dst_dtype))
    _assert_close(kernel(a.npu()), golden)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@tilelang.testing.requires_ascend
def test_parallel_eight_bit_widening_tail(dtype):
    """``i8 -> i16`` over a byte tail: the gather zero-extends, ``vcvt`` signs.

    A tail wider than half a register has no ``UNPK`` form, so the masked
    ``vgather2`` widens each byte into its own 16-bit lane. That widening is
    zero-extension (CANN intrinsic contract), which is already the value for
    ``uint8`` but only half of it for ``int8``; the signed pair therefore adds
    the same ``vcvt(PART_EVEN)`` the full-chunk recipe uses.
    """
    n = 100
    dst_dtype = "int16" if dtype == "int8" else "uint16"

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), C: T.Tensor((n,), dst_dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), dtype)
            c_ub = T.alloc_shared((n,), dst_dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast(dst_dtype, a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2(" in source, "the byte tail is widened by the masked gather"
    assert "simd_inst::vsts_norm(" in source
    if dtype == "int8":
        assert "simd_inst::vcvt" in source, "a signed byte needs the PART_EVEN sign extension"
        assert "PART_EVEN" in source
    else:
        assert "simd_inst::vcvt" not in source, "zero-extension is already the uint8 -> uint16 value"

    a = (torch.arange(n, dtype=torch.int32) % 251 - 125).to(getattr(torch, dtype))
    _assert_close(kernel(a.npu()), a.to(getattr(torch, dst_dtype)))


_VALUES = [0, 1, 2, 3, 4, 6, 8, -1, -2, -3, -4, -6, -8]
_VALUES_UNSIGNED = [0, 1, 2, 3, 4, 6, 8, 9, 10, 12, 16, 24, 32]


def _cast_values(src_dtype, dst_dtype):
    """Exact, in-range values for the pair (no rounding or saturation involved)."""
    if src_dtype.startswith("uint") or dst_dtype.startswith("uint"):
        return _VALUES_UNSIGNED
    return _VALUES


# (src, dst, vcvt_expected)
_CAST_CASES = [
    ("float32", "int32", True),
    ("float32", "int16", True),
    ("float32", "float16", True),
    ("float32", "float8_e4m3fn", True),
    ("float16", "int32", True),
    ("float16", "int16", True),
    ("float16", "int8", True),
    ("float16", "uint8", True),
    ("float16", "bfloat16", True),
    ("bfloat16", "int32", True),
    ("bfloat16", "float16", True),
    ("bfloat16", "float32", True),
    ("bfloat16", "float8_e4m3fn", True),
    ("int32", "float32", True),
    ("int32", "int16", True),
    ("int32", "uint16", True),
    ("uint32", "int16", True),
    ("uint32", "uint16", True),
    ("uint32", "uint8", True),
    ("int16", "float32", True),
    ("int16", "float16", True),
    ("int16", "uint32", True),
    ("uint16", "uint32", True),
    ("uint16", "uint8", True),
    ("int16", "uint16", False),  # same width: a register reinterpret
    ("uint16", "int16", False),
]


@pytest.mark.parametrize(
    "src_dtype,dst_dtype,has_vcvt",
    _CAST_CASES,
    ids=[f"{src}->{dst}" for src, dst, _ in _CAST_CASES],
)
@tilelang.testing.requires_ascend
def test_parallel_cast_pair(src_dtype, dst_dtype, has_vcvt):
    n = 128

    kernel, source = _compile(_cast_kernel(n, src_dtype, dst_dtype))
    if has_vcvt:
        assert "simd_inst::vcvt<" in source, "the conversion must be a vector vcvt"
    else:
        assert "simd_inst::vcvt" not in source, "a same-width sign change costs no instruction"
        assert "simd_inst::vld" in source and "simd_inst::vsts" in source

    values = _cast_values(src_dtype, dst_dtype)
    a = torch.tensor([values[i % len(values)] for i in range(n)], dtype=getattr(torch, src_dtype))
    golden = a.to(getattr(torch, dst_dtype))
    actual = kernel(a.npu()).cpu()
    if src_dtype in ("int16", "uint16") and dst_dtype in ("int16", "uint16"):
        # Same width: every bit pattern must survive.
        torch.testing.assert_close(actual.view(torch.int16), a.view(torch.int16), rtol=0, atol=0)
    else:
        _assert_close(actual, golden)


@tilelang.testing.requires_ascend
def test_parallel_bf16_to_f16_overflow_is_inf():
    """``bf16 -> f16`` overflow becomes ``+-inf``: this pair does not saturate.

    Both formats are 16-bit, but bf16 keeps f32's exponent (max ~3.4e38) while
    f16 stops at 65504. ``simd.py`` lists saturation only for ``float32 ->
    float16/bfloat16``, the fp8 pairs and the integer pairs; ``bfloat16 ->
    float16`` keeps the IEEE overflow result, which the device confirms.
    bf16's mantissa is shorter than f16's, so rounding never changes a value
    either — in-range inputs must be bit-exact.
    """
    n = 128

    kernel, source = _compile(_cast_kernel(n, "bfloat16", "float16"))
    assert "simd_inst::vcvt<" in source, "the conversion must be a vector vcvt"

    # 1e5 / 2^17 are bf16-representable and far above the f16 maximum; 65504 is
    # the maximum itself (it rounds up to 65536 in bf16, so it overflows too).
    values = [1e5, -1e5, 131072.0, -131072.0, 65504.0, 0.5, 3.0, -2.0]
    a = torch.tensor(values * (n // len(values)), dtype=torch.bfloat16)
    golden = a.to(torch.float16)  # torch also maps overflow to +-inf
    _assert_close(kernel(a.npu()).cpu(), golden)


# ---------------------------------------------------------------------------
# NZ layout, pack2 and the rejected shapes
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_compact_nz_store():
    @T.prim_func
    def main(A: T.Tensor((M, N), F32), C: T.Tensor((M, N), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            c_ub = T.alloc_shared((M, N), F32)
            T.annotate_layout({c_ub: make_ascend_compact_nz_layout(c_ub)})
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[i, j] = a_ub[i, j] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vsstb(" in source, "the NZ row is a block-strided scatter"
    assert "simd_inst::vsts_norm(" not in source

    a = (torch.arange(M * N, dtype=torch.float32) % 500) - 250.0
    golden = (a * 2.0).reshape(M, N)
    _assert_close(kernel(a.reshape(M, N).npu()), _nz_physical(golden, 8))


@tilelang.testing.requires_ascend
def test_parallel_elementwise_nz_and_row_reduction_are_planned_independently():
    """Elementwise + NZ and a row reduction in one VF, each in its own nest.

    The two recipes have no common vector axis. Keeping them in one
    ``T.Parallel`` nest would make the reduction ignore an axis, which P1
    rejects as a DSL error (it cannot fall back to scalar code inside a
    ``T.SimdVF``), so the program uses one nest per recipe.
    """
    @T.prim_func
    def main(
        A: T.Tensor((M, N), F32),
        S: T.Tensor((M, N), F32),
        C: T.Tensor((M, N), "bfloat16"),
        D: T.Tensor((M,), F32),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            s_ub = T.alloc_shared((M, N), F32)
            c_ub = T.alloc_shared((M, N), "bfloat16")
            d_ub = T.alloc_shared((M,), F32)
            T.annotate_layout({c_ub: make_ascend_compact_nz_layout(c_ub)})
            T.copy(A, a_ub)
            T.copy(S, s_ub)
            T.fill(d_ub, 0.0)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[i, j] = T.Cast("bfloat16", a_ub[i, j] * 2.0)
                for i in T.Parallel(M):
                    for k in T.serial(N):
                        d_ub[i] += s_ub[i, k]
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    kernel, source = _compile(main, out_idx=[-2, -1])
    assert "simd_inst::vld_x2<" in source, "the narrowed NZ store uses the paired form"
    assert "simd_inst::vsstb(" in source
    assert "simd_inst::vcadd(" in source, "the row reduction has its own nest"

    a = (torch.arange(M * N, dtype=torch.int32) % 64).to(torch.float32)
    s = (torch.arange(M * N, dtype=torch.int32) % 9).to(torch.float32)
    c_npu, d_npu = kernel(a.reshape(M, N).npu(), s.reshape(M, N).npu())

    golden = (a * 2.0).to(torch.bfloat16).reshape(M, N)
    _assert_close(c_npu, _nz_physical(golden, 16))
    _assert_close(d_npu, s.reshape(M, N).sum(dim=1))


@tilelang.testing.requires_ascend
def test_parallel_paired_nz_store_with_an_nd_sibling():
    """A narrowed NZ store next to an ND store uses the pack2 interchange.

    The two f32 chunk halves are de-interleaved by one ``vld2``, the ND half is
    re-interleaved by ``vintlv`` before its contiguous stores, and the bf16 NZ
    half is merged by ``vor`` into one full-register ``vsstb``.
    """
    @T.prim_func
    def main(A: T.Tensor((M, N), F32), C: T.Tensor((M, N), "bfloat16"), D: T.Tensor((M, N), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            c_ub = T.alloc_shared((M, N), "bfloat16")
            d_ub = T.alloc_shared((M, N), F32)
            T.annotate_layout({c_ub: make_ascend_compact_nz_layout(c_ub)})
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    d_ub[i, j] = a_ub[i, j] * 2.0
                    c_ub[i, j] = T.Cast("bfloat16", a_ub[i, j])
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    kernel, source = _compile(main, out_idx=None)
    assert "simd_inst::vld_x2<" in source, "one vld2 supplies both chunk halves"
    assert "simd_inst::vintlv(" in source, "the ND half is restored to chunk order"
    assert "simd_inst::vor(" in source, "the two bf16 parts merge into a full register"
    assert "simd_inst::vsstb(" in source
    assert source.count("simd_inst::vsts_norm(") == 2, "the ND half is two contiguous stores"

    a = (torch.arange(M * N, dtype=torch.int32) % 64).to(torch.float32)
    c_npu = torch.zeros((M, N), dtype=torch.bfloat16).npu()
    d_npu = torch.zeros((M, N), dtype=torch.float32).npu()
    kernel(a.reshape(M, N).npu(), c_npu, d_npu)

    _assert_close(c_npu, _nz_physical(a.to(torch.bfloat16).reshape(M, N), 16))
    _assert_close(d_npu, (a * 2.0).reshape(M, N))


if __name__ == "__main__":
    tilelang.testing.main()
