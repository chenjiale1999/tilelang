"""Offline structural tests for NZ-layout stores (``vsstb`` + pack2).

The target buffer carries an explicit Ascend NZ layout, so the planner
classifies the store as ``layout_kind == "nz"`` and the emitter scatters it
with a derived block stride.
"""

from __future__ import annotations


import pytest

import tilelang.ascend.language as T

from tilelang.ascend.analysis.parallel_vectorize import (
    Unsupported,
    emit_plan,
)
from tilelang.layout import (
    make_ascend_compact_nz_layout,
    make_ascend_major_mn_layout,
    make_ascend_nz_layout,
)
from testing.ascend.transform._parallel_vectorize_helpers import (
    _WIDE,
    _count_parallel_nests,
    _dump_plan_ir,
    _emitted_text,
    _nz,
    _nz_2d_kernel,
    _p1_items,
    _plan,
    _rejects,
)


# ===========================================================================
# 阶段二：目标 buffer 带显式 NZ 布局（寄存器直写 + 半寄存器配对形态）
# ===========================================================================
#
# 写一个注解成 Ascend NZ（compact ``[col/C0, row, col%C0]`` 或 canonical MajorK）
# 的 buffer 不能用连续写：planner 把该 store 判成 ``layout_kind == "nz"``，块步长
# 由布局前向映射推导，emitter 用 ``tl.simd.vsstb`` 散射。
#
# 窄化 store（``f32 -> bf16``）只填自己寄存器的一半，而 ``vsstb`` 没有 PK 变体，
# 所以阶段 2b 先把它加宽：一个 chunk 对的两半由一条 ``vld2(DINTLV_B32)`` 反交织
# 取出，分别窄化进目标寄存器的偶/奇 16-bit 槽（``vcvt PART_EVEN/PART_ODD``），
# ``vor`` 合并成满寄存器后一条 ``vsstb`` 散射整行；同段的连续目标在写回处用
# ``vintlv`` 还原。块步长是算出来的，不是硬编码。


def test_compact_nz_store_plan_emit_and_lowering():
    """Classify the NZ store, emit vsstb and replace the nest in one compile."""
    kernel = _nz_2d_kernel()
    layouts = _nz(kernel)
    plan = _plan(kernel, layouts)
    segment = plan.segments[0]
    store = segment.stores[0]

    assert segment.lanes == 64
    assert store.layout_kind == "nz"
    # Compact NZ is [col/C0, row, col%C0]; two neighbouring 32B blocks are
    # ``rows`` blocks apart, i.e. 128 * 8 fp32 elements / C0 = 128 blocks.
    assert store.nz_block_stride == _WIDE

    _dump_plan_ir("compact NZ store", plan)
    stmt = emit_plan(plan)
    text = str(stmt)

    assert "tl.simd.vsstb" in text
    # (block_stride << 16)：低 16 位是 POST_UPDATE 的前进量，普通直存恒 0。
    assert str(_WIDE << 16) in text
    # The register is scattered, never written contiguously.
    assert "tl.simd.vsts" not in text
    assert _count_parallel_nests(stmt) == 0


def test_canonical_nz_store_uses_the_same_block_stride():
    kernel = _nz_2d_kernel()
    layouts = _nz(kernel, make_ascend_nz_layout)
    store = _plan(kernel, layouts).segments[0].stores[0]

    assert store.layout_kind == "nz"
    assert store.nz_block_stride == _WIDE
    assert "tl.simd.vsstb" in _emitted_text(kernel, layouts)


def test_nd_destination_keeps_the_contiguous_store():
    """Zero-diff guard: no explicit layout ⇒ the historical ND form."""

    kernel = _nz_2d_kernel()
    store = _plan(kernel).segments[0].stores[0]

    assert store.layout_kind == "nd"
    assert store.nz_block_stride is None

    text = _emitted_text(_nz_2d_kernel())
    assert '"NORM_B32"' in text
    assert "vsstb" not in text


def test_p1_hand_off_exposes_the_annotated_layouts():
    """P1 hands the annotated buffer names over to P2 with the region."""

    @T.prim_func
    def k(A: T.Tensor((_WIDE, _WIDE), "float32"), C: T.Tensor((_WIDE, _WIDE), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            c_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            T.copy(A, a_ub)
            T.annotate_layout({c_ub: make_ascend_compact_nz_layout(c_ub)})
            with T.SimdVF():
                for i, j in T.Parallel(_WIDE, _WIDE):
                    c_ub[i, j] = T.exp(a_ub[i, j])
            T.copy(c_ub, C)

    items = _p1_items(k)
    lifted = [item for item in items if item.status == "LIFTED"]
    assert lifted, "the nest must be lifted"
    assert [
        capture.buffer.name
        for capture in lifted[0].captures
        if capture.layout is not None
    ] == ["c_ub"]


def test_explicit_non_nz_layout_is_rejected():
    kernel = _nz_2d_kernel()
    layouts = _nz(kernel, make_ascend_major_mn_layout)

    with pytest.raises(Unsupported, match="explicit non-NZ layout"):
        _plan(kernel, layouts)


def test_half_register_nz_store_is_paired_into_a_full_register():
    """Stage 2b：窄化 + NZ 用配对形态（vld2 + part0/1 + vor + 一条 vsstb）。"""

    @T.prim_func
    def k(A: T.Tensor((_WIDE, _WIDE), "float32"), B: T.Tensor((_WIDE, _WIDE), "float32"),
          C: T.Tensor((_WIDE, _WIDE), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            b_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            c_ub = T.alloc_shared((_WIDE, _WIDE), "bfloat16")
            T.copy(A, a_ub)
            for i, j in T.Parallel(_WIDE, _WIDE):
                b_ub[i, j] = T.exp(a_ub[i, j])
                c_ub[i, j] = T.Cast("bfloat16", b_ub[i, j])
            T.copy(b_ub, B)
            T.copy(c_ub, C)

    store = _plan(k, _nz(k)).segments[0].stores[1]
    assert store.store_lane_mode == "pk_half"
    assert store.layout_kind == "nz"
    assert store.nz_block_stride == _WIDE

    _dump_plan_ir("narrow + NZ (pack2)", _plan(k, _nz(k)))
    text = _emitted_text(k, _nz(k))
    # 两个 chunk 的 f32 值由一条 vld2 反交织取出，再分别写进目标寄存器的
    # 偶数/奇数 16-bit 槽。
    assert text.count("tl.simd.vld2(") == 1
    assert "DINTLV_B32" in text
    assert "tl.simd.vdintlv" not in text, "反交织已经折进 vld2，不应再发一次"
    assert "PART_EVEN" in text
    assert "PART_ODD" in text
    assert "tl.simd.vor" in text
    assert "uint16x128" in text
    # 同段的 ND 目标在写回处还原成连续两半：vintlv + 两条 vsts。
    assert "tl.simd.vintlv" in text
    assert text.count("tl.simd.vsts") == 2
    # 合并后是满寄存器：一条 vsstb 散射整行，不再需要半寄存器打包写。
    assert text.count("tl.simd.vsstb") == 1
    assert str(_WIDE << 16) in text
    assert '"PK_B32"' not in text


def test_paired_store_alone_loads_both_chunks():
    """单条窄化 + NZ store：一条 vld2 -> 两个 part -> 一条 vsstb（DSA 配方）。"""

    @T.prim_func
    def k(A: T.Tensor((_WIDE, _WIDE), "float32"), C: T.Tensor((_WIDE, _WIDE), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            c_ub = T.alloc_shared((_WIDE, _WIDE), "bfloat16")
            T.copy(A, a_ub)
            for i, j in T.Parallel(_WIDE, _WIDE):
                c_ub[i, j] = T.Cast("bfloat16", a_ub[i, j])
            T.copy(c_ub, C)

    text = _emitted_text(k, _nz(k))
    assert text.count("tl.simd.vld2(") == 1
    assert "tl.simd.vld(" not in text
    assert "tl.simd.vdintlv" not in text
    assert text.count("tl.simd.vsstb") == 1


def test_paired_store_loops_over_chunk_pairs():
    """4 chunks = 2 pairs: 每次迭代推进两个 chunk，vsstb 在循环内只发一条。"""

    rows, wide = 64, 256

    @T.prim_func
    def k(A: T.Tensor((rows, wide), "float32"), C: T.Tensor((rows, wide), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, wide), "float32")
            c_ub = T.alloc_shared((rows, wide), "bfloat16")
            T.copy(A, a_ub)
            for i, j in T.Parallel(rows, wide):
                c_ub[i, j] = T.Cast("bfloat16", a_ub[i, j])
            T.copy(c_ub, C)

    segment = _plan(k, _nz(k)).segments[0]
    assert segment.lanes == 64
    assert segment.chunks == 4

    text = _emitted_text(k, _nz(k))
    # 每对 chunk 一次迭代：chunk 索引 2*pair，步长 2 * lanes。
    assert "T.grid(%d, 2)" % rows in text
    assert "chunk_s0 * 128" in text
    assert text.count("tl.simd.vsstb") == 1
    assert str(rows << 16) in text


def test_paired_store_needs_whole_chunk_pairs():
    """64 列 = 1 个 chunk，凑不出一个 chunk 对。"""

    m = 64

    @T.prim_func
    def k(A: T.Tensor((_WIDE, m), "float32"), C: T.Tensor((_WIDE, m), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, m), "float32")
            c_ub = T.alloc_shared((_WIDE, m), "bfloat16")
            T.copy(A, a_ub)
            for i, j in T.Parallel(_WIDE, m):
                c_ub[i, j] = T.Cast("bfloat16", a_ub[i, j])
            T.copy(c_ub, C)

    with pytest.raises(Unsupported, match="whole chunk pairs"):
        _plan(k, _nz(k))


def test_paired_store_must_be_the_last_store():
    """合并寄存器只在配对结束后才存在，打包 store 后面不能再有语句。"""

    @T.prim_func
    def k(A: T.Tensor((_WIDE, _WIDE), "float32"), B: T.Tensor((_WIDE, _WIDE), "float32"),
          C: T.Tensor((_WIDE, _WIDE), "bfloat16"), D: T.Tensor((_WIDE, _WIDE), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            b_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            c_ub = T.alloc_shared((_WIDE, _WIDE), "bfloat16")
            d_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            T.copy(A, a_ub)
            for i, j in T.Parallel(_WIDE, _WIDE):
                b_ub[i, j] = T.exp(a_ub[i, j])
                c_ub[i, j] = T.Cast("bfloat16", b_ub[i, j])
                d_ub[i, j] = b_ub[i, j] * T.float32(2.0)
            T.copy(b_ub, B)
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    with pytest.raises(Unsupported, match="must be the last store"):
        _plan(k, _nz(k))


def test_nz_store_with_predicate_is_rejected():
    @T.prim_func
    def k(A: T.Tensor((_WIDE, _WIDE), "float32"), C: T.Tensor((_WIDE, _WIDE), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            c_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            T.copy(A, a_ub)
            for i, j in T.Parallel(_WIDE, _WIDE):
                if j < _WIDE - 8:
                    c_ub[i, j] = T.exp(a_ub[i, j])
            T.copy(c_ub, C)

    with _rejects("body.statement_predicate"):
        _plan(k, _nz(k))


def test_nz_store_with_partial_column_chunk_is_rejected():
    """96 columns is not a whole number of 64-element chunks."""

    m = 96

    @T.prim_func
    def k(A: T.Tensor((m, m), "float32"), C: T.Tensor((m, m), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((m, m), "float32")
            c_ub = T.alloc_shared((m, m), "float32")
            T.copy(A, a_ub)
            for i, j in T.Parallel(m, m):
                c_ub[i, j] = T.exp(a_ub[i, j])
            T.copy(c_ub, C)

    with _rejects("plan.partial_chunks"):
        _plan(k, _nz(k))
