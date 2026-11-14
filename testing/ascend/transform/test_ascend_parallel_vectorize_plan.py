"""Offline structural tests for the P2 plan / P3 emit entry points.

Covers the transform-level contract (multi-axis nests, tail chunks, serial
shells, reductions, guards, access-shape variants) and the rejection surface
of ``build_plan``.  Hardware independent.
"""

from __future__ import annotations

import contextlib

import pytest

import tilelang.ascend.language as T
from tvm import tirx

from tilelang.ascend.analysis.parallel_vectorize import (
    AccessClass,
    Unsupported,
    VectorProgram,
    build_plan,
    emit_plan,
)
from testing.ascend.transform._parallel_vectorize_helpers import (
    _alloc_buffers,
    _call_args,
    _chunk_iterations,
    _emitted_ops,
    _eval_lane,
    _first_parallel,
    _for_extents,
    _for_kinds,
    _negative_immediate_muls,
    _one_dimensional_copy,
    _over_reading_contiguous_load,
    _plan,
    _rejects,
    _single_segment,
)


# ---------------------------------------------------------------------------
# transform-level entry point
# ---------------------------------------------------------------------------


def test_plan_emit_replaces_the_parallel_nest():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i]

    stmt = emit_plan(_plan(k))

    kinds = _for_kinds(stmt)
    assert tirx.ForKind.PARALLEL not in kinds, "the T.Parallel nest must be replaced"
    assert kinds.count(tirx.ForKind.SERIAL) == 1  # the single axis -> chunk loop only
    assert "tl.simd.vld" in _emitted_ops(stmt)


def test_plan_emit_lowers_the_multidim_nest_as_a_whole():
    """Top-down lowering: a 2-axis ``T.Parallel(M, K)`` must be consumed *as a
    whole* (outer axis → serial, vector axis ``j`` → strip-mined chunks).  A
    bottom-up walk would replace the inner axis first, stranding the outer
    ``T.Parallel``."""
    M, K = 16, 128

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M, K), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            c = T.alloc_shared((M, K), "float32")
            for i, j in T.Parallel(M, K):
                c[i, j] = a[i, j]

    stmt = emit_plan(_plan(k))

    kinds = _for_kinds(stmt)
    assert tirx.ForKind.PARALLEL not in kinds, "the whole nest must be lowered"
    assert kinds.count(tirx.ForKind.SERIAL) == 2, "outer axis + chunk loop, no residue"
    ops = _emitted_ops(stmt)
    assert "tl.simd.vld" in ops and "tl.simd.vsts" in ops


def test_multidim_nest_keeps_every_axis():
    """The plan built from the nest head keeps all axes (outer ``i`` serial,
    inner ``j`` as the vector axis), which is what makes the whole-nest lowering
    possible."""
    M, K = 16, 128

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M, K), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            c = T.alloc_shared((M, K), "float32")
            for i, j in T.Parallel(M, K):
                c[i, j] = a[i, j]

    plan = _single_segment(k)
    assert [a.name for a in plan.axes] == ["i", "j"]
    assert plan.vector_axis.name == "j"


def test_plan_emit_lowers_multi_serial_program():
    """Top-down lowering also consumes a multi-serial ``T.Parallel`` body: the shared
    parallel axis becomes one serial loop, each segment contributes its own chunk loop,
    and source order (``c`` before ``d``) is preserved."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32"), D: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            d = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    d[i, l] = c[i, l] + 1.0

    stmt = emit_plan(_plan(k))

    kinds = _for_kinds(stmt)
    assert tirx.ForKind.PARALLEL not in kinds
    # 两段都展平成满 lane 合成轴：每段一个 chunk 循环。
    assert kinds.count(tirx.ForKind.SERIAL) == 2
    ops = _emitted_ops(stmt)
    assert ops.index("tl.simd.vmuls") < ops.index("tl.simd.vadds")


# ---------------------------------------------------------------------------
# 已知缺陷回归基线（探针 A–M）
#
# 下面每个用例都断言**正确的契约**，并用 ``xfail(strict=True)`` 标注当前尚未满足：
#   * 修复后测试会 XPASS，``strict=True`` 使其直接报错 —— 强制顺手删掉 xfail 标记；
#   * 若某个缺陷被"以另一种合理方式"修复（例如从静默错发改为明确拒绝），
#     用例里已把「拒绝」当作可接受结果一并接受，无需改动。
# 复现入口：``tilelang/ascend/analysis/test_drive_plan.py`` 尾部的 ``d*`` 用例；对应 ``log_poly.log`` 中的诊断。
# ---------------------------------------------------------------------------


def test_defect_d1_product_reduction_is_not_emitted_as_sum():
    """D1 已修：乘积归约**不会再被静默发成求和**。

    沿并行轴 i 的满 lane 向量化把 ``s[i] *= a[i,j]`` 发成逐 lane 的 ``vmul`` 读改写
    （serial j 保留为真实循环），乘积语义完整保留，也不会出现 ``vcadd``。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] * a[i, j]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vmul" in ops and "tl.simd.vcadd" not in ops


def test_defect_d2_plain_overwrite_is_not_treated_as_reduction():
    """D2 已修：覆写 ``s[i] = a[i,j]`` 不会被当成归约（不发 ``vcadd`` 求和）。

    沿并行轴 i 向量化后是逐 lane 赋值、j 循环保留为真实循环，结果与标量
    last-write-wins 完全一致。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i, j in T.Parallel(MS, KS):
                s[i] = a[i, j]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vsts" in ops and "tl.simd.vcadd" not in ops


def test_defect_d3_int_max_reduction_does_not_crash():
    """D3 已修：单位元按 dtype 派生（`tirx.min_value`），int32 max 归约不再抛 ValueError。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "int32"), S: T.Tensor((MS,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "int32")
            s = T.alloc_shared((MS,), "int32")
            for i, j in T.Parallel(MS, KS):
                s[i] = T.max(s[i], a[i, j])

    # 正确契约：发射要么成功，要么以 Unsupported 优雅拒绝；不得抛出 ValueError。
    with contextlib.suppress(Unsupported):
        emit_plan(_single_segment(k))


def test_defect_d3_f16_max_reduction_does_not_crash():
    """D3 已修：单位元按 dtype 派生，f16 max 归约不再超出值域抛 ValueError。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float16"), S: T.Tensor((MS,), "float16")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float16")
            s = T.alloc_shared((MS,), "float16")
            for i, j in T.Parallel(MS, KS):
                s[i] = T.max(s[i], a[i, j])

    with contextlib.suppress(Unsupported):
        emit_plan(_single_segment(k))


def test_defect_d4_uniform_addend_inside_reduction_is_not_dropped():
    """D4/D5 已修：归约体里的 lane 均匀加数不再被丢弃。

    改沿并行轴 i 向量化后，``s[i] = s[i] + a[i,j] + 1.0`` 取值保留 ``+1.0``
    （``vadds`` 的标量侧是 imm 1.0），旧值 ``s[i]`` 以 ``vload`` 直接参与读改写。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] + a[i, j] + 1.0

    segment = _single_segment(k)
    assert segment.reduce is None
    value = segment.stores[0].value
    assert value.recipe == "vadds"
    assert "1.0" in str(value.args[-1].scalar)
    ops = _emitted_ops(emit_plan(segment))
    assert "tl.simd.vadds" in ops and "tl.simd.vcadd" not in ops


def test_defect_d6_two_reductions_in_one_nest_are_both_emitted():
    """D6 已修：同一 nest 内两条归约各自保留（不再只发第一条）。

    沿并行轴 i 向量化后两条 store 都是逐 lane 读改写：两次 ``vadd`` + 两次 ``vsts``。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        B: T.Tensor((MS, KS), "float32"),
        S1: T.Tensor((MS,), "float32"),
        S2: T.Tensor((MS,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            s1 = T.alloc_shared((MS,), "float32")
            s2 = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s1[i] = s1[i] + a[i, j]
                    s2[i] = s2[i] + b[i, j]

    segment = _single_segment(k)
    assert [s.reduce for s in segment.stores] == [None, None]
    ops = _emitted_ops(emit_plan(segment))
    # 两条 store 都要回写（两次 vsts）与两次逐 lane 加法，不得静默丢 store。
    assert ops.count("tl.simd.vsts") >= 2
    assert ops.count("tl.simd.vadd") == 2
    assert "tl.simd.vcadd" not in ops


def test_defect_d13_lane_varying_overwrite_is_not_a_reduction():
    """D13 已修（统一判据）：``s[i] = a[i,j] + 1.0`` 是「后写覆盖」，不会被静默发成求和。

    沿并行轴 i 向量化后是逐 lane 的 ``vadds`` + ``vsts``（j 循环保留），不发 ``vcadd``。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i, j in T.Parallel(MS, KS):
                s[i] = a[i, j] + 1.0

    plan = _single_segment(k)
    assert plan.reduce is None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vadds" in ops and "tl.simd.vcadd" not in ops


def test_defect_d5_scaled_old_value_is_not_a_reduction():
    """D5/D13：旧值被缩放（``s[i] = s[i] * 2.0 + a[i,j]``）不再需要拒绝。

    沿并行轴 i 向量化是逐 lane 读改写：``vmuls(s, 2.0) → vadd(a) → vsts``，
    缩放系数完整保留，与标量语义一致。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] * 2.0 + a[i, j]

    plan = _single_segment(k)
    assert plan.reduce is None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vmuls" in ops and "tl.simd.vadd" in ops and "tl.simd.vcadd" not in ops


def test_defect_d6b_mixed_reductions_keep_their_own_recipe():
    """D6 已修：同一 nest 内求和与求最大**各自**保留自己的算子。

    沿并行轴向量化后分别是 ``vadd`` 与 ``vmax``（不再是跨 lane 的 ``vcadd``/``vcmax``）。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        B: T.Tensor((MS, KS), "float32"),
        S1: T.Tensor((MS,), "float32"),
        S2: T.Tensor((MS,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            s1 = T.alloc_shared((MS,), "float32")
            s2 = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s1[i] = s1[i] + a[i, j]
                    s2[i] = T.max(s2[i], b[i, j])

    segment = _single_segment(k)
    assert [store.reduce for store in segment.stores] == [None, None]
    ops = _emitted_ops(emit_plan(segment))
    assert "tl.simd.vadd" in ops and "tl.simd.vmax" in ops
    assert "tl.simd.vcadd" not in ops and "tl.simd.vcmax" not in ops


def test_reduce_segment_with_uniform_broadcast_sibling():
    """归约段内「取值沿轴均匀」的 store 是一次广播写：既不参与归约，也不被丢掉。

    沿并行轴 i 向量化后：``s1[i] += a[i,j]`` 是逐 lane 读改写，``s2[i] = 3.0``
    是广播写（``vdup`` + ``vsts``）。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S1: T.Tensor((MS,), "float32"), S2: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s1 = T.alloc_shared((MS,), "float32")
            s2 = T.alloc_shared((MS,), "float32")
            for i, j in T.Parallel(MS, KS):
                s1[i] = s1[i] + a[i, j]
                s2[i] = 3.0

    segment = _single_segment(k)
    assert segment.reduce is None
    assert [store.reduce for store in segment.stores] == [None, None]
    ops = _emitted_ops(emit_plan(segment))
    assert "tl.simd.vadd" in ops and "tl.simd.vdup" in ops
    assert ops.count("tl.simd.vsts") >= 2  # 读改写回写 + 广播写各一次


def test_defect_d13_stencil_shifted_read_is_not_an_accumulation():
    """D13/D5：``s[i] = s[i-1] + a[i,j]`` 的 ``s[i-1]`` 与写入地址**不同址**，不是累加。

    旧值可剥离的前提是「与目标地址同址回读」；跨位（stencil）读留在 ``varying`` 内并触发
    显式拒绝，而不是被误当成 ``dst op= x`` 后静默丢掉 ``s[i-1]``。
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i, j in T.Parallel(MS, KS):
                s[i] = s[i - 1] + a[i, j]

    with pytest.raises(Unsupported, match="accumulate position"):
        _single_segment(k)


def test_partial_reduction_uses_parallel_axis_in_multi_segment():
    """A0：``s[2*i] += a[i,j]`` + ``u[i,j] = s[2*i]`` 的段沿 serial 轴 j 是「部分归约」
    （lane 均匀地址 + 逐 lane 变化取值）→ 该轴被拒绝，轴回退到并行轴 i：
    ``s[2*i]`` 沿 i 是步长地址、逐 lane 互不冲突，语义正确。"""
    MS, KS = 8, 128

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), U: T.Tensor((MS, KS), "float32"), S: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            u = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((16,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[2 * i] = s[2 * i] + a[i, j]
                    u[i, j] = s[2 * i]
                for j in T.serial(KS):
                    u[i, j] = a[i, j] * 2.0

    program = _plan(k)
    assert isinstance(program, VectorProgram)
    assert len(program.segments) == 2
    assert program.segments[0].vector_axis.name == "i"
    assert program.segments[0].reduce is None
    emit_plan(program)  # build 成功 ⇒ emit 成功


def test_partial_reduction_falls_back_to_the_parallel_axis():
    """A0（新）：单段路径上，serial 轴被拒后回退到并行轴，结果仍然正确（不是静默错发）。"""
    MS, KS = 8, 128

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), U: T.Tensor((MS, KS), "float32"), S: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            u = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((16,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[2 * i] = s[2 * i] + a[i, j]
                    u[i, j] = s[2 * i]

    segment = _single_segment(k)
    # 不再沿 serial 轴 j 向量化（会丢累加），而是沿并行轴 i；j 保持为普通 serial 循环。
    assert segment.vector_axis.name == "i"
    ops = _emitted_ops(emit_plan(segment))
    assert "tl.simd.vcadd" not in ops  # 不是跨 lane 归约


def test_defect_d7_build_success_implies_emit_success_for_mixed_stores():
    """D7 已修：`is_uniform` 显式挡掉无 args 叶子，`vload_strided` 不再被误判 lane 均匀。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        D: T.Tensor((MS, KS), "float32"),
        S: T.Tensor((MS,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            d = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] + a[i, j]
                    d[i, j] = s[i]

    # 正确契约：build_plan 成功 ⇒ emit_plan 必须成功（不能出现「分析期 OK、发射期 Unsupported」）。
    plan = build_plan(_first_parallel(k))
    emit_plan(plan)


def test_defect_d8_negative_stride_below_minus_one():
    """D8 已修：``Δ ≤ -2`` 的负步长改走「每 lane 索引向量」通道，不再用无符号斜坡乘负步长。

    复现用例已修正为**合法**下标（``126 - 2*i``，``i < 64`` → 126..0）；旧用例
    ``a[N-2-2*i]`` 在 ``i ≥ 64`` 时下标为负，源程序本身就越界。
    """
    M = 64

    @T.prim_func
    def k(A: T.Tensor((128,), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((128,), "float32")
            c = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                c[i] = a[126 - 2 * i]

    try:
        plan = _single_segment(k)
        stmt = emit_plan(plan)
    except Unsupported:
        return
    ops = _emitted_ops(stmt)
    # 正确契约：要么拒绝，要么不得用「无符号 lane 斜坡 × 负步长」表达负步长访问。
    assert not _negative_immediate_muls(stmt)
    assert "tl.simd.vgather2" in ops
    # 数值 golden：lane 0..3 的下标必须是 126, 124, 122, 120（P0-1：不能整体取反）。
    assert [_eval_lane(plan.stores[0].value, lane, plan) for lane in range(4)] == [126, 124, 122, 120]


def test_defect_d8_negative_stride_store_is_scattered():
    """D8 写侧：``Δ ≤ -2`` 的负步长写同样走索引向量 ``vscatter``（不再用负步长斜坡）。"""
    M = 64

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((128,), "float32")
            for i in T.Parallel(M):
                c[126 - 2 * i] = a[i]

    try:
        plan = _single_segment(k)
        stmt = emit_plan(plan)
    except Unsupported:
        return
    ops = _emitted_ops(stmt)
    assert not _negative_immediate_muls(stmt)
    assert "tl.simd.vscatter" in ops
    # 数值 golden（写侧）：scatter 索引 lane 0..3 必须是 126, 124, 122, 120。
    assert [_eval_lane(plan.stores[0].index, lane, plan) for lane in range(4)] == [126, 124, 122, 120]


def test_defect_d9_symbolic_extent_is_not_silently_truncated():
    """D9 已修：符号 extent 走运行时 chunk 循环（``ceildiv(n, lanes)``），不再固定单 chunk。"""
    n = tirx.Var("n", "int32")

    @T.prim_func
    def k(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float32")
            c = T.alloc_shared((n,), "float32")
            for i in T.Parallel(n):
                c[i] = a[i]

    try:
        stmt = emit_plan(_single_segment(k))
    except Unsupported:
        return
    # 正确契约：要么拒绝，要么 chunk 循环必须由符号 extent 驱动（不能硬编码成单 chunk）。
    assert any(not isinstance(extent, tirx.IntImm) for extent in _for_extents(stmt))
    # 尾块车道数必须在运行时夹到 lanes 内（A5：update_mask 不接受超宽车道数）。
    assert "tl.simd.update_mask" in _emitted_ops(stmt)
    assert any(isinstance(node, tirx.Min) for node in _call_args(stmt, "tl.simd.update_mask"))


def test_short_reduction_is_unrolled_with_multiple_accumulator_chains():
    """短归约（``chunks <= 4``）：编译期展开 + 多条 SSA 累加链 + 树形合并。

    ``(4, 256)`` / lanes 64 → 4 个 chunk、无尾块：4 条链各自只含一块（链首直接取
    ``vcadd`` 结果，不需要 ``vdup`` 单位元），再两两合并。串行延迟从 4 级降到 2 级。
    """
    M, K = 4, 256

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), S: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            s = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                s[i] = s[i] + a[i, j]

    plan = _single_segment(k)
    assert plan.chunks == 4 and plan.tail == 0
    stmt = emit_plan(plan)
    ops = _emitted_ops(stmt)
    assert ops.count("tl.simd.vcadd") == 4  # 展开：每个 chunk 各一次
    # 3 次树形合并是向量 `vadd`（PAT_VL1，只算 lane 0）；并入旧值走标量-向量 `vadds`。
    assert ops.count("tl.simd.vadd") == 3
    assert ops.count("tl.simd.vadds") == 1
    assert "tl.simd.vdup" not in ops  # 链首直接取 vcadd 结果，不再需要单位元
    assert not _alloc_buffers(stmt)  # 展开路径不引入可变寄存器
    # 旧值按标量读一次并入，不再有 BRC 广播装载。
    assert "BRC_B32" not in str(stmt)
    # 跨 lane 归约之后的合并/回写都用只保留 lane 0 的谓词。
    assert 'pset(32, "PAT_VL1")' in str(stmt)


def test_long_reduction_keeps_a_real_chunk_loop_with_a_mutable_accumulator():
    """长归约（``chunks > 4``）：不再展开，改回真实 chunk 循环 + ``local.var`` 累加器。

    ``(2, 512)`` / lanes 64 → 8 个 chunk。展开会线性膨胀 8×3 条指令且关键路径仍是 8 级；
    循环版只有一条 ``vcadd``，累加器落成可写向量寄存器（codegen 发 ``vector_f32``）。
    """
    M, K = 2, 512

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), S: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            s = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                s[i] = s[i] + a[i, j]

    plan = _single_segment(k)
    assert plan.chunks == 8
    stmt = emit_plan(plan)
    ops = _emitted_ops(stmt)
    assert ops.count("tl.simd.vcadd") == 1  # 循环体内只有一份
    assert "tl.simd.vadd" in ops
    accumulators = _alloc_buffers(stmt)
    assert [buffer.scope() for buffer in accumulators] == ["local.var"]
    assert str(accumulators[0].dtype) == "float32x64"
    kinds = _for_kinds(stmt)
    assert kinds.count(tirx.ForKind.SERIAL) == 2  # 外层并行轴 i + chunk 循环
    # tail == 0 时 chunk 循环必须覆盖**全部** chunks 块（旧实现只跑 chunks-1 块，
    # 静默漏掉最后一块；strip-mine 收敛后由同一个骨架保证）。
    chunk_loops = [extent for extent in _for_extents(stmt) if isinstance(extent, tirx.IntImm) and int(extent.value) == plan.chunks]
    assert chunk_loops, f"expected a {plan.chunks}-iteration full-chunk loop (tail == 0)"


def test_symbolic_extent_reduction_uses_a_runtime_chunk_loop():
    """符号 extent 归约（D9 归约分支）：不再拒绝，改走「整块主循环 + peel 尾块」。

    主循环跑 ``E // lanes`` 个满 lane 整块；尾块在 ``if E % lanes > 0`` 下执行一次，
    运行期 ``update_mask`` 夹紧后交给 ``vcadd``（Add 单位元是 0，无需 ``vsel`` 填充）。
    """
    n = tirx.Var("n", "int32")

    @T.prim_func
    def k(A: T.Tensor((n,), "float32"), S: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float32")
            s = T.alloc_shared((1,), "float32")
            for i in T.Parallel(n):
                s[0] = s[0] + a[i]

    segment = _single_segment(k)
    assert segment.chunks is None
    stmt = emit_plan(segment)
    ops = _emitted_ops(stmt)
    assert ops.count("tl.simd.vcadd") == 2  # 主循环体 + peel 尾块各一份（未按 extent 展开）
    assert "tl.simd.vsel" not in ops  # Add 不需要单位元填充
    assert any(not isinstance(extent, tirx.IntImm) for extent in _for_extents(stmt))
    assert any(isinstance(node, tirx.Min) for node in _call_args(stmt, "tl.simd.update_mask"))
    assert [buffer.scope() for buffer in _alloc_buffers(stmt)] == ["local.var"]


def test_defect_d10_non_reduction_tail_load_is_bounded():
    """D10 已修：尾块装载带谓词（``vld`` 无谓词形式 → 退化成带谓词的 ``vgather2``）。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0

    try:
        stmt = emit_plan(_single_segment(k))
    except Unsupported:
        return
    # 正确契约：要么拒绝，要么不发出跨越源 buffer 最内维的整 lane 读。
    assert not _over_reading_contiguous_load(stmt)


def test_defect_d10_one_dimensional_tail_load_is_bounded():
    """D10 的一维形态：``(100,)`` 上尾块 ``vld(a[64], 64)`` 会越过分配范围。"""
    M = 100

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                c[i] = a[i]

    try:
        stmt = emit_plan(_single_segment(k))
    except Unsupported:
        return
    assert not _over_reading_contiguous_load(stmt)


def test_tail_chunk_loop_runs_exactly_one_iteration():
    """P0：尾块必须只覆盖最后一个 chunk ``[chunks-1, chunks)``。

    ``tirx.For(var, min, extent)`` 的第三个参数是 extent（不是 end）。旧实现写成
    ``For(tailchunk, full, chunks)``，于是从 ``full`` 起跑 ``chunks`` 次，导致越界读写。
    尾块现在展开成**直线代码**，因此这里分两种形态检查：整块循环的并集必须恰好是
    ``[0, chunks-1)``（有尾块时），或者全部展开时每个 chunk 恰好一次写。
    """
    lanes = 64
    for M in (100, 200, 300):
        chunks = (M + lanes - 1) // lanes
        stmt = emit_plan(_single_segment(_one_dimensional_copy(M)))
        assert not _over_reading_contiguous_load(stmt), f"M={M}: contiguous load escapes the buffer"
        iterations = _chunk_iterations(stmt)
        if not iterations:
            # 单/双 chunk 全部展开：每个 chunk 恰好一次写（含尾块掩码写）。
            assert _emitted_ops(stmt).count("tl.simd.vsts") == chunks, f"M={M}"
            continue
        covered: list = []
        for name, lo, extent in iterations:
            assert lo is not None and extent is not None, f"{name}: non-constant chunk span"
            assert lo >= 0 and lo + extent <= chunks, f"M={M}: chunk loop {name} escapes [0, {chunks})"
            covered.extend(range(lo, lo + extent))
        # 尾块展开后，剩下的整块循环必须**恰好**覆盖 [0, chunks-1)：不多跑（越界）也不少跑。
        expected = list(range(chunks - 1)) if M % lanes else list(range(chunks))
        assert sorted(covered) == expected, f"M={M}: chunk coverage {sorted(covered)} != {expected}"


def test_tail_chunk_guard_is_proved_free():
    """尾块谓词可证：``if i >= 128: c[i] = a[i]`` on a 130-element axis.

    ``i >= 128`` 对整条轴不可证（``i = 0`` 是反例），但在尾块 ``[128, 130)`` 上恒真，
    因此 ``mask_free_from`` 应指向最后一个 chunk，尾块不再生成比较。

    注意探测区间在本 fork 是 ``Range(begin, end)``，即 ``Range(128, 130)``；若误按 TVM
    传统的 ``Range(min, extent)`` 理解写成 ``Range(128, 2)``，会得到空区间，本用例会
    立刻失败（``mask_free_from is None``）。
    """
    N, lanes = 130, 64  # 130 = 2 full chunks + a 2-lane tail chunk

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if i >= 128:
                    c[i] = a[i]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert (plan.chunks, plan.tail) == (3, 2)
    store = plan.stores[0]
    # 整轴不可证时 until 保持 0；尾块可证时 from 指向 chunk 2。
    assert store.mask_free_until == 0
    assert store.mask_free_from == 2
    # 前缀两个 chunk 的谓词恒假，会被整块删除（见 dead_guard 用例）；
    # 剩下的尾块直接用尾掩码，不再生成比较。
    print(f"func:\n\n{k}\n\nemited:\n\n{emit_plan(plan)}")
    ops = _emitted_ops(emit_plan(plan))
    assert ops.count("tl.simd.vcmps") == 0
    assert ops.count("tl.simd.vsts") == 1


def test_dead_guard_chunks_are_elided():
    """谓词恒假的 chunk 整块删除：``if i >= 128: c[i] = a[i]`` on 130 elements.

    ``i >= 128`` 在 chunk 0/1（lane 0-127）上恒假，这两块原本照样发
    ``vci + vcmps + vld + vsts``（写不进任何数据）。plan 侧用
    ``GuardFreeBounds(not(mask))`` 求出死前缀，``chunk_lo/chunk_hi`` 把发射窗口
    收窄到 ``[2, 3)``，于是只剩下尾块那一条 ``vgather2`` + ``vsts``。
    """
    N = 130  # 2 full chunks (dead) + a 2-lane tail chunk (the only live one)

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if i >= 128:
                    c[i] = a[i]

    plan = _single_segment(k)
    assert (plan.chunk_lo, plan.chunk_hi) == (2, 3)

    stmt = emit_plan(plan)
    ops = _emitted_ops(stmt)
    # 死 chunk 循环完全消失：没有 chunk 循环、没有整块 vld/vcmps/vsts。
    assert _chunk_iterations(stmt) == []
    assert ops.count("tl.simd.vld") == 0
    assert ops.count("tl.simd.vcmps") == 0
    assert ops.count("tl.simd.vgather2") == 1
    assert ops.count("tl.simd.vsts") == 1


def test_unmasked_store_keeps_every_chunk_live():
    """无谓词 store 不会被误删：``chunk_lo/chunk_hi`` 保持默认整窗。"""
    N = 130

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i]

    plan = _single_segment(k)
    assert (plan.chunk_lo, plan.chunk_hi) == (0, 3)
    stmt = emit_plan(plan)
    # 整块循环仍然覆盖 chunk 0/1；尾块照旧走带掩码的 gather。
    assert [(name, lo, extent) for name, lo, extent in _chunk_iterations(stmt)] == [
        ("chunk", 0, 2)
    ]
    assert _emitted_ops(stmt).count("tl.simd.vld") == 1


def test_defect_d11_build_success_implies_emit_success_for_sibling_statement():
    """D11 已修：pre/post 的间接地址若沿向量轴不变（Δ==0），是 lane 内均匀的**标量**地址。

    ``c[b[0]] = 1.0`` 不属于「每 lane 索引向量」通道，但也不是拒绝理由：它沿 serial 轴均匀，
    发射成一次 ``ONEPT`` 单元素写；serial 主体照常向量化（部分向量化）。build 成功 ⇒ emit 成功。
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        B: T.Tensor((1,), "int32"),
        D: T.Tensor((MS, KS), "float32"),
        C: T.Tensor((MS,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((1,), "int32")
            d = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                c[b[0]] = 1.0
                for j in T.serial(KS):
                    d[i, j] = a[i, j] * 2.0

    plan = build_plan(_first_parallel(k))
    segment = plan.segments[0]
    assert segment.vector_axis.name == "i"  # 并行主体被向量化，serial 轴保留为循环
    assert segment.reduce is None
    assert [store.kind for store in segment.pre_stores] == [AccessClass.UNIFORM]
    stmt = emit_plan(plan)  # build 成功 ⇒ emit 成功
    ops = _emitted_ops(stmt)
    assert "tl.simd.vmuls" in ops  # 主体照常向量化
    # pre 与所有轴无关 → 提到最外层、退化成一次**标量** store（不再 vdup + ONEPT）。
    assert "c[b[0]] = T.float32(1.0)" in str(stmt)
    assert "ONEPT_B32" not in str(stmt)


def test_lane_invariant_indirect_load_is_uniform():
    """Δ==0 的间接地址（``a[b[0]]``）是 lane 内均匀标量读 → ``vdup`` 广播，不进索引向量通道。"""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((1,), "int32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((1,), "int32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[b[0]] * 2.0

    segment = _single_segment(k)
    assert segment.stores[0].kind == AccessClass.CONTIGUOUS
    ops = _emitted_ops(emit_plan(segment))
    assert "tl.simd.vdup" in ops
    assert "tl.simd.vgather2" not in ops and "tl.simd.vld" not in ops


def test_bool_store_is_rejected_at_build_time():
    """D11 同族：``bool`` 型 store 在 build 期拒绝，而不是 emit 期抛 ``ValueError``。"""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "bool")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "bool")
            for i in T.Parallel(N):
                c[i] = a[i] > 0.0

    with pytest.raises(Unsupported, match="not supported"):
        _single_segment(k)


def test_masked_branch_merging_falls_back_to_predication():
    """缺陷 D12 的后续：``if/else`` 同址合并时某一臂自带谓词（嵌套 else-if）→ 不再拒绝。

    没有单一的 select 条件，但两臂仍可各自谓词化：
    ``c = a>0 ? 1 : (b>0 ? 2 : c)`` 发射为两条互补掩码的 store。
    """
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if a[i] > 0.0:
                    c[i] = 1.0
                else:
                    if b[i] > 0.0:
                        c[i] = 2.0

    plan = _single_segment(k)
    assert [store.spec.buffer.name for store in plan.stores] == ["c", "c"]
    assert all(store.mask is not None for store in plan.stores), "each arm is predicated"
    assert "tl.simd.pand" in str(emit_plan(plan)) or "tl.simd.pnot" in str(emit_plan(plan))


# ---------------------------------------------------------------------------
# 设计层审计回归（R1–R7）
#
# 与上面的 D* 缺陷不同，这一组是「统一模型 / 发射契约」层面的问题：多为
# **build 成功但语义错误**（静默错发）或 **build 成功、emit 崩溃**。每个用例钉住
# 修复后的正确契约：要么拒绝，要么发射出与标量语义等价的形态。
# ---------------------------------------------------------------------------


def test_predicate_combine_follows_the_data_element_width():
    """R1：谓词-谓词运算（pand/por/pnot）的执行掩码必须跟随**数据元素**粒度。

    ``(100,)`` float16 → lanes = 128、``tail = 100``。用户谓词（``vcmp``，b16）与尾块
    谓词（``update_mask``，b16）做 ``pand`` 时若用固定 b32（64 lane），lane 64..127 会被
    清零，尾块只写一半。修复后掩码宽度只剩 b16。
    """
    M = 100

    @T.prim_func
    def k(A: T.Tensor((M,), "float16"), B: T.Tensor((M,), "float16"), C: T.Tensor((M,), "float16")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float16")
            b = T.alloc_shared((M,), "float16")
            c = T.alloc_shared((M,), "float16")
            for i in T.Parallel(M):
                if a[i] > b[i]:
                    c[i] = a[i]

    segment = _single_segment(k)
    assert segment.lanes == 128 and segment.tail == 100
    stmt = emit_plan(segment)
    assert "tl.simd.pand" in _emitted_ops(stmt)
    widths = {int(arg.value) for arg in _call_args(stmt, "tl.simd.pset") if isinstance(arg, tirx.IntImm)}
    assert widths == {16}, f"predicate ops must run at the data element width (b16), got {widths}"


def test_lane_uniform_store_condition_is_broadcast_as_a_predicate():
    """R2：``if a[0] > 0.0: c[i] = a[i]`` 的条件沿向量轴均匀。

    lane-uniform 条件不能落成**标量比较**（``__simd_vf__`` 内禁止标量指令），也不该
    当数据向量直接用。正确路径是向量谓词：``vdup(a[0])`` + ``vcmps`` 生成全 0 / 全 1
    的掩码，再与尾块掩码 ``pand``（``a[0]`` 的标量读只作为 ``vdup`` 的操作数）。
    """
    M = 100

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                if a[0] > 0.0:
                    c[i] = a[i]

    statement = emit_plan(_single_segment(k))  # build 成功 ⇒ emit 成功
    ops = _emitted_ops(statement)
    assert "tl.simd.vcmps" in ops, "the uniform guard is a vector compare"
    assert "tl.simd.pand" in ops, "conjoined with the tail mask (100 < 128)"
    assert "tl.simd.update_mask" in ops  # 尾块谓词仍然存在


def test_conditional_reduction_is_rejected():
    """R3：``if cond: s[0] += a[i]`` 的跨 lane 归约掩码只承载尾块谓词。

    旧实现**静默丢掉**用户谓词、对全部 lane 求和（结果错误）；现在 build 期显式拒绝。
    """
    M = 100

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), S: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            s = T.alloc_shared((1,), "float32")
            for i in T.Parallel(M):
                if i < 64:
                    s[0] = s[0] + a[i]

    with _rejects("reduce.conditional"):
        _single_segment(k)


def test_repeated_old_value_inside_a_reduction_is_rejected():
    """R4：``s[0] = s[0] + (s[0] + a[i,j])`` 的旧值出现两次。

    ``_strip_reduce`` 沿归约脊把两处旧值都剥掉，只留一个初值 → 会丢掉一份 ``s[0]``。
    现在按「剥离次数必须恰好 1」显式拒绝。
    """
    M, K = 4, 128

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), S: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            s = T.alloc_shared((1,), "float32")
            for i in T.Parallel(M):
                for j in T.serial(K):
                    s[0] = s[0] + (s[0] + a[i, j])

    with pytest.raises(Unsupported, match="appears more than once"):
        _single_segment(k)


def test_serial_loop_with_non_zero_lower_bound_is_rejected():
    """R5：迭代域建模成 ``0 <= x < extent``，``T.serial(1, K)`` 的下界会被丢掉。

    旧实现发射 ``for j in range(K-1)``（从 0 开始），下标整体错位一格。现在明确拒绝。
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(1, KS):
                    c[i, j] = a[i, j] * 2.0

    with pytest.raises(Unsupported, match="must start at 0"):
        _single_segment(k)


def test_multiple_writes_to_one_buffer_inside_a_shell_keep_source_order():
    """R6（更新）：同一 serial 体内 ``c[i,2*j]`` 与 ``c[i,j]`` 的多写不再被拒绝。

    这里 j 无法被吸收成轴（同一 buffer 的多条写目的地址不同），但它也不再需要被吸收：
    shell 作为**真实标量循环**保留，循环内两条向量 store 按源码顺序发射，跨迭代的
    「后写覆盖」由循环执行顺序保证，lane 之间仍各自写自己的地址。
    """
    MS, KS = 4, 128

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, 2 * KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, 2 * KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, 2 * j] = a[i, j]
                    c[i, j] = 2.0

    plan = _single_segment(k)
    assert [(str(axis.var), int(axis.extent)) for axis in plan.shells] == [("j", KS)]
    assert plan.vector_axis.name == "i"
    text = str(emit_plan(build_plan(_first_parallel(k))))
    assert "for j in range(128):" in text
    scatter_at_j2 = text.index("c[0, j * 2]")
    scatter_at_j = text.index("c[0, j], ")
    assert scatter_at_j2 < scatter_at_j


def test_non_absorbable_shell_is_kept_as_a_scalar_loop():
    """A shell the planner cannot absorb (last-write-wins) stays a real loop.

    ``c[i] = S[i, j]`` writes one address per lane, so ``j`` is neither a vector
    axis nor a reduction; the plan keeps it as a scalar shell around the chunk
    program instead of rejecting the nest (PR-2's shape).
    """
    BR, BC = 128, 128

    @T.prim_func
    def k(A: T.Tensor((BR, BC), "float32"), C: T.Tensor((BR,), "float32")):
        with T.Kernel(1):
            s = T.alloc_shared((BR, BC), "float32")
            c = T.alloc_shared((BR,), "float32")
            for i in T.Parallel(BR):
                for j in T.serial(BC):
                    c[i] = s[i, j]

    plan = _single_segment(k)
    assert [(str(axis.var), int(axis.extent)) for axis in plan.shells] == [("j", BC)]
    text = str(emit_plan(plan))
    # The shell is outside the chunk program: ``j`` wraps ``chunk``. The body is
    # a sequence (chunk loop + barrier), so the printer keeps both loops apart.
    assert "for j in range(128):" in text
    assert "for chunk in range(2):" in text
    assert text.index("for j in range(128):") < text.index("for chunk in range(2):")
    # 每次 shell 迭代末尾补 VST->VLD 屏障：跨迭代读改写不能竞争。
    assert text.count("tl.simd.mem_bar") == 1
    assert "VST_VLD" in text


def test_runtime_shell_bound_is_kept_as_a_scalar_loop():
    """A lane-invariant runtime trip count is carried by the plan, not rejected."""
    BR, BC = 128, 128

    @T.prim_func
    def k(A: T.Tensor((BR, BC), "float32"), C: T.Tensor((BR,), "float32")):
        with T.Kernel(2) as bx:
            s = T.alloc_shared((BR, BC), "float32")
            c = T.alloc_shared((BR,), "float32")
            t = T.min(BC, BC + bx)
            for i in T.Parallel(BR):
                for j in T.serial(t):
                    c[i] = s[i, j]

    plan = _single_segment(k)
    assert len(plan.shells) == 1
    assert str(plan.shells[0].var) == "j"
    assert not isinstance(plan.shells[0].extent, tirx.IntImm)
    text = str(emit_plan(plan))
    assert "for j in range(t):" in text
    assert "for chunk in range(2):" in text


def test_lane_dependent_shell_bound_is_rejected():
    """A shell bound that reads the lane index has no lane-uniform vector form."""
    BR = 64

    @T.prim_func
    def k(A: T.Tensor((BR, BR), "float32"), C: T.Tensor((BR,), "float32")):
        with T.Kernel(1):
            s = T.alloc_shared((BR, BR), "float32")
            c = T.alloc_shared((BR,), "float32")
            for i in T.Parallel(BR):
                for j in T.serial(i + 1):
                    c[i] = s[i, j]

    with pytest.raises(Unsupported):
        _single_segment(k)


def test_parallel_axis_reduction_with_non_uniform_pre_falls_back():
    """R7：沿**并行轴**做归约时，``pre``/``post`` 由 ``_emit_outer_store`` 发射并要求 lane 均匀。

    旧实现只在「沿 serial 轴」分支检查该门禁，于是 ``t[i] = 1.0``（沿 i 连续）+
    沿 i 的归约在 build 期通过、emit 期抛 ``Unsupported``。现在 build 期拒绝该轴，
    规划器回退到 serial 轴 ``j``（正确的归约），并且诊断里保留回退原因。
    """
    M, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((KS, M), "float32"), Tt: T.Tensor((M,), "float32"), S: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((KS, M), "float32")
            t = T.alloc_shared((M,), "float32")
            s = T.alloc_shared((1,), "float32")
            for i in T.Parallel(M):
                t[i] = 1.0
                for j in T.serial(KS):
                    s[0] = s[0] + a[j, i]

    program = _plan(k)
    segment = program.segments[0]
    assert segment.vector_axis.name == "j"
    assert any("-> fallback" in note for note in segment.diagnostics)
    emit_plan(program)  # build 成功 ⇒ emit 成功


def test_multi_segment_shares_prologue_and_passes_the_accumulator_register():
    """多段 program 的两类段间冗余：重复的 chunk 无关绑定、以及「回写后立刻重读」。

    段 0 把 ``s[i] += a[i,j]`` 累加进 ``local.var`` 寄存器并回写内存；段 1 立刻
    ``c[i,l] = s[i]*2``。修复后：

    * 全 lane 掩码与步长斜坡各只绑一次（``pset`` / ``vci`` 各 1 条）；
    * 段 1 的 ``vld s`` 直通段 0 的累加器寄存器（``vld`` 只剩段 0 的初值装载）。
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] + a[i, j]
                for l in T.serial(KS):
                    c[i, l] = s[i] * 2.0

    program = _plan(k)
    assert isinstance(program, VectorProgram) and len(program.segments) == 2
    ops = _emitted_ops(emit_plan(program))
    assert ops.count("tl.simd.pset") == 1, "全 lane 掩码应在 program 层只绑一次"
    assert ops.count("tl.simd.vci") == 1, "同一 stride 的步长斜坡应跨段复用"
    assert ops.count("tl.simd.vld") == 1, "段 1 的 s 读应直通段 0 的累加器寄存器"


def test_multi_segment_carry_is_not_used_across_chunks():
    """反例：段 0/1 都展平成 8 个 chunk（lane 基址逐 chunk 变化）时**不能**直通。

    ``p_serial_compose`` 的每一段各自跑 ``chunk_s<k>`` 循环；段 1 读段 0 写的 ``c``
    必须重新 ``vld``（两次）——直通只允许「单 chunk、lane 基址恒 0」的段。
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        C: T.Tensor((MS, KS), "float32"),
        D: T.Tensor((MS, KS), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            d = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    d[i, l] = c[i, l] + 1.0

    program = _plan(k)
    assert program.segments[0].chunks == 8 and program.segments[1].chunks == 8
    ops = _emitted_ops(emit_plan(program))
    assert ops.count("tl.simd.pset") == 1, "掩码共享仍然生效"
    assert ops.count("tl.simd.vld") == 2, "多 chunk 段之间不得直通寄存器"


def test_carry_is_invalidated_by_an_intermediate_write():
    """段间直通必须被中间写作废：``s[i] = 0.0`` 之后的重读不能再吃旧寄存器。

    段 0 的归约回写登记了 ``s[i]`` 的寄存器值；段 1 的 pre 覆写 ``s[i] = 0.0``
    会 ``note_write`` 掉该 buffer 的全部直通项，因此段 2 的 ``vld s`` 必须真实存在。
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] + a[i, j]
                s[i] = 0.0
                for l in T.serial(KS):
                    c[i, l] = s[i] * 2.0

    program = _plan(k)
    ops = _emitted_ops(emit_plan(program))
    # 段 0 的累加器初值装载 + 中间覆写后段 2 的真实重读。
    assert ops.count("tl.simd.vld") == 2, "中间写之后不得继续复用段 0 的寄存器"


def test_carry_tracks_the_newest_producer():
    """链式归约：段 1 的累加器初值可直通段 0 的寄存器，但消费段必须用段 1 的值。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        B: T.Tensor((MS, KS), "float32"),
        S: T.Tensor((MS,), "float32"),
        C: T.Tensor((MS, KS), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] + a[i, j]
                for j in T.serial(KS):
                    s[i] = s[i] + b[i, j]
                for l in T.serial(KS):
                    c[i, l] = s[i] * 2.0

    program = _plan(k)
    assert len(program.segments) == 3
    stmt = emit_plan(program)
    ops = _emitted_ops(stmt)
    assert ops.count("tl.simd.vld") == 1, "段 1 的初值与段 2 的读都应由直通/最新寄存器满足"
    assert "accs1_0[0]" in str(stmt), "消费段必须使用段 1 回写的寄存器（最新生产者）"
