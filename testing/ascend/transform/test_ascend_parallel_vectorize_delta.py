"""Offline structural tests for P2 planning: access classification and body shapes.

Pins the mathematical core (``finite_difference`` / ``classify_delta``) and the
planning scenarios derived from it: if/select, reductions, multiple stores,
serial-shell absorption, multi-segment programs and diagnostics.
Hardware independent; emission-width and NZ details live in the sibling files.
"""

from __future__ import annotations


import pytest

import tilelang.ascend.language as T
import tvm
from tvm import arith, tirx

from tilelang.ascend.analysis.parallel_vectorize import (
    AccessClass,
    Unsupported,
    VectorProgram,
    build_plan,
    classify_delta,
    emit_plan,
    finite_difference,
    render_plan,
)
from testing.ascend.transform._parallel_vectorize_helpers import (
    _emitted_ops,
    _eval_lane,
    _for_kinds,
    _outer_parallel_nests,
    _over_reading_contiguous_load,
    _plan,
    _recipes,
    _rejects,
    _single_segment,
)


# ---------------------------------------------------------------------------
# the mathematical core: Δ(offset) decides everything
# ---------------------------------------------------------------------------


def test_finite_difference_classifies_contiguous():
    analyzer = arith.Analyzer()
    i = tirx.Var("i", "int32")
    assert classify_delta(finite_difference(i, i, analyzer)) == (AccessClass.CONTIGUOUS, 1)


def test_finite_difference_classifies_uniform():
    analyzer = arith.Analyzer()
    i = tirx.Var("i", "int32")
    assert classify_delta(finite_difference(i * 0 + 7, i, analyzer)) == (AccessClass.UNIFORM, 0)


def test_finite_difference_classifies_strided():
    analyzer = arith.Analyzer()
    i = tirx.Var("i", "int32")
    assert classify_delta(finite_difference(i * 2, i, analyzer)) == (AccessClass.STRIDED, 2)


def test_finite_difference_absorbs_flatten_identity():
    """``i//16*16 + i%16 == i`` after simplification, so Δ == 1 without any
    template matching (this is what makes §4 Case II a *math* result)."""
    analyzer = arith.Analyzer()
    i = tirx.Var("i", "int32")
    flattened = tirx.floordiv(i, 16) * 16 + tirx.floormod(i, 16)
    assert classify_delta(finite_difference(flattened, i, analyzer)) == (AccessClass.CONTIGUOUS, 1)


def test_finite_difference_classifies_non_constant_as_piecewise():
    """``i*i`` has Δ == 2i+1, not a constant → ``PIECEWISE`` at the Δ layer.

    The Δ layer only decides *constant vs non-constant*; the structural
    refinement (constructible index vector → ``PIECEWISE`` / ``INDIRECT`` /
    ``PERMUTE``, otherwise ``TRUE_RANDOM``) happens in
    :func:`classify_access` (改进 4).
    """
    analyzer = arith.Analyzer()
    i = tirx.Var("i", "int32")
    assert classify_delta(finite_difference(i * i, i, analyzer))[0] == AccessClass.PIECEWISE


def test_const_int_proves_constant_beyond_literals():
    """``Δ`` need not already be a literal ``IntImm``: with an analyzer,
    ``(i//4 - i//4) + 3`` is *proved* to be the constant 3, so it classifies as
    ``STRIDED`` with stride 3 instead of falling into ``PIECEWISE``.  Without an
    analyzer the literal-only path cannot fold it."""
    analyzer = arith.Analyzer()
    i = tirx.Var("i", "int32")
    expr = (tirx.floordiv(i, 4) - tirx.floordiv(i, 4)) + 3
    assert classify_delta(expr, analyzer) == (AccessClass.STRIDED, 3)
    assert classify_delta(expr) == (AccessClass.PIECEWISE, None)


# ---------------------------------------------------------------------------
# §0 basic shapes
# ---------------------------------------------------------------------------


def test_copy():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.chunks == 2 and plan.tail == 0  # float32 -> 64 lanes, N=128
    assert plan.stores[0].kind == AccessClass.CONTIGUOUS
    assert _recipes(plan.stores[0].value) == ["vload"]

    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vld" in ops and "tl.simd.vsts" in ops


def test_scalar_broadcast():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[0]

    plan = _single_segment(k)
    assert plan.stores[0].value.recipe == "uniform_load"
    assert "tl.simd.vdup" in _emitted_ops(emit_plan(plan))


def test_fragment_nest_is_rejected_by_build_plan():
    """§0 Case III guard: a ``local.fragment`` operand has no memory form.

    P1 keeps such a nest out of P2/P3 entirely (it is reported OPAQUE, see
    ``test_ascend_vsimd_lift.py``), so the planner only needs to defend its own
    API: planning the nest directly must be refused instead of emitting an
    un-compilable ``vld``/``vsts`` pair against a V-core register tile.
    """
    N = 64

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float32")
            c_ub = T.alloc_shared((N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                frag = T.alloc_fragment((N,), "float32")
                for i in T.Parallel(N):
                    frag[i] = a_ub[i] + 1.0
                for i in T.Parallel(N):
                    c_ub[i] = frag[i] * 2.0
            T.copy(c_ub, C)

    for nest in _outer_parallel_nests(k):
        with pytest.raises(Unsupported, match="local.fragment"):
            build_plan(nest)


def test_if_else_merges_into_select():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if a[i] == 0.0:
                    c[i] = 0.0
                else:
                    c[i] = T.exp(b[i])

    plan = _single_segment(k)
    assert plan.stores[0].mask is None  # both arms write the same location
    assert plan.stores[0].value.recipe == "vsel"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vsel" in ops and "tl.simd.vcmps" in ops and "tl.simd.vexp" in ops


def test_if_single_arm_is_predicated_store():
    P = 8

    @T.prim_func
    def k(A: T.Tensor((P, P), "float32"), C: T.Tensor((P, P), "float32")):
        with T.Kernel(1):
            c = T.alloc_shared((P, P), "float32")
            for i, j in T.Parallel(P, P):
                if i < j:
                    c[i, j] = 0.0

    plan = _single_segment(k)
    # ``(8, 8)`` 的最内层只有 8 个 lane（寄存器是 64）→ P1 合成连续轴把它整条展平。
    assert plan.vector_axis.name.startswith("k_")
    store = plan.stores[0]
    assert store.mask is not None
    # 展平后 i、j 都随合成轴变化，``i < j`` 变成逐 lane 比较（不再是标量 vcmps）。
    assert store.mask.recipe == "vcmp"
    ops = _emitted_ops(emit_plan(plan))
    # 展平后 8x8 == 64 lane 恰好一个整 chunk，不再需要尾块谓词。
    assert "tl.simd.vcmp" in ops and "tl.simd.update_mask" not in ops


def test_statement_if_arm_load_is_guarded():
    """D16：语句级单臂 if 的臂内访存也是惰性的。``a[i-32]`` 在 lane 0..31 越界，
    装载必须挂上 store 谓词（发带掩码的 ``vgather2``），而不是整 lane ``vld a[-32]``。
    守卫在 chunk 1（``i ∈ [64,128)``）恒真 → 只有 chunk 0 付 ``vgather2``。"""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if i >= 32:
                    c[i] = a[i - 32]

    plan = _single_segment(k)
    store = plan.stores[0]
    assert store.mask is not None
    assert store.value.recipe == "vload" and store.value.predicate is not None
    assert plan.guard_boundaries == [1]  # chunk 1 起守卫恒真
    ops = _emitted_ops(emit_plan(plan))
    assert ops.count("tl.simd.vgather2") == 1  # chunk 0：越界风险段，带掩码
    assert ops.count("tl.simd.vld") == 1  # chunk 1：守卫恒真，回到普通 vld
    assert ops.count("tl.simd.vcmps") == 1  # 谓词只在带守卫的 chunk 0 计算


def test_statement_if_prefix_guard_is_dropped_before_boundary():
    """D16 镜像：**前缀**恒真守卫。``if i < 96: c[i] = a[i+32]`` 的 ``a[i+32]`` 只在
    ``i < 96`` 时安全（``i >= 96`` 会读到 ``a[128..159]``），守卫在 chunk 0 恒真 →
    chunk 0 退回普通 ``vld``，chunk 1 仍走带掩码的 ``vgather2``。"""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if i < 96:
                    c[i] = a[i + 32]

    plan = _single_segment(k)
    store = plan.stores[0]
    assert store.mask is not None
    assert store.value.recipe == "vload" and store.value.predicate is not None
    assert plan.guard_boundaries == [1]  # chunk 0 守卫恒真
    ops = _emitted_ops(emit_plan(plan))
    assert ops.count("tl.simd.vld") == 1  # chunk 0：守卫恒真，普通 vld
    assert ops.count("tl.simd.vgather2") == 1  # chunk 1：越界风险段，带掩码
    assert ops.count("tl.simd.vcmps") == 1  # 谓词只在带守卫的 chunk 1 计算


def test_guard_boundaries_cut_prefix_and_suffix_independently():
    """前缀与后缀守卫同段共存：两个切点 ``[1, 7]`` 把 8 个 chunk 切成三段。

    ``c`` 的后缀守卫（``i >= 32``）从 chunk 1 起恒真，``d`` 的前缀守卫
    （``i < 480``）到 chunk 6 为止恒真；两段恒真区间互补，只有 chunk 0 与 chunk 7
    各需要一次带掩码的 ``vgather2``。
    """
    M = 512

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((M,), "float32"), D: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((M,), "float32")
            d = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                if i >= 32:
                    c[i] = a[i - 32]
                if i < 480:
                    d[i] = a[i + 32]

    plan = _single_segment(k)
    assert plan.guard_boundaries == [1, 7]
    ops = _emitted_ops(emit_plan(plan))
    # chunk 0：c 带守卫 + d 恒真；chunk 1..6：两条都恒真；chunk 7：c 恒真 + d 带守卫。
    assert ops.count("tl.simd.vgather2") == 2
    assert ops.count("tl.simd.vld") == 4
    assert ops.count("tl.simd.vcmps") == 2


def test_statement_if_else_merge_keeps_branch_laziness():
    """D16：同址 if/else 合并成**惰性** ``if_then_else``（不是急切 ``Select``），
    真臂的越界读带条件守卫；常量 else 臂无装载。"""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if i >= 32:
                    c[i] = a[i - 32]
                else:
                    c[i] = 0.0

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vsel"
    assert value.args[1].recipe == "vload" and value.args[1].predicate is not None
    ops = _emitted_ops(emit_plan(plan))
    assert ops.count("tl.simd.vgather2") == 1
    assert ops.count("tl.simd.vld") == 1


def test_statement_if_safe_load_stays_unmasked():
    """D16 的细化：臂内地址**可证安全**时不加守卫，仍发普通 ``vld``（无性能回退）。"""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if a[i] != 0.0:
                    c[i] = a[i]

    plan = _single_segment(k)
    assert plan.stores[0].value.recipe == "vload" and plan.stores[0].value.predicate is None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vld" in ops and "tl.simd.vgather2" not in ops


def test_ternary():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = T.if_then_else(a[i] > b[i], a[i], b[i])

    plan = _single_segment(k)
    assert plan.stores[0].value.recipe == "vsel"
    # P0-3：两个臂的地址都能证明在 buffer 内 → 不需要谓词化，仍发普通 ``vld``。
    assert plan.stores[0].value.args[1].predicate is None
    assert plan.stores[0].value.args[2].predicate is None
    ops = _emitted_ops(emit_plan(plan))
    # a lane-varying condition compares per lane (vcmp); it is *not* broadcast.
    assert "tl.simd.vcmp" in ops
    assert "tl.simd.update_mask" not in ops
    assert "tl.simd.vgather2" not in ops


def test_lazy_conditional_with_unsafe_load_is_predicated():
    """P0-3：``if_then_else(i > 0, a[i-1], 0)`` 的 ``a[i-1]`` 在 lane 0 越界，
    所以真臂的装载必须带条件守卫（发带谓词的 ``vgather2``），而不是整 lane ``vld``。"""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = T.if_then_else(i > 0, a[i - 1], 0.0)

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vsel" and not value.is_uniform
    true_arm = value.args[1]
    assert true_arm.recipe == "vload" and true_arm.predicate is not None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vgather2" in ops  # 带守卫的装载通道
    assert "tl.simd.vcmps" in ops  # 条件本身（i > 0）


def test_lazy_conditional_with_unpredicable_uniform_load_is_rejected():
    """P0-3：惰性条件臂里的**lane 均匀**间接读无法谓词化（没有带掩码的标量读），
    build 期必须明确拒绝，而不是发一个无条件标量读。"""
    N = 128

    @T.prim_func
    def k(
        A: T.Tensor((N,), "float32"),
        D: T.Tensor((16,), "float32"),
        IDX: T.Tensor((1,), "int32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            d = T.alloc_shared((16,), "float32")
            idx = T.alloc_shared((1,), "int32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = T.if_then_else(a[i] > 0.0, d[idx[0]], 0.0)

    with pytest.raises(Unsupported, match="cannot be predicated"):
        _plan(k)


def test_lazy_conditional_with_safe_uniform_load_is_allowed():
    """P0-3 的细化：臂里的标量读地址**可证安全**时不必拒绝，也没必要带守卫。"""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((16,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((16,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = T.if_then_else(a[i] > 0.0, b[0] + 1.0, 0.0)

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.args[1].is_uniform  # b[0] + 1 折叠成标量 + vdup
    assert "tl.simd.vsel" in _emitted_ops(emit_plan(plan))


def test_ternary_with_loop_invariant_condition_broadcasts_predicate():
    """§1 Case II as written in the catalogue: the condition depends only on the
    outer axis ``i_s2``, so it is uniform along the vector axis ``i_v`` → the
    whole chunk takes the same branch.  It is materialised as a vector predicate
    (broadcast operand + ``vcmps``) fed to ``vsel`` — no branch hoisting, and no
    scalar compare (a scalar instruction inside ``__simd_vf__`` is refused by
    bisheng)."""
    S, DV = 16, 128

    @T.prim_func
    def k(G: T.Tensor((S,), "float32"), D: T.Tensor((S, DV), "float32")):
        with T.Kernel(1):
            g = T.alloc_shared((S,), "float32")
            dv = T.alloc_shared((S, DV), "float32")
            for i_s2, i_v in T.Parallel(S, DV):
                dv[i_s2, i_v] = T.if_then_else(10.0 - g[i_s2] <= 0, dv[i_s2, i_v] * g[i_s2], 0.0)

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vsel" and value.detail == "uniform-cond select"
    assert value.args[0].recipe == "vcmps"  # 向量谓词，不是 bool 标量
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcmps" in ops and "tl.simd.vsel" in ops
    assert "tl.simd.update_mask" not in ops, "no scalar-select mask"
    assert "tl.simd.vmuls" in ops and "tl.simd.vdup" in ops
    assert "tl.simd.vcmp" not in ops  # the condition is not compared per lane


def test_select_node_with_loop_invariant_condition_broadcasts_predicate():
    """Same path through the ``tirx.Select`` node (not ``T.if_then_else``)."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = tirx.Select(a[0] > 0.0, a[i], b[i])

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vsel" and value.detail == "uniform-cond select"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcmps" in ops and "tl.simd.vsel" in ops
    assert "tl.simd.update_mask" not in ops
    assert "tl.simd.vcmp" not in ops


def test_loop_invariant_condition_with_uniform_arm_broadcasts_and_dedups():
    """A uniform condition whose false arm is a scalar keeps working: the arm is
    materialised by ``vdup``, and the predicate is the vector one (broadcast +
    ``vcmps``)."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = T.if_then_else(a[0] > 0.0, b[i], 0.0)

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vsel" and value.args[2].is_uniform
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcmps" in ops and "tl.simd.vdup" in ops and "tl.simd.vsel" in ops
    assert "tl.simd.update_mask" not in ops


def test_uniform_select_is_a_vector_select_over_broadcast_arms():
    """When condition *and* both arms are loop-invariant, the select is still
    materialised with vector instructions: the arms are broadcast and the
    condition becomes a ``vcmps`` mask. A scalar ``?:`` (the old scalar-fold
    shape) would be an AIV-loop scalar instruction, which bisheng refuses."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = T.if_then_else(a[0] > 0.0, b[0], 2.0)

    plan = _single_segment(k)
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vsel" in ops and "tl.simd.vcmps" in ops
    assert "tl.simd.vdup" in ops and "tl.simd.vsts" in ops


# ---------------------------------------------------------------------------
# §2 operand rank mismatch / reduction
# ---------------------------------------------------------------------------


def test_operand_rank_mismatch_broadcasts_outer_load():
    M, K = 16, 128

    @T.prim_func
    def k(S: T.Tensor((K,), "float32"), C: T.Tensor((M, K), "float32")):
        with T.Kernel(1):
            g = T.alloc_shared((K,), "float32")
            d = T.alloc_shared((M, K), "float32")
            for i, j in T.Parallel(M, K):
                d[i, j] = g[i] - g[j]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "j"
    value = plan.stores[0].value
    # SIMD 层没有 vsubs：``uniform - vector`` 只能发 ``vsub(vdup(uniform), vector)``。
    # 只有 ``vector - scalar`` 才允许折叠成 ``vadds(vector, -scalar)``——反向折叠会
    # 得到 ``vector - scalar``（符号相反，P0-1）。
    assert value.recipe == "vsub"
    assert value.args[0].recipe == "vbrc"
    assert value.args[0].args[0].recipe == "uniform_load"
    assert value.args[1].recipe == "vload"
    # 数值 golden：i=3 时 lane j 必须是 3 - j（而不是 j - 3）。
    i_var = plan.axes[0].var
    got = [_eval_lane(value, lane, plan, env={i_var: 3}) for lane in range(8)]
    assert got == [3 - lane for lane in range(8)]
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vsub" in ops and "tl.simd.vdup" in ops and "tl.simd.vld" in ops
    assert "tl.simd.vadds" not in ops


def test_vector_minus_scalar_still_folds_to_vadds():
    """P0-1 的反向护栏：``vector - scalar`` 仍然要折叠成 ``vadds(vector, -scalar)``
    （标量取负是标量开销），不能被上一处修复误伤。"""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i] - 1.0

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vadds"
    assert value.args[0].recipe == "vload" and value.args[1].recipe == "uniform_expr"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vadds" in ops and "tl.simd.vsub" not in ops


def test_mixed_element_width_segment_is_rejected():
    """P0-2 + Stage 1：一段共享一个 chunk 宽度（chunk 步长 = lanes）。f16 与 int32
    混写时后者每个 chunk 只能覆盖一半元素，会静默漏写/越界；因此宽于 chunk 的
    store 必须是窄化转换（``vcvt`` 打包写），普通 store 仍然必须拒绝。"""
    N = 128

    @T.prim_func
    def k(
        AF: T.Tensor((N,), "float16"),
        AI: T.Tensor((N,), "int32"),
        CF: T.Tensor((N,), "float16"),
        CI: T.Tensor((N,), "int32"),
    ):
        with T.Kernel(1):
            af = T.alloc_shared((N,), "float16")
            ai = T.alloc_shared((N,), "int32")
            cf = T.alloc_shared((N,), "float16")
            ci = T.alloc_shared((N,), "int32")
            for i in T.Parallel(N):
                cf[i] = af[i]
                ci[i] = ai[i]

    with pytest.raises(Unsupported, match="not a narrowing conversion"):
        _plan(k)


def test_same_width_mixed_element_types_derive_per_load_dtype():
    """P0-2：f32 / int32 位宽相同可以同段；但每个值节点的 dtype 必须取**被访问
    buffer** 的元素类型（int32 的装载不能声称 ``float32x64``）。"""
    N = 128

    @T.prim_func
    def k(
        AF: T.Tensor((N,), "float32"),
        AI: T.Tensor((N,), "int32"),
        CF: T.Tensor((N,), "float32"),
        CI: T.Tensor((N,), "int32"),
    ):
        with T.Kernel(1):
            af = T.alloc_shared((N,), "float32")
            ai = T.alloc_shared((N,), "int32")
            cf = T.alloc_shared((N,), "float32")
            ci = T.alloc_shared((N,), "int32")
            for i in T.Parallel(N):
                cf[i] = af[i]
                ci[i] = ai[i]

    plan = _single_segment(k)
    assert plan.lanes == 64
    assert [store.value.dtype for store in plan.stores] == ["float32x64", "int32x64"]
    ops = _emitted_ops(emit_plan(plan))
    assert ops.count("tl.simd.vld") == 2 and ops.count("tl.simd.vsts") == 2


def test_reduction_detected():
    M, K = 4, 256

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            acc = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                acc[i] = acc[i] + a[i, j]

    plan = _single_segment(k)
    assert plan.reduce is not None
    assert plan.reduce.op == "vadd" and plan.reduce.recipe == "vcadd"
    assert plan.vector_axis.name == "j"
    # the store itself is lane-uniform (dst rank < src rank).
    assert plan.stores[0].kind == AccessClass.UNIFORM
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcadd" in ops and "tl.simd.vsts" in ops


def test_reduction_max_maps_to_vcmax():
    M, K = 4, 256

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            acc = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                acc[i] = T.max(acc[i], a[i, j])

    plan = _single_segment(k)
    assert plan.reduce is not None and plan.reduce.recipe == "vcmax"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcmax" in ops
    # accumulate form (``dst = max(dst, x)``) merges the old destination value.
    assert "tl.simd.vld" in ops


def test_reduction_with_tail_fills_identity():
    """``extent % lanes != 0``: the last chunk is a partial tile.  Its inactive
    lanes read out of range, so the tail predicate must be applied.  For **Add**
    the identity is 0 and ``MODE_ZEROING`` already zeroes inactive lanes, so the
    predicate is passed straight to ``vcadd`` (no ``vsel`` fill); Max/Min still
    need an explicit identity fill."""
    M, K = 4, 32  # K < 64 lanes -> tail != 0

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            acc = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                acc[i] = acc[i] + a[i, j]

    plan = _single_segment(k)
    assert plan.reduce is not None and plan.tail == 32
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.update_mask" in ops  # 尾块谓词
    assert "tl.simd.vsel" not in ops  # Add 不需要单位元填充
    assert "tl.simd.vcadd" in ops  # 跨 lane 归约（带尾块掩码）
    assert "tl.simd.vsts" in ops  # ONEPT 回写


# ---------------------------------------------------------------------------
# §3 multiple statements / scalar temporaries
# ---------------------------------------------------------------------------


def test_multi_store():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32"), D: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            d = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i]
                d[i] = a[i] * 2.0

    plan = _single_segment(k)
    assert len(plan.stores) == 2
    assert plan.stores[0].value.recipe == "vload"
    assert plan.stores[1].value.recipe == "vmuls"
    assert _emitted_ops(emit_plan(plan)).count("tl.simd.vsts") == 2


def test_store_strided_on_the_only_axis_is_scattered():
    """§3 Case I as written in the catalogue: ``g[i, 0]`` (shape ``(BM, 2)``) keeps
    the trailing index fixed, so Δ == 2 along the only axis → ``STRIDED``.  Since
    strided stores are now first-class in the index-vector channel, the axis *is*
    a candidate and the write is emitted as ``vscatter`` (改进 1)."""
    BM = 64

    @T.prim_func
    def k(A: T.Tensor((BM,), "float32"), C: T.Tensor((BM, 2), "float32")):
        with T.Kernel(1):
            mv = T.alloc_shared((BM,), "float32")
            g = T.alloc_shared((BM, 2), "float32")
            for i in T.Parallel(BM):
                g[i, 0] = mv[i]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.stores[0].kind == AccessClass.STRIDED
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vscatter" in ops and "tl.simd.vld" in ops


def test_multistore_with_unit_trailing_dim_is_contiguous():
    """The same multi-store shape becomes vectorizable once the trailing
    dimension is 1 (``g[i, 0]`` → ``Δ == 1``): the shape, not the code text,
    decides."""
    BM = 64

    @T.prim_func
    def k(A: T.Tensor((BM,), "float32"), B: T.Tensor((BM,), "float32"), C: T.Tensor((BM, 1), "float32"), D: T.Tensor((BM, 1), "float32")):
        with T.Kernel(1):
            mv = T.alloc_shared((BM,), "float32")
            mi = T.alloc_shared((BM,), "float32")
            g = T.alloc_shared((BM, 1), "float32")
            i2 = T.alloc_shared((BM, 1), "float32")
            for i in T.Parallel(BM):
                g[i, 0] = mv[i]
                i2[i, 0] = mi[i]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert [s.kind for s in plan.stores] == [AccessClass.CONTIGUOUS, AccessClass.CONTIGUOUS]
    assert _emitted_ops(emit_plan(plan)).count("tl.simd.vsts") == 2


def test_scalar_alias_folds_into_value():
    S, K = 16, 128

    @T.prim_func
    def k(A: T.Tensor((S, K), "float32"), C: T.Tensor((S, K), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((S, K), "float32")
            c = T.alloc_shared((S, K), "float32")
            for i, j in T.Parallel(S, K):
                acc: T.float32 = a[i, j] * 2.0
                c[i, j] = acc + 1.0

    plan = _single_segment(k)
    # ``acc`` is inlined: vadds(vmuls(vload, 2.0), 1.0)
    recipes = _recipes(plan.stores[0].value)
    assert recipes[0] == "vadds" and recipes[1] == "vmuls"


def test_floordiv_value_operator_strength_reduced():
    """§3 Case II complex scalar: ``FloorDiv`` / ``FloorMod`` used as a *value*
    operator.  ``i // 4`` is division by ``2^k``, so it strength-reduces to an
    arithmetic right shift (``vshrs``) and the float store then converts it
    (``vcvt``) — no integer ``vdiv`` needed."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i] + i // 4

    plan = _single_segment(k)
    recipes = _recipes(plan.stores[0].value)
    assert "vshrs" in recipes and "vcvt" in recipes
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vshrs" in ops and "tl.simd.vcvt" in ops


def test_floormod_value_operator_strength_reduced():
    """``i % 8`` → ``vand(i, 7)``: the two's-complement bitwise-and is exactly
    ``FloorMod`` for a power-of-two divisor, so a masked ``vdup`` constant feeds
    the existing vector-vector ``vand``."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "int32"), C: T.Tensor((N,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "int32")
            c = T.alloc_shared((N,), "int32")
            for i in T.Parallel(N):
                c[i] = a[i] + i % 8

    plan = _single_segment(k)
    recipes = _recipes(plan.stores[0].value)
    assert "vand" in recipes and "vbrc" in recipes
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vand" in ops and "tl.simd.vdup" in ops


def test_floordiv_by_non_power_of_two_is_rejected():
    """A non-power-of-two constant divisor has no exact 32-bit strength reduction
    (it needs 64-bit ``vmull`` magic numbers), so it is rejected rather than
    mis-emitted as a shift."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "int32"), C: T.Tensor((N,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "int32")
            c = T.alloc_shared((N,), "int32")
            for i in T.Parallel(N):
                c[i] = a[i] + i // 3

    with pytest.raises(Unsupported, match="non-power-of-two constant 3"):
        _plan(k)


# ---------------------------------------------------------------------------
# §4 expression subscripts / flattening / strided
# ---------------------------------------------------------------------------


def test_expression_subscript():
    # PAD is a whole 32B block (8 f32 elements): a block load needs a
    # 32B-aligned base, so a non-multiple offset (e.g. 3) is rejected by P2
    # (``plan.alignment``) and the segment falls back to another axis.
    B0, LEN, PAD = 2, 128, 8

    @T.prim_func
    def k(A: T.Tensor((B0, PAD + LEN), "float32"), C: T.Tensor((B0, LEN), "float32")):
        with T.Kernel(1):
            g = T.alloc_shared((B0, PAD + LEN), "float32")
            d = T.alloc_shared((B0, LEN), "float32")
            for b0, i_t in T.Parallel(B0, LEN):
                d[b0, i_t] = g[b0, PAD + i_t]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i_t"
    assert plan.stores[0].kind == AccessClass.CONTIGUOUS
    assert plan.stores[0].value.recipe == "vload"
    assert "tl.simd.vld" in _emitted_ops(emit_plan(plan))


def test_misaligned_stream_base_is_rejected():
    """A block ``vld``/``vsts`` needs a 32B-aligned base.

    ``a[i + 4]`` starts every chunk 4 f32 elements (16B) off the block boundary,
    and the emitter folds the whole element offset into the base pointer, so the
    instruction would decode the wrong 32B block (device report: a 400B f32 row
    stride returns 507035). The emitted form has no offset operand yet, so P2
    rejects the segment instead of miscompiling it.
    """
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N + 4,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N + 4,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i + 4]

    with _rejects("plan.alignment"):
        _plan(k)


def test_expression_subscript_reads_shared_operand_as_uniform():
    """§4 Case I (UB variant): ``Gb[0, S + i_s, 0]`` does not vary along the
    vector axis ``i_k`` → lane-uniform → a single scalar read broadcast by
    ``vdup``."""
    BS, S, DK, BH = 128, 16, 64, 8

    @T.prim_func
    def k(OUT: T.Tensor((S, DK), "float32")):
        with T.Kernel(1):
            gb = T.alloc_shared((2, BS * 8, BH), "float32")
            gs = T.alloc_shared((S, DK), "float32")
            for i_s, i_k in T.Parallel(S, DK):
                gs[i_s, i_k] = gb[0, 0 * S + i_s, 0]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i_k"
    assert plan.stores[0].kind == AccessClass.CONTIGUOUS
    assert plan.stores[0].value.recipe == "uniform_load"
    assert "tl.simd.vdup" in _emitted_ops(emit_plan(plan))


def test_highdim_flatten():
    M, LEN = 2, 128

    @T.prim_func
    def k(H: T.Tensor((M, 8, 16), "float32"), D: T.Tensor((M, 8, 16), "float32"), C: T.Tensor((M, LEN), "float32")):
        with T.Kernel(1):
            h = T.alloc_shared((M, 8, 16), "float32")
            d = T.alloc_shared((M, 8, 16), "float32")
            c = T.alloc_shared((M, LEN), "float32")
            for i_kv, _ in T.Parallel(M, LEN):
                c[i_kv, _] = h[i_kv, _ // 16, _ % 16] * d[i_kv, _ // 16, _ % 16]

    plan = _single_segment(k)
    # The 3D index flattens to ``i_kv*128 + _`` → Δ == 1 without a template match.
    assert plan.vector_axis.name == "_"
    assert plan.stores[0].kind == AccessClass.CONTIGUOUS
    assert plan.stores[0].value.recipe == "vmul"
    assert _recipes(plan.stores[0].value).count("vload") == 2


def test_block_tiled_fragment_layout_is_piecewise_gather():
    """§4 Case II block-tiled/store-permute: ``cs[i//MX, j//MY, i%MX, j%MY]`` is
    a *block* permutation of the indices, so ``Δ(offset)`` is not constant even
    after flattening.  MX=MY=4 are powers of two, so the per-lane element offset
    *is* constructible → ``PIECEWISE`` → the load is emitted as ``vgather2`` with
    the absolute offset vector (改进 1 + 4), while the contiguous store stays a
    plain ``vsts``."""
    BM, BN, MX, MY = 16, 16, 4, 4

    @T.prim_func
    def k(CS: T.Tensor((BM // MX, BN // MY, MX, MY), "float32"), C: T.Tensor((BM, BN), "float32")):
        with T.Kernel(1):
            cs = T.alloc_shared((BM // MX, BN // MY, MX, MY), "float32")
            c = T.alloc_shared((BM, BN), "float32")
            for i, j in T.Parallel(BM, BN):
                c[i, j] = cs[i // MX, j // MY, i % MX, j % MY]

    plan = _single_segment(k)
    assert plan.stores[0].kind == AccessClass.CONTIGUOUS
    value = plan.stores[0].value
    assert value.recipe == "gather" and value.detail == AccessClass.PIECEWISE.value
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vgather2" in ops and "tl.simd.vsts" in ops


def test_strided_access():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((2 * N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((2 * N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[2 * i]

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vload_strided" and value.detail == "stride=2"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vgather2" in ops


def test_indirect_gather_is_expressed():
    """Indirect access ``a[b[i]]`` has a ``BufferLoad`` inside its index, so the
    offset cannot be built from the loop axes alone.  It is classified
    ``INDIRECT`` and emitted as ``vgather2`` whose index register is the loaded
    per-lane ``b[i]`` vector (改进 1)."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "int32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "int32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[b[i]]

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "gather" and value.detail == AccessClass.INDIRECT.value
    assert "tl.simd.vgather2" in _emitted_ops(emit_plan(plan))


def test_reversal_is_permute_gather():
    """``a[N - 1 - i]`` has Δ == -1 along ``i``: a bijective reversal, not a
    stride.  It is classified ``PERMUTE`` and lowered through the same per-lane
    index-vector channel (改进 1 + 4)."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[N - 1 - i]

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "gather" and value.detail == AccessClass.PERMUTE.value
    # 数值 golden：chunk 0 的每 lane 下标必须正好是 127, 126, ...（P0-1：不能是 i-127）。
    assert [_eval_lane(value, lane, plan) for lane in range(4)] == [127, 126, 125, 124]
    assert [_eval_lane(value, lane, plan, base=64) for lane in range(4)] == [63, 62, 61, 60]
    assert "tl.simd.vgather2" in _emitted_ops(emit_plan(plan))


def test_true_random_access_is_rejected():
    """``a[i // 3]`` has a non-constant Δ with a *non*-power-of-two divisor, so its
    per-lane element offset cannot be built in the index-vector channel (改进 4:
    ``TRUE_RANDOM``); the vectorizer must reject rather than silently degrade."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i // 3]

    with _rejects("plan.no_vector_axis"):
        _plan(k)


# ---------------------------------------------------------------------------
# 改进 7: diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_list_every_candidate_axis_on_success():
    """A successful plan still records *every* axis' classification plus which one
    won, so a caller can see the full decision without re-deriving it (改进 7)."""
    N = 64

    @T.prim_func
    def k(A: T.Tensor((N, N), "float32"), C: T.Tensor((N, N), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N, N), "float32")
            c = T.alloc_shared((N, N), "float32")
            for i, j in T.Parallel(N, N):
                c[i, j] = a[i, j]

    plan = _single_segment(k)
    assert any(line.startswith("axis 'i'") for line in plan.diagnostics)
    assert any(line.startswith("axis 'j'") for line in plan.diagnostics)
    assert any(line.startswith("selected vector axis") for line in plan.diagnostics)
    assert "diagnostics:" in render_plan(plan)


def test_diagnostics_are_attached_to_a_rejection():
    """When no axis works, the raised ``Unsupported`` carries the per-axis notes
    (改进 7), not just a single flattened reason."""
    N = 64

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i // 3] = a[i]  # the *store* index is not constructible

    with pytest.raises(Unsupported) as excinfo:
        _plan(k)
    notes = excinfo.value.diagnostics
    assert any("true-random store" in note for note in notes)
    assert any("not a power of two" in note for note in notes)


# ---------------------------------------------------------------------------
# §5 atomics / §6 complex control flow
# ---------------------------------------------------------------------------


def test_atomic_rejected():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((1,), "float32")
            for i in T.Parallel(N):
                T.atomic_add(c[0], a[i])

    with pytest.raises(Unsupported, match="atomic"):
        _plan(k)


def test_nested_loop_rejected():
    S, KD = 16, 16

    @T.prim_func
    def k(A: T.Tensor((S, KD), "float32"), C: T.Tensor((S, KD), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((S, KD), "float32")
            c = T.alloc_shared((S, KD), "float32")
            cnt = T.alloc_shared((1,), "int32")
            for tm in T.Parallel(S):
                cnt[0] = 0
                for g_i in T.serial(4):
                    if a[tm, g_i] != 0.0:
                        cnt[0] += 1
                        c[tm, g_i] = a[tm, g_i]

    with _rejects("body.nested_loop"):
        _plan(k)


# ---------------------------------------------------------------------------
# §6 第一步：直接内嵌、干净的 T.serial 循环被吸收成额外候选轴（S1 / S3）
# ---------------------------------------------------------------------------


def test_direct_serial_copy_absorbed():
    """S1：``for i in P: for j in serial: c[i,j] = a[i,j]*2``。serial 体无循环携带
    依赖 → j 被吸收为候选轴；写沿 j 连续 → 选 j 向量化（连续 vld/vmuls/vsts）。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0

    plan = _single_segment(k)
    # ``KS=8 < lanes=64`` → P1 合成连续轴：整条 nesting 展平成一个满 lane 连续轴，
    # 512 个元素正好 8 个整 chunk。展平的**前提**是 serial 轴已被吸收进迭代域。
    assert plan.vector_axis.name.startswith("k_")
    assert plan.vector_axis.extent.value == MS * KS
    assert plan.stores[0].kind.value == "contiguous"
    assert plan.reduce is None
    stmt = emit_plan(plan)
    ops = _emitted_ops(stmt)
    # 满 lane 连续：只有 vld/vsts，没有带谓词的 gather，也没有尾块谓词。
    assert "tl.simd.vld" in ops and "tl.simd.vsts" in ops
    assert "tl.simd.vgather2" not in ops and "tl.simd.update_mask" not in ops
    assert not _over_reading_contiguous_load(stmt)


def test_direct_serial_predicated_absorbed():
    """S1 + 谓词：serial 体内的 ``if`` 降成谓词化 store，掩码沿向量轴逐 lane 变化。

    ``KS=8 < lanes=64`` → 同 :func:`test_direct_serial_copy_absorbed`，走 P1 合成连续轴；
    谓词 ``a[i, j] != 0`` 展平后是一整条 lane 上的向量比较（``vcmps``）。
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    if a[i, j] != 0.0:
                        c[i, j] = a[i, j]

    plan = _single_segment(k)
    assert plan.vector_axis.name.startswith("k_")
    assert plan.stores[0].mask is not None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcmps" in ops and "tl.simd.vsts" in ops
    assert "tl.simd.vgather2" not in ops


def test_flatten_is_skipped_when_a_load_still_needs_gather():
    """P1 的读侧门禁：展平只是把下标换成 ``k`` 的分解式，**消不掉** gather。

    ``d[i, j] = g[i] - g[j]`` 展平后两个读都是分片访问（``g[k//32]`` / ``g[k%32]``）→
    合成轴不接手，原单轴路径（``vadds`` + 1 条 gather，无 ``vdup``）保持不变。
    """
    S, M = 32, 16

    @T.prim_func
    def k(G: T.Tensor((S,), "float32"), D: T.Tensor((M, S), "float32")):
        with T.Kernel(1):
            g = T.alloc_shared((S,), "float32")
            d = T.alloc_shared((M, S), "float32")
            for i in T.Parallel(M):
                for j in T.Parallel(S):
                    d[i, j] = g[i] - g[j]

    plan = _single_segment(k)
    assert [axis.name for axis in plan.axes] == ["i", "j"]
    assert plan.vector_axis.name == "j"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vgather2" in ops and "tl.simd.vsub" in ops
    assert "tl.simd.vdup" in ops  # uniform 侧取负不适用：``g[i] - g[j]`` 是 scalar - vector


def test_direct_serial_reduction_absorbed():
    """S3：``for i in P: for j in serial: s[i] += a[i,j]``。

    内层 serial 轴只有 8 个 lane（8/64），选它做跨 lane 归约会浪费 56 个 lane；
    轴打分已按 ``active/lanes`` 折算 → 改选并行轴 i：``s_vec += a[:, j]``，
    serial 循环保留为真实循环（不再需要 ``vcadd``）。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] += a[i, j]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vadd" in ops and "tl.simd.vgather2" in ops
    assert "tl.simd.vcadd" not in ops


def test_direct_serial_private_scalar_absorbed():
    """S2（§6 第二步）：迭代私有**标量**累加器，沿并行轴向量化。

    ``acc``(local.var) 每个并行迭代各有一份 → 提升成 ``local.var`` **向量寄存器**：
    ``pre`` 的 ``acc = 0`` 折叠成 ``vdup`` 初值，主体每轮只做 ``vadd``，循环结束一次
    ``ONEPT`` 回写私有标量，``post`` 的 ``c[i] = acc`` 直接复用寄存器。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                acc = T.alloc_var("float32")
                acc = 0.0
                for j in T.serial(KS):
                    acc += a[i, j]
                c[i] = acc

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"  # 每 lane 一份私有累加器，沿并行轴向量化
    assert plan.reduce is None  # 不是跨 lane 归约
    assert [s.kind for s in plan.stores] == [AccessClass.UNIFORM]
    assert [s.spec.buffer.name for s in plan.pre_stores] == ["acc"]
    assert [s.spec.buffer.name for s in plan.post_stores] == ["c"]
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vadd" in ops
    assert "tl.simd.vcadd" not in ops  # 私有标量不做跨 lane 归约
    assert "tl.simd.vld" not in ops  # 初值来自 pre 折叠，主体旧值来自寄存器
    # ONEPT 回写私有标量（段外可见）+ 写回 c[i]；post 直接吃寄存器（不再 vdup 回读）。
    assert ops.count("tl.simd.vsts") == 2
    assert 'scope="local.var"' in str(emit_plan(plan))


def test_serial_shared_scalar_rejected():
    """S4 边界：归约到**共享标量** ``cnt[0] += 1``。目的地址沿 serial 轴不变、且取值
    不随轴变化（不是合法归约）→ 拒绝吸收。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            cnt = T.alloc_shared((1,), "int32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    if a[i, j] != 0.0:
                        cnt[0] += 1
                        c[i, j] = a[i, j]

    with _rejects("body.nested_loop"):
        _plan(k)


def test_serial_stencil_rejected():
    """边界：serial 体内的跨迭代读（stencil ``c[i,j-1]``）不是累积自读。

    P1/P2 不再把它当作「不可吸收的 shell」直接拒绝：shell 先按迭代展开（语义等价），
    跨地址读随后在轴选择阶段被识别为 stencil → 仍拒绝，原因更准确。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = c[i, j - 1] + a[i, j]

    with _rejects("plan.no_vector_axis") as excinfo:
        _plan(k)
    assert "stencil" in str(excinfo.value)


# ---------------------------------------------------------------------------
# §6 第二步：T.Parallel 体内多条语句，其中夹着一个干净的 T.serial（pre/post 段）
# ---------------------------------------------------------------------------


def test_serial_pre_statement_absorbed():
    """M2 前置语句 + serial：``s[i]=0.0; for j in serial: s[i]+=a[i,j]``。

    沿并行轴 i 向量化：chunk 循环包住 ``pre(s[i]=0) → for j: main``，pre 每个 chunk
    （而不是每个 j 迭代）执行一次。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                s[i] = 0.0
                for j in T.serial(KS):
                    s[i] += a[i, j]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    assert [s.spec.buffer.name for s in plan.pre_stores] == ["s"]
    assert [s.kind for s in plan.pre_stores] == [AccessClass.CONTIGUOUS]
    assert plan.post_stores == []
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vadd" in ops and "tl.simd.vsts" in ops
    assert "tl.simd.vcadd" not in ops


def test_serial_post_statement_absorbed():
    """M3 serial + 后置语句：``for j in serial: s[i]+=a[i,j]; s[i]=s[i]*2``。

    沿并行轴 i 向量化：chunk 循环包住 ``for j: main → post``，post 在寄存器/共享内存里
    每个 chunk 只做一次（``vmuls``），不再走跨 lane 归约。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] += a[i, j]
                s[i] = s[i] * 2.0

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    assert plan.pre_stores == []
    assert [s.spec.buffer.name for s in plan.post_stores] == ["s"]
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vadd" in ops and "tl.simd.vmuls" in ops
    assert "tl.simd.vcadd" not in ops


def test_serial_sandwiched_statements_absorbed():
    """前置 + serial + 后置（夹心）：``s[i]=0.0; for j: s[i]+=a[i,j]; s[i]=s[i]*2``。

    沿并行轴 i 向量化后，三段的顺序仍是 pre → for j: 主体 → post。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                s[i] = 0.0
                for j in T.serial(KS):
                    s[i] += a[i, j]
                s[i] = s[i] * 2.0

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    assert [s.spec.buffer.name for s in plan.pre_stores] == ["s"]
    assert [s.spec.buffer.name for s in plan.post_stores] == ["s"]
    # StoreRole: the plan (not the emitter) decides that the pre store feeds the
    # accumulator initial value and that the post owns the write-back.
    assert plan.pre_stores[0].role == "pre_folded"
    assert plan.stores[0].role == "post_superseded"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vadd" in ops and "tl.simd.vmuls" in ops
    # 累加器寄存器化：同址 pre 折成 ``vdup`` 寄存器初值，主体每轮只更新寄存器
    # （不再每轮 vld + vsts），post 直接读寄存器 → 整段只剩一次回写。
    assert ops.count("tl.simd.vsts") == 1
    assert "tl.simd.vld" not in ops
    text = str(emit_plan(plan))
    assert 'scope="local.var"' in text and "acc" in text


def test_serial_independent_store_group_absorbed():
    """M1 独立 store 组 + serial：``c[i,0]=1.0; for j: c[i,j]=a[i,j]*2``。

    沿并行轴 i 向量化；``c[i,0]`` 会被 ``j=0`` 那一轮必然覆写（主体不读 c）→
    pre 写是死写，build 期直接删除。主体沿 i 是步长写，走 ``vgather2``/``vscatter``。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                c[i, 0] = 1.0
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    assert plan.pre_stores == []  # 死写已删除
    assert [s.kind for s in plan.stores] == [AccessClass.STRIDED]
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vmuls" in ops and "tl.simd.vgather2" in ops and "tl.simd.vscatter" in ops


# ---------------------------------------------------------------------------
# §6 第三步：T.Parallel 体内**多个并列**的干净 T.serial 循环 → VectorProgram
# ---------------------------------------------------------------------------


def test_multi_serial_independent_segments():
    """两个并列的干净 serial 循环（互不相关）→ 每段独立向量化，共享外层并行轴 i
    （只在最外层包裹一次），两段按源码顺序发射。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        B: T.Tensor((MS, KS), "float32"),
        C: T.Tensor((MS, KS), "float32"),
        D: T.Tensor((MS, KS), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            d = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    d[i, l] = b[i, l] + 1.0

    program = _plan(k)
    assert isinstance(program, VectorProgram)
    # 两段都是纯 elementwise → 各自展平成满 lane 合成轴（i×j / i×l，512 元素）。
    assert program.axes == []
    assert [seg.vector_axis.name for seg in program.segments] == ["k_i_j", "k_i_l"]
    assert [seg.chunks for seg in program.segments] == [8, 8]
    assert [seg.stores[0].spec.buffer.name for seg in program.segments] == ["c", "d"]
    assert all(seg.stores[0].kind == AccessClass.CONTIGUOUS for seg in program.segments)

    stmt = emit_plan(program)
    kinds = _for_kinds(stmt)
    assert tirx.ForKind.PARALLEL not in kinds
    # 每段一个满 lane chunk 循环；没有 8/64 lane 的 serial 轴循环。
    assert kinds.count(tirx.ForKind.SERIAL) == 2


def test_multi_serial_reduce_then_use():
    """归约段 + 广播使用段：两段都沿并行轴 i 向量化（serial 循环保留），
    段间依赖 ``s[i]`` 是逐 lane 的，源序发射保持语义。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] += a[i, j]
                for l in T.serial(KS):
                    c[i, l] = s[i] * 2.0

    program = _plan(k)
    assert isinstance(program, VectorProgram)
    assert len(program.segments) == 2
    assert [seg.vector_axis.name for seg in program.segments] == ["i", "i"]
    # 向量轴 i 已吃成 lane 维：program 层不能再包 `for i`，否则整段会重复 64 次。
    assert program.axes == []
    assert all(seg.reduce is None for seg in program.segments)
    stmt = emit_plan(program)
    assert "for i in range(64)" not in str(stmt)
    ops = _emitted_ops(stmt)
    assert "tl.simd.vadd" in ops and "tl.simd.vmuls" in ops
    assert "tl.simd.vcadd" not in ops


def test_multi_serial_compose_keeps_source_order():
    """element-wise 组合：后段读前段写的 ``c[i,l]``。源序发射保证「先算 c 再算 d」，
    因此发射指令里 ``c`` 的算术前于 ``d`` 的算术。"""
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

    ops = _emitted_ops(emit_plan(_plan(k)))
    assert "tl.simd.vmuls" in ops and "tl.simd.vadds" in ops
    assert ops.index("tl.simd.vmuls") < ops.index("tl.simd.vadds")


def test_multi_serial_leading_statement_is_first_segment_pre():
    """首段之前的语句组归属首段的 ``pre``（在 chunk 循环之外每个并行迭代执行一次）。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                s[i] = 0.0
                for j in T.serial(KS):
                    s[i] += a[i, j]
                for l in T.serial(KS):
                    c[i, l] = s[i] * 2.0

    program = _plan(k)
    assert isinstance(program, VectorProgram)
    assert [store.spec.buffer.name for store in program.segments[0].pre_stores] == ["s"]
    assert program.segments[1].pre_stores == []


def test_multi_serial_one_unclean_is_rejected():
    """并列 serial 中有一段跨迭代读（stencil ``c[i,l-1]``）→ 不能吸收该段。

    shell 的逐迭代展开只在「不可吸收」时发生；展开后跨地址读在轴选择阶段被识别为
    stencil，整体仍拒绝（不做静默降级），但原因从「nested serial loop」变成具体的
    stencil 诊断。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    c[i, l] = c[i, l - 1] + a[i, l]

    with _rejects("plan.no_vector_axis") as excinfo:
        _plan(k)
    assert "stencil" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 级别 0/1：统一返回契约、等价性护栏、诊断补齐（缺陷 A/B）
# ---------------------------------------------------------------------------


def test_build_plan_always_returns_vector_program():
    """级别 1：``build_plan`` **恒**返回 :class:`VectorProgram`（1..N 段），调用方不再需要
    分支处理两种类型。单段用例里 ``program.axes`` 就是「并行且非向量轴」的提层前缀
    （此处只有外层 ``i``），向量轴 ``j`` 留在该段内 strip-mine。"""
    M, K = 16, 128

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M, K), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            c = T.alloc_shared((M, K), "float32")
            for i, j in T.Parallel(M, K):
                c[i, j] = a[i, j]

    program = _plan(k)
    assert isinstance(program, VectorProgram)
    assert len(program.segments) == 1
    assert [axis.name for axis in program.axes] == ["i"]
    assert all(axis.is_parallel for axis in program.axes)
    assert program.segments[0].vector_axis.name == "j"
    # program 级诊断对单段/多段**恒非空**（与多段路径对称）。
    assert any("single-segment program" in note for note in program.diagnostics)
    # 1 段 program 的人类可读文本与「单段 plan」逐字同形（只是多一行 program 级摘要）。
    assert render_plan(program) == render_plan(program.segments[0]) + ("\n  single-segment program: 1 segment, vector axis 'j'")
    assert not render_plan(program).startswith("program:")


def test_single_segment_program_emits_identically_to_single_plan():
    """等价性护栏（步骤 2）：同一份 IR 分别走「单段路径」（``emit_plan(VectorPlan)``：
    该段自己包全部非向量轴）与「1 段 program 的统一路径」（``emit_plan(VectorProgram)``：
    ``program.axes`` 提到最外层包一次 + 段内按变量身份跳过），断言两棵发射出的 TIR
    **逐节点结构等价**（``map_free_vars=True`` 即只按首现位置对应自由变量，变量名除外）。

    覆盖三类轴形状；第三类「向量轴就是外层并行轴」是 ``hoisted`` 必须按**变量身份**跳过
    （而非笼统跳过所有并行轴）才能保持等价的关键用例——否则段内的 ``j`` 会被漏包。
    """
    M, K = 64, 128

    @T.prim_func
    def multi_axis_copy(A: T.Tensor((M, K), "float32"), C: T.Tensor((M, K), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            c = T.alloc_shared((M, K), "float32")
            for i, j in T.Parallel(M, K):
                c[i, j] = a[i, j]

    MS, KS = 64, 8

    @T.prim_func
    def serial_absorbed(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0

    MD = 64

    @T.prim_func
    def outer_parallel_vector_axis(A: T.Tensor((MD, MD), "float32"), C: T.Tensor((MD, MD), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MD, MD), "float32")
            c = T.alloc_shared((MD, MD), "float32")
            for i, j in T.Parallel(MD, MD):
                c[j, i] = a[i, j]  # 联合打分选外层轴 i（让写连续）

    cases = [
        ("multi-axis copy", multi_axis_copy, "j", ["i"]),
        # ``(64, 8)`` 的最内层只有 8 lane → P1 展平成单轴，没有可提层的外层并行轴。
        ("serial absorbed", serial_absorbed, "k_i_j", []),
        ("outer parallel vector axis", outer_parallel_vector_axis, "i", []),
    ]
    for name, func, vector_axis, outer_axes in cases:
        program = _plan(func)
        assert isinstance(program, VectorProgram), name
        assert len(program.segments) == 1, name
        assert program.segments[0].vector_axis.name == vector_axis, name
        assert [axis.name for axis in program.axes] == outer_axes, name

        direct = emit_plan(program.segments[0])  # 单段路径：该段包全部轴
        unified = emit_plan(program)  # 统一路径：program.axes 提到最外层包一次
        assert tvm.ir.structural_equal(direct, unified, map_free_vars=True), name
        # 发射出的模板序列与循环种类也逐项一致（更强的行为信号）。
        assert _emitted_ops(direct) == _emitted_ops(unified), name
        assert _for_kinds(direct) == _for_kinds(unified), name


def test_multi_segment_program_diagnostics_are_populated():
    """缺陷 A：多段路径不再产生「diagnostics 恒空」的计划——program 层有摘要，每段也有
    自己的逐轴分类与「选中向量轴」说明，与单段路径的 ``plan.diagnostics`` 同构。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        B: T.Tensor((MS, KS), "float32"),
        C: T.Tensor((MS, KS), "float32"),
        D: T.Tensor((MS, KS), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            d = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    d[i, l] = b[i, l] + 1.0

    program = _plan(k)
    assert isinstance(program, VectorProgram)
    assert len(program.segments) == 2
    assert any("multi-segment program" in note for note in program.diagnostics)
    for segment in program.segments:
        assert any("axis 'i'" in note for note in segment.diagnostics)
        assert any(f"'{segment.vector_axis.name}'" in note for note in segment.diagnostics)
        assert any("segment vector axis" in note or "flattened segment" in note for note in segment.diagnostics)


def test_multi_segment_failure_reason_is_preserved():
    """缺陷 B：某一段构造失败时**不再被吞掉**——段级原因经 ``notes`` 合并进最终
    ``Unsupported.diagnostics``，而不是只剩回落单段路径后那句泛化的「nested serial loop」。

    第二段写 ``bool`` 目标（``T.simd`` 无 bool 向量 store）→ 该段在 store 规划时抛
    Unsupported；整体仍按既有行为拒绝，但原因随诊断一并可见。
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "bool")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            flags = T.alloc_shared((MS, KS), "bool")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    flags[i, l] = a[i, l] > 0.0

    with _rejects("unsupported") as excinfo:
        _plan(k)
    notes = excinfo.value.diagnostics
    assert any(note.startswith("segment 1") for note in notes)
    assert any("not supported" in note for note in notes)
