"""Offline structural tests for the unified polyhedral vectorizer (方案 §315-484).

Hardware independent and compiler independent: each case builds TileLang IR,
runs the polyhedral analysis (:func:`build_plan`) / codegen (:func:`emit_plan`)
directly, and inspects ``program.segments`` (the returned :class:`VectorProgram`;
single-segment cases go through :func:`_single_segment` to take ``segments[0]``)
plus the emitted ``T.simd.*`` call sequence.  No PTOAS / Bisheng / device
execution is involved.

**Implementation binding: C++ only.**  Every symbol this file imports from
``tilelang.ascend.analysis.polyhedral`` is a thin alias of the compiled C++
implementation (``_cpp`` over TVM FFI): plans, emissions, renders, Δ
classification and the nest driver are all produced by C++.  The Python
reference implementation has been moved out to a separate package
(``tilelang.ascend.analysis.polyhedral_reference``) and is never imported here;
:func:`test_cpp_implementation_is_the_only_one_loaded` asserts that binding at
runtime (no reference module in ``sys.modules``, every entry point owned by
``...polyhedral._cpp``, every plan an FFI ``VectorProgram`` object).

Covered contract (one test per catalogued case):

* §0 Case I   copy / scalar broadcast, and the no-operand (Copy / Brc) shape;
* §0 Case II  a global-memory operand is rejected (needs UB staging first);
* §1 Case I   ``if`` / ``if-else`` (two-arm merged to ``vsel``, single arm to a
  predicated store);
* §1 Case II  ternary (``T.if_then_else`` / ``tirx.Select``): a *loop-invariant*
  condition is broadcast to an all-or-nothing full-lane predicate
  (``update_mask(cond ? lanes : 0)``); a lane-varying condition compares per
  lane (``vcmp``); a select that is uniform everywhere stays scalar;
* §2 Case I   operand rank mismatch (``g[i] - g[j]``: broadcast one side);
* §2 Case II  reduction (``acc[i] += a[i, j]``: dst rank < src), including the
  ``max`` variant and the "extent % lanes != 0" tail (identity-filled via
  ``update_mask`` + ``vsel``, 改进 6);
* §3 Case I   multiple stores in one body; a store that is strided on the only
  axis is expressed through the per-lane index-vector channel (``vscatter``);
* §3 Case II  scalar temporaries (`Bind` aliases) fold into the value DAG;
  ``FloorDiv`` / ``FloorMod`` as a *value* operator strength-reduces to
  ``vshrs`` / ``vand`` for a power-of-two divisor (and is rejected otherwise);
* §4 Case I   expression subscript: accepted on UB, rejected on a GM operand;
* §4 Case II  high-dimensional flattening (``h[i//16, i%16]``), a strided load
  (``a[2*i]``), a reversal (``a[N-1-i]`` → ``PERMUTE``), and a block-tiled
  fragment layout (``PIECEWISE`` → a per-lane ``vgather2``);
* §4 gather   indirect access ``a[b[i]]`` is expressed through the index-vector
  channel; a truly structureless index (e.g. ``a[i//3]``) is rejected;
* §5          ``T.atomic_add`` is rejected (no vector+scalar atomic recipe);
* §6 第一步   a *directly-nested, clean* ``T.serial`` loop is absorbed as an extra
  candidate axis (S1 copy / predicated, S3 ``s[i] += a[i,j]`` reduction); a serial
  body with loop-carried state (shared scalar S4, stencil cross-iteration read) is
  *not* absorbed and the whole nest is rejected with "nested serial loop" (no
  silent downgrade).
* §6 第二步   a ``T.Parallel`` body that is a ``SeqStmt`` holding *exactly one*
  clean serial loop is also absorbed: the sibling statements before/after it
  become the ``pre`` / ``post`` segments, emitted *outside* the chunk loop once
  per parallel iteration (order ``pre → main → post``).  Covers M2 (init + serial),
  M3 (serial + post-processing), the sandwich (pre + serial + post) and M1 (an
  independent store group + serial), plus the private scalar accumulator S2.
* 改进 7      ``VectorPlan.diagnostics`` records every axis on success;
  ``Unsupported.diagnostics`` carries the per-axis notes on rejection.
* 统一契约    ``build_plan`` always returns a :class:`VectorProgram` (1..N segments),
  and ``program.axes`` holds only the hoisted ``T.Parallel`` axis prefix.  A
  1-segment program emits a TIR tree structurally identical (free variable names
  aside) to emitting its bare ``VectorPlan`` -- the equivalence guard for the
  §3.9 axis-hoisting order reasoning.
* 诊断不吞    a multi-segment program/segment carries diagnostics, and a segment
  build failure is merged into the final ``Unsupported.diagnostics`` instead of
  being swallowed by the single-segment fallback.
* transform   ``vectorize_parallel_nests`` lowers a whole multi-axis nest
  (outer axes → serial, vector axis → strip-mined chunks).

The mathematical core (``finite_difference`` / ``classify_delta``) is unit
tested directly, since every case above is decided by the sign/shape of
``Δ(offset)`` along the candidate vector axis.
"""

from __future__ import annotations

import contextlib
import sys

import pytest

import tilelang.ascend.language as T
import tvm
from tvm import arith, tirx

from tilelang.ascend.analysis.polyhedral import (
    AccessClass,
    Unsupported,
    VectorProgram,
    build_plan,
    classify_delta,
    emit_plan,
    finite_difference,
    render_plan,
    vectorize_parallel_nests,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _first_parallel(func) -> tirx.For:
    """Return the outermost ``T.Parallel`` loop of a lowered prim func.

    ``post_order_visit`` yields the innermost nest first, so the outermost one is
    the *last* collected: ``build_plan`` needs the head of the nest to recover
    every axis (``_split_nest`` walks all consecutive ``T.Parallel`` loops).
    """
    found: list = []

    def visit(node):
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL:
            found.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    assert found, "the kernel has no T.Parallel nest"
    return found[-1]


def _for_kinds(stmt) -> list:
    kinds: list = []

    def visit(node):
        if isinstance(node, tirx.For):
            kinds.append(node.kind)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return kinds


def _plan(func):
    return build_plan(_first_parallel(func))


def _single_segment(func):
    """``build_plan`` 的契约是**恒返回** :class:`VectorProgram`（级别 1）。单段用例想拿回
    那唯一一段的 :class:`VectorPlan` 时经这个 helper，顺带断言契约本身：恰好一段、
    ``program.axes`` 只含提层的并行轴前缀（详见 :func:`_outer_axes`）。"""
    program = build_plan(_first_parallel(func))
    assert isinstance(program, VectorProgram), "build_plan must always return a VectorProgram"
    assert len(program.segments) == 1, f"expected a single segment, got {len(program.segments)}"
    return program.segments[0]


def _emitted_ops(stmt) -> list:
    """Extract the ordered ``tl.simd.*`` op names emitted into *stmt*."""
    ops: list = []

    def visit(node):
        if isinstance(node, tirx.Call):
            name = str(getattr(node.op, "name", ""))
            if name.startswith("tl.simd."):
                ops.append(name)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return ops


def _call_args(stmt, op_name: str) -> list:
    """Collect the arguments of every ``op_name`` intrinsic call in *stmt*."""
    found: list = []

    def visit(node):
        if isinstance(node, tirx.Call) and str(getattr(node.op, "name", "")) == op_name:
            found.extend(node.args)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return found


def _recipes(node) -> list:
    """Flatten a :class:`ValueNode` DAG into its recipe names."""
    out = [node.recipe]
    for arg in node.args:
        out.extend(_recipes(arg))
    return out


def _eval_lane(node, lane: int, plan, env: dict | None = None, base: int = 0):
    """把 :class:`ValueNode` DAG 在单条 lane 上求值（数值 golden 用）。

    内存模型取**恒等**：一次装载返回它的元素偏移。这足以钉住"下标/取值算术"这一类
    缺陷（符号翻转、偏移错位）——它们在只有结构断言的测试里看不出来。

    ``env`` 给非向量轴变量赋值（如 ``{i: 3}``），``base`` 是该 chunk 的 lane 基址。
    """

    def offset(access, lane_value):
        mapping = dict(env or {})
        mapping[plan.vector_axis.var] = tirx.IntImm("int32", int(lane_value))
        simplified = arith.Analyzer().simplify(tirx.stmt_functor.substitute(access.offset, mapping))
        assert isinstance(simplified, (tirx.IntImm, tirx.FloatImm)), f"offset did not fold: {simplified}"
        return int(simplified.value)

    def scalar_value(expr):
        if isinstance(expr, (int, float)):
            return int(expr)
        simplified = arith.Analyzer().simplify(expr)
        if isinstance(simplified, (tirx.IntImm, tirx.FloatImm)):
            return int(simplified.value)
        simplified = arith.Analyzer().simplify(tirx.stmt_functor.substitute(simplified, dict(env or {})))
        assert isinstance(simplified, (tirx.IntImm, tirx.FloatImm)), f"scalar did not fold: {simplified}"
        return int(simplified.value)

    def ev(n):
        recipe = n.recipe
        if recipe == "lane_index":
            return base + lane
        if recipe in ("imm", "uniform_expr"):
            return scalar_value(n.scalar)
        if recipe == "uniform_load":
            return offset(n.access, 0)  # 沿向量轴不变：代入 0 即可
        if recipe in ("vload", "vload_strided"):
            return offset(n.access, base + lane)
        if recipe == "gather":
            return ev(n.args[0]) if n.args else offset(n.access, base + lane)
        if recipe == "vbrc":
            return ev(n.args[0])
        if recipe == "vadds":
            return ev(n.args[0]) + ev(n.args[1])
        if recipe == "vadd":
            return ev(n.args[0]) + ev(n.args[1])
        if recipe == "vsub":
            return ev(n.args[0]) - ev(n.args[1])
        if recipe in ("vmul", "vmuls"):
            return ev(n.args[0]) * ev(n.args[1])
        raise AssertionError(f"no lane evaluator for recipe '{recipe}'")

    return ev(node)


# ---------------------------------------------------------------------------
# implementation binding: C++ only, no Python reference implementation
# ---------------------------------------------------------------------------

#: Python oracle package that this suite must never load.
_REFERENCE_PACKAGE = "tilelang.ascend.analysis.polyhedral_reference"
#: Module that owns every entry point used below (the C++ FFI binding).
_CPP_MODULE = "tilelang.ascend.analysis.polyhedral._cpp"


def test_cpp_implementation_is_the_only_one_loaded():
    """Every assertion in this file is produced by the compiled implementation.

    Proves the dependency contract at runtime:

    * the Python reference package is not imported by the suite;
    * each API entry point is owned by ``...polyhedral._cpp`` (FFI), not by a
      Python module;
    * the plan objects flowing through the assertions are C++ FFI nodes, not the
      Python reference dataclasses.
    """
    loaded = [
        module
        for module in sys.modules
        if module == _REFERENCE_PACKAGE or module.startswith(_REFERENCE_PACKAGE + ".")
    ]
    assert loaded == [], f"the Python reference implementation was imported: {loaded}"

    for name, entry in (
        ("build_plan", build_plan),
        ("emit_plan", emit_plan),
        ("render_plan", render_plan),
        ("vectorize_parallel_nests", vectorize_parallel_nests),
        ("finite_difference", finite_difference),
        ("classify_delta", classify_delta),
    ):
        assert entry.__module__ == _CPP_MODULE, f"{name} is not the C++ entry point: {entry.__module__}"

    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i]

    program = build_plan(_first_parallel(k))
    assert type(program).__module__ == _CPP_MODULE, f"plan came from {type(program).__module__}"
    assert type(program.segments[0]).__module__ == _CPP_MODULE


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


def test_global_operand_rejected():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = A[i]

    with pytest.raises(Unsupported, match="global-memory operand"):
        _plan(k)


# ---------------------------------------------------------------------------
# §1 branches / ternaries
# ---------------------------------------------------------------------------


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
    whole chunk takes the same branch.  It is broadcast to an all-or-nothing
    full-lane predicate (``update_mask(cond ? lanes : 0)``) fed to ``vsel`` — no
    branch hoisting, no per-lane compare."""
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
    assert value.args[0].is_uniform
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.update_mask" in ops and "tl.simd.vsel" in ops
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
    assert "tl.simd.update_mask" in ops and "tl.simd.vsel" in ops
    assert "tl.simd.vcmp" not in ops


def test_loop_invariant_condition_with_uniform_arm_broadcasts_and_dedups():
    """A uniform condition whose false arm is a scalar keeps working: the arm is
    materialised by ``vdup``, the predicate is still the broadcast one."""
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
    assert "tl.simd.update_mask" in ops and "tl.simd.vdup" in ops and "tl.simd.vsel" in ops


def test_select_that_is_uniform_everywhere_stays_scalar():
    """When condition *and* both arms are loop-invariant, the whole select is
    lane-uniform: it collapses to one scalar value + ``vdup`` (no ``vsel``)."""
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
    assert plan.stores[0].value.is_uniform
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vsel" not in ops
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
    """P0-2：一段共享一个 lane 数（chunk 步长 = lanes）。f16 与 int32 混写时后者的
    向量寄存器只有一半 lane，会静默漏写/越界 → build 期必须拒绝。"""
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

    with pytest.raises(Unsupported, match="mixed element widths"):
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
    B0, LEN, PAD = 2, 128, 3

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


def test_expression_subscript_rejected_on_global_operand():
    """§4 Case I: the *same* expression subscript is rejected when the operand is
    a GM buffer — an expression subscript inside ``T.Parallel`` needs UB staging
    first, regardless of how simple the stride is."""
    S, DK = 16, 64

    @T.prim_func
    def k(Gb: T.Tensor((2, 1024, 8), "float32"), OUT: T.Tensor((S, DK), "float32")):
        with T.Kernel(1):
            gs = T.alloc_shared((S, DK), "float32")
            for i_s, i_k in T.Parallel(S, DK):
                gs[i_s, i_k] = Gb[0, 0 * S + i_s, 0]

    with pytest.raises(Unsupported, match="global-memory operand"):
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

    with pytest.raises(Unsupported, match="not a power of two"):
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

    with pytest.raises(Unsupported, match="nested serial loop"):
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

    with pytest.raises(Unsupported, match="nested serial loop"):
        _plan(k)


def test_serial_stencil_rejected():
    """边界：serial 体内的跨迭代读（stencil ``c[i,j-1]``）不是累积自读 → 拒绝吸收。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = c[i, j - 1] + a[i, j]

    with pytest.raises(Unsupported, match="nested serial loop"):
        _plan(k)


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
    """并列 serial 中有一段跨迭代读（stencil ``c[i,l-1]``）→ 不能吸收该段，整体回落到
    既有单段路径并按「nested serial loop」拒绝（不做静默降级）。"""
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

    with pytest.raises(Unsupported, match="nested serial loop"):
        _plan(k)


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

    第二段读 GM 张量 ``A``（未先 staging 到 UB）→ 该段在取值 DAG 构造时抛 Unsupported；
    整体仍按既有行为拒绝，但原因随诊断一并可见。
    """
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
                    c[i, l] = A[i, l]

    with pytest.raises(Unsupported, match="nested serial loop") as excinfo:
        _plan(k)
    notes = excinfo.value.diagnostics
    assert any(note.startswith("segment 1") for note in notes)
    assert any("global-memory operand" in note for note in notes)


# ---------------------------------------------------------------------------
# transform-level entry point
# ---------------------------------------------------------------------------


def test_vectorize_parallel_nests_replaces_parallel():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i]

    transformed = vectorize_parallel_nests(k.body)

    kinds = _for_kinds(transformed)
    assert tirx.ForKind.PARALLEL not in kinds, "the T.Parallel nest must be replaced"
    assert kinds.count(tirx.ForKind.SERIAL) == 1  # the single axis -> chunk loop only
    assert "tl.simd.vld" in _emitted_ops(transformed)


def test_vectorize_parallel_nests_lowers_multidim_nest_as_a_whole():
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

    transformed = vectorize_parallel_nests(k.body)

    kinds = _for_kinds(transformed)
    assert tirx.ForKind.PARALLEL not in kinds, "the whole nest must be lowered"
    assert kinds.count(tirx.ForKind.SERIAL) == 2, "outer axis + chunk loop, no residue"
    ops = _emitted_ops(transformed)
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


def test_vectorize_parallel_nests_lowers_multi_serial_program():
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

    transformed = vectorize_parallel_nests(k.body)

    kinds = _for_kinds(transformed)
    assert tirx.ForKind.PARALLEL not in kinds
    # 两段都展平成满 lane 合成轴：每段一个 chunk 循环。
    assert kinds.count(tirx.ForKind.SERIAL) == 2
    ops = _emitted_ops(transformed)
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


def _for_extents(stmt) -> list:
    """收集所有 ``For`` 的 extent（用于检查符号 extent 未被静默截断）。"""
    extents: list = []

    def visit(node):
        if isinstance(node, tirx.For):
            extents.append(node.extent)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return extents


def _negative_immediate_muls(stmt) -> bool:
    """是否存在 ``vmuls(<uint32 ramp>, <负常量>)`` —— 无符号斜坡乘负步长无意义（缺陷 D8）。"""
    bad: list = []

    def visit(node):
        if (
            isinstance(node, tirx.Call)
            and str(getattr(node.op, "name", "")) == "tl.simd.vmuls"
            and len(node.args) >= 2
            and isinstance(node.args[1], tirx.IntImm)
            and int(node.args[1].value) < 0
        ):
            bad.append(True)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return bool(bad)


def _loop_maxima(stmt) -> dict:
    """收集所有循环变量的最大取值（常量区间；用于判断访问跨度是否越界）。"""
    maxima: dict = {}

    def visit(node):
        if isinstance(node, tirx.For) and isinstance(node.min, tirx.IntImm) and isinstance(node.extent, tirx.IntImm):
            lo, extent = int(node.min.value), int(node.extent.value)
            maxima[node.loop_var] = lo + extent - 1 if extent > 0 else lo

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return maxima


def _over_reading_contiguous_load(stmt) -> bool:
    """连续 ``vld`` 的 access_ptr footprint 是否越过源 buffer 的分配范围。

    判据只有一条：``start + extent > buffer 的元素总数``。多维 buffer 里「整 lane 读跨越
    最内维」本身不是错误——只要是同一块紧凑分配（P1 展平后正是这种形态：``(64, 8)``
    上一次读 8 行），跨行读仍落在分配范围内。循环变量按**最大值**代入（只对沿该变量
    ``Δ`` 为非负常数的下标做，保证取到的是上界）。
    """
    analyzer = arith.Analyzer()
    maxima = _loop_maxima(stmt)
    bad: list = []

    def max_start(indices, strides):
        total = 0
        for index, stride in zip(indices, strides):
            expr = index
            for var, hi in maxima.items():
                delta = finite_difference(expr, var, analyzer)
                if isinstance(delta, tirx.IntImm) and int(delta.value) >= 0:
                    expr = tirx.stmt_functor.substitute(expr, {var: tirx.IntImm("int32", hi)})
                    expr = analyzer.simplify(expr)
            if not isinstance(expr, tirx.IntImm):
                return None
            total += int(expr.value) * stride
        return total

    def visit(node):
        if not (isinstance(node, tirx.Call) and str(getattr(node.op, "name", "")) == "tl.simd.vld"):
            return
        addr = node.args[0]
        if not (isinstance(addr, tirx.Call) and str(getattr(addr.op, "name", "")) == "tl.access_ptr"):
            return
        inner, extent = addr.args[0], addr.args[1]
        if not (isinstance(inner, tirx.BufferLoad) and isinstance(extent, tirx.IntImm)):
            return
        shape = inner.buffer.shape
        if not shape or not all(isinstance(dim, tirx.IntImm) for dim in shape):
            return
        strides, numel = [1] * len(shape), 1
        for k in range(len(shape) - 1, -1, -1):
            strides[k] = numel
            numel *= int(shape[k].value)
        start = max_start(inner.indices, strides)
        if start is not None and start + int(extent.value) > numel:
            bad.append(True)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return bool(bad)


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


def test_defect_d5_scaled_old_value_is_rejected():
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


def _alloc_buffers(stmt) -> list:
    """收集发射 TIR 里的 ``AllocBuffer`` 语句（归约的可变寄存器累加器）。"""
    found: list = []

    def visit(node):
        if isinstance(node, tirx.AllocBuffer):
            found.append(node.buffer)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return found


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


def _chunk_iterations(stmt) -> list[tuple]:
    """收集每个 chunk 循环的 ``(loop_var 名, 起始, 迭代次数)``（常量区间）。"""
    found: list = []

    def visit(node):
        if isinstance(node, tirx.For) and "chunk" in node.loop_var.name:
            lo = int(node.min.value) if isinstance(node.min, tirx.IntImm) else None
            extent = int(node.extent.value) if isinstance(node.extent, tirx.IntImm) else None
            found.append((node.loop_var.name, lo, extent))

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return found


def _one_dimensional_copy(M):
    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                c[i] = a[i]

    return k


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


def test_masked_branch_merging_is_rejected():
    """特征化用例（缺陷 D12）：``if/else`` 同址合并时某一臂自带谓词 → 明确拒绝。

    该路径在两份文档的 §4/§6 中**均未声明**（``normalize_body`` L615），此处把它固定成
    可观测行为；若后续支持掩码合并，本用例需同步更新。
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

    with pytest.raises(Unsupported, match="masked branch merging"):
        build_plan(_first_parallel(k))


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
    """R2：``if a[0] > 0.0: c[i] = a[i]`` 的条件沿向量轴均匀，是 **bool 标量**。

    旧实现把它交给 ``_Emitter.value`` 当数据向量 ``vdup``，而 bool 没有谓词宽度 →
    发射期 ``ValueError``。正确路径是 ``predicate()``：``update_mask(cond ? lanes : 0)``。
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
    assert "tl.simd.update_mask" in ops
    assert "tl.simd.vdup" not in ops, "a bool condition must not be broadcast as data"
    # 条件掩码与尾块掩码的合取仍然成立（尾块 100 < 128）。
    assert "tl.simd.pand" in ops


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

    with pytest.raises(Unsupported, match="conditional reduction"):
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


def test_multiple_writes_to_one_buffer_at_different_addresses_are_rejected():
    """R6：同一 serial 体内 ``c[i,2*j]`` 与 ``c[i,j]`` 会在不同迭代写到同一地址。

    标量顺序下的「后写覆盖」在向量化后不成立（lane 间写序未定义）；``_serial_is_clean``
    现在按「同一 buffer 的多条写必须目的地址相同」保守拒绝，原因进入诊断。
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

    with pytest.raises(Unsupported, match="nested serial loop") as excinfo:
        _single_segment(k)
    assert any("may alias" in note for note in excinfo.value.diagnostics)


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
