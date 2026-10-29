"""Transform-level tests for the whole Ascend polyhedral P1 -> P2 -> P3 path.

This file merges the two suites that used to live separately:

* **P1 -> P2 hand-off** (``AscendBuildP2PlanInputs`` / ``AscendBuildP2PlanInput``):
  every P1 LIFTED nest yields ``status == "LIFTED"`` plus a ``stmt`` that is the
  P1 reference expansion with the whole For chain switched back to
  ``ForKind.PARALLEL``; that ``stmt`` is a drop-in ``build_plan`` input (the plan
  and the emitted ``tl.simd.*`` statement are identical to planning the original
  ``T.Parallel`` nest); every P1 rejection keeps its status (``FALLBACK`` /
  ``OPAQUE`` / ``INTERNAL_ERROR``) and carries no ``stmt``; nests P1 can lift but
  P2 cannot plan are *Branch B* (``LIFTED`` + ``Unsupported``, not a bug).
* **Pass level** (``AscendPolyhedralVectorize`` and its three stages
  ``AscendLiftParallelToArray`` -> ``AscendPlanArray`` -> ``AscendLowerArrayPlan``):
  the combined pass replaces a vectorizable nest with the same body the offline
  driver emits, leaves a rejected nest untouched, is idempotent, hands off
  through the ``tl.vsimd.array_region`` / ``tl.vsimd.array_plan`` carriers, and
  is inserted into the pipeline only for ``tl.ascend.polyhedral.impl == "cpp"``.

Known boundary: an outer ``Bind`` (``tail = T.min(n - bx * bn, bn)``) is inlined
by ``RestoreBindings`` instead of being re-declared, so the emitted text spells
``T.min(...)`` where the original plan keeps the bound variable. The plan
structure is still identical; the difference is recorded in
:func:`test_tail_bind_restored`.

No device execution and no PTOAS / Bisheng compilation is involved.
"""

from __future__ import annotations

import pytest

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.ascend import transform as at
from tilelang.ascend.analysis.polyhedral import (
    AccessClass,
    Unsupported,
    VectorProgram,
    build_plan,
    emit_plan,
    render_plan,
    vectorize_parallel_nests,
)
from tvm import tirx
from tvm.ir import structural_equal

N = 64
M = 8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _report(func) -> str:
    return str(at.AscendAnalyzeVSimdRegionReport(func))


def _outer_parallel_nests(func) -> list:
    """Outermost ``T.Parallel`` nests in source order (one per P1 nest)."""
    nests: list = []

    def walk(stmt):
        if isinstance(stmt, tirx.For):
            if stmt.kind == tirx.ForKind.PARALLEL:
                nests.append(stmt)
                return
            walk(stmt.body)
            return
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                walk(child)
            return
        if isinstance(stmt, tirx.SBlockRealize):
            walk(stmt.block)
            return
        body = getattr(stmt, "body", None)
        if body is not None:
            walk(body)

    walk(func.body)
    return nests


def _parallel_nests(stmt) -> list:
    """Every ``T.Parallel`` For in the statement, in post-order."""
    nests: list = []

    def visit(node):
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL:
            nests.append(node)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return nests


def _only_parallel_nest(stmt):
    nests = _parallel_nests(stmt)
    assert len(nests) == 1, f"expected exactly one T.Parallel nest, got {len(nests)}"
    return nests[0]


def _inputs(func):
    return at.AscendBuildP2PlanInputs(func)


def _lifted(func, index: int = 0):
    items = _inputs(func)
    assert items, "no P1 nest was reported"
    item = items[index]
    assert item["status"] == "LIFTED", item.get("reason")
    assert "stmt" in item, "LIFTED input must carry a statement"
    return item


def _for_kinds(stmt) -> list:
    kinds: list = []

    def visit(node):
        if isinstance(node, tirx.For):
            kinds.append(node.kind)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return kinds


def _stored_buffers(stmt) -> list:
    names: list = []

    def visit(node):
        if isinstance(node, tirx.BufferStore):
            names.append(node.buffer.name)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return names


def _has_decl_buffer(stmt) -> bool:
    found: list = []

    def visit(node):
        if isinstance(node, tirx.DeclBuffer):
            found.append(node)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return bool(found)


def _plan_and_emit_parity(func, index: int = 0) -> None:
    """The P1 input must plan/emit exactly like the original parallel nest."""
    item = _lifted(func, index)
    original = _outer_parallel_nests(func)[index]
    original_plan = build_plan(original)
    p1_plan = build_plan(item["stmt"])
    assert render_plan(original_plan) == render_plan(p1_plan)
    assert structural_equal(emit_plan(original_plan), emit_plan(p1_plan), map_free_vars=True)
    print(f"\n{func}\n  \np1_r:\n{item}\nplan:\norigin:\n{original_plan}\np1_plan:\n{p1_plan}\n, emit:\nemit_p1:\n{emit_plan(p1_plan)}")


def _assert_branch_b(func, match: str, index: int = 0):
    """P1 LIFTED but P2 rejects: capture ``Unsupported`` explicitly."""
    item = _lifted(func, index)
    with pytest.raises(Unsupported, match=match):
        build_plan(item["stmt"])


def _simd_calls(stmt) -> list[str]:
    """Op names of every ``tl.simd.*`` call in the statement."""
    names: list[str] = []

    def visit(node):
        if isinstance(node, tirx.Call) and isinstance(node.op, tirx.op.Op) and node.op.name.startswith("tl.simd."):
            names.append(node.op.name)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return names


def _run_pass(func):
    """Run the combined C++ pass on a PrimFunc and return the rewritten PrimFunc."""
    return at.AscendPolyhedralVectorize()(tvm.IRModule.from_expr(func))["main"]


def _run(func, pass_factory):
    """Run one staged pass (P1/P2/P3) on a PrimFunc."""
    return pass_factory()(tvm.IRModule.from_expr(func))["main"]


def _carriers(stmt, attr_key: str) -> list:
    """Every AttrStmt with the given carrier key, in post-order."""
    found: list = []

    def visit(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == attr_key:
            found.append(node)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return found


def _lower(func, impl: str | None = None) -> str:
    """Lower to Ascend device source under the requested ``impl`` (None = default)."""
    config = {} if impl is None else {"tl.ascend.polyhedral.impl": impl}
    with tilelang.tvm.target.Target("ascend"), tilelang.transform.PassContext(config=config):
        return tilelang.lower(func, target="ascend").kernel_source


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def _kernel_1d():
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float32")
            c_ub = T.alloc_shared((N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    return main


def _kernel_2d():
    @T.prim_func
    def main(A: T.Tensor((M, N), "float32"), C: T.Tensor((M, N), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), "float32")
            c_ub = T.alloc_shared((M, N), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[i, j] = a_ub[i, j] + 1.0
            T.copy(c_ub, C)

    return main


def _kernel_global_operand():
    """GM operand inside T.Parallel: P2 rejects, the pass must keep the nest."""

    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            c_ub = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c_ub[i] = A[i]

    return main


def _kernel_gm_operand_in_vf():
    """GM load inside T.SimdVF: P1 lifts it, P2 rejects it (Branch B)."""

    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            c_ub = T.alloc_shared((N,), "float32")
            with T.SimdVF():
                for i in T.Parallel(N):
                    c_ub[i] = A[i]
            T.copy(c_ub, C)

    return main


# ===========================================================================
# Part 1: P1 -> P2 hand-off (``AscendBuildP2PlanInputs``)
# ===========================================================================


# ---------------------------------------------------------------------------
# 1-D golden (p.log / test_simple_region_lifted)
# ---------------------------------------------------------------------------


def test_simple_input_lifted_1d():
    main = _kernel_1d()
    item = _lifted(main)

    assert item["vf_index"] == 0
    assert item["nest_index"] == 0
    stmt = item["stmt"]
    assert isinstance(stmt, tirx.For)
    assert stmt.kind == tirx.ForKind.PARALLEL
    assert stmt.loop_var.name == "i"
    assert int(stmt.min) == 0
    assert int(stmt.extent) == N
    assert _for_kinds(stmt) == [tirx.ForKind.PARALLEL]
    assert _stored_buffers(stmt) == ["c_ub"]
    assert len(_outer_parallel_nests(main)) == 1


def test_simple_plan_fields_1d():
    program = build_plan(_lifted(_kernel_1d())["stmt"])
    assert isinstance(program, VectorProgram)
    assert program.axes == []
    assert len(program.segments) == 1
    segment = program.segments[0]
    assert segment.vector_axis.name == "i"
    assert segment.lanes == 64
    assert segment.chunks == 1
    assert segment.tail == 0
    assert segment.reduce is None
    assert len(segment.stores) == 1
    assert segment.stores[0].kind == AccessClass.CONTIGUOUS


def test_simple_plan_emit_parity_1d():
    _plan_and_emit_parity(_kernel_1d())


def test_report_golden_unchanged():
    report = _report(_kernel_1d())
    assert "VF#0 nest 0: status=LIFTED" in report
    assert "domain: i in [0, 64)" in report
    assert "W c_ub[i] = a_ub[i] + T.float32(1.0) shape=[64] axes=[0] dep={0}" in report
    assert "R a_ub[i] shape=[64] axes=[0] dep={0}" in report
    assert "for i in range(64)" in report
    assert "verify: OK" in report
    assert "tl.vsimd.region" in report


# ---------------------------------------------------------------------------
# 2-D / N-D
# ---------------------------------------------------------------------------


def test_2d_input_is_parallel_chain():
    main = _kernel_2d()
    report = _report(main)
    assert "domain: i in [0, 8) j in [0, 64)" in report
    assert "shape=[8,64] axes=[0,1] dep={0,1}" in report
    assert "T.grid(8, 64)" in report
    assert "verify: OK" in report

    stmt = _lifted(main)["stmt"]
    assert _for_kinds(stmt) == [tirx.ForKind.PARALLEL, tirx.ForKind.PARALLEL]
    assert stmt.loop_var.name == "i"
    assert stmt.body.loop_var.name == "j"
    assert int(stmt.extent) == M
    assert int(stmt.body.extent) == N


def test_2d_plan_emit_parity():
    _plan_and_emit_parity(_kernel_2d())


def test_2d_broadcast_parity():
    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        Bias: T.Tensor((N,), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), "float32")
            bias_ub = T.alloc_shared((N,), "float32")
            c_ub = T.alloc_shared((M, N), "float32")
            T.copy(A, a_ub)
            T.copy(Bias, bias_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[i, j] = a_ub[i, j] + bias_ub[j]
            T.copy(c_ub, C)

    assert "R bias_ub[j] shape=[8,64] axes=[0,1] dep={1}" in _report(main)
    _plan_and_emit_parity(main)


def test_2d_transpose_parity():
    @T.prim_func
    def main(A: T.Tensor((N, M), "float32"), C: T.Tensor((M, N), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N, M), "float32")
            c_ub = T.alloc_shared((M, N), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[i, j] = a_ub[j, i]
            T.copy(c_ub, C)

    assert "R a_ub[j, i] shape=[8,64] axes=[0,1] dep={0,1}" in _report(main)
    _plan_and_emit_parity(main)


def test_2d_flattened_index_parity():
    @T.prim_func
    def main(A: T.Tensor((M * N,), "float32"), C: T.Tensor((M * N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M * N,), "float32")
            c_ub = T.alloc_shared((M * N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[i * N + j] = a_ub[i * N + j] + 1.0
            T.copy(c_ub, C)

    _plan_and_emit_parity(main)


def test_3d_parity():
    b0, b1, b2 = 2, 8, 16

    @T.prim_func
    def main(
        A: T.Tensor((b0, b1, b2), "float32"),
        C: T.Tensor((b0, b1, b2), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((b0, b1, b2), "float32")
            c_ub = T.alloc_shared((b0, b1, b2), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j, k in T.Parallel(b0, b1, b2):
                    c_ub[i, j, k] = a_ub[i, j, k] + 1.0
            T.copy(c_ub, C)

    stmt = _lifted(main)["stmt"]
    assert _for_kinds(stmt) == [tirx.ForKind.PARALLEL] * 3
    _plan_and_emit_parity(main)


def test_4d_parity():
    b0, b1, b2, b3 = 2, 2, 4, 8

    @T.prim_func
    def main(
        A: T.Tensor((b0, b1, b2, b3), "float32"),
        C: T.Tensor((b0, b1, b2, b3), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((b0, b1, b2, b3), "float32")
            c_ub = T.alloc_shared((b0, b1, b2, b3), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j, k, l in T.Parallel(b0, b1, b2, b3):
                    c_ub[i, j, k, l] = a_ub[i, j, k, l] + 1.0
            T.copy(c_ub, C)

    stmt = _lifted(main)["stmt"]
    assert _for_kinds(stmt) == [tirx.ForKind.PARALLEL] * 4
    _plan_and_emit_parity(main)


# ---------------------------------------------------------------------------
# boundary matrix (§6.3)
# ---------------------------------------------------------------------------


def _kernel_1d_extent(extent: int):
    @T.prim_func
    def main(A: T.Tensor((extent,), "float32"), C: T.Tensor((extent,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((extent,), "float32")
            c_ub = T.alloc_shared((extent,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(extent):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    return main


def test_tail_extent_100():
    main = _kernel_1d_extent(100)
    segment = build_plan(_lifted(main)["stmt"]).segments[0]
    assert segment.chunks == 2
    assert segment.tail == 36
    _plan_and_emit_parity(main)


def test_extent_1():
    main = _kernel_1d_extent(1)
    segment = build_plan(_lifted(main)["stmt"]).segments[0]
    assert segment.chunks == 1
    assert segment.tail == 1
    _plan_and_emit_parity(main)


def test_symbolic_extent():
    e = T.dynamic("E")

    @T.prim_func
    def main(A: T.Tensor((e,), "float32"), C: T.Tensor((e,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((e,), "float32")
            c_ub = T.alloc_shared((e,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(e):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    assert "shape_syms:" in _report(main)
    segment = build_plan(_lifted(main)["stmt"]).segments[0]
    # Symbolic extent: the runtime chunk loop carries a symbolic tail.
    assert segment.chunks is None
    assert segment.tail is None
    _plan_and_emit_parity(main)


def test_float16_lanes_128():
    @T.prim_func
    def main(A: T.Tensor((N,), "float16"), C: T.Tensor((N,), "float16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float16")
            c_ub = T.alloc_shared((N,), "float16")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    segment = build_plan(_lifted(main)["stmt"]).segments[0]
    assert segment.lanes == 128
    _plan_and_emit_parity(main)


def test_int32_lanes_64():
    @T.prim_func
    def main(A: T.Tensor((N,), "int32"), C: T.Tensor((N,), "int32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "int32")
            c_ub = T.alloc_shared((N,), "int32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    c_ub[i] = a_ub[i] + 1
            T.copy(c_ub, C)

    segment = build_plan(_lifted(main)["stmt"]).segments[0]
    assert segment.lanes == 64
    _plan_and_emit_parity(main)


def test_multi_store():
    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
        D: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float32")
            c_ub = T.alloc_shared((N,), "float32")
            d_ub = T.alloc_shared((N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    c_ub[i] = a_ub[i] + 1.0
                    d_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    assert _stored_buffers(_lifted(main)["stmt"]) == ["c_ub", "d_ub"]
    segment = build_plan(_lifted(main)["stmt"]).segments[0]
    assert len(segment.stores) == 2
    _plan_and_emit_parity(main)


def test_value_select():
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float32")
            c_ub = T.alloc_shared((N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    c_ub[i] = T.if_then_else(a_ub[i] > 0.0, a_ub[i], 0.0)
            T.copy(c_ub, C)

    segment = build_plan(_lifted(main)["stmt"]).segments[0]
    recipes = [store.value.recipe for store in segment.stores]
    assert "vsel" in recipes
    _plan_and_emit_parity(main)


def test_view_region_parity():
    """A local view's DeclBuffer is dropped; the store still references it."""

    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            base = T.alloc_shared((N,), "float32")
            T.copy(A, base)
            with T.SimdVF():
                for i in T.Parallel(N // 2):
                    v = T.view(base, shape=(N // 2, 2), dtype="float32")
                    v[i, 0] = v[i, 0] + 1.0
            T.copy(base, C)

    report = _report(main)
    assert "local_views:" in report
    assert "v -> base" in report
    assert "decl_buffer" in report

    stmt = _lifted(main)["stmt"]
    assert not _has_decl_buffer(stmt)
    assert _stored_buffers(stmt) == ["v"]
    _plan_and_emit_parity(main)


def test_tail_bind_restored():
    """Outer Bind is inlined (known boundary: emit text, not plan shape, differs)."""
    n, bn = 1000, 256

    @T.prim_func
    def main(A: T.Tensor((n + bn,), "float32"), C: T.Tensor((n + bn,), "float32")):
        with T.Kernel(T.ceildiv(n, bn)) as bx:
            a_ub = T.alloc_shared((n + bn,), "float32")
            c_ub = T.alloc_shared((n + bn,), "float32")
            T.copy(A, a_ub)
            tail = T.min(n - bx * bn, bn)
            with T.SimdVF():
                for i in T.Parallel(tail):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    item = _lifted(main)
    stmt = item["stmt"]
    # The bind var must not survive as a free variable.
    free = {var.name for var in tirx.analysis.undefined_vars(stmt, [])}
    assert "tail" not in free
    assert "bx" in free
    assert "T.min" in str(stmt)

    # Plan structure still matches the original nest (symbolic extent).
    original_plan = build_plan(_outer_parallel_nests(main)[0])
    p1_plan = build_plan(stmt)
    assert render_plan(original_plan) == render_plan(p1_plan)
    # Known difference: the emitted tail expression is the inlined T.min(...)
    # instead of the bound var, so only the plan shape is compared.
    assert not structural_equal(emit_plan(original_plan), emit_plan(p1_plan), map_free_vars=True)


def test_multi_nest_single_vf():
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float32")
            c_ub = T.alloc_shared((N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    c_ub[i] = a_ub[i] + 1.0
                for i in T.Parallel(N):
                    a_ub[i] = c_ub[i] + 2.0
            T.copy(c_ub, C)

    items = _inputs(main)
    assert [(item["vf_index"], item["nest_index"]) for item in items] == [(0, 0), (0, 1)]
    assert [item["status"] for item in items] == ["LIFTED", "LIFTED"]
    assert _stored_buffers(items[0]["stmt"]) == ["c_ub"]
    assert _stored_buffers(items[1]["stmt"]) == ["a_ub"]


# ---------------------------------------------------------------------------
# negative matrix (§6.4)
# ---------------------------------------------------------------------------


def _assert_no_stmt_rejected(func, allowed, reason_match: str):
    items = _inputs(func)
    assert len(items) == 1
    item = items[0]
    assert item["status"] in allowed, (item["status"], item.get("reason"))
    assert "stmt" not in item
    assert reason_match in item["reason"]


def test_inner_bind_is_opaque():
    @T.prim_func
    def main(A: T.Tensor((M * N,), "float32"), C: T.Tensor((M * N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M * N,), "float32")
            c_ub = T.alloc_shared((M * N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    t = i * N + j
                    c_ub[t] = a_ub[t] + 1.0
            T.copy(c_ub, C)

    _assert_no_stmt_rejected(main, {"OPAQUE", "FALLBACK"}, "tirx.Bind")


def test_reduction_falls_back():
    @T.prim_func
    def main(A: T.Tensor((M, N), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), "float32")
            c_ub = T.alloc_shared((M,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[i] += a_ub[i, j]
            T.copy(c_ub, C)

    _assert_no_stmt_rejected(main, {"FALLBACK"}, "ILLEGAL")


def test_local_var_is_opaque():
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float32")
            c_ub = T.alloc_shared((N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                v = T.alloc_var("float32", init=0.0)
                for i in T.Parallel(N):
                    c_ub[i] = a_ub[i] + v
            T.copy(c_ub, C)

    _assert_no_stmt_rejected(main, {"OPAQUE"}, "local.var")


def test_opaque_evaluate_is_opaque():
    @T.prim_func
    def main(A: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    T.evaluate(T.call_extern("float32", "opaque_side_effect", a_ub[i]))

    _assert_no_stmt_rejected(main, {"OPAQUE", "FALLBACK"}, "opaque")


def test_illegal_memory_falls_back():
    n = 256

    @T.prim_func
    def main(A: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n - 32):
                    a_ub[i] = a_ub[i + 32]

    items = _inputs(main)
    assert len(items) == 1
    assert items[0]["status"] == "FALLBACK"
    assert "memory legality is ILLEGAL" in items[0]["reason"]
    assert "stmt" not in items[0]


def test_mixed_legality_marks_whole_vf():
    """PR-A aggregates per VF: one illegal nest drags the legal sibling down."""
    n = 256

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n - 32):
                    c_ub[i] = a_ub[i] + 1.0
                for i in T.Parallel(n - 32):
                    a_ub[i] = a_ub[i + 32]
            T.copy(c_ub, C)

    items = _inputs(main)
    assert len(items) == 2
    assert [item["status"] for item in items] == ["FALLBACK", "FALLBACK"]
    assert all("stmt" not in item for item in items)


# ---------------------------------------------------------------------------
# Branch B: P1 LIFTED but P2 rejects (§10.2)
# ---------------------------------------------------------------------------


def test_gm_operand_branch_b():
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1), T.SimdVF():
            for i in T.Parallel(N):
                C[i] = A[i] + 1.0

    _assert_branch_b(main, "global-memory operand")


def test_bool_store_branch_b():
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "bool")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float32")
            c_ub = T.alloc_shared((N,), "bool")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    c_ub[i] = a_ub[i] > 0.0
            T.copy(c_ub, C)

    _assert_branch_b(main, "not supported")


def test_nonzero_min_branch_b():
    """TileLang has no nonzero-min T.Parallel, so patch the loop min directly."""

    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float32")
            c_ub = T.alloc_shared((N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    def rewrite(node):
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL:
            return tirx.For(node.loop_var, tirx.IntImm("int32", 1), node.extent, node.kind, node.body)
        return None

    body = tirx.stmt_functor.ir_transform(main.body, None, rewrite, ["tirx.For"])
    patched = main.with_body(body)

    item = _lifted(patched)
    assert int(item["stmt"].min) == 1
    with pytest.raises(Unsupported, match="must start at 0"):
        build_plan(item["stmt"])


def test_stencil_tail_branch_b():
    """`a_ub[i] = a_ub[i + bn]` is liftable but not vectorizable (stencil)."""
    n, bn = 1000, 256

    @T.prim_func
    def main(A: T.Tensor((n + bn,), "float32")):
        with T.Kernel(T.ceildiv(n, bn)) as bx:
            a_ub = T.alloc_shared((n + bn,), "float32")
            T.copy(A, a_ub)
            tail = T.min(n - bx * bn, bn)
            with T.SimdVF():
                for i in T.Parallel(tail):
                    a_ub[i] = a_ub[i + bn]

    assert "status=LIFTED" in _report(main)
    _assert_branch_b(main, "stencil")


# ---------------------------------------------------------------------------
# empty
# ---------------------------------------------------------------------------


def test_no_vf_returns_empty():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((16,), "float32")
            T.copy(A, a_ub)

    assert _inputs(main) == []


# ===========================================================================
# Part 2: pass level (``AscendPolyhedralVectorize`` + its three stages)
# ===========================================================================


def test_pass_replaces_vectorizable_nest():
    main = _kernel_1d()
    after = _run_pass(main)

    assert not tvm.ir.structural_equal(after.body, main.body, map_free_vars=True)
    assert _parallel_nests(after.body) == [], "the T.Parallel nest survived the pass"

    calls = _simd_calls(after.body)
    assert "tl.simd.vld" in calls
    assert "tl.simd.vsts" in calls
    assert any(name in ("tl.simd.vadd", "tl.simd.vadds") for name in calls)


def test_pass_matches_offline_driver():
    """The pass body is exactly the offline driver's output (single oracle)."""
    main = _kernel_1d()
    after = _run_pass(main)

    expected = vectorize_parallel_nests(main.body)
    tvm.ir.assert_structural_equal(after.body, expected, map_free_vars=True)


def test_pass_leaves_rejected_nest_untouched():
    main = _kernel_global_operand()
    with pytest.raises(Unsupported, match="global-memory operand"):
        build_plan(_only_parallel_nest(main.body))

    after = _run_pass(main)
    assert len(_parallel_nests(after.body)) == 1
    tvm.ir.assert_structural_equal(after.body, main.body, map_free_vars=True)


def test_pass_is_idempotent():
    once = _run_pass(_kernel_1d())
    twice = _run_pass(once)

    tvm.ir.assert_structural_equal(twice.body, once.body, map_free_vars=True)


# ---------------------------------------------------------------------------
# staged passes (P1 -> P2 -> P3 carriers)
# ---------------------------------------------------------------------------


def test_staged_passes_exchange_carriers():
    main = _kernel_1d()

    lifted = _run(main, at.AscendLiftParallelToArray)
    assert len(_carriers(lifted.body, "tl.vsimd.array_region")) == 1
    assert len(_parallel_nests(lifted.body)) == 1, "P1 must not rewrite the nest"
    assert _simd_calls(lifted.body) == []

    planned = _run(lifted, at.AscendPlanArray)
    assert _carriers(planned.body, "tl.vsimd.array_region") == []
    plans = _carriers(planned.body, "tl.vsimd.array_plan")
    assert len(plans) == 1
    assert len(plans[0].node.segments) == 1
    assert len(_parallel_nests(planned.body)) == 1

    emitted = _run(planned, at.AscendLowerArrayPlan)
    assert _carriers(emitted.body, "tl.vsimd.array_plan") == []
    assert _parallel_nests(emitted.body) == []
    assert "tl.simd.vsts" in _simd_calls(emitted.body)

    combined = _run_pass(main)
    tvm.ir.assert_structural_equal(combined.body, emitted.body, map_free_vars=True)


def test_p2_plans_the_carried_region():
    """P2 must build its plan from the P1 region, not re-read the nest body.

    The plan carried by the ``tl.vsimd.array_plan`` marker is exactly the plan
    built from the P1 hand-off statement (the region's reference expansion), so
    the two stage interfaces agree field for field.
    """
    main = _kernel_1d()

    planned = _run(_run(main, at.AscendLiftParallelToArray), at.AscendPlanArray)
    carried = _carriers(planned.body, "tl.vsimd.array_plan")
    assert len(carried) == 1

    from_region = build_plan(_lifted(main)["stmt"])
    assert render_plan(carried[0].node) == render_plan(from_region)


def test_lift_leaves_rejected_nest_untouched():
    """P1 rejection ⇒ no carrier and a byte-identical body (legacy still owns it)."""
    n = 256

    @T.prim_func
    def main(A: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n - 32):
                    a_ub[i] = a_ub[i + 32]

    lifted = _run(main, at.AscendLiftParallelToArray)
    assert _carriers(lifted.body, "tl.vsimd.array_region") == []
    tvm.ir.assert_structural_equal(lifted.body, main.body, map_free_vars=True)


def test_lift_is_idempotent():
    once = _run(_kernel_1d(), at.AscendLiftParallelToArray)
    twice = _run(once, at.AscendLiftParallelToArray)

    assert len(_carriers(twice.body, "tl.vsimd.array_region")) == 1


def test_p2_unwraps_nest_it_cannot_plan():
    main = _kernel_gm_operand_in_vf()

    lifted = _run(main, at.AscendLiftParallelToArray)
    assert len(_carriers(lifted.body, "tl.vsimd.array_region")) == 1

    planned = _run(lifted, at.AscendPlanArray)
    assert _carriers(planned.body, "tl.vsimd.array_region") == []
    assert _carriers(planned.body, "tl.vsimd.array_plan") == []
    assert len(_parallel_nests(planned.body)) == 1

    emitted = _run(planned, at.AscendLowerArrayPlan)
    tvm.ir.assert_structural_equal(emitted.body, main.body, map_free_vars=True)


# ---------------------------------------------------------------------------
# pipeline gate (``tl.ascend.polyhedral.impl``)
# ---------------------------------------------------------------------------


@pytest.fixture
def pass_calls(monkeypatch) -> list:
    """Record how often the pipeline constructs the pass."""
    calls: list = []
    real = at.AscendPolyhedralVectorize

    def spy():
        calls.append(True)
        return real()

    monkeypatch.setattr(at, "AscendPolyhedralVectorize", spy)
    return calls


def test_pipeline_default_does_not_insert_pass(pass_calls):
    source = _lower(_kernel_1d())

    assert pass_calls == []
    # The legacy lowering still produces the SIMD_VF body on its own.
    assert "simd_inst::" in source


def test_pipeline_inserts_pass_only_for_cpp(pass_calls):
    _lower(_kernel_1d(), "legacy")
    assert pass_calls == []

    _lower(_kernel_1d(), "cpp")
    assert pass_calls == [True]


def test_cpp_changes_device_source():
    legacy = _lower(_kernel_1d())
    cpp = _lower(_kernel_1d(), "cpp")

    assert cpp != legacy
    # a_ub[i] + 1.0: legacy broadcasts first (vdup + vadd), P3 uses the
    # scalar-vector recipe (vadds).
    assert "vdup" in legacy
    assert "vadds" in cpp
    assert "vadds" not in legacy


if __name__ == "__main__":
    tilelang.testing.main()
