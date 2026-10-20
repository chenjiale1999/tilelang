"""Offline structural tests for the unified polyhedral vectorizer (方案 §315-484).

Hardware independent and compiler independent: each case builds TileLang IR,
runs the polyhedral analysis (:func:`build_plan`) / codegen (:func:`emit_plan`)
directly, and inspects the resulting :class:`VectorPlan` plus the emitted
``T.simd.*`` call sequence.  No PTOAS / Bisheng / device execution is involved.

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
  body with loop-carried state (private scalar accumulator S2, shared scalar S4,
  stencil cross-iteration read) is *not* absorbed and the whole nest is rejected
  with "nested serial loop" (no silent downgrade).
* 改进 7      ``VectorPlan.diagnostics`` records every axis on success;
  ``Unsupported.diagnostics`` carries the per-axis notes on rejection.
* transform   ``vectorize_parallel_nests`` lowers a whole multi-axis nest
  (outer axes → serial, vector axis → strip-mined chunks).

The mathematical core (``finite_difference`` / ``classify_delta``) is unit
tested directly, since every case above is decided by the sign/shape of
``Δ(offset)`` along the candidate vector axis.
"""

from __future__ import annotations

import pytest

import tilelang.ascend.language as T
from tvm import arith, tirx

from tilelang.ascend.analysis.polyhedral import (
    AccessClass,
    Unsupported,
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


def _recipes(node) -> list:
    """Flatten a :class:`ValueNode` DAG into its recipe names."""
    out = [node.recipe]
    for arg in node.args:
        out.extend(_recipes(arg))
    return out


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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
    assert plan.vector_axis.name == "j"
    store = plan.stores[0]
    assert store.mask is not None
    # ``i < j`` with a uniform ``i`` → ``vcmps(j, i, "gt")`` (predicate flipped).
    assert store.mask.recipe == "vcmps" and store.mask.detail == "gt"
    ops = _emitted_ops(emit_plan(plan))
    # tail mask (P=8 < lanes) and the predicate combine via pand.
    assert "tl.simd.update_mask" in ops and "tl.simd.pand" in ops


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

    plan = _plan(k)
    assert plan.stores[0].value.recipe == "vsel"
    ops = _emitted_ops(emit_plan(plan))
    # a lane-varying condition compares per lane (vcmp); it is *not* broadcast.
    assert "tl.simd.vcmp" in ops
    assert "tl.simd.update_mask" not in ops


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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
    assert plan.vector_axis.name == "j"
    value = plan.stores[0].value
    assert value.recipe == "vsub"
    # ``g[i]`` is lane-uniform along j → broadcast; ``g[j]`` is a vector load.
    assert value.args[0].recipe == "vbrc"
    assert value.args[1].recipe == "vload"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vdup" in ops and "tl.simd.vld" in ops and "tl.simd.vsub" in ops


def test_reduction_detected():
    M, K = 4, 256

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            acc = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                acc[i] = acc[i] + a[i, j]

    plan = _plan(k)
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

    plan = _plan(k)
    assert plan.reduce is not None and plan.reduce.recipe == "vcmax"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcmax" in ops
    # accumulate form (``dst = max(dst, x)``) merges the old destination value.
    assert "tl.simd.vld" in ops


def test_reduction_with_tail_fills_identity():
    """``extent % lanes != 0``: the last chunk is a partial tile.  Its inactive
    lanes read out of range, so they must be replaced by the reduction identity
    (``update_mask`` + ``vsel``) before the cross-lane reduction (改进 6)."""
    M, K = 4, 32  # K < 64 lanes -> tail != 0

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            acc = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                acc[i] = acc[i] + a[i, j]

    plan = _plan(k)
    assert plan.reduce is not None and plan.tail == 32
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.update_mask" in ops  # 尾块谓词
    assert "tl.simd.vsel" in ops  # 单位元填充
    assert "tl.simd.vcadd" in ops  # 跨 lane 归约
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
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

    plan = _plan(k)
    value = plan.stores[0].value
    assert value.recipe == "gather" and value.detail == AccessClass.PERMUTE.value
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

    plan = _plan(k)
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

    plan = _plan(k)
    assert plan.vector_axis.name == "j"
    assert [ax.name for ax in plan.axes] == ["i", "j"]
    assert all(not ax.is_parallel for ax in plan.axes if ax.name == "j")
    assert plan.stores[0].kind.value == "contiguous"
    assert plan.reduce is None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vld" in ops and "tl.simd.vsts" in ops


def test_direct_serial_predicated_absorbed():
    """S1 + 谓词：serial 体内的 ``if`` 降成谓词化 store，掩码沿 serial 向量轴逐 lane 变化。"""
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

    plan = _plan(k)
    assert plan.vector_axis.name == "j"
    assert plan.stores[0].mask is not None
    assert "tl.simd.vcmps" in _emitted_ops(emit_plan(plan))


def test_direct_serial_reduction_absorbed():
    """S3：``for i in P: for j in serial: s[i] += a[i,j]``。沿 j 的累积归约是合法的
    循环携带依赖（同址自读）→ j 被吸收并选为向量轴，走 vcadd 跨 lane 归约。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] += a[i, j]

    plan = _plan(k)
    assert plan.vector_axis.name == "j"
    assert plan.reduce is not None and plan.reduce.recipe == "vcadd"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcadd" in ops and "tl.simd.vsts" in ops


def test_serial_private_scalar_rejected():
    """S2 边界：迭代私有**标量**累加器。``acc``(local.var) 的 init/store 把 serial 循环
    夹在中间，不是"直接内嵌"；且从 SCoP 看 acc[0] 是 lane 均匀的共享标量（与 S4 同形）。
    → 拒绝吸收，退回 "nested serial loop" 明确拒绝。"""
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

    with pytest.raises(Unsupported, match="nested serial loop"):
        _plan(k)


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

    plan = _plan(k)
    assert [a.name for a in plan.axes] == ["i", "j"]
    assert plan.vector_axis.name == "j"
