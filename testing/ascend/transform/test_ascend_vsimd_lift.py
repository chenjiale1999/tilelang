"""Phase 0 F3: the P1 pass attaches a region to every top-level T.Parallel loop."""

import pytest

import tilelang
import tilelang.ascend.language as T
import tvm
from tvm import tirx
from tilelang.ascend import transform as at
from tilelang.ascend import vsimd_ir as ir

REGION_KEY = "tl.vsimd.region"


def _lift(func):
    mod = tvm.IRModule({"main": func})
    return at.AscendLiftVSimd()(mod)


def _regions(mod):
    """Return the region annotations in body order."""
    regions = []

    def visit(node):
        if isinstance(node, tirx.For) and REGION_KEY in dict(node.annotations):
            regions.append(node.annotations[REGION_KEY])

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return regions


def _vf_bodies(mod):
    bodies = []

    def visit(node):
        if isinstance(node, tirx.SBlock) and node.name_hint == "SIMD_VF":
            bodies.append(node.body)

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return bodies


def _strip_for_annotations(stmt):
    """Drop For annotations so a lifted body can be compared structurally."""

    def visit(node):
        if isinstance(node, tirx.For) and dict(node.annotations):
            return tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                {},
            )
        return None

    return tirx.stmt_functor.ir_transform(stmt, None, visit)


def test_region_attached_to_parallel_and_body_unchanged():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    out = _lift(main)
    before = _vf_bodies(tvm.IRModule({"main": main}))[0]
    after = _vf_bodies(out)[0]
    assert tvm.ir.structural_equal(_strip_for_annotations(before), _strip_for_annotations(after))

    regions = _regions(out)
    assert len(regions) == 1
    region = regions[0]
    assert region.status == "LIFTED"
    assert region.reason_code == ""
    assert int(region.vf_index) == 0
    assert int(region.region_index) == 0
    assert [str(axis.extent) for axis in region.axes] == ["64"]
    assert [op.kind for op in region.ops] == ["store"]
    assert [capture.buffer.name for capture in region.captures] == [
        "c_ub",
        "a_ub",
    ]


def test_independent_parallels_lift_independently():
    @T.prim_func
    def main(
        A: T.Tensor((96,), "float32"),
        D: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((96,), "float32")
            d_ub = T.alloc_shared((64,), "float32")
            c_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            T.copy(D, d_ub)
            with T.SimdVF():
                for i in T.Parallel(64):
                    a_ub[i] = a_ub[i + 32]  # cross-iteration RAW: illegal
                for i in T.Parallel(64):
                    c_ub[i] = d_ub[i] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 2
    assert regions[0].status == "OPAQUE"
    assert regions[0].reason_code == "ILLEGAL"
    assert regions[1].status == "LIFTED"


def test_shared_buffer_dependency_does_not_block_the_consumer():
    """A buffer dependency must not force the consumer down.

    Shared/global buffers always have the scalar fallback, so P1 keeps the
    regions independent and leaves the decision to P2.
    """

    @T.prim_func
    def main(
        A: T.Tensor((96,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((96,), "float32")
            c_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(64):
                    a_ub[i] = a_ub[i + 32]
                for i in T.Parallel(64):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert [region.status for region in regions] == ["OPAQUE", "LIFTED"]


def test_serial_wrapped_parallel_is_lifted():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            c_ub = T.alloc_shared((64,), "float32")
            tmp = T.alloc_shared((32,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for w in T.serial(2):
                    for i in T.Parallel(32):
                        tmp[i] = 1.0 + w
                for i in T.Parallel(64):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    mod = _lift(main)
    regions = _regions(mod)
    assert len(regions) == 2
    assert [region.status for region in regions] == ["LIFTED", "LIFTED"]
    assert [int(axis.extent) for axis in regions[0].axes] == [32]
    assert [int(axis.extent) for axis in regions[1].axes] == [64]
    # The enclosing serial loop stays in place: it becomes the scalar shell
    # around the emitted vector code and keeps binding ``w``.
    serial = []

    def visit(node):
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.SERIAL:
            serial.append(node)

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert any(node.loop_var.name == "w" for node in serial)


def test_serial_wrapper_does_not_hide_the_nest():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for w in T.serial(2):
                    for i in T.Parallel(64):
                        a_ub[i] = 1.0 + w
            T.copy(a_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"
    assert [int(axis.extent) for axis in regions[0].axes] == [64]


def test_no_parallel_no_annotation():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n, 64), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, 64), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                T.reduce_sum(a_ub, c_ub, dim=1, clear=True)
            T.copy(c_ub, C)

    out = _lift(main)
    assert _regions(out) == []


def test_assume_wrapper_is_transparent():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                T.attr(n > 0, "tl.assume", "test assume")
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"


def test_dynamic_shell_bound_is_lifted():
    """A lane-invariant runtime shell bound is liftable (P2 keeps it as a loop)."""
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n, 64), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, 64), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    for j in T.serial(64):
                        c_ub[i] = a_ub[i, j]
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"


def test_lane_dependent_shell_bound_is_opaque():
    """A shell bound that reads the lane index is not lane invariant -> OPAQUE."""
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n, n), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, n), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    for j in T.serial(i + 1):
                        c_ub[i] = a_ub[i, j]
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    assert regions[0].status == "OPAQUE"
    assert "lane index" in regions[0].reason


def test_multi_shell_body_is_distributed():
    """A nest body mixing two serial shells is distributed into one nest per shell."""
    n, m = 64, 32

    @T.prim_func
    def main(A: T.Tensor((n, m), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, m), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    for j in T.serial(m):
                        a_ub[i, j] = a_ub[i, j] * 0.5
                    for j in T.serial(m):
                        c_ub[i] = T.max(c_ub[i], a_ub[i, j])
            T.copy(c_ub, C)

    mod = _lift(main)
    regions = _regions(mod)
    assert len(regions) == 2
    assert [region.status for region in regions] == ["LIFTED", "LIFTED"]
    kinds = [[op.kind for op in region.ops] for region in regions]
    assert kinds == [["serial"], ["serial"]]
    # Both distributed nests keep the same domain and source order.
    assert [[int(axis.extent) for axis in region.axes] for region in regions] == [[n], [n]]
    parallels = []

    def visit(node):
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL:
            parallels.append(node)

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert len(parallels) == 2


def test_region_annotation_survives_simplify():
    n = 32

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    lifted = _lift(main)
    simplified = tilelang.transform.Simplify()(lifted)
    regions = _regions(simplified)
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"


def test_if_inside_parallel_is_lifted():
    """A bundled ``if`` is modelled as a guarded op and stays vectorizable."""
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if a_ub[i] > 0.0:
                        c_ub[i] = a_ub[i] + 1.0
                    else:
                        c_ub[i] = 0.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    region = regions[0]
    assert region.status == "LIFTED"
    assert region.reason_code == ""
    assert [op.kind for op in region.ops] == ["if"]
    # The branch condition is the op guard; both arms are modelled.
    assert "a_ub" in str(region.ops[0].guard)
    assert [op.kind for op in region.ops[0].ops] == ["store"]
    assert [op.kind for op in region.ops[0].else_ops] == ["store"]


def test_tileop_inside_parallel_is_opaque_per_nest():
    """A tile-op inside ``T.Parallel`` is not modelled (front-end forbids it).

    The semantic check rejects such a program before P1 runs, so P1 has no
    tile-op grammar: the nest is reported OPAQUE with a diagnostic and the
    sibling Parallel in the same VF still lifts on its own.
    """
    br, bc = 64, 128

    @T.prim_func
    def main(A: T.Tensor((br, bc), "float32"), M: T.Tensor((br,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((br, bc), "float32")
            m_ub = T.alloc_shared((br,), "float32")
            T.copy(A, s_ub)
            with T.SimdVF():
                for i in T.Parallel(br):
                    T.reduce_max(s_ub[i, :], m_ub[i : i + 1], dim=1, clear=False)
                for i in T.Parallel(br):
                    m_ub[i] = s_ub[i, 0] + 1.0
            T.copy(m_ub, M)

    regions = _regions(_lift(main))
    assert [region.status for region in regions] == ["OPAQUE", "LIFTED"]
    assert regions[0].reason_code == "UNSUPPORTED"
    assert "tl.tileop.reduce" in regions[0].reason


def test_unsupported_tileop_rejects_only_its_nest():
    """An unmodelled tile-op is a per-nest fallback, not a VF-level poison."""
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for _i in T.Parallel(n):
                    T.fill(c_ub, 0.0)
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert [region.status for region in regions] == ["OPAQUE", "LIFTED"]
    assert regions[0].reason_code == "UNSUPPORTED"
    assert "tl.tileop.fill" in regions[0].reason
    assert regions[1].reason_code == ""


def test_copy_tileop_does_not_poison_the_sibling_nest():
    """A copy inside ``T.Parallel`` is a per-nest fallback, not a VF poison."""
    m, n = 8, 32

    @T.prim_func
    def main(A: T.Tensor((n, m), "float32"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, m), "float32")
            c_ub = T.alloc_shared((m, n), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(m):
                    T.copy(a_ub[:, i], c_ub[i, :], transpose=True)
                for i in T.Parallel(m):
                    c_ub[i, 0] = c_ub[i, 0] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert [region.status for region in regions] == ["OPAQUE", "LIFTED"]
    assert regions[0].reason_code == "UNSUPPORTED"
    assert "tl.tileop.ascend_copy" in regions[0].reason


def test_sibling_parallels_are_unsupported_not_internal():
    """A Parallel body that is a sequence of loops is not one rectangular nest.

    The nest model collapses only a body that is exactly one Parallel loop; a
    body that mixes sibling loops with other statements must be reported as
    unsupported input, not as a P1 internal error. This pins the target-FA shape
    and the unit-extent variant that used to reach the reconstruction oracle.
    """
    br, bc = 64, 128

    @T.prim_func
    def fa_shape(A: T.Tensor((br, bc), "float32"), M: T.Tensor((br,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((br, bc), "float32")
            m_ub = T.alloc_shared((br,), "float32")
            T.copy(A, s_ub)
            with T.SimdVF():
                for i in T.Parallel(br):
                    for j in T.Parallel(bc):
                        s_ub[i, j] = s_ub[i, j] * 0.5
                    T.reduce_max(s_ub[i, 0:bc], m_ub[i : i + 1], dim=1, clear=False)
            T.copy(m_ub, M)

    @T.prim_func
    def unit_siblings(A: T.Tensor((4,), "float32"), C: T.Tensor((4,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((4,), "float32")
            c_ub = T.alloc_shared((4,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(4):
                    for _j in T.Parallel(1):
                        a_ub[i] = a_ub[i] * 2.0
                    for _k in T.Parallel(1):
                        c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    for func in (fa_shape, unit_siblings):
        regions = _regions(_lift(func))
        assert len(regions) == 1
        region = regions[0]
        assert region.status == "OPAQUE"
        assert region.reason_code == "UNSUPPORTED"
        assert "not a single rectangular nest" in region.reason


def test_unused_parallel_axis_is_rejected():
    """A Parallel axis the body never mentions is a hard error in P1.

    P2 would hoist the dead axis into an outer serial loop and replay the whole
    vector body once per dead index. The nest is not left to the legacy path:
    inside a ``T.SimdVF`` that path emits a scalar loop into the ``__simd_vf__``
    helper, which bisheng refuses, so the DSL mistake is reported directly.
    """
    m, n = 8, 64

    @T.prim_func
    def main(S: T.Tensor((m, n), "float32"), D: T.Tensor((m,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((m, n), "float32")
            d_ub = T.alloc_shared((m,), "float32")
            T.copy(S, s_ub)
            T.fill(d_ub, 0.0)
            with T.SimdVF():
                for i, _j in T.Parallel(m, n):
                    for k in T.serial(n):
                        d_ub[i] = d_ub[i] + s_ub[i, k]
            T.copy(d_ub, D)

    with pytest.raises(Exception, match="never used"):
        _lift(main)


def test_unit_extent_parallel_axis_is_rejected():
    """``T.Parallel(1)`` carries no lane, so the nest would be scalar.

    ``Simplify`` folds the counter into the body before P1 (``i`` becomes 0), so
    the nest has no vector domain and the ``__simd_vf__`` helper would hold a
    scalar loop, which bisheng refuses ("Unsupported scalar instruction in AIV
    loop"). P1 reports that cause directly (device report: ``rows, k = 1, 100``).
    """
    n = 100

    @T.prim_func
    def main(A: T.Tensor((1, n), "float32"), C: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((1, n), "float32")
            c_ub = T.alloc_shared((1,), "float32")
            T.copy(A, a_ub)
            T.fill(c_ub, 0.0)
            with T.SimdVF():
                for i in T.Parallel(1):
                    for j in T.serial(n):
                        c_ub[i] = c_ub[i] + a_ub[i, j]
            T.copy(c_ub, C)

    with pytest.raises(Exception, match="extent 1"):
        _lift(main)


def test_reduction_over_a_dead_parallel_axis_is_rejected():
    """An accumulation that ignores a Parallel axis is a hard error in P1.

    ``d_ub[i] += s_ub[i, k]`` never reads ``j``, so the ``T.Parallel(m, n)``
    nest would replay the whole reduction once per ``j`` (device report D3,
    ``mixed_recipe``'s D output). The scalar program has the same defect, so the
    nest is not silently "fixed": P1 reports it and stops the compile.
    """
    m, n = 8, 64

    @T.prim_func
    def main(
        A: T.Tensor((m, n), "float32"),
        S: T.Tensor((m, n), "float32"),
        C: T.Tensor((m, n), "float32"),
        D: T.Tensor((m,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((m, n), "float32")
            s_ub = T.alloc_shared((m, n), "float32")
            c_ub = T.alloc_shared((m, n), "float32")
            d_ub = T.alloc_shared((m,), "float32")
            T.copy(A, a_ub)
            T.copy(S, s_ub)
            T.fill(d_ub, 0.0)
            with T.SimdVF():
                for i, j in T.Parallel(m, n):
                    c_ub[i, j] = a_ub[i, j] * 2.0
                    for k in T.serial(n):
                        d_ub[i] = d_ub[i] + s_ub[i, k]
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    with pytest.raises(Exception, match="ignores T.Parallel axis 'j'"):
        _lift(main)


def test_reduction_along_the_vector_axis_is_still_lifted():
    """A reduction that *does* read the axis (``d[i] += s[i, j]``) stays legal."""
    m, n = 8, 64

    @T.prim_func
    def main(S: T.Tensor((m, n), "float32"), D: T.Tensor((m,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((m, n), "float32")
            d_ub = T.alloc_shared((m,), "float32")
            T.copy(S, s_ub)
            T.fill(d_ub, 0.0)
            with T.SimdVF():
                for i, j in T.Parallel(m, n):
                    d_ub[i] = d_ub[i] + s_ub[i, j]
            T.copy(d_ub, D)

    regions = _regions(_lift(main))
    assert [region.status for region in regions] == ["LIFTED"]


def test_manual_vf_region_keeps_footprint():
    """A manual VF rejects its regions, but they still record what they touch."""

    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                v = T.alloc_var("float32", init=0.0)
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + v
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    region = regions[0]
    assert region.status == "OPAQUE"
    # Manual is a VF-level verdict, so the region reports UNSUPPORTED and the
    # VF-level message stays in the reason text.
    assert region.reason_code == "UNSUPPORTED"
    assert "manual" in region.reason
    assert "VF-level rejection" in region.reason

    captures = {capture.buffer.name: capture for capture in region.captures}
    assert {"a_ub", "c_ub"} <= set(captures)
    assert all(capture.kind == "unknown" for capture in region.captures)
    op = region.ops[0]
    assert op.kind == "opaque"
    assert "c_ub" in [buffer.name for buffer in op.writes]
    assert "a_ub" in [buffer.name for buffer in op.reads]


def test_view_and_base_share_one_capture():
    """A footprint seeing a T.view and its base must not duplicate the capture.

    Capture identity is the data var, and the verifier rejects two captures for
    one data var; here the opaque region is produced from the syntactic
    footprint, which sees both buffers.
    """

    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            base = T.alloc_shared((n,), "float32")
            T.copy(A, base)
            with T.SimdVF():
                v = T.alloc_var("float32", init=0.0)  # forces the manual path
                for i in T.Parallel(n // 2):
                    view = T.view(base, shape=(n // 2, 2), dtype="float32")
                    view[i, 0] = base[2 * i] + v
            T.copy(base, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    region = regions[0]
    assert region.status == "OPAQUE"
    data_vars = [str(capture.buffer.data) for capture in region.captures]
    assert len(data_vars) == len(set(data_vars))
    assert ir.verify_region(region) == ""


def test_p1_can_be_disabled_by_config():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ASCEND_SIMD_LIFT: False}):
        out = _lift(main)
    assert _regions(out) == []


def test_default_pipeline_vectorizes_the_parallel_nest():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    with tvm.target.Target("ascend"):
        out = tilelang.lower(main)
    # The default pipeline runs P1 -> P2 -> P3, so the nest becomes vector code.
    assert "main_kernel_simd_vf_0" in out.kernel_source
    assert "simd_inst::" in out.kernel_source
    assert "tl.vsimd.region" not in out.kernel_source
    assert "tl.vsimd.plan" not in out.kernel_source


def _rewrite_loop_bounds(func, kind, *, min_value=None, step=None):
    """Rewrite the first loop of ``kind`` into a hand-written shifted/strided form.

    ``T.Parallel`` / ``T.serial`` always build a unit-step loop from the frontend,
    so a non-zero ``min`` or a non-unit ``step`` can only come from hand-written
    TIR; this helper produces exactly that input for the P1 shape gate.
    """

    def visit(node):
        if isinstance(node, tirx.For) and node.kind == kind:
            return tirx.For(
                node.loop_var,
                node.min if min_value is None else min_value,
                node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                node.annotations,
                node.step if step is None else step,
            )
        return None

    return func.with_body(tirx.stmt_functor.ir_transform(func.body, None, visit))


def _parallel_kernel(n=64):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    return main


def test_parallel_with_non_zero_min_is_opaque():
    """The region models ``[0, extent)``: a shifted Parallel must not be lifted."""
    out = _lift(_rewrite_loop_bounds(_parallel_kernel(), tirx.ForKind.PARALLEL, min_value=tirx.IntImm("int32", 3)))
    regions = _regions(out)
    assert len(regions) == 1
    assert regions[0].status == "OPAQUE"
    assert regions[0].reason_code == "UNSUPPORTED"
    assert "must start at 0" in regions[0].reason


def test_parallel_with_non_unit_step_is_opaque():
    """``step != 1`` visits a different index set, so the nest falls back."""
    out = _lift(_rewrite_loop_bounds(_parallel_kernel(), tirx.ForKind.PARALLEL, step=tirx.IntImm("int32", 2)))
    regions = _regions(out)
    assert len(regions) == 1
    assert regions[0].status == "OPAQUE"
    assert regions[0].reason_code == "UNSUPPORTED"
    assert "must have step 1" in regions[0].reason


def test_serial_shell_with_non_unit_step_is_opaque():
    """The planner replicates a shell as ``min + iter``; a stride breaks that."""
    n, steps = 16, 4

    @T.prim_func
    def main(A: T.Tensor((n, steps), "float32"), C: T.Tensor((n, steps), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, steps), "float32")
            c_ub = T.alloc_shared((n, steps), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    for k in T.serial(steps):
                        c_ub[i, k] = a_ub[i, k] + 1.0
            T.copy(c_ub, C)

    assert _regions(_lift(main))[0].status == "LIFTED"

    out = _lift(_rewrite_loop_bounds(main, tirx.ForKind.SERIAL, step=tirx.IntImm("int32", 2)))
    regions = _regions(out)
    assert len(regions) == 1
    assert regions[0].status == "OPAQUE"
    assert regions[0].reason_code == "UNSUPPORTED"
    assert "T.serial shell inside T.Parallel must have step 1" in regions[0].reason


if __name__ == "__main__":
    tilelang.testing.main()
