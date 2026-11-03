"""Phase 1: P1 metadata pass structural tests."""

import tilelang
import tilelang.ascend.language as T
import tvm
from tilelang.ascend import transform as at

PROGRAM_KEY = "tl.vsimd.program"


def _lift(func):
    mod = tvm.IRModule({"main": func})
    return at.AscendLiftVSimd()(mod)


def _programs(mod):
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.SBlock):
            if PROGRAM_KEY in node.annotations:
                found.append(node.annotations[PROGRAM_KEY])

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    for program in found:
        assert "validation_error" not in program
    return found


def _vf_bodies(mod):
    bodies = []

    def visit(node):
        if isinstance(node, tvm.tirx.SBlock) and node.name_hint == "SIMD_VF":
            bodies.append(node.body)

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return bodies


def _items(program):
    return list(program["items"])


def _assert_dense_spans(items):
    for index, item in enumerate(items):
        assert int(item["span_begin"]) == index
        assert int(item["span_end"]) == index + 1


def test_simple_program_attached_and_body_unchanged():
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
    before_vf = _vf_bodies(tvm.IRModule({"main": main}))
    after_vf = _vf_bodies(out)
    assert len(before_vf) == len(after_vf) == 1
    assert tvm.ir.structural_equal(before_vf[0], after_vf[0])
    programs = _programs(out)
    assert len(programs) == 1
    items = _items(programs[0])
    assert len(items) == 1
    _assert_dense_spans(items)
    assert items[0]["kind"] == "REGION"
    assert items[0]["status"] == "LIFTED"
    assert items[0]["span_begin"] == 0
    assert items[0]["span_end"] == 1


def test_independent_regions_partial_lift():
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
                    a_ub[i] = a_ub[i + 32]
                for i in T.Parallel(64):
                    c_ub[i] = d_ub[i] + 1.0
            T.copy(c_ub, C)

    programs = _programs(_lift(main))
    assert len(programs) == 1
    items = _items(programs[0])
    assert len(items) == 2
    _assert_dense_spans(items)
    assert items[0]["kind"] == "REGION"
    assert items[0]["status"] == "OPAQUE"
    assert items[1]["kind"] == "REGION"
    assert items[1]["status"] == "LIFTED"
    # No dependency between the two regions: they must not share a group.
    assert items[0]["group"] != items[1]["group"]
    group_ids = [int(group["id"]) for group in programs[0]["groups"]]
    assert group_ids == sorted(group_ids)


def test_fa_copy_reduce_fragment_program():
    """FA-like single-VF fragment + copy/reduce chain.

    The shared FA example is kept in its pre-fragment form until the P2/P3
    group path supports COPY/REDUCE. This synthetic kernel exercises the same
    P1 records without changing the example used by the legacy P2/P3 tests.
    """

    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((128, 128), "float32")
            m_old = T.alloc_shared((128,), "float32")
            T.copy(A, s_ub)
            with T.SimdVF():
                s_frag = T.alloc_fragment((128, 128), "float32")
                m_new = T.alloc_fragment((128,), "float32")
                l_cur = T.alloc_fragment((128,), "float32")
                for i, j in T.Parallel(128, 128):
                    s_frag[i, j] = s_ub[i, j] * 2.0
                T.copy(m_old, m_new)
                T.reduce_max(s_frag, m_new, dim=1, clear=False)
                T.reduce_sum(s_frag, l_cur, dim=1, clear=True)
                for i in T.Parallel(128):
                    m_old[i] = m_new[i] + l_cur[i]
            T.copy(m_old, C)

    programs = _programs(_lift(main))
    assert len(programs) == 1
    program = programs[0]
    assert int(program["vf_index"]) == 0
    items = _items(program)
    _assert_dense_spans(items)
    kinds = [item["kind"] for item in items]
    assert "REGION" in kinds
    assert "COPY" in kinds
    assert "REDUCE" in kinds
    copy_items = [item for item in items if item["kind"] == "COPY"]
    assert copy_items
    assert all("src" in item and "dst" in item for item in copy_items)
    assert all(item["src"]["dtype"] for item in copy_items)
    assert all(item["src_region"]["extents"] for item in copy_items)
    assert all(item["dst_region"]["extents"] for item in copy_items)
    assert len({int(item["group"]) for item in items}) == 1
    assert all(item["group_status"] == "P1_LIFTABLE" for item in items)
    assert any(capture["storage"] == "FRAGMENT" for capture in program["captures"])
    assert program["fragments"]
    fragment_names = {fragment["name"] for fragment in program["fragments"]}
    assert len(fragment_names) == 3
    assert any("s_frag" in name for name in fragment_names)
    assert any("m_new" in name for name in fragment_names)
    assert any("l_cur" in name for name in fragment_names)
    for fragment in program["fragments"]:
        assert fragment["definition_scope"] == "IN_VF"
        assert int(fragment["definition_vf"]) == 0
        assert not bool(fragment["cross_vf"])
        assert not bool(fragment["escapes_vf"])


def test_group_opaque_downgrades_consumer():
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

    programs = _programs(_lift(main))
    assert len(programs) == 1
    items = _items(programs[0])
    _assert_dense_spans(items)
    assert items[0]["group"] == items[1]["group"]
    assert items[0]["group_status"] == "OPAQUE"
    assert items[0]["status"] == "OPAQUE"
    assert items[1]["status"] == "OPAQUE"
    assert items[1]["analysis_status"] == "LIFTED"


def test_metadata_bind_is_preserve_not_liftable():
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

    x = tvm.tirx.Var("x", "int32")
    bind = tvm.tirx.Bind(x, tvm.tirx.IntImm("int32", 1))

    def inject(node):
        if isinstance(node, tvm.tirx.SBlock) and node.name_hint == "SIMD_VF":
            return tvm.tirx.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                tvm.tirx.SeqStmt([bind, node.body]),
                node.init,
                node.alloc_buffers,
                node.match_buffers,
                node.annotations,
            )
        return None

    body = tvm.tirx.stmt_functor.ir_transform(main.body, None, inject)
    func = main.with_body(body)
    programs = _programs(_lift(func))
    assert len(programs) == 1
    items = _items(programs[0])
    _assert_dense_spans(items)
    assert items[0]["kind"] == "BIND"
    assert items[0]["status"] == "PRESERVE"
    assert items[0]["group_status"] == "PRESERVE"
    assert items[1]["status"] == "LIFTED"


def test_parallel_ordinal_matches_capture_under_serial():
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

    programs = _programs(_lift(main))
    assert len(programs) == 1
    items = _items(programs[0])
    _assert_dense_spans(items)
    assert items[0]["kind"] == "OPAQUE"
    assert items[1]["kind"] == "REGION"
    assert items[1]["parallel_nest_index"] == 1
    assert [int(x["extent"]) for x in items[1]["region"]["domain"]] == [64]


def test_nested_parallel_without_top_level_is_all_opaque():
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

    programs = _programs(_lift(main))
    assert len(programs) == 1
    items = _items(programs[0])
    _assert_dense_spans(items)
    assert items
    assert all(item["status"] == "OPAQUE" for item in items)
    assert all(item["group_status"] == "OPAQUE" for item in items)


def test_nested_parallel_keeps_metadata_preserve_in_all_opaque_program():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for w in T.serial(2):
                    for i in T.Parallel(n):
                        a_ub[i] = 1.0 + w
            T.copy(a_ub, C)

    view = tvm.tirx.decl_buffer((n,), "float32", name="v", scope="shared")
    decl = tvm.tirx.DeclBuffer(view)

    def inject(node):
        if isinstance(node, tvm.tirx.SBlock) and node.name_hint == "SIMD_VF":
            return tvm.tirx.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                tvm.tirx.SeqStmt([decl, node.body]),
                node.init,
                node.alloc_buffers,
                node.match_buffers,
                node.annotations,
            )
        return None

    body = tvm.tirx.stmt_functor.ir_transform(main.body, None, inject)
    programs = _programs(_lift(main.with_body(body)))
    assert len(programs) == 1
    items = _items(programs[0])
    _assert_dense_spans(items)
    assert [item["kind"] for item in items] == ["DECL", "OPAQUE"]
    # The all-OPAQUE fallback must leave metadata items PRESERVE: clobbering
    # them to OPAQUE breaks the metadata/preserve invariant that the P1
    # verifier (and P2's reader) enforces.
    assert items[0]["status"] == "PRESERVE"
    assert items[0]["group_status"] == "PRESERVE"
    assert items[1]["status"] == "OPAQUE"


def test_assume_is_program_metadata():
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

    programs = _programs(_lift(main))
    assert len(programs) == 1
    items = _items(programs[0])
    assert [item["kind"] for item in items] == ["REGION"]
    _assert_dense_spans(items)
    assert len(programs[0]["assumptions"]) == 1
    assert "condition" in programs[0]["assumptions"][0]
    assert int(programs[0]["assumptions"][0]["anchor"]) == 0
    assert int(programs[0]["assumptions"][0]["anchor_end"]) == 1
    assert items[0]["status"] == "LIFTED"


def test_no_parallel_no_program():
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

    programs = _programs(_lift(main))
    assert programs == []


def test_program_survives_simplify():
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
    assert len(_programs(simplified)) == 1


def test_cross_vf_fragment_metadata():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((n,), "float32")
            with T.SimdVF():
                for i in T.Parallel(n):
                    frag[i] = A[i]
            with T.SimdVF():
                for i in T.Parallel(n):
                    C[i] = frag[i] + 1.0

    programs = _programs(_lift(main))
    assert programs
    for program in programs:
        _assert_dense_spans(_items(program))
    fragments = {
        f["name"]: f for p in programs for f in p["fragments"]
    }
    assert "frag" in fragments
    assert fragments["frag"]["definition_scope"] == "OUTSIDE_VF"
    assert bool(fragments["frag"]["cross_vf"]) is True
    assert bool(fragments["frag"]["escapes_vf"]) is False
    assert {int(v) for v in fragments["frag"]["use_vfs"]} == {0, 1}


def test_fragment_outside_vf_single_use_is_not_cross_vf():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((n,), "float32")
            with T.SimdVF():
                for i in T.Parallel(n):
                    frag[i] = A[i] + 1.0
                    C[i] = frag[i]

    programs = _programs(_lift(main))
    assert len(programs) == 1
    fragments = {f["name"]: f for f in programs[0]["fragments"]}
    assert "frag" in fragments
    assert fragments["frag"]["definition_scope"] == "OUTSIDE_VF"
    assert bool(fragments["frag"]["cross_vf"]) is False
    assert bool(fragments["frag"]["escapes_vf"]) is False


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

    with tvm.transform.PassContext(
        config={tilelang.PassConfigKey.TL_ASCEND_SIMD_LIFT: False}
    ):
        out = _lift(main)
    assert _programs(out) == []


def test_default_pipeline_keeps_legacy_lowering():
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
    # The pass is metadata-only; the legacy parallel lowering still runs.
    assert out is not None
    assert "main_kernel_simd_vf_0" in out.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
