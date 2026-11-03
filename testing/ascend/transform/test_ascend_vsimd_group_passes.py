"""P2/P3 group passes over the P1 ``tl.vsimd.program`` annotation."""

import tilelang
import tilelang.ascend.language as T
import tvm
from tilelang.ascend import transform as at

PROGRAM_KEY = "tl.vsimd.program"
GROUP_PLAN_KEY = "tl.vsimd.group_plan"


def _annotations(mod, key):
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.SBlock) and key in node.annotations:
            found.append(node.annotations[key])

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return found


def _simd_ops(mod):
    ops = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            name = str(getattr(node.op, "name", ""))
            if name.startswith("tl.simd."):
                ops.append(name)

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return ops


def _lift(func):
    print(f"Before lift: {func}")
    # return at.AscendLiftVSimd()(tvm.IRModule({"main": func}))
    mod = at.AscendLiftVSimd()(tvm.IRModule({"main": func}))
    print(f"After lift: {mod}")
    return mod


def _plan(mod):
    mod = at.AscendPlanVSimd()(mod)
    print(f"After plan: {mod}")
    return mod


def _emit(mod):
    mod = at.AscendEmitVSimd()(mod)
    print(f"After emit: {mod}")
    return mod


@T.prim_func
def _simple(A: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
    with T.Kernel(1):
        a_ub = T.alloc_shared((64,), "float32")
        c_ub = T.alloc_shared((64,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            for i in T.Parallel(64):
                c_ub[i] = a_ub[i] + 1.0
        T.copy(c_ub, C)


@T.prim_func
def _partial(A: T.Tensor((96,), "float32"), D: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
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


@T.prim_func
def _copy_reduce(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128,), "float32")):
    with T.Kernel(1):
        s_ub = T.alloc_shared((128, 128), "float32")
        m_old = T.alloc_shared((128,), "float32")
        m_new = T.alloc_shared((128,), "float32")
        T.copy(A, s_ub)
        with T.SimdVF():
            T.copy(m_old, m_new)
            T.reduce_max(s_ub, m_new, dim=1, clear=False)
            for i in T.Parallel(128):
                m_old[i] = m_new[i] + 1.0
        T.copy(m_old, C)


@T.prim_func
def _reduce_clear(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128,), "float32")):
    with T.Kernel(1):
        s_ub = T.alloc_shared((128, 128), "float32")
        m_ub = T.alloc_shared((128,), "float32")
        out_ub = T.alloc_shared((128,), "float32")
        T.copy(A, s_ub)
        with T.SimdVF():
            T.reduce_sum(s_ub, m_ub, dim=1, clear=True)
            for i in T.Parallel(128):
                out_ub[i] = m_ub[i] + 1.0
        T.copy(out_ub, C)


@T.prim_func
def _partial_copy(A: T.Tensor((128, 128), "float32"), C: T.Tensor((64, 64), "float32")):
    with T.Kernel(1):
        a_ub = T.alloc_shared((128, 128), "float32")
        c_ub = T.alloc_shared((64, 64), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            T.copy(a_ub[32:96, 16:80], c_ub)
            for i, j in T.Parallel(64, 64):
                c_ub[i, j] = c_ub[i, j] + 1.0
        T.copy(c_ub, C)


@T.prim_func
def _reduce_dim0(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128,), "float32")):
    with T.Kernel(1):
        a_ub = T.alloc_shared((128, 128), "float32")
        m_ub = T.alloc_shared((128,), "float32")
        out_ub = T.alloc_shared((128,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            T.reduce_sum(a_ub, m_ub, dim=0, clear=False)
            for i in T.Parallel(128):
                out_ub[i] = m_ub[i]
        T.copy(out_ub, C)


@T.prim_func
def _fragment(A: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
    with T.Kernel(1):
        a_ub = T.alloc_shared((64,), "float32")
        c_ub = T.alloc_shared((64,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            frag = T.alloc_fragment((64,), "float32")
            for i in T.Parallel(64):
                frag[i] = a_ub[i] + 1.0
            for i in T.Parallel(64):
                c_ub[i] = frag[i] * 2.0
        T.copy(c_ub, C)


@T.prim_func
def _outside_fragment(A: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
    with T.Kernel(1):
        a_ub = T.alloc_shared((64,), "float32")
        c_ub = T.alloc_shared((64,), "float32")
        frag = T.alloc_fragment((64,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            for i in T.Parallel(64):
                frag[i] = a_ub[i] + 1.0
            for i in T.Parallel(64):
                c_ub[i] = frag[i] * 2.0
        T.copy(c_ub, C)


def test_simple_group_is_planned_and_emitted():
    mod = _lift(_simple)
    programs = _annotations(mod, PROGRAM_KEY)
    assert len(programs) == 1
    assert programs[0]["items"][0]["status"] == "LIFTED"

    mod = _plan(mod)
    plans = _annotations(mod, GROUP_PLAN_KEY)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.vf_index == 0
    assert plan.fragment_mode == "none"
    assert len(plan.groups) == 1
    group = plan.groups[0]
    assert group.status == "PLANNED"
    assert len(group.items) == 1
    item = group.items[0]
    assert (item.kind, item.plan_kind) == ("REGION", "REGION")
    assert item.region_plan is not None
    assert item.copy_plan is None and item.reduce_plan is None

    mod = _emit(mod)
    assert _annotations(mod, PROGRAM_KEY) == []
    assert _annotations(mod, GROUP_PLAN_KEY) == []
    body = str(mod["main"].body)
    assert "tl.simd.vld" in body
    assert "tl.simd.vsts" in body


def test_partial_lift_plans_only_the_lifted_group():
    mod = _emit(_plan(_lift(_partial)))
    body = str(mod["main"].body)
    # The independent legal region is vectorized.
    assert "tl.simd.vadds" in body
    # The illegal region stays a T.Parallel for the legacy fallback.
    assert "T.parallel" in body
    assert _annotations(mod, PROGRAM_KEY) == []
    assert _annotations(mod, GROUP_PLAN_KEY) == []


def test_copy_reduce_group_is_planned_and_emitted():
    """Phase B: the top-level COPY and REDUCE are planned as synthesized nests."""
    mod = _plan(_lift(_copy_reduce))
    plans = _annotations(mod, GROUP_PLAN_KEY)
    assert len(plans) == 1
    group = plans[0].groups[0]
    assert group.status == "PLANNED", group.reason
    assert [item.plan_kind for item in group.items] == ["COPY", "REDUCE", "REGION"]
    copy_item, reduce_item, _ = group.items
    assert copy_item.copy_plan is not None
    assert copy_item.copy_plan.kind == "ND"
    assert (copy_item.copy_plan.src_scope, copy_item.copy_plan.dst_scope) == ("SHARED", "SHARED")
    assert reduce_item.reduce_plan is not None
    assert (reduce_item.reduce_plan.reduce_kind, reduce_item.reduce_plan.dim) == ("max", 1)
    assert reduce_item.reduce_plan.clear is False
    assert reduce_item.reduce_plan.init is None

    mod = _emit(mod)
    body = str(mod["main"].body)
    # No partial lift: the whole group is replaced.
    assert "T.parallel" not in body
    assert "tl.tileop.reduce" not in body
    assert "tl.simd.vld" in body and "tl.simd.vsts" in body
    assert "tl.simd.vmax" in body
    assert _annotations(mod, PROGRAM_KEY) == []
    assert _annotations(mod, GROUP_PLAN_KEY) == []


def test_reduce_clear_true_emits_an_identity_init():
    mod = _plan(_lift(_reduce_clear))
    plans = _annotations(mod, GROUP_PLAN_KEY)
    group = plans[0].groups[0]
    assert group.status == "PLANNED", group.reason
    reduce_item = group.items[0]
    assert reduce_item.reduce_plan.clear is True
    assert reduce_item.reduce_plan.init is not None

    mod = _emit(mod)
    body = str(mod["main"].body)
    assert "T.parallel" not in body
    assert "tl.simd.vld" in body and "tl.simd.vsts" in body


def test_partial_region_copy_uses_the_region_indices():
    """Phase B copy honours the tile-region origin, not just the extent."""
    mod = _plan(_lift(_partial_copy))
    group = _annotations(mod, GROUP_PLAN_KEY)[0].groups[0]
    assert group.status == "PLANNED", group.reason
    copy_item = group.items[0]
    assert copy_item.plan_kind == "COPY"
    assert copy_item.copy_plan.kind == "ND"
    assert [str(extent) for extent in copy_item.copy_plan.dst_extents] == ["64", "64"]

    mod = _emit(mod)
    body = str(mod["main"].body)
    assert "T.parallel" not in body
    assert "tl.simd.vld" in body and "tl.simd.vsts" in body


def test_reduce_along_dim0_is_planned():
    mod = _plan(_lift(_reduce_dim0))
    group = _annotations(mod, GROUP_PLAN_KEY)[0].groups[0]
    assert group.status == "PLANNED", group.reason
    reduce_item = group.items[0]
    assert reduce_item.reduce_plan.dim == 0
    assert reduce_item.reduce_plan.reduce_kind == "sum"

    mod = _emit(mod)
    body = str(mod["main"].body)
    assert "T.parallel" not in body
    assert "tl.simd.vsts" in body


def test_in_vf_fragment_uses_the_register_file():
    mod = _plan(_lift(_fragment))
    plans = _annotations(mod, GROUP_PLAN_KEY)
    plan = plans[0]
    assert plan.fragment_mode == "register"
    assert plan.fragment_regs is not None
    assert [(entry.name, entry.slots, entry.lanes) for entry in plan.fragment_regs.entries] == [("frag", 1, 64)]
    assert all(group.status == "PLANNED" for group in plan.groups)

    mod = _emit(mod)
    body = str(mod["main"].body)
    assert "local.var" in body
    assert "local.fragment" not in body
    assert _annotations(mod, PROGRAM_KEY) == []
    assert _annotations(mod, GROUP_PLAN_KEY) == []


def test_fragment_allocated_outside_the_vf_is_rejected():
    """Cross-VF fragment: P1 records OUTSIDE_VF, P2 rejects the group directly."""
    mod = _lift(_outside_fragment)
    program = _annotations(mod, PROGRAM_KEY)[0]
    assert any(fragment["definition_scope"] == "OUTSIDE_VF" for fragment in program["fragments"])

    mod = _plan(mod)
    plans = _annotations(mod, GROUP_PLAN_KEY)
    assert len(plans) == 1
    group = plans[0].groups[0]
    assert group.status == "REJECTED"
    assert "allocated outside this VF" in group.reason

    original = _outside_fragment
    mod = _emit(mod)
    # Nothing was replaced: the whole VF stays for the legacy lowering.
    assert _simd_ops(mod) == []
    tvm.ir.assert_structural_equal(mod["main"].body, original.body, map_free_vars=True)
    assert _annotations(mod, PROGRAM_KEY) == []
    assert _annotations(mod, GROUP_PLAN_KEY) == []


def test_one_rejected_region_rejects_the_whole_group():
    """A P2-rejected REGION must not leave its producer vectorized."""

    @T.prim_func
    def k(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32"), D: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((128,), "float32")
            c = T.alloc_shared((128,), "float32")
            d = T.alloc_shared((128,), "float32")
            T.copy(A, a)
            with T.SimdVF():
                for i in T.Parallel(128):
                    c[i] = a[i] + 1.0
                for i in T.Parallel(128):
                    d[i] = c[i // 3]
            T.copy(c, C)
            T.copy(d, D)

    mod = _plan(_lift(k))
    plans = _annotations(mod, GROUP_PLAN_KEY)
    assert plans and len(plans[0].groups) == 1
    group = plans[0].groups[0]
    assert group.status == "REJECTED"
    assert "not a power of two" in " ".join(group.diagnostics)
    mod = _emit(mod)
    assert _simd_ops(mod) == []


def test_body_change_rejects_the_whole_vf():
    mod = _lift(_simple)
    # Simulate a body-mutating pass between P1 and P2: the legacy lowering
    # replaces the T.Parallel body, so the P1 fingerprint no longer matches.
    mod = at.AscendSimdVFLowerParallel()(mod)
    mod = _plan(mod)
    assert _annotations(mod, GROUP_PLAN_KEY) == []
    mod = _emit(mod)
    assert _annotations(mod, PROGRAM_KEY) == []


def test_pipeline_uses_the_new_group_path_by_default():
    with tvm.target.Target("ascend"), tilelang.transform.PassContext():
        out = tilelang.lower(_simple)
    assert "simd_inst::" in out.kernel_source
    assert "tl.vsimd.program" not in out.kernel_source
    assert "tl.vsimd.group_plan" not in out.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
