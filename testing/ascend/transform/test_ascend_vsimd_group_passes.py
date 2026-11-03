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
    print(f"Found {len(found)} {key} annotations,all: {found}")
    return found


def _lift(func):
    print(func)
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


def test_simple_group_is_planned_and_emitted():
    mod = _lift(_simple)
    programs = _annotations(mod, PROGRAM_KEY)
    assert len(programs) == 1
    assert programs[0]["items"][0]["status"] == "LIFTED"

    mod = _plan(mod)
    plans = _annotations(mod, GROUP_PLAN_KEY)
    assert len(plans) == 1
    group = plans[0]["groups"][0]
    assert group["status"] == "PLANNED"
    assert len(group["items"]) == 1

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


def test_copy_reduce_group_is_rejected_as_a_whole():
    mod = _plan(_lift(_copy_reduce))
    plans = _annotations(mod, GROUP_PLAN_KEY)
    assert len(plans) == 1
    assert all(group["status"] == "REJECTED" for group in plans[0]["groups"])

    mod = _emit(mod)
    body = str(mod["main"].body)
    # No partial lift: the copy, reduce and region stay together.
    assert "tl.simd." not in body
    assert _annotations(mod, PROGRAM_KEY) == []
    assert _annotations(mod, GROUP_PLAN_KEY) == []


def test_body_change_rejects_the_whole_vf():
    mod = _lift(_simple)
    # Simulate a body-mutating pass between P1 and P2: the legacy lowering
    # replaces the T.Parallel body, so the P1 fingerprint no longer matches.
    mod = at.AscendSimdVFLowerParallel()(mod)
    mod = _plan(mod)
    assert _annotations(mod, GROUP_PLAN_KEY) == []
    mod = _emit(mod)
    assert _annotations(mod, PROGRAM_KEY) == []


def test_pipeline_uses_the_new_group_path():
    with tvm.transform.PassContext(
        config={tilelang.PassConfigKey.TL_ASCEND_VSIMD_VECTORIZE: True}
    ):
        with tvm.target.Target("ascend"):
            out = tilelang.lower(_simple)
    assert "simd_inst::" in out.kernel_source
    assert "tl.vsimd.program" not in out.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
