"""P1 delivery and P1→P2→P3 pass-pipeline contract tests.

Offline: checks the annotation hand-off between ``AscendLiftVSimd`` →
``AscendPlanVSimd`` → ``AscendEmitVSimd``, idempotence, rejected nests staying
byte-identical, and the ``AscendVectorizeVSimd`` pipeline switch.
"""

from __future__ import annotations


import pytest

import tilelang
import tilelang.ascend.language as T
import tvm

from tilelang.ascend import transform as at
from tilelang.ascend.analysis.parallel_vectorize import (
    build_plan,
    emit_plan,
    render_plan,
)
from testing.ascend.transform._parallel_vectorize_helpers import (
    _count_parallel_nests,
    _emitted_ops,
    _lower_source,
    _outer_parallel_nests,
    _p1_items,
    _p1_kernel_1d,
    _p1_kernel_extent,
    _region_annotations,
    _region_plans,
    _run_lift,
    _run_new_path,
    _run_plan,
    _single_segment,
)


# ===========================================================================
# P1 交付、三段 pass 与流水线接入
# ===========================================================================
#
# 这一组钉的是“nest 之外的接口”。P1 的 item/group/capture 全量字段由
# ``test_ascend_vsimd_{lift,region,capture_scope,memory_legality}.py`` 覆盖，
# P2/P3 的规划与发射语义由上面各节的场景用例覆盖（多 store、三元/if、
# 归约、broadcast、transpose/flatten、混合宽度、NZ 等）。这里不再重复这些
# 断言，只覆盖三件事：
#
#   * 交付：LIFTED 的 nest 可直接喂给 ``build_plan``；被拒 nest 记录为
#     OPAQUE/带 reason（P1 侧只留一条代表性用例）；
#   * 三段：P1/P2/P3 通过 ``tl.vsimd.region`` / ``tl.vsimd.plan``
#     annotation（都挂在 Parallel 上）交接，幂等，被拒 nest 逐字节不变；
#   * 流水线：``AscendVectorizeVSimd`` 是新路径的唯一入口，
#     默认 PassContext 下就运行。


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 交付契约：LIFTED 的 stmt 可直接规划；被拒 nest 只带状态
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extent, chunks, tail", [(100, 2, 36), (1, 1, 1)], ids=["tail", "unit"])
def test_plan_tail_and_unit_extent(extent, chunks, tail):
    func = _p1_kernel_extent(extent)
    segment = build_plan(_outer_parallel_nests(func)[0]).segments[0]
    assert (segment.chunks, segment.tail) == (chunks, tail)


def test_plan_symbolic_extent():
    """符号 extent：运行时 chunk 循环 + 符号尾块，plan 不折叠成常量。"""
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

    segment = build_plan(_outer_parallel_nests(main)[0]).segments[0]
    assert segment.chunks is None and segment.tail is None


def test_p1_reduction_is_lifted_and_planned():
    """归约写：P1 不再按"非单射"拒绝，P2 规划成跨 lane ``vcadd``。

    ``c_ub[i] += a_ub[i, j]`` 的下标确实非单射，但它是**累积写**（读写同一地址），
    属于归约而不是冲突写入，P1 必须放行。``n == lanes`` 时走短归约形态：
    一次 ``vld`` + ``vcadd`` + ``vadds`` + 只保留 lane 0 的 ``vsts``。
    """
    m, n = 8, 64

    @T.prim_func
    def main(A: T.Tensor((m, n), "float32"), C: T.Tensor((m,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((m, n), "float32")
            c_ub = T.alloc_shared((m,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(m, n):
                    c_ub[i] += a_ub[i, j]
            T.copy(c_ub, C)

    assert _p1_items(main)[0].status == "LIFTED"
    segment = _single_segment(main)
    assert segment.vector_axis.name == "j"
    stmt = emit_plan(segment)
    ops = _emitted_ops(stmt)
    assert ops.count("tl.simd.vcadd") == 1
    assert ops.count("tl.simd.vadds") == 1
    assert ops.count("tl.simd.vsts") == 1
    # 归约结果只占 lane 0，回写必须用只保留 lane 0 的谓词。
    assert 'pset(32, "PAT_VL1")' in str(stmt)


# ---------------------------------------------------------------------------
# 三段 pass：annotation 交接、幂等、被拒 nest 不变
# ---------------------------------------------------------------------------


def test_staged_passes_exchange_annotations():
    func = _p1_kernel_1d()

    lifted = _run_lift(func)
    regions = _region_annotations(lifted.body)
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"
    assert _count_parallel_nests(lifted.body) == 1, "P1 must not rewrite the nest"
    assert _emitted_ops(lifted.body) == []

    planned = _run_plan(lifted)
    planned_regions = _region_annotations(planned.body)
    assert len(planned_regions) == 1
    assert planned_regions[0].status == "LIFTED", "P2 must keep the P1 annotation"
    plans = _region_plans(planned.body)
    assert len(plans) == 1
    plan = plans[0]
    assert int(plan.version) == 1
    assert (int(plan.vf_index), int(plan.region_index)) == (0, 0)
    assert len(plan.program.segments) == 1
    assert _count_parallel_nests(planned.body) == 1
    # P2 的 plan 与直接规划交接 stmt 的 plan 逐字段一致。
    assert render_plan(plan.program) == render_plan(
        build_plan(_outer_parallel_nests(func)[0])
    )

    emitted = _run_new_path(func)
    assert _region_plans(emitted.body) == []
    assert _region_annotations(emitted.body) == []
    assert _count_parallel_nests(emitted.body) == 0
    assert "tl.simd.vsts" in _emitted_ops(emitted.body)
    tvm.ir.assert_structural_equal(emitted.body, _run_new_path(func).body, map_free_vars=True)


def test_passes_are_idempotent():
    once = _run_lift(_p1_kernel_1d())
    twice = _run_lift(once)
    assert len(_region_annotations(twice.body)) == 1

    combined = _run_new_path(_p1_kernel_1d())
    again = _run_new_path(combined)
    tvm.ir.assert_structural_equal(again.body, combined.body, map_free_vars=True)


def test_rejected_nests_fall_through_every_stage():
    # P1 拒绝（非法内存）⇒ 记录为 OPAQUE（不可 lift），IR 逐字节不变；新链路也不动它。
    n = 256

    @T.prim_func
    def main(A: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n - 32):
                    a_ub[i] = a_ub[i + 32]

    illegal = main
    lifted = _run_lift(illegal)
    # P1 still records the nest, but as an OPAQUE region (memory legality is
    # ILLEGAL): nothing is liftable, so the body is byte-identical.
    regions = _region_annotations(lifted.body)
    assert regions and all(region.status != "LIFTED" for region in regions)
    # P2 writes no plan (nothing is liftable) and P3 restores the original body.
    assert _region_plans(_run_plan(lifted).body) == []
    tvm.ir.assert_structural_equal(_run_new_path(illegal).body, illegal.body, map_free_vars=True)


# ---------------------------------------------------------------------------
# 流水线接入：P1/P2/P3 同一开关
# ---------------------------------------------------------------------------


@pytest.fixture
def pass_calls(monkeypatch) -> list:
    """Record how often the pipeline constructs the pass."""
    calls: list = []
    real = at.AscendVectorizeVSimd

    def spy():
        calls.append(True)
        return real()

    monkeypatch.setattr(at, "AscendVectorizeVSimd", spy)
    return calls


def test_disabling_the_lift_skips_the_whole_chain(pass_calls):
    _lower_source(
        _p1_kernel_1d(),
        config={tilelang.PassConfigKey.TL_ASCEND_SIMD_LIFT: False},
    )

    assert pass_calls == [], "P2/P3 must not run without the P1 annotation"
