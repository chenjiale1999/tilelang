"""C++-only facade for the polyhedral P2/P3 vectorizer.

Every symbol here is backed by the compiled C++ implementation
(``src/ascend/transform/polyhedral``) through the TVM FFI; nothing in this
package plans, emits, or renders in Python. The original Python planner/emitter
is kept as an oracle in :mod:`tilelang.ascend.analysis.polyhedral_reference` and
is never imported by this module or by the tests that use it.

Data flow (all C++):

============================================ ===============================
``build_plan(stmt)``                         P2: ``T.Parallel`` nest -> ``VectorProgram``
``emit_plan(plan)``                          P3: plan -> ``T.simd.*`` TIR
``render_plan(plan)``                        plan -> readable text (diff oracle)
``vectorize_parallel_nests(stmt)``           pre-order replacement driver
``finite_difference`` / ``classify_delta``   the mathematical core
``Axis`` / ``AccessInfo`` / ``StoreSpec`` /  FFI plan objects (fields match the
``ValueNode`` / ``StorePlan`` /              C++ ``def_ro`` registration)
``ReduceInfo`` / ``VectorPlan`` /
``VectorProgram``
============================================ ===============================

The pipeline pass lives in ``tilelang.ascend.transform``
(``AscendLiftParallelToArray`` -> ``AscendPlanArray`` -> ``AscendLowerArrayPlan``,
composed by ``AscendPolyhedralVectorize``).
"""

from __future__ import annotations

from ._contract import AccessClass, Unsupported
from ._cpp import (
    AccessInfo,
    Axis,
    ReduceInfo,
    StorePlan,
    StoreSpec,
    ValueNode,
    VectorPlan,
    VectorProgram,
    build_plan,
    classify_delta,
    emit_plan,
    finite_difference,
    render_plan,
    vectorize_parallel_nests,
)

__all__ = [
    "Axis",
    "AccessClass",
    "AccessInfo",
    "StoreSpec",
    "ValueNode",
    "StorePlan",
    "ReduceInfo",
    "VectorPlan",
    "VectorProgram",
    "Unsupported",
    "finite_difference",
    "classify_delta",
    "build_plan",
    "emit_plan",
    "render_plan",
    "vectorize_parallel_nests",
]
