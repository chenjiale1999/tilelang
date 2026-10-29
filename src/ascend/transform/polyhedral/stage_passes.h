/*!
 * \file stage_passes.h
 * \brief Staged P1/P2/P3 lowering of ``T.Parallel``.
 *
 * The three stages are separate passes over the same PrimFunc and communicate
 * through two AttrStmt markers, so every stage consumes exactly what the
 * previous stage produced:
 *
 *   P1  LiftParallelToArray  read-only P1 analysis; keeps only nests whose
 *                            VSimdRegionLift.lifted is true and wraps each of
 *                            them in ``AttrStmt(VSimdRegion, "tl.vsimd.array_region")``.
 *                            The nest itself is not rewritten.
 *   P2  PlanArray            consumes the carried **region** (never the nest
 *                            body), plans it and rewrites the marker to
 *                            ``AttrStmt(VectorProgram, "tl.vsimd.array_plan")``.
 *                            An unplannable nest is unwrapped untouched.
 *   P3  LowerArrayPlan       consumes ``tl.vsimd.array_plan`` markers and
 *                            replaces them with the emitted ``T.simd.*`` TIR.
 *
 * Any stage that cannot handle a nest leaves the original IR unchanged: the
 * carrier is unwrapped back to the exact nest statement, which the legacy
 * ``AscendSimdVFLowerParallel`` then lowers as before.
 *
 * ``VectorizeParallelNests`` runs the three stages in order; it is the pass the
 * Ascend pipeline inserts (``tl.ascend.polyhedral.impl = "cpp"``).
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_STAGE_PASSES_H_
#define TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_STAGE_PASSES_H_

#include <tvm/tirx/function.h>

namespace tvm {
namespace tl {
namespace polyhedral {

/*! \brief P1 -> P2 carrier attribute (value = the typed :class:`VSimdRegion`). */
inline constexpr const char* kArrayRegionAttr = "tl.vsimd.array_region";
/*! \brief P2 -> P3 carrier attribute (value = the :class:`VectorProgram`). */
inline constexpr const char* kArrayPlanAttr = "tl.vsimd.array_plan";

/*!
 * \brief P1: run the vsimd analysis and mark every lifted nest.
 *
 * The function body is otherwise unchanged; a nest P1 does not lift is left
 * exactly as it was, so ``AscendSimdVFLowerParallel`` still handles it.
 */
TVM_DLL tirx::PrimFunc LiftParallelToArray(tirx::PrimFunc func);

/*!
 * \brief P2: plan every ``tl.vsimd.array_region`` nest.
 *
 * Rejected nests are unwrapped (never left marked), so a nest P1 lifted but P2
 * cannot plan falls back to the legacy lowering.
 */
TVM_DLL tirx::PrimFunc PlanArray(tirx::PrimFunc func);

/*!
 * \brief P3: replace every ``tl.vsimd.array_plan`` marker with its TIR.
 *
 * Leftover region markers (P2 skipped) are unwrapped so P1 plumbing never
 * reaches codegen.
 */
TVM_DLL tirx::PrimFunc LowerArrayPlan(tirx::PrimFunc func);

/*! \brief P1 -> P2 -> P3 in one pass (pipeline entry point). */
TVM_DLL tirx::PrimFunc VectorizeParallelNests(tirx::PrimFunc func);

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_STAGE_PASSES_H_
