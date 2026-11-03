/*!
 * \file vsimd_group_passes.h
 * \brief P2/P3 passes that consume the P1 ``tl.vsimd.program`` annotation.
 *
 * P2 plans every ``P1_LIFTABLE`` group atomically into
 * ``tl.vsimd.group_plan``; P3 emits the planned items and replaces their
 * canonical spans, leaving rejected groups to the legacy lowering.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_VSIMD_GROUP_PASSES_H_
#define TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_VSIMD_GROUP_PASSES_H_

#include <tvm/tirx/function.h>

namespace tvm {
namespace tl {
namespace polyhedral {

/*! \brief P2: plan P1 groups; body is unchanged. */
TVM_DLL tirx::PrimFunc PlanVSimd(tirx::PrimFunc func);

/*! \brief P3: emit planned groups and replace their item spans. */
TVM_DLL tirx::PrimFunc EmitVSimd(tirx::PrimFunc func);

/*! \brief P2 + P3 (combination entry point). */
TVM_DLL tirx::PrimFunc VectorizeVSimd(tirx::PrimFunc func);

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_VSIMD_GROUP_PASSES_H_
