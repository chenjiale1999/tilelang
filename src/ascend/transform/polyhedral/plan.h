/*!
 * \file plan.h
 * \brief Planning layer: nest splitting, axis ranking, reductions, build_plan.
 *
 * C++ counterpart of ``polyhedral/plan.py``. ``BuildPlan`` always returns a
 * :class:`VectorProgram` with 1..N segments; the plan holds no instruction
 * names at all (instruction selection is the emit layer's job).
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_PLAN_H_
#define TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_PLAN_H_

#include <optional>

#include "model.h"

namespace tvm {
namespace tl {
namespace polyhedral {

/*!
 * \brief ``T.Parallel`` nest -> :class:`VectorProgram` (1..N segments).
 *
 * ``lanes`` defaults to the vector register width derived from the store
 * element dtype (``2048 / elem_bits``), so callers normally leave it empty.
 */
TVM_DLL VectorProgram BuildPlan(const tirx::Stmt& stmt,
                                std::optional<int64_t> lanes = std::nullopt);

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_PLAN_H_
