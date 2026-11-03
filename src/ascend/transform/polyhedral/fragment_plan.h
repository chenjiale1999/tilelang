/*!
 * \file fragment_plan.h
 * \brief VF-level fragment register planning (V1-V4) shared by P2/P3.
 *
 * Fragment register mode is VF-atomic: either every planned nest of the
 * ``SIMD_VF`` block addresses its fragments as V-core registers, or none does.
 * P2 runs the validation over the planned programs and publishes the resulting
 * :class:`FragmentRegisterMap` in the group plan; P3 builds the register file
 * from it and rewrites the block's allocations.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_FRAGMENT_PLAN_H_
#define TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_FRAGMENT_PLAN_H_

#include <tvm/tirx/stmt.h>

#include <string>
#include <vector>

#include "model.h"

namespace tvm {
namespace tl {
namespace polyhedral {

/*! \brief Outcome of the VF-level fragment register validation. */
struct VfFragmentPlan {
  /*! \brief True when the VF's fragments live in registers. */
  bool enabled{false};
  /*! \brief Why register mode was disabled (empty on success). */
  std::string reason;
  /*! \brief The VF register map; only defined when ``enabled``. */
  FragmentRegisterMap map;
};

/*!
 * \brief Decide whether the VF's fragments can use the register file.
 *
 * \param body           The ``SIMD_VF`` body (unchanged, P1 item order).
 * \param programs       Plans of the groups P2 accepted, in source order.
 * \param planned_spans  Canonical item spans those plans replace. A fragment
 *                       access outside them escapes its register plan (V1).
 *
 * Returns ``enabled == false`` (and a reason) on any violation; the caller must
 * then fall back atomically for the whole VF.
 */
TVM_DLL VfFragmentPlan PlanVfFragmentRegisters(
    const tirx::Stmt& body, const std::vector<VectorProgram>& programs,
    const std::vector<int>& planned_spans);

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_FRAGMENT_PLAN_H_
