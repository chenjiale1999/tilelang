/*!
 * \file emit.h
 * \brief Emission layer: ``VectorPlan`` / ``VectorProgram`` -> ``T.simd.*`` TIR.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_EMIT_H_
#define TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_EMIT_H_

#include <tvm/tirx/stmt.h>

#include "model.h"

namespace tvm {
namespace tl {
namespace polyhedral {

/*!
 * \brief Emit a plan (or program) as ``T.simd.*`` TIR.
 *
 * The plan is read-only; every node / ordering rule matches the Python emitter
 * node for node (SSA names included).
 */
TVM_DLL tirx::Stmt EmitPlan(const ffi::ObjectRef& plan);

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_EMIT_H_
