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

/*!
 * \brief ``simd_inst::mem_bar(VST_VLD)``.
 *
 * Closes a vector store before the next load. Emitted at the end of every
 * scalar-shell iteration (``plan.shells``) and by P3 when the planned Parallel
 * sits inside a serial loop, so a read-modify-write body never reads a value
 * from before its own store.
 */
TVM_DLL tirx::Stmt VstVldBarrier();

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_EMIT_H_
