/*!
 * \file render.h
 * \brief Human-readable plan rendering (diagnostics / test oracle).
 *
 * A 1-segment program renders byte-identically to its bare segment plan (the
 * unified representation does not change the readable output).
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_RENDER_H_
#define TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_RENDER_H_

#include <string>

#include "model.h"

namespace tvm {
namespace tl {
namespace polyhedral {

/*! \brief Render a :class:`VectorPlan` or :class:`VectorProgram` to text. */
TVM_DLL std::string RenderPlan(const ffi::ObjectRef& plan);

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_RENDER_H_
