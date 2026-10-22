/*!
 * \file vsimd_memory_legality.h
 * \brief Read-only memory-legality analysis for SIMD_VF bodies (P1).
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_VSIMD_MEMORY_LEGALITY_H_
#define TVM_TL_ASCEND_TRANSFORM_VSIMD_MEMORY_LEGALITY_H_

#include <tvm/ffi/string.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

/*!
 * \brief Analyze the memory legality of every SIMD_VF block in \p func.
 *
 * The analysis is intentionally read-only: it does not rewrite the function.
 * It reports, per SIMD_VF and per T.Parallel nest, whether the stores are
 * injective and whether cross-iteration dependences exist. Manual SIMD VFs
 * (tl.simd.*, local.var, nd2nz helpers) are reported as opaque.
 */
TVM_DLL ffi::String AnalyzeVSimdMemoryLegality(const tirx::PrimFunc &func);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VSIMD_MEMORY_LEGALITY_H_
