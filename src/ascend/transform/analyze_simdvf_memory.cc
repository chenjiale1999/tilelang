/*!
 * \file analyze_simdvf_memory.cc
 * \brief Debug pass / FFI entry for the read-only P1 memory-legality analysis.
 */
#include "vsimd_memory_legality.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

ffi::String AscendAnalyzeSimdVFMemoryReport(const PrimFunc &func) {
  return AnalyzeVSimdMemoryLegality(func);
}

tvm::transform::Pass AscendAnalyzeSimdVFMemory() {
  auto pass_func = [](PrimFunc f, const IRModule &mod, PassContext ctx) {
    LOG(INFO) << "\n" << AnalyzeVSimdMemoryLegality(f);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendAnalyzeSimdVFMemory", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendAnalyzeSimdVFMemoryReport",
                        AscendAnalyzeSimdVFMemoryReport);
  refl::GlobalDef().def("tl.transform.AscendAnalyzeSimdVFMemory",
                        AscendAnalyzeSimdVFMemory);
}

} // namespace tl
} // namespace tvm
