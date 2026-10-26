/*!
 * \file analyze_vsimd_region.cc
 * \brief Debug pass / FFI entry for the PR-C minimal region lift.
 */
#include "vsimd_region.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

ffi::String AscendAnalyzeVSimdRegionReport(const PrimFunc &func) {
  return AnalyzeVSimdRegionReport(func);
}

tvm::transform::Pass AscendAnalyzeVSimdRegion() {
  auto pass_func = [](PrimFunc f, const IRModule &mod, PassContext ctx) {
    LOG(INFO) << "\n" << AnalyzeVSimdRegionReport(f);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendAnalyzeVSimdRegion", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendAnalyzeVSimdRegionReport",
                        AscendAnalyzeVSimdRegionReport);
  refl::GlobalDef().def("tl.transform.AscendAnalyzeVSimdRegion",
                        AscendAnalyzeVSimdRegion);
}

} // namespace tl
} // namespace tvm
