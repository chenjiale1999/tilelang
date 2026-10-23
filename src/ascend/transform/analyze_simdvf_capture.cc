/*!
 * \file analyze_simdvf_capture.cc
 * \brief Debug pass / FFI entry for the read-only PR-B capture/scope analysis.
 */
#include "vsimd_capture_scope.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

ffi::String AscendAnalyzeSimdVFCaptureReport(const PrimFunc &func) {
  return AnalyzeVSimdCaptureScope(func);
}

ffi::Array<ffi::Map<ffi::String, ffi::Any>>
AscendAnalyzeSimdVFCaptureInfo(const PrimFunc &func) {
  return CaptureInfoToFFI(AnalyzeVSimdCaptureInfo(func));
}

tvm::transform::Pass AscendAnalyzeSimdVFCapture() {
  auto pass_func = [](PrimFunc f, const IRModule &mod, PassContext ctx) {
    LOG(INFO) << "\n" << AnalyzeVSimdCaptureScope(f);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendAnalyzeSimdVFCapture", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendAnalyzeSimdVFCaptureReport",
                        AscendAnalyzeSimdVFCaptureReport);
  refl::GlobalDef().def("tl.transform.AscendAnalyzeSimdVFCaptureInfo",
                        AscendAnalyzeSimdVFCaptureInfo);
  refl::GlobalDef().def("tl.transform.AscendAnalyzeSimdVFCapture",
                        AscendAnalyzeSimdVFCapture);
}

} // namespace tl
} // namespace tvm
