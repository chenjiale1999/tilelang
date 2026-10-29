/*!
 * \file ffi_polyhedral.cc
 * \brief FFI entry points and the driver pass for the C++ polyhedral P2/P3 port.
 *
 * Every entry point converts :class:`polyhedral::Unsupported` into the
 * ``Map{"ok": false, "message": ..., "diagnostics": [...]}`` shape so the Python
 * wrapper can re-raise the *identical* ``Unsupported`` (message + diagnostics
 * ordering are part of the contract, D3).
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <string>
#include <vector>

#include "access.h"
#include "emit.h"
#include "plan.h"
#include "primitives.h"
#include "render.h"
#include "stage_passes.h"

namespace tvm {
namespace tl {
namespace polyhedral {

// Pipeline switch: ``legacy`` (default) keeps the historical behavior, ``cpp``
// enables this vectorizer (the only implementation behind the pass).
TVM_REGISTER_PASS_CONFIG_OPTION("tl.ascend.polyhedral.impl", ffi::String);

namespace {

ffi::Map<ffi::String, ffi::Any> MakeError(const Unsupported& error) {
  ffi::Map<ffi::String, ffi::Any> result;
  result.Set("ok", ffi::Any(false));
  result.Set("message", ffi::Any(ffi::String(error.message())));
  ffi::Array<ffi::String> diagnostics;
  for (const std::string& note : error.diagnostics()) {
    diagnostics.push_back(note);
  }
  result.Set("diagnostics", ffi::Any(diagnostics));
  return result;
}

/*! \brief Run ``body`` and bridge ``Unsupported`` into the Map protocol. */
template <typename Fn>
ffi::Map<ffi::String, ffi::Any> Guarded(Fn body) {
  try {
    return body();
  } catch (const Unsupported& error) {
    return MakeError(error);
  }
}

/*! \brief ``{ok: true, key: value}`` — the success shape of every entry point. */
ffi::Map<ffi::String, ffi::Any> MakeOk(ffi::String key, ffi::Any value) {
  ffi::Map<ffi::String, ffi::Any> result;
  result.Set("ok", ffi::Any(true));
  result.Set(key, std::move(value));
  return result;
}

/*! \brief ``AxisClass`` -> ``[kind, stride?]`` (the shape Python reads). */
ffi::Array<ffi::Any> AxisClassToArray(const AxisClass& classified) {
  ffi::Array<ffi::Any> out;
  out.push_back(ffi::Any(ffi::String(AccessClassName(classified.kind))));
  out.push_back(classified.stride.has_value() ? ffi::Any(*classified.stride)
                                              : ffi::Any());
  return out;
}

/*! \brief FFI optional lanes -> ``std::optional<int64_t>`` (empty = derive). */
std::optional<int64_t> OptionalLanes(const ffi::Optional<int64_t>& lanes) {
  return lanes.has_value() ? std::optional<int64_t>(*lanes) : std::nullopt;
}

/*!
 * \brief Wrap one ``PrimFunc -> PrimFunc`` stage as a named PrimFunc pass.
 *
 * All four registered passes have the same shape (module/context unused), so the
 * boilerplate lives here instead of being repeated per stage.
 */
tvm::transform::Pass MakeStagePass(const char* name,
                                   tirx::PrimFunc (*stage)(tirx::PrimFunc)) {
  auto pass_func = [stage](tirx::PrimFunc func, const IRModule&,
                           const tvm::transform::PassContext&) {
    return stage(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0, name, {});
}

/*!
 * \brief Pre-order replacement of every ``T.Parallel`` nest (``driver.py``).
 *
 * Planning a whole nest replaces *the entire consecutive Parallel chain*; if it
 * fails we keep descending so inner nests still get a chance. Bottom-up would
 * leave the outer ``T.Parallel`` behind, hence pre-order.
 */
class NestVectorizer : public tirx::StmtMutator {
 public:
  explicit NestVectorizer(std::optional<int64_t> lanes) : lanes_(lanes) {}

  tirx::Stmt VisitStmt_(const tirx::ForNode* op) final {
    if (op->kind == tirx::ForKind::kParallel) {
      try {
        VectorProgram program = BuildPlan(ffi::GetRef<tirx::Stmt>(op), lanes_);
        return EmitPlan(program);
      } catch (const Unsupported&) {
        // Keep the nest untouched; legacy lowering takes over.
      }
    }
    return tirx::StmtMutator::VisitStmt_(op);
  }

 private:
  std::optional<int64_t> lanes_;
};

}  // namespace

ffi::Array<ffi::Any> AscendPolyhedralAxisClass(const AccessInfo& info, const Axis& axis) {
  return AxisClassToArray(AxisClassOf(info, axis));
}

ffi::Map<ffi::String, ffi::Any> AscendPolyhedralBuildPlan(
    const tirx::Stmt& stmt, ffi::Optional<int64_t> lanes) {
  return Guarded([&]() {
    VectorProgram program = BuildPlan(stmt, OptionalLanes(lanes));
    return MakeOk("program", ffi::Any(program));
  });
}

ffi::Map<ffi::String, ffi::Any> AscendPolyhedralEmitPlan(const ffi::ObjectRef& plan) {
  return Guarded([&]() { return MakeOk("stmt", ffi::Any(EmitPlan(plan))); });
}

ffi::Map<ffi::String, ffi::Any> AscendPolyhedralRenderPlan(const ffi::ObjectRef& plan) {
  return Guarded(
      [&]() { return MakeOk("text", ffi::Any(ffi::String(RenderPlan(plan)))); });
}

ffi::Map<ffi::String, ffi::Any> AscendPolyhedralVectorizeNests(
    const tirx::Stmt& stmt, ffi::Optional<int64_t> lanes) {
  return Guarded([&]() {
    NestVectorizer vectorizer(OptionalLanes(lanes));
    return MakeOk("stmt", ffi::Any(vectorizer(stmt)));
  });
}

PrimExpr AscendPolyhedralFiniteDifference(const PrimExpr& offset, const tirx::Var& var,
                                          bool use_analyzer) {
  arith::Analyzer analyzer;
  return FiniteDifference(offset, var, use_analyzer ? &analyzer : nullptr);
}

ffi::Array<ffi::Any> AscendPolyhedralClassifyDelta(const PrimExpr& delta,
                                                   bool use_analyzer) {
  arith::Analyzer analyzer;
  return AxisClassToArray(ClassifyDelta(delta, use_analyzer ? &analyzer : nullptr));
}

/*! \brief P1 pass: mark every nest P1 lifts with its region carrier. */
tvm::transform::Pass AscendLiftParallelToArray() {
  return MakeStagePass("tl.AscendLiftParallelToArray", LiftParallelToArray);
}

/*! \brief P2 pass: consume the P1 region carrier, attach the plan carrier. */
tvm::transform::Pass AscendPlanArray() {
  return MakeStagePass("tl.AscendPlanArray", PlanArray);
}

/*! \brief P3 pass: consume plan carriers, emit ``T.simd.*``. */
tvm::transform::Pass AscendLowerArrayPlan() {
  return MakeStagePass("tl.AscendLowerArrayPlan", LowerArrayPlan);
}

/*! \brief The combined pass: P1 -> P2 -> P3 on one PrimFunc. */
tvm::transform::Pass AscendPolyhedralVectorize() {
  return MakeStagePass("tl.AscendPolyhedralVectorize", VectorizeParallelNests);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendPolyhedralAxisClass",
                        AscendPolyhedralAxisClass);
  refl::GlobalDef().def("tl.transform.AscendPolyhedralBuildPlan",
                        AscendPolyhedralBuildPlan);
  refl::GlobalDef().def("tl.transform.AscendPolyhedralEmitPlan",
                     AscendPolyhedralEmitPlan);
  refl::GlobalDef().def("tl.transform.AscendPolyhedralRenderPlan",
                        AscendPolyhedralRenderPlan);
  refl::GlobalDef().def("tl.transform.AscendPolyhedralVectorizeNests",
                        AscendPolyhedralVectorizeNests);
  refl::GlobalDef().def("tl.transform.AscendPolyhedralFiniteDifference",
                        AscendPolyhedralFiniteDifference);
  refl::GlobalDef().def("tl.transform.AscendPolyhedralClassifyDelta",
                        AscendPolyhedralClassifyDelta);
  refl::GlobalDef().def("tl.transform.AscendLiftParallelToArray",
                        AscendLiftParallelToArray);
  refl::GlobalDef().def("tl.transform.AscendPlanArray", AscendPlanArray);
  refl::GlobalDef().def("tl.transform.AscendLowerArrayPlan", AscendLowerArrayPlan);
  refl::GlobalDef().def("tl.transform.AscendPolyhedralVectorize",
                        AscendPolyhedralVectorize);
}

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm
