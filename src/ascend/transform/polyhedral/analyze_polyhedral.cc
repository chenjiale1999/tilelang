/*!
 * \file analyze_polyhedral.cc
 * \brief Offline analysis/planning FFI for the polyhedral P2/P3 implementation.
 *
 * These entry points are the test-facing counterpart of the pipeline passes
 * (``AscendPlanVSimd`` / ``AscendEmitVSimd``): they plan and emit a raw
 * ``T.Parallel`` nest without P1 annotations, and expose the plan text and the
 * Δ-classification helpers used by the offline test suite.
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

#include <optional>
#include <string>
#include <vector>

#include "access.h"
#include "emit.h"
#include "model.h"
#include "plan.h"
#include "primitives.h"

namespace tvm {
namespace tl {
namespace polyhedral {

namespace {

ffi::Map<ffi::String, ffi::Any> MakeError(const Unsupported& error) {
  ffi::Map<ffi::String, ffi::Any> result;
  result.Set("ok", ffi::Any(false));
  result.Set("code", ffi::Any(ffi::String(error.code())));
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

/*! \brief FFI plan inputs -> ``PlanOptions`` (empty optional = derive). */
PlanOptions MakePlanOptions(const ffi::Optional<ffi::Map<tirx::Buffer, Layout>>& layouts) {
  PlanOptions options;
  if (layouts.has_value()) {
    for (const auto& entry : *layouts) {
      options.layouts.emplace_back(entry.first, entry.second);
    }
  }
  return options;
}

/* ------------------------------------------------------------------------- */
/* plan rendering (test oracle)                                              */
/* ------------------------------------------------------------------------- */

std::string OptionalIntText(const ffi::Optional<int64_t>& value) {
  return value.has_value() ? std::to_string(*value) : "None";
}

/*! \brief Python ``f"{buffer.name}{list(indices)}"``. */
std::string SinkText(const StoreSpec& spec) {
  std::string out = spec->buffer->name.operator std::string() + "[";
  for (size_t i = 0; i < spec->indices.size(); ++i) {
    out += (i == 0 ? "" : ", ") + PrintExpr(spec->indices[i]);
  }
  return out + "]";
}

std::string PyStrList(const std::vector<std::string>& items) {
  std::string out = "[";
  for (size_t i = 0; i < items.size(); ++i) out += (i == 0 ? "'" : ", '") + items[i] + "'";
  return out + "]";
}

void RenderValue(const ValueNode& node, std::vector<std::string>* lines, int depth) {
  std::string pad(static_cast<size_t>(depth) * 2, ' ');
  std::string detail = node->detail.empty()
                           ? std::string()
                           : " (" + node->detail.operator std::string() + ")";
  lines->push_back(pad + "- " + node->recipe.operator std::string() + ": " +
                   node->dtype.operator std::string() + detail);
  for (const ffi::ObjectRef& arg : node->args) {
    RenderValue(Downcast<ValueNode>(arg), lines, depth + 1);
  }
}

std::vector<std::string> RenderSinglePlan(const VectorPlan& plan,
                                          const std::vector<std::string>& extra_diagnostics) {
  std::vector<std::string> lines;
  lines.push_back("vector_axis=" + plan->vector_axis->name.operator std::string() +
                  " lanes=" + std::to_string(plan->lanes) +
                  " chunks=" + OptionalIntText(plan->chunks) +
                  " tail=" + OptionalIntText(plan->tail));
  ReduceInfo reduce = ReduceOf(plan);
  if (reduce.defined()) {
    lines.push_back("reduce(axis=" + reduce->axis->name.operator std::string() + ", op=" +
                    reduce->op.operator std::string() + " -> " +
                    reduce->recipe.operator std::string() + ")");
  }
  const std::pair<const char*, ffi::Array<StorePlan>> groups[] = {
      {"pre", plan->pre_stores}, {"store", plan->stores}, {"post", plan->post_stores}};
  for (const auto& group : groups) {
    for (const StorePlan& store : group.second) {
      // Mixed element widths / destination layouts: only non-default stores
      // carry an extra note, so the render of an ordinary plan stays identical.
      const std::string lane_mode = store->store_lane_mode.operator std::string();
      std::string note =
          lane_mode == "full"
              ? ""
              : " store_lanes=" + lane_mode + "/" + std::to_string(store->store_lanes);
      const std::string layout_kind = store->layout_kind.operator std::string();
      if (layout_kind != "nd") {
        note += " layout=" + layout_kind + "[" +
                std::to_string(store->nz_block_stride.has_value()
                                   ? *store->nz_block_stride
                                   : 0) +
                "]";
      }
      lines.push_back(std::string(group.first) + " " + SinkText(store->spec) + " [" +
                      AccessClassName(store->kind) + "] mask=" +
                      (store->mask.has_value() ? "yes" : "no") + note);
      RenderValue(store->value, &lines, 1);
    }
  }
  if (!plan->diagnostics.empty() || !extra_diagnostics.empty()) {
    lines.push_back("diagnostics:");
    for (const ffi::String& note : plan->diagnostics) {
      lines.push_back("  " + note.operator std::string());
    }
    for (const std::string& note : extra_diagnostics) lines.push_back("  " + note);
  }
  return lines;
}

std::string RenderPlanText(const ffi::ObjectRef& plan) {
  if (const auto* single = plan.as<VectorPlanNode>()) {
    std::vector<std::string> lines = RenderSinglePlan(ffi::GetRef<VectorPlan>(single), {});
    std::string out;
    for (size_t i = 0; i < lines.size(); ++i) out += (i == 0 ? "" : "\n") + lines[i];
    return out;
  }
  const auto* program = plan.as<VectorProgramNode>();
  TVM_FFI_ICHECK(program != nullptr) << "render_plan expects a VectorPlan or VectorProgram";
  VectorProgram node = ffi::GetRef<VectorProgram>(program);
  if (node->segments.size() == 1) {
    std::vector<std::string> extra;
    for (const ffi::String& note : node->diagnostics) {
      extra.push_back(note.operator std::string());
    }
    std::vector<std::string> lines = RenderSinglePlan(node->segments[0], extra);
    std::string out;
    for (size_t i = 0; i < lines.size(); ++i) out += (i == 0 ? "" : "\n") + lines[i];
    return out;
  }

  std::vector<std::string> names;
  for (const Axis& axis : node->axes) names.push_back(axis->name.operator std::string());
  std::vector<std::string> lines;
  lines.push_back("program: axes=" + PyStrList(names) +
                  " segments=" + std::to_string(node->segments.size()));
  for (size_t index = 0; index < node->segments.size(); ++index) {
    lines.push_back("-- segment " + std::to_string(index) + " --");
    for (const std::string& line : RenderSinglePlan(node->segments[index], {})) {
      lines.push_back(line);
    }
  }
  if (!node->diagnostics.empty()) {
    lines.push_back("diagnostics:");
    for (const ffi::String& note : node->diagnostics) {
      lines.push_back("  " + note.operator std::string());
    }
  }
  std::string out;
  for (size_t i = 0; i < lines.size(); ++i) out += (i == 0 ? "" : "\n") + lines[i];
  return out;
}

}  // namespace

ffi::Array<ffi::Any> AscendPolyhedralAxisClass(const AccessInfo& info, const Axis& axis) {
  return AxisClassToArray(AxisClassOf(info, axis));
}

ffi::Map<ffi::String, ffi::Any> AscendPolyhedralBuildPlan(
    const tirx::Stmt& stmt, ffi::Optional<int64_t> lanes,
    ffi::Optional<ffi::Map<tirx::Buffer, Layout>> layouts) {
  return Guarded([&]() {
    VectorProgram program =
        BuildPlan(stmt, OptionalLanes(lanes), MakePlanOptions(layouts));
    return MakeOk("program", ffi::Any(program));
  });
}

ffi::Map<ffi::String, ffi::Any> AscendPolyhedralEmitPlan(const ffi::ObjectRef& plan) {
  return Guarded([&]() { return MakeOk("stmt", ffi::Any(EmitPlan(plan))); });
}

ffi::Map<ffi::String, ffi::Any> AscendPolyhedralRenderPlan(const ffi::ObjectRef& plan) {
  return Guarded(
      [&]() { return MakeOk("text", ffi::Any(ffi::String(RenderPlanText(plan)))); });
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
  refl::GlobalDef().def("tl.transform.AscendPolyhedralFiniteDifference",
                        AscendPolyhedralFiniteDifference);
  refl::GlobalDef().def("tl.transform.AscendPolyhedralClassifyDelta",
                        AscendPolyhedralClassifyDelta);
}

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm
