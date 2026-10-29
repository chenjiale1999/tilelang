/*!
 * \file render.cc
 * \brief Human-readable plan rendering.
 */
#include "render.h"

#include <string>
#include <vector>

#include "primitives.h"

namespace tvm {
namespace tl {
namespace polyhedral {

namespace {

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
      lines.push_back(std::string(group.first) + " " + SinkText(store->spec) + " [" +
                      store->kind_value.operator std::string() + "] mask=" +
                      (store->mask.has_value() ? "yes" : "no"));
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

}  // namespace

std::string RenderPlan(const ffi::ObjectRef& plan) {
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

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm
