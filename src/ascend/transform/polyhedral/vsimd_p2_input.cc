/*!
 * \file vsimd_p2_input.cc
 * \brief Implementation of the P1 -> P2 hand-off (parallel For chain inputs).
 */
#include "vsimd_p2_input.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../vsimd_memory_legality.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

namespace {

/*!
 * \brief Recursive For-chain conversion.
 *
 * ``require_loop`` is true for the top-level statement (the expansion must
 * start with a loop) and false for loop bodies (the innermost body is kept
 * verbatim: a BufferStore or a SeqStmt of BufferStores).
 */
bool AsParallelNestImpl(const Stmt &expanded, bool require_loop, Stmt *out,
                        std::string *error) {
  const ForNode *loop = nullptr;
  if (const auto *for_node = expanded.as<ForNode>()) {
    loop = for_node;
  } else if (const auto *seq = expanded.as<SeqStmtNode>()) {
    for (const Stmt &child : seq->seq) {
      if (const auto *for_node = child.as<ForNode>()) {
        if (loop != nullptr) {
          *error = "expanded has multiple sibling loops";
          return false;
        }
        loop = for_node;
      }
    }
    if (loop != nullptr) {
      // Only DeclBuffer declarations may accompany the loop; the stores live in
      // its body.
      for (const Stmt &child : seq->seq) {
        if (!child.as<ForNode>() && !child.as<DeclBufferNode>()) {
          *error = "unexpected statement next to the expanded loop: " +
                   std::string(child->GetTypeKey());
          return false;
        }
      }
    }
  }

  if (loop == nullptr) {
    if (require_loop) {
      *error = "expanded has no loop";
      return false;
    }
    // Innermost body (stores only): keep verbatim.
    *out = expanded;
    return true;
  }
  if (loop->kind != ForKind::kSerial) {
    *error = "unexpected expanded loop kind";
    return false;
  }
  Stmt body;
  if (!AsParallelNestImpl(loop->body, /*require_loop=*/false, &body, error))
    return false;
  *out = For(loop->loop_var, loop->min, loop->extent, ForKind::kParallel, body);
  return true;
}

Map<String, Any> P2InputToMap(const VSimdP2Input &item) {
  Map<String, Any> map;
  map.Set("vf_index", Any(item.vf_index));
  map.Set("nest_index", Any(item.nest_index));
  map.Set("status", Any(String(item.status)));
  map.Set("reason", Any(String(item.reason)));
  // The statement is only ever handed out for LIFTED nests; the typed region
  // stays on the C++ side (its access expressions are not stringified).
  if (item.stmt.defined())
    map.Set("stmt", Any(item.stmt));
  return map;
}

} // namespace

bool AsParallelNest(const Stmt &expanded, Stmt *out, std::string *error) {
  error->clear();
  return AsParallelNestImpl(expanded, /*require_loop=*/true, out, error);
}

bool RegionToPlanInput(const VSimdRegion &region, Stmt *out,
                       std::string *error) {
  return AsParallelNest(ExpandVSimdRegionToLoops(region), out, error);
}

Stmt RestoreBindings(const Stmt &stmt,
                     const std::vector<std::pair<Var, PrimExpr>> &bindings) {
  if (bindings.empty() || !stmt.defined())
    return stmt;
  std::unordered_map<const VarNode *, PrimExpr> values;
  for (const auto &binding : bindings) {
    // Bind values that are calls (e.g. buffer declarations / type
    // constructors) are not value aliases; same rule as P2 _collect_aliases.
    if (binding.second.as<CallNode>() != nullptr)
      continue;
    values[binding.first.get()] = binding.second;
  }
  if (values.empty())
    return stmt;

  Stmt result = stmt;
  // Aliases chains are short; bound the passes to avoid a cyclic Bind set.
  for (int pass = 0; pass < 8; ++pass) {
    std::unordered_map<const VarNode *, PrimExpr> free_values;
    for (const Var &var : UndefinedVars(result, {})) {
      auto it = values.find(var.get());
      if (it != values.end())
        free_values[it->first] = it->second;
    }
    if (free_values.empty())
      break;
    Stmt replaced = Substitute(result, free_values);
    if (replaced.same_as(result))
      break;
    result = replaced;
  }
  return result;
}

std::vector<VSimdP2Input> BuildP2PlanInputs(const PrimFunc &func) {
  std::vector<std::string> statuses = AnalyzeVSimdMemoryStatuses(func);
  std::vector<VSimdCaptureInfo> infos = AnalyzeVSimdCaptureInfo(func);

  std::vector<VSimdP2Input> out;
  out.reserve(infos.size());
  for (const VSimdCaptureInfo &info : infos) {
    VSimdP2Input item;
    item.vf_index = info.vf_index;
    item.nest_index = info.nest_index;

    if (info.nest_index < 0) {
      item.status = "OPAQUE";
      item.reason = info.message;
      out.push_back(std::move(item));
      continue;
    }

    std::string memory_status =
        (info.vf_index >= 0 &&
         static_cast<size_t>(info.vf_index) < statuses.size())
            ? statuses[info.vf_index]
            : "MISSING";

    VSimdRegionLift lift = BuildMinimalRegion(info, memory_status);
    if (!lift.lifted) {
      item.status = lift.internal_error ? "INTERNAL_ERROR" : "FALLBACK";
      item.reason = lift.reason;
      out.push_back(std::move(item));
      continue;
    }

    item.region = lift.region;
    Stmt parallel;
    std::string error;
    if (!RegionToPlanInput(lift.region, &parallel, &error)) {
      LOG(WARNING) << "[vsimd] P2 input adaptation failed (compiler bug): "
                   << error;
      item.status = "INTERNAL_ERROR";
      item.reason = error;
      out.push_back(std::move(item));
      continue;
    }
    // The reference expansion only rebuilds stores, so outer Bind definitions
    // used by an extent/index must be inlined back before P2 sees them.
    parallel = RestoreBindings(parallel, lift.region.bindings);

    item.status = "LIFTED";
    item.stmt = parallel;
    out.push_back(std::move(item));
  }
  return out;
}

Array<Map<String, Any>> AscendBuildP2PlanInputs(const PrimFunc &func) {
  Array<Map<String, Any>> out;
  for (const VSimdP2Input &item : BuildP2PlanInputs(func))
    out.push_back(P2InputToMap(item));
  return out;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendBuildP2PlanInputs",
                        AscendBuildP2PlanInputs);
}

} // namespace tl
} // namespace tvm
