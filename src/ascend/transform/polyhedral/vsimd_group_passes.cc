/*!
 * \file vsimd_group_passes.cc
 * \brief P2/P3 passes over the P1 ``tl.vsimd.program`` annotation.
 */
#include "vsimd_group_passes.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../vsimd_handoff.h"
#include "../vsimd_program.h"
#include "../vsimd_region.h"
#include "emit.h"
#include "model.h"
#include "plan.h"
#include "primitives.h"

namespace tvm {
namespace tl {
namespace polyhedral {

using namespace tirx;
using namespace tirx::transform;

using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

namespace {

constexpr const char *kProgramKey = "tl.vsimd.program";
constexpr const char *kGroupPlanKey = "tl.vsimd.group_plan";

static bool ReadIntField(const Map<String, Any> &map, const char *key, int *out,
                         std::string *error) {
  auto value = map.Get(key);
  if (!value) {
    if (error != nullptr)
      *error = std::string("missing '") + key + "'";
    return false;
  }
  auto imm = value.value().as<IntImm>();
  if (!imm) {
    if (error != nullptr)
      *error = std::string("'") + key + "' is not an integer";
    return false;
  }
  *out = static_cast<int>(imm.value()->value);
  return true;
}

static bool ReadStringField(const Map<String, Any> &map, const char *key,
                            std::string *out, std::string *error) {
  auto value = map.Get(key);
  if (!value) {
    if (error != nullptr)
      *error = std::string("missing '") + key + "'";
    return false;
  }
  auto str = value.value().as<String>();
  if (!str) {
    if (error != nullptr)
      *error = std::string("'") + key + "' is not a string";
    return false;
  }
  *out = std::string(str.value());
  return true;
}

static Array<String> ToStringArray(const std::vector<std::string> &items) {
  std::vector<String> out;
  out.reserve(items.size());
  for (const std::string &item : items)
    out.push_back(String(item));
  return Array<String>(out.begin(), out.end());
}

static bool GroupUsesFragment(const VSimdProgram &program,
                              const VSimdProgramGroup &group) {
  if (program.fragments.empty())
    return false;
  std::unordered_set<const VarNode *> fragments;
  for (const VSimdProgramFragment &fragment : program.fragments) {
    if (fragment.buffer.defined())
      fragments.insert(fragment.buffer->data.get());
  }
  if (fragments.empty())
    return false;
  for (int index : group.items) {
    if (index < 0 || index >= static_cast<int>(program.items.size()))
      return true;
    const VSimdProgramItem &item = program.items[index];
    for (const Buffer &buffer : item.read_buffers) {
      if (buffer.defined() && fragments.count(buffer->data.get()) != 0)
        return true;
    }
    for (const Buffer &buffer : item.write_buffers) {
      if (buffer.defined() && fragments.count(buffer->data.get()) != 0)
        return true;
    }
  }
  return false;
}

static Map<String, Any> PlanOneGroup(const VSimdProgram &program,
                                     const VSimdProgramGroup &group,
                                     const VSimdFlattenedBody &body,
                                     const SBlock &vf) {
  Map<String, Any> result;
  result.Set("group_id", Any(Integer(group.id)));

  std::string reason;
  std::vector<std::string> diagnostics;
  std::vector<Map<String, Any>> item_plans;
  bool ok = true;

  // Fragment groups are deferred to the legacy path in v1: the existing VF
  // register planner is VF-atomic and cannot be combined with a partially
  // planned group.
  if (GroupUsesFragment(program, group)) {
    ok = false;
    reason = "fragment groups are not planned by the v1 group planner";
  }

  for (int index : group.items) {
    if (!ok)
      break;
    if (index < 0 || index >= static_cast<int>(program.items.size())) {
      ok = false;
      reason = "group references an invalid item index";
      break;
    }
    const VSimdProgramItem &item = program.items[index];
    if (item.kind != "REGION") {
      ok = false;
      reason = "COPY/REDUCE plan is not implemented in the v1 group planner";
      diagnostics.push_back("item " + std::to_string(index) + ": " + item.kind);
      break;
    }
    if (item.status != "LIFTED") {
      ok = false;
      reason = "REGION item is not effectively LIFTED";
      diagnostics.push_back("item " + std::to_string(index) + ": " + item.status);
      break;
    }
    if (item.span_begin < 0 ||
        item.span_begin >= static_cast<int>(body.stmts.size()) ||
        item.span_end != item.span_begin + 1) {
      ok = false;
      reason = "item span is out of range or not unit-length";
      diagnostics.push_back("item " + std::to_string(index));
      break;
    }

    VSimdRegion region;
    std::string error;
    if (!ReadVSimdRegion(item.meta, program.vf_index, &region, &error)) {
      ok = false;
      reason = "invalid region payload: " + error;
      diagnostics.push_back("item " + std::to_string(index) + ": " + error);
      break;
    }

    PlanOptions options;
    options.layouts = region.layouts;
    options.assumptions = region.assumptions;
    options.bindings = region.bindings;
    options.shape_syms = region.shape_syms;
    options.vf_alloc_scope.assign(vf->alloc_buffers.begin(),
                                  vf->alloc_buffers.end());

    try {
      VectorProgram plan =
          BuildPlan(body.stmts[item.span_begin], std::nullopt, options);
      Map<String, Any> item_plan;
      item_plan.Set("item_index", Any(Integer(index)));
      item_plan.Set("span_begin", Any(Integer(item.span_begin)));
      item_plan.Set("span_end", Any(Integer(item.span_end)));
      item_plan.Set("kind", Any(String(item.kind)));
      item_plan.Set("plan_kind", Any(String("REGION")));
      item_plan.Set("region_plan", Any(plan));
      item_plans.push_back(std::move(item_plan));
    } catch (const Unsupported &e) {
      ok = false;
      reason = e.message();
      diagnostics.push_back("item " + std::to_string(index) + ": " + e.message());
      for (const std::string &diag : e.diagnostics())
        diagnostics.push_back(diag);
      break;
    }
  }

  result.Set("status", Any(String(ok ? "PLANNED" : "REJECTED")));
  result.Set("reason", Any(String(reason)));
  result.Set("diagnostics", Any(ToStringArray(diagnostics)));
  if (ok) {
    result.Set("items", Any(Array<Map<String, Any>>(item_plans.begin(),
                                                    item_plans.end())));
  } else {
    // Never publish a partial group plan: an unplanned group is left verbatim
    // and handled by the legacy lowering.
    item_plans.clear();
    result.Set("items", Any(Array<Map<String, Any>>()));
  }
  return result;
}

class GroupPlanner final : public StmtMutator {
 public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF")
      return StmtMutator::VisitStmt_(op);

    SBlock block = ffi::GetRef<SBlock>(op);
    auto annotation = op->annotations.find(kProgramKey);
    if (annotation == op->annotations.end())
      return block;
    auto program_map = (*annotation).second.as<Map<String, Any>>();
    if (!program_map)
      return block;

    VSimdProgram program;
    std::string error;
    if (!ReadVSimdProgram(program_map.value(), &program, &error)) {
      LOG(WARNING) << "[vsimd] P2 skipped a VF with an invalid P1 program: "
                   << error;
      return block;
    }
    if (program.raw.Get("validation_error")) {
      LOG(WARNING) << "[vsimd] P2 skipped a VF with a P1 validation_error";
      return block;
    }

    VSimdFlattenedBody body = FlattenVSimdBody(block->body);
    if (program.item_count >= 0 &&
        program.item_count != static_cast<int>(body.stmts.size())) {
      LOG(WARNING) << "[vsimd] P2 skipped a VF whose body item count changed "
                   << "between P1 and P2";
      return block;
    }
    if (program.has_body_fingerprint &&
        program.body_fingerprint != body.fingerprint) {
      LOG(WARNING) << "[vsimd] P2 skipped a VF whose body changed between P1 "
                   << "and P2";
      return block;
    }

    std::vector<Map<String, Any>> groups;
    for (const VSimdProgramGroup &group : program.groups) {
      if (group.status != "P1_LIFTABLE")
        continue;
      groups.push_back(PlanOneGroup(program, group, body, block));
    }

    Map<String, Any> plan_set;
    plan_set.Set("version", Any(Integer(1)));
    plan_set.Set("vf_index", Any(Integer(program.vf_index)));
    plan_set.Set("groups", Any(Array<Map<String, Any>>(groups.begin(),
                                                       groups.end())));

    SBlockNode *writer = block.CopyOnWrite();
    writer->annotations.Set(kGroupPlanKey, plan_set);
    return block;
  }
};

class GroupEmitter final : public StmtMutator {
 public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF")
      return StmtMutator::VisitStmt_(op);

    SBlock block = ffi::GetRef<SBlock>(op);
    auto annotation = op->annotations.find(kGroupPlanKey);
    if (annotation == op->annotations.end()) {
      // P2 did not plan this VF (no program, invalid program, or body change).
      // P3 is still responsible for not leaking the P1 annotation to codegen.
      if (op->annotations.count(kProgramKey) != 0) {
        SBlockNode *writer = block.CopyOnWrite();
        writer->annotations.erase(kProgramKey);
      }
      return block;
    }
    auto plan_set = (*annotation).second.as<Map<String, Any>>();
    if (!plan_set)
      return block;

    VSimdFlattenedBody body = FlattenVSimdBody(block->body);
    std::map<int, Stmt> replacements;
    std::string error;
    auto groups_value = plan_set.value().Get("groups");
    if (groups_value) {
      auto groups = groups_value.value().as<Array<Map<String, Any>>>();
      if (groups) {
        for (const Map<String, Any> &group : groups.value()) {
          std::string status;
          if (!ReadStringField(group, "status", &status, &error) ||
              status != "PLANNED")
            continue;
          auto items_value = group.Get("items");
          if (!items_value)
            continue;
          auto items = items_value.value().as<Array<Map<String, Any>>>();
          if (!items)
            continue;

          std::vector<int> spans;
          std::vector<Stmt> emitted;
          bool ok = true;
          for (const Map<String, Any> &item : items.value()) {
            int span = -1;
            if (!ReadIntField(item, "span_begin", &span, &error) || span < 0 ||
                span >= static_cast<int>(body.stmts.size())) {
              ok = false;
              break;
            }
            auto plan_value = item.Get("region_plan");
            if (!plan_value) {
              ok = false;
              break;
            }
            auto plan = plan_value.value().as<VectorProgram>();
            if (!plan) {
              ok = false;
              break;
            }
            try {
              emitted.push_back(EmitPlan(ffi::ObjectRef(plan.value()), nullptr));
              spans.push_back(span);
            } catch (const Unsupported &e) {
              LOG(WARNING) << "[vsimd] P3 rejected a group at span " << span
                           << ": " << e.message();
              ok = false;
              break;
            }
          }
          if (!ok)
            continue;  // whole-group rollback: keep every original statement
          for (size_t i = 0; i < spans.size(); ++i)
            replacements[spans[i]] = emitted[i];
        }
      }
    }

    SBlockNode *writer = block.CopyOnWrite();
    if (!replacements.empty())
      writer->body = ReplaceVSimdSpans(writer->body, replacements);
    // The annotation describes the original body; it must never survive the
    // rewrite or reach the later passes / codegen.
    writer->annotations.erase(kProgramKey);
    writer->annotations.erase(kGroupPlanKey);
    return block;
  }
};

tvm::transform::Pass MakeGroupPass(const char *name,
                                   tirx::PrimFunc (*fn)(tirx::PrimFunc)) {
  auto pass_func = [fn](tirx::PrimFunc func, const IRModule &,
                        PassContext) { return fn(std::move(func)); };
  return CreatePrimFuncPass(pass_func, 0, name, {});
}

}  // namespace

tirx::PrimFunc PlanVSimd(tirx::PrimFunc func) {
  GroupPlanner planner;
  tirx::PrimFuncNode *writer = func.CopyOnWrite();
  writer->body = planner(std::move(writer->body));
  return func;
}

tirx::PrimFunc EmitVSimd(tirx::PrimFunc func) {
  GroupEmitter emitter;
  tirx::PrimFuncNode *writer = func.CopyOnWrite();
  writer->body = emitter(std::move(writer->body));
  return func;
}

tirx::PrimFunc VectorizeVSimd(tirx::PrimFunc func) {
  func = PlanVSimd(std::move(func));
  return EmitVSimd(std::move(func));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendPlanVSimd", []() {
    return MakeGroupPass("tl.AscendPlanVSimd", PlanVSimd);
  });
  refl::GlobalDef().def("tl.transform.AscendEmitVSimd", []() {
    return MakeGroupPass("tl.AscendEmitVSimd", EmitVSimd);
  });
  refl::GlobalDef().def("tl.transform.AscendVectorizeVSimd", []() {
    return MakeGroupPass("tl.AscendVectorizeVSimd", VectorizeVSimd);
  });
}

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm
