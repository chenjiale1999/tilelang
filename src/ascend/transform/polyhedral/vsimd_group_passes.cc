/*!
 * \file vsimd_group_passes.cc
 * \brief P2/P3 passes over the P1 ``tl.vsimd.program`` annotation.
 */
#include "vsimd_group_passes.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../vsimd_handoff.h"
#include "../vsimd_program.h"
#include "../vsimd_region.h"
#include "emit.h"
#include "fragment_plan.h"
#include "group_plan.h"
#include "layout/layout.h"
#include "model.h"
#include "plan.h"
#include "primitives.h"
#include "vsimd_payload_read.h"

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

/*! \brief Why one group/item could not be planned. */
struct PlanFailure {
  std::string reason;
  std::vector<std::string> diagnostics;
};

Array<String> ToStringArray(const std::vector<std::string> &items) {
  std::vector<String> out;
  out.reserve(items.size());
  for (const std::string &item : items) out.push_back(String(item));
  return Array<String>(out.begin(), out.end());
}

/*! \brief Whether the storage scope is the UB read by SimdVF memory ops. */
bool IsUbScope(const std::string &scope) {
  return scope == "SHARED" || scope == "shared" || scope == "shared.dyn";
}

PrimExpr AddLoop(const PrimExpr &base, const Var &var) {
  return base + Cast(base.dtype(), var);
}

Var MakeLoopVar(const std::string &prefix, int index) {
  return Var(prefix + std::to_string(index), DataType::Int(32));
}

std::string ShowExpr(const PrimExpr &expr) {
  std::ostringstream os;
  os << expr;
  return os.str();
}

/*!
 * \brief Union of the region layouts of one program, keyed by buffer data Var.
 *
 * COPY/REDUCE items carry no layout payload of their own; the layouts P1
 * recorded on the REGION items of the same VF are the only source of truth
 * (an NZ destination must not be stored in ND order).
 */
std::vector<std::pair<Buffer, Layout>> CollectProgramLayouts(
    const VSimdProgram &program) {
  std::vector<std::pair<Buffer, Layout>> layouts;
  std::unordered_set<const VarNode *> seen;
  for (const VSimdProgramItem &item : program.items) {
    if (item.kind != "REGION" || item.status != "LIFTED") continue;
    VSimdRegion region;
    std::string error;
    if (!ReadVSimdRegion(item.meta, program.vf_index, &region, &error)) continue;
    for (const auto &entry : region.layouts) {
      if (seen.insert(entry.first->data.get()).second) layouts.push_back(entry);
    }
  }
  return layouts;
}

PlanOptions MakeOptions(const SBlock &vf,
                        const std::vector<std::pair<Buffer, Layout>> &layouts) {
  PlanOptions options;
  options.layouts = layouts;
  options.vf_alloc_scope.assign(vf->alloc_buffers.begin(),
                                vf->alloc_buffers.end());
  return options;
}

/* ------------------------------------------------------------------------- */
/* REGION                                                                    */
/* ------------------------------------------------------------------------- */

bool PlanRegionItem(const VSimdProgram &program, const VSimdProgramItem &item,
                    const VSimdFlattenedBody &body, const SBlock &vf,
                    VSimdItemPlan *out, PlanFailure *failure) {
  if (item.status != "LIFTED") {
    failure->reason = "REGION item is not effectively LIFTED";
    failure->diagnostics.push_back("item " + std::to_string(item.index) + ": " +
                                   item.status);
    return false;
  }
  if (item.span_begin < 0 ||
      item.span_begin >= static_cast<int>(body.stmts.size()) ||
      item.span_end != item.span_begin + 1) {
    failure->reason = "item span is out of range or not unit-length";
    failure->diagnostics.push_back("item " + std::to_string(item.index));
    return false;
  }

  VSimdRegion region;
  std::string error;
  if (!ReadVSimdRegion(item.meta, program.vf_index, &region, &error)) {
    failure->reason = "invalid region payload: " + error;
    failure->diagnostics.push_back("item " + std::to_string(item.index) + ": " +
                                   error);
    return false;
  }

  PlanOptions options = MakeOptions(vf, region.layouts);
  options.assumptions = region.assumptions;
  options.bindings = region.bindings;
  options.shape_syms = region.shape_syms;

  try {
    VectorProgram plan =
        BuildPlan(body.stmts[item.span_begin], std::nullopt, options);
    *out = MakeVSimdRegionItemPlan(item.index, item.span_begin, item.span_end,
                                   item.kind, std::move(plan));
    return true;
  } catch (const Unsupported &e) {
    failure->reason = e.message();
    failure->diagnostics.push_back("item " + std::to_string(item.index) +
                                   ": " + e.message());
    for (const std::string &diag : e.diagnostics())
      failure->diagnostics.push_back(diag);
    return false;
  }
}

/* ------------------------------------------------------------------------- */
/* COPY (Phase B)                                                            */
/* ------------------------------------------------------------------------- */

/*!
 * \brief Synthesize ``for idx in T.Parallel(dst_extents):
 *        dst[dst_indices + idx] = src[src_indices + stride * idx]``.
 *
 * The copy is element-wise; a source extent of 1 along a dimension broadcasts.
 */
bool BuildCopyNest(const VSimdCopyReducePayload &payload, Stmt *out,
                   PlanFailure *failure) {
  const int rank = static_cast<int>(payload.dst_extents.size());
  if (rank == 0) {
    failure->reason = "a rank-0 copy has no parallel domain";
    return false;
  }
  if (static_cast<int>(payload.src_extents.size()) != rank) {
    failure->reason = "copy src/dst ranks differ (" +
                      std::to_string(payload.src_extents.size()) + " vs " +
                      std::to_string(rank) + ")";
    return false;
  }
  arith::Analyzer analyzer;
  std::vector<Var> vars;
  std::vector<PrimExpr> dst_indices;
  std::vector<PrimExpr> src_indices;
  for (int d = 0; d < rank; ++d) {
    vars.push_back(MakeLoopVar("cp_i", d));
    if (!analyzer.CanProveEqual(payload.src_extents[d], payload.dst_extents[d]) &&
        !tirx::is_one(payload.src_extents[d])) {
      failure->reason =
          "copy dimension " + std::to_string(d) + " has unequal extents " +
          ShowExpr(payload.src_extents[d]) + " and " +
          ShowExpr(payload.dst_extents[d]) +
          " (only equal or broadcast is supported)";
      return false;
    }
    const bool broadcast = tirx::is_one(payload.src_extents[d]);
    dst_indices.push_back(AddLoop(payload.dst_indices[d], vars[d]));
    src_indices.push_back(broadcast ? payload.src_indices[d]
                                    : AddLoop(payload.src_indices[d], vars[d]));
  }
  Stmt body = BufferStore(payload.dst, BufferLoad(payload.src, src_indices),
                          dst_indices);
  for (int d = rank - 1; d >= 0; --d) {
    body = For(vars[d], IntImm32(0), payload.dst_extents[d], ForKind::kParallel,
               body);
  }
  *out = std::move(body);
  return true;
}

bool PlanCopyItem(const VSimdProgram &program, const VSimdProgramItem &item,
                  const SBlock &vf,
                  const std::vector<std::pair<Buffer, Layout>> &program_layouts,
                  VSimdItemPlan *out, PlanFailure *failure) {
  VSimdCopyReducePayload payload;
  std::string error;
  if (!ReadVSimdCopy(item, &payload, &error)) {
    failure->reason = "invalid COPY payload: " + error;
    failure->diagnostics.push_back("item " + std::to_string(item.index) + ": " +
                                   error);
    return false;
  }
  if (!IsUbScope(payload.src_scope)) {
    failure->reason = "copy source '" + payload.src->name.operator std::string() +
                      "' is not UB (scope " + payload.src_scope + ")";
    return false;
  }
  if (!IsUbScope(payload.dst_scope)) {
    failure->reason =
        "copy destination '" + payload.dst->name.operator std::string() +
        "' is not UB (scope " + payload.dst_scope + ")";
    return false;
  }
  Stmt nest;
  if (!BuildCopyNest(payload, &nest, failure)) return false;
  try {
    VectorProgram plan =
        BuildPlan(nest, std::nullopt, MakeOptions(vf, program_layouts));
    const bool broadcast = std::any_of(
        payload.src_extents.begin(), payload.src_extents.end(),
        [](const PrimExpr &extent) { return tirx::is_one(extent); });
    *out = MakeVSimdCopyItemPlan(
        item.index, item.span_begin, item.span_end,
        MakeVSimdCopyPlan(broadcast ? "BROADCAST" : "ND", payload.src_scope,
                          payload.dst_scope,
                          Array<PrimExpr>(payload.src_extents.begin(),
                                          payload.src_extents.end()),
                          Array<PrimExpr>(payload.dst_extents.begin(),
                                          payload.dst_extents.end()),
                          std::move(plan)));
    return true;
  } catch (const Unsupported &e) {
    failure->reason = "COPY: " + e.message();
    failure->diagnostics.push_back("item " + std::to_string(item.index) +
                                   ": " + e.message());
    for (const std::string &diag : e.diagnostics())
      failure->diagnostics.push_back(diag);
    return false;
  }
}

/* ------------------------------------------------------------------------- */
/* REDUCE (Phase B)                                                          */
/* ------------------------------------------------------------------------- */

/*! \brief ``max``/``sum``/``min`` -> the reduction recipe of the emitter. */
bool ReduceRecipeOf(const std::string &kind, std::string *recipe) {
  if (kind == "max") {
    *recipe = "vcmax";
  } else if (kind == "min") {
    *recipe = "vcmin";
  } else if (kind == "sum") {
    *recipe = "vcadd";
  } else {
    return false;
  }
  return true;
}

/*!
 * \brief Map every source dimension of a reduce onto its destination dimension.
 *
 * Supported shapes: the destination drops the reduced dimension (the usual
 * ``reduce(src(m, k), dst(m), dim=1)``), or keeps it with extent 1.
 */
bool MapReduceDims(const VSimdReducePayload &payload, bool *kept_dim,
                   std::vector<int> *src_to_dst, PlanFailure *failure) {
  const int src_rank = static_cast<int>(payload.src_extents.size());
  const int dst_rank = static_cast<int>(payload.dst_extents.size());
  if (payload.dim < 0 || payload.dim >= src_rank) {
    failure->reason = "reduce dim " + std::to_string(payload.dim) +
                      " is outside the source rank " + std::to_string(src_rank);
    return false;
  }
  if (dst_rank == src_rank) {
    if (!tirx::is_one(payload.dst_extents[payload.dim])) {
      failure->reason =
          "reduce destination keeps the reduced dimension but its extent is not 1";
      return false;
    }
    *kept_dim = true;
    src_to_dst->assign(src_rank, 0);
    for (int t = 0; t < src_rank; ++t) (*src_to_dst)[t] = t;
  } else if (dst_rank + 1 == src_rank) {
    *kept_dim = false;
    src_to_dst->assign(src_rank, -1);
    for (int t = 0; t < src_rank; ++t) {
      if (t < payload.dim) (*src_to_dst)[t] = t;
      if (t > payload.dim) (*src_to_dst)[t] = t - 1;
    }
  } else {
    failure->reason = "reduce source/destination ranks are incompatible (" +
                      std::to_string(src_rank) + " -> " +
                      std::to_string(dst_rank) + ")";
    return false;
  }

  arith::Analyzer analyzer;
  for (int t = 0; t < src_rank; ++t) {
    if (t == payload.dim) continue;
    const int d = (*src_to_dst)[t];
    if (!analyzer.CanProveEqual(payload.src_extents[t], payload.dst_extents[d]) &&
        !tirx::is_one(payload.src_extents[t])) {
      failure->reason = "reduce dimension " + std::to_string(t) +
                        " does not match its destination extent";
      return false;
    }
  }
  return true;
}

/*!
 * \brief Synthesize the reduction nest of a ``T.reduce_<kind>`` item.
 *
 * ``dst[dst_idx] = op(dst[dst_idx], src[src_idx])`` over the whole source
 * domain: the planner's cross-lane reduction classification turns the
 * lane-varying source reads into ``vcadd``/``vcmax``/``vcmin``.
 */
bool BuildReduceNest(const VSimdReducePayload &payload, bool kept_dim,
                     const std::vector<int> &src_to_dst, Stmt *out,
                     PlanFailure *failure) {
  const int src_rank = static_cast<int>(payload.src_extents.size());
  std::vector<Var> vars;
  std::vector<PrimExpr> src_indices;
  for (int t = 0; t < src_rank; ++t) {
    vars.push_back(MakeLoopVar("rd_i", t));
    src_indices.push_back(AddLoop(payload.src_indices[t], vars[t]));
  }

  std::vector<PrimExpr> dst_indices;
  const int dst_rank = static_cast<int>(payload.dst_extents.size());
  for (int d = 0; d < dst_rank; ++d) {
    int src_dim = -1;
    for (int t = 0; t < src_rank; ++t) {
      if (src_to_dst[t] == d) {
        src_dim = t;
        break;
      }
    }
    if (src_dim < 0) {
      failure->reason = "reduce destination dimension " + std::to_string(d) +
                        " has no source dimension";
      return false;
    }
    const bool reduced = kept_dim && src_dim == payload.dim;
    dst_indices.push_back(
        reduced ? payload.dst_indices[d]
                : AddLoop(payload.dst_indices[d], vars[src_dim]));
  }

  BufferLoad old_value(payload.dst, Array<PrimExpr>(dst_indices.begin(),
                                                    dst_indices.end()));
  BufferLoad new_value(payload.src, src_indices);
  PrimExpr value;
  if (payload.reduce_kind == "max") {
    value = Max(old_value, new_value);
  } else if (payload.reduce_kind == "min") {
    value = Min(old_value, new_value);
  } else {
    value = Add(old_value, new_value);
  }
  Stmt body = BufferStore(payload.dst, value, dst_indices);
  for (int t = src_rank - 1; t >= 0; --t) {
    body = For(vars[t], IntImm32(0), payload.src_extents[t], ForKind::kParallel,
               body);
  }
  *out = std::move(body);
  return true;
}

/*! \brief ``for idx in T.Parallel(dst_extents): dst[dst_idx + idx] = identity``. */
bool BuildReduceInitNest(const VSimdReducePayload &payload,
                         const std::string &recipe, Stmt *out,
                         PlanFailure *failure) {
  const int dst_rank = static_cast<int>(payload.dst_extents.size());
  if (dst_rank == 0) {
    failure->reason = "a rank-0 reduce destination has no parallel domain";
    return false;
  }
  PrimExpr identity =
      ReduceIdentity(recipe, ElemDtype(payload.dst->dtype));
  std::vector<Var> vars;
  std::vector<PrimExpr> dst_indices;
  for (int d = 0; d < dst_rank; ++d) {
    vars.push_back(MakeLoopVar("ri_i", d));
    dst_indices.push_back(AddLoop(payload.dst_indices[d], vars[d]));
  }
  Stmt body = BufferStore(payload.dst, identity, dst_indices);
  for (int d = dst_rank - 1; d >= 0; --d) {
    body = For(vars[d], IntImm32(0), payload.dst_extents[d], ForKind::kParallel,
               body);
  }
  *out = std::move(body);
  return true;
}

bool PlanReduceItem(const VSimdProgram &program, const VSimdProgramItem &item,
                    const SBlock &vf,
                    const std::vector<std::pair<Buffer, Layout>> &program_layouts,
                    VSimdItemPlan *out, PlanFailure *failure) {
  VSimdReducePayload payload;
  std::string error;
  if (!ReadVSimdReduce(item, &payload, &error)) {
    failure->reason = "invalid REDUCE payload: " + error;
    failure->diagnostics.push_back("item " + std::to_string(item.index) + ": " +
                                   error);
    return false;
  }
  std::string recipe;
  if (!ReduceRecipeOf(payload.reduce_kind, &recipe)) {
    failure->reason = "reduce kind '" + payload.reduce_kind +
                      "' is not supported yet (max/sum/min)";
    return false;
  }
  if (!IsUbScope(payload.src_scope) || !IsUbScope(payload.dst_scope)) {
    failure->reason = "reduce operands must both be UB ('" +
                      payload.src->name.operator std::string() + "' / '" +
                      payload.dst->name.operator std::string() + "')";
    return false;
  }
  if (payload.src_extents.empty() || payload.dst_extents.empty()) {
    failure->reason = "reduce payload has an empty source or destination region";
    return false;
  }

  bool kept_dim = false;
  std::vector<int> src_to_dst;
  if (!MapReduceDims(payload, &kept_dim, &src_to_dst, failure)) return false;

  Stmt nest;
  if (!BuildReduceNest(payload, kept_dim, src_to_dst, &nest, failure))
    return false;
  std::optional<Stmt> init_nest;
  if (payload.clear) {
    Stmt init;
    if (!BuildReduceInitNest(payload, recipe, &init, failure)) return false;
    init_nest = std::move(init);
  }

  PlanOptions options = MakeOptions(vf, program_layouts);
  try {
    VectorProgram program_plan = BuildPlan(nest, std::nullopt, options);
    ffi::Optional<VectorProgram> init_plan;
    if (init_nest.has_value()) {
      init_plan = ffi::Optional<VectorProgram>(
          BuildPlan(*init_nest, std::nullopt, options));
    }
    *out = MakeVSimdReduceItemPlan(
        item.index, item.span_begin, item.span_end,
        MakeVSimdReducePlan(payload.reduce_kind, payload.dim, payload.clear,
                            std::move(init_plan), std::move(program_plan)));
    return true;
  } catch (const Unsupported &e) {
    failure->reason = "REDUCE: " + e.message();
    failure->diagnostics.push_back("item " + std::to_string(item.index) +
                                   ": " + e.message());
    for (const std::string &diag : e.diagnostics())
      failure->diagnostics.push_back(diag);
    return false;
  }
}

/* ------------------------------------------------------------------------- */
/* group planning                                                            */
/* ------------------------------------------------------------------------- */

bool GroupUsesOutsideFragment(const VSimdProgram &program,
                              const VSimdProgramGroup &group,
                              std::string *name) {
  std::unordered_set<const VarNode *> outside;
  std::unordered_map<const VarNode *, std::string> names;
  for (const VSimdProgramFragment &fragment : program.fragments) {
    if (fragment.definition_scope != "OUTSIDE_VF" || !fragment.buffer.defined())
      continue;
    outside.insert(fragment.buffer->data.get());
    names.emplace(fragment.buffer->data.get(), fragment.name);
  }
  if (outside.empty()) return false;
  for (int index : group.items) {
    if (index < 0 || index >= static_cast<int>(program.items.size())) continue;
    const VSimdProgramItem &item = program.items[index];
    for (const Buffer &buffer : item.read_buffers) {
      auto it = outside.find(buffer->data.get());
      if (it != outside.end()) {
        *name = names[*it];
        return true;
      }
    }
    for (const Buffer &buffer : item.write_buffers) {
      auto it = outside.find(buffer->data.get());
      if (it != outside.end()) {
        *name = names[*it];
        return true;
      }
    }
  }
  return false;
}

VSimdGroupPlan PlanOneGroup(
    const VSimdProgram &program, const VSimdProgramGroup &group,
    const VSimdFlattenedBody &body, const SBlock &vf,
    const std::vector<std::pair<Buffer, Layout>> &program_layouts) {
  // A fragment allocated outside the VF is a cross-VF value: its memory
  // identity cannot be replaced by a register and the register file is VF
  // local, so P2 rejects the group outright (P1 still records the fact).
  std::string outside_fragment;
  if (GroupUsesOutsideFragment(program, group, &outside_fragment)) {
    return MakeRejectedGroupPlan(
        group.id, "fragment '" + outside_fragment +
                      "' is allocated outside this VF (cross-VF fragment); "
                      "stage it through UB first");
  }

  PlanFailure failure;
  std::vector<VSimdItemPlan> items;
  for (int index : group.items) {
    if (index < 0 || index >= static_cast<int>(program.items.size())) {
      failure.reason = "group references an invalid item index";
      break;
    }
    const VSimdProgramItem &item = program.items[index];
    VSimdItemPlan plan;
    bool ok = false;
    if (item.kind == "REGION") {
      ok = PlanRegionItem(program, item, body, vf, &plan, &failure);
    } else if (item.kind == "COPY") {
      ok = PlanCopyItem(program, item, vf, program_layouts, &plan, &failure);
    } else if (item.kind == "REDUCE") {
      ok = PlanReduceItem(program, item, vf, program_layouts, &plan, &failure);
    } else {
      failure.reason = "unexpected " + item.kind + " item in a P1_LIFTABLE group";
    }
    if (!ok) break;
    items.push_back(std::move(plan));
  }

  if (!failure.reason.empty() || items.size() != group.items.size()) {
    return MakeRejectedGroupPlan(group.id,
                                 failure.reason.empty()
                                     ? "group planning stopped early"
                                     : failure.reason,
                                 failure.diagnostics);
  }
  VSimdGroupPlan plan = MakeVSimdGroupPlan(group.id, "PLANNED");
  VSimdGroupPlanNode *writer = plan.CopyOnWrite();
  writer->items = Array<VSimdItemPlan>(items.begin(), items.end());
  return plan;
}

/*! \brief VectorPrograms of the planned items of one group, in item order. */
void CollectPrograms(const VSimdGroupPlan &group,
                     std::vector<VectorProgram> *programs,
                     std::vector<int> *spans) {
  if (group->status.operator std::string() != "PLANNED") return;
  for (const VSimdItemPlan &item : group->items) {
    if (item->region_plan.has_value()) {
      programs->push_back(*item->region_plan);
      spans->push_back(static_cast<int>(item->span_begin));
    } else if (item->copy_plan.has_value()) {
      programs->push_back((*item->copy_plan)->program);
      spans->push_back(static_cast<int>(item->span_begin));
    } else if (item->reduce_plan.has_value()) {
      const VSimdReducePlan &reduce = *item->reduce_plan;
      if (reduce->init.has_value()) {
        programs->push_back(*reduce->init);
        spans->push_back(static_cast<int>(item->span_begin));
      }
      programs->push_back(reduce->program);
      spans->push_back(static_cast<int>(item->span_begin));
    }
  }
}

class GroupPlanner final : public StmtMutator {
 public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF")
      return StmtMutator::VisitStmt_(op);

    SBlock block = ffi::GetRef<SBlock>(op);
    auto annotation = op->annotations.find(kProgramKey);
    if (annotation == op->annotations.end()) return block;
    auto program_map = (*annotation).second.as<Map<String, Any>>();
    if (!program_map) return block;

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

    const std::vector<std::pair<Buffer, Layout>> program_layouts =
        CollectProgramLayouts(program);
    std::vector<VSimdGroupPlan> groups;
    for (const VSimdProgramGroup &group : program.groups) {
      if (group.status != "P1_LIFTABLE") continue;
      groups.push_back(
          PlanOneGroup(program, group, body, block, program_layouts));
    }

    // Fragment registers are VF atomic: decide once for the whole block.
    std::vector<VectorProgram> programs;
    std::vector<int> spans;
    for (const VSimdGroupPlan &group : groups)
      CollectPrograms(group, &programs, &spans);
    VfFragmentPlan fragments = PlanVfFragmentRegisters(block->body, programs, spans);
    if (!fragments.enabled && !fragments.reason.empty()) {
      LOG(WARNING) << "[vsimd] P2 rejected a VF's groups: " << fragments.reason;
      std::vector<VSimdGroupPlan> rejected;
      for (const VSimdGroupPlan &group : groups) {
        if (group->status.operator std::string() != "PLANNED") {
          rejected.push_back(group);
          continue;
        }
        rejected.push_back(MakeRejectedGroupPlan(
            group->group_id, "fragment registers rejected: " + fragments.reason));
      }
      groups = std::move(rejected);
    }

    VSimdGroupPlanSet plan_set =
        MakeVSimdGroupPlanSet(program.vf_index, std::move(groups));
    VSimdGroupPlanSetNode *writer = plan_set.CopyOnWrite();
    if (fragments.enabled) {
      writer->fragment_regs =
          ffi::Optional<FragmentRegisterMap>(fragments.map);
      writer->fragment_mode = ffi::String("register");
    } else if (!fragments.reason.empty()) {
      writer->fragment_mode = ffi::String("rejected");
      writer->fragment_reason = ffi::String(fragments.reason);
    }

    SBlockNode *block_writer = block.CopyOnWrite();
    block_writer->annotations.Set(kGroupPlanKey, plan_set);
    return block;
  }
};

/* ------------------------------------------------------------------------- */
/* P3                                                                        */
/* ------------------------------------------------------------------------- */

/*! \brief Replace the allocations of register-managed fragments. */
void RewriteFragmentAllocations(SBlockNode *writer,
                                const FragmentRegisterFile &registers) {
  Array<Buffer> kept;
  for (const Buffer &buffer : writer->alloc_buffers) {
    if (registers.Find(buffer->data) != nullptr) continue;
    kept.push_back(buffer);
  }
  for (const Buffer &reg : registers.Buffers()) kept.push_back(reg);
  writer->alloc_buffers = std::move(kept);

  if (writer->annotations.count(attr::kLayoutMap) == 0) return;
  ffi::Map<Buffer, Layout> old =
      Downcast<ffi::Map<Buffer, Layout>>(writer->annotations.at(attr::kLayoutMap));
  ffi::Map<Buffer, Layout> remapped;
  for (const auto &entry : old) {
    if (registers.Find(entry.first->data) != nullptr) continue;
    remapped.Set(entry.first, entry.second);
  }
  if (remapped.empty()) {
    writer->annotations.erase(attr::kLayoutMap);
  } else {
    writer->annotations.Set(attr::kLayoutMap, remapped);
  }
}

/*! \brief Emit one planned item; throws ``Unsupported`` on a P3 failure. */
Stmt EmitItem(const VSimdItemPlan &item, FragmentRegisterFile *registers) {
  if (item->region_plan.has_value())
    return EmitPlan(ffi::ObjectRef(*item->region_plan), registers);
  if (item->copy_plan.has_value()) {
    return EmitPlan(ffi::ObjectRef((*item->copy_plan)->program), registers);
  }
  if (item->reduce_plan.has_value()) {
    const VSimdReducePlan &reduce = *item->reduce_plan;
    std::vector<Stmt> stmts;
    if (reduce->init.has_value())
      stmts.push_back(EmitPlan(ffi::ObjectRef(*reduce->init), registers));
    stmts.push_back(EmitPlan(ffi::ObjectRef(reduce->program), registers));
    return SeqStmt(stmts);
  }
  throw Unsupported("empty item plan");
}

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
    const auto *plan_node = (*annotation).second.as<VSimdGroupPlanSetNode>();
    if (plan_node == nullptr) return block;
    VSimdGroupPlanSet plans = ffi::GetRef<VSimdGroupPlanSet>(plan_node);

    FragmentRegisterFile registers;
    if (plans->fragment_regs.has_value())
      registers = FragmentRegisterFile::Build((*plans->fragment_regs)->entries);

    VSimdFlattenedBody body = FlattenVSimdBody(block->body);
    std::map<int, Stmt> replacements;
    bool emit_failed = false;
    for (const VSimdGroupPlan &group : plans->groups) {
      if (group->status.operator std::string() != "PLANNED") continue;
      std::vector<int> spans;
      std::vector<Stmt> emitted;
      bool ok = true;
      for (const VSimdItemPlan &item : group->items) {
        if (item->span_begin < 0 ||
            item->span_begin >= static_cast<int>(body.stmts.size())) {
          ok = false;
          break;
        }
        try {
          emitted.push_back(EmitItem(item, &registers));
          spans.push_back(static_cast<int>(item->span_begin));
        } catch (const Unsupported &e) {
          LOG(WARNING) << "[vsimd] P3 rejected a group at span "
                       << item->span_begin << ": " << e.message();
          ok = false;
          break;
        }
      }
      if (!ok) {
        emit_failed = true;
        continue;  // whole-group rollback: keep every original statement
      }
      for (size_t i = 0; i < spans.size(); ++i)
        replacements[spans[i]] = emitted[i];
    }

    if (emit_failed && !registers.empty()) {
      // A rolled-back group may still use a register-managed fragment as
      // memory; keeping the register file would corrupt it. Fall back for the
      // whole VF.
      replacements.clear();
    }

    SBlockNode *writer = block.CopyOnWrite();
    if (!replacements.empty())
      writer->body = ReplaceVSimdSpans(writer->body, replacements);
    if (!registers.empty() && !replacements.empty())
      RewriteFragmentAllocations(writer, registers);
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
