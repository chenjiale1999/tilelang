/*!
 * \file fragment_plan.cc
 * \brief VF-level fragment register validation (V1 escape, V2 budget,
 *        V3 consistency, V4 lifecycle).
 */
#include "fragment_plan.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/stmt_functor.h>

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../vsimd_handoff.h"
#include "primitives.h"

namespace tvm {
namespace tl {
namespace polyhedral {

using namespace tirx;

namespace {

/*! \brief Whether a program contains at least one register-mode segment. */
bool ProgramHasFragmentRegs(const VectorProgram &program) {
  for (const VectorPlan &segment : program->segments) {
    if (!segment->fragment_regs.empty()) return true;
  }
  return false;
}

/*!
 * \brief V1 escape scan: every fragment access must live inside a planned item.
 *
 * A fragment used by an unplanned ``T.copy`` / ``T.reduce_*`` / ``T.simd.*`` /
 * scalar statement keeps its memory identity, which cannot coexist with the
 * register file; the whole VF then falls back to scalar loops.
 */
class FragmentEscapeScanner final : public StmtExprVisitor {
 public:
  explicit FragmentEscapeScanner(std::set<const VarNode *> managed)
      : managed_(std::move(managed)) {}

  bool escaped{false};

  void VisitExpr_(const BufferLoadNode *op) final {
    if (managed_.count(op->buffer->data.get()) != 0) escaped = true;
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (managed_.count(op->buffer->data.get()) != 0) escaped = true;
    StmtExprVisitor::VisitStmt_(op);
  }

 private:
  std::set<const VarNode *> managed_;
};

/*! \brief Chunk slots touched by one fragment access. */
bool FragmentSlots(const AccessInfo &access, const VectorPlan &plan,
                   std::vector<int64_t> *slots, std::string *error) {
  if (!plan->chunks.has_value() || *plan->chunks <= 0) {
    *error = "'" + access->buffer->name.operator std::string() +
             "' has a dynamic chunk count";
    return false;
  }
  arith::Analyzer analyzer;
  PrimExpr base_expr = tirx::Substitute(
      access->offset, [&](const Var &var) -> ffi::Optional<PrimExpr> {
        if (var.same_as(plan->vector_axis->var)) {
          return ffi::Optional<PrimExpr>(IntImm32(0));
        }
        return ffi::Optional<PrimExpr>();
      });
  std::optional<int64_t> base = ConstInt(analyzer.Simplify(base_expr), &analyzer);
  if (!base.has_value() || *base % plan->lanes != 0) {
    *error = "'" + access->buffer->name.operator std::string() +
             "' has a non-constant or unaligned register slot";
    return false;
  }
  int64_t base_slot = *base / plan->lanes;
  for (int64_t chunk = 0; chunk < *plan->chunks; ++chunk) {
    slots->push_back(base_slot + chunk);
  }
  return true;
}

/*! \brief Whether a value DAG reads a fragment register. */
bool ValueReadsRegister(const ValueNode &node,
                        std::vector<AccessInfo> *reads) {
  if (!node.defined()) return true;
  if (node->recipe.operator std::string() == "reg_load") {
    reads->push_back(*node->access);
  }
  for (const ffi::ObjectRef &arg : node->args) {
    if (!ValueReadsRegister(Downcast<ValueNode>(arg), reads)) return false;
  }
  if (node->predicate.has_value() &&
      !ValueReadsRegister(Downcast<ValueNode>(*node->predicate), reads)) {
    return false;
  }
  return true;
}

/*! \brief V4: per-VF ``init[data][slot]`` state machine. */
class FragmentLifecycle {
 public:
  using Key = std::pair<const VarNode *, int64_t>;

  bool CheckRead(const AccessInfo &access, const VectorPlan &plan,
                 std::string *error) {
    std::vector<int64_t> slots;
    if (!FragmentSlots(access, plan, &slots, error)) return false;
    for (int64_t slot : slots) {
      auto it = defined_.find(Key(access->buffer->data.get(), slot));
      if (it == defined_.end() || !it->second) {
        *error = "read of uninitialized chunk " + std::to_string(slot) +
                 " of '" + access->buffer->name.operator std::string() + "'";
        return false;
      }
    }
    return true;
  }

  bool CheckStoreReads(const StorePlan &store, const VectorPlan &plan,
                       std::string *error) {
    std::vector<AccessInfo> reads;
    if (!ValueReadsRegister(store->value, &reads)) return false;
    if (store->mask.has_value() && !ValueReadsRegister(*store->mask, &reads)) {
      return false;
    }
    if (store->index.has_value() && !ValueReadsRegister(*store->index, &reads)) {
      return false;
    }
    for (const AccessInfo &access : reads) {
      if (!CheckRead(access, plan, error)) return false;
    }
    return true;
  }

  bool ApplyWrite(const StorePlan &store, const VectorPlan &plan,
                  std::string *error) {
    std::vector<int64_t> slots;
    if (!FragmentSlots(store->access, plan, &slots, error)) return false;
    const bool masked = store->mask.has_value();
    for (int64_t slot : slots) {
      Key key(store->spec->buffer->data.get(), slot);
      auto it = defined_.find(key);
      const bool defined = it != defined_.end() && it->second;
      if (masked && !defined) {
        *error = "partial write to chunk " + std::to_string(slot) + " of '" +
                 store->spec->buffer->name.operator std::string() +
                 "' before it is initialized";
        return false;
      }
      defined_[key] = true;
    }
    return true;
  }

 private:
  std::map<Key, bool> defined_;
};

}  // namespace

VfFragmentPlan PlanVfFragmentRegisters(
    const Stmt &body, const std::vector<VectorProgram> &programs,
    const std::vector<int> &planned_spans) {
  VfFragmentPlan result;
  bool any = false;
  for (const VectorProgram &program : programs) {
    any = any || ProgramHasFragmentRegs(program);
  }
  if (!any) return result;

  // Managed fragment data Vars (the V1 scan set).
  std::set<const VarNode *> managed;
  for (const VectorProgram &program : programs) {
    for (const VectorPlan &segment : program->segments) {
      for (const FragmentReg &reg : segment->fragment_regs) {
        managed.insert(reg->data.get());
      }
    }
  }

  // V1: no fragment access may escape its register plan. Planned spans are
  // replaced by their emitted form, so only the other canonical items are
  // scanned.
  std::set<int> planned(planned_spans.begin(), planned_spans.end());
  VSimdFlattenedBody flat = FlattenVSimdBody(body);
  FragmentEscapeScanner scanner(managed);
  for (size_t i = 0; i < flat.stmts.size(); ++i) {
    if (planned.count(static_cast<int>(i)) != 0) continue;
    scanner(flat.stmts[i]);
    if (scanner.escaped) {
      result.reason =
          "a fragment is also used outside its register plan "
          "(T.copy / T.reduce_* / T.simd.* / scalar statement); stage it "
          "through UB first";
      return result;
    }
  }

  // V3: one register shape per data Var (same name different data Var is two
  // independent entries; the identity key never merges them).
  std::vector<FragmentReg> entries;
  std::map<const VarNode *, size_t> index;
  for (const VectorProgram &program : programs) {
    for (const VectorPlan &segment : program->segments) {
      for (const FragmentReg &reg : segment->fragment_regs) {
        auto it = index.find(reg->data.get());
        if (it == index.end()) {
          index.emplace(reg->data.get(), entries.size());
          entries.push_back(reg);
          continue;
        }
        const FragmentReg &seen = entries[it->second];
        if (seen->slots != reg->slots || seen->lanes != reg->lanes ||
            seen->dtype.operator std::string() !=
                reg->dtype.operator std::string()) {
          result.reason = "inconsistent register shape for fragment '" +
                          reg->name.operator std::string() + "'";
          return result;
        }
      }
    }
  }

  // V2: pinned manual slots + the automatic-temporary reserve.
  int64_t total = 0;
  std::string detail;
  for (const FragmentReg &reg : entries) {
    total += reg->slots;
    detail += (detail.empty() ? "" : ", ") + reg->name.operator std::string() +
              "=" + std::to_string(reg->slots);
  }
  if (total > kManualVRegPool || total + kAutoTempVRegReserve > kMaxVRegsPerVf) {
    result.reason = std::to_string(total) + " fragment registers required (" +
                    detail + "), the manual budget is " +
                    std::to_string(kManualVRegPool);
    return result;
  }

  // V4: program-order lifecycle across the nests of this VF.
  FragmentLifecycle lifecycle;
  for (const VectorProgram &program : programs) {
    for (const VectorPlan &segment : program->segments) {
      std::string error;
      auto process = [&](const StorePlan &store) -> bool {
        if (!lifecycle.CheckStoreReads(store, segment, &error)) return false;
        if (store->storage.operator std::string() == "register" &&
            !lifecycle.ApplyWrite(store, segment, &error)) {
          return false;
        }
        return true;
      };
      for (const StorePlan &store : segment->pre_stores) {
        if (!process(store)) {
          result.reason = error;
          return result;
        }
      }
      for (const StorePlan &store : segment->stores) {
        if (!process(store)) {
          result.reason = error;
          return result;
        }
      }
      for (const StorePlan &store : segment->post_stores) {
        if (!process(store)) {
          result.reason = error;
          return result;
        }
      }
    }
  }

  result.enabled = true;
  result.map = MakeFragmentRegisterMap(std::move(entries));
  return result;
}

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm
