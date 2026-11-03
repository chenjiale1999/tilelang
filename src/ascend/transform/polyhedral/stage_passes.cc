/*!
 * \file stage_passes.cc
 * \brief Staged P1/P2/P3 lowering of ``T.Parallel`` (region carrier -> plan -> TIR).
 */
#include "stage_passes.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "emit.h"
#include "layout/layout.h"
#include "model.h"
#include "plan.h"
#include "primitives.h"
#include "vsimd_p2_input.h"

namespace tvm {
namespace tl {
namespace polyhedral {

using tirx::AttrStmt;
using tirx::Buffer;
using tirx::ForKind;
using tirx::ForNode;
using tirx::SBlock;
using tirx::SBlockNode;
using tirx::Stmt;
using tirx::StmtExprVisitor;
using tirx::StmtMutator;

/*!
 * \brief FFI payload of ``kArrayRegionAttr``: the typed P1 region.
 *
 * The region stays a C++ struct (its access/value fields are not part of any
 * FFI contract); only the carrier has to be an Object so it can live in an
 * AttrStmt inside the IR between P1 and P2.
 */
class VSimdRegionRefNode : public ffi::Object {
 public:
  VSimdRegion region;

  static constexpr const bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.vsimd.Region", VSimdRegionRefNode, ffi::Object);

  static void RegisterReflection() {
    tvm::ffi::reflection::ObjectDef<VSimdRegionRefNode>();
  }
};

class VSimdRegionRef : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(VSimdRegionRef, ffi::ObjectRef,
                                             VSimdRegionRefNode);

  static VSimdRegionRef Make(const VSimdRegion& region) {
    auto node = ffi::make_object<VSimdRegionRefNode>();
    node->region = region;
    return VSimdRegionRef(std::move(node));
  }
};

namespace {

/*! \brief Outermost ``T.Parallel`` roots of one SIMD_VF body, in P1 nest order. */
class NestRootCollector final : public StmtExprVisitor {
 public:
  std::vector<Stmt> roots;

  void VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kParallel) {
      roots.push_back(ffi::GetRef<Stmt>(op));
      return;  // inner levels belong to the same nest
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

/*! \brief Whether an already-marked function is being lifted a second time. */
class MarkerFinder final : public StmtExprVisitor {
 public:
  bool found{false};

  void VisitStmt_(const tirx::AttrStmtNode* op) final {
    if (op->attr_key == kArrayRegionAttr || op->attr_key == kArrayPlanAttr) {
      found = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

/*! \brief Wrap the marked nest statements with their P1 region carrier. */
class RegionMarkerInserter final : public StmtMutator {
 public:
  explicit RegionMarkerInserter(std::unordered_map<const ForNode*, VSimdRegion> regions)
      : regions_(std::move(regions)) {}

  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kParallel) {
      Stmt self = ffi::GetRef<Stmt>(op);
      auto it = regions_.find(op);
      if (it == regions_.end()) return self;  // not a lifted root: keep verbatim
      return AttrStmt(VSimdRegionRef::Make(it->second), kArrayRegionAttr, Integer(1),
                      std::move(self));
    }
    return StmtMutator::VisitStmt_(op);
  }

 private:
  std::unordered_map<const ForNode*, VSimdRegion> regions_;
};

/*!
 * \brief P1 driver: attach a region carrier to every nest P1 lifted.
 *
 * ``BuildP2PlanInputs`` performs the whole P1 analysis (PR-A legality, PR-B
 * capture closure, PR-C/D lift + expander self-check) and returns one record per
 * nest in report order. This pass only maps those records back to the actual
 * nest statements (same traversal order) and wraps them.
 */
class VfRegionMarker final : public StmtMutator {
 public:
  explicit VfRegionMarker(std::vector<VSimdP2Input> inputs)
      : inputs_(std::move(inputs)) {}

  Stmt VisitStmt_(const SBlockNode* op) final {
    if (op->name_hint != "SIMD_VF") {
      return StmtMutator::VisitStmt_(op);
    }

    const int vf_index = vf_index_++;
    size_t begin = cursor_;
    while (cursor_ < inputs_.size() && inputs_[cursor_].vf_index == vf_index) {
      ++cursor_;
    }

    Stmt visited = StmtMutator::VisitStmt_(op);
    // An opaque VF carries a single record with ``nest_index == -1`` and no
    // region, so nothing is wrapped.
    if (cursor_ == begin || inputs_[begin].nest_index < 0) {
      return visited;
    }

    NestRootCollector collector;
    collector(op->body);
    std::unordered_map<const ForNode*, VSimdRegion> regions;
    for (size_t i = 0; i < cursor_ - begin && i < collector.roots.size(); ++i) {
      const VSimdP2Input& item = inputs_[begin + i];
      if (item.status != "LIFTED") continue;
      const auto* loop = collector.roots[i].as<ForNode>();
      if (loop != nullptr) regions.emplace(loop, item.region);
    }
    if (regions.empty()) {
      return visited;
    }

    SBlock block = Downcast<SBlock>(visited);
    SBlockNode* writer = block.CopyOnWrite();
    Stmt marked = RegionMarkerInserter(std::move(regions))(writer->body);
    writer->body = std::move(marked);
    return block;
  }

 private:
  std::vector<VSimdP2Input> inputs_;
  size_t cursor_{0};
  int vf_index_{0};
};

/*!
 * \brief P2 driver: plan the region carried by every marker.
 *
 * P2 consumes **only** the region field of the P1 lift (it never re-interprets
 * the nest body): the plan input is rebuilt from the region through
 * ``RegionToPlanInput``. A nest the planner rejects — or a region whose
 * expansion is not a single parallel chain (a P1 internal inconsistency, not
 * user IR) — is unwrapped exactly as it was, so the function body stays
 * byte-identical and the nest stays a serial loop.
 */
class PlanMarkerReplacer final : public StmtMutator {
 public:
  Stmt VisitStmt_(const SBlockNode* op) final {
    if (op->name_hint != "SIMD_VF") {
      return StmtMutator::VisitStmt_(op);
    }
    // The VF allocation scope is the G0 contract: only fragments allocated by
    // this block are registered; a fragment captured from outside is rejected.
    std::vector<Buffer> saved = std::move(vf_alloc_scope_);
    vf_alloc_scope_.assign(op->alloc_buffers.begin(), op->alloc_buffers.end());
    Stmt visited = StmtMutator::VisitStmt_(op);
    vf_alloc_scope_ = std::move(saved);
    return visited;
  }

  Stmt VisitStmt_(const tirx::AttrStmtNode* op) final {
    if (op->attr_key != kArrayRegionAttr) {
      return StmtMutator::VisitStmt_(op);
    }

    // The marker body is the original nest: it is the zero-change fallback.
    Stmt nest = StmtMutator::VisitStmt(op->body);
    const auto* carrier = op->node.as<VSimdRegionRefNode>();
    if (carrier != nullptr) {
      Stmt input;
      std::string error;
      if (RegionToPlanInput(carrier->region, &input, &error)) {
        // The rebuilt statement carries the stores but none of the region's
        // context; hand the typed fields over explicitly (layouts decide ND vs
        // NZ stores, the rest is the reserved analyzer context).
        PlanOptions options;
        options.layouts = carrier->region.layouts;
        options.assumptions = carrier->region.assumptions;
        options.bindings = carrier->region.bindings;
        options.shape_syms = carrier->region.shape_syms;
        options.vf_alloc_scope = vf_alloc_scope_;
        try {
          VectorProgram program = BuildPlan(input, std::nullopt, options);
          return AttrStmt(program, kArrayPlanAttr, Integer(1), std::move(nest));
        } catch (const Unsupported&) {
          // Branch B: P1 lifted the nest but P2 cannot plan it.
        }
      } else {
        LOG(WARNING) << "[vsimd] P1 region expansion failed (compiler bug): "
                     << error;
      }
    }
    return nest;
  }

 private:
  std::vector<Buffer> vf_alloc_scope_;
};

/* ------------------------------------------------------------------------- */
/* P2.5: VF-level fragment register planning (V1-V4)                         */
/* ------------------------------------------------------------------------- */

/*! \brief Whether a program contains at least one register-mode segment. */
bool ProgramHasFragmentRegs(const VectorProgram& program) {
  for (const VectorPlan& segment : program->segments) {
    if (!segment->fragment_regs.empty()) return true;
  }
  return false;
}

/*! \brief Collect the plan carriers of one VF body in source order. */
class PlanCarrierCollector final : public StmtExprVisitor {
 public:
  std::vector<VectorProgram> programs;

  void VisitStmt_(const tirx::AttrStmtNode* op) final {
    if (op->attr_key == kArrayPlanAttr) {
      if (const auto* program = op->node.as<VectorProgramNode>()) {
        programs.push_back(ffi::GetRef<VectorProgram>(program));
      }
      return;  // the marker body is the original nest (no nested plans)
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

/*! \brief Unwrap every carrier of one VF body (the atomic fallback). */
class PlanCarrierStripper final : public StmtMutator {
 public:
  Stmt VisitStmt_(const tirx::AttrStmtNode* op) final {
    if (op->attr_key == kArrayPlanAttr || op->attr_key == kArrayRegionAttr) {
      return StmtMutator::VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }
};

/*!
 * \brief V1 escape scan: every fragment access must live inside a register plan.
 *
 * A fragment used by ``T.copy`` / ``T.reduce_*`` / ``T.simd.*`` / a scalar
 * statement keeps its memory identity, which cannot coexist with the register
 * file; the whole VF then falls back to scalar loops.
 */
class FragmentEscapeScanner final : public StmtExprVisitor {
 public:
  explicit FragmentEscapeScanner(std::set<const tirx::VarNode*> managed)
      : managed_(std::move(managed)) {}

  bool escaped{false};

  void VisitStmt_(const tirx::AttrStmtNode* op) final {
    if (op->attr_key == kArrayPlanAttr) {
      const auto* program = op->node.as<VectorProgramNode>();
      if (program != nullptr &&
          ProgramHasFragmentRegs(ffi::GetRef<VectorProgram>(program))) {
        return;  // P2 planned the whole nest: nothing escapes here
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const tirx::BufferLoadNode* op) final {
    if (managed_.count(op->buffer->data.get()) != 0) escaped = true;
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tirx::BufferStoreNode* op) final {
    if (managed_.count(op->buffer->data.get()) != 0) escaped = true;
    StmtExprVisitor::VisitStmt_(op);
  }

 private:
  std::set<const tirx::VarNode*> managed_;
};

/*! \brief Chunk slots touched by one fragment access. */
bool FragmentSlots(const AccessInfo& access, const VectorPlan& plan,
                   std::vector<int64_t>* slots, std::string* error) {
  if (!plan->chunks.has_value() || *plan->chunks <= 0) {
    *error = "'" + access->buffer->name.operator std::string() +
             "' has a dynamic chunk count";
    return false;
  }
  arith::Analyzer analyzer;
  PrimExpr base_expr = tirx::Substitute(
      access->offset, [&](const tirx::Var& var) -> ffi::Optional<PrimExpr> {
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
bool ValueReadsRegister(const ValueNode& node,
                        std::vector<AccessInfo>* reads) {
  if (!node.defined()) return true;
  if (node->recipe.operator std::string() == "reg_load") {
    reads->push_back(*node->access);
  }
  for (const ffi::ObjectRef& arg : node->args) {
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
  using Key = std::pair<const tirx::VarNode*, int64_t>;

  bool CheckRead(const AccessInfo& access, const VectorPlan& plan,
                 std::string* error) {
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

  bool CheckStoreReads(const StorePlan& store, const VectorPlan& plan,
                       std::string* error) {
    std::vector<AccessInfo> reads;
    if (!ValueReadsRegister(store->value, &reads)) return false;
    if (store->mask.has_value() && !ValueReadsRegister(*store->mask, &reads)) {
      return false;
    }
    if (store->index.has_value() && !ValueReadsRegister(*store->index, &reads)) {
      return false;
    }
    for (const AccessInfo& access : reads) {
      if (!CheckRead(access, plan, error)) return false;
    }
    return true;
  }

  bool ApplyWrite(const StorePlan& store, const VectorPlan& plan,
                  std::string* error) {
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

/*!
 * \brief P2.5 driver: plan the register file of every ``SIMD_VF`` block.
 *
 * All checks are VF-atomic: a single escape, inconsistency, budget overflow or
 * lifecycle error drops every plan carrier of the block, so no nest of the VF
 * is half-register / half-memory.
 */
class FragmentRegisterPlanner final : public StmtMutator {
 public:
  Stmt VisitStmt_(const SBlockNode* op) final {
    if (op->name_hint != "SIMD_VF") {
      return StmtMutator::VisitStmt_(op);
    }
    SBlock block = Downcast<SBlock>(StmtMutator::VisitStmt_(op));
    return PlanBlock(block);
  }

 private:
  static SBlock Reject(SBlock block, const std::string& reason) {
    LOG(WARNING) << "[vsimd] fragment registers disabled for a SIMD_VF block: "
                 << reason;
    SBlockNode* writer = block.CopyOnWrite();
    writer->body = PlanCarrierStripper()(writer->body);
    if (writer->annotations.count(kFragmentRegsAttr) != 0) {
      writer->annotations.erase(kFragmentRegsAttr);
    }
    return block;
  }

  static SBlock PlanBlock(const SBlock& block) {
    PlanCarrierCollector collector;
    collector(block->body);
    bool any = false;
    for (const VectorProgram& program : collector.programs) {
      any = any || ProgramHasFragmentRegs(program);
    }
    if (!any) return block;

    // Managed fragment data Vars (the V1 scan set).
    std::set<const tirx::VarNode*> managed;
    for (const VectorProgram& program : collector.programs) {
      for (const VectorPlan& segment : program->segments) {
        for (const FragmentReg& reg : segment->fragment_regs) {
          managed.insert(reg->data.get());
        }
      }
    }

    // V1: no fragment access may escape its register plan.
    FragmentEscapeScanner scanner(managed);
    scanner(block->body);
    if (scanner.escaped) {
      return Reject(block,
                    "a fragment is also used outside its register plan "
                    "(T.copy / T.reduce_* / T.simd.* / scalar statement); "
                    "stage it through UB first");
    }

    // V3: one register shape per data Var (same name different data Var is
    // two independent entries; the identity key never merges them).
    std::vector<FragmentReg> entries;
    std::map<const tirx::VarNode*, size_t> index;
    for (const VectorProgram& program : collector.programs) {
      for (const VectorPlan& segment : program->segments) {
        for (const FragmentReg& reg : segment->fragment_regs) {
          auto it = index.find(reg->data.get());
          if (it == index.end()) {
            index.emplace(reg->data.get(), entries.size());
            entries.push_back(reg);
            continue;
          }
          const FragmentReg& seen = entries[it->second];
          if (seen->slots != reg->slots || seen->lanes != reg->lanes ||
              seen->dtype.operator std::string() != reg->dtype.operator std::string()) {
            return Reject(block, "inconsistent register shape for fragment '" +
                                     reg->name.operator std::string() + "'");
          }
        }
      }
    }

    // V2: pinned manual slots + the automatic-temporary reserve.
    int64_t total = 0;
    std::string detail;
    for (const FragmentReg& reg : entries) {
      total += reg->slots;
      detail += (detail.empty() ? "" : ", ") +
                reg->name.operator std::string() + "=" +
                std::to_string(reg->slots);
    }
    if (total > kManualVRegPool || total + kAutoTempVRegReserve > kMaxVRegsPerVf) {
      return Reject(block, std::to_string(total) +
                               " fragment registers required (" + detail +
                               "), the manual budget is " +
                               std::to_string(kManualVRegPool));
    }

    // V4: program-order lifecycle across the nests of this VF.
    FragmentLifecycle lifecycle;
    for (const VectorProgram& program : collector.programs) {
      for (const VectorPlan& segment : program->segments) {
        std::string error;
        auto process = [&](const StorePlan& store) -> bool {
          if (!lifecycle.CheckStoreReads(store, segment, &error)) return false;
          if (store->storage.operator std::string() == "register" &&
              !lifecycle.ApplyWrite(store, segment, &error)) {
            return false;
          }
          return true;
        };
        for (const StorePlan& store : segment->pre_stores) {
          if (!process(store)) return Reject(block, error);
        }
        for (const StorePlan& store : segment->stores) {
          if (!process(store)) return Reject(block, error);
        }
        for (const StorePlan& store : segment->post_stores) {
          if (!process(store)) return Reject(block, error);
        }
      }
    }

    SBlock result = block;
    SBlockNode* writer = result.CopyOnWrite();
    writer->annotations.Set(kFragmentRegsAttr,
                            MakeFragmentRegisterMap(std::move(entries)));
    return result;
  }
};

/*! \brief P3 driver: emit every plan marker, unwrap anything left over. */
class PlanMarkerLower final : public StmtMutator {
 public:
  Stmt VisitStmt_(const SBlockNode* op) final {
    if (op->name_hint != "SIMD_VF") {
      return StmtMutator::VisitStmt_(op);
    }
    // Enter the VF: build its register file from the P2.5 annotation.
    FragmentRegisterFile saved = std::move(registers_);
    const bool saved_inside = inside_vf_;
    const bool saved_active = registers_active_;
    registers_ = FragmentRegisterFile();
    registers_active_ = false;
    inside_vf_ = true;
    if (op->annotations.count(kFragmentRegsAttr) != 0) {
      const auto* map =
          op->annotations.at(kFragmentRegsAttr).as<FragmentRegisterMapNode>();
      if (map != nullptr) {
        registers_ =
            FragmentRegisterFile::Build(ffi::GetRef<FragmentRegisterMap>(map)->entries);
        registers_active_ = !registers_.empty();
      }
    }
    Stmt visited = StmtMutator::VisitStmt_(op);
    FragmentRegisterFile done = std::move(registers_);
    registers_ = std::move(saved);
    registers_active_ = saved_active;
    inside_vf_ = saved_inside;

    if (done.empty()) return visited;
    SBlock block = Downcast<SBlock>(visited);
    SBlockNode* writer = block.CopyOnWrite();
    // The fragment allocations become register declarations; their layout-map
    // entries must go too so AscendLowerTileOp never remaps them.
    ffi::Array<Buffer> kept;
    for (const Buffer& buffer : writer->alloc_buffers) {
      if (done.Find(buffer->data) != nullptr) continue;
      kept.push_back(buffer);
    }
    for (const Buffer& reg : done.Buffers()) kept.push_back(reg);
    writer->alloc_buffers = std::move(kept);
    if (writer->annotations.count(attr::kLayoutMap) != 0) {
      ffi::Map<Buffer, Layout> old =
          Downcast<ffi::Map<Buffer, Layout>>(writer->annotations.at(attr::kLayoutMap));
      ffi::Map<Buffer, Layout> remapped;
      for (const auto& entry : old) {
        if (done.Find(entry.first->data) != nullptr) continue;
        remapped.Set(entry.first, entry.second);
      }
      if (remapped.empty()) {
        writer->annotations.erase(attr::kLayoutMap);
      } else {
        writer->annotations.Set(attr::kLayoutMap, remapped);
      }
    }
    return block;
  }

  Stmt VisitStmt_(const tirx::AttrStmtNode* op) final {
    if (op->attr_key == kArrayPlanAttr) {
      const auto* program = op->node.as<VectorProgramNode>();
      if (program != nullptr) {
        FragmentRegisterFile* file = registers_active_ ? &registers_ : nullptr;
        // Inside a VF, a register-mode plan without a VF file is a P2.5
        // inconsistency; passing the (empty) file makes EmitPlan reject it so
        // the nest falls back to scalar instead of losing cross-nest values.
        if (inside_vf_ && !registers_active_) file = &registers_;
        try {
          return EmitPlan(ffi::GetRef<VectorProgram>(program), file);
        } catch (const Unsupported& error) {
          LOG(WARNING) << "[vsimd] P3 rejected a planned nest: " << error.message();
        }
      }
      return StmtMutator::VisitStmt(op->body);
    }
    if (op->attr_key == kArrayRegionAttr) {
      // P2 did not run: never leak the P1 plumbing into the emitted IR.
      return StmtMutator::VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }

 private:
  FragmentRegisterFile registers_;
  bool registers_active_{false};
  bool inside_vf_{false};
};

Stmt RewriteBody(const Stmt& body, StmtMutator* mutator) { return (*mutator)(body); }

}  // namespace

tirx::PrimFunc LiftParallelToArray(tirx::PrimFunc func) {
  // ``BuildP2PlanInputs`` *is* the P1 lift: per nest it runs PR-A memory
  // legality, PR-B capture closure and ``BuildMinimalRegion``, which returns
  // ``VSimdRegionLift{lifted, internal_error, reason, region}`` after checking
  // its own reference expansion. Those four fields are encoded as ``status``
  // (LIFTED / INTERNAL_ERROR / FALLBACK / OPAQUE); only ``lifted == true``
  // nests get a carrier, so P2/P3 only ever see lifted regions.
  std::vector<VSimdP2Input> inputs = BuildP2PlanInputs(func);
  if (inputs.empty()) return func;

  MarkerFinder finder;
  finder(func->body);
  if (finder.found) return func;  // already lifted; never stack carriers

  VfRegionMarker marker(std::move(inputs));
  tirx::PrimFuncNode* writer = func.CopyOnWrite();
  writer->body = RewriteBody(std::move(writer->body), &marker);
  return func;
}

tirx::PrimFunc PlanArray(tirx::PrimFunc func) {
  PlanMarkerReplacer planner;
  tirx::PrimFuncNode* writer = func.CopyOnWrite();
  writer->body = RewriteBody(std::move(writer->body), &planner);
  return func;
}

tirx::PrimFunc PlanFragmentRegisters(tirx::PrimFunc func) {
  FragmentRegisterPlanner planner;
  tirx::PrimFuncNode* writer = func.CopyOnWrite();
  writer->body = RewriteBody(std::move(writer->body), &planner);
  return func;
}

tirx::PrimFunc LowerArrayPlan(tirx::PrimFunc func) {
  PlanMarkerLower emitter;
  tirx::PrimFuncNode* writer = func.CopyOnWrite();
  writer->body = RewriteBody(std::move(writer->body), &emitter);
  return func;
}

tirx::PrimFunc VectorizeParallelNests(tirx::PrimFunc func) {
  func = LiftParallelToArray(std::move(func));
  func = PlanArray(std::move(func));
  func = PlanFragmentRegisters(std::move(func));
  return LowerArrayPlan(std::move(func));
}

TVM_FFI_STATIC_INIT_BLOCK() { VSimdRegionRefNode::RegisterReflection(); }

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm
