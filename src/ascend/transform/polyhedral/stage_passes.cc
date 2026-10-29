/*!
 * \file stage_passes.cc
 * \brief Staged P1/P2/P3 lowering of ``T.Parallel`` (region carrier -> plan -> TIR).
 */
#include "stage_passes.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "emit.h"
#include "plan.h"
#include "primitives.h"
#include "vsimd_p2_input.h"

namespace tvm {
namespace tl {
namespace polyhedral {

using tirx::AttrStmt;
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
 * user IR) — is unwrapped exactly as it was, so the legacy lowering still owns
 * it and the function body stays byte-identical.
 */
class PlanMarkerReplacer final : public StmtMutator {
 public:
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
        try {
          VectorProgram program = BuildPlan(input);
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
};

/*! \brief P3 driver: emit every plan marker, unwrap anything left over. */
class PlanMarkerLower final : public StmtMutator {
 public:
  Stmt VisitStmt_(const tirx::AttrStmtNode* op) final {
    if (op->attr_key == kArrayPlanAttr) {
      const auto* program = op->node.as<VectorProgramNode>();
      if (program != nullptr) {
        try {
          return EmitPlan(ffi::GetRef<VectorProgram>(program));
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

tirx::PrimFunc LowerArrayPlan(tirx::PrimFunc func) {
  PlanMarkerLower emitter;
  tirx::PrimFuncNode* writer = func.CopyOnWrite();
  writer->body = RewriteBody(std::move(writer->body), &emitter);
  return func;
}

tirx::PrimFunc VectorizeParallelNests(tirx::PrimFunc func) {
  func = LiftParallelToArray(std::move(func));
  func = PlanArray(std::move(func));
  return LowerArrayPlan(std::move(func));
}

TVM_FFI_STATIC_INIT_BLOCK() { VSimdRegionRefNode::RegisterReflection(); }

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm
