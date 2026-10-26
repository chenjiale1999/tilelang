/*!
 * \file vsimd_analysis_common.h
 * \brief Shared read-only IR helpers for the P1 vsimd analyses (PR-A/PR-B).
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_VSIMD_ANALYSIS_COMMON_H_
#define TVM_TL_ASCEND_TRANSFORM_VSIMD_ANALYSIS_COMMON_H_

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace vsimd {

using namespace tirx;
using ffi::Array;

using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

inline void AddUndefinedVars(const PrimExpr &expr, VarSet *vars) {
  if (!expr.defined())
    return;
  for (const Var &v : tirx::UndefinedVars(expr))
    vars->insert(v);
}

/*! \brief One T.Parallel loop level: loop var and its iteration range. */
struct LoopInfo {
  Var var;
  PrimExpr min;
  PrimExpr extent;
};

/*!
 * \brief One buffer access inside a parallel nest.
 *
 * `value` is only meaningful for writes; it is held so a reference expander
 * can reproduce the stored expression without re-inspecting the IR.
 */
struct AccessInfo {
  Buffer buffer;
  Array<PrimExpr> indices;
  PrimExpr value;
  bool is_write{false};
};

/*! \brief A single top-level T.Parallel nest collected from a SIMD_VF. */
struct ParallelNest {
  std::vector<LoopInfo> loops;      // parallel domain, outer to inner
  std::vector<AccessInfo> accesses; // loads/stores in the nest
  std::vector<Buffer> local_views;  // DeclBuffer/AllocBuffer in the VF body
  bool unsupported{false};          // control flow / local.var / TileOp
  std::string reason;               // why unsupported
};

inline bool IsManualCall(const CallNode *call) {
  const auto *op = call->op.as<OpNode>();
  if (op == nullptr)
    return false;
  std::string name = op->name;
  return name.rfind("tl.simd.", 0) == 0 ||
         name.rfind("tl.ascend_nd2nz", 0) == 0;
}

class ManualDetector : public StmtExprVisitor {
public:
  bool manual{false};

  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      if (buffer.scope() == "local.var")
        manual = true;
    }
    for (const auto &match : op->match_buffers) {
      if (match->buffer.scope() == "local.var")
        manual = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (IsManualCall(op))
      manual = true;
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    if (op->buffer.scope() == "local.var")
      manual = true;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    if (op->buffer.scope() == "local.var")
      manual = true;
    StmtExprVisitor::VisitStmt_(op);
  }
};

class NestCollector : public StmtExprVisitor {
public:
  std::vector<ParallelNest> nests;
  bool unsupported{false};
  std::string unsupported_reason;

  explicit NestCollector(bool allow_control_flow = false)
      : allow_control_flow_(allow_control_flow) {}

  // Positive statement grammar for a liftable T.Parallel body. Unknown
  // statements are rejected instead of being silently ignored; this prevents
  // effects or declarations from disappearing during region expansion.
  // Future CF1-CF3 support must extend this allowlist explicitly.
  void VisitStmt(const Stmt &stmt) final {
    if (depth_ == 0) {
      StmtExprVisitor::VisitStmt(stmt);
      return;
    }
    if (stmt.as<BufferStoreNode>() || stmt.as<SeqStmtNode>() ||
        stmt.as<DeclBufferNode>() || stmt.as<BindNode>() ||
        stmt.as<EvaluateNode>()) {
      StmtExprVisitor::VisitStmt(stmt);
      return;
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      if (loop->kind == ForKind::kParallel ||
          (allow_control_flow_ && loop->kind == ForKind::kSerial)) {
        StmtExprVisitor::VisitStmt(stmt);
        return;
      }
    }
    if (allow_control_flow_ && stmt.as<IfThenElseNode>()) {
      StmtExprVisitor::VisitStmt(stmt);
      return;
    }
    unsupported = true;
    unsupported_reason =
        "unsupported statement inside T.Parallel: " + stmt->GetTypeKey();
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      bool outer = depth_ == 0;
      if (outer) {
        loops_.clear();
        accesses_.clear();
        local_views_.clear();
      }
      loops_.push_back(LoopInfo{op->loop_var, op->min, op->extent});
      ++depth_;
      StmtExprVisitor::VisitStmt_(op);
      --depth_;
      if (outer) {
        ParallelNest nest;
        nest.loops = loops_;
        nest.accesses = accesses_;
        nest.local_views = local_views_;
        nests.push_back(std::move(nest));
        loops_.clear();
        accesses_.clear();
        local_views_.clear();
      }
    } else {
      if (depth_ > 0 && !unsupported) {
        unsupported = true;
        unsupported_reason = "serial loop inside T.Parallel";
      }
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    if (depth_ > 0 && !unsupported) {
      unsupported = true;
      unsupported_reason = "if inside T.Parallel";
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileNode *op) final {
    if (depth_ > 0 && !unsupported) {
      unsupported = true;
      unsupported_reason = "while inside T.Parallel";
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (depth_ > 0) {
      if (op->buffer.scope() == "local.var") {
        unsupported = true;
        unsupported_reason = "local.var access inside T.Parallel";
        return;
      }
      AccessInfo info;
      info.buffer = op->buffer;
      info.indices = op->indices;
      info.value = op->value;
      info.is_write = true;
      accesses_.push_back(std::move(info));
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (depth_ > 0) {
      if (op->buffer.scope() == "local.var") {
        unsupported = true;
        unsupported_reason = "local.var access inside T.Parallel";
        return;
      }
      AccessInfo info;
      info.buffer = op->buffer;
      info.indices = op->indices;
      info.is_write = false;
      accesses_.push_back(std::move(info));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    local_views_.push_back(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    local_views_.push_back(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    // A bare Evaluate is an effectful/opaque statement. The region capture
    // only models values and BufferStores, so lifting it would silently drop
    // the effect. Reject it here instead of letting it pass A/B/C.
    if (depth_ > 0) {
      unsupported = true;
      unsupported_reason = "opaque Evaluate inside T.Parallel";
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (depth_ > 0 && !unsupported) {
      const auto *callee = op->op.as<OpNode>();
      if (callee != nullptr) {
        std::string name = callee->name;
        if (name.rfind("tl.tileop.", 0) == 0 ||
            name.rfind("tl.ascend_", 0) == 0) {
          unsupported = true;
          unsupported_reason = "TileOp/copy inside T.Parallel";
          return;
        }
        if (name == "tirx.call_extern" || name == "tirx.call_pure_extern") {
          unsupported = true;
          unsupported_reason = "opaque call_extern inside T.Parallel";
          return;
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

private:
  int depth_{0};
  bool allow_control_flow_{false};
  std::vector<LoopInfo> loops_;
  std::vector<AccessInfo> accesses_;
  std::vector<Buffer> local_views_;
};

class AssumptionFilter : public StmtExprVisitor {
public:
  bool ok{true};
  void VisitExpr_(const BufferLoadNode *) final { ok = false; }
  void VisitExpr_(const ProducerLoadNode *) final { ok = false; }
  void VisitExpr_(const CallNode *) final { ok = false; }
};

inline bool IsSupportedAssumption(const PrimExpr &cond) {
  if (!cond.defined() || !cond.dtype().is_bool())
    return false;
  if (tirx::SideEffect(cond) > tirx::CallEffectKind::kPure)
    return false;
  AssumptionFilter filter;
  filter(cond);
  return filter.ok;
}

} // namespace vsimd
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VSIMD_ANALYSIS_COMMON_H_
