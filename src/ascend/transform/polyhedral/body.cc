/*!
 * \file body.cc
 * \brief Body normalization implementation.
 */
#include "body.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "primitives.h"

namespace tvm {
namespace tl {
namespace polyhedral {

namespace {

/*! \brief Literal zero only: statement normalization has no analyzer. */
bool IsZeroLiteral(const PrimExpr& expr) {
  if (const auto* int_imm = expr.as<IntImmNode>()) return int_imm->value == 0;
  return false;
}

/*! \brief ``else`` that carries no work: absent, or ``Evaluate(0)``. */
bool IsTruthyElse(const ffi::Optional<tirx::Stmt>& else_case) {
  if (!else_case.has_value()) return true;
  const tirx::Stmt& stmt = *else_case;
  const auto* evaluate = stmt.as<tirx::EvaluateNode>();
  return evaluate != nullptr && IsZeroLiteral(evaluate->value);
}

}  // namespace

StoreSpec MakeStoreSpec(tirx::Buffer buffer, ffi::Array<PrimExpr> indices,
                        PrimExpr value, ffi::Optional<PrimExpr> mask) {
  auto node = ffi::make_object<StoreSpecNode>();
  node->buffer = std::move(buffer);
  node->indices = std::move(indices);
  node->value = std::move(value);
  node->mask = std::move(mask);
  return StoreSpec(std::move(node));
}

AliasMap CollectAliases(const tirx::Stmt& stmt) {
  AliasMap aliases;
  tirx::PostOrderVisit(stmt, [&](const ffi::ObjectRef& node) {
    const auto* bind = node.as<tirx::BindNode>();
    if (bind == nullptr) return;
    // Buffer declarations / type constructors are Calls, not value aliases.
    if (bind->value.as<tirx::CallNode>() != nullptr) return;
    aliases.values[bind->var.get()] = bind->value;
  });
  return aliases;
}

PrimExpr ApplyAliases(const PrimExpr& expr, const AliasMap& aliases) {
  if (aliases.empty()) return expr;
  PrimExpr current = expr;
  for (int pass = 0; pass < 8; ++pass) {
    PrimExpr replaced = tirx::Substitute(
        current, [&](const tirx::Var& var) -> ffi::Optional<PrimExpr> {
          auto it = aliases.values.find(var.get());
          if (it == aliases.values.end()) return ffi::Optional<PrimExpr>();
          return ffi::Optional<PrimExpr>(it->second);
        });
    if (replaced.same_as(current)) break;
    current = replaced;
  }
  return current;
}

namespace {

/* ------------------------------------------------------------------------- */
/* nested tile-op expansion                                                  */
/* ------------------------------------------------------------------------- */

/*! \brief One ``tl.region`` argument: buffer, base indices and extents. */
struct TileOpRegion {
  tirx::Buffer buffer;
  ffi::Array<PrimExpr> indices;
  ffi::Array<PrimExpr> extents;
};

bool IsSupportedTileOp(const tirx::CallNode* call) {
  if (call == nullptr) return false;
  const auto* op = call->op.as<OpNode>();
  if (op == nullptr) return false;
  return op->name == "tl.tileop.copy" || op->name == "tl.tileop.ascend_copy" ||
         op->name == "tl.tileop.reduce";
}

bool ContainsTileOpCall(const tirx::Stmt& stmt) {
  bool found = false;
  tirx::PostOrderVisit(stmt, [&](const ffi::ObjectRef& node) {
    if (!found) {
      if (const auto* call = node.as<tirx::CallNode>())
        found = IsSupportedTileOp(call);
    }
  });
  return found;
}

bool DecodeTileOpRegion(const PrimExpr& expr, TileOpRegion* out) {
  const auto* call = expr.as<tirx::CallNode>();
  if (call == nullptr || call->args.size() < 3) return false;
  const auto* op = call->op.as<OpNode>();
  if (op == nullptr || op->name != "tl.region") return false;
  const auto* load = call->args[0].as<tirx::BufferLoadNode>();
  if (load == nullptr) return false;
  out->buffer = load->buffer;
  out->indices = load->indices;
  std::vector<PrimExpr> extents;
  for (size_t i = 2; i < call->args.size(); ++i) extents.push_back(call->args[i]);
  out->extents = ffi::Array<PrimExpr>(extents.begin(), extents.end());
  return out->indices.size() == out->extents.size();
}

bool IsUbScope(const tirx::Buffer& buffer) {
  std::string scope = buffer.scope().operator std::string();
  return scope == "shared" || scope == "shared.dyn" || scope == "SHARED";
}

tirx::Var MakeLoopVar(const char* prefix, int index) {
  return tirx::Var(std::string(prefix) + std::to_string(index), DataType::Int(32));
}

PrimExpr AddLoop(const PrimExpr& base, const tirx::Var& var) {
  return base + tirx::Cast(base.dtype(), var);
}

tirx::Stmt BuildCopyNestFromRegions(const TileOpRegion& src, const TileOpRegion& dst,
                                    arith::Analyzer* analyzer) {
  const int rank = static_cast<int>(dst.extents.size());
  if (rank == 0) throw Unsupported("a rank-0 copy has no parallel domain");
  if (static_cast<int>(src.extents.size()) != rank)
    throw Unsupported("copy src/dst ranks differ (" +
                      std::to_string(src.extents.size()) + " vs " +
                      std::to_string(rank) + ")");
  if (!IsUbScope(src.buffer))
    throw Unsupported("copy source '" + src.buffer->name.operator std::string() +
                      "' is not UB (scope " + src.buffer.scope().operator std::string() + ")");
  if (!IsUbScope(dst.buffer))
    throw Unsupported("copy destination '" + dst.buffer->name.operator std::string() +
                      "' is not UB (scope " + dst.buffer.scope().operator std::string() + ")");

  std::vector<tirx::Var> vars;
  std::vector<bool> looped;
  std::vector<PrimExpr> dst_indices;
  std::vector<PrimExpr> src_indices;
  for (int d = 0; d < rank; ++d) {
    if (!analyzer->CanProveEqual(src.extents[d], dst.extents[d]) &&
        !tirx::is_one(src.extents[d])) {
      throw Unsupported("copy dimension " + std::to_string(d) +
                        " has unequal source/destination extents "
                        "(only equal or broadcast is supported)");
    }
    vars.push_back(MakeLoopVar("tile_cp_i", d));
    // An extent-1 tile dimension is not a loop: keeping it would leave a
    // ``range(1)`` in the IR and block the flatten channel (P11).
    looped.push_back(!tirx::is_one(dst.extents[d]));
    const bool broadcast = tirx::is_one(src.extents[d]);
    dst_indices.push_back(looped[d] ? AddLoop(dst.indices[d], vars[d]) : dst.indices[d]);
    src_indices.push_back(
        broadcast || !looped[d] ? src.indices[d] : AddLoop(src.indices[d], vars[d]));
  }
  tirx::Stmt body = tirx::BufferStore(dst.buffer, tirx::BufferLoad(src.buffer, src_indices),
                                      dst_indices);
  for (int d = rank - 1; d >= 0; --d) {
    if (!looped[d]) continue;
    body = tirx::For(vars[d], IntImm32(0), dst.extents[d], tirx::ForKind::kParallel, body);
  }
  return body;
}

struct ReduceDimMap {
  bool kept_dim{false};
  std::vector<int> src_to_dst;
};

ReduceDimMap MapReduceDims(const TileOpRegion& src, const TileOpRegion& dst, int64_t dim,
                           arith::Analyzer* analyzer) {
  const int src_rank = static_cast<int>(src.extents.size());
  const int dst_rank = static_cast<int>(dst.extents.size());
  if (dim < 0 || dim >= src_rank) {
    throw Unsupported("reduce dim " + std::to_string(dim) +
                      " is outside the source rank " + std::to_string(src_rank));
  }
  ReduceDimMap map;
  if (dst_rank == src_rank) {
    if (!tirx::is_one(dst.extents[dim])) {
      throw Unsupported(
          "reduce destination keeps the reduced dimension but its extent is not 1");
    }
    map.kept_dim = true;
    map.src_to_dst.resize(src_rank);
    for (int t = 0; t < src_rank; ++t) map.src_to_dst[t] = t;
  } else if (dst_rank + 1 == src_rank) {
    map.kept_dim = false;
    map.src_to_dst.assign(src_rank, -1);
    for (int t = 0; t < src_rank; ++t) {
      if (t < dim) map.src_to_dst[t] = t;
      if (t > dim) map.src_to_dst[t] = t - 1;
    }
  } else {
    throw Unsupported("reduce source/destination ranks are incompatible (" +
                      std::to_string(src_rank) + " -> " + std::to_string(dst_rank) + ")");
  }
  for (int t = 0; t < src_rank; ++t) {
    if (t == dim) continue;
    const int d = map.src_to_dst[t];
    if (d < 0 || d >= dst_rank) {
      throw Unsupported("reduce destination dimension has no source dimension");
    }
    if (!analyzer->CanProveEqual(src.extents[t], dst.extents[d]) &&
        !tirx::is_one(src.extents[t])) {
      throw Unsupported("reduce dimension " + std::to_string(t) +
                        " does not match its destination extent");
    }
  }
  return map;
}

tirx::Stmt BuildReduceNestFromRegions(const TileOpRegion& src, const TileOpRegion& dst,
                                      const std::string& kind, int64_t dim, bool clear,
                                      arith::Analyzer* analyzer) {
  if (!IsUbScope(src.buffer) || !IsUbScope(dst.buffer)) {
    throw Unsupported("reduce operands must both be UB ('" +
                      src.buffer->name.operator std::string() + "' / '" +
                      dst.buffer->name.operator std::string() + "')");
  }
  if (src.extents.empty() || dst.extents.empty())
    throw Unsupported("reduce payload has an empty source or destination region");
  std::string recipe;
  if (kind == "max") {
    recipe = "vcmax";
  } else if (kind == "min") {
    recipe = "vcmin";
  } else if (kind == "sum") {
    recipe = "vcadd";
  } else {
    throw Unsupported("reduce kind '" + kind + "' is not supported (max/sum/min)");
  }

  const ReduceDimMap map = MapReduceDims(src, dst, dim, analyzer);
  const int src_rank = static_cast<int>(src.extents.size());
  std::vector<tirx::Var> vars;
  std::vector<bool> looped;
  std::vector<PrimExpr> src_indices;
  for (int t = 0; t < src_rank; ++t) {
    vars.push_back(MakeLoopVar("tile_rd_i", t));
    // An extent-1 tile dimension is not a loop (P11).
    looped.push_back(!tirx::is_one(src.extents[t]));
    src_indices.push_back(looped[t] ? AddLoop(src.indices[t], vars[t]) : src.indices[t]);
  }

  std::vector<PrimExpr> dst_indices;
  const int dst_rank = static_cast<int>(dst.extents.size());
  for (int d = 0; d < dst_rank; ++d) {
    int src_dim = -1;
    for (int t = 0; t < src_rank; ++t) {
      if (map.src_to_dst[t] == d) {
        src_dim = t;
        break;
      }
    }
    if (src_dim < 0) throw Unsupported("reduce destination dimension has no source dimension");
    const bool reduced = map.kept_dim && src_dim == dim;
    dst_indices.push_back(reduced || !looped[src_dim] ? dst.indices[d]
                                                      : AddLoop(dst.indices[d], vars[src_dim]));
  }

  tirx::BufferLoad old_value(dst.buffer, dst_indices);
  tirx::BufferLoad new_value(src.buffer, src_indices);
  PrimExpr value;
  if (kind == "max") {
    value = tirx::Max(old_value, new_value);
  } else if (kind == "min") {
    value = tirx::Min(old_value, new_value);
  } else {
    value = tirx::Add(old_value, new_value);
  }
  // The reduced dimension is the innermost serial loop: the destination
  // read-modify-write then has an inner axis along which its address is
  // invariant, which is exactly the accumulator-promotion criterion. The
  // emitter lifts it into a register (one ``vld``/``vdup`` plus one ``vsts``
  // per chunk) instead of one shared read and write per reduction step.
  tirx::Stmt body = tirx::BufferStore(dst.buffer, value, dst_indices);
  body = tirx::For(vars[dim], IntImm32(0), src.extents[dim], tirx::ForKind::kSerial, body);
  if (clear) {
    // The identity is a pre-store of the same address, outside the serial
    // loop: the plan folds it into the accumulator's initial value, so it
    // never reaches shared memory.
    PrimExpr identity = ReduceIdentity(recipe, ElemDtype(dst.buffer->dtype));
    tirx::Stmt init = tirx::BufferStore(dst.buffer, identity, dst_indices);
    body = tirx::SeqStmt({init, body});
  }
  for (int t = src_rank - 1; t >= 0; --t) {
    if (t == dim || !looped[t]) continue;
    body = tirx::For(vars[t], IntImm32(0), src.extents[t], tirx::ForKind::kParallel, body);
  }
  return body;
}

tirx::Stmt ExpandOneTileOp(const tirx::CallNode* call, arith::Analyzer* analyzer) {
  const auto* op = call->op.as<OpNode>();
  std::string name;
  if (op != nullptr) name = op->name.operator std::string();
  TileOpRegion src;
  TileOpRegion dst;
  if (name == "tl.tileop.reduce") {
    if (call->args.size() < 5)
      throw Unsupported("reduce call has too few arguments");
    if (!DecodeTileOpRegion(call->args[0], &src) ||
        !DecodeTileOpRegion(call->args[1], &dst)) {
      throw Unsupported("reduce operands are not tl.region calls");
    }
    const auto* kind_imm = call->args[2].as<tirx::StringImmNode>();
    const auto* dim_imm = call->args[3].as<tirx::IntImmNode>();
    const auto* clear_imm = call->args[4].as<tirx::IntImmNode>();
    if (kind_imm == nullptr || dim_imm == nullptr || clear_imm == nullptr) {
      throw Unsupported("reduce kind/dim/clear are not immediates");
    }
    return BuildReduceNestFromRegions(src, dst, kind_imm->value, dim_imm->value,
                                      clear_imm->value != 0, analyzer);
  }
  if (name == "tl.tileop.copy" || name == "tl.tileop.ascend_copy") {
    if (call->args.size() < 2 || !DecodeTileOpRegion(call->args[0], &src) ||
        !DecodeTileOpRegion(call->args[1], &dst)) {
      throw Unsupported("copy operands are not tl.region calls");
    }
    return BuildCopyNestFromRegions(src, dst, analyzer);
  }
  throw Unsupported("unsupported tile-op inside T.Parallel: " + name);
}

class TileOpExpander final : public tirx::StmtExprMutator {
 public:
  explicit TileOpExpander(arith::Analyzer* analyzer) : analyzer_(analyzer) {}

  bool expanded() const { return expanded_; }

  tirx::Stmt VisitStmt_(const tirx::EvaluateNode* op) final {
    if (const auto* call = op->value.as<tirx::CallNode>(); IsSupportedTileOp(call)) {
      expanded_ = true;
      return ExpandOneTileOp(call, analyzer_);
    }
    return tirx::StmtExprMutator::VisitStmt_(op);
  }

 private:
  arith::Analyzer* analyzer_;
  bool expanded_{false};
};

/* ------------------------------------------------------------------------- */
/* fragment staging fusion                                                   */
/* ------------------------------------------------------------------------- */

bool IsCopyCall(const tirx::CallNode* call) {
  if (call == nullptr) return false;
  const auto* op = call->op.as<OpNode>();
  if (op == nullptr) return false;
  return op->name == "tl.tileop.copy" || op->name == "tl.tileop.ascend_copy";
}

bool IsFragmentBuffer(const tirx::Buffer& buffer) {
  return buffer.scope().operator std::string() == "local.fragment";
}

bool IsZeroIndex(const PrimExpr& expr) {
  const auto* imm = expr.as<tirx::IntImmNode>();
  return imm != nullptr && imm->value == 0;
}

bool IsFullFragmentRegion(const TileOpRegion& region, arith::Analyzer* analyzer) {
  if (!IsFragmentBuffer(region.buffer)) return false;
  if (region.indices.size() != region.buffer->shape.size()) return false;
  for (size_t i = 0; i < region.indices.size(); ++i) {
    if (!IsZeroIndex(region.indices[i])) return false;
    if (!analyzer->CanProveEqual(region.extents[i], region.buffer->shape[i])) return false;
  }
  return true;
}

bool SameRegion(const TileOpRegion& lhs, const TileOpRegion& rhs,
                arith::Analyzer* analyzer) {
  if (!lhs.buffer.same_as(rhs.buffer)) return false;
  if (lhs.indices.size() != rhs.indices.size()) return false;
  for (size_t i = 0; i < lhs.indices.size(); ++i) {
    if (!analyzer->CanProveEqual(lhs.indices[i], rhs.indices[i])) return false;
  }
  return true;
}

bool DecodeCopyStmt(const tirx::Stmt& stmt, TileOpRegion* src, TileOpRegion* dst) {
  const auto* evaluate = stmt.as<tirx::EvaluateNode>();
  if (evaluate == nullptr) return false;
  const auto* call = evaluate->value.as<tirx::CallNode>();
  if (!IsCopyCall(call) || call->args.size() < 2) return false;
  return DecodeTileOpRegion(call->args[0], src) && DecodeTileOpRegion(call->args[1], dst);
}

bool DecodeReduceStmt(const tirx::Stmt& stmt, TileOpRegion* src, TileOpRegion* dst,
                      std::string* kind, int64_t* dim, bool* clear) {
  const auto* evaluate = stmt.as<tirx::EvaluateNode>();
  if (evaluate == nullptr) return false;
  const auto* call = evaluate->value.as<tirx::CallNode>();
  if (call == nullptr || call->args.size() < 5) return false;
  const auto* op = call->op.as<OpNode>();
  if (op == nullptr || op->name != "tl.tileop.reduce") return false;
  const auto* kind_imm = call->args[2].as<tirx::StringImmNode>();
  const auto* dim_imm = call->args[3].as<tirx::IntImmNode>();
  const auto* clear_imm = call->args[4].as<tirx::IntImmNode>();
  if (kind_imm == nullptr || dim_imm == nullptr || clear_imm == nullptr) return false;
  if (!DecodeTileOpRegion(call->args[0], src) ||
      !DecodeTileOpRegion(call->args[1], dst))
    return false;
  *kind = kind_imm->value;
  *dim = dim_imm->value;
  *clear = clear_imm->value != 0;
  return true;
}

/*!
 * \brief Fuse the ``T.reduce_*`` frontend staging pattern back to a direct
 *        shared-to-shared reduction.
 *
 * ``T.reduce_*`` on shared buffers is macro-expanded into
 * ``copy(shared_src, frag_src); reduce(frag_src, frag_dst); copy(frag_dst,
 * shared_dst)`` (plus a ``copy(shared_dst, frag_dst)`` pre-copy when
 * ``clear=false``). The fragments are VF-level temporaries, not user data: the
 * memory round-trip is exactly what the direct reduce nest already models, so
 * the whole sequence is replaced by one reduction nest on the shared regions.
 * Unmatched staging copies stay in place and are rejected by the expander.
 */
class ReduceStagingFuser final : public tirx::StmtExprMutator {
 public:
  explicit ReduceStagingFuser(arith::Analyzer* analyzer) : analyzer_(analyzer) {}

  bool fused() const { return fused_; }

  tirx::Stmt VisitStmt_(const tirx::SeqStmtNode* op) final {
    std::vector<tirx::Stmt> out;
    for (size_t i = 0; i < op->seq.size();) {
      tirx::Stmt fused_stmt;
      size_t consumed = 0;
      if (TryFuse(op->seq, i, &fused_stmt, &consumed)) {
        if (const auto* fused_seq = fused_stmt.as<tirx::SeqStmtNode>()) {
          for (const tirx::Stmt& child : fused_seq->seq) out.push_back(child);
        } else {
          out.push_back(std::move(fused_stmt));
        }
        i += consumed;
        fused_ = true;
      } else {
        out.push_back(VisitStmt(op->seq[i]));
        ++i;
      }
    }
    if (out.size() == 1) return out[0];
    return tirx::SeqStmt(out);
  }

 private:
  bool TryFuse(const ffi::Array<tirx::Stmt>& stmts, size_t pos, tirx::Stmt* out,
               size_t* consumed) {
    // clear=true: copy(shared_src, frag_src); reduce(frag_src, frag_dst, true);
    //             copy(frag_dst, shared_dst)
    if (pos + 2 < stmts.size()) {
      TileOpRegion src_shared;
      TileOpRegion src_frag;
      TileOpRegion reduce_src;
      TileOpRegion reduce_dst;
      TileOpRegion out_frag;
      TileOpRegion dst_shared;
      std::string kind;
      int64_t dim = 0;
      bool clear = true;
      if (DecodeCopyStmt(stmts[pos], &src_shared, &src_frag) &&
          IsFullFragmentRegion(src_frag, analyzer_) &&
          DecodeReduceStmt(stmts[pos + 1], &reduce_src, &reduce_dst, &kind, &dim,
                           &clear) &&
          clear && reduce_src.buffer.same_as(src_frag.buffer) &&
          IsFullFragmentRegion(reduce_src, analyzer_) &&
          IsFullFragmentRegion(reduce_dst, analyzer_) &&
          DecodeCopyStmt(stmts[pos + 2], &out_frag, &dst_shared) &&
          out_frag.buffer.same_as(reduce_dst.buffer) &&
          IsFullFragmentRegion(out_frag, analyzer_) &&
          !IsFragmentBuffer(dst_shared.buffer)) {
        *out = BuildReduceNestFromRegions(src_shared, dst_shared, kind, dim, true,
                                          analyzer_);
        *consumed = 3;
        return true;
      }
    }
    // clear=false: copy(shared_dst, frag_dst); copy(shared_src, frag_src);
    //              reduce(frag_src, frag_dst, false); copy(frag_dst, shared_dst)
    if (pos + 3 < stmts.size()) {
      TileOpRegion pre_src;
      TileOpRegion pre_dst;
      TileOpRegion src_shared;
      TileOpRegion src_frag;
      TileOpRegion reduce_src;
      TileOpRegion reduce_dst;
      TileOpRegion out_frag;
      TileOpRegion dst_shared;
      std::string kind;
      int64_t dim = 0;
      bool clear = true;
      if (DecodeCopyStmt(stmts[pos], &pre_src, &pre_dst) &&
          IsFullFragmentRegion(pre_dst, analyzer_) &&
          DecodeCopyStmt(stmts[pos + 1], &src_shared, &src_frag) &&
          IsFullFragmentRegion(src_frag, analyzer_) &&
          DecodeReduceStmt(stmts[pos + 2], &reduce_src, &reduce_dst, &kind, &dim,
                           &clear) &&
          !clear && reduce_src.buffer.same_as(src_frag.buffer) &&
          IsFullFragmentRegion(reduce_src, analyzer_) &&
          reduce_dst.buffer.same_as(pre_dst.buffer) &&
          IsFullFragmentRegion(reduce_dst, analyzer_) &&
          DecodeCopyStmt(stmts[pos + 3], &out_frag, &dst_shared) &&
          out_frag.buffer.same_as(reduce_dst.buffer) &&
          IsFullFragmentRegion(out_frag, analyzer_) &&
          !IsFragmentBuffer(dst_shared.buffer) &&
          SameRegion(pre_src, dst_shared, analyzer_)) {
        *out = BuildReduceNestFromRegions(src_shared, dst_shared, kind, dim, false,
                                          analyzer_);
        *consumed = 4;
        return true;
      }
    }
    return false;
  }

  arith::Analyzer* analyzer_;
  bool fused_{false};
};

}  // namespace

std::pair<tirx::Stmt, bool> ExpandParallelTileOps(const tirx::Stmt& stmt,
                                                  arith::Analyzer* analyzer) {
  arith::Analyzer local;
  arith::Analyzer* active = analyzer != nullptr ? analyzer : &local;
  if (!stmt.defined()) return {stmt, false};
  if (!ContainsTileOpCall(stmt)) return {stmt, false};
  ReduceStagingFuser fuser(active);
  tirx::Stmt fused = fuser(stmt);
  if (!ContainsTileOpCall(fused)) return {fused, fuser.fused()};
  TileOpExpander expander(active);
  tirx::Stmt out = expander(fused);
  return {std::move(out), fuser.fused() || expander.expanded()};
}

void NormalizeBodyInto(const tirx::Stmt& stmt, const AliasMap& aliases,
                       std::vector<StoreSpec>* out, ffi::Optional<PrimExpr> mask) {
  if (const auto* seq = stmt.as<tirx::SeqStmtNode>()) {
    for (const tirx::Stmt& child : seq->seq) {
      NormalizeBodyInto(child, aliases, out, mask);
    }
    return;
  }

  if (const auto* store = stmt.as<tirx::BufferStoreNode>()) {
    PrimExpr value = ApplyAliases(store->value, aliases);
    out->push_back(MakeStoreSpec(store->buffer, store->indices, value, mask));
    return;
  }

  if (const auto* evaluate = stmt.as<tirx::EvaluateNode>()) {
    if (IsZeroLiteral(evaluate->value)) return;
    if (const auto* call = evaluate->value.as<tirx::CallNode>()) {
      if (IsAtomicName(CallTail(ffi::GetRef<tirx::Call>(call)))) {
        throw Unsupported(
            "atomic op (T.atomic_add) is not supported yet: no vector+scalar atomic recipe");
      }
      std::string name = "Evaluate";
      if (const auto* op = call->op.as<OpNode>()) name = op->name;
      throw Unsupported("opaque Evaluate inside a planned nest is not supported: " + name);
    }
    throw Unsupported("opaque Evaluate inside a planned nest is not supported: " +
                      TypeKeyTail(evaluate->value));
  }

  if (const auto* branch = stmt.as<tirx::IfThenElseNode>()) {
    PrimExpr condition = ApplyAliases(branch->condition, aliases);
    std::vector<StoreSpec> then_stores;
    NormalizeBodyInto(branch->then_case, aliases, &then_stores, std::nullopt);
    std::vector<StoreSpec> else_stores;
    if (!IsTruthyElse(branch->else_case)) {
      NormalizeBodyInto(*branch->else_case, aliases, &else_stores, std::nullopt);
    }

    if (else_stores.empty()) {
      // Single-arm if -> predication: every write carries (cond & mask).
      ffi::Optional<PrimExpr> combined(
          mask.has_value() ? logical_and(*mask, condition) : condition);
      for (const StoreSpec& spec_const : then_stores) {
        StoreSpec spec = spec_const;
        ffi::Optional<PrimExpr> spec_mask = spec->mask;
        spec.CopyOnWrite()->mask =
            spec_mask.has_value()
                ? ffi::Optional<PrimExpr>(logical_and(*combined, *spec_mask))
                : combined;
        out->push_back(spec);
      }
      return;
    }

    if (then_stores.size() == 1 && else_stores.size() == 1) {
      const StoreSpec& then_spec = then_stores[0];
      const StoreSpec& else_spec = else_stores[0];
      if (SameBuffer(then_spec->buffer, else_spec->buffer) &&
          SameIndices(then_spec->indices, else_spec->indices)) {
        // Statement-level if/else arms are *lazy*: use ``if_then_else`` so the
        // arm loads stay guarded (a plain Select would load eagerly, D16).
        PrimExpr merged = if_then_else(condition, then_spec->value, else_spec->value);
        if (then_spec->mask.has_value() || else_spec->mask.has_value()) {
          throw Unsupported(
              "masked branch merging is not supported: one arm of this same-address if/else is itself "
              "predicated (e.g. a nested else-if), so there is no single select condition; "
              "rewrite it as two consecutive masked stores");
        }
        out->push_back(MakeStoreSpec(then_spec->buffer, then_spec->indices, merged, mask));
        return;
      }
    }

    throw Unsupported(
        "divergent branch arms: an if/else whose arms write different locations cannot be vectorized (方案 §8.1)");
  }

  if (const auto* loop = stmt.as<tirx::ForNode>()) {
    throw Unsupported(
        "nested " + ForKindName(loop->kind) +
        " loop inside T.Parallel is not affine-vectorizable "
        "(方案 §6): only directly-nested, dependence-free serial loops are absorbed as axes");
  }

  if (stmt.as<tirx::BindNode>() != nullptr) {
    return;
  }

  throw Unsupported("unsupported statement in T.Parallel body: " + TypeKeyTail(stmt));
}

std::vector<StoreSpec> NormalizeBody(const tirx::Stmt& stmt, const AliasMap& aliases) {
  std::vector<StoreSpec> out;
  NormalizeBodyInto(stmt, aliases, &out, std::nullopt);
  return out;
}

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm
