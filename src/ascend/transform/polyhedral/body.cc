/*!
 * \file body.cc
 * \brief Body normalization implementation.
 */
#include "body.h"

#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
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
    if (const auto* call = evaluate->value.as<tirx::CallNode>()) {
      if (IsAtomicName(CallTail(ffi::GetRef<tirx::Call>(call)))) {
        throw Unsupported(
            "atomic op (T.atomic_add) is not supported yet: no vector+scalar atomic recipe");
      }
    }
    return;
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
