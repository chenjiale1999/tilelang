/*!
 * \file body.h
 * \brief Body normalization: multi-statement / branches -> ordered stores.
 *
 * C++ counterpart of ``polyhedral/body.py``. The rules are operator-agnostic:
 *
 *  * ``SeqStmt``       -> flatten in order (§3);
 *  * ``BufferStore``   -> one store;
 *  * single-arm ``if`` -> predicated store (``mask &= cond``);
 *  * two-arm ``if``    -> lazy ``if_then_else(cond, a, b)`` when both arms write
 *                         the same location;
 *  * everything else   -> explicit ``Unsupported`` (no silent downgrade).
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_BODY_H_
#define TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_BODY_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace polyhedral {

/*! \brief Where to write, what to write, and an optional predicate. */
class StoreSpecNode : public ffi::Object {
 public:
  tirx::Buffer buffer;
  ffi::Array<PrimExpr> indices;
  PrimExpr value;
  ffi::Optional<PrimExpr> mask;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.polyhedral.StoreSpec", StoreSpecNode,
                                    ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StoreSpecNode>()
        .def_ro("buffer", &StoreSpecNode::buffer)
        .def_ro("indices", &StoreSpecNode::indices)
        .def_ro("value", &StoreSpecNode::value)
        .def_ro("mask", &StoreSpecNode::mask);
  }
};

class StoreSpec : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StoreSpec, ffi::ObjectRef, StoreSpecNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StoreSpecNode);
};

/*! \brief Build a store spec. */
StoreSpec MakeStoreSpec(tirx::Buffer buffer, ffi::Array<PrimExpr> indices,
                        PrimExpr value, ffi::Optional<PrimExpr> mask = std::nullopt);

/*!
 * \brief Scalar aliases introduced by ``Bind`` (``acc = a[i,j] * 2``).
 *
 * Only real value bindings participate; a Bind whose value is a Call (buffer
 * declaration / type constructor) is skipped, so §3 scalar temporaries fold
 * into the final expression.
 */
struct AliasMap {
  std::unordered_map<const tirx::VarNode*, PrimExpr> values;
  bool empty() const { return values.empty(); }
};

AliasMap CollectAliases(const tirx::Stmt& stmt);

/*! \brief Inline aliases (bounded passes to avoid a cyclic Bind chain). */
PrimExpr ApplyAliases(const PrimExpr& expr, const AliasMap& aliases);

/*! \brief Normalize a body into an ordered list of store specs. */
std::vector<StoreSpec> NormalizeBody(const tirx::Stmt& stmt, const AliasMap& aliases);

/*! \brief Normalize with an inherited predicate (recursion entry point). */
void NormalizeBodyInto(const tirx::Stmt& stmt, const AliasMap& aliases,
                       std::vector<StoreSpec>* out, ffi::Optional<PrimExpr> mask);

/*!
 * \brief Expand tile-op calls inside a ``T.Parallel`` body into explicit loops.
 *
 * ``T.copy`` becomes an elementwise copy nest and ``T.reduce_*`` becomes a
 * reduction nest (the reduced dimension stays serial, the destination-mapped
 * dimensions stay parallel); ``clear=true`` prepends an identity-init nest.
 * The second element of the result is ``true`` only when at least one call was
 * expanded. Unsupported shapes throw :class:`Unsupported` instead of leaving
 * the call in place (which P3 would otherwise drop together with its span).
 */
std::pair<tirx::Stmt, bool> ExpandParallelTileOps(const tirx::Stmt& stmt,
                                                  arith::Analyzer* analyzer);

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_BODY_H_
