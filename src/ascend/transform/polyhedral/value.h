/*!
 * \file value.h
 * \brief Value DAG: lift scalar expressions into per-lane vector value nodes.
 *
 * C++ counterpart of ``polyhedral/value.py``. The decision is again purely the
 * finite difference along the vector axis:
 *
 *   Delta_v == 1  -> vload             (contiguous)
 *   Delta_v == 0  -> uniform_load      (scalar read + vdup broadcast)
 *   Delta_v == k  -> vload_strided     (vgather2 with a stride ramp)
 *   indirect      -> gather            (per-lane index vector)
 *   loop var      -> lane_index        (vci)
 *
 * ``ValueNode`` is instruction-selection agnostic: ``recipe`` is a primitive
 * name, the concrete ``T.simd.*`` instruction is chosen by the emit layer.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_VALUE_H_
#define TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_VALUE_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/tirx/expr.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "access.h"

namespace tvm {
namespace tl {
namespace polyhedral {

class ValueNodeNode;

/*!
 * \brief One value node: a primitive recipe plus its operands.
 *
 * ``predicate`` is the *load guard* (P0-3): the value is only valid on lanes
 * where the predicate holds, which happens inside lazy ``if_then_else`` arms.
 * ``predicate_free_until`` / ``predicate_free_from`` give the leading/trailing
 * chunk ranges where the guard is provably true, so the emitter can drop the
 * masked gather in those blocks.
 */
class ValueNodeNode : public ffi::Object {
 public:
  ffi::String recipe;
  ffi::String dtype;
  ffi::Array<ffi::ObjectRef> args;  // ValueNode elements
  ffi::Optional<AccessInfo> access;
  ffi::Optional<PrimExpr> scalar;
  ffi::String detail;
  ffi::Optional<ffi::ObjectRef> predicate;  // ValueNode
  int64_t predicate_free_until{0};
  ffi::Optional<int64_t> predicate_free_from;
  bool is_uniform{false};

  static constexpr const bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.polyhedral.ValueNode", ValueNodeNode,
                                    ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ValueNodeNode>()
        .def_ro("recipe", &ValueNodeNode::recipe)
        .def_ro("dtype", &ValueNodeNode::dtype)
        .def_ro("args", &ValueNodeNode::args)
        .def_ro("access", &ValueNodeNode::access)
        .def_ro("scalar", &ValueNodeNode::scalar)
        .def_ro("detail", &ValueNodeNode::detail)
        .def_ro("predicate", &ValueNodeNode::predicate)
        .def_ro("predicate_free_until", &ValueNodeNode::predicate_free_until)
        .def_ro("predicate_free_from", &ValueNodeNode::predicate_free_from)
        .def_ro("is_uniform", &ValueNodeNode::is_uniform);
  }
};

class ValueNode : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ValueNode, ffi::ObjectRef, ValueNodeNode);
};

/*! \brief Value node fields that the constructors of Python set by keyword. */
struct ValueSpec {
  std::string recipe;
  std::string dtype;
  std::vector<ValueNode> args;
  ffi::Optional<AccessInfo> access;
  ffi::Optional<PrimExpr> scalar;
  std::string detail;
  ValueNode predicate;  // undefined == None
  int64_t predicate_free_until{0};
  std::optional<int64_t> predicate_free_from;
};

/*! \brief Build a value node (``is_uniform`` is derived, exactly as Python). */
ValueNode MakeValueNode(ValueSpec spec);

/*! \brief Shorthand for a value node without access / guard fields. */
ValueNode MakeValue(const std::string& recipe, const std::string& dtype,
                    std::vector<ValueNode> args = {}, std::string detail = "");

/*! \brief The guard node of a value (undefined when there is none). */
ValueNode PredicateOf(const ValueNode& node);
/*! \brief Operands of a value node as typed value nodes. */
std::vector<ValueNode> ArgsOf(const ValueNode& node);

/*! \brief Per-lane expression context (axes, vector axis, analyzer, guard). */
struct ValueContext {
  std::vector<Axis> axes;
  Axis vector_axis;
  arith::Analyzer* analyzer{nullptr};
  ValueNode guard;  // undefined == no guard
  int64_t guard_free_until{0};
  std::optional<int64_t> guard_free_from;
  int64_t lanes{0};
};

/*! \brief Narrow the context to a new guard (no-op when ``guard`` is None). */
ValueContext WithGuard(const ValueContext& ctx, const ValueNode& guard,
                       int64_t free_until = 0,
                       std::optional<int64_t> free_from = std::nullopt);

/*! \brief Prefix conjunction of two guard-free ranges: the smaller one. */
int64_t CombineFreeUntil(int64_t lhs, int64_t rhs);
/*! \brief Suffix conjunction: ``None`` when either side cannot be proven. */
std::optional<int64_t> CombineFreeFrom(std::optional<int64_t> lhs,
                                      std::optional<int64_t> rhs);

/*! \brief ``[0, until)`` / ``[from, chunks)`` chunk ranges where ``expr`` holds. */
std::pair<int64_t, std::optional<int64_t>> GuardFreeBounds(const PrimExpr& expr,
                                                           const ValueContext& ctx);

/*! \brief Lift a scalar expression into a vector value node. */
ValueNode BuildValue(const PrimExpr& expr, const ValueContext& ctx);

/*! \brief Build the per-lane absolute element offset vector of an access. */
ValueNode IndexValue(const AccessInfo& info, const ValueContext& ctx);

/*! \brief Turn a classified access into a load node. */
ValueNode LoadNode(const AccessInfo& info, const ValueContext& ctx);

/*! \brief Whether a lane-uniform scalar is a safe operand under a lazy guard. */
ValueNode UniformGuarded(const PrimExpr& expr, const ValueContext& ctx);

/*! \brief Broadcast a lane-uniform value to a vector (``vbrc``). */
ValueNode Broadcast(const ValueNode& node, const ValueContext& ctx);

/*! \brief ``pnot`` / scalar negate. */
ValueNode NegatePredicate(const ValueNode& node);
/*! \brief ``pand`` / scalar and; undefined operand means "no guard". */
ValueNode AndPredicate(const ValueNode& lhs, const ValueNode& rhs);

/*! \brief Whether the value DAG reads ``buffer`` anywhere. */
bool ReadsBuffer(const ValueNode& node, const tirx::Buffer& buffer);
/*! \brief Same-buffer, same-address load node inside the DAG (RMW old value). */
ValueNode SameAddressLoad(const ValueNode& node, const tirx::Buffer& buffer,
                          const ffi::Array<PrimExpr>& indices);
/*! \brief Whether the DAG reads ``buffer`` at a *different* address. */
bool ReadsOtherAddress(const ValueNode& node, const tirx::Buffer& buffer,
                       const ffi::Array<PrimExpr>& indices);

/*!
 * \brief Whether the access provably stays inside the buffer for all lanes.
 *
 * Only used to decide whether a load inside a lazy conditional needs a guard:
 * provably safe loads keep emitting plain ``vld``.
 */
bool AccessProvablyInBounds(const AccessInfo& info, const ValueContext& ctx,
                            int64_t lo = 0, std::optional<int64_t> hi = std::nullopt);

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_VALUE_H_
