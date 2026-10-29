/*!
 * \file access.h
 * \brief SCoP modelling: iteration axes, affine memory access classification.
 *
 * C++ counterpart of ``polyhedral/access.py``. The model is:
 *
 *   iteration domain  D = { 0 <= x_k < E_k }
 *   access function   off(x) = elem_offset + sum_d idx_d * stride_d
 *   classification    Delta_v(off) = off(x_v + 1) - off(x_v)
 *
 * Everything downstream (contiguity / broadcast / stride / gather) is decided
 * by the shape of ``Delta_v`` — never by matching code shapes.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_ACCESS_H_
#define TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_ACCESS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace polyhedral {

/*! \brief Access shape along one axis (``AccessClass``). */
enum class AccessClass {
  kContiguous,  // Delta == 1
  kUniform,     // Delta == 0
  kStrided,     // Delta == k, |k| > 1
  kPermute,     // Delta == -1 (bijective reversal)
  kPiecewise,   // non-constant Delta, but the index vector is constructible
  kIndirect,    // the subscript itself contains a BufferLoad
  kTrueRandom,  // non-constant Delta, index vector not constructible -> reject
};

/*! \brief ``AccessClass.value`` text used by diagnostics / rendering. */
const std::string& AccessClassName(AccessClass kind);
/*! \brief Inverse of :func:`AccessClassName`. */
AccessClass AccessClassFromName(const std::string& name);

class AxisNode;

/*!
 * \brief One iteration axis ``0 <= var < extent``.
 *
 * ``kind`` separates ``T.Parallel`` axes from absorbed ``T.serial`` axes: a
 * directly-nested, dependence-free serial loop is part of the iteration domain
 * too, so it is an equally valid candidate vector axis.
 */
class AxisNode : public ffi::Object {
 public:
  tirx::Var var;
  PrimExpr extent;
  int64_t index{0};  // position in the nest, 0 = outermost
  int64_t kind{1};   // tirx::ForKind as int (1 == kParallel)
  ffi::String name;
  bool is_parallel{true};

  static constexpr const bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.polyhedral.Axis", AxisNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AxisNode>()
        .def_ro("var", &AxisNode::var)
        .def_ro("extent", &AxisNode::extent)
        .def_ro("index", &AxisNode::index)
        .def_ro("kind", &AxisNode::kind)
        .def_ro("name", &AxisNode::name)
        .def_ro("is_parallel", &AxisNode::is_parallel);
  }
};

class Axis : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Axis, ffi::ObjectRef, AxisNode);
};

/*! \brief Build an axis; ``name`` mirrors the Python ``@property``. */
Axis MakeAxis(tirx::Var var, PrimExpr extent, int64_t index, tirx::ForKind kind);

/*! \brief One axis' classification: kind + stride (``None`` when not constant). */
struct AxisClass {
  AccessClass kind{AccessClass::kTrueRandom};
  std::optional<int64_t> stride;
};

/*!
 * \brief One buffer access plus its per-axis classification.
 *
 * ``per_axis`` is keyed by ``axis.index`` (position in this iteration domain),
 * never by the variable name: two different serial loops may share a name.
 */
class AccessInfoNode : public ffi::Object {
 public:
  tirx::Buffer buffer;
  ffi::Array<PrimExpr> indices;
  PrimExpr offset;
  bool is_store{false};
  bool indirect{false};
  bool in_global{false};
  ffi::String reason;
  // Not reflected: tests never read it, and it is keyed by axis.index.
  std::map<int64_t, AxisClass> per_axis;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.polyhedral.AccessInfo", AccessInfoNode,
                                    ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AccessInfoNode>()
        .def_ro("buffer", &AccessInfoNode::buffer)
        .def_ro("indices", &AccessInfoNode::indices)
        .def_ro("offset", &AccessInfoNode::offset)
        .def_ro("is_store", &AccessInfoNode::is_store)
        .def_ro("indirect", &AccessInfoNode::indirect)
        .def_ro("in_global", &AccessInfoNode::in_global)
        .def_ro("reason", &AccessInfoNode::reason);
  }
};

class AccessInfo : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AccessInfo, ffi::ObjectRef,
                                             AccessInfoNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AccessInfoNode);
};

/*! \brief ``info.axis_class(axis)`` — FFI-visible so Python emit can reuse it. */
AxisClass AxisClassOf(const AccessInfo& info, const Axis& axis);

/*!
 * \brief Recognise an integer constant.
 *
 * Literals first; when an analyzer is given, ``simplify`` then
 * ``const_int_bound`` (equal bounds prove a constant).
 */
std::optional<int64_t> ConstInt(const PrimExpr& expr, arith::Analyzer* analyzer);

/*! \brief Row-major compact strides (used when ``buffer.strides`` is empty). */
std::vector<PrimExpr> BufferStrides(const tirx::Buffer& buffer);

/*! \brief Linearize an N-D access: ``off = elem_offset + sum idx_d * stride_d``. */
PrimExpr LinearElementOffset(const tirx::Buffer& buffer,
                             const ffi::Array<PrimExpr>& indices,
                             arith::Analyzer* analyzer);

/*! \brief ``Delta_v(off) = off(v + 1) - off(v)`` (simplified). */
PrimExpr FiniteDifference(const PrimExpr& offset, const tirx::Var& var,
                          arith::Analyzer* analyzer);

/*! \brief Map a Delta value onto an :class:`AccessClass` (Δ layer only). */
AxisClass ClassifyDelta(const PrimExpr& delta, arith::Analyzer* analyzer);

/*!
 * \brief Whether the per-lane index vector can be built for ``expr``.
 *
 * Returns ``std::nullopt`` when constructible, otherwise a one-line diagnostic
 * (the ``TRUE_RANDOM`` justification).
 */
std::optional<std::string> IndexVectorReason(const PrimExpr& expr,
                                             arith::Analyzer* analyzer);

/*! \brief Whether the expression contains a ``BufferLoad`` anywhere. */
bool ContainsBufferLoad(const PrimExpr& expr);

/*! \brief All ``Var`` nodes of an expression, in post-order. */
std::vector<tirx::Var> FreeVars(const PrimExpr& expr);

/*!
 * \brief Build the affine model of one access and classify it per axis.
 *
 * Δ gives the constant shapes; non-constant Δ is refined structurally into
 * INDIRECT (subscript contains a load), PIECEWISE (index vector buildable) or
 * TRUE_RANDOM (neither). An indirect access that is *invariant* along the axis
 * stays UNIFORM: it is a lane-uniform scalar address (e.g. ``c[b[0]] = 1.0``).
 */
AccessInfo ClassifyAccess(const tirx::Buffer& buffer,
                          const ffi::Array<PrimExpr>& indices,
                          const std::vector<Axis>& axes, arith::Analyzer* analyzer,
                          bool is_store);

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_ACCESS_H_
