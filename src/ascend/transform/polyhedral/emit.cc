/*!
 * \file emit.cc
 * \brief Emission layer implementation (strip-mining, masks, T.simd.* ops).
 */
#include "emit.h"

#include <tvm/ir/repr.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "access.h"
#include "ascend/op/builtin.h"
#include "primitives.h"
#include "value.h"

namespace tvm {
namespace tl {
namespace polyhedral {

namespace {

using tirx::ForKind;
using tirx::Stmt;

/* ------------------------------------------------------------------------- */
/* T.simd.* call builders (mirror tilelang/ascend/language/simd.py)           */
/* ------------------------------------------------------------------------- */

PrimExpr SimdCall(const Op& op, DataType dtype, const std::vector<PrimExpr>& args) {
  ffi::Array<PrimExpr> array;
  for (const PrimExpr& arg : args) array.push_back(arg);
  return tirx::Call(dtype, op, array);
}

PrimExpr StrImm(const std::string& value) { return tirx::StringImm(value); }

/*! \brief ``T.access_ptr(base, "r"/"w", extent)``. */
PrimExpr AccessPtr(const PrimExpr& base_load, int64_t extent, int rw_mask) {
  return tirx::Call(DataType::Handle(), access_ptr(),
              {base_load, IntImm(DataType::Int(32), extent),
               IntImm(DataType::Int(32), rw_mask)});
}

DataType BoolMaskDtype() { return DataType::Bool(256); }

PrimExpr Vld(const PrimExpr& load, const tirx::Buffer& buffer, const std::string& dist) {
  int lanes = VRegLanes(ElemDtype(buffer->dtype));
  int64_t extent = lanes;
  std::string elem = ElemDtype(buffer->dtype);
  // BRC_B* broadcasts one element of the suffix width.
  if (dist == "BRC_B8" || dist == "BRC_B16" || dist == "BRC_B32") {
    int brc_bits = dist == "BRC_B8" ? 8 : (dist == "BRC_B16" ? 16 : 32);
    int elem_bits = 0;
    ElemBits(elem, &elem_bits);
    extent = brc_bits / elem_bits;
  }
  PrimExpr ptr = AccessPtr(load, extent, /*rw=*/1);
  return SimdCall(simd_vld(), VecDataType(elem),
                  {ptr, StrImm(dist)});
}

PrimExpr Vsts(const PrimExpr& load, const tirx::Buffer& buffer, const PrimExpr& src,
              const PrimExpr& mask, const std::string& dist) {
  int lanes = VRegLanes(ElemDtype(buffer->dtype));
  int64_t extent = lanes;
  if (dist.compare(0, 6, "ONEPT_") == 0) {
    extent = 1;
  } else if (dist.compare(0, 3, "PK_") == 0) {
    extent = lanes / 2;
  } else if (dist.compare(0, 4, "PK4_") == 0) {
    extent = lanes / 4;
  }
  PrimExpr ptr = AccessPtr(load, extent, /*rw=*/2);
  return SimdCall(simd_vsts(), DataType::Void(), {ptr, src, mask, StrImm(dist)});
}

PrimExpr Vgather2(const PrimExpr& base_load, const tirx::Buffer& buffer,
                  const PrimExpr& index, const PrimExpr& mask) {
  std::string elem = ElemDtype(buffer->dtype);
  int lanes = VRegLanes(elem);
  PrimExpr ptr = AccessPtr(base_load, lanes, /*rw=*/1);
  std::string out_elem = elem;
  if (elem == "int8") out_elem = "int16";
  if (elem == "uint8") out_elem = "uint16";
  return SimdCall(simd_vgather2(), VecDataType(out_elem),
                  {ptr, index, mask});
}

PrimExpr Vscatter(const PrimExpr& src, const PrimExpr& base_load,
                  const tirx::Buffer& buffer, const PrimExpr& index,
                  const PrimExpr& mask) {
  int lanes = VRegLanes(ElemDtype(buffer->dtype));
  PrimExpr ptr = AccessPtr(base_load, lanes, /*rw=*/2);
  return SimdCall(simd_vscatter(), DataType::Void(), {src, ptr, index, mask});
}

PrimExpr Vdup(const PrimExpr& scalar, const std::string& elem, const PrimExpr& mask) {
  return SimdCall(simd_vdup(), VecDataType(elem),
                  {scalar, mask, StrImm(ModeZeroing())});
}

PrimExpr Vci(const PrimExpr& index, const std::string& elem) {
  return SimdCall(simd_vci(), VecDataType(elem),
                  {index, StrImm("INC_ORDER")});
}

PrimExpr Pset(int bits, const std::string& dist) {
  return SimdCall(simd_pset(), BoolMaskDtype(),
                  {IntImm(DataType::Int(32), bits), StrImm(dist)});
}

PrimExpr UpdateMask(const PrimExpr& count, int bits) {
  return SimdCall(simd_update_mask(), BoolMaskDtype(),
                  {count, IntImm(DataType::Int(32), bits)});
}

PrimExpr MethodMask(const Op& op, const std::vector<PrimExpr>& args) {
  return SimdCall(op, BoolMaskDtype(), args);
}

/*! \brief ``getattr(simd, recipe)`` for the scalar-vector binary family. */
const Op& ScalarBinaryOp(const std::string& recipe) {
  if (recipe == "vadds") return simd_vadds();
  if (recipe == "vmuls") return simd_vmuls();
  if (recipe == "vmaxs") return simd_vmaxs();
  if (recipe == "vmins") return simd_vmins();
  if (recipe == "vshrs") return simd_vshrs();
  if (recipe == "vshls") return simd_vshls();
  TVM_FFI_ICHECK(false) << "unknown scalar-vector recipe '" << recipe << "'";
  return simd_vadds();
}

/*! \brief ``getattr(simd, recipe)`` for the elementwise binary family. */
const Op& BinaryOp(const std::string& recipe) {
  if (recipe == "vadd") return simd_vadd();
  if (recipe == "vsub") return simd_vsub();
  if (recipe == "vmul") return simd_vmul();
  if (recipe == "vdiv") return simd_vdiv();
  if (recipe == "vmax") return simd_vmax();
  if (recipe == "vmin") return simd_vmin();
  if (recipe == "vand") return simd_vand();
  if (recipe == "vor") return simd_vor();
  if (recipe == "vxor") return simd_vxor();
  if (recipe == "vshl") return simd_vshl();
  if (recipe == "vshr") return simd_vshr();
  TVM_FFI_ICHECK(false) << "unknown binary recipe '" << recipe << "'";
  return simd_vadd();
}

/*! \brief ``getattr(simd, recipe)`` for the elementwise unary family. */
const Op& UnaryOp(const std::string& recipe) {
  if (recipe == "vneg") return simd_vneg();
  if (recipe == "vabs") return simd_vabs();
  if (recipe == "vexp") return simd_vexp();
  if (recipe == "vsqrt") return simd_vsqrt();
  if (recipe == "vln") return simd_vln();
  TVM_FFI_ICHECK(false) << "unknown unary recipe '" << recipe << "'";
  return simd_vneg();
}

/*! \brief Cross-lane reduction op; ``vcadd`` widens 16-bit integer inputs. */
PrimExpr ReduceCall(const std::string& recipe, const PrimExpr& src, const PrimExpr& mask) {
  std::string src_dtype = ffi::DLDataTypeToString(src.dtype()).operator std::string();
  std::string elem = ElemDtype(src_dtype);
  std::string out = src_dtype;
  if (recipe == "vcadd") {
    if (elem == "int16") out = VecDtype("int32");
    if (elem == "uint16") out = VecDtype("uint32");
  }
  const Op& op = recipe == "vcadd" ? simd_vcadd() : (recipe == "vcmax" ? simd_vcmax() : simd_vcmin());
  return SimdCall(op, DataTypeOf(out),
                  {src, mask, StrImm(ModeZeroing())});
}

/*!
 * \brief ``T.simd.vcvt`` argument protocol for the default round/sat/part.
 *
 * Mirrors ``language/simd.py:vcvt`` with ``round="ROUND_R"``, ``sat=True`` and
 * ``part=0``; only the pairs P2 can produce are needed, everything else keeps
 * the Python "Unsupported vcvt conversion" rejection.
 */
PrimExpr Vcvt(const PrimExpr& src, const std::string& target_elem, const PrimExpr& mask) {
  std::string src_elem = ElemDtype(ffi::DLDataTypeToString(src.dtype()).operator std::string());
  const std::string round = "ROUND_R";
  const std::string sat = "RS_ENABLE";
  const std::string part2 = "PART_EVEN";
  const std::string part4 = "PART_P0";
  std::vector<PrimExpr> args = {src, mask};
  auto push = [&](const std::string& value) { args.push_back(StrImm(value)); };
  // ``PART_EVEN`` matches ``simd.vcvt(part=0)`` (the language default) and is
  // what the verified cast kernel emits before a packed ``PK_B*`` store: a
  // narrowing conversion fills the even 16-bit slots of the wider destination
  // register, and the packed store writes exactly those lanes.
  if (src_elem == "float32") {
    if (target_elem == "int64") {
      push(round); push(sat); push(part2); push(ModeZeroing());
    } else if (target_elem == "int32") {
      push(round); push(sat); push(ModeZeroing());
    } else if (target_elem == "int16") {
      push(round); push(sat); push(part2); push(ModeZeroing());
    } else if (target_elem == "float16" || target_elem == "bfloat16") {
      push(round); push(sat); push(part2); push(ModeZeroing());
    } else if (target_elem == "float8_e4m3" || target_elem == "float8_e4m3fn" ||
               target_elem == "float8_e5m2") {
      push(round); push(sat); push(part4); push(ModeZeroing());
    }
  } else if (src_elem == "float16") {
    if (target_elem == "int32") {
      push(round); push(part2); push(ModeZeroing());
    } else if (target_elem == "int16") {
      push(round); push(sat); push(ModeZeroing());
    } else if (target_elem == "int8" || target_elem == "uint8") {
      push(round); push(sat); push(part2); push(ModeZeroing());
    } else if (target_elem == "bfloat16") {
      push(round); push(ModeZeroing());
    } else if (target_elem == "float32") {
      push(part2); push(ModeZeroing());
    }
  } else if (src_elem == "bfloat16") {
    if (target_elem == "int32") {
      push(round); push(sat); push(part2); push(ModeZeroing());
    } else if (target_elem == "float16") {
      push(sat); push(round); push(ModeZeroing());
    } else if (target_elem == "float32") {
      push(part2); push(ModeZeroing());
    }
  } else if (src_elem == "int32" && target_elem == "float32") {
    push(round); push(ModeZeroing());
  } else if (src_elem == "int16" && target_elem == "float32") {
    push(part2); push(ModeZeroing());
  } else if (src_elem == "int16" && target_elem == "float16") {
    push(round); push(ModeZeroing());
  } else if (src_elem == "int32" && (target_elem == "int16" || target_elem == "uint16")) {
    push(sat); push(part2); push(ModeZeroing());
  } else if (src_elem == "int16" && (target_elem == "int32" || target_elem == "uint32")) {
    push(part2); push(ModeZeroing());
  }
  if (args.size() == 2) {
    throw Unsupported("Unsupported vcvt conversion " + src_elem + "->" + target_elem);
  }
  return SimdCall(simd_vcvt(), VecDataType(target_elem), args);
}

/*! \brief ``tirx.reinterpret`` a signed index register as unsigned. */
PrimExpr UnsignedIndex(const PrimExpr& reg) {
  std::string dtype = ffi::DLDataTypeToString(reg.dtype()).operator std::string();
  if (dtype.compare(0, 3, "int") == 0) {
    dtype.replace(0, 3, "uint");
    return reinterpret(DataTypeOf(dtype), reg);
  }
  return reg;
}

/* ------------------------------------------------------------------------- */
/* structural cache keys                                                     */
/* ------------------------------------------------------------------------- */

/*!
 * \brief Fast structural key of an expression (stand-in for Python ``str(expr)``).
 *
 * ``ffi::ReprPrint`` costs ~7 us per call, which dominates the whole emitter;
 * this serializer walks the same node shapes and appends to a string, keeping
 * the "structurally equal => same key" guarantee that SSA reuse relies on.
 */
void ExprKeyInto(const PrimExpr& expr, std::string* out) {
  if (!expr.defined()) {
    *out += "~";
    return;
  }
  if (const auto* var = expr.as<tirx::VarNode>()) {
    *out += "v:";
    *out += var->name_hint.operator std::string();
    return;
  }
  if (const auto* imm = expr.as<IntImmNode>()) {
    *out += "i:";
    *out += std::to_string(imm->value);
    *out += ":";
    *out += std::to_string(imm->dtype.bits());
    return;
  }
  if (const auto* imm = expr.as<FloatImmNode>()) {
    *out += "f:";
    *out += std::to_string(imm->value);
    *out += ":";
    *out += std::to_string(imm->dtype.bits());
    return;
  }
  if (const auto* load = expr.as<tirx::BufferLoadNode>()) {
    *out += "L:";
    *out += load->buffer->name.operator std::string();
    *out += "(";
    for (const PrimExpr& index : load->indices) {
      ExprKeyInto(index, out);
      *out += ",";
    }
    *out += ")";
    return;
  }
  const char* binary = nullptr;
  if (expr.as<tirx::AddNode>()) {
    binary = "Add";
  } else if (expr.as<tirx::SubNode>()) {
    binary = "Sub";
  } else if (expr.as<tirx::MulNode>()) {
    binary = "Mul";
  } else if (expr.as<tirx::DivNode>()) {
    binary = "Div";
  } else if (expr.as<tirx::ModNode>()) {
    binary = "Mod";
  } else if (expr.as<tirx::FloorDivNode>()) {
    binary = "FloorDiv";
  } else if (expr.as<tirx::FloorModNode>()) {
    binary = "FloorMod";
  } else if (expr.as<tirx::MinNode>()) {
    binary = "Min";
  } else if (expr.as<tirx::MaxNode>()) {
    binary = "Max";
  } else if (expr.as<tirx::EQNode>()) {
    binary = "EQ";
  } else if (expr.as<tirx::NENode>()) {
    binary = "NE";
  } else if (expr.as<tirx::LTNode>()) {
    binary = "LT";
  } else if (expr.as<tirx::LENode>()) {
    binary = "LE";
  } else if (expr.as<tirx::GTNode>()) {
    binary = "GT";
  } else if (expr.as<tirx::GENode>()) {
    binary = "GE";
  } else if (expr.as<tirx::AndNode>()) {
    binary = "And";
  } else if (expr.as<tirx::OrNode>()) {
    binary = "Or";
  }
  if (binary != nullptr) {
    *out += binary;
    *out += "(";
    const auto* base = static_cast<const tirx::CmpOpNode<tirx::EQNode>*>(nullptr);
    (void)base;
    PrimExpr a;
    PrimExpr b;
    if (const auto* op = expr.as<tirx::AddNode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::SubNode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::MulNode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::DivNode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::ModNode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::FloorDivNode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::FloorModNode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::MinNode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::MaxNode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::EQNode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::NENode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::LTNode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::LENode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::GTNode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::GENode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::AndNode>()) {
      a = op->a; b = op->b;
    } else if (const auto* op = expr.as<tirx::OrNode>()) {
      a = op->a; b = op->b;
    }
    ExprKeyInto(a, out);
    *out += ",";
    ExprKeyInto(b, out);
    *out += ")";
    return;
  }
  if (const auto* op = expr.as<tirx::NotNode>()) {
    *out += "Not(";
    ExprKeyInto(op->a, out);
    *out += ")";
    return;
  }
  if (const auto* op = expr.as<tirx::CastNode>()) {
    *out += "Cast:";
    *out += ffi::DLDataTypeToString(op->dtype).operator std::string();
    *out += "(";
    ExprKeyInto(op->value, out);
    *out += ")";
    return;
  }
  if (const auto* op = expr.as<tirx::SelectNode>()) {
    *out += "Select(";
    ExprKeyInto(op->condition, out);
    *out += ",";
    ExprKeyInto(op->true_value, out);
    *out += ",";
    ExprKeyInto(op->false_value, out);
    *out += ")";
    return;
  }
  if (const auto* op = expr.as<tirx::CallNode>()) {
    *out += "C:";
    if (const auto* opn = op->op.as<OpNode>()) *out += opn->name;
    *out += "(";
    for (const PrimExpr& arg : op->args) {
      ExprKeyInto(arg, out);
      *out += ",";
    }
    *out += ")";
    return;
  }
  // Fallback: rare shapes keep the exact printed form.
  *out += PrintExpr(expr);
}

std::string ExprKey(const PrimExpr& expr) {
  std::string out;
  ExprKeyInto(expr, &out);
  return out;
}

/*! \brief Structural identity of a buffer (never Python object identity). */
std::string BufferKey(const tirx::Buffer& buffer) {
  // PrintExpr costs microseconds per call and this runs for every buffer of
  // every segment; the fast serializer gives the same equivalence relation.
  std::string out = buffer->data->name_hint.operator std::string() + "|" +
                    ffi::DLDataTypeToString(buffer->dtype).operator std::string() + "|" +
                    buffer.scope().operator std::string() + "|(";
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    out += (i == 0 ? "" : ",") + ExprKey(buffer->shape[i]);
  }
  out += ")|(";
  for (size_t i = 0; i < buffer->strides.size(); ++i) {
    out += (i == 0 ? "" : ",") + ExprKey(buffer->strides[i]);
  }
  out += ")|" + ExprKey(buffer->elem_offset);
  return out;
}

std::string IndexKey(const ffi::Array<PrimExpr>& indices) {
  std::string out;
  for (size_t i = 0; i < indices.size(); ++i) {
    out += (i == 0 ? "" : ",") + ExprKey(indices[i]);
  }
  return out;
}

std::string ValueKey(const ValueNode& node, const std::map<std::string, int64_t>& store_gen);

std::string AccessKey(const AccessInfo& access, const std::map<std::string, int64_t>& store_gen) {
  std::string buffer_key = BufferKey(access->buffer);
  int64_t generation = 0;
  if (!access->is_store) {
    auto it = store_gen.find(buffer_key);
    if (it != store_gen.end()) generation = it->second;
  }
  return buffer_key + "#" + IndexKey(access->indices) + "#" + ExprKey(access->offset) + "#" +
         std::to_string(generation);
}

std::string ValueKey(const ValueNode& node, const std::map<std::string, int64_t>& store_gen) {
  std::string out = node->recipe.operator std::string() + "|" +
                    node->dtype.operator std::string() + "|" +
                    node->detail.operator std::string() + "|";
  out += node->scalar.has_value() ? ExprKey(*node->scalar) : std::string("-");
  out += "|";
  if (node->access.has_value()) out += AccessKey(*node->access, store_gen);
  out += "|(";
  for (size_t i = 0; i < node->args.size(); ++i) {
    out += (i == 0 ? "" : ";") + ValueKey(Downcast<ValueNode>(node->args[i]), store_gen);
  }
  out += ")|";
  ValueNode predicate = PredicateOf(node);
  if (predicate.defined()) out += ValueKey(predicate, store_gen);
  return out;
}

std::vector<std::string> LoadBufferKeys(const PrimExpr& expr) {
  std::vector<std::string> keys;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef& node) {
    const auto* load = node.as<tirx::BufferLoadNode>();
    if (load != nullptr) keys.push_back(BufferKey(load->buffer));
  });
  return keys;
}

/* ------------------------------------------------------------------------- */
/* program-shared prologue                                                   */
/* ------------------------------------------------------------------------- */

class ProgramPrologue {
 public:
  std::vector<Stmt> statements;
  std::map<std::pair<std::string, int>, tirx::Var> masks;
  std::map<std::pair<int64_t, const void*>, tirx::Var> ramps;
};

/*! \brief Segment-to-segment register forwarding entry. */
struct CarryEntry {
  tirx::Var axis;
  PrimExpr value;
};

/* ------------------------------------------------------------------------- */
/* _Emitter                                                                  */
/* ------------------------------------------------------------------------- */

class Emitter {
 public:
  Emitter(const VectorPlan& plan, PrimExpr chunk, std::string prefix, bool masked_tail = false,
          int64_t chunk_min = 0, std::optional<int64_t> chunk_max = std::nullopt,
          ProgramPrologue* shared = nullptr,
          std::map<std::string, CarryEntry>* carry = nullptr, bool allow_carry = false)
      : plan_(plan),
        chunk_(std::move(chunk)),
        prefix_(std::move(prefix)),
        masked_tail_(masked_tail),
        chunk_min_(chunk_min),
        chunk_max_(chunk_max),
        shared_(shared),
        carry_(carry != nullptr ? carry : &owned_carry_),
        allow_carry_(allow_carry) {
    lane_base_ = chunk_ * IntImm32(plan_->lanes);
    for (const StorePlan& store : plan_->pre_stores) written_buffers_.push_back(store->spec->buffer);
    for (const StorePlan& store : plan_->stores) written_buffers_.push_back(store->spec->buffer);
    for (const StorePlan& store : plan_->post_stores) written_buffers_.push_back(store->spec->buffer);
  }

  const VectorPlan& plan() const { return plan_; }
  const std::string& prefix() const { return prefix_; }
  int64_t counter() const { return counter_; }
  void set_counter(int64_t value) { counter_ = value; }
  std::vector<Stmt>& lets() { return lets_; }
  std::vector<Stmt>& prologue() { return prologue_; }
  PrimExpr lane_base() const { return lane_base_; }
  void set_lane_base(PrimExpr value) { lane_base_ = std::move(value); }
  bool masked_tail() const { return masked_tail_; }
  void set_masked_tail(bool value) { masked_tail_ = value; }
  void clear_forward() { forward_.clear(); }

  /* -- SSA binding ------------------------------------------------------ */
  tirx::Var Bind(const PrimExpr& expr, std::vector<Stmt>* target) {
    tirx::Var var("v" + prefix_ + std::to_string(counter_), expr.dtype());
    ++counter_;
    target->push_back(tirx::Bind(var, expr));
    return var;
  }

  tirx::Var PrologueBind(const PrimExpr& expr) {
    tirx::Var var = Bind(expr, &prologue_);
    prologue_vars_.insert(var.get());
    return var;
  }

  /* -- masks ------------------------------------------------------------ */
  std::optional<tirx::Var> SharedMask(const std::pair<std::string, int>& key,
                                      const std::string& name, const PrimExpr& expr) {
    if (shared_ == nullptr) return std::nullopt;
    tirx::Var var;
    auto it = shared_->masks.find(key);
    if (it == shared_->masks.end()) {
      var = tirx::Var(name, DataType::Bool(256));
      shared_->statements.push_back(tirx::Bind(var, expr));
      shared_->masks[key] = var;
    } else {
      var = it->second;
    }
    prologue_vars_.insert(var.get());
    shared_mask_ids_.insert(var.get());
    return var;
  }

  PrimExpr AllMask(int bits) {
    auto key = std::make_pair(std::string("all"), bits);
    auto it = mask_vars_.find(key);
    if (it != mask_vars_.end()) return it->second;
    PrimExpr expr = Pset(bits, "PAT_ALL");
    std::optional<tirx::Var> shared_var = SharedMask(key, "mask_" + std::to_string(bits), expr);
    PrimExpr var;
    if (shared_var.has_value()) {
      var = *shared_var;
    } else {
      tirx::Var local_var("mask" + prefix_ + std::to_string(bits), DataType::Bool(256));
      prologue_.push_back(tirx::Bind(local_var, expr));
      prologue_vars_.insert(local_var.get());
      var = local_var;
    }
    mask_vars_[key] = var;
    return var;
  }

  PrimExpr Lane0Mask(int bits) {
    auto key = std::make_pair(std::string("lane0"), bits);
    auto it = mask_vars_.find(key);
    if (it != mask_vars_.end()) return it->second;
    PrimExpr expr = Pset(bits, "PAT_VL1");
    std::optional<tirx::Var> shared_var = SharedMask(key, "lane0_" + std::to_string(bits), expr);
    PrimExpr var;
    if (shared_var.has_value()) {
      var = *shared_var;
    } else {
      tirx::Var local_var("lane0" + prefix_ + std::to_string(bits), DataType::Bool(256));
      prologue_.push_back(tirx::Bind(local_var, expr));
      prologue_vars_.insert(local_var.get());
      var = local_var;
    }
    mask_vars_[key] = var;
    return var;
  }

  PrimExpr DataMask(const std::string& dtype_like) { return AllMask(MaskBits(dtype_like)); }

  PrimExpr TailMask(int bits) {
    if (!masked_tail_) return PrimExpr();
    auto key = std::make_pair(std::string("tail"), bits);
    auto it = mask_vars_.find(key);
    if (it != mask_vars_.end()) return it->second;
    PrimExpr var;
    if (plan_->tail.has_value()) {
      var = PrologueBind(UpdateMask(IntImm32(*plan_->tail), bits));
    } else {
      PrimExpr remaining = plan_->vector_axis->extent - chunk_ * IntImm32(plan_->lanes);
      PrimExpr count = min(remaining, IntImm32(plan_->lanes));
      var = Bind(UpdateMask(count, bits), &lets_);
    }
    mask_vars_[key] = var;
    return var;
  }

  PrimExpr PredicateCombine(const PrimExpr& lhs, const PrimExpr& rhs, int bits) {
    return Bind(MethodMask(simd_pand(), {lhs, rhs, AllMask(bits)}), &lets_);
  }

  PrimExpr Predicate(const ValueNode& node, int bits) {
    std::string key = CachedValueKey(node) + "@" + std::to_string(bits);
    auto it = predicates_.find(key);
    if (it != predicates_.end()) return it->second;
    PrimExpr mask;
    if (node->is_uniform) {
      PrimExpr lane_count = tirx::Select(Scalar(node), IntImm32(plan_->lanes), IntImm32(0));
      mask = Bind(UpdateMask(lane_count, bits), &lets_);
    } else if (node->recipe == "pnot") {
      mask = Bind(MethodMask(simd_pnot(), {Predicate(Downcast<ValueNode>(node->args[0]), bits),
                                           AllMask(bits)}),
                  &lets_);
    } else if (node->recipe == "pand") {
      mask = PredicateCombine(Predicate(Downcast<ValueNode>(node->args[0]), bits),
                              Predicate(Downcast<ValueNode>(node->args[1]), bits), bits);
    } else if (node->recipe == "por") {
      mask = Bind(MethodMask(simd_por(), {Predicate(Downcast<ValueNode>(node->args[0]), bits),
                                          Predicate(Downcast<ValueNode>(node->args[1]), bits),
                                          AllMask(bits)}),
                  &lets_);
    } else {
      mask = Value(node);
    }
    predicates_[key] = mask;
    return mask;
  }

  PrimExpr RampMask(int bits) {
    PrimExpr tail = TailMask(bits);
    return tail.defined() ? tail : AllMask(bits);
  }

  bool GuardFree(int64_t free_until, const ffi::Optional<int64_t>& free_from) {
    if (free_from.has_value() && *free_from <= chunk_min_) return true;
    return chunk_max_.has_value() && free_until >= *chunk_max_;
  }

  PrimExpr LoadGuard(const ValueNode& node, int bits) {
    PrimExpr mask = TailMask(bits);
    ValueNode guard_node = PredicateOf(node);
    if (guard_node.defined() &&
        GuardFree(node->predicate_free_until, node->predicate_free_from)) {
      guard_node = ValueNode();
    }
    if (guard_node.defined()) {
      PrimExpr guard = Predicate(guard_node, bits);
      mask = mask.defined() ? PredicateCombine(mask, guard, bits) : guard;
    }
    return mask;
  }

  /* -- addresses -------------------------------------------------------- */
  ffi::Array<PrimExpr> AddrIndices(const AccessInfo& access, bool uniform) {
    PrimExpr base = uniform ? IntImm32(0) : lane_base_;
    ffi::Array<PrimExpr> out;
    for (const PrimExpr& index : access->indices) {
      out.push_back(analyzer_.Simplify(tirx::Substitute(
          index, [&](const tirx::Var& var) -> ffi::Optional<PrimExpr> {
            if (var.same_as(plan_->vector_axis->var)) return ffi::Optional<PrimExpr>(base);
            return ffi::Optional<PrimExpr>();
          })));
    }
    return out;
  }

  PrimExpr Addr(const AccessInfo& access, bool uniform = false) {
    return tirx::BufferLoad(access->buffer, AddrIndices(access, uniform));
  }

  PrimExpr IndexAddr(const AccessInfo& access) {
    ffi::Array<PrimExpr> zeros;
    for (size_t i = 0; i < access->indices.size(); ++i) zeros.push_back(IntImm32(0));
    return tirx::BufferLoad(access->buffer, zeros);
  }

  /* -- scalar operands -------------------------------------------------- */
  PrimExpr Scalar(const ValueNode& node) {
    const std::string& recipe = node->recipe;
    if (recipe == "imm" || recipe == "uniform_expr") {
      if (!node->scalar.has_value()) {
        throw Unsupported("cannot scalarize vector value '" + recipe + "'");
      }
      return *node->scalar;
    }
    if (recipe == "uniform_load") {
      return Addr(*node->access, /*uniform=*/true);
    }
    if (recipe == "vbrc") {
      return Scalar(Downcast<ValueNode>(node->args[0]));
    }
    throw Unsupported("cannot scalarize vector value '" + recipe + "'");
  }

  /* -- writes / forwarding ---------------------------------------------- */
  void NoteWrite(const tirx::Buffer& buffer) {
    const std::string& key = CachedBufferKey(buffer);
    store_gen_[key] = store_gen_.count(key) != 0 ? store_gen_[key] + 1 : 1;
    value_keys_.clear();
    if (!forward_.empty()) {
      for (auto it = forward_.begin(); it != forward_.end();) {
        if (it->first.rfind(key, 0) == 0) {
          it = forward_.erase(it);
        } else {
          ++it;
        }
      }
    }
    if (!carry_->empty()) {
      for (auto it = carry_->begin(); it != carry_->end();) {
        if (it->first.rfind(key, 0) == 0) {
          it = carry_->erase(it);
        } else {
          ++it;
        }
      }
    }
  }

  void NoteStore(const StorePlan& store, const PrimExpr& value) {
    if (AccessClassFromName(store->kind_value.operator std::string()) == AccessClass::kContiguous) {
      forward_[CachedBufferKey(store->spec->buffer) + "^" + IndexKey(store->spec->indices)] = value;
    }
  }

  void Promote(const ValueNode& node, const PrimExpr& vector) {
    promoted_[CachedValueKey(node)] = vector;
  }

  void RegisterCarry(const StorePlan& store, const PrimExpr& value, const tirx::Var& axis) {
    carry_->insert_or_assign(
        CachedBufferKey(store->spec->buffer) + "^" + IndexKey(store->spec->indices),
        CarryEntry{axis, value});
  }

  PrimExpr CarryEntryLookup(const std::string& fwd_key, const std::string& dtype) {
    if (!allow_carry_ || carry_->empty()) return PrimExpr();
    auto it = carry_->find(fwd_key);
    if (it == carry_->end()) return PrimExpr();
    if (!it->second.axis.same_as(plan_->vector_axis->var)) return PrimExpr();
    if (ffi::DLDataTypeToString(it->second.value.dtype()).operator std::string() != dtype) {
      return PrimExpr();
    }
    return it->second.value;
  }

  /* -- values ------------------------------------------------------------ */
  PrimExpr Value(const ValueNode& node) {
    const std::string& key = CachedValueKey(node);
    auto it = memo_.find(key);
    if (it != memo_.end()) return it->second;
    PrimExpr expr = Build(node);
    memo_[key] = expr;
    return expr;
  }

  PrimExpr VectorOperand(const ValueNode& node) {
    const std::string& key = CachedValueKey(node);
    auto it = promoted_.find(key);
    if (it != promoted_.end()) return it->second;
    if (IsScalarOnlyRecipe(node->recipe.operator std::string())) return Dup(node);
    return Value(node);
  }

  PrimExpr Dup(const ValueNode& node) {
    std::string elem = ElemDtype(node->dtype.operator std::string());
    PrimExpr scalar = Scalar(node);
    if (ScalarHoistable(node)) {
      auto key = std::make_pair(PrintExpr(scalar), elem);
      auto it = dups_.find(key);
      if (it != dups_.end()) return it->second;
      tirx::Var var = PrologueBind(Vdup(scalar, elem, DataMask(elem)));
      dups_[key] = var;
      return var;
    }
    return Bind(Vdup(scalar, elem, DataMask(elem)), &lets_);
  }

  bool ScalarHoistable(const ValueNode& node) {
    PrimExpr expr = Scalar(node);
    if (!expr.defined()) return true;
    std::vector<tirx::Var> axis_vars;
    for (const Axis& axis : plan_->axes) axis_vars.push_back(axis->var);
    std::set<std::string> written;
    for (const tirx::Buffer& buffer : written_buffers_) written.insert(BufferKey(buffer));
    bool blocked = false;
    tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef& sub) {
      const auto* load = sub.as<tirx::BufferLoadNode>();
      if (load != nullptr && written.count(BufferKey(load->buffer)) != 0) blocked = true;
      const auto* var = sub.as<tirx::VarNode>();
      if (var != nullptr) {
        for (const tirx::Var& axis_var : axis_vars) {
          if (axis_var.same_as(ffi::GetRef<tirx::Var>(var))) blocked = true;
        }
      }
    });
    return !blocked;
  }

  /* -- ramp -------------------------------------------------------------- */
  PrimExpr Ramp(int64_t stride, const PrimExpr& mask) {
    auto key = std::make_pair(stride, mask.get());
    auto it = ramps_.find(key);
    if (it != ramps_.end()) return it->second;
    PrimExpr expr = SimdCall(
        simd_vmuls(), VecDataType("uint32"),
        {Vci(IntImm32(0), "uint32"), IntImm32(stride), mask, StrImm(ModeZeroing())});
    PrimExpr cached;
    const tirx::VarNode* mask_var = mask.as<tirx::VarNode>();
    if (prologue_vars_.count(mask_var) != 0) {
      if (shared_ != nullptr && shared_mask_ids_.count(mask_var) != 0) {
        auto shared_it = shared_->ramps.find(key);
        if (shared_it == shared_->ramps.end()) {
          tirx::Var var("ramp_" + std::to_string(stride), expr.dtype());
          shared_->statements.push_back(tirx::Bind(var, expr));
          shared_->ramps[key] = var;
          cached = var;
        } else {
          cached = shared_it->second;
        }
        prologue_vars_.insert(cached.as<tirx::VarNode>());
      } else {
        cached = PrologueBind(expr);
      }
    } else {
      cached = Bind(expr, &lets_);
    }
    ramps_[key] = cached;
    return cached;
  }

  void SetMemo(const std::string& key, const PrimExpr& value) { memo_[key] = value; }

  void ClearChunkCaches() {
    memo_.clear();
    predicates_.clear();
  }

  /*!
   * \brief Structural value key, memoized per node.
   *
   * Printing scalars / access keys is by far the most expensive part of the
   * emitter. The plan DAG is immutable, and the key only depends on the write
   * generation of each buffer, so the cache is dropped whenever a write bumps a
   * generation (``NoteWrite``).
   */
  const std::string& CachedValueKey(const ValueNode& node) {
    auto it = value_keys_.find(node.get());
    if (it != value_keys_.end()) return it->second;
    return value_keys_.emplace(node.get(), ValueKey(node, store_gen_)).first->second;
  }

  const std::string& CachedBufferKey(const tirx::Buffer& buffer) {
    auto it = buffer_keys_.find(buffer.get());
    if (it != buffer_keys_.end()) return it->second;
    return buffer_keys_.emplace(buffer.get(), BufferKey(buffer)).first->second;
  }

  void AppendLets(std::vector<Stmt> stmts) {
    for (Stmt& stmt : stmts) lets_.push_back(std::move(stmt));
  }

  std::vector<Stmt> TakeLets() {
    std::vector<Stmt> out = std::move(lets_);
    lets_.clear();
    return out;
  }

  const std::map<std::string, int64_t>& store_gen() const { return store_gen_; }
  ProgramPrologue* shared() const { return shared_; }
  std::map<std::string, CarryEntry>* carry() const { return carry_; }
  bool allow_carry() const { return allow_carry_; }

  std::vector<Stmt> lets_;
  std::vector<Stmt> prologue_;

 private:
  PrimExpr Build(const ValueNode& node);

  const VectorPlan& plan_;
  PrimExpr chunk_;
  std::string prefix_;
  PrimExpr lane_base_;
  bool masked_tail_;
  int64_t chunk_min_;
  std::optional<int64_t> chunk_max_;
  ProgramPrologue* shared_;
  std::map<std::string, CarryEntry> owned_carry_;
  std::map<std::string, CarryEntry>* carry_;
  bool allow_carry_;
  int64_t counter_{0};
  std::map<std::pair<std::string, int>, PrimExpr> mask_vars_;
  std::map<std::string, PrimExpr> predicates_;
  std::map<std::pair<int64_t, const void*>, PrimExpr> ramps_;
  std::set<const tirx::VarNode*> prologue_vars_;
  std::set<const tirx::VarNode*> shared_mask_ids_;
  std::map<std::string, int64_t> store_gen_;
  std::vector<tirx::Buffer> written_buffers_;
  std::map<std::pair<std::string, std::string>, PrimExpr> dups_;
  std::map<std::pair<std::string, std::string>, PrimExpr> vci_;
  std::map<std::string, PrimExpr> forward_;
  std::map<std::string, PrimExpr> promoted_;
  std::map<std::string, PrimExpr> memo_;
  std::map<const ValueNodeNode*, std::string> value_keys_;
  std::map<const tirx::BufferNode*, std::string> buffer_keys_;
  arith::Analyzer analyzer_;
};

PrimExpr Emitter::Build(const ValueNode& node) {
  const std::string& recipe = node->recipe;
  const std::vector<ValueNode> args = ArgsOf(node);

  if (recipe == "imm") {
    if (!node->scalar.has_value()) {
      throw Unsupported("cannot scalarize vector value 'imm'");
    }
    return *node->scalar;
  }
  if (recipe == "lane_index") {
    // Absolute lane value = chunk base + lane id (lane-varying compares).
    std::string elem = ElemDtype(node->dtype.operator std::string());
    auto key = std::make_pair(PrintExpr(lane_base_), elem);
    auto it = vci_.find(key);
    if (it != vci_.end()) return it->second;
    tirx::Var var = Bind(Vci(lane_base_, elem), &lets_);
    vci_[key] = var;
    return var;
  }
  if (recipe == "vload") {
    std::string fwd_key = CachedBufferKey((*node->access)->buffer) + "^" +
                          IndexKey((*node->access)->indices);
    auto fwd = forward_.find(fwd_key);
    if (fwd != forward_.end() &&
        ffi::DLDataTypeToString(fwd->second.dtype()).operator std::string() ==
            node->dtype.operator std::string()) {
      return fwd->second;
    }
    if (!masked_tail_ && !PredicateOf(node).defined()) {
      PrimExpr carried = CarryEntryLookup(fwd_key, node->dtype.operator std::string());
      if (carried.defined()) return carried;
    }
    int bits = MaskBits(node->dtype.operator std::string());
    PrimExpr mask = LoadGuard(node, bits);
    if (!mask.defined()) {
      return Bind(Vld(Addr(*node->access), (*node->access)->buffer, "NORM"), &lets_);
    }
    // vld has no predicate form: the tail/guard case degrades to a masked
    // vgather2 with the relative lane offsets (D10 / P0-3).
    return Bind(Vgather2(Addr(*node->access), (*node->access)->buffer, Vci(IntImm32(0), "uint32"),
                         mask),
                &lets_);
  }
  if (recipe == "vload_strided") {
    AxisClass classified = AxisClassOf(*node->access, plan_->vector_axis);
    int bits = MaskBits(node->dtype.operator std::string());
    PrimExpr guard = LoadGuard(node, bits);
    if (!guard.defined()) guard = AllMask(bits);
    PrimExpr index = Ramp(classified.stride.value_or(0), RampMask(bits));
    return Bind(
        Vgather2(Addr(*node->access), (*node->access)->buffer, index, guard), &lets_);
  }
  if (recipe == "gather") {
    int bits = MaskBits(node->dtype.operator std::string());
    PrimExpr mask = LoadGuard(node, bits);
    if (!mask.defined()) mask = AllMask(bits);
    PrimExpr index = UnsignedIndex(Value(args[0]));
    return Bind(Vgather2(IndexAddr(*node->access), (*node->access)->buffer, index, mask), &lets_);
  }
  if (recipe == "uniform_load" || recipe == "uniform_expr" || recipe == "vbrc") {
    ValueNode operand = recipe == "vbrc" ? args[0] : node;
    return Dup(operand);
  }
  const std::string recipe_str = recipe;
  if (IsScalarVectorRecipe(recipe_str) && args.size() == 2) {
    // Scalar-vector variant: the "scalar" operand may already be a promoted
    // vector register (a local.var accumulator), in which case the vector-vector
    // form is required.
    PrimExpr vector = Value(args[0]);
    const std::string& operand_key = CachedValueKey(args[1]);
    auto promoted = memo_.find(operand_key);
    if (promoted != memo_.end()) {
      const Op& op = BinaryOp(ScalarToVectorBinary(recipe_str));
      return Bind(SimdCall(op, DataTypeOf(node->dtype.operator std::string()),
                           {vector, promoted->second, DataMask(node->dtype.operator std::string()),
                            StrImm(ModeZeroing())}),
                  &lets_);
    }
    return Bind(SimdCall(ScalarBinaryOp(recipe_str),
                         DataTypeOf(node->dtype.operator std::string()),
                         {vector, Scalar(args[1]), DataMask(node->dtype.operator std::string()),
                          StrImm(ModeZeroing())}),
                &lets_);
  }
  if (recipe == "vshrs" || recipe == "vshls") {
    return Bind(SimdCall(ScalarBinaryOp(recipe_str),
                         DataTypeOf(node->dtype.operator std::string()),
                         {Value(args[0]), *node->scalar,
                          DataMask(node->dtype.operator std::string()), StrImm(ModeZeroing())}),
                &lets_);
  }
  if (IsElementwiseUnaryRecipe(recipe_str) && args.size() == 1) {
    return Bind(SimdCall(UnaryOp(recipe_str),
                         DataTypeOf(node->dtype.operator std::string()),
                         {Value(args[0]), DataMask(node->dtype.operator std::string()),
                          StrImm(ModeZeroing())}),
                &lets_);
  }
  if (IsElementwiseBinaryRecipe(recipe_str) && args.size() == 2) {
    return Bind(SimdCall(BinaryOp(recipe_str),
                         DataTypeOf(node->dtype.operator std::string()),
                         {Value(args[0]), Value(args[1]),
                          DataMask(node->dtype.operator std::string()), StrImm(ModeZeroing())}),
                &lets_);
  }
  if (recipe == "vcmp") {
    return Bind(SimdCall(simd_vcmp(), BoolMaskDtype(),
                         {Value(args[0]), Value(args[1]), DataMask(args[0]->dtype.operator std::string()),
                          StrImm(node->detail.operator std::string())}),
                &lets_);
  }
  if (recipe == "vcmps") {
    return Bind(SimdCall(simd_vcmps(), BoolMaskDtype(),
                         {Value(args[0]), Scalar(args[1]),
                          DataMask(args[0]->dtype.operator std::string()),
                          StrImm(node->detail.operator std::string())}),
                &lets_);
  }
  if (recipe == "vsel") {
    // args = [cond, true, false]; vsel order is (t, f, mask).
    PrimExpr true_value = VectorOperand(args[1]);
    PrimExpr false_value = VectorOperand(args[2]);
    int bits = MaskBits(ElemDtype(node->dtype.operator std::string()));
    return Bind(SimdCall(simd_vsel(),
                         DataTypeOf(node->dtype.operator std::string()),
                         {true_value, false_value, Predicate(args[0], bits)}),
                &lets_);
  }
  if (recipe == "pand" || recipe == "por") {
    const Op& op = recipe == "pand" ? simd_pand() : simd_por();
    return Bind(MethodMask(op, {Value(args[0]), Value(args[1]), AllMask(kPredicateMaskBits)}),
                &lets_);
  }
  if (recipe == "pnot") {
    return Bind(MethodMask(simd_pnot(), {Value(args[0]), AllMask(kPredicateMaskBits)}), &lets_);
  }
  if (recipe == "vcvt") {
    PrimExpr src = Value(args[0]);
    std::string target = node->detail.operator std::string();
    int bits = std::min(MaskBits(ffi::DLDataTypeToString(src.dtype()).operator std::string()),
                        MaskBits(target));
    return Bind(Vcvt(src, target, AllMask(bits)), &lets_);
  }
  throw Unsupported("no emitter for value recipe '" + recipe + "'");
}

/* ------------------------------------------------------------------------- */
/* store emission                                                            */
/* ------------------------------------------------------------------------- */

PrimExpr StoreMask(const StorePlan& store, Emitter* emitter, int bits) {
  PrimExpr mask;
  if (store->mask.has_value() &&
      !emitter->GuardFree(store->mask_free_until, store->mask_free_from)) {
    // The predicate is provably true in this block: use the all-lane mask and
    // skip the whole vcmps chain.
    mask = emitter->Predicate(*store->mask, bits);
  }
  PrimExpr tail = emitter->TailMask(bits);
  if (tail.defined()) {
    mask = mask.defined() ? emitter->PredicateCombine(mask, tail, bits) : tail;
  }
  if (!mask.defined()) mask = emitter->AllMask(bits);
  return mask;
}

void EmitVectorWrite(const StorePlan& store, Emitter* emitter, int bits, const PrimExpr& mask) {
  PrimExpr value = emitter->VectorOperand(store->value);
  bool uniform = AccessClassFromName(store->kind_value.operator std::string()) ==
                 AccessClass::kUniform;
  std::string dist;
  const std::string lane_mode = store->store_lane_mode.operator std::string();
  if (lane_mode == "pk_half" || lane_mode == "pk_quarter") {
    // Mixed element widths: the value register is wider than the chunk, so the
    // store must write exactly ``plan.lanes`` elements through the packed form
    // (verified cast-kernel recipe: ``vcvt`` with ``PART_EVEN`` + ``PK_B*``).
    int64_t pack_bits = store->store_pack_bits.has_value() ? *store->store_pack_bits : bits;
    dist = (lane_mode == "pk_half" ? "PK_B" : "PK4_B") + std::to_string(pack_bits);
  } else {
    dist = (uniform ? "ONEPT_B" : "NORM_B") + std::to_string(bits);
  }
  emitter->lets_.push_back(tirx::Evaluate(
      Vsts(emitter->Addr(store->access, uniform), store->spec->buffer, value, mask, dist)));
  emitter->NoteWrite(store->spec->buffer);
  emitter->NoteStore(store, value);
}

void EmitStore(const VectorPlan& plan, const StorePlan& store, Emitter* emitter) {
  std::string elem = ElemDtype(store->spec->buffer->dtype);
  int bits = MaskBits(elem);
  PrimExpr mask = StoreMask(store, emitter, bits);
  AccessClass kind = AccessClassFromName(store->kind_value.operator std::string());

  if (kind == AccessClass::kContiguous || kind == AccessClass::kUniform) {
    EmitVectorWrite(store, emitter, bits, mask);
    return;
  }

  PrimExpr value = emitter->VectorOperand(store->value);
  if (kind == AccessClass::kStrided) {
    // Constant *positive* stride writes use the stride ramp; a negative stride
    // falls through to the index-vector channel (D8).
    AxisClass classified = AxisClassOf(store->access, plan->vector_axis);
    if (classified.stride.has_value() && *classified.stride > 0) {
      PrimExpr index = emitter->Ramp(*classified.stride, emitter->RampMask(bits));
      emitter->lets_.push_back(tirx::Evaluate(
          Vscatter(value, emitter->Addr(store->access), store->spec->buffer, index, mask)));
      emitter->NoteWrite(store->spec->buffer);
      return;
    }
  }

  if (!store->index.has_value()) {
    throw Unsupported("no per-lane index vector for a " +
                      store->kind_value.operator std::string() + " store");
  }
  PrimExpr index = UnsignedIndex(emitter->Value(*store->index));
  emitter->lets_.push_back(
      tirx::Evaluate(Vscatter(value, emitter->IndexAddr(store->access), store->spec->buffer,
                             index, mask)));
  emitter->NoteWrite(store->spec->buffer);
}

void EmitOuterStore(const StorePlan& store, Emitter* emitter) {
  AccessClass kind = AccessClassFromName(store->kind_value.operator std::string());
  if (kind != AccessClass::kUniform) {
    throw Unsupported(
        "a statement outside the serial loop must be uniform along the vector axis (got " +
        store->kind_value.operator std::string() + ")");
  }
  std::string elem = ElemDtype(store->spec->buffer->dtype);
  int bits = MaskBits(elem);
  if (!store->mask.has_value() && store->value->is_uniform) {
    PrimExpr scalar;
    bool ok = true;
    try {
      scalar = emitter->Scalar(store->value);
    } catch (const Unsupported&) {
      ok = false;
    }
    if (ok && scalar.defined()) {
      emitter->lets_.push_back(
          tirx::BufferStore(store->spec->buffer, scalar,
                            emitter->AddrIndices(store->access, /*uniform=*/true)));
      emitter->NoteWrite(store->spec->buffer);
      return;
    }
  }
  PrimExpr mask = store->mask.has_value() ? emitter->Predicate(*store->mask, bits)
                                          : emitter->AllMask(bits);
  EmitVectorWrite(store, emitter, bits, mask);
}

/* ------------------------------------------------------------------------- */
/* strip-mining                                                              */
/* ------------------------------------------------------------------------- */

std::vector<Stmt> SeqOfVec(const std::vector<Stmt>& stmts) {
  ffi::Array<Stmt> array;
  for (const Stmt& stmt : stmts) array.push_back(stmt);
  Stmt folded = SeqOf(array);
  if (!folded.defined()) return {};
  return {folded};
}

struct Block {
  std::vector<Stmt> prologue;
  std::vector<Stmt> lets;
};

using MakeBlock = std::function<Block(const PrimExpr& chunk_index, bool masked_tail,
                                      const std::string& label, int64_t chunk_min,
                                      std::optional<int64_t> chunk_max)>;

std::pair<std::vector<Stmt>, std::vector<Stmt>> StripMine(const VectorPlan& plan,
                                                          const tirx::Var& chunk,
                                                          const MakeBlock& make_block) {
  std::vector<Stmt> prologue;

  auto block = [&](const PrimExpr& index, bool masked_tail, const std::string& label,
                   int64_t chunk_min, std::optional<int64_t> chunk_max) {
    Block result = make_block(index, masked_tail, label, chunk_min, chunk_max);
    for (Stmt& stmt : result.prologue) prologue.push_back(std::move(stmt));
    return result.lets;
  };

  std::function<std::vector<Stmt>(int64_t, int64_t, const std::string&)> range_blocks =
      [&](int64_t lo, int64_t hi, const std::string& label) -> std::vector<Stmt> {
    if (hi <= lo) return {};
    if (hi - lo == 1) {
      return SeqOfVec(block(IntImm32(lo), false, label, lo, hi));
    }
    std::vector<Stmt> body = SeqOfVec(block(chunk, false, label, lo, hi));
    std::vector<Stmt> out;
    out.push_back(tirx::For(chunk, IntImm32(lo), IntImm32(hi), ForKind::kSerial,
                            body.empty() ? Stmt() : body[0]));
    return out;
  };

  if (!plan->chunks.has_value()) {
    PrimExpr extent = plan->vector_axis->extent;
    PrimExpr full = tirx::FloorDiv(extent, IntImm32(plan->lanes));
    std::vector<Stmt> loop_body =
        SeqOfVec(block(chunk, false, "", 0, std::nullopt));
    std::vector<Stmt> blocks;
    blocks.push_back(tirx::For(chunk, IntImm32(0), full, ForKind::kSerial,
                               loop_body.empty() ? Stmt() : loop_body[0]));
    PrimExpr tail_chunk = tirx::FloorDiv(extent, IntImm32(plan->lanes));
    std::vector<Stmt> tail_body =
        SeqOfVec(block(tail_chunk, true, "tail_", 0, std::nullopt));
    blocks.push_back(tirx::IfThenElse(
        tirx::GT(tirx::FloorMod(extent, IntImm32(plan->lanes)), IntImm32(0)),
        tail_body.empty() ? Stmt() : tail_body[0], std::nullopt));
    return {prologue, blocks};
  }

  std::vector<int64_t> cuts(plan->guard_boundaries.begin(), plan->guard_boundaries.end());

  std::function<std::vector<Stmt>(int64_t, int64_t)> split_range =
      [&](int64_t lo, int64_t hi) -> std::vector<Stmt> {
    std::vector<int64_t> points = {lo};
    for (int64_t cut : cuts) {
      if (cut > lo && cut < hi) points.push_back(cut);
    }
    points.push_back(hi);
    std::vector<Stmt> blocks;
    for (size_t index = 0; index + 1 < points.size(); ++index) {
      std::string label = points.size() == 2 ? "" : "gr" + std::to_string(index) + "_";
      for (Stmt& stmt : range_blocks(points[index], points[index + 1], label)) {
        blocks.push_back(std::move(stmt));
      }
    }
    return blocks;
  };

  bool has_tail = plan->tail.has_value() && *plan->tail != 0;
  if (!has_tail) {
    if (*plan->chunks == 1) {
      // A single chunk is straight-line code: ``for chunk in range(1)`` only
      // confuses the backend.
      return {prologue, block(IntImm32(0), false, "", 0, 1)};
    }
    return {prologue, split_range(0, *plan->chunks)};
  }

  int64_t full = *plan->chunks - 1;
  std::vector<Stmt> blocks = split_range(0, full);
  for (Stmt& stmt : SeqOfVec(block(IntImm32(full), true, "tail_", full, full + 1))) {
    blocks.push_back(std::move(stmt));
  }
  return {prologue, blocks};
}

std::pair<std::vector<Stmt>, std::vector<Stmt>> EmitElementwiseChunks(
    const VectorPlan& plan, const tirx::Var& chunk, const std::string& prefix,
    ProgramPrologue* shared, std::map<std::string, CarryEntry>* carry, bool allow_carry) {
  MakeBlock make_block = [&](const PrimExpr& chunk_index, bool masked_tail,
                             const std::string& label, int64_t chunk_min,
                             std::optional<int64_t> chunk_max) -> Block {
    Emitter body(plan, chunk_index, prefix + label, masked_tail, chunk_min, chunk_max, shared,
                 carry, allow_carry);
    for (const StorePlan& store : plan->stores) EmitStore(plan, store, &body);
    return Block{body.prologue_, body.lets_};
  };
  return StripMine(plan, chunk, make_block);
}

/* ------------------------------------------------------------------------- */
/* LICM helpers                                                              */
/* ------------------------------------------------------------------------- */

std::pair<std::vector<Stmt>, std::vector<Stmt>> SplitLoopInvariant(
    const std::vector<Stmt>& lets, const std::vector<Axis>& inner,
    const std::set<std::string>& written_keys, const std::vector<tirx::Var>& pinned) {
  std::vector<tirx::Var> inner_vars;
  for (const Axis& axis : inner) inner_vars.push_back(axis->var);
  std::vector<tirx::Var> loop_defined = pinned;
  std::vector<Stmt> hoisted;
  std::vector<Stmt> kept;
  for (const Stmt& stmt : lets) {
    const auto* bind = stmt.as<tirx::BindNode>();
    if (bind == nullptr) {
      kept.push_back(stmt);
      continue;
    }
    bool blocked = false;
    for (const tirx::Var& var : FreeVars(bind->value)) {
      for (const tirx::Var& inner_var : inner_vars) {
        if (var.same_as(inner_var)) blocked = true;
      }
      for (const tirx::Var& loop_var : loop_defined) {
        if (var.same_as(loop_var)) blocked = true;
      }
    }
    for (const std::string& key : LoadBufferKeys(bind->value)) {
      if (written_keys.count(key) != 0) blocked = true;
    }
    if (!blocked) {
      hoisted.push_back(stmt);
      continue;
    }
    loop_defined.push_back(bind->var);
    kept.push_back(stmt);
  }
  return {hoisted, kept};
}

bool StoreIsLoopInvariant(const StorePlan& store, const VectorPlan& plan,
                          const std::set<std::string>& written_keys) {
  std::vector<tirx::Var> axis_vars;
  for (const Axis& axis : plan->axes) axis_vars.push_back(axis->var);
  std::vector<PrimExpr> exprs(store->spec->indices.begin(), store->spec->indices.end());
  exprs.push_back(store->spec->value);
  if (store->spec->mask.has_value()) exprs.push_back(*store->spec->mask);
  bool blocked = false;
  for (const PrimExpr& expr : exprs) {
    tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef& node) {
      const auto* var = node.as<tirx::VarNode>();
      if (var != nullptr) {
        ffi::ObjectRef ref = ffi::GetRef<ffi::ObjectRef>(var);
        for (const tirx::Var& axis_var : axis_vars) {
          if (axis_var.same_as(ffi::GetRef<tirx::Var>(var))) blocked = true;
        }
        (void)ref;
      }
      const auto* load = node.as<tirx::BufferLoadNode>();
      if (load != nullptr && written_keys.count(BufferKey(load->buffer)) != 0) blocked = true;
    });
  }
  return !blocked;
}

std::vector<tirx::Var> StmtVars(const std::vector<Stmt>& stmts) {
  std::vector<tirx::Var> vars;
  for (const Stmt& stmt : stmts) {
    if (const auto* bind = stmt.as<tirx::BindNode>()) vars.push_back(bind->var);
  }
  return vars;
}

bool HasHoisted(const std::vector<Axis>& hoisted, const Axis& axis) {
  for (const Axis& item : hoisted) {
    if (item->var.same_as(axis->var)) return true;
  }
  return false;
}

/* ------------------------------------------------------------------------- */
/* reduction emission (forward declarations)                                 */
/* ------------------------------------------------------------------------- */

void EmitReduce(const VectorPlan& plan, Emitter* emitter);
PrimExpr EmitReduceChunk(const VectorPlan& plan, const StorePlan& store, Emitter* body,
                         const ReduceInfo& reduce, int bits, const PrimExpr& mask,
                         const PrimExpr& fill);

/* ------------------------------------------------------------------------- */
/* segment emission                                                          */
/* ------------------------------------------------------------------------- */

std::vector<Axis> InnerAxes(const VectorPlan& plan, const std::vector<Axis>& hoisted) {
  std::vector<Axis> inner;
  for (const Axis& axis : plan->axes) {
    if (axis->index <= plan->vector_axis->index) continue;
    if (HasHoisted(hoisted, axis)) continue;
    inner.push_back(axis);
  }
  return inner;
}

std::pair<std::vector<Stmt>, Stmt> EmitSegmentStmts(
    const VectorPlan& plan, const tirx::Var& chunk, const std::vector<Axis>& hoisted,
    const std::string& prefix, bool single, ProgramPrologue* shared,
    std::map<std::string, CarryEntry>* carry, bool allow_carry);

std::pair<std::vector<Stmt>, Stmt> EmitParallelAxisParts(
    const VectorPlan& plan, const tirx::Var& chunk, const std::string& prefix,
    const std::vector<Axis>& hoisted, bool single, ProgramPrologue* shared,
    std::map<std::string, CarryEntry>* carry, bool allow_carry);

std::pair<std::vector<Stmt>, Stmt> EmitSegmentStmts(
    const VectorPlan& plan, const tirx::Var& chunk, const std::vector<Axis>& hoisted,
    const std::string& prefix, bool single, ProgramPrologue* shared,
    std::map<std::string, CarryEntry>* carry, bool allow_carry) {
  // A parallel vector axis with inner axes uses the hand-written three-part
  // structure (pre -> for inner: main -> post) so pre/post run once per chunk.
  std::vector<Axis> inner = InnerAxes(plan, hoisted);
  if (plan->vector_axis->is_parallel && !inner.empty() && !ReduceOf(plan).defined()) {
    return EmitParallelAxisParts(plan, chunk, prefix, hoisted, single, shared, carry,
                                 allow_carry);
  }

  Emitter emitter(plan, chunk, prefix, false, 0, std::nullopt, shared, carry, allow_carry);
  std::set<std::string> written_keys;
  for (const StorePlan& store : plan->pre_stores) written_keys.insert(BufferKey(store->spec->buffer));
  for (const StorePlan& store : plan->stores) written_keys.insert(BufferKey(store->spec->buffer));
  for (const StorePlan& store : plan->post_stores) written_keys.insert(BufferKey(store->spec->buffer));

  std::vector<StorePlan> invariant_pre;
  std::vector<StorePlan> loop_pre;
  for (const StorePlan& store : plan->pre_stores) {
    if (single && StoreIsLoopInvariant(store, plan, written_keys)) {
      invariant_pre.push_back(store);
    } else {
      loop_pre.push_back(store);
    }
  }

  for (const StorePlan& store : loop_pre) EmitOuterStore(store, &emitter);
  std::vector<Stmt> pre_lets = emitter.TakeLets();
  emitter.clear_forward();

  std::vector<Stmt> main_prologue;
  std::vector<Stmt> main_lets;
  ReduceInfo reduce = ReduceOf(plan);
  if (reduce.defined()) {
    EmitReduce(plan, &emitter);
    main_lets = emitter.TakeLets();
  } else {
    auto chunks = EmitElementwiseChunks(plan, chunk, prefix, shared, carry, allow_carry);
    main_prologue = chunks.first;
    main_lets = chunks.second;
  }

  for (const StorePlan& store : plan->post_stores) EmitOuterStore(store, &emitter);
  std::vector<Stmt> post_lets = emitter.TakeLets();

  std::vector<Axis> wrap_axes;
  for (const Axis& axis : plan->axes) {
    if (axis->var.same_as(plan->vector_axis->var)) continue;
    if (HasHoisted(hoisted, axis)) continue;
    wrap_axes.push_back(axis);
  }

  std::vector<Stmt> hoisted_body;
  std::vector<Stmt> flat_main;
  for (const Stmt& stmt : main_lets) {
    if (const auto* seq = stmt.as<tirx::SeqStmtNode>()) {
      for (const Stmt& child : seq->seq) flat_main.push_back(child);
    } else {
      flat_main.push_back(stmt);
    }
  }
  bool all_bind_or_eval = !flat_main.empty();
  for (const Stmt& stmt : flat_main) {
    if (stmt.as<tirx::BindNode>() == nullptr && stmt.as<tirx::EvaluateNode>() == nullptr) {
      all_bind_or_eval = false;
    }
  }
  if (single && (!wrap_axes.empty() || !hoisted.empty()) && all_bind_or_eval) {
    std::vector<Axis> inner_and_hoisted = wrap_axes;
    inner_and_hoisted.insert(inner_and_hoisted.end(), hoisted.begin(), hoisted.end());
    auto split = SplitLoopInvariant(flat_main, inner_and_hoisted, written_keys,
                                   StmtVars(pre_lets));
    hoisted_body = split.first;
    flat_main = split.second;
  }

  std::vector<Stmt> body_stmts = pre_lets;
  body_stmts.insert(body_stmts.end(), flat_main.begin(), flat_main.end());
  body_stmts.insert(body_stmts.end(), post_lets.begin(), post_lets.end());
  Stmt body = SeqOfVec(body_stmts).empty() ? Stmt() : SeqOfVec(body_stmts)[0];
  for (auto it = plan->axes.rbegin(); it != plan->axes.rend(); ++it) {
    const Axis& axis = *it;
    if (axis->var.same_as(plan->vector_axis->var)) continue;
    if (HasHoisted(hoisted, axis)) continue;
    body = tirx::For(axis->var, IntImm32(0), axis->extent, ForKind::kSerial, body);
  }

  std::vector<Stmt> hoisted_lets;
  if (!invariant_pre.empty()) {
    Emitter pre_emitter(plan, IntImm32(0), prefix + "hopre_", false, 0, std::nullopt, shared,
                        carry, false);
    for (const StorePlan& store : invariant_pre) EmitOuterStore(store, &pre_emitter);
    for (Stmt& stmt : pre_emitter.prologue_) emitter.prologue_.push_back(std::move(stmt));
    hoisted_lets = pre_emitter.TakeLets();
  }

  std::vector<Stmt> prologue = emitter.prologue_;
  prologue.insert(prologue.end(), main_prologue.begin(), main_prologue.end());
  prologue.insert(prologue.end(), hoisted_body.begin(), hoisted_body.end());
  prologue.insert(prologue.end(), hoisted_lets.begin(), hoisted_lets.end());
  return {prologue, body};
}

/* ------------------------------------------------------------------------- */
/* reduction emission                                                        */
/* ------------------------------------------------------------------------- */

void EmitReduceTail(const StorePlan& store, Emitter* emitter, const ReduceInfo& reduce,
                    const std::string& elem, int bits, PrimExpr acc) {
  PrimExpr lane0 = emitter->Lane0Mask(bits);
  if (reduce->accumulate) {
    std::string element_op;
    ReduceElementFor(reduce->recipe.operator std::string(), &element_op);
    PrimExpr old = emitter->Addr(store->access, /*uniform=*/true);
    std::string acc_elem =
        ElemDtype(ffi::DLDataTypeToString(acc.dtype()).operator std::string());
    if (acc_elem == elem) {
      acc = emitter->Bind(
          SimdCall(ScalarBinaryOp(VecScalarBinary(element_op)),
                   DataTypeOf(ffi::DLDataTypeToString(acc.dtype()).operator std::string()),
                   {acc, old, lane0, StrImm(ModeZeroing())}),
          &emitter->lets_);
    } else {
      PrimExpr old_vec = emitter->Bind(
          Vld(old, store->spec->buffer, "BRC_B" + std::to_string(bits)), &emitter->lets_);
      acc = emitter->Bind(
          SimdCall(BinaryOp(element_op),
                   DataTypeOf(ffi::DLDataTypeToString(acc.dtype()).operator std::string()),
                   {acc, old_vec, lane0, StrImm(ModeZeroing())}),
          &emitter->lets_);
    }
  }
  PrimExpr dst = emitter->Addr(store->access, /*uniform=*/true);
  emitter->lets_.push_back(
      tirx::Evaluate(Vsts(dst, store->spec->buffer, acc, lane0, "ONEPT_B" + std::to_string(bits))));
  emitter->NoteWrite(store->spec->buffer);
}

PrimExpr EmitReduceChunk(const VectorPlan& plan, const StorePlan& store, Emitter* body,
                         const ReduceInfo& reduce, int bits, const PrimExpr& mask,
                         const PrimExpr& fill) {
  PrimExpr loaded = body->Value(*reduce->varying);
  PrimExpr tail = body->masked_tail() ? body->TailMask(bits) : PrimExpr();
  if (tail.defined()) {
    if (reduce->recipe == "vcadd") {
      return body->Bind(ReduceCall("vcadd", loaded, tail), &body->lets_);
    }
    loaded = body->Bind(
        SimdCall(simd_vsel(),
                 DataTypeOf(ffi::DLDataTypeToString(loaded.dtype()).operator std::string()),
                 {loaded, fill, tail}),
        &body->lets_);
  }
  return body->Bind(ReduceCall(reduce->recipe.operator std::string(), loaded, mask),
                    &body->lets_);
}

void EmitReduceUnrolled(const VectorPlan& plan, const StorePlan& store, Emitter* emitter,
                        const ReduceInfo& reduce, int64_t chunks) {
  std::string elem = ElemDtype(store->spec->buffer->dtype);
  int bits = MaskBits(elem);
  PrimExpr mask = emitter->AllMask(bits);
  std::string element_op;
  ReduceElementFor(reduce->recipe.operator std::string(), &element_op);

  bool has_tail = plan->tail.has_value() && *plan->tail != 0;
  PrimExpr fill;
  if (has_tail && reduce->recipe != "vcadd") {
    fill = emitter->Bind(
        Vdup(ReduceIdentity(reduce->recipe.operator std::string(), elem), elem, mask),
        &emitter->lets_);
  }

  int64_t last_chunk = chunks - 1;
  std::vector<PrimExpr> chains;
  for (int64_t ck = 0; ck < chunks; ++ck) {
    emitter->set_lane_base(IntImm32(ck * plan->lanes));
    // Addresses change per chunk: the memo and predicate caches must follow.
    emitter->ClearChunkCaches();
    emitter->set_masked_tail(has_tail && ck == last_chunk);
    chains.push_back(EmitReduceChunk(plan, store, emitter, reduce, bits, mask, fill));
  }
  emitter->set_masked_tail(false);

  while (chains.size() > 1) {
    PrimExpr lane0 = emitter->Lane0Mask(bits);
    std::vector<PrimExpr> merged;
    for (size_t k = 0; k + 1 < chains.size(); k += 2) {
      merged.push_back(emitter->Bind(
          SimdCall(BinaryOp(element_op),
                   DataTypeOf(ffi::DLDataTypeToString(chains[k].dtype()).operator std::string()),
                   {chains[k], chains[k + 1], lane0, StrImm(ModeZeroing())}),
          &emitter->lets_));
    }
    if (chains.size() % 2 != 0) merged.push_back(chains.back());
    chains = merged;
  }

  EmitReduceTail(store, emitter, reduce, elem, bits, chains[0]);
}

void EmitReduceLoop(const VectorPlan& plan, const StorePlan& store, Emitter* emitter,
                    const ReduceInfo& reduce) {
  std::string elem = ElemDtype(store->spec->buffer->dtype);
  int bits = MaskBits(elem);
  PrimExpr mask = emitter->AllMask(bits);
  std::string element_op;
  ReduceElementFor(reduce->recipe.operator std::string(), &element_op);
  PrimExpr identity = ReduceIdentity(reduce->recipe.operator std::string(), elem);

  tirx::Buffer acc = tirx::decl_buffer(
      {IntImm32(1)}, VecDataType(elem),
      "acc" + emitter->prefix() + std::to_string(emitter->counter()), "local.var");
  emitter->set_counter(emitter->counter() + 1);
  // The accumulator lives outside the chunk (and outer axis) loops: one register
  // per segment, re-initialized on every outer iteration.
  emitter->prologue_.push_back(tirx::AllocBuffer(acc));
  emitter->lets_.push_back(tirx::BufferStore(acc, Vdup(identity, elem, mask), {IntImm32(0)}));

  bool may_have_tail = !plan->chunks.has_value() ||
                       (plan->tail.has_value() && *plan->tail != 0);
  PrimExpr fill;
  if (may_have_tail && reduce->recipe != "vcadd") {
    fill = emitter->Bind(Vdup(identity, elem, mask), &emitter->lets_);
  }

  MakeBlock build_chunk = [&](const PrimExpr& chunk_index, bool masked_tail,
                              const std::string& label, int64_t chunk_min,
                              std::optional<int64_t> chunk_max) -> Block {
    Emitter body(plan, chunk_index, emitter->prefix() + label, masked_tail, chunk_min,
                 chunk_max, emitter->shared(), emitter->carry(), false);
    PrimExpr reduced = EmitReduceChunk(plan, store, &body, reduce, bits, mask, fill);
    PrimExpr lane0 = emitter->Lane0Mask(bits);
    PrimExpr total = body.Bind(
        SimdCall(BinaryOp(element_op),
                 DataTypeOf(ffi::DLDataTypeToString(reduced.dtype()).operator std::string()),
                 {tirx::BufferLoad(acc, {IntImm32(0)}), reduced, lane0, StrImm(ModeZeroing())}),
        &body.lets_);
    body.lets_.push_back(tirx::BufferStore(acc, total, {IntImm32(0)}));
    return Block{body.prologue_, body.lets_};
  };

  // Each reduction gets its own chunk loop (sibling reductions inside one
  // segment must not share the loop variable).
  tirx::Var loop_var(emitter->prefix() + "ck" + std::to_string(emitter->counter()),
                     DataType::Int(32));
  emitter->set_counter(emitter->counter() + 1);
  auto blocks = StripMine(plan, loop_var, build_chunk);
  for (Stmt& stmt : blocks.first) emitter->prologue_.push_back(std::move(stmt));
  for (Stmt& stmt : blocks.second) emitter->lets_.push_back(std::move(stmt));

  PrimExpr final_value =
      emitter->Bind(tirx::BufferLoad(acc, {IntImm32(0)}), &emitter->lets_);
  EmitReduceTail(store, emitter, reduce, elem, bits, final_value);
}

void EmitOneReduce(const VectorPlan& plan, const StorePlan& store, Emitter* emitter) {
  ReduceInfo reduce = ReduceOf(store);
  std::optional<int64_t> chunks =
      plan->chunks.has_value() ? std::optional<int64_t>(*plan->chunks) : std::nullopt;
  if (chunks.has_value() && *chunks == 0) {
    // Zero iterations: the destination keeps its value (and no empty chain).
    return;
  }
  if (chunks.has_value() && *chunks <= kReduceUnrollLimit) {
    EmitReduceUnrolled(plan, store, emitter, reduce, *chunks);
  } else {
    EmitReduceLoop(plan, store, emitter, reduce);
  }
}

void EmitReduce(const VectorPlan& plan, Emitter* emitter) {
  for (const StorePlan& store : plan->stores) {
    ReduceInfo reduce = ReduceOf(store);
    if (!reduce.defined()) {
      EmitOuterStore(store, emitter);  // lane-uniform value: one broadcast write
      continue;
    }
    EmitOneReduce(plan, store, emitter);
  }
}

/* ------------------------------------------------------------------------- */
/* parallel-axis pre/main/post structure                                     */
/* ------------------------------------------------------------------------- */

std::pair<std::vector<Stmt>, Stmt> EmitParallelAxisParts(
    const VectorPlan& plan, const tirx::Var& chunk, const std::string& prefix,
    const std::vector<Axis>& hoisted, bool single, ProgramPrologue* shared,
    std::map<std::string, CarryEntry>* carry, bool allow_carry) {
  TVM_FFI_ICHECK(!ReduceOf(plan).defined())
      << "a parallel vector axis with pre/post must go through the elementwise path";
  std::vector<Axis> inner = InnerAxes(plan, hoisted);
  std::vector<Stmt> prologue;
  std::set<std::string> written_keys;
  for (const StorePlan& store : plan->pre_stores) written_keys.insert(BufferKey(store->spec->buffer));
  for (const StorePlan& store : plan->stores) written_keys.insert(BufferKey(store->spec->buffer));
  for (const StorePlan& store : plan->post_stores) written_keys.insert(BufferKey(store->spec->buffer));

  std::vector<StorePlan> candidates;
  for (const StorePlan& store : plan->stores) {
    if (store->promote.has_value()) candidates.push_back(store);
  }
  // A pre write folded into an accumulator initial value must NOT be hoisted:
  // folding means it no longer writes memory, and the hoisted init depends on it.
  std::set<const StorePlanNode*> foldable;
  for (const StorePlan& store : candidates) {
    for (const StorePlan& pre : plan->pre_stores) {
      if (!pre->mask.has_value() && pre->value->is_uniform &&
          SameBuffer(pre->spec->buffer, store->spec->buffer) &&
          SameIndices(pre->spec->indices, store->spec->indices)) {
        foldable.insert(pre.get());
      }
    }
  }
  std::vector<StorePlan> invariant_pre;
  std::vector<StorePlan> loop_pre;
  for (const StorePlan& store : plan->pre_stores) {
    if (single && foldable.count(store.get()) == 0 &&
        StoreIsLoopInvariant(store, plan, written_keys)) {
      invariant_pre.push_back(store);
    } else {
      loop_pre.push_back(store);
    }
  }
  std::set<const StorePlanNode*> invariant_ids;
  for (const StorePlan& store : invariant_pre) invariant_ids.insert(store.get());
  std::vector<StorePlan> remaining_pre;
  for (const StorePlan& store : plan->pre_stores) {
    if (invariant_ids.count(store.get()) == 0) remaining_pre.push_back(store);
  }
  loop_pre = remaining_pre;

  auto finish = [&](const Stmt& body) -> std::pair<std::vector<Stmt>, Stmt> {
    if (!invariant_pre.empty()) {
      Emitter pre_emitter(plan, IntImm32(0), prefix + "hopre_", false, 0, std::nullopt, shared,
                          carry, false);
      for (const StorePlan& store : invariant_pre) EmitOuterStore(store, &pre_emitter);
      for (Stmt& stmt : pre_emitter.prologue_) prologue.push_back(std::move(stmt));
      std::vector<Stmt> lets = pre_emitter.TakeLets();
      std::vector<Stmt> out = prologue;
      out.insert(out.end(), lets.begin(), lets.end());
      return {out, body};
    }
    return {prologue, body};
  };

  MakeBlock make_block = [&](const PrimExpr& chunk_var, bool masked_tail,
                             const std::string& label, int64_t chunk_min,
                             std::optional<int64_t> chunk_max) -> Block {
    // SSA names of a program segment are prefixed with ``s{i}_`` so sibling
    // segments never collide.
    const std::string body_prefix = prefix + label;
    Emitter body_emitter(plan, chunk_var, body_prefix, masked_tail, chunk_min, chunk_max,
                         shared, carry, allow_carry);
    // A same-address pre write with a uniform value folds into the register
    // initial value instead of a memory write.
    std::map<const StorePlanNode*, StorePlan> folded;
    std::set<const StorePlanNode*> consumed_pre;
    for (const StorePlan& store : candidates) {
      for (const StorePlan& pre : loop_pre) {
        if (consumed_pre.count(pre.get()) != 0 || pre->mask.has_value() ||
            !pre->value->is_uniform) {
          continue;
        }
        if (SameBuffer(pre->spec->buffer, store->spec->buffer) &&
            SameIndices(pre->spec->indices, store->spec->indices)) {
          // Folding drops the memory write; a sibling pre reading the buffer
          // would see a stale value, so keep the write in that case.
          bool sibling_reads = false;
          for (const StorePlan& other : loop_pre) {
            if (other.get() != pre.get() && ReadsBuffer(other->value, pre->spec->buffer)) {
              sibling_reads = true;
            }
          }
          if (sibling_reads) continue;
          folded[store.get()] = pre;
          consumed_pre.insert(pre.get());
          break;
        }
      }
    }
    for (const StorePlan& store : loop_pre) {
      if (consumed_pre.count(store.get()) == 0) EmitStore(plan, store, &body_emitter);
    }
    std::vector<Stmt> pre_lets = body_emitter.TakeLets();
    // pre sits outside the inner loop: forwarding must not cross the boundary.
    body_emitter.clear_forward();

    // Accumulators: address invariant along the inner axes -> load into a
    // local.var vector register, update in the loop, write back once.
    std::vector<std::tuple<StorePlan, ValueNode, tirx::Buffer>> acc_order;
    std::map<const StorePlanNode*, size_t> acc_map;
    for (const StorePlan& store : candidates) {
      ValueNode old = *store->promote;
      tirx::Buffer acc = tirx::decl_buffer(
          {IntImm32(1)}, DataTypeOf(VecDtype(ElemDtype(store->spec->buffer->dtype))),
          "acc" + body_prefix + std::to_string(body_emitter.counter()), "local.var");
      body_emitter.set_counter(body_emitter.counter() + 1);
      body_emitter.prologue_.push_back(tirx::AllocBuffer(acc));
      PrimExpr init;
      auto fold_it = folded.find(store.get());
      if (fold_it != folded.end()) {
        init = body_emitter.Dup(fold_it->second->value);
      } else if (AccessClassFromName(store->kind_value.operator std::string()) ==
                 AccessClass::kContiguous) {
        init = body_emitter.Value(old);
      } else {
        init = body_emitter.Dup(old);
      }
      body_emitter.lets_.push_back(tirx::BufferStore(acc, init, {IntImm32(0)}));
      body_emitter.SetMemo(ValueKey(old, body_emitter.store_gen()),
                           tirx::BufferLoad(acc, {IntImm32(0)}));
      acc_map[store.get()] = acc_order.size();
      acc_order.push_back(std::make_tuple(store, old, acc));
    }
    std::vector<Stmt> init_lets = body_emitter.TakeLets();

    for (const StorePlan& store : plan->stores) {
      auto it = acc_map.find(store.get());
      if (it == acc_map.end()) {
        EmitStore(plan, store, &body_emitter);
        continue;
      }
      tirx::Buffer acc = std::get<2>(acc_order[it->second]);
      PrimExpr value = body_emitter.VectorOperand(store->value);
      body_emitter.lets_.push_back(tirx::BufferStore(acc, value, {IntImm32(0)}));
    }
    std::vector<Stmt> main_lets = body_emitter.TakeLets();
    body_emitter.clear_forward();

    // Write-back: when a post overwrites the same address and reads the old
    // value, hand the write-back to the post (register forwarding, saves a vsts).
    std::set<const StorePlanNode*> superseded;
    for (const StorePlan& store : candidates) {
      for (const StorePlan& post : plan->post_stores) {
        if (!post->mask.has_value() &&
            SameBuffer(post->spec->buffer, store->spec->buffer) &&
            SameIndices(post->spec->indices, store->spec->indices) &&
            SameAddressLoad(post->value, post->spec->buffer, post->spec->indices).defined()) {
          superseded.insert(store.get());
        }
      }
    }
    for (const auto& entry : acc_order) {
      const StorePlan& store = std::get<0>(entry);
      ValueNode old = std::get<1>(entry);
      tirx::Buffer acc = std::get<2>(entry);
      PrimExpr value = tirx::BufferLoad(acc, {IntImm32(0)});
      if (superseded.count(store.get()) == 0) {
        int bits = MaskBits(ElemDtype(store->spec->buffer->dtype));
        if (AccessClassFromName(store->kind_value.operator std::string()) ==
            AccessClass::kContiguous) {
          PrimExpr mask = StoreMask(store, &body_emitter, bits);
          body_emitter.lets_.push_back(tirx::Evaluate(
              Vsts(body_emitter.Addr(store->access), store->spec->buffer, value, mask,
                   "NORM_B" + std::to_string(bits))));
        } else {
          body_emitter.lets_.push_back(tirx::Evaluate(
              Vsts(body_emitter.Addr(store->access, /*uniform=*/true), store->spec->buffer,
                   value, body_emitter.AllMask(bits), "ONEPT_B" + std::to_string(bits))));
        }
        body_emitter.NoteWrite(store->spec->buffer);
      }
      body_emitter.Promote(old, value);
      // Segment-to-segment register forwarding: a single-chunk, tail-free,
      // predicate-free contiguous write-back whose address is invariant along
      // the inner axes *is* the memory content of the whole vector.
      bool before_axis_hoisted = true;
      for (const Axis& axis : plan->axes) {
        if (axis->index >= plan->vector_axis->index) continue;
        if (!HasHoisted(hoisted, axis)) before_axis_hoisted = false;
      }
      if (carry != nullptr && superseded.count(store.get()) == 0 && !store->mask.has_value() &&
          AccessClassFromName(store->kind_value.operator std::string()) ==
              AccessClass::kContiguous &&
          plan->chunks.has_value() && *plan->chunks == 1 &&
          !(plan->tail.has_value() && *plan->tail != 0) && before_axis_hoisted) {
        body_emitter.RegisterCarry(store, value, plan->vector_axis->var);
      }
    }
    std::vector<Stmt> final_lets = body_emitter.TakeLets();

    for (const StorePlan& store : plan->post_stores) {
      EmitStore(plan, store, &body_emitter);
    }
    std::vector<Stmt> post_lets = body_emitter.TakeLets();

    std::set<std::string> licm_written = written_keys;
    for (const auto& entry : acc_order) {
      licm_written.insert(BufferKey(std::get<2>(entry)));
    }
    auto split = SplitLoopInvariant(main_lets, inner, licm_written, {});
    std::vector<Stmt> hoisted_main = split.first;
    std::vector<Stmt> loop_main = split.second;

    std::vector<Stmt> prefix_stmts = pre_lets;
    prefix_stmts.insert(prefix_stmts.end(), init_lets.begin(), init_lets.end());
    prefix_stmts.insert(prefix_stmts.end(), hoisted_main.begin(), hoisted_main.end());
    if (loop_main.empty()) {
      std::vector<Stmt> out = prefix_stmts;
      out.insert(out.end(), final_lets.begin(), final_lets.end());
      out.insert(out.end(), post_lets.begin(), post_lets.end());
      return Block{body_emitter.prologue_, out};
    }
    Stmt main_stmt = SeqOfVec(loop_main)[0];
    for (auto it = inner.rbegin(); it != inner.rend(); ++it) {
      main_stmt = tirx::For((*it)->var, IntImm32(0), (*it)->extent, ForKind::kSerial, main_stmt);
    }
    std::vector<Stmt> out = prefix_stmts;
    out.push_back(main_stmt);
    out.insert(out.end(), final_lets.begin(), final_lets.end());
    out.insert(out.end(), post_lets.begin(), post_lets.end());
    return Block{body_emitter.prologue_, out};
  };

  auto blocks = StripMine(plan, chunk, make_block);
  for (Stmt& stmt : blocks.first) prologue.push_back(std::move(stmt));
  return finish(SeqOfVec(blocks.second)[0]);
}

/* ------------------------------------------------------------------------- */
/* program emission                                                          */
/* ------------------------------------------------------------------------- */

bool SegmentConsumesCarry(const VectorPlan& segment, const std::vector<Axis>& hoisted) {
  if (!segment->chunks.has_value() || *segment->chunks != 1) return false;
  for (const Axis& axis : segment->axes) {
    if (axis->index >= segment->vector_axis->index) continue;
    if (!HasHoisted(hoisted, axis)) return false;
  }
  return true;
}

Stmt EmitProgram(const VectorProgram& program) {
  for (const Axis& axis : program->axes) {
    TVM_FFI_ICHECK(axis->is_parallel)
        << "program.axes may only hold hoisted T.Parallel axes";
  }
  ProgramPrologue shared;
  std::map<std::string, CarryEntry> carry;
  std::vector<Stmt> prologue;
  std::vector<Stmt> bodies;
  for (size_t index = 0; index < program->segments.size(); ++index) {
    tirx::Var seg_chunk("chunk_s" + std::to_string(index), DataType::Int(32));
    std::vector<Axis> hoisted(program->axes.begin(), program->axes.end());
    auto emitted = EmitSegmentStmts(program->segments[index], seg_chunk, hoisted,
                                    "s" + std::to_string(index) + "_",
                                    program->segments.size() == 1, &shared, &carry,
                                    SegmentConsumesCarry(program->segments[index], hoisted));
    for (Stmt& stmt : emitted.first) prologue.push_back(std::move(stmt));
    bodies.push_back(emitted.second);
  }
  Stmt body = SeqOfVec(bodies)[0];
  for (auto it = program->axes.rbegin(); it != program->axes.rend(); ++it) {
    body = tirx::For((*it)->var, IntImm32(0), (*it)->extent, ForKind::kSerial, body);
  }
  std::vector<Stmt> all;
  for (Stmt& stmt : shared.statements) all.push_back(std::move(stmt));
  for (Stmt& stmt : prologue) all.push_back(std::move(stmt));
  all.push_back(body);
  return SeqOfVec(all)[0];
}

}  // namespace

tirx::Stmt EmitPlan(const ffi::ObjectRef& plan) {
  if (const auto* program = plan.as<VectorProgramNode>()) {
    return EmitProgram(ffi::GetRef<VectorProgram>(program));
  }
  const auto* single = plan.as<VectorPlanNode>();
  TVM_FFI_ICHECK(single != nullptr) << "emit_plan expects a VectorPlan or VectorProgram";
  VectorPlan node = ffi::GetRef<VectorPlan>(single);
  tirx::Var chunk("chunk", DataType::Int(32));
  auto emitted = EmitSegmentStmts(node, chunk, {}, "", /*single=*/true, nullptr, nullptr, false);
  if (emitted.first.empty()) return emitted.second;
  std::vector<Stmt> all = emitted.first;
  all.push_back(emitted.second);
  return SeqOfVec(all)[0];
}

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm
