/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file simdvf_alloc_fragment.cc
 * \brief Lower SimdVF-local ``T.alloc_fragment`` tiles into register arrays.
 *
 * A ``T.alloc_fragment`` allocated inside ``T.SimdVF`` is a logical V-core
 * register tile, while one hardware vector register holds exactly 2048 bits
 * (256 bytes).  A tile that is larger than one register therefore has no valid
 * single-register representation, and the elementwise loops around it cannot be
 * planned by P2/P3 until the tile is expressed at register granularity.
 *
 * This pass runs before ``AscendLiftVSimd`` (P1) and rewrites every fragment
 * that is larger than one register into an array of register-sized fragments,
 * then rewrites its elementwise uses into chunk-wise ``T.Parallel`` nests:
 *
 *   frag = T.alloc_fragment((128,), "float32")      // 2 x vector_f32
 *   for i in T.serial(128): frag[i] = frag[i] * 2
 *   T.copy(frag, out_ub)
 *
 * becomes
 *
 *   frag_r0 = T.alloc_fragment((64,), "float32")
 *   frag_r1 = T.alloc_fragment((64,), "float32")
 *   for i in T.Parallel(64): frag_r0[i] = frag_r0[i] * 2
 *   for i in T.Parallel(64): frag_r1[i] = frag_r1[i] * 2
 *   for i in T.Parallel(64): out_ub[i] = frag_r0[i]
 *   for i in T.Parallel(64): out_ub[64 + i] = frag_r1[i]
 *
 * Each rewritten nest touches exactly one register per chunk, so P2's
 * fragment-register mode turns every one of them into pure ``local.var``
 * vector register operations.
 *
 * The rewrite is conservative and VF-local: a fragment is lowered only when
 * *every* use of it matches one of the supported elementwise patterns.  Any
 * other use (strided / indirect indexing, nested control flow, tile ops with a
 * partial region, ...) leaves the whole fragment untouched and the VF keeps the
 * previous behavior.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "layout/layout.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

/*! \brief One V-core vector register holds 2048 bits (256 bytes). */
constexpr int64_t kVectorRegisterBits = 2048;

/*! \brief Upper bound on the register array length of one fragment. */
constexpr int64_t kMaxRegistersPerFragment = 16;

bool IsFragment(const Buffer& buffer) {
  return buffer.defined() && buffer.scope() == "local.fragment";
}

bool GetStaticInt(const PrimExpr& expr, int64_t* out) {
  const auto* imm = expr.as<IntImmNode>();
  if (imm == nullptr) {
    return false;
  }
  *out = imm->value;
  return true;
}

/*! \brief Element types accepted by the fragment register file. */
bool SupportedElementDtype(const DataType& dtype) {
  if (dtype.lanes() != 1) {
    return false;
  }
  if (dtype.is_float16() || dtype.is_bfloat16()) {
    return true;
  }
  if (dtype.is_float()) {
    return dtype.bits() == 32;
  }
  if (dtype.is_int() || dtype.is_uint()) {
    return dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32;
  }
  return false;
}

bool StaticElementCount(const Buffer& buffer, int64_t* count) {
  int64_t total = 1;
  for (const PrimExpr& dim : buffer->shape) {
    int64_t value = 0;
    if (!GetStaticInt(dim, &value) || value <= 0) {
      return false;
    }
    total *= value;
  }
  *count = total;
  return true;
}

/*!
 * \brief Whether *loop* is the identity ``T.parallel(1)`` store wrapper.
 *
 * ``AddWrapperForSingleBufStore`` wraps every lone fragment store that is not
 * already inside a tile-operation loop, so a serial elementwise loop shows up
 * as ``for i in serial(N): for _ in parallel(1): frag[i] = ...``.  The wrapper
 * carries no iteration semantics and must be transparent here.
 */
bool IsIdentityParallel(const ForNode* loop) {
  if (loop->kind != ForKind::kParallel) {
    return false;
  }
  int64_t extent = 0;
  return GetStaticInt(loop->extent, &extent) && extent == 1 && is_zero(loop->min);
}

/*! \brief Flatten ``SeqStmt`` and identity ``T.parallel(1)`` wrappers. */
void FlattenElementwiseStmts(const Stmt& stmt, std::vector<Stmt>* out) {
  if (const auto* seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt& item : seq->seq) {
      FlattenElementwiseStmts(item, out);
    }
    return;
  }
  if (const auto* loop = stmt.as<ForNode>()) {
    if (IsIdentityParallel(loop)) {
      FlattenElementwiseStmts(loop->body, out);
      return;
    }
  }
  out->push_back(stmt);
}

/*! \brief Register layout of one lowered fragment. */
struct RegisterArray {
  Buffer source;
  int64_t total{0};
  int64_t lanes{0};
  std::vector<Buffer> regs;
};

/*! \brief One top-level statement of a SIMD_VF body and its fragment uses. */
struct UnitScan {
  Stmt stmt;
  std::set<const VarNode*> touched;
  /*! \brief ``true`` when the statement matches a supported rewrite pattern. */
  bool supported{false};
  /*! \brief Iteration domain of a supported loop / copy. */
  int64_t extent{0};
};

void FlattenSeq(const Stmt& stmt, std::vector<Stmt>* out) {
  if (const auto* seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt& item : seq->seq) {
      FlattenSeq(item, out);
    }
    return;
  }
  out->push_back(stmt);
}

/*! \brief Collect the statements of one SIMD_VF body (scope marker stripped). */
void CollectUnits(const Stmt& stmt, std::vector<Stmt>* out) {
  if (const auto* seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt& item : seq->seq) {
      CollectUnits(item, out);
    }
    return;
  }
  if (const auto* attr = stmt.as<AttrStmtNode>()) {
    if (attr->attr_key == "tl.simdvf_scope") {
      CollectUnits(attr->body, out);
      return;
    }
  }
  if (stmt.as<DeclBufferNode>() != nullptr) {
    // Buffer declarations carry no computation; the cleanup stage rewrites them.
    return;
  }
  out->push_back(stmt);
}

/*! \brief Decode a ``T.region(buffer[index], access, extent)`` argument. */
bool DecodeRegion(const PrimExpr& expr, Buffer* buffer, PrimExpr* base,
                  int64_t* extent) {
  const auto* call = expr.as<CallNode>();
  if (call == nullptr) {
    return false;
  }
  const auto* op = call->op.as<OpNode>();
  if (op == nullptr || op->name != "tl.region" || call->args.size() != 3) {
    return false;
  }
  const auto* load = call->args[0].as<BufferLoadNode>();
  if (load == nullptr || load->indices.size() != 1 ||
      load->predicate.defined()) {
    return false;
  }
  int64_t size = 0;
  if (!GetStaticInt(call->args[2], &size) || size <= 0) {
    return false;
  }
  *buffer = load->buffer;
  *base = load->indices[0];
  *extent = size;
  return true;
}

/*! \brief Scan one top-level statement for supported elementwise patterns. */
class UnitScanner {
 public:
  explicit UnitScanner(const std::set<const VarNode*>& candidates)
      : candidates_(candidates) {}

  UnitScan Scan(const Stmt& stmt) {
    UnitScan unit;
    unit.stmt = stmt;
    if (const auto* loop = stmt.as<ForNode>()) {
      ScanLoop(loop, &unit);
    } else if (const auto* eval = stmt.as<EvaluateNode>()) {
      if (const auto* call = eval->value.as<CallNode>()) {
        ScanCopy(call, &unit);
      }
    }
    if (!unit.supported) {
      // Still record the candidate fragments the statement touches, so an
      // unsupported use can veto those fragments.
      tirx::PostOrderVisit(stmt, [&](const ObjectRef& node) {
        if (const auto* load = node.as<BufferLoadNode>()) {
          if (candidates_.count(load->buffer->data.get()) != 0) {
            unit.touched.insert(load->buffer->data.get());
          }
        } else if (const auto* store = node.as<BufferStoreNode>()) {
          if (candidates_.count(store->buffer->data.get()) != 0) {
            unit.touched.insert(store->buffer->data.get());
          }
        }
      });
    }
    return unit;
  }

 private:
  void ScanLoop(const ForNode* loop, UnitScan* unit) {
    if (!is_zero(loop->min)) {
      return;
    }
    if (loop->step.defined() && !is_one(loop->step.value())) {
      return;
    }
    if (loop->kind != ForKind::kParallel && loop->kind != ForKind::kSerial) {
      return;
    }
    if (loop->thread_binding.defined() || !loop->annotations.empty()) {
      // Keep annotated loops untouched instead of silently dropping hints that
      // may refer to the original loop variable.
      return;
    }
    int64_t extent = 0;
    if (!GetStaticInt(loop->extent, &extent) || extent <= 0) {
      return;
    }

    std::vector<Stmt> body;
    FlattenElementwiseStmts(loop->body, &body);
    bool only_stores = true;
    for (const Stmt& item : body) {
      if (item.as<BufferStoreNode>() == nullptr) {
        only_stores = false;
        break;
      }
    }

    bool nested_control_flow = false;
    bool other_write = false;
    bool bad_access = false;
    std::set<const VarNode*> touched;
    tirx::PostOrderVisit(loop->body, [&](const ObjectRef& node) {
      if (const auto* nested = node.as<ForNode>()) {
        if (!IsIdentityParallel(nested)) {
          nested_control_flow = true;
        }
        return;
      }
      if (node.as<IfThenElseNode>() != nullptr) {
        nested_control_flow = true;
        return;
      }
      if (const auto* store = node.as<BufferStoreNode>()) {
        if (candidates_.count(store->buffer->data.get()) != 0) {
          touched.insert(store->buffer->data.get());
          if (store->predicate.defined() || store->indices.size() != 1 ||
              !store->indices[0].same_as(loop->loop_var)) {
            bad_access = true;
          }
        } else {
          other_write = true;
        }
        return;
      }
      if (const auto* load = node.as<BufferLoadNode>()) {
        if (candidates_.count(load->buffer->data.get()) != 0) {
          touched.insert(load->buffer->data.get());
          if (load->predicate.defined() || load->indices.size() != 1 ||
              !load->indices[0].same_as(loop->loop_var)) {
            bad_access = true;
          }
        }
      }
    });

    if (touched.empty() || bad_access || nested_control_flow) {
      return;
    }
    if (loop->kind == ForKind::kSerial && (!only_stores || other_write)) {
      // A serial loop is only turned into a parallel one when every iteration
      // writes exactly one element of a lowered fragment and nothing else.
      return;
    }
    unit->supported = true;
    unit->touched = std::move(touched);
    unit->extent = extent;
  }

  void ScanCopy(const CallNode* call, UnitScan* unit) {
    const auto* op = call->op.as<OpNode>();
    if (op == nullptr ||
        (op->name != "tl.tileop.copy" && op->name != "tl.tileop.ascend_copy")) {
      return;
    }
    if (call->args.size() != 2) {
      return;
    }
    Buffer src;
    Buffer dst;
    PrimExpr src_base;
    PrimExpr dst_base;
    int64_t src_extent = 0;
    int64_t dst_extent = 0;
    if (!DecodeRegion(call->args[0], &src, &src_base, &src_extent) ||
        !DecodeRegion(call->args[1], &dst, &dst_base, &dst_extent)) {
      return;
    }
    if (src_extent != dst_extent || src->dtype != dst->dtype ||
        src.same_as(dst)) {
      return;
    }
    const bool src_fragment = candidates_.count(src->data.get()) != 0;
    const bool dst_fragment = candidates_.count(dst->data.get()) != 0;
    if (!src_fragment && !dst_fragment) {
      return;
    }
    if (src_fragment && !is_zero(src_base)) {
      return;
    }
    if (dst_fragment && !is_zero(dst_base)) {
      return;
    }
    if (src_fragment) {
      unit->touched.insert(src->data.get());
    }
    if (dst_fragment) {
      unit->touched.insert(dst->data.get());
    }
    unit->supported = true;
    unit->extent = src_extent;
  }

  const std::set<const VarNode*>& candidates_;
};

/*! \brief Replace loop-variable uses and fragment accesses of one chunk. */
class ChunkRewriter : public StmtExprMutator {
 public:
  ChunkRewriter(const Var& var, const Var& chunk_var, PrimExpr base,
                const std::map<const VarNode*, RegisterArray>* lowered,
                int64_t slot)
      : var_(var),
        chunk_var_(chunk_var),
        base_(std::move(base)),
        lowered_(lowered),
        slot_(slot) {}

 private:
  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == var_.get()) {
      return base_ + chunk_var_;
    }
    return GetRef<Var>(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto it = lowered_->find(op->buffer->data.get());
    if (it == lowered_->end()) {
      return StmtExprMutator::VisitExpr_(op);
    }
    TVM_FFI_ICHECK(op->indices.size() == 1 && op->indices[0].same_as(var_))
        << "SimdVF fragment lowering lost the elementwise access pattern of '"
        << op->buffer->name;
    return BufferLoad(it->second.regs[slot_], {chunk_var_});
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto it = lowered_->find(op->buffer->data.get());
    if (it == lowered_->end()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    TVM_FFI_ICHECK(op->indices.size() == 1 && op->indices[0].same_as(var_))
        << "SimdVF fragment lowering lost the elementwise access pattern of '"
        << op->buffer->name;
    PrimExpr value = VisitExpr(op->value);
    return BufferStore(it->second.regs[slot_], value, {chunk_var_});
  }

  Stmt VisitStmt_(const ForNode* op) final {
    if (IsIdentityParallel(op)) {
      return VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Var var_;
  Var chunk_var_;
  PrimExpr base_;
  const std::map<const VarNode*, RegisterArray>* lowered_;
  int64_t slot_;
};

/*! \brief Lowering plan of one SIMD_VF block. */
struct BlockPlan {
  std::set<const VarNode*> candidates;
  std::map<const VarNode*, RegisterArray> lowered;
};

/*! \brief Per-module lowering state shared by the rewrite stages. */
struct LoweringState {
  std::map<const SBlockNode*, BlockPlan> plans;
  std::map<const VarNode*, RegisterArray> lowered;
};

/*! \brief Build the register array declaration of one candidate fragment. */
bool MakeRegisterArray(const Buffer& buffer, RegisterArray* array) {
  if (!IsFragment(buffer) || !SupportedElementDtype(buffer->dtype)) {
    return false;
  }
  int64_t total = 0;
  if (!StaticElementCount(buffer, &total)) {
    return false;
  }
  const int64_t lanes = kVectorRegisterBits / buffer->dtype.bits();
  if (lanes <= 0 || total % lanes != 0) {
    return false;
  }
  const int64_t chunks = total / lanes;
  if (chunks < 2 || chunks > kMaxRegistersPerFragment) {
    // One register already is the natural form; larger arrays are left to the
    // existing fragment path instead of exploding the IR.
    return false;
  }
  array->source = buffer;
  array->total = total;
  array->lanes = lanes;
  for (int64_t slot = 0; slot < chunks; ++slot) {
    std::string name =
        buffer->name.operator std::string() + "_r" + std::to_string(slot);
    array->regs.push_back(tirx::decl_buffer(
        {IntImm(DataType::Int(32), lanes)}, buffer->dtype, name,
        "local.fragment"));
  }
  return true;
}

/*!
 * \brief Decide which fragments of one SIMD_VF block are lowered.
 *
 * The decision is a fixpoint: a fragment is lowered only when every statement
 * that touches it supports the elementwise rewrite and all fragments of a
 * statement share the same register width and chunk count.
 */
BlockPlan PlanBlock(const SBlockNode* block) {
  BlockPlan plan;

  std::map<const VarNode*, RegisterArray> candidates;
  for (const Buffer& buffer : block->alloc_buffers) {
    RegisterArray array;
    if (MakeRegisterArray(buffer, &array)) {
      candidates[buffer->data.get()] = std::move(array);
    }
  }
  if (candidates.empty()) {
    return plan;
  }

  for (const auto& entry : candidates) {
    plan.candidates.insert(entry.first);
  }

  std::vector<Stmt> units;
  CollectUnits(block->body, &units);

  UnitScanner scanner(plan.candidates);
  std::vector<UnitScan> scans;
  scans.reserve(units.size());
  for (const Stmt& unit : units) {
    scans.push_back(scanner.Scan(unit));
  }

  std::set<const VarNode*> lowered = plan.candidates;
  bool changed = true;
  while (changed) {
    changed = false;
    for (const UnitScan& scan : scans) {
      if (scan.touched.empty()) {
        continue;
      }
      bool ok = scan.supported;
      int64_t lanes = -1;
      if (ok) {
        for (const VarNode* key : scan.touched) {
          if (lowered.count(key) == 0) {
            ok = false;
            break;
          }
          const RegisterArray& array = candidates.at(key);
          if (array.total != scan.extent) {
            ok = false;
            break;
          }
          if (lanes < 0) {
            lanes = array.lanes;
          } else if (lanes != array.lanes) {
            ok = false;
            break;
          }
        }
      }
      if (ok) {
        continue;
      }
      for (const VarNode* key : scan.touched) {
        if (lowered.erase(key) != 0) {
          changed = true;
        }
      }
    }
  }

  for (const VarNode* key : lowered) {
    plan.lowered.emplace(key, candidates.at(key));
  }
  return plan;
}

/*! \brief Build the chunk loops that replace one supported unit. */
std::vector<Stmt> RewriteUnit(
    const UnitScan& scan,
    const std::map<const VarNode*, RegisterArray>& lowered) {
  int64_t lanes = -1;
  int64_t chunks = 0;
  for (const VarNode* key : scan.touched) {
    const RegisterArray& array = lowered.at(key);
    lanes = array.lanes;
    chunks = static_cast<int64_t>(array.regs.size());
    break;
  }
  if (lanes <= 0 || chunks <= 0) {
    return {};
  }

  std::vector<Stmt> result;
  result.reserve(static_cast<size_t>(chunks));

  if (const auto* loop = scan.stmt.as<ForNode>()) {
    const DataType index_dtype = loop->loop_var.dtype();
    for (int64_t slot = 0; slot < chunks; ++slot) {
      Var chunk_var(loop->loop_var->name_hint + "_c" + std::to_string(slot),
                    index_dtype);
      PrimExpr base = IntImm(index_dtype, slot * lanes);
      Stmt body = ChunkRewriter(loop->loop_var, chunk_var, base, &lowered, slot)(
          loop->body);
      result.push_back(For(chunk_var, IntImm(index_dtype, 0),
                           IntImm(index_dtype, lanes), ForKind::kParallel,
                           std::move(body)));
    }
    return result;
  }

  const auto* eval = scan.stmt.as<EvaluateNode>();
  const auto* call = eval == nullptr ? nullptr : eval->value.as<CallNode>();
  if (call == nullptr || call->args.size() != 2) {
    return {};
  }
  Buffer src;
  Buffer dst;
  PrimExpr src_base;
  PrimExpr dst_base;
  int64_t src_extent = 0;
  int64_t dst_extent = 0;
  if (!DecodeRegion(call->args[0], &src, &src_base, &src_extent) ||
      !DecodeRegion(call->args[1], &dst, &dst_base, &dst_extent)) {
    return {};
  }
  const auto src_it = lowered.find(src->data.get());
  const auto dst_it = lowered.find(dst->data.get());
  if (src_it == lowered.end() && dst_it == lowered.end()) {
    return {};
  }
  const DataType index_dtype = DataType::Int(32);
  for (int64_t slot = 0; slot < chunks; ++slot) {
    Var index("simdvf_frag_i" + std::to_string(slot), index_dtype);
    PrimExpr offset = IntImm(index_dtype, slot * lanes);
    PrimExpr value;
    if (src_it != lowered.end()) {
      value = BufferLoad(src_it->second.regs[slot], {index});
    } else {
      value = BufferLoad(src, {src_base + offset + index});
    }
    if (dst_it != lowered.end()) {
      result.push_back(For(
          index, IntImm(index_dtype, 0), IntImm(index_dtype, lanes),
          ForKind::kParallel,
          BufferStore(dst_it->second.regs[slot], value, {index})));
    } else {
      result.push_back(For(
          index, IntImm(index_dtype, 0), IntImm(index_dtype, lanes),
          ForKind::kParallel,
          BufferStore(dst, value, {dst_base + offset + index})));
    }
  }
  return result;
}

Array<Stmt> ToArray(const std::vector<Stmt>& stmts) {
  Array<Stmt> array;
  for (const Stmt& stmt : stmts) {
    array.push_back(stmt);
  }
  return array;
}

/*! \brief Rebuild a SIMD_VF body with the planned units replaced. */
Stmt RewriteBody(
    const Stmt& stmt,
    const std::map<const StmtNode*, std::vector<Stmt>>& replacements) {
  if (const auto* seq = stmt.as<SeqStmtNode>()) {
    std::vector<Stmt> items;
    for (const Stmt& item : seq->seq) {
      Stmt rewritten = RewriteBody(item, replacements);
      FlattenSeq(rewritten, &items);
    }
    return SeqStmt(ToArray(items));
  }
  if (const auto* attr = stmt.as<AttrStmtNode>()) {
    if (attr->attr_key == "tl.simdvf_scope") {
      Stmt body = RewriteBody(attr->body, replacements);
      return AttrStmt(attr->node, attr->attr_key, attr->value,
                      std::move(body));
    }
  }
  auto it = replacements.find(stmt.get());
  if (it != replacements.end()) {
    return SeqStmt(ToArray(it->second));
  }
  return stmt;
}

/*! \brief Stage 1: collect one plan per SIMD_VF block. */
class PlanCollector : public StmtExprMutator {
 public:
  explicit PlanCollector(LoweringState* state) : state_(state) {}

  Stmt VisitStmt_(const SBlockNode* op) final {
    if (op->name_hint != "SIMD_VF") {
      return StmtExprMutator::VisitStmt_(op);
    }
    BlockPlan plan = PlanBlock(op);
    if (plan.lowered.empty()) {
      return GetRef<Stmt>(op);
    }
    for (const auto& entry : plan.lowered) {
      // A fragment shared by two VF blocks cannot be lowered safely: the first
      // registration wins and the second VF keeps the original tile.
      if (state_->lowered.count(entry.first) != 0) {
        return GetRef<Stmt>(op);
      }
    }
    for (const auto& entry : plan.lowered) {
      state_->lowered.emplace(entry.first, entry.second);
    }
    state_->plans.emplace(op, std::move(plan));
    return GetRef<Stmt>(op);
  }

 private:
  LoweringState* state_;
};

/*! \brief Stage 2: apply the collected plans. */
class PlanApplier : public StmtExprMutator {
 public:
  explicit PlanApplier(LoweringState* state) : state_(state) {}

  Stmt VisitStmt_(const SBlockNode* op) final {
    auto plan_it = state_->plans.find(op);
    if (plan_it == state_->plans.end()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    const BlockPlan& plan = plan_it->second;

    std::vector<Stmt> units;
    CollectUnits(op->body, &units);
    UnitScanner scanner(plan.candidates);
    std::map<const StmtNode*, std::vector<Stmt>> replacements;
    for (const Stmt& unit : units) {
      UnitScan scan = scanner.Scan(unit);
      if (scan.touched.empty()) {
        continue;
      }
      bool all_lowered = true;
      for (const VarNode* key : scan.touched) {
        if (plan.lowered.count(key) == 0) {
          all_lowered = false;
          break;
        }
      }
      if (!all_lowered) {
        continue;
      }
      std::vector<Stmt> rewritten = RewriteUnit(scan, plan.lowered);
      if (!rewritten.empty()) {
        replacements.emplace(unit.get(), std::move(rewritten));
      }
    }

    SBlock block = GetRef<SBlock>(op);
    SBlockNode* writer = block.CopyOnWrite();
    Array<Buffer> allocs;
    for (const Buffer& buffer : op->alloc_buffers) {
      auto frag = plan.lowered.find(buffer->data.get());
      if (frag == plan.lowered.end()) {
        allocs.push_back(buffer);
        continue;
      }
      for (const Buffer& reg : frag->second.regs) {
        allocs.push_back(reg);
      }
    }
    writer->alloc_buffers = std::move(allocs);
    writer->body = RewriteBody(op->body, replacements);
    return block;
  }

 private:
  LoweringState* state_;
};

/*!
 * \brief Stage 3: rewrite buffer declarations and drop stale layout entries.
 *
 * The lowered fragment's data Var is no longer allocated, so its ``DeclBuffer``
 * is replaced by the register declarations and every ``layout_map`` entry that
 * still refers to it is dropped.  The register buffers have no layout entry,
 * which P2 treats as the identity layout.
 */
class DeclCleanup : public StmtExprMutator {
 public:
  explicit DeclCleanup(const std::map<const VarNode*, RegisterArray>& lowered)
      : lowered_(lowered) {}

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    auto it = lowered_.find(op->buffer->data.get());
    if (it == lowered_.end()) {
      return GetRef<Stmt>(op);
    }
    std::vector<Stmt> decls;
    for (const Buffer& reg : it->second.regs) {
      decls.push_back(DeclBuffer(reg));
    }
    return SeqStmt(ToArray(decls));
  }

  Stmt VisitStmt_(const SBlockNode* op) final {
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    auto it = block->annotations.find(attr::kLayoutMap);
    if (it == block->annotations.end()) {
      return block;
    }
    auto old = Downcast<Map<Buffer, Layout>>((*it).second);
    Map<Buffer, Layout> kept;
    bool removed = false;
    for (const auto& entry : old) {
      if (lowered_.count(entry.first->data.get()) != 0) {
        removed = true;
        continue;
      }
      kept.Set(entry.first, entry.second);
    }
    if (!removed) {
      return block;
    }
    SBlockNode* writer = block.CopyOnWrite();
    if (kept.empty()) {
      writer->annotations.erase(attr::kLayoutMap);
    } else {
      writer->annotations.Set(attr::kLayoutMap, kept);
    }
    return block;
  }

 private:
  const std::map<const VarNode*, RegisterArray>& lowered_;
};

Stmt LowerSimdVFAllocFragment(Stmt stmt) {
  LoweringState state;
  PlanCollector collector(&state);
  Stmt collected = collector(std::move(stmt));
  if (state.plans.empty()) {
    return collected;
  }
  PlanApplier applier(&state);
  Stmt applied = applier(std::move(collected));
  DeclCleanup cleanup(state.lowered);
  return cleanup(std::move(applied));
}

}  // namespace

tvm::transform::Pass AscendSimdVFallocFragment() {
  auto pass_func = [](PrimFunc func, const IRModule&,
                      const tvm::transform::PassContext&) {
    PrimFuncNode* fptr = func.CopyOnWrite();
    fptr->body = LowerSimdVFAllocFragment(std::move(fptr->body));
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.AscendSimdVFallocFragment", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendSimdVFallocFragment",
                        AscendSimdVFallocFragment);
}

}  // namespace tl
}  // namespace tvm
