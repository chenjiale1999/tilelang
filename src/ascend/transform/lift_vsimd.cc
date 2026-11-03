/*!
 * \file lift_vsimd.cc
 * \brief P1 metadata pass: per-VF VSimdProgram annotation (Phase 1).
 *
 * The pass is deliberately non-destructive: it never changes the SIMD_VF body.
 * It records per-Region/Op status, spans, captures, groups and fragment usage
 * in a typed-map annotation so P2/P3 can plan without re-running scope analysis.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vsimd_capture_scope.h"
#include "vsimd_handoff.h"
#include "vsimd_memory_legality.h"
#include "vsimd_program.h"
#include "vsimd_region.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;
using namespace vsimd;
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

namespace {

constexpr const char *kProgramKey = "tl.vsimd.program";

struct VarHash {
  size_t operator()(const Var &v) const { return ffi::ObjectPtrHash()(v); }
};
struct VarEq {
  bool operator()(const Var &a, const Var &b) const { return a.same_as(b); }
};
using VarSet = std::unordered_set<Var, VarHash, VarEq>;
using VarIntMap = std::unordered_map<Var, int, VarHash, VarEq>;

static String ExprStr(const PrimExpr &e) {
  std::ostringstream os;
  os << e;
  return String(os.str());
}

static Array<String> ToStrings(const std::vector<std::string> &items) {
  std::vector<String> out;
  out.reserve(items.size());
  for (const std::string &s : items)
    out.push_back(String(s));
  return Array<String>(out.begin(), out.end());
}

static Array<Integer> ToInts(const std::vector<int> &items) {
  std::vector<Integer> out;
  out.reserve(items.size());
  for (int i : items)
    out.push_back(Integer(i));
  return Array<Integer>(out.begin(), out.end());
}

static bool IsFragment(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "local.fragment";
}

static void AddBufferUnique(std::vector<Buffer> *buffers, const Buffer &b) {
  for (const Buffer &existing : *buffers) {
    if (existing.same_as(b))
      return;
  }
  buffers->push_back(b);
}

class BufferCollector : public StmtExprVisitor {
public:
  std::vector<Buffer> buffers;
  void VisitExpr_(const BufferLoadNode *op) final {
    AddBufferUnique(&buffers, op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    AddBufferUnique(&buffers, op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
};

static std::vector<Buffer> BuffersIn(const PrimExpr &expr) {
  BufferCollector collector;
  if (expr.defined())
    collector(expr);
  return collector.buffers;
}

class BufferAccessCollector : public StmtExprVisitor {
public:
  std::vector<Buffer> reads;
  std::vector<Buffer> writes;
  bool unknown_effects{false};

  void VisitExpr_(const BufferLoadNode *op) final {
    AddBufferUnique(&reads, op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    AddBufferUnique(&writes, op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    if (const auto *callee = op->op.as<OpNode>()) {
      const std::string name = callee->name;
      if (name == "tirx.call_extern" || name == "tirx.call_pure_extern" ||
          name.rfind("tl.", 0) == 0) {
        unknown_effects = true;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

static void SplitAccesses(const Stmt &stmt, std::vector<Buffer> *reads,
                          std::vector<Buffer> *writes) {
  BufferAccessCollector collector;
  if (stmt.defined())
    collector(stmt);
  *reads = collector.reads;
  *writes = collector.writes;
  if (collector.unknown_effects) {
    for (const Buffer &buffer : collector.reads)
      AddBufferUnique(writes, buffer);
  }
}

static bool HasParallelItem(const std::vector<Stmt> &stmts) {
  for (const Stmt &stmt : stmts) {
    if (const auto *loop = stmt.as<ForNode>()) {
      if (loop->kind == ForKind::kParallel)
        return true;
    }
  }
  return false;
}

/* Pre-order ordinal of outer Parallel nests, matching PR-B NestCollector.
 * A Parallel nested under a serial loop / if / attr still gets an ordinal;
 * a Parallel nested inside another Parallel does not (it is part of the same
 * nest). */
class ParallelOrdinalCollector : public StmtExprVisitor {
public:
  void VisitStmt_(const ForNode *op) final {
    if (op->kind != ForKind::kParallel) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    if (depth_ == 0)
      ordinals.emplace(op, next_++);
    ++depth_;
    StmtExprVisitor::VisitStmt_(op);
    --depth_;
  }
  std::unordered_map<const ForNode *, int> ordinals;

private:
  int depth_{0};
  int next_{0};
};

class SimdVFCollector : public StmtExprVisitor {
public:
  bool found{false};
  bool has_parallel{false};
  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint == "SIMD_VF")
      found = true;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel)
      has_parallel = true;
    StmtExprVisitor::VisitStmt_(op);
  }
};

/* Fragment definitions/uses in module order. */
class FragmentDefCollector : public StmtExprVisitor {
public:
  int vf_index{0};
  VarIntMap definition;
  VarIntMap first_use;

  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF") {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    int vf = vf_index++;
    for (const Buffer &buffer : op->alloc_buffers) {
      if (IsFragment(buffer))
        definition.emplace(buffer->data, vf);
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

struct ItemRecord {
  Map<String, Any> meta;
  std::vector<Buffer> reads;
  std::vector<Buffer> writes;
  bool p1_liftable{false};
  bool is_metadata{false};
  int group{-1};
  std::string analysis_status{"OPAQUE"};
  std::string group_status{"OPAQUE"};
};

class DSU {
public:
  explicit DSU(size_t n) : parent_(n), rank_(n, 0) {
    for (size_t i = 0; i < n; ++i)
      parent_[i] = i;
  }
  int Find(int x) {
    if (parent_[x] != x)
      parent_[x] = Find(parent_[x]);
    return parent_[x];
  }
  void Unite(int a, int b) {
    a = Find(a);
    b = Find(b);
    if (a == b)
      return;
    if (rank_[a] < rank_[b])
      std::swap(a, b);
    parent_[b] = a;
    if (rank_[a] == rank_[b])
      ++rank_[a];
  }

private:
  std::vector<int> parent_;
  std::vector<int> rank_;
};

class FragmentOutsideUseCollector : public StmtExprVisitor {
public:
  VarSet outside_uses;
  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF") {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    int saved = in_vf_;
    in_vf_ = true;
    StmtExprVisitor::VisitStmt_(op);
    in_vf_ = saved;
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    if (!in_vf_ && IsFragment(op->buffer))
      outside_uses.insert(op->buffer->data);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    if (!in_vf_ && IsFragment(op->buffer))
      outside_uses.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  bool in_vf_{false};
};

struct ProgramInputs {
  std::vector<VSimdCaptureInfo> infos;
  std::vector<VSimdMemoryRegionStatus> memory_regions;
  VarIntMap fragment_definition;
  std::unordered_map<Var, std::vector<int>, VarHash, VarEq> fragment_uses;
  VarSet outside_fragment_uses;
};

static String StorageKind(const Buffer &buffer) {
  if (!buffer.defined())
    return String("UNKNOWN");
  const std::string scope = buffer.scope();
  if (scope == "local.fragment")
    return String("FRAGMENT");
  if (scope.find("shared") != std::string::npos)
    return String("SHARED");
  if (scope == "global")
    return String("GLOBAL");
  return String("UNKNOWN");
}

static Map<String, Any> BufferInfo(const Buffer &buffer) {
  Map<String, Any> info;
  info.Set("buffer", Any(buffer));
  info.Set("name", Any(String(buffer->name)));
  std::ostringstream dtype;
  dtype << buffer->dtype;
  info.Set("dtype", Any(String(dtype.str())));
  std::vector<std::string> shape;
  for (const PrimExpr &extent : buffer->shape)
    shape.push_back(std::string(ExprStr(extent)));
  info.Set("shape", Any(ToStrings(shape)));
  info.Set("scope", Any(StorageKind(buffer)));
  return info;
}

static Array<String> BufferNames(const std::vector<Buffer> &buffers) {
  std::vector<std::string> names;
  for (const Buffer &b : buffers)
    names.push_back(b->name);
  return ToStrings(names);
}

static void SetAccessMeta(ItemRecord *rec) {
  rec->meta.Set("reads", Any(BufferNames(rec->reads)));
  rec->meta.Set("writes", Any(BufferNames(rec->writes)));
  rec->meta.Set("read_buffers",
                Any(Array<Buffer>(rec->reads.begin(), rec->reads.end())));
  rec->meta.Set("write_buffers",
                Any(Array<Buffer>(rec->writes.begin(), rec->writes.end())));
}

static Map<String, Any> RegionPayload(const VSimdRegion &region) {
  Map<String, Any> map;

  std::vector<Map<String, Any>> domain;
  for (const LoopInfo &loop : region.domain) {
    Map<String, Any> axis;
    axis.Set("var", Any(loop.var));
    axis.Set("min", Any(loop.min));
    axis.Set("extent", Any(loop.extent));
    domain.push_back(axis);
  }

  std::vector<Map<String, Any>> accesses;
  for (const VSimdAccess &wrapped : region.accesses) {
    Map<String, Any> access;
    access.Set("buffer", Any(wrapped.access.buffer));
    access.Set("indices", Any(wrapped.access.indices));
    access.Set("value", Any(wrapped.access.value));
    access.Set("is_write", Any(Bool(wrapped.access.is_write)));
    access.Set("dep_axes", Any(ToInts(wrapped.dep_axes)));
    access.Set("value_axes", Any(ToInts(wrapped.value_axes)));
    access.Set("value_shape",
               Any(Array<PrimExpr>(wrapped.value_shape.begin(),
                                   wrapped.value_shape.end())));
    access.Set("has_layout", Any(Bool(wrapped.has_layout)));
    if (wrapped.has_layout)
      access.Set("layout", Any(wrapped.layout));
    accesses.push_back(access);
  }

  std::vector<Map<String, Any>> captures;
  for (const auto &capture : region.captures) {
    Map<String, Any> item;
    item.Set("buffer", Any(capture.first));
    item.Set("kind", Any(String(capture.second)));
    item.Set("storage", Any(StorageKind(capture.first)));
    captures.push_back(item);
  }

  std::vector<Map<String, Any>> outer_vars;
  for (const auto &var : region.outer_vars) {
    Map<String, Any> item;
    item.Set("var", Any(var.first));
    item.Set("kind", Any(String(var.second)));
    outer_vars.push_back(item);
  }

  std::vector<Map<String, Any>> local_views;
  for (const auto &view : region.local_views) {
    Map<String, Any> item;
    item.Set("view", Any(view.first));
    item.Set("base", Any(view.second));
    local_views.push_back(item);
  }

  std::vector<Map<String, Any>> layouts;
  for (const auto &layout : region.layouts) {
    Map<String, Any> item;
    item.Set("buffer", Any(layout.first));
    item.Set("layout", Any(layout.second));
    layouts.push_back(item);
  }

  std::vector<Map<String, Any>> bindings;
  for (const auto &binding : region.bindings) {
    Map<String, Any> item;
    item.Set("var", Any(binding.first));
    item.Set("value", Any(binding.second));
    bindings.push_back(item);
  }

  map.Set("domain", Any(Array<Map<String, Any>>(domain.begin(), domain.end())));
  map.Set("accesses",
          Any(Array<Map<String, Any>>(accesses.begin(), accesses.end())));
  map.Set("captures",
          Any(Array<Map<String, Any>>(captures.begin(), captures.end())));
  map.Set("outer_vars",
          Any(Array<Map<String, Any>>(outer_vars.begin(), outer_vars.end())));
  map.Set("shape_syms", Any(Array<Var>(region.shape_syms.begin(),
                                       region.shape_syms.end())));
  map.Set("local_views",
          Any(Array<Map<String, Any>>(local_views.begin(), local_views.end())));
  map.Set("layouts",
          Any(Array<Map<String, Any>>(layouts.begin(), layouts.end())));
  map.Set("assumptions", Any(Array<PrimExpr>(region.assumptions.begin(),
                                             region.assumptions.end())));
  map.Set("bindings",
          Any(Array<Map<String, Any>>(bindings.begin(), bindings.end())));
  map.Set("dump", Any(String(DumpVSimdRegion(region))));
  return map;
}

struct TileRegionInfo {
  Buffer buffer;
  Array<PrimExpr> indices;
  Array<PrimExpr> extents;
  String access_type;
};

static bool DecodeTileRegion(const PrimExpr &expr, TileRegionInfo *out) {
  const auto *call = expr.as<CallNode>();
  if (call == nullptr)
    return false;
  const auto *op = call->op.as<OpNode>();
  if (op == nullptr ||
      (op->name != "tl.region" && op->name != "tirx.region"))
    return false;
  if (call->args.size() < 2)
    return false;
  const auto *load = call->args[0].as<BufferLoadNode>();
  if (load == nullptr)
    return false;
  out->buffer = load->buffer;
  out->indices = load->indices;
  if (const auto *access = call->args[1].as<StringImmNode>())
    out->access_type = String(access->value);
  std::vector<PrimExpr> extents;
  for (size_t i = 2; i < call->args.size(); ++i)
    extents.push_back(call->args[i]);
  out->extents = Array<PrimExpr>(extents.begin(), extents.end());
  return true;
}

static Map<String, Any> TileRegionMap(const PrimExpr &expr) {
  TileRegionInfo info;
  Map<String, Any> map;
  if (!DecodeTileRegion(expr, &info))
    return map;
  map.Set("buffer", Any(info.buffer));
  map.Set("indices", Any(info.indices));
  map.Set("extents", Any(info.extents));
  map.Set("access_type", Any(info.access_type));
  return map;
}

static Map<String, Any> CopyReducePayload(const CallNode *call,
                                          const std::string &kind) {
  Map<String, Any> map;
  map.Set("kind", Any(String(kind)));
  map.Set("op", Any(String(call->op.as<OpNode>()->name)));
  std::vector<std::string> args;
  for (const PrimExpr &arg : call->args)
    args.push_back(std::string(ExprStr(arg)));
  map.Set("args", Any(ToStrings(args)));
  if (call->args.size() > 0) {
    Map<String, Any> src = TileRegionMap(call->args[0]);
    if (!src.empty())
      map.Set("src_region", Any(src));
  }
  if (call->args.size() > 1) {
    Map<String, Any> dst = TileRegionMap(call->args[1]);
    if (!dst.empty())
      map.Set("dst_region", Any(dst));
  }
  if (const auto *imm = call->args.size() > 2
                            ? call->args[2].as<StringImmNode>()
                            : nullptr) {
    map.Set("reduce_kind", Any(String(imm->value)));
  }
  if (call->args.size() > 3)
    map.Set("reduce_dim", Any(call->args[3]));
  if (call->args.size() > 4)
    map.Set("clear", Any(call->args[4]));
  return map;
}

static std::vector<Buffer> FirstArgBuffers(const CallNode *call) {
  if (call->args.empty())
    return {};
  return BuffersIn(call->args[0]);
}

static std::vector<Buffer> SecondArgBuffers(const CallNode *call) {
  if (call->args.size() < 2)
    return {};
  return BuffersIn(call->args[1]);
}

static void AddRegionItem(std::vector<ItemRecord> *items,
                          const VSimdCaptureInfo &info,
                          const std::string &memory_status, int span) {
  ItemRecord rec;
  VSimdRegionLift lift = BuildMinimalRegion(info, memory_status);
  std::string status = lift.lifted ? "LIFTED" : "OPAQUE";
  rec.p1_liftable = lift.lifted;
  rec.analysis_status = status;
  rec.meta.Set("kind", Any(String("REGION")));
  rec.meta.Set("span_begin", Any(Integer(span)));
  rec.meta.Set("span_end", Any(Integer(span + 1)));
  rec.meta.Set("status", Any(String(status)));
  rec.meta.Set("nest_index", Any(Integer(info.nest_index)));
  rec.meta.Set("parallel_nest_index", Any(Integer(info.nest_index)));
  rec.meta.Set("reason", Any(String(lift.reason)));
  if (lift.lifted)
    rec.meta.Set("region", Any(RegionPayload(lift.region)));
  for (const AccessInfo &acc : info.accesses) {
    if (acc.is_write)
      AddBufferUnique(&rec.writes, acc.buffer);
    else
      AddBufferUnique(&rec.reads, acc.buffer);
  }
  SetAccessMeta(&rec);
  items->push_back(std::move(rec));
}

static void AddCopyReduceItem(std::vector<ItemRecord> *items,
                              const CallNode *call, const std::string &kind,
                              int span) {
  ItemRecord rec;
  rec.p1_liftable = true;
  rec.analysis_status = "TYPED";
  rec.meta = CopyReducePayload(call, kind);
  rec.meta.Set("span_begin", Any(Integer(span)));
  rec.meta.Set("span_end", Any(Integer(span + 1)));
  rec.meta.Set("status", Any(String("TYPED")));
  rec.reads = FirstArgBuffers(call);
  rec.writes = SecondArgBuffers(call);
  // `clear=false` reductions read the old destination value, so the dst is a
  // dependency input as well as an output. Without this edge a producer of the
  // destination and the reduction would end up in different transaction groups.
  if (kind == "REDUCE" && call->args.size() > 4) {
    bool clear = true;
    if (const auto *imm = call->args[4].as<IntImmNode>())
      clear = imm->value != 0;
    if (!clear) {
      for (const Buffer &buffer : rec.writes)
        AddBufferUnique(&rec.reads, buffer);
    }
  }
  SetAccessMeta(&rec);
  if (!rec.reads.empty())
    rec.meta.Set("src", Any(BufferInfo(rec.reads.front())));
  if (!rec.writes.empty())
    rec.meta.Set("dst", Any(BufferInfo(rec.writes.front())));
  items->push_back(std::move(rec));
}

static void AddOpaqueItem(std::vector<ItemRecord> *items, const Stmt &stmt,
                          int span, const std::string &reason) {
  ItemRecord rec;
  rec.p1_liftable = false;
  rec.analysis_status = "OPAQUE";
  rec.meta.Set("kind", Any(String("OPAQUE")));
  rec.meta.Set("span_begin", Any(Integer(span)));
  rec.meta.Set("span_end", Any(Integer(span + 1)));
  rec.meta.Set("status", Any(String("OPAQUE")));
  rec.meta.Set("reason", Any(String(reason)));
  rec.reads.clear();
  rec.writes.clear();
  SplitAccesses(stmt, &rec.reads, &rec.writes);
  SetAccessMeta(&rec);
  items->push_back(std::move(rec));
}

static void AddBindItem(std::vector<ItemRecord> *items,
                        const BindNode *bind, int span) {
  ItemRecord rec;
  rec.p1_liftable = false;
  rec.is_metadata = true;
  rec.analysis_status = "PRESERVE";
  rec.meta.Set("kind", Any(String("BIND")));
  rec.meta.Set("span_begin", Any(Integer(span)));
  rec.meta.Set("span_end", Any(Integer(span + 1)));
  rec.meta.Set("status", Any(String("PRESERVE")));
  rec.meta.Set("var", Any(bind->var));
  rec.meta.Set("value", Any(bind->value));
  SetAccessMeta(&rec);
  items->push_back(std::move(rec));
}

static void AddBufferDeclItem(std::vector<ItemRecord> *items,
                              const Buffer &buffer, const std::string &kind,
                              int span) {
  ItemRecord rec;
  rec.p1_liftable = false;
  rec.is_metadata = true;
  rec.analysis_status = "PRESERVE";
  rec.meta.Set("kind", Any(String(kind)));
  rec.meta.Set("span_begin", Any(Integer(span)));
  rec.meta.Set("span_end", Any(Integer(span + 1)));
  rec.meta.Set("status", Any(String("PRESERVE")));
  rec.meta.Set("buffer", Any(buffer));
  rec.meta.Set("storage", Any(StorageKind(buffer)));
  SetAccessMeta(&rec);
  items->push_back(std::move(rec));
}

static bool IsCopyOp(const std::string &name) {
  return name.find("copy") != std::string::npos;
}
static bool IsReduceOp(const std::string &name) {
  return name.find("reduce") != std::string::npos;
}

static void BuildGroups(std::vector<ItemRecord> *items,
                        const std::vector<Buffer> *all_buffers) {
  DSU dsu(items->size());
  VarIntMap last_writer;
  VarIntMap fragment_item;
  auto is_fragment_var = [&](const Var &var) {
    for (const Buffer &buffer : *all_buffers) {
      if (buffer->data.same_as(var))
        return IsFragment(buffer);
    }
    return false;
  };

  for (size_t i = 0; i < items->size(); ++i) {
    ItemRecord &item = (*items)[i];
    for (const Buffer &buffer : item.reads) {
      auto it = last_writer.find(buffer->data);
      if (it != last_writer.end())
        dsu.Unite(static_cast<int>(i), it->second);
    }
    for (const Buffer &buffer : item.reads) {
      if (!is_fragment_var(buffer->data))
        continue;
      auto it = fragment_item.find(buffer->data);
      if (it != fragment_item.end())
        dsu.Unite(static_cast<int>(i), it->second);
      else
        fragment_item.emplace(buffer->data, static_cast<int>(i));
    }
    for (const Buffer &buffer : item.writes) {
      if (is_fragment_var(buffer->data)) {
        auto it = fragment_item.find(buffer->data);
        if (it != fragment_item.end())
          dsu.Unite(static_cast<int>(i), it->second);
        else
          fragment_item.emplace(buffer->data, static_cast<int>(i));
      }
      last_writer[buffer->data] = static_cast<int>(i);
    }
  }

  std::unordered_map<int, int> root_to_group;
  std::vector<bool> group_liftable;
  std::vector<bool> group_has_meta;
  std::vector<bool> group_has_nonmeta;
  std::vector<int> group_sizes;
  for (size_t i = 0; i < items->size(); ++i) {
    int root = dsu.Find(static_cast<int>(i));
    auto it = root_to_group.find(root);
    int group;
    if (it == root_to_group.end()) {
      group = static_cast<int>(root_to_group.size());
      root_to_group.emplace(root, group);
      group_liftable.push_back(true);
      group_has_meta.push_back(false);
      group_has_nonmeta.push_back(false);
      group_sizes.push_back(0);
    } else {
      group = it->second;
    }
    ItemRecord &item = (*items)[i];
    item.group = group;
    if (item.is_metadata)
      group_has_meta[group] = true;
    else
      group_has_nonmeta[group] = true;
    group_liftable[group] = group_liftable[group] && item.p1_liftable;
    ++group_sizes[group];
  }

  for (ItemRecord &item : *items) {
    if (group_has_meta[item.group] && !group_has_nonmeta[item.group]) {
      item.group_status = "PRESERVE";
    } else if (group_has_meta[item.group] || !group_liftable[item.group]) {
      item.group_status = "OPAQUE";
    } else {
      item.group_status = "P1_LIFTABLE";
    }
    item.meta.Set("group", Any(Integer(item.group)));
    item.meta.Set("group_status", Any(String(item.group_status)));
    item.meta.Set("analysis_status", Any(String(item.analysis_status)));
    std::string effective;
    if (item.group_status == "PRESERVE")
      effective = "PRESERVE";
    else if (item.group_status == "OPAQUE")
      effective = "OPAQUE";
    else
      effective = item.analysis_status;
    item.meta.Set("status", Any(String(effective)));
    item.meta.Set("effective_status", Any(String(effective)));
  }
}

static Map<String, Any> BuildProgram(const SBlock &vf, int vf_index,
                                     const ProgramInputs &inputs) {
  VSimdFlattenedBody flat = FlattenVSimdBody(vf->body);
  const std::vector<Stmt> &stmts = flat.stmts;
  const std::vector<Map<String, Any>> &assumption_meta = flat.assumptions;
  const std::vector<Map<String, Any>> &metadata_attrs = flat.attrs;

  ParallelOrdinalCollector ordinals;
  ordinals(vf->body);
  const bool has_top_level_parallel = HasParallelItem(stmts);
  const bool has_any_parallel = !ordinals.ordinals.empty();

  std::unordered_map<int, VSimdCaptureInfo> infos_by_nest;
  for (const VSimdCaptureInfo &info : inputs.infos) {
    if (info.vf_index == vf_index && info.nest_index >= 0)
      infos_by_nest.emplace(info.nest_index, info);
  }
  std::unordered_map<int, std::string> memory_by_nest;
  for (const VSimdMemoryRegionStatus &status : inputs.memory_regions) {
    if (status.vf_index == vf_index)
      memory_by_nest.emplace(status.nest_index, status.status);
  }

  std::vector<ItemRecord> items;
  for (size_t i = 0; i < stmts.size(); ++i) {
    const Stmt &stmt = stmts[i];
    int span = static_cast<int>(i);
    if (const auto *loop = stmt.as<ForNode>()) {
      if (loop->kind == ForKind::kParallel) {
        auto ordinal_it = ordinals.ordinals.find(loop);
        const int ordinal =
            ordinal_it == ordinals.ordinals.end() ? -1 : ordinal_it->second;
        auto it = infos_by_nest.find(ordinal);
        std::string memory = "OPAQUE";
        auto mit = memory_by_nest.find(ordinal);
        if (mit != memory_by_nest.end())
          memory = mit->second;
        if (it != infos_by_nest.end()) {
          AddRegionItem(&items, it->second, memory, span);
        } else {
          AddOpaqueItem(&items, stmt, span,
                        "no capture info for Parallel nest " +
                            std::to_string(ordinal));
        }
        continue;
      }
      AddOpaqueItem(&items, stmt, span, "non-Parallel loop");
      continue;
    }

    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *call = eval->value.as<CallNode>()) {
        if (const auto *op = call->op.as<OpNode>()) {
          std::string name = op->name;
          if (IsCopyOp(name)) {
            AddCopyReduceItem(&items, call, "COPY", span);
            continue;
          }
          if (IsReduceOp(name)) {
            AddCopyReduceItem(&items, call, "REDUCE", span);
            continue;
          }
        }
      }
      AddOpaqueItem(&items, stmt, span, "unsupported Evaluate");
      continue;
    }

    if (const auto *bind = stmt.as<BindNode>()) {
      AddBindItem(&items, bind, span);
      continue;
    }

    if (stmt.as<AttrStmtNode>()) {
      AddOpaqueItem(&items, stmt, span,
                    "VF-level AttrStmt is kept opaque in v1");
      continue;
    }

    if (const auto *alloc = stmt.as<AllocBufferNode>()) {
      AddBufferDeclItem(&items, alloc->buffer, "ALLOC", span);
      continue;
    }
    if (const auto *decl = stmt.as<DeclBufferNode>()) {
      AddBufferDeclItem(&items, decl->buffer, "DECL", span);
      continue;
    }

    if (stmt.as<IfThenElseNode>()) {
      AddOpaqueItem(&items, stmt, span,
                    "control flow inside VF is not lifted in v1");
      continue;
    }
    if (stmt.as<SBlockNode>()) {
      AddOpaqueItem(&items, stmt, span,
                    "nested SBlock inside VF is not lifted in v1");
      continue;
    }
    if (stmt.as<WhileNode>() || stmt.as<AssertStmtNode>()) {
      AddOpaqueItem(&items, stmt, span,
                    "loop/assert inside VF is not lifted in v1");
      continue;
    }
    AddOpaqueItem(&items, stmt, span, "unsupported statement");
  }

  if (!has_any_parallel) {
    Map<String, Any> empty;
    return empty;
  }
  if (!has_top_level_parallel) {
    LOG(WARNING)
        << "AscendLiftVSimd: VF has a Parallel nest but no top-level "
           "T.Parallel; attaching an all-OPAQUE program";
    for (ItemRecord &item : items) {
      // Metadata items (BIND/ALLOC/DECL) are not liftable work; they must stay
      // PRESERVE so the program keeps its metadata/preserve invariant (a
      // PRESERVE item in an all-OPAQUE VF is still copied verbatim by P2).
      if (item.is_metadata)
        continue;
      item.p1_liftable = false;
      item.analysis_status = "OPAQUE";
    }
  }

  // Copy/Reduce operands may not appear in any Parallel capture; include them
  // in the program capture/fragment tables as well.
  std::vector<Buffer> used_buffers;
  for (const VSimdCaptureInfo &info : inputs.infos) {
    if (info.vf_index != vf_index)
      continue;
    for (const auto &capture : info.captures)
      AddBufferUnique(&used_buffers, capture.first);
  }
  for (const ItemRecord &item : items) {
    for (const Buffer &buffer : item.reads)
      AddBufferUnique(&used_buffers, buffer);
    for (const Buffer &buffer : item.writes)
      AddBufferUnique(&used_buffers, buffer);
  }
  BuildGroups(&items, &used_buffers);

  std::vector<Map<String, Any>> item_maps;
  std::vector<Map<String, Any>> group_maps;
  std::map<int, std::vector<int>> group_items;
  for (size_t i = 0; i < items.size(); ++i) {
    group_items[items[i].group].push_back(static_cast<int>(i));
    item_maps.push_back(items[i].meta);
  }
  for (const auto &kv : group_items) {
    Map<String, Any> group;
    group.Set("id", Any(Integer(kv.first)));
    group.Set("status", Any(String(items[kv.second[0]].group_status)));
    std::vector<int> ids = kv.second;
    group.Set("items", Any(ToInts(ids)));
    group_maps.push_back(group);
  }

  std::vector<Map<String, Any>> captures;
  for (const Buffer &buffer : used_buffers) {
    bool used = false;
    for (const ItemRecord &item : items) {
      for (const Buffer &b : item.reads)
        used = used || b.same_as(buffer);
      for (const Buffer &b : item.writes)
        used = used || b.same_as(buffer);
    }
    if (used) {
      Map<String, Any> capture;
      capture.Set("buffer", Any(buffer));
      capture.Set("name", Any(String(buffer->name)));
      capture.Set("storage", Any(StorageKind(buffer)));
      captures.push_back(capture);
    }
  }

  std::vector<Map<String, Any>> fragments;
  for (const Buffer &buffer : used_buffers) {
    if (!IsFragment(buffer))
      continue;
    std::vector<int> uses;
    auto use_it = inputs.fragment_uses.find(buffer->data);
    if (use_it != inputs.fragment_uses.end())
      uses = use_it->second;
    if (std::find(uses.begin(), uses.end(), vf_index) == uses.end())
      uses.push_back(vf_index);
    Map<String, Any> frag;
    frag.Set("buffer", Any(buffer));
    frag.Set("name", Any(String(buffer->name)));
    auto def_it = inputs.fragment_definition.find(buffer->data);
    int def = def_it == inputs.fragment_definition.end() ? -1 : def_it->second;
    frag.Set("definition_vf", Any(Integer(def)));
    frag.Set("definition_scope",
             Any(String(def >= 0 ? "IN_VF" : "OUTSIDE_VF")));
    std::sort(uses.begin(), uses.end());
    uses.erase(std::unique(uses.begin(), uses.end()), uses.end());
    frag.Set("use_vfs", Any(ToInts(uses)));
    bool uses_outside_vf =
        inputs.outside_fragment_uses.count(buffer->data) != 0;
    bool cross_vf = uses.size() > 1;
    if (def >= 0) {
      for (int vf : uses)
        cross_vf = cross_vf || vf != def;
    }
    bool escapes_vf = def >= 0 &&
                      (uses_outside_vf || std::any_of(
                           uses.begin(), uses.end(),
                           [&](int vf) { return vf != def; }));
    frag.Set("cross_vf", Any(Bool(cross_vf)));
    frag.Set("escapes_vf", Any(Bool(escapes_vf)));
    frag.Set("uses_outside_vf", Any(Bool(uses_outside_vf)));
    fragments.push_back(frag);
  }

  Map<String, Any> program;
  program.Set("version", Any(Integer(1)));
  program.Set("vf_index", Any(Integer(vf_index)));
  program.Set("provenance", Any(String("LIFTED")));
  program.Set("item_count", Any(Integer(static_cast<int64_t>(stmts.size()))));
  program.Set("body_fingerprint",
              Any(IntImm(DataType::Int(64),
                         static_cast<int64_t>(flat.fingerprint))));
  program.Set("assumptions",
              Any(Array<Map<String, Any>>(assumption_meta.begin(),
                                          assumption_meta.end())));
  program.Set("attrs", Any(Array<Map<String, Any>>(metadata_attrs.begin(),
                                                   metadata_attrs.end())));
  program.Set("items", Any(Array<Map<String, Any>>(item_maps.begin(),
                                                   item_maps.end())));
  program.Set("groups", Any(Array<Map<String, Any>>(group_maps.begin(),
                                                    group_maps.end())));
  program.Set("captures",
              Any(Array<Map<String, Any>>(captures.begin(), captures.end())));
  program.Set("fragments", Any(Array<Map<String, Any>>(fragments.begin(),
                                                       fragments.end())));
  VSimdProgram typed;
  std::string validation_error;
  if (!ReadVSimdProgram(program, &typed, &validation_error)) {
    LOG(WARNING) << "AscendLiftVSimd: invalid VSimdProgram: "
                 << validation_error;
    program.Set("validation_error", Any(String(validation_error)));
  }
  return program;
}

class LiftMutator : public StmtExprMutator {
public:
  LiftMutator(ProgramInputs inputs) : inputs_(std::move(inputs)) {}

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF")
      return StmtExprMutator::VisitStmt_(op);
    int vf = vf_index_++;
    SBlock block = ffi::GetRef<SBlock>(op);
    Map<String, Any> program = BuildProgram(block, vf, inputs_);
    if (program.defined() && !program.empty()) {
      block.CopyOnWrite()->annotations.Set(kProgramKey, program);
    }
    return block;
  }

private:
  ProgramInputs inputs_;
  int vf_index_{0};
};

} // namespace

tvm::transform::Pass AscendLiftVSimd() {
  auto pass_func = [](PrimFunc f, const IRModule &mod, PassContext ctx) {
    if (!ctx->GetConfig<Bool>("tl.ascend_simd_lift", Bool(true)).value())
      return f;
    SimdVFCollector collector;
    collector(f->body);
    if (!collector.found || !collector.has_parallel)
      return f;

    ProgramInputs inputs;
    inputs.infos = AnalyzeVSimdCaptureInfo(f);
    inputs.memory_regions = AnalyzeVSimdMemoryRegions(f);

    FragmentDefCollector defs;
    defs(f->body);
    inputs.fragment_definition = defs.definition;
    FragmentOutsideUseCollector outside_uses;
    outside_uses(f->body);
    inputs.outside_fragment_uses = outside_uses.outside_uses;
    for (const VSimdCaptureInfo &info : inputs.infos) {
      for (const auto &capture : info.captures) {
        if (IsFragment(capture.first)) {
          inputs.fragment_uses[capture.first->data].push_back(info.vf_index);
        }
      }
    }

    LiftMutator mutator(std::move(inputs));
    PrimFuncNode *write_ptr = f.CopyOnWrite();
    write_ptr->body = mutator(std::move(write_ptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendLiftVSimd", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendLiftVSimd", []() {
    return AscendLiftVSimd();
  });
}

} // namespace tl
} // namespace tvm
