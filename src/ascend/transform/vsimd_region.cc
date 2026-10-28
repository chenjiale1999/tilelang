/*!
 * \file vsimd_region.cc
 * \brief Minimal 1-D VSimdRegion builder, dumper, and reference expander.
 */
#include "vsimd_region.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "vsimd_memory_legality.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace vsimd;
using ffi::String;

namespace {

static std::string AccessString(const VSimdAccess &wrapped) {
  const AccessInfo &acc = wrapped.access;
  std::ostringstream os;
  os << (acc.is_write ? "W " : "R ") << acc.buffer->name << "[";
  for (size_t i = 0; i < acc.indices.size(); ++i) {
    if (i)
      os << ", ";
    os << acc.indices[i];
  }
  os << "]";
  if (acc.is_write && acc.value.defined())
    os << " = " << acc.value;
  os << " shape=[";
  for (size_t i = 0; i < wrapped.value_shape.size(); ++i) {
    if (i)
      os << ",";
    os << wrapped.value_shape[i];
  }
  os << "] axes=[";
  for (size_t i = 0; i < wrapped.value_axes.size(); ++i) {
    if (i)
      os << ",";
    os << wrapped.value_axes[i];
  }
  os << "] dep={";
  for (size_t i = 0; i < wrapped.dep_axes.size(); ++i) {
    if (i)
      os << ",";
    os << wrapped.dep_axes[i];
  }
  os << "}";
  os << " layout=" << (wrapped.has_layout ? "annotated" : "none");
  return os.str();
}

static bool ExprUsesVar(const PrimExpr &expr, const Var &var) {
  if (!expr.defined())
    return false;
  for (const Var &used : tirx::UndefinedVars(expr)) {
    if (used.same_as(var))
      return true;
  }
  return false;
}

static std::vector<int> ComputeDepAxes(const AccessInfo &acc,
                                       const std::vector<LoopInfo> &domain) {
  std::vector<int> dep;
  for (size_t axis = 0; axis < domain.size(); ++axis) {
    bool used = false;
    for (const PrimExpr &index : acc.indices) {
      if (ExprUsesVar(index, domain[axis].var)) {
        used = true;
        break;
      }
    }
    if (used)
      dep.push_back(static_cast<int>(axis));
  }
  return dep;
}

} // namespace

VSimdRegionLift BuildMinimalRegion(const VSimdCaptureInfo &info,
                                   const std::string &memory_status) {
  VSimdRegionLift result;
  if (info.status != "CLOSED") {
    result.reason = "capture/scope is not CLOSED";
    return result;
  }
  if (memory_status != "LEGAL") {
    result.reason = "memory legality is " + memory_status;
    return result;
  }
  if (info.domain.empty()) {
    result.reason = "region requires at least one T.Parallel axis";
    return result;
  }

  bool has_write = false;
  for (const AccessInfo &acc : info.accesses) {
    if (acc.buffer.scope() == "local.var") {
      result.reason = "local.var access is not liftable";
      return result;
    }
    if (acc.buffer->shape.size() != acc.indices.size()) {
      result.reason = "access rank does not match buffer rank";
      return result;
    }
    has_write = has_write || acc.is_write;
  }
  if (!has_write) {
    result.reason = "nest has no BufferStore";
    return result;
  }

  result.lifted = true;
  result.region.vf_index = info.vf_index;
  result.region.nest_index = info.nest_index;
  result.region.domain = info.domain;
  for (const AccessInfo &acc : info.accesses) {
    VSimdAccess wrapped;
    wrapped.access = acc;
    wrapped.dep_axes = ComputeDepAxes(acc, info.domain);
    for (size_t axis = 0; axis < info.domain.size(); ++axis) {
      wrapped.value_axes.push_back(static_cast<int>(axis));
      wrapped.value_shape.push_back(info.domain[axis].extent);
    }
    for (const auto &layout : info.layouts) {
      if (layout.first->data.same_as(acc.buffer->data)) {
        wrapped.has_layout = true;
        wrapped.layout = layout.second;
        break;
      }
    }
    result.region.accesses.push_back(std::move(wrapped));
  }
  result.region.captures = info.captures;
  result.region.outer_vars = info.outer_vars;
  result.region.shape_syms = info.shape_syms;
  result.region.local_views = info.local_views;
  result.region.layouts = info.layouts;
  result.region.assumptions = info.assumptions;
  result.region.bindings = info.bindings;

  Stmt expanded = ExpandVSimdRegionToLoops(result.region);
  VSimdRegionVerify verify = VerifyExpandedRegion(result.region, expanded);
  if (!verify.ok) {
    LOG(WARNING) << "[vsimd] internal region expansion verification failed "
                    "(compiler bug): "
                 << verify.reason;
    result.lifted = false;
    result.internal_error = true;
    result.reason = "internal expander verification failed: " + verify.reason;
  }
  return result;
}

VSimdRegionLift BuildMinimal1DRegion(const VSimdCaptureInfo &info,
                                     const std::string &memory_status) {
  if (info.domain.size() != 1) {
    VSimdRegionLift result;
    result.reason = "PR-C minimal lift only supports one T.Parallel level";
    return result;
  }
  return BuildMinimalRegion(info, memory_status);
}

std::string DumpVSimdRegion(const VSimdRegion &region) {
  std::ostringstream os;
  os << "region VF#" << region.vf_index << " nest#" << region.nest_index
     << "\n";
  os << "  domain:";
  for (const LoopInfo &loop : region.domain) {
    os << " " << loop.var->name_hint << " in [" << loop.min << ", "
       << (loop.min + loop.extent) << ")";
  }
  os << "\n";
  os << "  captures:";
  for (const auto &capture : region.captures)
    os << " " << capture.first->name << "[" << capture.second << "]";
  os << "\n";
  if (!region.outer_vars.empty()) {
    os << "  outer:";
    for (const auto &var : region.outer_vars)
      os << " " << var.first->name_hint << "[" << var.second << "]";
    os << "\n";
  }
  if (!region.shape_syms.empty()) {
    os << "  shape_syms:";
    for (const Var &var : region.shape_syms)
      os << " " << var->name_hint;
    os << "\n";
  }
  if (!region.local_views.empty()) {
    os << "  local_views:";
    for (const auto &view : region.local_views)
      os << " " << view.first->name << " -> " << view.second->name;
    os << "\n";
  }
  if (!region.layouts.empty()) {
    os << "  layouts:";
    for (const auto &layout : region.layouts)
      os << " " << layout.first->name;
    os << "\n";
  }
  if (!region.assumptions.empty()) {
    os << "  assumptions:";
    for (const PrimExpr &expr : region.assumptions)
      os << " " << expr;
    os << "\n";
  }
  os << "  accesses:\n";
  for (const VSimdAccess &wrapped : region.accesses)
    os << "    " << AccessString(wrapped) << "\n";
  return os.str();
}

Stmt ExpandVSimdRegionToLoops(const VSimdRegion &region) {
  std::vector<Stmt> stores;
  for (const VSimdAccess &wrapped : region.accesses) {
    const AccessInfo &acc = wrapped.access;
    if (acc.is_write)
      stores.push_back(BufferStore(acc.buffer, acc.value, acc.indices));
  }
  if (stores.empty())
    return Evaluate(0);
  Stmt result = stores.size() == 1 ? stores[0] : SeqStmt(stores);
  // Reference expansion is intentionally serial scalar loops; it is a
  // semantic oracle, not the P3 physical lowering.
  for (auto it = region.domain.rbegin(); it != region.domain.rend(); ++it) {
    result = For(it->var, it->min, it->extent, ForKind::kSerial, result);
  }
  // T.view aliases are DeclBuffer definitions, not allocations. They must be
  // declared before the reconstructed loop or the expanded IR references an
  // undeclared view buffer.
  for (auto it = region.local_views.rbegin();
       it != region.local_views.rend(); ++it) {
    result = SeqStmt({DeclBuffer(it->first), result});
  }
  return result;
}

Stmt BuildVSimdRegionCarrier(const VSimdRegion &region) {
  // Debug-only carrier. It is not executable and never enters the default
  // pipeline; the executable P2/P3 carrier will consume VSimdRegion directly.
  return AttrStmt(StringImm(DumpVSimdRegion(region)), "tl.vsimd.region", 1,
                  Evaluate(0));
}

VSimdRegionVerify VerifyExpandedRegion(const VSimdRegion &region,
                                       const Stmt &expanded) {
  using BufferSet =
      std::unordered_set<Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
  BufferSet visible;
  for (const auto &capture : region.captures)
    visible.insert(capture.first);
  for (const auto &view : region.local_views)
    visible.insert(view.first);

  class VerifyVisitor : public StmtExprVisitor {
  public:
    BufferSet declared;
    std::vector<const BufferStoreNode *> stores;
    std::vector<const ForNode *> loops;
    std::vector<Buffer> loads;
    std::string error;

    void VisitStmt_(const DeclBufferNode *op) final {
      declared.insert(op->buffer);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const ForNode *op) final {
      loops.push_back(op);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      stores.push_back(op);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const BufferLoadNode *op) final {
      loads.push_back(op->buffer);
      StmtExprVisitor::VisitExpr_(op);
    }
  } verify;

  verify(expanded);

  VSimdRegionVerify result;
  for (const auto &view : region.local_views) {
    if (!verify.declared.count(view.first)) {
      result.ok = false;
      result.reason = "local view is not declared: " +
                      std::string(view.first->name);
      return result;
    }
  }

  auto visible_buffer = [&](const Buffer &buffer) {
    return visible.count(buffer) || verify.declared.count(buffer);
  };
  for (const Buffer &buffer : verify.loads) {
    if (!visible_buffer(buffer)) {
      result.ok = false;
      result.reason = "load from undeclared buffer: " +
                      std::string(buffer->name);
      return result;
    }
  }

  if (region.domain.empty()) {
    if (!verify.loops.empty()) {
      result.ok = false;
      result.reason = "expanded has a loop but region domain is empty";
      return result;
    }
  } else {
    if (verify.loops.size() != region.domain.size()) {
      result.ok = false;
      result.reason = "expanded loop count does not match region domain";
      return result;
    }
    for (size_t i = 0; i < region.domain.size(); ++i) {
      const ForNode *loop = verify.loops[i];
      const LoopInfo &domain = region.domain[i];
      if (!loop->loop_var.same_as(domain.var) ||
          !ffi::StructuralEqual()(loop->min, domain.min) ||
          !ffi::StructuralEqual()(loop->extent, domain.extent)) {
        result.ok = false;
        result.reason = "expanded domain does not match region domain";
        return result;
      }
    }
  }

  std::vector<const AccessInfo *> writes;
  for (const VSimdAccess &wrapped : region.accesses) {
    if (wrapped.access.is_write)
      writes.push_back(&wrapped.access);
  }
  if (verify.stores.size() != writes.size()) {
    result.ok = false;
    result.reason = "expanded store count does not match region stores";
    return result;
  }
  for (size_t i = 0; i < writes.size(); ++i) {
    const BufferStoreNode *store = verify.stores[i];
    const AccessInfo &acc = *writes[i];
    if (!store->buffer.same_as(acc.buffer) ||
        !ffi::StructuralEqual()(store->indices, acc.indices) ||
        !ffi::StructuralEqual()(store->value, acc.value)) {
      result.ok = false;
      result.reason = "expanded store does not match region access";
      return result;
    }
  }
  return result;
}

String AnalyzeVSimdRegionReport(const PrimFunc &func) {
  std::vector<std::string> statuses = AnalyzeVSimdMemoryStatuses(func);
  std::vector<VSimdCaptureInfo> infos = AnalyzeVSimdCaptureInfo(func);

  std::ostringstream out;
  for (const VSimdCaptureInfo &info : infos) {
    if (info.nest_index < 0) {
      out << "VF#" << info.vf_index << " status=OPAQUE";
      if (!info.message.empty())
        out << " reason=" << info.message;
      out << "\n";
      continue;
    }
    std::string memory_status =
        info.vf_index >= 0 &&
                static_cast<size_t>(info.vf_index) < statuses.size()
            ? statuses[info.vf_index]
            : "MISSING";
    VSimdRegionLift lift = BuildMinimalRegion(info, memory_status);
    out << "VF#" << info.vf_index << " nest " << info.nest_index << ": ";
    if (!lift.lifted) {
      if (lift.internal_error) {
        out << "status=INTERNAL_ERROR reason=" << lift.reason << "\n";
      } else {
        out << "status=FALLBACK reason=" << lift.reason << "\n";
      }
      continue;
    }
    Stmt expanded_stmt = ExpandVSimdRegionToLoops(lift.region);
    out << "status=LIFTED\n";
    out << DumpVSimdRegion(lift.region);
    std::ostringstream expanded;
    expanded << expanded_stmt;
    out << "  expanded:\n    " << expanded.str() << "\n";
    out << "  verify: OK\n";
    std::ostringstream carrier;
    carrier << BuildVSimdRegionCarrier(lift.region);
    out << "  carrier:\n    " << carrier.str() << "\n";
  }
  if (out.str().empty())
    return String("no SIMD_VF blocks\n");
  return String(out.str());
}

} // namespace tl
} // namespace tvm
