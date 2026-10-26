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

static std::string AccessString(const AccessInfo &acc) {
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
  return os.str();
}

} // namespace

VSimdRegionLift BuildMinimal1DRegion(const VSimdCaptureInfo &info,
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
  if (info.domain.size() != 1) {
    result.reason = "PR-C minimal lift only supports one T.Parallel level";
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
  result.region.accesses = info.accesses;
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
  for (const AccessInfo &acc : region.accesses)
    os << "    " << AccessString(acc) << "\n";
  return os.str();
}

Stmt ExpandVSimdRegionToLoops(const VSimdRegion &region) {
  std::vector<Stmt> stores;
  for (const AccessInfo &acc : region.accesses) {
    if (acc.is_write)
      stores.push_back(BufferStore(acc.buffer, acc.value, acc.indices));
  }
  if (stores.empty())
    return Evaluate(0);
  Stmt body = stores.size() == 1 ? stores[0] : SeqStmt(stores);
  Stmt result;
  if (region.domain.empty()) {
    result = body;
  } else {
    const LoopInfo &loop = region.domain[0];
    // Reference expansion is intentionally a serial scalar loop; it is a
    // semantic oracle, not the P3 physical lowering.
    result = For(loop.var, loop.min, loop.extent, ForKind::kSerial, body);
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
    if (verify.loops.size() != 1) {
      result.ok = false;
      result.reason = "expanded loop count does not match region domain";
      return result;
    }
    const ForNode *loop = verify.loops[0];
    const LoopInfo &domain = region.domain[0];
    if (!loop->loop_var.same_as(domain.var) ||
        !ffi::StructuralEqual()(loop->min, domain.min) ||
        !ffi::StructuralEqual()(loop->extent, domain.extent)) {
      result.ok = false;
      result.reason = "expanded domain does not match region domain";
      return result;
    }
  }

  std::vector<const AccessInfo *> writes;
  for (const AccessInfo &acc : region.accesses) {
    if (acc.is_write)
      writes.push_back(&acc);
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
    VSimdRegionLift lift = BuildMinimal1DRegion(info, memory_status);
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
