/*!
 * \file vsimd_region.h
 * \brief Minimal 1-D VSimdRegion carrier, dumper, and reference expander (PR-C).
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_VSIMD_REGION_H_
#define TVM_TL_ASCEND_TRANSFORM_VSIMD_REGION_H_

#include <tvm/ffi/string.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <string>
#include <utility>
#include <vector>

#include "vsimd_analysis_common.h"
#include "vsimd_capture_scope.h"

namespace tvm {
namespace tl {

/*!
 * \brief Hardware-independent region carrier for a minimal 1-D Parallel nest.
 *
 * The region is the typed hand-off from P1 to the later P2/P3 stages. It keeps
 * the logical access maps and store values; it does not contain physical
 * layout, lane, or instruction choices. The TIR side of PR-C is a debug-only
 * AttrStmt carrier (see BuildVSimdRegionCarrier); the executable P2/P3 carrier
 * will consume this structure directly later.
 */
struct VSimdRegion {
  int vf_index{-1};   // source SIMD_VF index
  int nest_index{-1}; // source T.Parallel nest index

  // Parallel domain; PR-C only accepts exactly one loop level.
  std::vector<vsimd::LoopInfo> domain;

  // Loads and stores in source order; stores carry their value expression.
  std::vector<vsimd::AccessInfo> accesses;

  // Context needed to recreate the region after lift: buffer captures,
  // scalar/thread captures, shape symbols, local views, layouts, arithmetic
  // contract, and relevant Bind defs. Layouts are mandatory for the later
  // layout-aware store / nd2nz / vsstb path, so they are not dropped here.
  std::vector<std::pair<tirx::Buffer, std::string>> captures;
  std::vector<std::pair<tirx::Var, std::string>> outer_vars;
  std::vector<tirx::Var> shape_syms;
  std::vector<std::pair<tirx::Buffer, tirx::Buffer>> local_views;
  std::vector<std::pair<tirx::Buffer, Layout>> layouts;
  std::vector<PrimExpr> assumptions;
  std::vector<std::pair<tirx::Var, PrimExpr>> bindings;
};

/*! \brief Result of the minimal 1-D lift decision for one nest. */
struct VSimdRegionLift {
  bool lifted{false};
  bool internal_error{false}; // expander self-check failed: compiler bug, not user IR
  std::string reason;         // fallback/internal-error reason
  VSimdRegion region;         // valid only when lifted=true
};

/*!
 * \brief Build a region only for a proven, closed, single-level 1-D nest.
 *
 * `memory_status` is the PR-A per-VF status (LEGAL is required). All other
 * cases return lifted=false with a diagnostic; this function never mutates IR.
 */
TVM_DLL VSimdRegionLift
BuildMinimal1DRegion(const VSimdCaptureInfo &info,
                     const std::string &memory_status);

/*! \brief Reconstruct scalar serial loops from a region (reference expander). */
TVM_DLL tirx::Stmt ExpandVSimdRegionToLoops(const VSimdRegion &region);

/*! \brief Result of the cheap structural expander/lift consistency check. */
struct VSimdRegionVerify {
  bool ok{true};
  std::string reason;
};

/*!
 * \brief Check that the expanded statement still declares every local view,
 * uses only captured/declared buffers, and contains exactly the region stores
 * in source order. This is a structural guard, not a numerical oracle.
 */
TVM_DLL VSimdRegionVerify VerifyExpandedRegion(const VSimdRegion &region,
                                               const tirx::Stmt &expanded);

/*! \brief Debug-only TIR carrier marker containing the region dump. */
TVM_DLL tirx::Stmt BuildVSimdRegionCarrier(const VSimdRegion &region);

/*! \brief Human-readable region dump. */
TVM_DLL std::string DumpVSimdRegion(const VSimdRegion &region);

/*! \brief Lift every eligible nest and render the region/expanded/carrier dump. */
TVM_DLL ffi::String AnalyzeVSimdRegionReport(const tirx::PrimFunc &func);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VSIMD_REGION_H_
