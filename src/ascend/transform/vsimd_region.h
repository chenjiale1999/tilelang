/*!
 * \file vsimd_region.h
 * \brief VSimdRegion carrier, dumper, and reference expander (PR-C/PR-D).
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
 * \brief One domain-bound access with its logical value metadata.
 *
 * Access vs value: `access` is the buffer/map side (what memory is touched);
 * `value_shape`/`value_axes`/`dep_axes` describe the logical value side.
 * Do not conflate them.
 *
 * v1 value metadata is intentionally a placeholder: every domain-bound
 * access consumes/produces a value whose logical shape/axes equal the
 * parallel domain. These fields are NOT a value identity and NOT the access
 * map. Once value ops / fusion are introduced, a value will carry its own
 * shape/axes (for example a reduced value), and these fields will describe
 * that value at this access site. Broadcast is therefore never inferred from
 * value_shape/value_axes; use dep_axes (the axes actually used by the map).
 */
struct VSimdAccess {
  vsimd::AccessInfo access;
  std::vector<int> dep_axes;      // axes used by access.indices only
  std::vector<int> value_axes;    // v1: domain axes; future: value axes
  std::vector<PrimExpr> value_shape; // v1: domain extents; future: value shape
};

/*!
 * \brief Hardware-independent region carrier for a rectangular Parallel nest.
 *
 * The region is the typed hand-off from P1 to the later P2/P3 stages. It keeps
 * the logical access maps and store values; it does not contain physical
 * layout, lane, or instruction choices. The TIR side is a debug-only AttrStmt
 * carrier (see BuildVSimdRegionCarrier); the executable P2/P3 carrier will
 * consume this structure directly later.
 */
struct VSimdRegion {
  int vf_index{-1};   // source SIMD_VF index
  int nest_index{-1}; // source T.Parallel nest index

  // Parallel domain, outer to inner. PR-C accepts one level; PR-D accepts
  // arbitrary rectangular rank (the expander/verifier are recursive).
  std::vector<vsimd::LoopInfo> domain;

  // Loads and stores in source order; each carries value shape/axes/dep_axes.
  std::vector<VSimdAccess> accesses;

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

/*! \brief Result of the region lift decision for one nest. */
struct VSimdRegionLift {
  bool lifted{false};
  bool internal_error{false}; // expander self-check failed: compiler bug, not user IR
  std::string reason;         // fallback/internal-error reason
  VSimdRegion region;         // valid only when lifted=true
};

/*!
 * \brief Build a region for a proven, closed, rectangular nest.
 *
 * `memory_status` is the PR-A per-VF status (LEGAL is required). All other
 * cases return lifted=false with a diagnostic; this function never mutates IR.
 * PR-D enables arbitrary rectangular rank, broadcast (dep_axes), index/stride
 * and transpose maps.
 */
TVM_DLL VSimdRegionLift
BuildMinimalRegion(const VSimdCaptureInfo &info,
                   const std::string &memory_status);

/*! \brief Backward-compatible PR-C entry: require exactly one parallel axis. */
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
