/*!
 * \file vsimd_p2_input.h
 * \brief P1 -> P2 hand-off: per-nest plan inputs whose statement is a parallel
 *        For chain that the existing ``build_plan`` can consume.
 *
 * The hand-off deliberately reuses P1: every decision (memory legality,
 * capture/scope closure, G1-G6, store reconstruction) stays in
 * ``vsimd_memory_legality`` / ``vsimd_capture_scope`` / ``vsimd_region``. This
 * layer only adapts the reference expansion (``For(kSerial)``) back to
 * ``For(kParallel)`` and restores the outer ``Bind`` definitions that the
 * reference expander does not reproduce.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_VSIMD_P2_INPUT_H_
#define TVM_TL_ASCEND_TRANSFORM_VSIMD_P2_INPUT_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <string>
#include <utility>
#include <vector>

#include "../vsimd_region.h"

namespace tvm {
namespace tl {

/*!
 * \brief One nest's P1 -> P2 hand-off record.
 *
 * ``status == "LIFTED"`` means every P1 check passed and ``stmt`` holds the
 * reference expansion with each loop kind switched back to ``kParallel``, so it
 * can be handed straight to the Python ``build_plan``. A P1 guarantee does not
 * imply a P2 guarantee: ``build_plan`` re-analyses the statement and may still
 * reject it (Branch B: GM operand, bool store, true-random index, nonzero
 * lower bound).
 */
struct VSimdP2Input {
  int vf_index{-1};
  int nest_index{-1};

  // LIFTED / FALLBACK / INTERNAL_ERROR / OPAQUE / NOT_FOUND
  std::string status;
  std::string reason;

  // Valid when LIFTED: typed PR-D region, kept for debugging and for a future
  // PlanOptions hand-off (accesses carry dep_axes/value_axes/value_shape).
  VSimdRegion region;

  // Valid when LIFTED: N-level ``For(kParallel)`` chain (1..N levels).
  tirx::Stmt stmt;
};

/*!
 * \brief Return every nest's P2 input, in P1 report order.
 *
 * Nests whose capture is opaque (no analyzable T.Parallel nest) are reported
 * with ``status == "OPAQUE"`` and no statement, matching the P1 report.
 */
TVM_DLL std::vector<VSimdP2Input>
BuildP2PlanInputs(const tirx::PrimFunc &func);

/*!
 * \brief Rewrite the reference expansion into a parallel For chain.
 *
 * ``ExpandVSimdRegionToLoops`` emits ``For(kSerial)`` levels as a semantic
 * oracle. ``build_plan`` only starts its SCoP at a ``T.Parallel`` axis, so the
 * whole chain is switched back to ``kParallel``. This is a pure ForKind change:
 * the original domain was parallel and PR-A proved there are no cross-iteration
 * dependences.
 *
 * Accepted shapes (``DeclBuffer`` siblings are dropped: P3 emits stores in
 * place, where the view is still in scope):
 *   - ``For(kSerial, var, min, extent, body)`` (recursively);
 *   - ``SeqStmt`` with exactly one loop and only ``DeclBuffer`` siblings.
 *
 * Returns false and fills ``error`` for sibling loops, a missing loop, or a
 * non-serial loop (all of which mean a P1 expander bug).
 */
TVM_DLL bool AsParallelNest(const tirx::Stmt &expanded, tirx::Stmt *out,
                            std::string *error);

/*!
 * \brief Rebuild the P2 plan input from a lifted region (the P1 -> P2 contract).
 *
 * This is the only conversion P2 needs: the region is the typed hand-off, the
 * returned statement is its reference expansion with every loop switched back
 * to ``ForKind::kParallel`` (``AsParallelNest(ExpandVSimdRegionToLoops(region))``).
 *
 * Outer ``Bind`` variables are deliberately left free here: P3 replaces the nest
 * in place, where the binding is still in scope, and the planner treats a free
 * extent as a symbolic one. Only the offline hand-off
 * (:func:`BuildP2PlanInputs`) inlines them, because its statement is detached
 * from that scope.
 *
 * Returns false and fills \p error when the expansion is not a single parallel
 * chain; that is a P1 internal inconsistency (never user IR), so callers must
 * fall back to the untouched nest.
 */
TVM_DLL bool RegionToPlanInput(const VSimdRegion &region, tirx::Stmt *out,
                               std::string *error);

/*!
 * \brief Inline outer ``Bind`` definitions that remain free in \p stmt.
 *
 * The reference expander only rebuilds stores, so an extent/index that used an
 * outer ``Bind`` (e.g. ``tail = T.min(n - bx * bn, bn)``) would hand an
 * undefined Var to ``build_plan``. Only variables that are free in \p stmt are
 * substituted; Bind values that are calls are skipped (same rule as P2's
 * ``_collect_aliases``). Chained bindings are resolved with a bounded number of
 * passes. Remaining undefined vars are left alone so ``build_plan`` reports them
 * as Unsupported (Branch B).
 */
TVM_DLL tirx::Stmt
RestoreBindings(const tirx::Stmt &stmt,
                const std::vector<std::pair<tirx::Var, PrimExpr>> &bindings);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VSIMD_P2_INPUT_H_
