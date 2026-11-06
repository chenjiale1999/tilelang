/*!
 * \file plan.cc
 * \brief Planning layer implementation (nest split / axis ranking / reductions).
 */
#include "plan.h"

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "primitives.h"

namespace tvm {
namespace tl {
namespace polyhedral {

namespace {

using tirx::ForKind;
using tirx::Stmt;

/*! \brief Python repr of a list of strings: ``['a', 'b']``. */
std::string PyStrList(const std::vector<std::string>& items) {
  std::string out = "[";
  for (size_t i = 0; i < items.size(); ++i) {
    out += (i == 0 ? "'" : ", '") + items[i] + "'";
  }
  return out + "]";
}

/*! \brief Python repr of a list of ints: ``[0, 5]``. */
std::string PyIntList(const std::vector<int64_t>& items) {
  std::string out = "[";
  for (size_t i = 0; i < items.size(); ++i) {
    out += (i == 0 ? "" : ", ") + std::to_string(items[i]);
  }
  return out + "]";
}

template <typename T>
ffi::Array<T> ToArray(const std::vector<T>& items) {
  ffi::Array<T> out;
  for (const T& item : items) out.push_back(item);
  return out;
}

std::vector<std::string> ToStrings(const ffi::Array<ffi::String>& items) {
  std::vector<std::string> out;
  out.reserve(items.size());
  for (const ffi::String& item : items) out.push_back(item.operator std::string());
  return out;
}

ffi::Array<ffi::String> ToStringArray(const std::vector<std::string>& items) {
  ffi::Array<ffi::String> out;
  for (const std::string& item : items) out.push_back(item);
  return out;
}

std::string JoinNames(const std::vector<Axis>& axes) {
  std::string out;
  for (size_t i = 0; i < axes.size(); ++i) {
    out += (i == 0 ? "" : ", ") + axes[i]->name.operator std::string();
  }
  return out;
}

/*! \brief ``_seq`` over statement vectors. */
Stmt SeqStmts(const std::vector<Stmt>& stmts) {
  ffi::Array<Stmt> array;
  for (const Stmt& stmt : stmts) array.push_back(stmt);
  return SeqOf(array);
}

/*! \brief Substitute a variable mapping into an expression. */
PrimExpr SubstituteVars(const PrimExpr& expr,
                        const std::map<const tirx::VarNode*, PrimExpr>& mapping) {
  return tirx::Substitute(expr, [&](const tirx::Var& var) -> ffi::Optional<PrimExpr> {
    auto it = mapping.find(var.get());
    if (it == mapping.end()) return ffi::Optional<PrimExpr>();
    return ffi::Optional<PrimExpr>(it->second);
  });
}

Stmt SubstituteVarsStmt(const Stmt& stmt,
                        const std::map<const tirx::VarNode*, PrimExpr>& mapping) {
  return tirx::Substitute(stmt, [&](const tirx::Var& var) -> ffi::Optional<PrimExpr> {
    auto it = mapping.find(var.get());
    if (it == mapping.end()) return ffi::Optional<PrimExpr>();
    return ffi::Optional<PrimExpr>(it->second);
  });
}

int AxisWeight(AccessClass kind) {
  switch (kind) {
    case AccessClass::kContiguous:
      return 8;
    case AccessClass::kPiecewise:
      return 4;
    case AccessClass::kStrided:
      return 3;
    case AccessClass::kPermute:
      return 3;
    case AccessClass::kIndirect:
      return 2;
    case AccessClass::kUniform:
      return 1;
    case AccessClass::kTrueRandom:
      return 0;
  }
  return 0;
}

/*! \brief ``_SegmentSpecs``: normalized analysis input of one segment. */
struct SegmentSpecs {
  std::vector<Axis> axes;
  std::optional<Axis> serial_axis;
  std::vector<StoreSpec> pre_specs;
  std::vector<AccessInfo> pre_infos;
  std::vector<StoreSpec> main_specs;
  std::vector<AccessInfo> main_infos;
  std::vector<StoreSpec> post_specs;
  std::vector<AccessInfo> post_infos;
  int64_t lanes{0};

  std::vector<AccessInfo> AllInfos() const {
    std::vector<AccessInfo> infos;
    infos.insert(infos.end(), pre_infos.begin(), pre_infos.end());
    infos.insert(infos.end(), main_infos.begin(), main_infos.end());
    infos.insert(infos.end(), post_infos.begin(), post_infos.end());
    return infos;
  }
  std::vector<StoreSpec> AllSpecs() const {
    std::vector<StoreSpec> specs;
    specs.insert(specs.end(), pre_specs.begin(), pre_specs.end());
    specs.insert(specs.end(), main_specs.begin(), main_specs.end());
    specs.insert(specs.end(), post_specs.begin(), post_specs.end());
    return specs;
  }
  VectorPlan plan_for(const Axis& vector_axis, arith::Analyzer* analyzer,
                      const PlanOptions& options) const;
};

/* ------------------------------------------------------------------------- */
/* forward declarations                                                       */
/* ------------------------------------------------------------------------- */

VectorPlan BuildPlanForAxis(const std::vector<Axis>& axes,
                            std::optional<Axis> serial_axis,
                            const std::vector<StoreSpec>& pre_specs,
                            const std::vector<AccessInfo>& pre_infos,
                            const std::vector<StoreSpec>& main_specs,
                            const std::vector<AccessInfo>& main_infos,
                            const std::vector<StoreSpec>& post_specs,
                            const std::vector<AccessInfo>& post_infos,
                            const Axis& vector_axis, int64_t lanes,
                            arith::Analyzer* analyzer, const PlanOptions& options);

VectorPlan SegmentSpecs::plan_for(const Axis& vector_axis,
                                  arith::Analyzer* analyzer,
                                  const PlanOptions& options) const {
  return BuildPlanForAxis(axes, serial_axis, pre_specs, pre_infos, main_specs, main_infos,
                          post_specs, post_infos, vector_axis, lanes, analyzer, options);
}

/* ------------------------------------------------------------------------- */
/* iteration domain                                                          */
/* ------------------------------------------------------------------------- */

void RequireZeroMin(const tirx::For& loop, arith::Analyzer* analyzer) {
  std::optional<int64_t> min = ConstInt(loop->min, analyzer);
  if (!min.has_value() || *min != 0) {
    throw Unsupported(ForKindName(loop->kind) + " loop '" +
                      loop->loop_var->name_hint.operator std::string() +
                      "' must start at 0 to be vectorized (min=" + PrintExpr(loop->min) +
                      ")");
  }
}

std::pair<std::vector<Axis>, Stmt> AbsorbParallelChain(const Stmt& stmt,
                                                       arith::Analyzer* analyzer) {
  std::vector<Axis> axes;
  Stmt current = stmt;
  while (const auto* loop = current.as<tirx::ForNode>()) {
    if (loop->kind != ForKind::kParallel) break;
    RequireZeroMin(ffi::GetRef<tirx::For>(loop), analyzer);
    axes.push_back(MakeAxis(loop->loop_var, loop->extent, static_cast<int64_t>(axes.size()),
                            ForKind::kParallel));
    current = loop->body;
  }
  return {axes, current};
}

void RecordSerialReason(std::vector<std::string>* notes, const tirx::For& loop,
                        const std::string& reason) {
  if (notes == nullptr) return;
  std::string note = "serial loop '" + loop->loop_var->name_hint.operator std::string() +
                     "': " + (reason.empty() ? std::string("loop-carried dependence") : reason);
  if (std::find(notes->begin(), notes->end(), note) == notes->end()) {
    notes->push_back(note);
  }
}

bool IsCumulativeWrite(const tirx::BufferStoreNode* store) {
  bool found = false;
  tirx::PostOrderVisit(store->value, [&](const ffi::ObjectRef& node) {
    const auto* load = node.as<tirx::BufferLoadNode>();
    if (load != nullptr && SameBuffer(load->buffer, store->buffer) &&
        SameIndices(load->indices, store->indices)) {
      found = true;
    }
  });
  return found;
}

bool ValueVariesAlong(const PrimExpr& expr, const Axis& axis,
                      arith::Analyzer* analyzer);

/*!
 * \brief Whether a directly-nested ``T.serial`` loop can be absorbed (方案 §6).
 *
 * Absorbing makes the loop iteration a candidate vector axis, so loop-carried
 * dependences must be excluded. The criterion is conservative on purpose:
 * reject rather than miscompile.
 */
std::pair<bool, std::string> SerialIsClean(const tirx::For& loop,
                                           arith::Analyzer* analyzer) {
  tirx::Var var = loop->loop_var;
  std::vector<const tirx::BufferStoreNode*> stores;
  std::vector<const tirx::BufferLoadNode*> loads;
  tirx::PostOrderVisit(loop->body, [&](const ffi::ObjectRef& node) {
    if (const auto* store = node.as<tirx::BufferStoreNode>()) {
      stores.push_back(store);
    } else if (const auto* load = node.as<tirx::BufferLoadNode>()) {
      loads.push_back(load);
    }
  });

  Axis axis = MakeAxis(var, loop->extent, 0, ForKind::kSerial);
  std::vector<std::pair<const tirx::BufferStoreNode*, PrimExpr>> offsets;
  for (const tirx::BufferStoreNode* store : stores) {
    offsets.emplace_back(store,
                         LinearElementOffset(store->buffer, store->indices, analyzer));
  }
  for (const auto& entry : offsets) {
    std::optional<int64_t> delta =
        ConstInt(FiniteDifference(entry.second, var, analyzer), analyzer);
    if (delta.has_value() && *delta == 0) {
      if (!IsCumulativeWrite(entry.first)) {
        return {false, "write to " + entry.first->buffer->name.operator std::string() +
                           " is invariant along '" + var->name_hint.operator std::string() +
                           "' but not a reduction"};
      }
      if (!ValueVariesAlong(entry.first->value, axis, analyzer)) {
        return {false, "reduction into " +
                           entry.first->buffer->name.operator std::string() +
                           " does not vary along '" + var->name_hint.operator std::string() +
                           "'"};
      }
    }
  }

  for (size_t i = 0; i < offsets.size(); ++i) {
    for (size_t j = i + 1; j < offsets.size(); ++j) {
      if (SameBuffer(offsets[i].first->buffer, offsets[j].first->buffer)) {
        tirx::ExprDeepEqual equal;
        if (!equal(offsets[i].second, offsets[j].second)) {
          return {false,
                  "multiple writes to " +
                      offsets[i].first->buffer->name.operator std::string() +
                      " at different addresses along '" +
                      var->name_hint.operator std::string() +
                      "' may alias across iterations (last-write-wins is not preserved by vectorization, R6)"};
        }
      }
    }
  }

  for (const tirx::BufferLoadNode* load : loads) {
    bool touches_written = false;
    bool same_address = false;
    for (const auto& entry : offsets) {
      if (!SameBuffer(load->buffer, entry.first->buffer)) continue;
      touches_written = true;
      if (SameIndices(load->indices, entry.first->indices)) same_address = true;
    }
    if (touches_written && !same_address) {
      return {false, "cross-iteration read of " +
                         load->buffer->name.operator std::string() + " along '" +
                         var->name_hint.operator std::string() + "'"};
    }
  }
  return {true, ""};
}

/*! \brief Run/body skeleton of a nest: axes + (pre, main, post) statements. */
struct NestSplit {
  std::vector<Axis> axes;
  std::optional<Axis> serial_axis;
  std::vector<Stmt> pre;
  Stmt main;
  std::vector<Stmt> post;
};

NestSplit SplitNest(const Stmt& stmt, arith::Analyzer* analyzer,
                    std::vector<std::string>* notes) {
  NestSplit split;
  auto absorbed = AbsorbParallelChain(stmt, analyzer);
  split.axes = std::move(absorbed.first);
  Stmt current = absorbed.second;
  std::optional<Axis> serial_axis;

  while (const auto* loop_ptr = current.as<tirx::ForNode>()) {
    tirx::For loop = ffi::GetRef<tirx::For>(loop_ptr);
    ForKind kind = loop->kind;
    RequireZeroMin(loop, analyzer);
    if (kind == ForKind::kSerial) {
      auto clean = SerialIsClean(loop, analyzer);
      if (!clean.first) {
        RecordSerialReason(notes, loop, clean.second);
        break;
      }
    } else if (kind != ForKind::kParallel) {
      break;
    }
    Axis axis = MakeAxis(loop->loop_var, loop->extent,
                         static_cast<int64_t>(split.axes.size()), kind);
    split.axes.push_back(axis);
    if (kind == ForKind::kSerial) serial_axis = axis;
    current = loop->body;
  }

  if (const auto* seq = current.as<tirx::SeqStmtNode>()) {
    std::vector<size_t> loop_positions;
    for (size_t i = 0; i < seq->seq.size(); ++i) {
      if (seq->seq[i].as<tirx::ForNode>() != nullptr) loop_positions.push_back(i);
    }
    // Only an exactly-one clean serial loop splits the body into pre/serial/post;
    // anything else keeps the existing "nested serial loop" rejection.
    if (loop_positions.size() == 1) {
      size_t pos = loop_positions[0];
      tirx::For loop = Downcast<tirx::For>(seq->seq[pos]);
      if (loop->kind == ForKind::kSerial) {
        auto clean = SerialIsClean(loop, analyzer);
        if (clean.first) {
          serial_axis = MakeAxis(loop->loop_var, loop->extent,
                                 static_cast<int64_t>(split.axes.size()), ForKind::kSerial);
          split.axes.push_back(*serial_axis);
          for (size_t i = 0; i < pos; ++i) split.pre.push_back(seq->seq[i]);
          for (size_t i = pos + 1; i < seq->seq.size(); ++i) split.post.push_back(seq->seq[i]);
          current = loop->body;
        } else {
          RecordSerialReason(notes, loop, clean.second);
        }
      }
    }
  }

  bool has_parallel = false;
  for (const Axis& axis : split.axes) has_parallel = has_parallel || axis->is_parallel;
  if (!has_parallel) throw Unsupported("no T.Parallel axis found");

  split.serial_axis = serial_axis;
  split.main = current;
  return split;
}

/*! \brief Absorb a pure serial chain into axes; nullopt keeps existing behavior. */
std::optional<std::pair<std::vector<Axis>, Stmt>> AbsorbSerialChain(
    const Stmt& stmt, arith::Analyzer* analyzer, int64_t base_index) {
  const auto* loop_ptr = stmt.as<tirx::ForNode>();
  if (loop_ptr == nullptr || loop_ptr->kind != ForKind::kSerial) return std::nullopt;
  std::vector<Axis> axes;
  Stmt current = stmt;
  while (const auto* current_loop = current.as<tirx::ForNode>()) {
    if (current_loop->kind != ForKind::kSerial) break;
    std::optional<int64_t> min = ConstInt(current_loop->min, analyzer);
    if (!min.has_value() || *min != 0) return std::nullopt;
    if (!SerialIsClean(ffi::GetRef<tirx::For>(current_loop), analyzer).first) {
      return std::nullopt;
    }
    axes.push_back(MakeAxis(current_loop->loop_var, current_loop->extent,
                            base_index + static_cast<int64_t>(axes.size()),
                            ForKind::kSerial));
    current = current_loop->body;
  }
  if (current.as<tirx::ForNode>() != nullptr) return std::nullopt;
  return std::make_pair(std::move(axes), current);
}

/*!
 * \brief Absorb a consecutive loop chain of parallel and clean serial loops.
 *
 * Used by the expanded tile-op multi-segment path: a body statement may be a
 * parallel copy/reduce nest whose reduced dimension is serial. A parallel axis
 * needs no dependence proof; a serial axis keeps the :func:`SerialIsClean`
 * requirement. Returns ``nullopt`` when the chain cannot be absorbed.
 */
std::optional<std::pair<std::vector<Axis>, Stmt>> AbsorbLoopChain(
    const Stmt& stmt, arith::Analyzer* analyzer, int64_t base_index) {
  const auto* loop_ptr = stmt.as<tirx::ForNode>();
  if (loop_ptr == nullptr) return std::nullopt;
  std::vector<Axis> axes;
  Stmt current = stmt;
  while (const auto* current_loop = current.as<tirx::ForNode>()) {
    const ForKind kind = current_loop->kind;
    if (kind != ForKind::kParallel && kind != ForKind::kSerial) return std::nullopt;
    RequireZeroMin(ffi::GetRef<tirx::For>(current_loop), analyzer);
    if (kind == ForKind::kSerial &&
        !SerialIsClean(ffi::GetRef<tirx::For>(current_loop), analyzer).first) {
      return std::nullopt;
    }
    axes.push_back(MakeAxis(current_loop->loop_var, current_loop->extent,
                            base_index + static_cast<int64_t>(axes.size()), kind));
    current = current_loop->body;
  }
  if (axes.empty()) return std::nullopt;
  return std::make_pair(std::move(axes), current);
}

/* ------------------------------------------------------------------------- */
/* scoring / diagnostics                                                     */
/* ------------------------------------------------------------------------- */

int64_t ActiveLanes(const Axis& axis, arith::Analyzer* analyzer, int64_t lanes) {
  std::optional<int64_t> extent = ConstInt(axis->extent, analyzer);
  if (!extent.has_value()) return lanes;
  return std::max<int64_t>(std::min<int64_t>(*extent, lanes), 1);
}

int64_t AxisWeightSum(const std::vector<AccessInfo>& infos, const Axis& axis) {
  int64_t total = 0;
  for (const AccessInfo& info : infos) total += AxisWeight(AxisClassOf(info, axis).kind);
  return total;
}

int64_t AxisScore(const std::vector<AccessInfo>& store_infos,
                  const std::vector<AccessInfo>& load_infos, const Axis& axis,
                  arith::Analyzer* analyzer, int64_t lanes) {
  int64_t weight = 100 * AxisWeightSum(store_infos, axis) + 10 * AxisWeightSum(load_infos, axis);
  return weight * ActiveLanes(axis, analyzer, lanes) / lanes + axis->index;
}

std::vector<Axis> RankVectorAxes(const std::vector<Axis>& axes,
                                 const std::vector<AccessInfo>& store_infos,
                                 const std::vector<AccessInfo>& load_infos,
                                 arith::Analyzer* analyzer, int64_t lanes) {
  std::vector<std::pair<int64_t, Axis>> ranked;
  for (const Axis& axis : axes) {
    bool rejected = false;
    for (const AccessInfo& info : store_infos) {
      if (AxisClassOf(info, axis).kind == AccessClass::kTrueRandom) {
        rejected = true;
        break;
      }
    }
    if (rejected) continue;
    ranked.emplace_back(AxisScore(store_infos, load_infos, axis, analyzer, lanes), axis);
  }
  // Python's ``list.sort(reverse=True)`` is stable: equal scores keep nest order.
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const std::pair<int64_t, Axis>& lhs,
                      const std::pair<int64_t, Axis>& rhs) { return lhs.first > rhs.first; });
  std::vector<Axis> out;
  out.reserve(ranked.size());
  for (const auto& entry : ranked) out.push_back(entry.second);
  return out;
}

std::vector<std::string> AxisDiagnostics(const std::vector<Axis>& axes,
                                         const std::vector<AccessInfo>& store_infos,
                                         const std::vector<AccessInfo>& load_infos,
                                         arith::Analyzer* analyzer, int64_t lanes) {
  std::vector<std::string> lines;
  for (const Axis& axis : axes) {
    std::string label = "axis '" + axis->name.operator std::string() + "'[" +
                        std::to_string(axis->index) + "]" +
                        (axis->is_parallel ? "" : "(serial)");
    std::vector<std::string> store_classes;
    bool rejected = false;
    for (const AccessInfo& info : store_infos) {
      AccessClass kind = AxisClassOf(info, axis).kind;
      store_classes.push_back(AccessClassName(kind));
      if (kind == AccessClass::kTrueRandom) rejected = true;
    }
    if (rejected) {
      std::set<std::string> reasons;
      for (const AccessInfo& info : store_infos) {
        if (!info->reason.empty()) reasons.insert(info->reason.operator std::string());
      }
      std::string detail;
      if (!reasons.empty()) {
        detail = " (";
        bool first = true;
        for (const std::string& reason : reasons) {
          detail += (first ? "" : "; ") + reason;
          first = false;
        }
        detail += ")";
      }
      lines.push_back(label + ": store=" + PyStrList(store_classes) +
                      " -> rejected (true-random store)" + detail);
      continue;
    }
    std::vector<std::string> load_classes;
    for (const AccessInfo& info : load_infos) {
      load_classes.push_back(AccessClassName(AxisClassOf(info, axis).kind));
    }
    int64_t active = ActiveLanes(axis, analyzer, lanes);
    lines.push_back(label + ": store=" + PyStrList(store_classes) + " load=" +
                    PyStrList(load_classes) + " -> candidate (score=" +
                    std::to_string(AxisScore(store_infos, load_infos, axis, analyzer, lanes)) +
                    ", active=" + std::to_string(active) + "/" + std::to_string(lanes) + ")");
  }
  return lines;
}

/* ------------------------------------------------------------------------- */
/* body normalization helpers                                                */
/* ------------------------------------------------------------------------- */

std::vector<PrimExpr> CollectLoadExprs(const std::vector<StoreSpec>& specs) {
  std::vector<PrimExpr> found;
  for (const StoreSpec& spec : specs) {
    tirx::PostOrderVisit(spec->value, [&](const ffi::ObjectRef& node) {
      if (node.as<tirx::BufferLoadNode>() != nullptr) found.push_back(Downcast<PrimExpr>(node));
    });
    if (spec->mask.has_value()) {
      tirx::PostOrderVisit(*spec->mask, [&](const ffi::ObjectRef& node) {
        if (node.as<tirx::BufferLoadNode>() != nullptr) found.push_back(Downcast<PrimExpr>(node));
      });
    }
  }
  return found;
}

std::vector<AccessInfo> LoadInfos(const std::vector<Axis>& axes,
                                  const std::vector<StoreSpec>& specs,
                                  arith::Analyzer* analyzer) {
  std::vector<AccessInfo> infos;
  for (const PrimExpr& expr : CollectLoadExprs(specs)) {
    const auto* load = expr.as<tirx::BufferLoadNode>();
    infos.push_back(ClassifyAccess(load->buffer, load->indices, axes, analyzer,
                                   /*is_store=*/false));
  }
  return infos;
}

bool ValueVariesAlong(const PrimExpr& expr, const Axis& axis,
                      arith::Analyzer* analyzer) {
  bool varies = false;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef& node) {
    const auto* load = node.as<tirx::BufferLoadNode>();
    if (load == nullptr) return;
    AccessInfo info = ClassifyAccess(load->buffer, load->indices, {axis}, analyzer,
                                     /*is_store=*/false);
    if (AxisClassOf(info, axis).kind != AccessClass::kUniform) varies = true;
  });
  if (varies) return true;
  for (const tirx::Var& var : FreeVars(expr)) {
    if (axis->var.same_as(var)) return true;
  }
  return false;
}

bool IsReduceAxis(const Axis& axis, const std::vector<StoreSpec>& specs,
                  const std::vector<AccessInfo>& store_infos,
                  arith::Analyzer* analyzer) {
  for (const AccessInfo& info : store_infos) {
    if (AxisClassOf(info, axis).kind != AccessClass::kUniform) return false;
  }
  for (const StoreSpec& spec : specs) {
    if (ValueVariesAlong(spec->value, axis, analyzer)) return true;
  }
  return false;
}

std::string ReduceOpOf(const std::vector<StoreSpec>& specs) {
  for (const StoreSpec& spec : specs) {
    std::string op;
    if (ReduceOpFor(TypeKeyTail(spec->value), &op)) return op;
  }
  std::set<std::string> names;
  for (const StoreSpec& spec : specs) names.insert(TypeKeyTail(spec->value));
  std::string joined;
  if (names.empty()) {
    joined = "<empty>";
  } else {
    bool first = true;
    for (const std::string& name : names) {
      joined += (first ? "" : ", ") + name;
      first = false;
    }
  }
  throw Unsupported("unsupported reduction operator(s) " + joined +
                    " (方案 §2 Case II): only Add/Max/Min reductions are supported");
}

bool ExprReadsBuffers(const PrimExpr& expr, const std::vector<tirx::Buffer>& buffers) {
  bool found = false;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef& node) {
    const auto* load = node.as<tirx::BufferLoadNode>();
    if (load == nullptr) return;
    for (const tirx::Buffer& buffer : buffers) {
      if (SameBuffer(load->buffer, buffer)) found = true;
    }
  });
  return found;
}

/* ------------------------------------------------------------------------- */
/* flatten channel                                                           */
/* ------------------------------------------------------------------------- */

struct FlattenResult {
  Axis flat_axis;
  std::map<const tirx::VarNode*, PrimExpr> mapping;
  int64_t total{0};
};

std::optional<FlattenResult> FlattenAxes(const std::vector<Axis>& axes,
                                         arith::Analyzer* analyzer) {
  if (axes.size() < 2) return std::nullopt;
  // An extent-1 axis carries no lane information: mapping it to 0 keeps the
  // synthesized axis free of ``range(1)`` padding, so e.g. a sliced tile-op
  // dimension (``b[i, :]`` keeps a leading 1) does not block flattening.
  std::vector<Axis> chain;
  std::vector<int64_t> extents;
  std::map<const tirx::VarNode*, PrimExpr> trivial;
  for (const Axis& axis : axes) {
    std::optional<int64_t> extent = ConstInt(axis->extent, analyzer);
    if (!extent.has_value() || *extent < 1) return std::nullopt;
    if (*extent == 1) {
      trivial[axis->var.get()] = IntImm32(0);
      continue;
    }
    if ((*extent & (*extent - 1)) != 0) return std::nullopt;
    chain.push_back(axis);
    extents.push_back(*extent);
  }
  if (chain.size() < 2) return std::nullopt;
  int64_t total = 1;
  for (int64_t extent : extents) total *= extent;

  std::string name = "k_";
  for (size_t i = 0; i < chain.size(); ++i) {
    name += (i == 0 ? "" : "_") + chain[i]->name.operator std::string();
  }
  tirx::Var var(name, DataType::Int(32));

  FlattenResult result;
  result.flat_axis = MakeAxis(var, IntImm32(total), 0, ForKind::kParallel);
  result.total = total;
  result.mapping = std::move(trivial);
  int64_t suffix = 1;
  int64_t count = static_cast<int64_t>(chain.size());
  for (int64_t position = 0; position < count; ++position) {
    // Right-to-left decomposition; the outermost axis is NOT taken modulo, since
    // ``(k // suffix) % E`` would collapse to ``k % (prod E)`` and lose Delta==1.
    const Axis& axis = chain[static_cast<size_t>(count - 1 - position)];
    int64_t extent = extents[static_cast<size_t>(count - 1 - position)];
    PrimExpr mapping;
    if (position == 0) {
      mapping = floormod(var, IntImm32(extent));
    } else if (position == count - 1) {
      mapping = floordiv(var, IntImm32(suffix));
    } else {
      mapping = floormod(floordiv(var, IntImm32(suffix)), IntImm32(extent));
    }
    result.mapping[axis->var.get()] = mapping;
    suffix *= extent;
  }
  return result;
}

SegmentSpecs PrepareSegment(const std::vector<Axis>& axes,
                            std::optional<Axis> serial_axis,
                            const std::vector<Stmt>& pre_stmts, const Stmt& main_stmt,
                            const std::vector<Stmt>& post_stmts, const AliasMap& aliases,
                            arith::Analyzer* analyzer, std::optional<int64_t> lanes);

std::pair<std::optional<SegmentSpecs>, std::string> TryFlatten(
    const std::vector<Axis>& axes, const std::vector<Stmt>& pre_stmts, const Stmt& main_stmt,
    const std::vector<Stmt>& post_stmts, int64_t lanes, arith::Analyzer* analyzer) {
  if (!pre_stmts.empty() || !post_stmts.empty()) return {std::nullopt, ""};
  if (axes.size() < 2) return {std::nullopt, ""};
  // The innermost *real* axis decides whether flattening is needed; a trailing
  // extent-1 axis carries no lanes.
  std::optional<int64_t> innermost;
  int64_t real_axes = 0;
  for (auto it = axes.rbegin(); it != axes.rend(); ++it) {
    std::optional<int64_t> extent = ConstInt((*it)->extent, analyzer);
    if (!extent.has_value()) return {std::nullopt, ""};
    if (*extent == 1) continue;
    if (!innermost.has_value()) innermost = extent;
    ++real_axes;
  }
  if (real_axes < 2) return {std::nullopt, ""};
  if (!innermost.has_value() || *innermost >= lanes) return {std::nullopt, ""};
  std::optional<FlattenResult> flat = FlattenAxes(axes, analyzer);
  if (!flat.has_value()) {
    return {std::nullopt, "flatten rejected: extent product of [" + JoinNames(axes) +
                              "] is not a power-of-two chain"};
  }
  Axis flat_axis = flat->flat_axis;
  int64_t total = flat->total;
  Stmt body = SubstituteVarsStmt(main_stmt, flat->mapping);
  SegmentSpecs specs;
  try {
    specs = PrepareSegment({flat_axis}, std::nullopt, {}, body, {},
                           CollectAliases(body), analyzer, lanes);
  } catch (const Unsupported& error) {
    return {std::nullopt, "flatten rejected: " + error.message()};
  }
  for (const AccessInfo& info : specs.main_infos) {
    if (AxisClassOf(info, flat_axis).kind != AccessClass::kContiguous) {
      return {std::nullopt,
              "flatten rejected: not every store is contiguous along the synthesized axis '" +
                  flat_axis->name.operator std::string() + "' (extent " +
                  std::to_string(total) + ")"};
    }
  }
  // Read-side gate: flattening only removes contiguity, never a gather.
  std::vector<AccessInfo> load_infos = LoadInfos({flat_axis}, specs.AllSpecs(), analyzer);
  for (const AccessInfo& info : load_infos) {
    AccessClass kind = AxisClassOf(info, flat_axis).kind;
    if (kind != AccessClass::kContiguous && kind != AccessClass::kUniform) {
      return {std::nullopt,
              "flatten rejected: a load along the synthesized axis '" +
                  flat_axis->name.operator std::string() +
                  "' still needs an index vector (gather) — flattening would not remove it"};
    }
  }
  std::vector<tirx::Buffer> targets;
  for (const StoreSpec& spec : specs.main_specs) targets.push_back(spec->buffer);
  for (size_t i = 0; i < targets.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (SameBuffer(targets[i], targets[j])) {
        return {std::nullopt,
                "flatten rejected: two statements write the same buffer '" +
                    targets[i]->name.operator std::string() +
                    "' along the synthesized axis '" +
                    flat_axis->name.operator std::string() +
                    "' (their addresses may overlap across k) — extent " +
                    std::to_string(total)};
      }
    }
  }
  for (const StoreSpec& spec : specs.main_specs) {
    const std::vector<PrimExpr> exprs = {spec->value,
                                         spec->mask.has_value() ? *spec->mask : PrimExpr()};
    for (const PrimExpr& expr : exprs) {
      if (expr.defined() && ExprReadsBuffers(expr, targets)) {
        return {std::nullopt,
                "flatten rejected: a written buffer is also read inside the nest (no write-conflict-free flattening) — extent " +
                    std::to_string(total)};
      }
    }
  }
  return {specs, ""};
}

/* ------------------------------------------------------------------------- */
/* segment preparation / reduction helpers                                    */
/* ------------------------------------------------------------------------- */

void RequireSupportedElement(const tirx::Buffer& buffer) {
  std::string elem = ElemDtype(buffer->dtype);
  try {
    MaskBits(elem);
  } catch (const Unsupported& error) {
    // Python converts the ValueError into Unsupported at build time (D11).
    throw Unsupported("store element dtype '" + elem + "' of buffer '" +
                      buffer->name.operator std::string() + "' is not supported: " +
                      error.message());
  }
}

std::string PyBufferDtypeName(const tirx::Buffer& buffer) {
  return ffi::DLDataTypeToString(buffer->dtype).operator std::string();
}

/*! \brief Natural vector-register width of a store's element type (2048/bits). */
int64_t StoreRegisterLanes(const StoreSpec& spec) {
  return VRegLanes(ElemDtype(PyBufferDtypeName(spec->buffer)));
}

/*! \brief Element bit width shared by every load inside ``expr`` (0 = unknown/mixed). */
int CommonLoadBits(const PrimExpr& expr) {
  int bits = 0;
  bool mixed = false;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef& node) {
    const auto* load = node.as<tirx::BufferLoadNode>();
    if (load == nullptr) return;
    int width = load->buffer->dtype.bits();
    if (bits == 0) {
      bits = width;
    } else if (bits != width) {
      mixed = true;
    }
  });
  return mixed ? 0 : bits;
}

/*!
 * \brief Register lanes of a genuine narrowing conversion, if any.
 *
 * A cast only narrows the *register* when the converted expression is itself a
 * full register of its own element type: every load inside it must share that
 * element width (``cast(bf16, f32_load)``). A cast that merely closes a
 * widening round trip (``f16 load -> f32 arithmetic -> f16 store``) keeps the
 * store's own register and must not shrink the chunk.
 */
std::optional<int64_t> NarrowingSourceLanes(const PrimExpr& value,
                                            const tirx::Buffer& buffer) {
  const auto* cast = value.as<tirx::CastNode>();
  if (cast == nullptr) return std::nullopt;
  if (cast->value.dtype().bits() <= buffer->dtype.bits()) return std::nullopt;
  if (CommonLoadBits(cast->value) != cast->value.dtype().bits()) return std::nullopt;
  return VRegLanes(ElemDtype(ffi::DLDataTypeToString(cast->value.dtype())));
}

/*!
 * \brief Lanes of the vector a store hands to its write port.
 *
 * A narrowing conversion fills only ``VRegLanes(src)`` lanes of the wider
 * destination register; every other recipe produces a full destination
 * register.
 */
int64_t ProducedLanes(const PrimExpr& value, const tirx::Buffer& buffer) {
  std::optional<int64_t> narrowed = NarrowingSourceLanes(value, buffer);
  if (narrowed.has_value()) return *narrowed;
  return VRegLanes(ElemDtype(PyBufferDtypeName(buffer)));
}

/*!
 * \brief Pick the chunk width of one segment (mixed element widths, Stage 1).
 *
 * The chunk covers ``lanes`` logical elements; a store whose element type has
 * ``2048 / bits`` register lanes writes exactly ``lanes`` of them, so every
 * store width must be ``lanes``, ``2 * lanes`` or ``4 * lanes``. Stage 1 keeps
 * the model deliberately narrow:
 *
 *  - ``lanes`` is the smallest width the stores actually produce, so one chunk
 *    is never wider than the data a store hands to its write port;
 *  - a store wider than the chunk must be a narrowing conversion whose source
 *    register is exactly ``lanes`` wide (it is emitted in the packed form);
 *  - mixed widths with sibling (``pre``/``post``) statements or a statement
 *    predicate are rejected here; the extent and access-shape checks live in
 *    ``BuildPlanForAxis`` once the vector axis is known.
 */
int64_t SelectSegmentLanes(const SegmentSpecs& specs,
                           std::optional<int64_t> override_lanes) {
  int64_t min_effective = 0;
  for (const StoreSpec& spec : specs.main_specs) {
    int64_t effective =
        std::min(StoreRegisterLanes(spec), ProducedLanes(spec->value, spec->buffer));
    min_effective = min_effective == 0 ? effective : std::min(min_effective, effective);
  }
  if (min_effective == 0) return override_lanes.value_or(0);
  int64_t lanes = override_lanes.value_or(min_effective);
  if (lanes <= 0) return min_effective;

  bool mixed = false;
  for (const StoreSpec& spec : specs.main_specs) {
    int64_t register_lanes = StoreRegisterLanes(spec);
    std::string name = spec->buffer->name.operator std::string();
    if (register_lanes % lanes != 0 || register_lanes / lanes > 4) {
      throw Unsupported("store '" + name + "' has " + std::to_string(register_lanes) +
                        " register lanes, which the segment chunk width " +
                        std::to_string(lanes) + " cannot cover (split the nest)");
    }
    if (register_lanes / lanes > 1) mixed = true;
  }
  if (!mixed) return lanes;

  if (!specs.pre_specs.empty() || !specs.post_specs.empty()) {
    throw Unsupported(
        "mixed element widths in one segment are only supported without sibling statements "
        "around the loop; split the nest or stage an explicit cast");
  }
  for (const StoreSpec& spec : specs.main_specs) {
    int64_t register_lanes = StoreRegisterLanes(spec);
    int64_t ratio = register_lanes / lanes;
    std::string name = spec->buffer->name.operator std::string();
    if (spec->mask.has_value()) {
      throw Unsupported(
          "mixed element widths in one segment are not supported with a statement predicate on "
          "store '" +
          name + "'",
          {}, "body.statement_predicate");
    }
    const auto* cast = spec->value.as<tirx::CastNode>();
    if (ratio == 1) {
      if (cast != nullptr && cast->value.dtype().bits() < spec->buffer->dtype.bits()) {
        throw Unsupported("store '" + name +
                          "' widens its value inside a mixed-width segment; widening stores are "
                          "not supported yet");
      }
      continue;
    }
    std::optional<int64_t> narrowed = NarrowingSourceLanes(spec->value, spec->buffer);
    if (!narrowed.has_value()) {
      throw Unsupported("store '" + name + "' needs " + std::to_string(ratio) +
                        "x the segment chunk width " + std::to_string(lanes) +
                        " but its value is not a narrowing conversion; split the nest or stage "
                        "an explicit cast");
    }
    if (*narrowed != lanes) {
      throw Unsupported("store '" + name + "' narrows from a " + std::to_string(*narrowed) +
                        "-lane register but the segment chunk width is " +
                        std::to_string(lanes) + " (split the nest)");
    }
  }
  return lanes;
}

SegmentSpecs PrepareSegment(const std::vector<Axis>& axes,
                            std::optional<Axis> serial_axis,
                            const std::vector<Stmt>& pre_stmts, const Stmt& main_stmt,
                            const std::vector<Stmt>& post_stmts, const AliasMap& aliases,
                            arith::Analyzer* analyzer, std::optional<int64_t> lanes) {
  SegmentSpecs specs;
  specs.axes = axes;
  specs.serial_axis = serial_axis;
  if (!pre_stmts.empty()) specs.pre_specs = NormalizeBody(SeqStmts(pre_stmts), aliases);
  specs.main_specs = NormalizeBody(main_stmt, aliases);
  if (!post_stmts.empty()) specs.post_specs = NormalizeBody(SeqStmts(post_stmts), aliases);
  if (specs.main_specs.empty()) {
    throw Unsupported("T.Parallel body has no store to vectorize");
  }
  std::vector<StoreSpec> all = specs.AllSpecs();
  for (const StoreSpec& spec : all) RequireSupportedElement(spec->buffer);
  specs.lanes = SelectSegmentLanes(specs, lanes);

  auto classify = [&](const std::vector<StoreSpec>& group) {
    std::vector<AccessInfo> infos;
    for (const StoreSpec& spec : group) {
      infos.push_back(ClassifyAccess(spec->buffer, spec->indices, axes, analyzer,
                                     /*is_store=*/true));
    }
    return infos;
  };
  specs.pre_infos = classify(specs.pre_specs);
  specs.main_infos = classify(specs.main_specs);
  specs.post_infos = classify(specs.post_specs);
  return specs;
}

bool IsSameConst(const PrimExpr& expr, const PrimExpr& value) {
  const auto* lhs_int = expr.as<IntImmNode>();
  const auto* rhs_int = value.as<IntImmNode>();
  if (lhs_int != nullptr && rhs_int != nullptr) {
    return static_cast<double>(lhs_int->value) == static_cast<double>(rhs_int->value);
  }
  const auto* lhs_float = expr.as<FloatImmNode>();
  const auto* rhs_float = value.as<FloatImmNode>();
  if (lhs_float != nullptr && rhs_float != nullptr) {
    return lhs_float->value == rhs_float->value;
  }
  return false;
}

void FoldIdentityInit(VectorPlan& plan) {
  if (plan->pre_stores.empty()) return;
  std::vector<StorePlan> kept;
  for (const StorePlan& pre : plan->pre_stores) {
    bool folded = false;
    if (pre->kind == AccessClass::kUniform &&
        !pre->mask.has_value()) {
      for (size_t store_index = 0; store_index < plan->stores.size(); ++store_index) {
        const StorePlan& store = plan->stores[store_index];
        ReduceInfo reduce = ReduceOf(store);
        if (!reduce.defined() || !reduce->accumulate) continue;
        if (!(SameBuffer(pre->spec->buffer, store->spec->buffer) &&
              SameIndices(pre->spec->indices, store->spec->indices))) {
          continue;
        }
        std::string elem = ElemDtype(PyBufferDtypeName(store->spec->buffer));
        PrimExpr identity = ReduceIdentity(reduce->recipe.operator std::string(), elem);
        if (IsSameConst(pre->spec->value, identity)) {
          // Defect L1: `reduce` is a by-value ObjectRef; mutating its COW copy
          // alone loses the change. Write the updated reduce back to the store
          // (the store still has exclusive ownership here, so COW is a no-op
          // copy of the node we are about to replace).
          ReduceInfo updated_reduce = reduce;
          updated_reduce.CopyOnWrite()->accumulate = false;
          StorePlan updated_store = store;
          updated_store.CopyOnWrite()->reduce =
              ffi::Optional<ffi::ObjectRef>(updated_reduce);
          plan.CopyOnWrite()->stores.Set(static_cast<int64_t>(store_index),
                                         updated_store);
          folded = true;
          break;
        }
      }
    }
    if (!folded) kept.push_back(pre);
  }
  plan.CopyOnWrite()->pre_stores = ffi::Array<StorePlan>(kept);
}

void DropDeadPreStores(VectorPlan& plan, arith::Analyzer* analyzer) {
  if (plan->pre_stores.empty()) return;
  std::vector<tirx::Var> serial_vars;
  for (const Axis& axis : plan->axes) {
    if (!axis->is_parallel) serial_vars.push_back(axis->var);
  }
  if (serial_vars.empty()) return;
  // A zero-trip serial loop never runs the body: pre writes stay live.
  for (const Axis& axis : plan->axes) {
    if (axis->is_parallel) continue;
    std::optional<int64_t> extent = ConstInt(axis->extent, analyzer);
    if (extent.has_value() && *extent == 0) return;
  }
  std::vector<StorePlan> kept;
  for (const StorePlan& pre : plan->pre_stores) {
    bool dead = false;
    for (const StorePlan& store : plan->stores) {
      if (store->mask.has_value() || !SameBuffer(pre->spec->buffer, store->spec->buffer)) {
        continue;
      }
      // A read-modify-write depends on the value the pre write left behind.
      if (ReadsBuffer(store->value, store->spec->buffer)) continue;
      PrimExpr offset = store->access->offset;
      std::map<const tirx::VarNode*, PrimExpr> mapping;
      for (const tirx::Var& var : serial_vars) mapping[var.get()] = IntImm32(0);
      offset = SubstituteVars(offset, mapping);
      PrimExpr diff = analyzer->Simplify(offset - pre->access->offset);
      std::optional<int64_t> constant = ConstInt(diff, analyzer);
      if (constant.has_value() && *constant == 0) {
        dead = true;
        break;
      }
    }
    if (!dead) kept.push_back(pre);
  }
  plan.CopyOnWrite()->pre_stores = ffi::Array<StorePlan>(kept);
}

/*!
 * \brief Whether ``pre`` can be folded into ``store``'s accumulator initial value.
 *
 * Folding drops the pre store's memory write, so the value must be exactly the
 * accumulator's initial value (uniform, unpredicated, same address) and no
 * sibling pre store may read the buffer afterwards (it would see a stale value).
 */
bool PreFoldableInto(const StorePlan& pre, const StorePlan& store,
                     const std::vector<StorePlan>& pre_stores) {
  if (pre->mask.has_value() || !pre->value->is_uniform) return false;
  if (!SameBuffer(pre->spec->buffer, store->spec->buffer)) return false;
  if (!SameIndices(pre->spec->indices, store->spec->indices)) return false;
  for (const StorePlan& other : pre_stores) {
    if (other.same_as(pre)) continue;
    if (ReadsBuffer(other->value, pre->spec->buffer)) return false;
  }
  return true;
}

/*!
 * \brief Record the two non-default store handlings in the plan.
 *
 * Both rules used to be re-derived by the emitter, and the identity-init rule
 * was derived twice with different strength (the hoisting pass used the weak
 * form, the emission block the strong one). Marking them here makes the plan
 * authoritative and keeps both emitter paths consistent.
 */
void MarkStoreRoles(VectorPlan& plan) {
  // PRE_FOLDED: the pre store writes the accumulator's initial value.
  std::set<const StorePlanNode*> consumed;
  for (int64_t index = 0; index < static_cast<int64_t>(plan->stores.size()); ++index) {
    const StorePlan& store = plan->stores[index];
    if (!store->promote.has_value()) continue;
    for (int64_t pre_index = 0;
         pre_index < static_cast<int64_t>(plan->pre_stores.size()); ++pre_index) {
      const StorePlan& pre = plan->pre_stores[pre_index];
      if (consumed.count(pre.get()) != 0) continue;
      if (!PreFoldableInto(pre, store, std::vector<StorePlan>(
                                          plan->pre_stores.begin(), plan->pre_stores.end()))) {
        continue;
      }
      consumed.insert(pre.get());
      StorePlan updated = pre;
      updated.CopyOnWrite()->role = ffi::String(StoreRoleName(StoreRole::kPreFolded));
      plan.CopyOnWrite()->pre_stores.Set(pre_index, updated);
      break;
    }
  }

  // POST_SUPERSEDED: a post store rewrites the same address and consumes the
  // accumulator value, so the write-back belongs to the post (register
  // forwarding, one ``vsts`` instead of two).
  for (int64_t index = 0; index < static_cast<int64_t>(plan->stores.size()); ++index) {
    const StorePlan& store = plan->stores[index];
    if (!store->promote.has_value()) continue;
    bool superseded = false;
    for (const StorePlan& post : plan->post_stores) {
      if (post->mask.has_value()) continue;
      if (!SameBuffer(post->spec->buffer, store->spec->buffer)) continue;
      if (!SameIndices(post->spec->indices, store->spec->indices)) continue;
      if (!SameAddressLoad(post->value, post->spec->buffer, post->spec->indices)
               .defined()) {
        continue;
      }
      superseded = true;
      break;
    }
    if (!superseded) continue;
    StorePlan updated = store;
    updated.CopyOnWrite()->role = ffi::String(StoreRoleName(StoreRole::kPostSuperseded));
    plan.CopyOnWrite()->stores.Set(index, updated);
  }
}

/*!
 * \brief Shrink the segment to the chunks whose stores can actually fire.
 *
 * ``GuardFreeBounds`` already proves "this predicate holds over a chunk range"
 * for the mask-free fast path. Asking it about ``not(mask)`` yields the
 * complementary fact: the leading / trailing chunks where every store's
 * predicate is *false*, i.e. the whole chunk performs no write. Those chunks
 * used to be emitted (a compare, a load and a masked store that writes nothing).
 *
 * Only whole-chunk elision is considered, and only when every store is masked
 * and no pre/post statement shares the chunk body: a single unmasked store, or a
 * sibling statement, keeps the chunk live.
 */
void MarkLiveChunkRange(VectorPlan& plan, const ValueContext& ctx) {
  if (!plan->chunks.has_value() || *plan->chunks <= 1) return;
  if (plan->stores.empty() || !plan->pre_stores.empty() || !plan->post_stores.empty()) return;
  if (ReduceOf(plan).defined()) return;
  const int64_t chunks = *plan->chunks;
  int64_t live_lo = 0;
  int64_t live_hi = chunks;
  for (const StorePlan& store : plan->stores) {
    if (!store->mask.has_value()) return;  // an unconditional write keeps every chunk
    const auto dead = GuardFreeBounds(logical_not(*store->spec->mask), ctx);
    live_lo = std::max(live_lo, dead.first);
    live_hi = std::min(live_hi, dead.second.value_or(chunks));
  }
  if (live_lo <= 0 && live_hi >= chunks) return;      // nothing to elide
  if (live_lo >= live_hi) return;                     // degenerate: the whole nest is dead
  plan.CopyOnWrite()->chunk_lo = live_lo;
  plan.CopyOnWrite()->chunk_hi = live_hi;
}

std::pair<ValueNode, int64_t> StripReduce(const ValueNode& value, const std::string& op,
                                          const StoreSpec& site) {
  std::string scalar_recipe;
  ReduceScalarRecipeFor(op, &scalar_recipe);
  std::set<ValueRecipe> spine = {ValueRecipeFromName(op),
                                 ValueRecipeFromName(scalar_recipe)};
  const tirx::Buffer& buffer = site->buffer;
  int64_t accumulated = 0;
  bool repeated = false;

  std::function<ValueNode(const ValueNode&)> strip = [&](const ValueNode& node) -> ValueNode {
    if (node->id == ValueRecipe::kUniformLoad && node->access.has_value() &&
        SameBuffer((*node->access)->buffer, buffer) &&
        SameIndices((*node->access)->indices, site->indices)) {
      ++accumulated;
      if (accumulated > 1) repeated = true;
      return ValueNode();
    }
    if (spine.count(node->id) != 0 && node->args.size() == 2) {
      std::vector<ValueNode> kept;
      for (const ffi::ObjectRef& arg : node->args) {
        ValueNode child = strip(Downcast<ValueNode>(arg));
        if (child.defined()) kept.push_back(child);
      }
      if (kept.empty()) return ValueNode();
      if (kept.size() == 1) return kept[0];
      ValueSpec spec;
      spec.recipe = node->recipe;
      spec.dtype = node->dtype;
      spec.args = kept;
      spec.detail = node->detail;
      return MakeValueNode(std::move(spec));
    }
    return node;
  };

  ValueNode varying = strip(value);
  if (repeated) {
    throw Unsupported("the old value of '" + buffer->name.operator std::string() +
                      "' appears more than once inside the reduction expression "
                      "(方案 §2 Case II): only a single same-address old-value operand can be folded into the "
                      "reduction's initial value");
  }
  if (varying.defined() && ReadsBuffer(varying, buffer)) {
    throw Unsupported("the reduction body reads the destination buffer '" +
                      buffer->name.operator std::string() +
                      "' outside the accumulate position "
                      "(方案 §2 Case II): the old value must be a direct operand of the reduction operator");
  }
  return {varying, accumulated};
}

ReduceInfo PlanReduce(const StorePlan& store, const Axis& vector_axis) {
  const tirx::Buffer& buffer = store->spec->buffer;
  if (store->mask.has_value()) {
    // A conditional reduction has no vector form yet: the cross-lane mask only
    // carries the tail predicate, so the user guard would be silently dropped.
    throw Unsupported("conditional reduction into '" +
                      buffer->name.operator std::string() +
                      "' is not supported yet: the cross-lane reduction mask only "
                      "carries the tail predicate, so the guard would be silently dropped "
                      "(方案 §2 Case II / R3)",
                      {}, "reduce.conditional");
  }
  if (!ReadsBuffer(store->value, buffer)) {
    throw Unsupported("store to '" + buffer->name.operator std::string() +
                      "' is lane-uniform along axis '" +
                      vector_axis->name.operator std::string() +
                      "' but its value does not read '" +
                      buffer->name.operator std::string() +
                      "': only the accumulate form 'dst[...] = dst[...] <op> x' (op ∈ Add/Max/Min) is a "
                      "reduction (方案 §2 Case II); a lane-varying value written to a lane-uniform address is a "
                      "last-write-wins overwrite and has no vector form");
  }
  std::string op = ReduceOpOf({store->spec});
  auto stripped = StripReduce(store->value, op, store->spec);
  if (stripped.second == 0 || !stripped.first.defined()) {
    throw Unsupported("the old value of '" + buffer->name.operator std::string() +
                      "' must be a direct operand of the '" + op +
                      "' reduction "
                      "(方案 §2 Case II); it cannot be scaled or nested inside another operation");
  }
  std::string recipe;
  ReduceRecipeFor(op, &recipe);
  return MakeReduceInfo(vector_axis, op, recipe, stripped.first, /*accumulate=*/true);
}

/*! \brief Segment-level reduction classification (D13 single criterion). */
ReduceInfo ClassifyReduce(VectorPlan& plan, arith::Analyzer* analyzer) {
  std::vector<StorePlan> stores;
  for (const StorePlan& store : plan->stores) {
    // A private scalar has one instance per parallel iteration: a lane-uniform
    // write along a parallel axis is a per-lane accumulator, not a reduction.
    if (plan->vector_axis->is_parallel && IsPrivateScope(store->spec->buffer)) continue;
    stores.push_back(store);
  }
  if (stores.empty()) return ReduceInfo();

  std::vector<StoreSpec> specs;
  std::vector<AccessInfo> infos;
  for (const StorePlan& store : stores) {
    specs.push_back(store->spec);
    infos.push_back(store->access);
  }
  if (!IsReduceAxis(plan->vector_axis, specs, infos, analyzer)) return ReduceInfo();

  ReduceInfo primary;
  for (size_t i = 0; i < stores.size(); ++i) {
    if (!ValueVariesAlong(stores[i]->spec->value, plan->vector_axis, analyzer)) {
      continue;  // lane-uniform value: a single broadcast write, no reduction
    }
    ReduceInfo reduce = PlanReduce(stores[i], plan->vector_axis);
    for (size_t j = 0; j < plan->stores.size(); ++j) {
      if (plan->stores[j].get() == stores[i].get()) {
        StorePlan copy = plan->stores[j];
        copy.CopyOnWrite()->reduce = ffi::Optional<ffi::ObjectRef>(reduce);
        plan.CopyOnWrite()->stores.Set(static_cast<int64_t>(j), copy);
        break;
      }
    }
    if (!primary.defined()) primary = reduce;
  }
  return primary;
}

/* ------------------------------------------------------------------------- */
/* stencil / guard cuts / per-axis plan                                      */
/* ------------------------------------------------------------------------- */

std::optional<std::string> SpecCrossAddressRead(
    const StoreSpec& spec,
    const std::vector<std::pair<tirx::Buffer, ffi::Array<PrimExpr>>>& sites,
    const Axis& vector_axis, arith::Analyzer* analyzer) {
  std::vector<std::string> bad;
  tirx::PostOrderVisit(spec->value, [&](const ffi::ObjectRef& node) {
    const auto* load = node.as<tirx::BufferLoadNode>();
    if (load == nullptr) return;
    bool touches = false;
    bool same_address = false;
    for (const auto& site : sites) {
      if (!SameBuffer(load->buffer, site.first)) continue;
      touches = true;
      if (SameIndices(load->indices, site.second)) same_address = true;
    }
    if (!touches || same_address) return;
    AccessInfo info =
        ClassifyAccess(load->buffer, load->indices, {vector_axis}, analyzer, false);
    if (AxisClassOf(info, vector_axis).kind != AccessClass::kUniform) {
      bad.push_back(load->buffer->name.operator std::string());
    }
  });
  if (bad.empty()) return std::nullopt;
  return bad.front();
}

std::optional<std::string> CrossAddressRead(const std::vector<StoreSpec>& specs,
                                            const Axis& vector_axis,
                                            arith::Analyzer* analyzer) {
  std::vector<std::pair<tirx::Buffer, ffi::Array<PrimExpr>>> sites;
  for (const StoreSpec& spec : specs) sites.emplace_back(spec->buffer, spec->indices);
  for (const StoreSpec& spec : specs) {
    std::optional<std::string> name = SpecCrossAddressRead(spec, sites, vector_axis, analyzer);
    if (name.has_value()) {
      return "read of '" + *name +
             "' at a different address than any write in the same nest is a stencil along "
             "vector axis '" +
             vector_axis->name.operator std::string() +
             "': vectorizing would read a different element than the scalar order";
    }
  }
  return std::nullopt;
}

std::vector<int64_t> GuardCuts(const VectorPlan& plan) {
  if (!plan->chunks.has_value() || *plan->chunks <= 1) return {};
  int64_t chunks = *plan->chunks;
  std::set<int64_t> cuts;
  std::vector<ValueNode> stack;
  std::vector<StorePlan> stores(plan->stores.begin(), plan->stores.end());
  if (plan->vector_axis->is_parallel) {
    stores.insert(stores.end(), plan->pre_stores.begin(), plan->pre_stores.end());
    stores.insert(stores.end(), plan->post_stores.begin(), plan->post_stores.end());
  }
  for (const StorePlan& store : stores) {
    if (store->mask.has_value()) {
      cuts.insert(store->mask_free_until);
      if (store->mask_free_from.has_value()) cuts.insert(*store->mask_free_from);
    }
    stack.push_back(store->value);
    if (store->mask.has_value()) stack.push_back(*store->mask);
    if (store->index.has_value()) stack.push_back(*store->index);
  }
  while (!stack.empty()) {
    ValueNode node = stack.back();
    stack.pop_back();
    ValueNode predicate = PredicateOf(node);
    if (predicate.defined()) {
      cuts.insert(node->predicate_free_until);
      if (node->predicate_free_from.has_value()) cuts.insert(*node->predicate_free_from);
    }
    for (const ffi::ObjectRef& arg : node->args) stack.push_back(Downcast<ValueNode>(arg));
  }
  std::vector<int64_t> interior;
  for (int64_t cut : cuts) {
    if (cut > 0 && cut < chunks) interior.push_back(cut);
  }
  if (static_cast<int64_t>(interior.size()) > kGuardCutLimit) return {};
  return interior;
}

/*! \brief Program-layer axes: the leading run of non-vector parallel axes. */
std::vector<Axis> OuterAxes(const ffi::Array<Axis>& axes, const Axis& vector_axis) {
  std::vector<Axis> outer;
  for (const Axis& axis : axes) {
    if (axis->var.same_as(vector_axis->var) || !axis->is_parallel) break;
    outer.push_back(axis);
  }
  return outer;
}

/*! \brief Whether the chunk is narrower than at least one store's register. */
bool HasPartialLaneStore(const VectorPlan& plan) {
  for (const StorePlan& store : plan->stores) {
    if (store->store_lane_mode.operator std::string() != "full") return true;
  }
  return false;
}

/*! \brief Collect the register widths of every real load in a value DAG. */
void CollectLoadLanes(const ValueNode& node, std::set<int64_t>* lanes) {
  if (!node.defined()) return;
  if (node->id == ValueRecipe::kVLoad || node->id == ValueRecipe::kVLoadStrided ||
      node->id == ValueRecipe::kGather) {
    lanes->insert(VRegLanes(ElemDtype(node->dtype.operator std::string())));
  }
  for (const ffi::ObjectRef& arg : node->args) {
    CollectLoadLanes(Downcast<ValueNode>(arg), lanes);
  }
  if (node->predicate.has_value()) {
    CollectLoadLanes(Downcast<ValueNode>(*node->predicate), lanes);
  }
}

/*!
 * \brief Stage 1 guards for a segment whose chunk is narrower than some store.
 *
 * A partial-lane segment writes ``lanes`` elements per chunk out of a wider
 * store register. It is only proved for straight-line, unpredicated,
 * contiguous element-wise stores whose values load exactly ``lanes`` wide and
 * whose extent divides evenly (no tail); everything else is rejected (the nest
 * stays a serial loop).
 */
void ValidatePartialLaneSegment(const VectorPlan& plan,
                                const std::optional<int64_t>& extent, int64_t lanes) {
  if (!extent.has_value()) {
    throw Unsupported(
        "mixed element widths in one segment require a static vector-axis extent that is a "
        "multiple of the chunk width " +
        std::to_string(lanes) + " (got a symbolic extent)");
  }
  if (*extent % lanes != 0) {
    throw Unsupported(
        "mixed element widths in one segment require the vector-axis extent (" +
        std::to_string(*extent) + ") to be a multiple of the chunk width " +
        std::to_string(lanes) + " (tails are not supported yet)");
  }
  for (const StorePlan& store : plan->stores) {
    const std::string name = store->spec->buffer->name.operator std::string();
    const std::string kind = AccessClassName(store->kind);
    if (store->kind != AccessClass::kContiguous && store->kind != AccessClass::kUniform) {
      throw Unsupported(
          "mixed element widths in one segment only support contiguous or uniform stores ('" +
          name + "' is " + kind + ")");
    }
    std::set<int64_t> load_lanes;
    CollectLoadLanes(store->value, &load_lanes);
    if (store->mask.has_value()) CollectLoadLanes(*store->mask, &load_lanes);
    if (store->index.has_value()) CollectLoadLanes(*store->index, &load_lanes);
    for (int64_t width : load_lanes) {
      if (width != lanes) {
        throw Unsupported("mixed element widths in one segment require every load to be " +
                          std::to_string(lanes) + "-lane wide (store '" + name +
                          "' loads a " + std::to_string(width) + "-lane register)");
      }
    }
  }
}

/*! \brief Whether any store of the segment targets an explicitly annotated NZ buffer. */
bool HasNzStore(const VectorPlan& plan) {
  for (const StorePlan& store : plan->stores) {
    if (store->layout_kind.operator std::string() == "nz") return true;
  }
  return false;
}

/*!
 * \brief Stage 2b: validate the paired form of a half-register NZ store.
 *
 * ``vsstb`` has no packed variant, so a ``pk_half`` NZ store cannot be scattered
 * as planned: the value register only holds ``lanes`` valid elements out of its
 * ``2 * lanes`` lanes. It is emitted in the *paired* form instead — two
 * consecutive chunks are de-interleaved into one full register (``vdintlv`` +
 * two ``vcvt`` parts + ``vor``) and written with a single ``vsstb``, the same
 * recipe as the hand-written FA softmax. That form only exists when the segment
 * is a clean element-wise body over whole chunk pairs and the packed store owns
 * its destination buffer, so everything else stays ``Unsupported``.
 */
void ValidatePackedNzSegment(const VectorPlan& plan,
                             const std::vector<Axis>& axes,
                             const Axis& vector_axis,
                             const std::optional<int64_t>& extent, int64_t lanes) {
  std::optional<size_t> packed;
  for (size_t i = 0; i < plan->stores.size(); ++i) {
    const StorePlan& store = plan->stores[i];
    if (store->layout_kind.operator std::string() != "nz") continue;
    if (store->store_lane_mode.operator std::string() != "pk_half") continue;
    if (packed.has_value()) {
      throw Unsupported(
          "at most one half-register NZ store per segment is supported (split the nest)");
    }
    packed = i;
  }
  if (!packed.has_value()) return;

  const StorePlan& target = plan->stores[*packed];
  const std::string name = target->spec->buffer->name.operator std::string();
  // Defect L3: the paired emitter requires the vector axis to be the innermost
  // axis; without this gate P2 succeeds and P3 rejects the same plan.
  for (const Axis& axis : axes) {
    if (axis->index > vector_axis->index) {
      throw Unsupported("the packed NZ store to '" + name +
                        "' needs the vector axis to be the innermost axis "
                        "(split the nest)");
    }
  }
  if (*packed != plan->stores.size() - 1) {
    throw Unsupported("the packed NZ store to '" + name +
                      "' must be the last store of the segment (the merged register only exists "
                      "after both chunks)");
  }
  for (const StorePlan& store : plan->stores) {
    if (store.same_as(target)) continue;
    if (store->spec->buffer->data.same_as(target->spec->buffer->data)) {
      throw Unsupported("the packed NZ store to '" + name +
                        "' must own its buffer (another store of the segment writes it)");
    }
  }
  if (target->value->recipe.operator std::string() != "vcvt" ||
      target->value->args.size() != 1) {
    throw Unsupported("the packed NZ store to '" + name +
                      "' needs a plain narrowing cast as its value");
  }
  if (!extent.has_value() || *extent % (2 * lanes) != 0) {
    throw Unsupported("the packed NZ store to '" + name +
                      "' needs the vector axis to cover whole chunk pairs (extent=" +
                      (extent.has_value() ? std::to_string(*extent) : std::string("?")) +
                      ", pair=" + std::to_string(2 * lanes) + ")");
  }
}

/*! \brief Find the explicit layout of a store buffer (same buffer or shared data var). */
const Layout* LookupStoreLayout(const PlanOptions& options, const tirx::Buffer& buffer) {
  for (const auto& entry : options.layouts) {
    if (entry.first.same_as(buffer) || entry.first->data.same_as(buffer->data)) {
      return &entry.second;
    }
  }
  return nullptr;
}

/*! \brief Row-major linearized physical offset of a logical index under ``layout``. */
PrimExpr PhysicalOffset(const Layout& layout, const ffi::Array<PrimExpr>& logical) {
  ffi::Array<PrimExpr> out = layout->Forward(logical);
  ffi::Array<PrimExpr> shape = layout->OutputShape();
  PrimExpr offset = IntImm(DataType::Int(64), 0);
  for (size_t i = 0; i < out.size(); ++i) {
    offset = offset * tirx::Cast(DataType::Int(64), shape[i]) +
             tirx::Cast(DataType::Int(64), out[i]);
  }
  return offset;
}

/*!
 * \brief Classify one store's destination layout (Stage 2).
 *
 * Default is ND: a buffer without an explicit layout annotation keeps the plain
 * contiguous store. A buffer explicitly annotated with an Ascend NZ layout is
 * scattered as 32B blocks straight from the register (``vsstb``); the block and
 * element strides are derived from the layout forward map instead of being
 * hard-coded, and any shape the scatter cannot express stays ``Unsupported`` so
 * the nest is left as a serial loop instead of writing a wrong layout.
 *
 * Returns the classification instead of mutating ``store``: a by-value
 * ``StorePlan`` parameter would bump the node refcount and turn the caller's
 * ``CopyOnWrite`` into a private copy (the write would be lost).
 */
struct StoreLayoutPlan {
  bool nz{false};
  int64_t block_stride{0};  // 32B blocks between two neighbouring C0 groups
};

StoreLayoutPlan ClassifyStoreLayout(const StorePlan& store, const PlanOptions& options,
                                    int64_t lanes, arith::Analyzer* analyzer) {
  StoreLayoutPlan result;
  const tirx::Buffer& buffer = store->spec->buffer;
  const Layout* layout = LookupStoreLayout(options, buffer);
  if (layout == nullptr || !layout->defined() ||
      layout->as<FragmentNode>() != nullptr) {
    return result;  // no explicit (physical) layout: default ND
  }
  const std::string name = buffer->name.operator std::string();
  if (!IsAscendNZLayout(*layout, buffer)) {
    throw Unsupported("store to '" + name +
                      "' has an explicit non-NZ layout; only ND (default) and Ascend NZ "
                      "destination layouts are supported");
  }
  const std::string lane_mode = store->store_lane_mode.operator std::string();
  if (lane_mode == "pk_quarter") {
    throw Unsupported("NZ store to '" + name +
                      "' would need a quarter-register source; only the half-register packed "
                      "form is implemented");
  }
  if (store->kind != AccessClass::kContiguous) {
    throw Unsupported("NZ store to '" + name +
                      "' requires a contiguous access along the vector axis (got " +
                      AccessClassName(store->kind) + ")");
  }
  if (store->mask.has_value()) {
    throw Unsupported("NZ store to '" + name +
                      "' does not support a statement predicate yet",
                      {}, "body.statement_predicate");
  }
  const int c0 = AscendC0(buffer->dtype.bits());
  std::optional<int64_t> cols = ConstInt(buffer->shape.back(), analyzer);
  if (c0 <= 0 || lanes % c0 != 0) {
    throw Unsupported("NZ store to '" + name + "' needs a chunk width multiple of the 32B block "
                      "(" + std::to_string(c0) + " elements), got " + std::to_string(lanes));
  }
  if (!cols.has_value() || *cols % lanes != 0 || lanes > *cols) {
    throw Unsupported(
        "NZ store to '" + name +
            "' needs a static column extent covering a whole number of chunks (cols=" +
            (cols.has_value() ? std::to_string(*cols) : std::string("?")) +
            ", chunk=" + std::to_string(lanes) + ")",
        {}, "plan.partial_chunks");
  }
  // A half-register source is written in the paired form (Stage 2b): two chunks
  // are merged into one full register, so the columns must cover whole pairs.
  if (lane_mode == "pk_half" && *cols % (2 * lanes) != 0) {
    throw Unsupported("NZ store to '" + name + "' narrows to a half register, so it needs whole " +
                      "chunk pairs (cols=" + std::to_string(*cols) + ", pair=" +
                      std::to_string(2 * lanes) + ")");
  }
  // Block stride from the layout forward map: the distance between the 32B block
  // holding (row, col) and the one holding (row, col + C0).
  ffi::Array<PrimExpr> indices = store->spec->indices;
  ffi::Array<PrimExpr> next;
  for (size_t i = 0; i < indices.size(); ++i) {
    next.push_back(i + 1 == indices.size() ? indices[i] + IntImm32(c0) : indices[i]);
  }
  PrimExpr step = analyzer->Simplify(PhysicalOffset(*layout, next) -
                                     PhysicalOffset(*layout, indices));
  std::optional<int64_t> block = ConstInt(step, analyzer);
  if (!block.has_value() || *block <= 0 || *block % c0 != 0) {
    throw Unsupported("NZ store to '" + name +
                      "' has a non-constant 32B block stride under its layout");
  }
  result.nz = true;
  result.block_stride = *block / c0;
  return result;
}

/* ------------------------------------------------------------------------- */
/* fragment register planning (G0/G0b + G1-G8)                              */
/* ------------------------------------------------------------------------- */

/*! \brief One fragment access site of the nest (a store target or a load). */
struct FragmentAccessSite {
  tirx::Buffer buffer;
  ffi::Array<PrimExpr> indices;
  bool is_store{false};
};

/*! \brief P2's per-nest fragment decision (design §2.1.2). */
struct FragmentMode {
  bool enabled{false};
  std::vector<FragmentReg> regs;
};

bool IsRegisterFragment(const tirx::Buffer& buffer) {
  return StorageClassOf(buffer) == StorageClass::kFragment;
}

/*! \brief Element dtypes v1 maps onto a vector register (G7). */
bool FragmentDtypeSupported(const std::string& elem) {
  static const std::set<std::string> supported = {
      "int8",    "int16",   "int32",   "uint8",    "uint16",
      "uint32",  "float16", "bfloat16", "float32"};
  return supported.count(elem) != 0;
}

/*!
 * \brief G6: a fragment layout must be the identity on the flattened index.
 *
 * A padded / replicated / permuted fragment layout would make the register slot
 * of an element depend on more than ``off / lanes``; the register file cannot
 * represent that, so the nest stays scalar.
 */
bool FragmentLayoutIsIdentity(const PlanOptions& options, const tirx::Buffer& buffer) {
  const Layout* layout = LookupStoreLayout(options, buffer);
  if (layout == nullptr || !layout->defined()) return true;
  const auto* fragment = layout->as<FragmentNode>();
  if (fragment == nullptr) return false;
  if (!tirx::is_one(fragment->ReplicateExtent())) return false;
  ffi::Array<PrimExpr> vars;
  for (size_t i = 0; i < fragment->InputDim(); ++i) {
    vars.push_back(tirx::Var("frag_i" + std::to_string(i), DataType::Int(32)));
  }
  ffi::Array<PrimExpr> forward = (*layout)->Forward(vars);
  if (forward.size() != vars.size()) return false;
  for (size_t i = 0; i < forward.size(); ++i) {
    const auto* var = forward[i].as<tirx::VarNode>();
    if (var == nullptr || !ffi::GetRef<tirx::Var>(var).same_as(vars[i])) return false;
  }
  return true;
}

/*! \brief Collect every fragment load nested in one expression. */
void CollectFragmentLoads(const PrimExpr& expr,
                          std::vector<FragmentAccessSite>* out) {
  if (!expr.defined()) return;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef& node) {
    const auto* load = node.as<tirx::BufferLoadNode>();
    if (load == nullptr || !IsRegisterFragment(load->buffer)) return;
    out->push_back(FragmentAccessSite{load->buffer, load->indices, false});
  });
}

/*!
 * \brief G0/G0b + G1-G8 over every fragment access of one nest.
 *
 * The decision is atomic per nest: any violation throws, and P2 then attaches
 * no plan carrier, so the whole nest falls back to the scalar path instead of
 * producing an un-compilable ``vld``/``vsts`` pair.
 */
FragmentMode ClassifyFragmentRegisters(
    const std::vector<StoreSpec>& pre_specs,
    const std::vector<StoreSpec>& main_specs,
    const std::vector<StoreSpec>& post_specs, const std::vector<Axis>& axes,
    const Axis& vector_axis, int64_t lanes, const PlanOptions& options,
    arith::Analyzer* analyzer) {
  std::vector<FragmentAccessSite> sites;
  auto collect_spec = [&](const StoreSpec& spec, bool is_store) {
    if (is_store && IsRegisterFragment(spec->buffer)) {
      sites.push_back(FragmentAccessSite{spec->buffer, spec->indices, true});
    }
    CollectFragmentLoads(spec->value, &sites);
    if (spec->mask.has_value()) CollectFragmentLoads(*spec->mask, &sites);
    for (const PrimExpr& index : spec->indices) CollectFragmentLoads(index, &sites);
  };
  for (const StoreSpec& spec : pre_specs) collect_spec(spec, true);
  for (const StoreSpec& spec : main_specs) collect_spec(spec, true);
  for (const StoreSpec& spec : post_specs) collect_spec(spec, true);
  if (sites.empty()) return FragmentMode();

  // G0b: offline callers have no VF context; never plan a fragment blindly.
  if (options.vf_alloc_scope.empty()) {
    throw Unsupported(
        "fragment register: no VF allocation scope provided; fragment register "
        "mode requires the P1 region hand-off");
  }
  std::set<const tirx::VarNode*> scope;
  for (const tirx::Buffer& buffer : options.vf_alloc_scope) {
    scope.insert(buffer->data.get());
  }
  std::set<const tirx::VarNode*> axis_vars;
  for (const Axis& axis : axes) axis_vars.insert(axis->var.get());

  std::optional<int64_t> extent = ConstInt(vector_axis->extent, analyzer);
  if (!vector_axis->is_parallel) {
    throw Unsupported("fragment register: the vector axis '" +
                      vector_axis->name.operator std::string() +
                      "' must be a T.Parallel axis");
  }
  for (const Axis& axis : axes) {
    if (axis->index > vector_axis->index) {
      throw Unsupported(
          "fragment register: the vector axis '" +
          vector_axis->name.operator std::string() +
          "' must be the innermost axis of the nest (found inner axis '" +
          axis->name.operator std::string() + "')");
    }
  }
  if (!extent.has_value()) {
    throw Unsupported("fragment register: the vector axis extent must be a "
                      "compile-time constant");
  }
  int64_t chunks = (*extent + lanes - 1) / lanes;
  if (chunks <= 0 || chunks > kFragmentUnrollLimit) {
    throw Unsupported("fragment register: " + std::to_string(chunks) +
                      " chunks per nest is outside the expandable range [1, " +
                      std::to_string(kFragmentUnrollLimit) + "]");
  }

  FragmentMode mode;
  mode.enabled = true;
  std::map<const tirx::VarNode*, FragmentReg> by_data;
  for (const FragmentAccessSite& site : sites) {
    const tirx::Buffer& buffer = site.buffer;
    const std::string name = buffer->name.operator std::string();
    // G0: the fragment must be allocated by the current VF.
    if (scope.count(buffer->data.get()) == 0) {
      throw Unsupported(
          "fragment register: '" + name +
          "' is allocated outside this VF; bisheng would capture it as a "
          "pointer. Move T.alloc_fragment inside T.SimdVF");
    }
    // G6: no padded / replicated / permuted fragment layout.
    if (!FragmentLayoutIsIdentity(options, buffer)) {
      const Layout* layout = LookupStoreLayout(options, buffer);
      throw Unsupported("fragment register: '" + name +
                            "' has a non-identity layout (" +
                            (layout != nullptr ? (*layout)->DebugOutput()
                                               : std::string("unknown")) +
                            ")",
                        {}, "fragment.non_identity_layout");
    }
    // G7: supported element dtype and matching lane width.
    std::string elem = ffi::DLDataTypeToString(buffer->dtype.element_of())
                           .operator std::string();
    if (!FragmentDtypeSupported(elem)) {
      throw Unsupported("fragment register: unsupported dtype '" + elem +
                        "' for '" + name + "'");
    }
    if (VRegLanes(elem) != lanes) {
      throw Unsupported("fragment register: unsupported lane width for '" + name +
                        "' (fragment " + std::to_string(VRegLanes(elem)) +
                        " lanes, chunk " + std::to_string(lanes) + ")");
    }
    // G7: static, non-empty shape -> a fixed register count.
    int64_t elems = 1;
    for (const PrimExpr& dim : buffer->shape) {
      std::optional<int64_t> value = ConstInt(dim, analyzer);
      if (!value.has_value() || *value <= 0) {
        throw Unsupported("fragment register: unsupported shape for '" + name +
                          "' (all fragment dimensions must be static and positive)");
      }
      elems *= *value;
    }
    int64_t slots = (elems + lanes - 1) / lanes;
    if (slots <= 0) {
      throw Unsupported("fragment register: unsupported shape for '" + name +
                        "' (empty register file)");
    }

    AccessInfo info =
        ClassifyAccess(buffer, site.indices, axes, analyzer, site.is_store);
    // G1: contiguous along the vector axis (reads *and* writes).
    AxisClass classified = AxisClassOf(info, vector_axis);
    if (classified.kind != AccessClass::kContiguous) {
      throw Unsupported(
          "fragment register: '" + name + "' is not contiguous along vector axis '" +
              vector_axis->name.operator std::string() + "' (Δ=" +
              AccessClassName(classified.kind) +
              (classified.stride.has_value()
                   ? ", stride=" + std::to_string(*classified.stride)
                   : std::string()) +
              ")",
          {}, "fragment.not_contiguous");
    }
    // G2/G3/G4: slot = off / lanes must be a compile-time constant that only
    // depends on the vector axis.
    for (const tirx::Var& var : FreeVars(info->offset)) {
      if (var.same_as(vector_axis->var)) continue;
      if (axis_vars.count(var.get()) != 0) {
        throw Unsupported("fragment register: '" + name +
                          "' slot changes along axis '" + var->name_hint +
                          "'; a register has no dynamic index");
      }
      throw Unsupported("fragment register: '" + name +
                        "' slot is not a compile-time constant");
    }
    PrimExpr base_expr =
        SubstituteVars(info->offset, {{vector_axis->var.get(), IntImm32(0)}});
    std::optional<int64_t> base = ConstInt(analyzer->Simplify(base_expr), analyzer);
    if (!base.has_value()) {
      throw Unsupported("fragment register: '" + name +
                        "' slot is not a compile-time constant");
    }
    if (*base % lanes != 0) {
      throw Unsupported("fragment register: '" + name + "' chunk base " +
                        std::to_string(*base) + " is not aligned to " +
                        std::to_string(lanes) + " lanes");
    }
    int64_t base_slot = *base / lanes;
    if (base_slot + chunks > slots) {
      throw Unsupported("fragment register: '" + name + "' access spans slot " +
                        std::to_string(base_slot + chunks - 1) +
                        " but only " + std::to_string(slots) +
                        " registers are allocated");
    }
    // G8: v1 only writes whole chunks; a partial (tail) chunk would need the
    // old register value and P2 cannot prove it is initialized across nests.
    if (site.is_store && (*extent % lanes) != 0) {
      throw Unsupported("fragment register: '" + name + "' partially writes chunk " +
                        std::to_string(chunks - 1) +
                        " before it is initialized; write whole chunks "
                        "(T.if_then_else fills every lane) or pad the extent");
    }
    auto it = by_data.find(buffer->data.get());
    if (it == by_data.end()) {
      FragmentReg reg = MakeFragmentReg(buffer->data, name, slots, lanes, elem);
      by_data.emplace(buffer->data.get(), reg);
      mode.regs.push_back(reg);
    } else if (it->second->slots != slots || it->second->lanes != lanes ||
               it->second->dtype.operator std::string() != elem) {
      throw Unsupported("fragment register: '" + name +
                        "' has inconsistent register shapes across its views");
    }
  }
  return mode;
}

VectorPlan BuildPlanForAxis(const std::vector<Axis>& axes,
                            std::optional<Axis> serial_axis,
                            const std::vector<StoreSpec>& pre_specs,
                            const std::vector<AccessInfo>& pre_infos,
                            const std::vector<StoreSpec>& main_specs,
                            const std::vector<AccessInfo>& main_infos,
                            const std::vector<StoreSpec>& post_specs,
                            const std::vector<AccessInfo>& post_infos,
                            const Axis& vector_axis, int64_t lanes,
                            arith::Analyzer* analyzer, const PlanOptions& options) {
  bool along_serial =
      serial_axis.has_value() && vector_axis->var.same_as((*serial_axis)->var);
  if (!pre_specs.empty() || !post_specs.empty()) {
    int64_t serial_count = 0;
    for (const Axis& axis : axes) serial_count += axis->is_parallel ? 0 : 1;
    if (!along_serial && !(vector_axis->is_parallel && serial_count == 1)) {
      throw Unsupported(
          "a T.Parallel body with sibling statements around a serial loop can only be vectorized "
          "along that serial axis or a single outer parallel axis (方案 §6 第二步)");
    }
  }

  std::optional<int64_t> extent = ConstInt(vector_axis->extent, analyzer);
  // Defect L2: a zero-extent plan has no chunks, and the emitter indexes an
  // empty statement sequence. Reject it here instead of producing a plan that
  // crashes (or silently truncates) in P3.
  if (extent.has_value() && *extent == 0) {
    throw Unsupported("zero-extent parallel axis '" +
                      vector_axis->name.operator std::string() + "'");
  }
  std::optional<int64_t> chunks;
  std::optional<int64_t> tail;
  if (extent.has_value()) {
    chunks = (*extent + lanes - 1) / lanes;
    tail = *extent % lanes;
  }

  // Fragment register mode (G0-G8): decided before any value is built, because
  // a fragment load becomes a ``reg_load`` recipe instead of a ``vload``.
  FragmentMode fragment = ClassifyFragmentRegisters(
      pre_specs, main_specs, post_specs, axes, vector_axis, lanes, options, analyzer);
  if (fragment.enabled) {
    for (const StoreSpec& spec : pre_specs) {
      if (IsRegisterFragment(spec->buffer)) {
        throw Unsupported("fragment register: a pre-loop store to '" +
                          spec->buffer->name.operator std::string() +
                          "' has no register form; split the nest");
      }
    }
    for (const StoreSpec& spec : post_specs) {
      if (IsRegisterFragment(spec->buffer)) {
        throw Unsupported("fragment register: a post-loop store to '" +
                          spec->buffer->name.operator std::string() +
                          "' has no register form; split the nest");
      }
    }
  }

  ValueContext ctx;
  ctx.axes = axes;
  ctx.vector_axis = vector_axis;
  ctx.analyzer = analyzer;
  ctx.lanes = lanes;
  ctx.register_fragments = fragment.enabled;

  VectorPlan plan = MakeVectorPlan(ToArray(axes), vector_axis, lanes,
                                   chunks, tail);
  // The plan is uniquely owned here, so take the writable node once instead of
  // re-checking the refcount at every field write (P6).
  VectorPlanNode* writer = plan.CopyOnWrite();

  auto plan_store = [&](const StoreSpec& spec, const AccessInfo& info) -> StorePlan {
    AxisClass classified = AxisClassOf(info, vector_axis);
    ffi::Optional<ValueNode> mask;
    if (spec->mask.has_value()) {
      mask = ffi::Optional<ValueNode>(BuildValue(*spec->mask, ctx));
    }
    int64_t mask_free_until = 0;
    std::optional<int64_t> mask_free_from;
    if (spec->mask.has_value()) {
      auto bounds = GuardFreeBounds(*spec->mask, ctx);
      mask_free_until = bounds.first;
      mask_free_from = bounds.second;
    }
    ValueContext value_ctx = ctx;
    if (mask.has_value()) {
      if (!ctx.guard.defined()) {
        value_ctx = WithGuard(ctx, *mask, mask_free_until, mask_free_from);
      } else {
        value_ctx = WithGuard(ctx, AndPredicate(ctx.guard, *mask),
                              CombineFreeUntil(ctx.guard_free_until, mask_free_until),
                              CombineFreeFrom(ctx.guard_free_from, mask_free_from));
      }
    }
    ValueNode value = BuildValue(spec->value, value_ctx);
    bool needs_index = classified.kind == AccessClass::kPermute ||
                       classified.kind == AccessClass::kPiecewise ||
                       classified.kind == AccessClass::kIndirect ||
                       (classified.kind == AccessClass::kStrided &&
                        classified.stride.has_value() && *classified.stride < 0);
    ffi::Optional<ValueNode> index;
    if (needs_index) {
      index = ffi::Optional<ValueNode>(IndexValue(info, ctx));
    }
    StorePlan store = MakeStorePlan(spec, info, value, mask, classified.kind, index);
    store.CopyOnWrite()->mask_free_until = mask_free_until;
    if (mask_free_from.has_value()) {
      store.CopyOnWrite()->mask_free_from = ffi::Optional<int64_t>(*mask_free_from);
    }
    // Mixed element widths (Stage 1): record how many of this store's register
    // lanes one chunk actually writes. ``full`` keeps the historical form.
    int64_t register_lanes = VRegLanes(ElemDtype(PyBufferDtypeName(spec->buffer)));
    int64_t ratio = lanes > 0 ? register_lanes / lanes : 1;
    store.CopyOnWrite()->store_lanes = register_lanes;
    if (ratio == 2) {
      store.CopyOnWrite()->store_lane_mode = ffi::String("pk_half");
      store.CopyOnWrite()->lane_mode_id = StoreLaneMode::kPkHalf;
    } else if (ratio == 4) {
      store.CopyOnWrite()->store_lane_mode = ffi::String("pk_quarter");
      store.CopyOnWrite()->lane_mode_id = StoreLaneMode::kPkQuarter;
    } else {
      store.CopyOnWrite()->store_lane_mode = ffi::String("full");
      store.CopyOnWrite()->lane_mode_id = StoreLaneMode::kFull;
    }
    if (ratio > 1) {
      store.CopyOnWrite()->store_pack_bits = ffi::Optional<int64_t>(2048 / lanes);
    }
    // Fragment register target (design §2.1.3): no memory instruction is
    // emitted; the chunk maps to one register slot. A packed / narrowing store
    // would hand a wider register to a half chunk, which the register file
    // cannot express (G5).
    if (fragment.enabled && IsRegisterFragment(spec->buffer)) {
      const std::string lane_mode =
          store->store_lane_mode.operator std::string();
      if (lane_mode != "full") {
        throw Unsupported("fragment register: '" +
                          spec->buffer->name.operator std::string() +
                          "' needs a packed/narrowing store (" + lane_mode + ")");
      }
      store.CopyOnWrite()->storage = ffi::String("register");
      store.CopyOnWrite()->storage_id = StoreStorage::kRegister;
    }
    // Destination layout (Stage 2): the classification is applied here, where
    // the plan node is still uniquely owned, so ``CopyOnWrite`` stays in place.
    StoreLayoutPlan layout_plan = ClassifyStoreLayout(store, options, lanes, analyzer);
    if (layout_plan.nz) {
      store.CopyOnWrite()->layout_kind = ffi::String("nz");
      store.CopyOnWrite()->layout_id = StoreLayoutKind::kNz;
      store.CopyOnWrite()->nz_block_stride =
          ffi::Optional<int64_t>(layout_plan.block_stride);
      store.CopyOnWrite()->nz_elem_stride = ffi::Optional<int64_t>(1);
    }
    return store;
  };

  ffi::Array<StorePlan> stores;
  for (size_t i = 0; i < main_specs.size(); ++i) {
    stores.push_back(plan_store(main_specs[i], main_infos[i]));
  }
  ffi::Array<StorePlan> pre_stores;
  for (size_t i = 0; i < pre_specs.size(); ++i) {
    pre_stores.push_back(plan_store(pre_specs[i], pre_infos[i]));
  }
  ffi::Array<StorePlan> post_stores;
  for (size_t i = 0; i < post_specs.size(); ++i) {
    post_stores.push_back(plan_store(post_specs[i], post_infos[i]));
  }
  writer->stores = stores;
  writer->pre_stores = pre_stores;
  writer->post_stores = post_stores;
  if (fragment.enabled) {
    ffi::Array<FragmentReg> regs;
    for (const FragmentReg& reg : fragment.regs) regs.push_back(reg);
    writer->fragment_regs = regs;
  }

  // Mixed element widths (Stage 1): a chunk narrower than some store register
  // is only proved for the straight-line element-wise subset.
  const bool partial_lanes = HasPartialLaneStore(plan);
  if (partial_lanes) ValidatePartialLaneSegment(plan, extent, lanes);

  // Destination layouts (Stage 2): an NZ scatter has no tail or sibling form.
  const bool nz_stores = HasNzStore(plan);
  if (nz_stores) {
    if (!extent.has_value() || *extent % lanes != 0) {
      throw Unsupported(
          "NZ stores require the vector-axis extent to be a multiple of the chunk width " +
          std::to_string(lanes) + " (tails are not supported yet)");
    }
    if (!pre_specs.empty() || !post_specs.empty()) {
      throw Unsupported(
          "NZ stores are only supported without sibling statements around the loop");
    }
  }
  // Stage 2b: a half-register NZ store is widened to a full register by pairing
  // two chunks; that shape has its own segment-level contract.
  ValidatePackedNzSegment(plan, axes, vector_axis, extent, lanes);

  // Along a serial axis, pre/post are siblings of the chunk loop and must be
  // lane-uniform; along a parallel axis they live inside the chunk body.
  if (along_serial) {
    const std::vector<std::pair<std::string, ffi::Array<StorePlan>>> groups = {
        {"before", plan->pre_stores}, {"after", plan->post_stores}};
    for (const auto& group : groups) {
      for (const StorePlan& store : group.second) {
        if (store->kind != AccessClass::kUniform) {
          throw Unsupported(
              "a statement " + group.first +
              " the serial loop must be uniform along vector axis '" +
              vector_axis->name.operator std::string() + "' (got " +
              AccessClassName(store->kind) + ") (方案 §6 第二步)");
        }
      }
    }
  }

  if (vector_axis->is_parallel) {
    std::vector<StoreSpec> all_specs;
    all_specs.insert(all_specs.end(), pre_specs.begin(), pre_specs.end());
    all_specs.insert(all_specs.end(), main_specs.begin(), main_specs.end());
    all_specs.insert(all_specs.end(), post_specs.begin(), post_specs.end());
    std::optional<std::string> stencil = CrossAddressRead(all_specs, vector_axis, analyzer);
    if (stencil.has_value()) throw Unsupported(*stencil);
    // Accumulator promotion is a plan-time decision: emit only consumes it.
    for (int64_t i = 0; i < static_cast<int64_t>(plan->stores.size()); ++i) {
      StorePlan store = plan->stores[i];
      ValueNode promote = AccumulatorCandidate(store, plan, analyzer);
      if (promote.defined()) {
        store.CopyOnWrite()->promote = ffi::Optional<ValueNode>(promote);
        writer->stores.Set(i, store);
      }
    }
    for (const StorePlan& store : plan->stores) {
      bool promoted = store->promote.has_value();
      if (store->kind == AccessClass::kUniform &&
          !store->value->is_uniform && IsPrivateScope(store->spec->buffer) && !promoted) {
        throw Unsupported(
            "store to private scalar '" + store->spec->buffer->name.operator std::string() +
            "' varies along vector axis '" + vector_axis->name.operator std::string() +
            "': vectorizing the outer axis needs a promotable accumulator "
            "(a per-iteration private scalar cannot be shared by all lanes) (方案 §2 Case II)");
      }
    }
  }

  ReduceInfo reduce = ClassifyReduce(plan, analyzer);
  if (fragment.enabled && reduce.defined()) {
    throw Unsupported(
        "fragment register: a reduction into a fragment register has no register "
        "form (the cross-lane result is a scalar); keep the fragment element-wise "
        "or reduce into UB");
  }
  if (reduce.defined() && partial_lanes) {
    throw Unsupported(
        "a reduction cannot share a segment with a wider-lane store (mixed element widths would "
        "fold the unused lanes into the accumulator); split the nest or stage an explicit cast");
  }
  if (reduce.defined() && nz_stores) {
    throw Unsupported(
        "a reduction store cannot be scattered into an NZ destination yet; split the nest");
  }
  if (reduce.defined()) writer->reduce = ffi::Optional<ffi::ObjectRef>(reduce);
  if (reduce.defined()) {
    // The reduction path hands pre/post to the outer-store emitter, which needs
    // lane-uniform statements (R7: close the build-ok/emit-Unsupported gap).
    const std::vector<std::pair<std::string, ffi::Array<StorePlan>>> groups = {
        {"before", plan->pre_stores}, {"after", plan->post_stores}};
    for (const auto& group : groups) {
      for (const StorePlan& store : group.second) {
        if (store->kind != AccessClass::kUniform) {
          throw Unsupported(
              "a statement " + group.first +
              " the serial loop must be uniform along the reduction vector axis '" +
              vector_axis->name.operator std::string() + "' (got " +
              AccessClassName(store->kind) + ") (方案 §6 第二步)");
        }
      }
    }
    FoldIdentityInit(plan);
  }
  DropDeadPreStores(plan, analyzer);
  MarkStoreRoles(plan);
  MarkLiveChunkRange(plan, ctx);

  // Defect A0: a partial reduction on an absorbed serial axis has no vector form.
  if (!reduce.defined()) {
    for (const StorePlan& store : plan->stores) {
      if (store->kind != AccessClass::kUniform || store->value->is_uniform) {
        continue;
      }
      if (vector_axis->is_parallel && store->promote.has_value()) continue;
      if (vector_axis->is_parallel) {
        throw Unsupported(
            "store to '" + store->spec->buffer->name.operator std::string() +
            "' is lane-uniform along vector axis '" +
            vector_axis->name.operator std::string() +
            "' but its value varies with it: a lane-varying value written to a "
            "lane-uniform address has no vector form (方案 §2 Case II)");
      }
      throw Unsupported(
          "store to '" + store->spec->buffer->name.operator std::string() +
          "' is lane-uniform along the absorbed serial axis '" +
          vector_axis->name.operator std::string() +
          "' but its value varies with it: a partial reduction mixed with "
          "element-wise stores has no vector form (方案 §2 Case II)");
    }
  }

  std::vector<int64_t> cuts = GuardCuts(plan);
  ffi::Array<int64_t> guard_boundaries;
  for (int64_t cut : cuts) guard_boundaries.push_back(cut);
  writer->guard_boundaries = guard_boundaries;
  return plan;
}

void NoteGuardSplit(VectorPlan& plan) {
  if (plan->guard_boundaries.empty()) return;
  std::vector<int64_t> cuts(plan->guard_boundaries.begin(), plan->guard_boundaries.end());
  std::vector<std::string> diagnostics = ToStrings(plan->diagnostics);
  diagnostics.push_back("guard-free chunk cuts at " + PyIntList(cuts) +
                        ": masked gathers revert to plain vld");
  plan.CopyOnWrite()->diagnostics = ToStringArray(diagnostics);
}

/*! \brief Plan one segment: flatten channel first, then ranked axis fallback. */
std::pair<VectorPlan, bool> PlanSegment(const std::vector<Axis>& axes,
                                        std::optional<Axis> serial_axis,
                                        const std::vector<Stmt>& pre_stmts,
                                        const Stmt& main_stmt,
                                        const std::vector<Stmt>& post_stmts,
                                        const AliasMap& aliases, arith::Analyzer* analyzer,
                                        std::optional<int64_t> lanes, bool single,
                                        const std::vector<std::string>& base_notes,
                                        const PlanOptions& options,
                                        bool allow_flatten = true) {
  std::string flatten_word = single ? "nest" : "segment";
  SegmentSpecs specs = PrepareSegment(axes, serial_axis, pre_stmts, main_stmt, post_stmts,
                                      aliases, analyzer, lanes);
  std::vector<AccessInfo> all_infos = specs.AllInfos();
  std::vector<AccessInfo> load_infos = LoadInfos(axes, specs.AllSpecs(), analyzer);
  std::vector<std::string> diagnostics = base_notes;
  std::vector<std::string> axis_notes =
      AxisDiagnostics(axes, all_infos, load_infos, analyzer, specs.lanes);
  diagnostics.insert(diagnostics.end(), axis_notes.begin(), axis_notes.end());

  auto flattened = allow_flatten
                       ? TryFlatten(axes, pre_stmts, main_stmt, post_stmts, specs.lanes,
                                    analyzer)
                       : std::pair<std::optional<SegmentSpecs>, std::string>{
                             std::nullopt, std::string()};
  if (!flattened.second.empty()) diagnostics.push_back(flattened.second);
  if (flattened.first.has_value()) {
    const SegmentSpecs& flat_specs = *flattened.first;
    Axis flat_axis = flat_specs.axes[0];
    bool ok = true;
    VectorPlan plan;
    try {
      plan = flat_specs.plan_for(flat_axis, analyzer, options);
    } catch (const Unsupported& error) {
      diagnostics.push_back("flattened axis '" + flat_axis->name.operator std::string() +
                            "': " + error.message());
      ok = false;
    }
    if (ok) {
      std::vector<std::string> notes;
      notes.push_back("flattened " + flatten_word + " [" + JoinNames(axes) + "] -> '" +
                      flat_axis->name.operator std::string() + "' (extent " +
                      PrintExpr(flat_axis->extent) + ", full-lane chunks)");
      notes.insert(notes.end(), diagnostics.begin(), diagnostics.end());
      plan.CopyOnWrite()->diagnostics = ToStringArray(notes);
      NoteGuardSplit(plan);
      return {plan, true};
    }
  }

  std::vector<std::string> errors;
  // Specificity survives the per-axis fallback: the first classified reason
  // (ranked axes are tried in order) becomes the rejection code.
  std::string first_classified_code;
  std::vector<Axis> ranked = RankVectorAxes(axes, all_infos, load_infos, analyzer, specs.lanes);
  if (ranked.empty()) {
    throw Unsupported(
        "no vectorizable axis: every store is true-random (its index cannot be built in the index-vector channel) on all axes",
        diagnostics, "plan.no_vector_axis");
  }
  for (const Axis& axis : ranked) {
    VectorPlan plan;
    try {
      plan = specs.plan_for(axis, analyzer, options);
    } catch (const Unsupported& error) {
      // Per-axis failure reasons survive even on success (改进 7).
      errors.push_back("axis '" + axis->name.operator std::string() + "': " + error.message());
      if (first_classified_code.empty() &&
          error.code() != Unsupported::kDefaultCode) {
        first_classified_code = error.code();
      }
      diagnostics.push_back("axis '" + axis->name.operator std::string() + "': " +
                            error.message() + " -> fallback");
      for (const std::string& note : error.diagnostics()) diagnostics.push_back("  " + note);
      continue;
    }
    if (single) {
      diagnostics.push_back(
          "selected vector axis '" + axis->name.operator std::string() + "' (score=" +
          std::to_string(AxisScore(all_infos, load_infos, axis, analyzer, specs.lanes)) + ")");
    } else {
      diagnostics.push_back(
          "segment vector axis '" + axis->name.operator std::string() + "' (active=" +
          std::to_string(ActiveLanes(axis, analyzer, specs.lanes)) + "/" +
          std::to_string(specs.lanes) + ", 方案 §6 第三步)");
    }
    plan.CopyOnWrite()->diagnostics = ToStringArray(diagnostics);
    NoteGuardSplit(plan);
    return {plan, false};
  }
  std::string joined;
  for (size_t i = 0; i < errors.size(); ++i) joined += (i == 0 ? "" : "; ") + errors[i];
  std::string code =
      first_classified_code.empty() ? "plan.no_vector_axis" : first_classified_code;
  throw Unsupported("no vectorizable axis: " + joined, diagnostics, code);
}

/*! \brief Plan one serial chain segment (multi-segment path). */
std::pair<VectorPlan, bool> BuildSegment(const std::vector<Axis>& axes,
                                         const Axis& serial_axis,
                                         const std::vector<Stmt>& pre_stmts,
                                         const Stmt& main_stmt,
                                         const std::vector<Stmt>& post_stmts,
                                         const AliasMap& aliases, arith::Analyzer* analyzer,
                                         std::optional<int64_t> lanes,
                                         const PlanOptions& options,
                                         bool allow_flatten = true) {
  return PlanSegment(axes, std::optional<Axis>(serial_axis), pre_stmts, main_stmt, post_stmts,
                     aliases, analyzer, lanes, /*single=*/false, {}, options, allow_flatten);
}

/*!
 * \brief Multi-segment path for an expanded tile-op body.
 *
 * Every statement group of the parallel body becomes one segment: consecutive
 * non-loop statements form a store segment (axes = outer parallel axes), each
 * loop chain (parallel copy / serial reduction dim) forms a loop segment with
 * ``outer + chain`` axes. Statements that are siblings of the single innermost
 * serial loop stay in the same segment as pre/post, so e.g. a reduction's
 * identity initializer is folded into the accumulator register instead of
 * becoming a shared-memory write of its own.
 */
std::optional<VectorProgram> TryBuildSplitAllProgram(
    const Stmt& stmt, arith::Analyzer* analyzer, std::optional<int64_t> lanes,
    std::vector<std::string>* notes, const PlanOptions& options) {
  auto absorbed = AbsorbParallelChain(stmt, analyzer);
  const std::vector<Axis>& parallel_axes = absorbed.first;
  Stmt current = absorbed.second;
  const auto* seq = current.as<tirx::SeqStmtNode>();
  if (parallel_axes.empty() || seq == nullptr) return std::nullopt;

  struct SegmentSource {
    std::vector<Axis> axes;
    std::optional<Axis> serial_axis;
    std::vector<Stmt> pre;
    Stmt main;
    std::vector<Stmt> post;
  };
  AliasMap aliases = CollectAliases(current);

  // A single clean serial loop at the end of a loop chain with siblings before
  // it: the siblings become the segment's pre stores. That is how a reduction's
  // identity initializer stays in the same segment as the reduction and folds
  // into the accumulator register instead of a shared-memory write.
  auto split_trailing_serial = [&](SegmentSource* source) {
    const auto* body_seq = source->main.as<tirx::SeqStmtNode>();
    if (body_seq == nullptr) return;
    std::vector<size_t> loop_positions;
    for (size_t pos = 0; pos < body_seq->seq.size(); ++pos) {
      if (body_seq->seq[pos].as<tirx::ForNode>() != nullptr) loop_positions.push_back(pos);
    }
    if (loop_positions.size() != 1) return;
    const size_t pos = loop_positions[0];
    tirx::For loop = Downcast<tirx::For>(body_seq->seq[pos]);
    std::optional<int64_t> min = ConstInt(loop->min, analyzer);
    auto clean = loop->kind == ForKind::kSerial && min.has_value() && *min == 0
                     ? SerialIsClean(loop, analyzer)
                     : std::pair<bool, std::string>{false, ""};
    if (!clean.first) {
      if (!clean.second.empty()) RecordSerialReason(notes, loop, clean.second);
      return;
    }
    Axis axis = MakeAxis(loop->loop_var, loop->extent,
                         static_cast<int64_t>(source->axes.size()), ForKind::kSerial);
    source->axes.push_back(axis);
    source->serial_axis = axis;
    for (size_t i = 0; i < pos; ++i) source->pre.push_back(body_seq->seq[i]);
    for (size_t i = pos + 1; i < body_seq->seq.size(); ++i)
      source->post.push_back(body_seq->seq[i]);
    source->main = loop->body;
  };

  auto build_loop_source = [&](const Stmt& child) -> SegmentSource {
    auto chain = AbsorbLoopChain(child, analyzer,
                                 static_cast<int64_t>(parallel_axes.size()));
    if (!chain.has_value()) {
      const auto* loop = child.as<tirx::ForNode>();
      if (loop != nullptr && loop->kind == ForKind::kSerial) {
        RecordSerialReason(notes, Downcast<tirx::For>(child),
                           SerialIsClean(Downcast<tirx::For>(child), analyzer).second);
      }
      throw Unsupported(
          "tile-op segment rejected: a child loop is not a clean parallel/serial chain",
          *notes);
    }
    SegmentSource source;
    source.axes = parallel_axes;
    source.axes.insert(source.axes.end(), chain->first.begin(), chain->first.end());
    for (const Axis& axis : chain->first) {
      if (!axis->is_parallel) source.serial_axis = axis;
    }
    source.main = chain->second;
    split_trailing_serial(&source);
    return source;
  };

  // Identify the identity initializer pattern: every statement before the one
  // loop chain is a uniform literal store to the same address as a store inside
  // the chain. Such a body is one segment, so the initializer can be folded.
  auto identity_init_of = [&](const Stmt& init, const Stmt& chain) -> bool {
    const auto* store = init.as<tirx::BufferStoreNode>();
    if (store == nullptr) return false;
    if (store->value.as<FloatImmNode>() == nullptr &&
        store->value.as<IntImmNode>() == nullptr) {
      return false;
    }
    ffi::Array<PrimExpr> indices;
    for (const PrimExpr& index : store->indices) indices.push_back(ApplyAliases(index, aliases));
    bool matched = false;
    tirx::PostOrderVisit(chain, [&](const ffi::ObjectRef& node) {
      const auto* other = node.as<tirx::BufferStoreNode>();
      if (other == nullptr || matched) return;
      if (!SameBuffer(other->buffer, store->buffer)) return;
      if (other->indices.size() != indices.size()) return;
      tirx::ExprDeepEqual equal;
      for (size_t i = 0; i < indices.size(); ++i) {
        if (!equal(ApplyAliases(other->indices[i], aliases), indices[i])) return;
      }
      matched = true;
    });
    return matched;
  };

  std::vector<SegmentSource> sources;
  std::vector<size_t> loop_children;
  for (size_t i = 0; i < seq->seq.size(); ++i) {
    if (seq->seq[i].as<tirx::ForNode>() != nullptr) loop_children.push_back(i);
  }
  bool merged_init = false;
  if (loop_children.size() == 1 && loop_children[0] + 1 == seq->seq.size() &&
      loop_children[0] > 0) {
    merged_init = true;
    for (size_t i = 0; i < loop_children[0] && merged_init; ++i) {
      merged_init = identity_init_of(seq->seq[i], seq->seq[loop_children[0]]);
    }
    if (merged_init) {
      SegmentSource source = build_loop_source(seq->seq[loop_children[0]]);
      for (size_t i = 0; i < loop_children[0]; ++i) source.pre.push_back(seq->seq[i]);
      sources.push_back(std::move(source));
    }
  }

  if (!merged_init) {
    std::vector<Stmt> pending;
    auto flush_pending = [&]() {
      if (pending.empty()) return;
      SegmentSource source;
      source.main = pending.size() == 1 ? pending.front() : SeqStmts(pending);
      source.axes = parallel_axes;
      sources.push_back(std::move(source));
      pending.clear();
    };

    // Two adjacent pure-parallel loop chains over the same domain and disjoint
    // buffers become one nest: their chunk loops then interleave (one loop with
    // both bodies) instead of running back to back, which gives the scheduler
    // independent load/store pairs to overlap.
    // One traversal per child collects everything the merge decision needs:
    // the buffers it touches and whether it carries a serial loop. Walking the
    // same chain three times (buffers, serial flag, then again for the merge)
    // was the dominant cost of this loop.
    struct ChildFacts {
      std::set<const tirx::VarNode*> buffers;
      bool has_serial{false};
    };
    auto collect_facts = [](const Stmt& stmt) {
      ChildFacts facts;
      tirx::PostOrderVisit(stmt, [&](const ffi::ObjectRef& node) {
        if (const auto* load = node.as<tirx::BufferLoadNode>()) {
          facts.buffers.insert(load->buffer->data.get());
        } else if (const auto* store = node.as<tirx::BufferStoreNode>()) {
          facts.buffers.insert(store->buffer->data.get());
        } else if (const auto* loop = node.as<tirx::ForNode>()) {
          if (loop->kind == ForKind::kSerial) facts.has_serial = true;
        }
      });
      return facts;
    };
    std::function<bool(const Stmt&, const Stmt&)> same_loop_shape =
        [&](const Stmt& lhs, const Stmt& rhs) -> bool {
      const auto* a = lhs.as<tirx::ForNode>();
      const auto* b = rhs.as<tirx::ForNode>();
      if (a == nullptr || b == nullptr) return a == nullptr && b == nullptr;
      tirx::ExprDeepEqual equal;
      if (a->kind != b->kind || !equal(a->min, b->min) || !equal(a->extent, b->extent)) {
        return false;
      }
      return same_loop_shape(a->body, b->body);
    };
    // Rebuild ``lhs`` with ``rhs`` appended to its innermost body; both chains
    // have the same shape, so the recursion bottoms out together.
    std::function<Stmt(const Stmt&, const Stmt&)> merge_chain_bodies =
        [&](const Stmt& lhs, const Stmt& rhs) -> Stmt {
      const auto* a = lhs.as<tirx::ForNode>();
      const auto* b = rhs.as<tirx::ForNode>();
      if (a == nullptr || b == nullptr) return SeqStmts({lhs, rhs});
      return tirx::For(a->loop_var, a->min, a->extent, a->kind,
                       merge_chain_bodies(a->body, b->body));
    };
    auto zip_loop_vars = [&](const Stmt& lhs, const Stmt& rhs) {
      std::map<const tirx::VarNode*, PrimExpr> mapping;
      const tirx::ForNode* a = lhs.as<tirx::ForNode>();
      const tirx::ForNode* b = rhs.as<tirx::ForNode>();
      while (a != nullptr && b != nullptr) {
        mapping[b->loop_var.get()] = a->loop_var;
        a = a->body.as<tirx::ForNode>();
        b = b->body.as<tirx::ForNode>();
      }
      return mapping;
    };

    std::optional<Stmt> previous_chain;
    ChildFacts previous_facts;
    for (const Stmt& child : seq->seq) {
      if (child.as<tirx::ForNode>() == nullptr) {
        pending.push_back(child);
        previous_chain.reset();
        continue;
      }
      ChildFacts facts = collect_facts(child);
      if (previous_chain.has_value() && !facts.has_serial &&
          !previous_facts.has_serial && same_loop_shape(*previous_chain, child)) {
        bool disjoint = true;
        for (const tirx::VarNode* buffer : facts.buffers) {
          if (previous_facts.buffers.count(buffer) != 0) {
            disjoint = false;
            break;
          }
        }
        if (disjoint) {
          Stmt merged = merge_chain_bodies(
              *previous_chain, SubstituteVarsStmt(child, zip_loop_vars(*previous_chain, child)));
          sources.back() = build_loop_source(merged);
          previous_chain = merged;
          previous_facts.buffers.insert(facts.buffers.begin(), facts.buffers.end());
          continue;
        }
      }
      flush_pending();
      sources.push_back(build_loop_source(child));
      previous_chain = child;
      previous_facts = std::move(facts);
    }
    flush_pending();
  }
  if (sources.empty()) return std::nullopt;
  std::vector<VectorPlan> segments;
  for (size_t i = 0; i < sources.size(); ++i) {
    try {
      // Flatten only loop-free parallel segments: it merges the whole
      // contiguous domain into a full-lane axis, which is the only way a
      // contiguous access avoids the masked gather. A segment with a serial
      // axis must keep its axis table intact (the serial dim is the
      // accumulator's inner axis).
      std::pair<VectorPlan, bool> segment =
          PlanSegment(sources[i].axes, sources[i].serial_axis, sources[i].pre,
                      sources[i].main, sources[i].post, aliases, analyzer, lanes,
                      /*single=*/false, {}, options,
                      /*allow_flatten=*/!sources[i].serial_axis.has_value());
      segments.push_back(segment.first);
    } catch (const Unsupported& error) {
      notes->push_back("segment " + std::to_string(i) + ": " + error.message());
      for (const std::string& note : error.diagnostics()) notes->push_back("  " + note);
      throw Unsupported("tile-op segment planning failed: " + error.message(), *notes);
    }
  }

  // Only the outer parallel prefix in front of the first consumed vector axis
  // may be hoisted; each segment carries the full axis table and the emitter
  // skips hoisted axes by identity.
  std::vector<int64_t> consumed;
  for (const VectorPlan& segment : segments) {
    if (segment->vector_axis->is_parallel) consumed.push_back(segment->vector_axis->index);
  }
  std::optional<int64_t> first_consumed;
  for (int64_t index : consumed) {
    first_consumed = first_consumed.has_value() ? std::min(*first_consumed, index) : index;
  }
  ffi::Array<Axis> hoisted;
  for (const Axis& axis : parallel_axes) {
    if (!first_consumed.has_value() || axis->index < *first_consumed)
      hoisted.push_back(axis);
  }
  VectorProgram program = MakeVectorProgram(hoisted, ToArray(segments));
  ffi::Array<ffi::String> diagnostics;
  diagnostics.push_back("multi-segment program: " + std::to_string(segments.size()) +
                        " segments, source order preserved (tile-op expansion)");
  program.CopyOnWrite()->diagnostics = diagnostics;
  return program;
}

/*! \brief Multi-segment path: several sibling clean serial loops. */
std::optional<VectorProgram> TryBuildProgram(const Stmt& stmt, arith::Analyzer* analyzer,
                                             std::optional<int64_t> lanes,
                                             std::vector<std::string>* notes,
                                             const PlanOptions& options,
                                             bool split_all = false,
                                             bool allow_flatten = true) {
  if (split_all) return TryBuildSplitAllProgram(stmt, analyzer, lanes, notes, options);
  auto absorbed = AbsorbParallelChain(stmt, analyzer);
  const std::vector<Axis>& parallel_axes = absorbed.first;
  Stmt current = absorbed.second;
  if (parallel_axes.empty() || current.as<tirx::SeqStmtNode>() == nullptr) {
    return std::nullopt;
  }
  const auto* seq = current.as<tirx::SeqStmtNode>();

  struct Item {
    bool is_segment{false};
    std::pair<std::vector<Axis>, Stmt> chain;  // valid when is_segment
    Stmt stmt;                                 // valid otherwise
  };
  std::vector<Item> items;
  for (const Stmt& child : seq->seq) {
    if (child.as<tirx::ForNode>() != nullptr) {
      auto chain = AbsorbSerialChain(child, analyzer,
                                     static_cast<int64_t>(parallel_axes.size()));
      if (!chain.has_value()) {
        // Keep the concrete reason instead of the generic "nested serial loop".
        RecordSerialReason(notes, Downcast<tirx::For>(child),
                           SerialIsClean(Downcast<tirx::For>(child), analyzer).second);
        return std::nullopt;
      }
      Item item;
      item.is_segment = true;
      item.chain = std::move(*chain);
      items.push_back(std::move(item));
    } else {
      Item item;
      item.is_segment = false;
      item.stmt = child;
      items.push_back(std::move(item));
    }
  }
  std::vector<size_t> seg_positions;
  for (size_t i = 0; i < items.size(); ++i) {
    if (items[i].is_segment) seg_positions.push_back(i);
  }
  if (seg_positions.size() < 2) return std::nullopt;

  AliasMap aliases = CollectAliases(current);
  std::vector<VectorPlan> segments;
  bool flattened_any = false;
  int64_t previous = -1;
  for (size_t order = 0; order < seg_positions.size(); ++order) {
    size_t pos = seg_positions[order];
    std::vector<Stmt> pre;
    for (int64_t k = previous + 1; k < static_cast<int64_t>(pos); ++k) {
      pre.push_back(items[static_cast<size_t>(k)].stmt);
    }
    const std::vector<Axis>& serial_axes = items[pos].chain.first;
    const Stmt& body = items[pos].chain.second;
    std::vector<Stmt> post;
    if (order + 1 == seg_positions.size()) {
      for (size_t k = pos + 1; k < items.size(); ++k) post.push_back(items[k].stmt);
    }
    std::vector<Axis> segment_axes = parallel_axes;
    segment_axes.insert(segment_axes.end(), serial_axes.begin(), serial_axes.end());
    std::pair<VectorPlan, bool> segment;
    try {
      segment = BuildSegment(segment_axes, serial_axes.back(), pre, body, post, aliases,
                            analyzer, lanes, options);
    } catch (const Unsupported& error) {
      notes->push_back("segment " + std::to_string(order) + " (T.serial '" +
                       serial_axes.back()->name.operator std::string() + "'): " +
                       error.message());
      for (const std::string& note : error.diagnostics()) notes->push_back("  " + note);
      return std::nullopt;
    }
    segments.push_back(segment.first);
    flattened_any = flattened_any || segment.second;
    previous = static_cast<int64_t>(pos);
  }

  // The program layer may only hoist axes in front of the first axis some
  // segment consumes as its vector axis; a flattened segment consumes the whole
  // parallel prefix.
  std::vector<int64_t> consumed;
  for (const VectorPlan& segment : segments) {
    if (segment->vector_axis->is_parallel) consumed.push_back(segment->vector_axis->index);
  }
  std::optional<int64_t> first_consumed;
  for (int64_t index : consumed) {
    first_consumed = first_consumed.has_value() ? std::min(*first_consumed, index) : index;
  }
  ffi::Array<Axis> hoisted;
  if (!flattened_any) {
    for (const Axis& axis : parallel_axes) {
      if (!first_consumed.has_value() || axis->index < *first_consumed) hoisted.push_back(axis);
    }
  }
  VectorProgram program = MakeVectorProgram(hoisted, ToArray(segments));
  ffi::Array<ffi::String> diagnostics;
  diagnostics.push_back("multi-segment program: " + std::to_string(segments.size()) +
                        " segments, source order preserved" +
                        (hoisted.empty()
                             ? ", flattened/consumed parallel axes are wrapped by each segment"
                             : ""));
  program.CopyOnWrite()->diagnostics = diagnostics;
  return program;
}

}  // namespace

VectorProgram BuildPlan(const tirx::Stmt& stmt, std::optional<int64_t> lanes,
                        const PlanOptions& options) {
  arith::Analyzer analyzer;
  // P1 hands over the arithmetic contract of the region. Enter it before any
  // zero-min / injectivity / lane proof so a VF-local tl.assume is usable here
  // (the fields used to be write-only).
  for (const PrimExpr& assumption : options.assumptions) {
    if (assumption.defined())
      analyzer.EnterConstraint(assumption);
  }
  for (const auto& binding : options.bindings) {
    // Call-valued bindings are not value aliases (same rule as the old
    // RestoreBindings helper); binding them would confuse the analyzer.
    if (binding.first.defined() && binding.second.defined() &&
        binding.second.as<tirx::CallNode>() == nullptr) {
      analyzer.Bind(binding.first, binding.second);
    }
  }
  std::vector<std::string> notes;
  std::pair<tirx::Stmt, bool> expanded = ExpandParallelTileOps(stmt, &analyzer);
  const bool expanded_tileops = expanded.second;
  std::optional<VectorProgram> program =
      TryBuildProgram(expanded.first, &analyzer, lanes, &notes, options,
                      /*split_all=*/expanded_tileops,
                      /*allow_flatten=*/!expanded_tileops);
  if (program.has_value()) return *program;

  VectorPlan plan;
  bool flattened = false;
  try {
    NestSplit split = SplitNest(expanded.first, &analyzer, &notes);
    std::vector<Stmt> pieces = split.pre;
    pieces.push_back(split.main);
    pieces.insert(pieces.end(), split.post.begin(), split.post.end());
    AliasMap aliases = CollectAliases(SeqStmts(pieces));
    auto planned = PlanSegment(split.axes, split.serial_axis, split.pre, split.main,
                               split.post, aliases, &analyzer, lanes, /*single=*/true, notes,
                               options, /*allow_flatten=*/!expanded_tileops);
    plan = planned.first;
    flattened = planned.second;
  } catch (const Unsupported& error) {
    // The fallback path rejects here (typically "nested serial loop"): merge the
    // segment-level reasons tried by the multi-segment path instead of dropping
    // them (defect B). ``_plan_segment`` already merged ``notes`` when possible.
    bool already_merged = !error.diagnostics().empty() && notes.size() <= error.diagnostics().size() &&
                          std::equal(notes.begin(), notes.end(), error.diagnostics().begin());
    if (!notes.empty() && !already_merged) {
      std::string joined;
      for (size_t i = 0; i < notes.size(); ++i) joined += (i == 0 ? "" : "; ") + notes[i];
      std::vector<std::string> merged = notes;
      merged.insert(merged.end(), error.diagnostics().begin(), error.diagnostics().end());
      throw Unsupported(error.message() + " [" + joined + "]", merged, error.code());
    }
    throw;
  }

  ffi::Array<Axis> program_axes;
  if (!flattened) {
    for (const Axis& axis : OuterAxes(plan->axes, plan->vector_axis)) {
      program_axes.push_back(axis);
    }
  }
  ffi::Array<VectorPlan> segments;
  segments.push_back(plan);
  VectorProgram result = MakeVectorProgram(program_axes, segments);
  std::string detail = flattened
                           ? "flattened axis '" + plan->vector_axis->name.operator std::string() + "'"
                           : "vector axis '" + plan->vector_axis->name.operator std::string() + "'";
  ffi::Array<ffi::String> diagnostics;
  diagnostics.push_back("single-segment program: 1 segment, " + detail);
  result.CopyOnWrite()->diagnostics = diagnostics;
  return result;
}

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm
