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
  VectorPlan plan_for(const Axis& vector_axis, arith::Analyzer* analyzer) const;
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
                            arith::Analyzer* analyzer);

VectorPlan SegmentSpecs::plan_for(const Axis& vector_axis,
                                  arith::Analyzer* analyzer) const {
  return BuildPlanForAxis(axes, serial_axis, pre_specs, pre_infos, main_specs, main_infos,
                          post_specs, post_infos, vector_axis, lanes, analyzer);
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
  std::vector<int64_t> extents;
  for (const Axis& axis : axes) {
    std::optional<int64_t> extent = ConstInt(axis->extent, analyzer);
    if (!extent.has_value() || *extent < 2 || (*extent & (*extent - 1)) != 0) {
      return std::nullopt;
    }
    extents.push_back(*extent);
  }
  int64_t total = 1;
  for (int64_t extent : extents) total *= extent;

  std::string name = "k_";
  for (size_t i = 0; i < axes.size(); ++i) {
    name += (i == 0 ? "" : "_") + axes[i]->name.operator std::string();
  }
  tirx::Var var(name, DataType::Int(32));

  FlattenResult result;
  result.flat_axis = MakeAxis(var, IntImm32(total), 0, ForKind::kParallel);
  result.total = total;
  int64_t suffix = 1;
  int64_t count = static_cast<int64_t>(axes.size());
  for (int64_t position = 0; position < count; ++position) {
    // Right-to-left decomposition; the outermost axis is NOT taken modulo, since
    // ``(k // suffix) % E`` would collapse to ``k % (prod E)`` and lose Delta==1.
    const Axis& axis = axes[static_cast<size_t>(count - 1 - position)];
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
  std::optional<int64_t> innermost = ConstInt(axes.back()->extent, analyzer);
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
          name + "'");
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
    if (pre->kind_value == AccessClassName(AccessClass::kUniform) &&
        !pre->mask.has_value()) {
      for (const StorePlan& store : plan->stores) {
        ReduceInfo reduce = ReduceOf(store);
        if (!reduce.defined() || !reduce->accumulate) continue;
        if (!(SameBuffer(pre->spec->buffer, store->spec->buffer) &&
              SameIndices(pre->spec->indices, store->spec->indices))) {
          continue;
        }
        std::string elem = ElemDtype(PyBufferDtypeName(store->spec->buffer));
        PrimExpr identity = ReduceIdentity(reduce->recipe.operator std::string(), elem);
        if (IsSameConst(pre->spec->value, identity)) {
          reduce.CopyOnWrite()->accumulate = false;
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

std::pair<ValueNode, int64_t> StripReduce(const ValueNode& value, const std::string& op,
                                          const StoreSpec& site) {
  std::string scalar_recipe;
  ReduceScalarRecipeFor(op, &scalar_recipe);
  std::set<std::string> spine = {op, scalar_recipe};
  const tirx::Buffer& buffer = site->buffer;
  int64_t accumulated = 0;
  bool repeated = false;

  std::function<ValueNode(const ValueNode&)> strip = [&](const ValueNode& node) -> ValueNode {
    if (node->recipe == "uniform_load" && node->access.has_value() &&
        SameBuffer((*node->access)->buffer, buffer) &&
        SameIndices((*node->access)->indices, site->indices)) {
      ++accumulated;
      if (accumulated > 1) repeated = true;
      return ValueNode();
    }
    if (spine.count(node->recipe.operator std::string()) != 0 && node->args.size() == 2) {
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
                      "carries the tail predicate, so the guard would be silently dropped (方案 §2 Case II / R3)");
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
  const std::string recipe = node->recipe.operator std::string();
  if (recipe == "vload" || recipe == "vload_strided" || recipe == "gather") {
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
 * whose extent divides evenly (no tail); everything else keeps falling back to
 * the legacy lowering.
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
    const std::string kind = store->kind_value.operator std::string();
    if (kind != AccessClassName(AccessClass::kContiguous) &&
        kind != AccessClassName(AccessClass::kUniform)) {
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

VectorPlan BuildPlanForAxis(const std::vector<Axis>& axes,
                            std::optional<Axis> serial_axis,
                            const std::vector<StoreSpec>& pre_specs,
                            const std::vector<AccessInfo>& pre_infos,
                            const std::vector<StoreSpec>& main_specs,
                            const std::vector<AccessInfo>& main_infos,
                            const std::vector<StoreSpec>& post_specs,
                            const std::vector<AccessInfo>& post_infos,
                            const Axis& vector_axis, int64_t lanes,
                            arith::Analyzer* analyzer) {
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
  std::optional<int64_t> chunks;
  std::optional<int64_t> tail;
  if (extent.has_value()) {
    chunks = (*extent + lanes - 1) / lanes;
    tail = *extent % lanes;
  }

  ValueContext ctx;
  ctx.axes = axes;
  ctx.vector_axis = vector_axis;
  ctx.analyzer = analyzer;
  ctx.lanes = lanes;

  VectorPlan plan = MakeVectorPlan(ToArray(axes), vector_axis, lanes,
                                   chunks, tail);

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
    } else if (ratio == 4) {
      store.CopyOnWrite()->store_lane_mode = ffi::String("pk_quarter");
    } else {
      store.CopyOnWrite()->store_lane_mode = ffi::String("full");
    }
    if (ratio > 1) {
      store.CopyOnWrite()->store_pack_bits = ffi::Optional<int64_t>(2048 / lanes);
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
  plan.CopyOnWrite()->stores = stores;
  plan.CopyOnWrite()->pre_stores = pre_stores;
  plan.CopyOnWrite()->post_stores = post_stores;

  // Mixed element widths (Stage 1): a chunk narrower than some store register
  // is only proved for the straight-line element-wise subset.
  const bool partial_lanes = HasPartialLaneStore(plan);
  if (partial_lanes) ValidatePartialLaneSegment(plan, extent, lanes);

  // Along a serial axis, pre/post are siblings of the chunk loop and must be
  // lane-uniform; along a parallel axis they live inside the chunk body.
  if (along_serial) {
    const std::vector<std::pair<std::string, ffi::Array<StorePlan>>> groups = {
        {"before", plan->pre_stores}, {"after", plan->post_stores}};
    for (const auto& group : groups) {
      for (const StorePlan& store : group.second) {
        if (store->kind_value != AccessClassName(AccessClass::kUniform)) {
          throw Unsupported(
              "a statement " + group.first +
              " the serial loop must be uniform along vector axis '" +
              vector_axis->name.operator std::string() + "' (got " +
              store->kind_value.operator std::string() + ") (方案 §6 第二步)");
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
        plan.CopyOnWrite()->stores.Set(i, store);
      }
    }
    for (const StorePlan& store : plan->stores) {
      bool promoted = store->promote.has_value();
      if (store->kind_value == AccessClassName(AccessClass::kUniform) &&
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
  if (reduce.defined() && partial_lanes) {
    throw Unsupported(
        "a reduction cannot share a segment with a wider-lane store (mixed element widths would "
        "fold the unused lanes into the accumulator); split the nest or stage an explicit cast");
  }
  if (reduce.defined()) plan.CopyOnWrite()->reduce = ffi::Optional<ffi::ObjectRef>(reduce);
  if (reduce.defined()) {
    // The reduction path hands pre/post to the outer-store emitter, which needs
    // lane-uniform statements (R7: close the build-ok/emit-Unsupported gap).
    const std::vector<std::pair<std::string, ffi::Array<StorePlan>>> groups = {
        {"before", plan->pre_stores}, {"after", plan->post_stores}};
    for (const auto& group : groups) {
      for (const StorePlan& store : group.second) {
        if (store->kind_value != AccessClassName(AccessClass::kUniform)) {
          throw Unsupported(
              "a statement " + group.first +
              " the serial loop must be uniform along the reduction vector axis '" +
              vector_axis->name.operator std::string() + "' (got " +
              store->kind_value.operator std::string() + ") (方案 §6 第二步)");
        }
      }
    }
    FoldIdentityInit(plan);
  }
  DropDeadPreStores(plan, analyzer);

  // Defect A0: a partial reduction on an absorbed serial axis has no vector form.
  if (!reduce.defined()) {
    for (const StorePlan& store : plan->stores) {
      if (store->kind_value != AccessClassName(AccessClass::kUniform) ||
          store->value->is_uniform) {
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
  plan.CopyOnWrite()->guard_boundaries = guard_boundaries;
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
                                        const std::vector<std::string>& base_notes) {
  std::string flatten_word = single ? "nest" : "segment";
  SegmentSpecs specs = PrepareSegment(axes, serial_axis, pre_stmts, main_stmt, post_stmts,
                                      aliases, analyzer, lanes);
  std::vector<AccessInfo> all_infos = specs.AllInfos();
  std::vector<AccessInfo> load_infos = LoadInfos(axes, specs.AllSpecs(), analyzer);
  std::vector<std::string> diagnostics = base_notes;
  std::vector<std::string> axis_notes =
      AxisDiagnostics(axes, all_infos, load_infos, analyzer, specs.lanes);
  diagnostics.insert(diagnostics.end(), axis_notes.begin(), axis_notes.end());

  auto flattened = TryFlatten(axes, pre_stmts, main_stmt, post_stmts, specs.lanes, analyzer);
  if (!flattened.second.empty()) diagnostics.push_back(flattened.second);
  if (flattened.first.has_value()) {
    const SegmentSpecs& flat_specs = *flattened.first;
    Axis flat_axis = flat_specs.axes[0];
    bool ok = true;
    VectorPlan plan;
    try {
      plan = flat_specs.plan_for(flat_axis, analyzer);
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
  std::vector<Axis> ranked = RankVectorAxes(axes, all_infos, load_infos, analyzer, specs.lanes);
  if (ranked.empty()) {
    throw Unsupported(
        "no vectorizable axis: every store is true-random (its index cannot be built in the index-vector channel) on all axes",
        diagnostics);
  }
  for (const Axis& axis : ranked) {
    VectorPlan plan;
    try {
      plan = specs.plan_for(axis, analyzer);
    } catch (const Unsupported& error) {
      // Per-axis failure reasons survive even on success (改进 7).
      errors.push_back("axis '" + axis->name.operator std::string() + "': " + error.message());
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
  throw Unsupported("no vectorizable axis: " + joined, diagnostics);
}

/*! \brief Plan one serial chain segment (multi-segment path). */
std::pair<VectorPlan, bool> BuildSegment(const std::vector<Axis>& axes,
                                         const Axis& serial_axis,
                                         const std::vector<Stmt>& pre_stmts,
                                         const Stmt& main_stmt,
                                         const std::vector<Stmt>& post_stmts,
                                         const AliasMap& aliases, arith::Analyzer* analyzer,
                                         std::optional<int64_t> lanes) {
  return PlanSegment(axes, std::optional<Axis>(serial_axis), pre_stmts, main_stmt, post_stmts,
                     aliases, analyzer, lanes, /*single=*/false, {});
}

/*! \brief Multi-segment path: several sibling clean serial loops. */
std::optional<VectorProgram> TryBuildProgram(const Stmt& stmt, arith::Analyzer* analyzer,
                                             std::optional<int64_t> lanes,
                                             std::vector<std::string>* notes) {
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
                            analyzer, lanes);
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

VectorProgram BuildPlan(const tirx::Stmt& stmt, std::optional<int64_t> lanes) {
  arith::Analyzer analyzer;
  std::vector<std::string> notes;
  std::optional<VectorProgram> program = TryBuildProgram(stmt, &analyzer, lanes, &notes);
  if (program.has_value()) return *program;

  VectorPlan plan;
  bool flattened = false;
  try {
    NestSplit split = SplitNest(stmt, &analyzer, &notes);
    std::vector<Stmt> pieces = split.pre;
    pieces.push_back(split.main);
    pieces.insert(pieces.end(), split.post.begin(), split.post.end());
    AliasMap aliases = CollectAliases(SeqStmts(pieces));
    auto planned = PlanSegment(split.axes, split.serial_axis, split.pre, split.main,
                               split.post, aliases, &analyzer, lanes, /*single=*/true, notes);
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
      throw Unsupported(error.message() + " [" + joined + "]", merged);
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
