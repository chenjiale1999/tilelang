/*!
 * \file group_plan.cc
 * \brief Builders and FFI reflection registration of the typed group plan.
 */
#include "group_plan.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/node_functor.h>
#include <tvm/script/printer/ir_docsifier.h>

#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace polyhedral {

VSimdCopyPlan MakeVSimdCopyPlan(std::string kind, std::string src_scope,
                                std::string dst_scope,
                                ffi::Array<PrimExpr> src_extents,
                                ffi::Array<PrimExpr> dst_extents,
                                VectorProgram program) {
  auto node = ffi::make_object<VSimdCopyPlanNode>();
  node->kind = ffi::String(std::move(kind));
  node->src_scope = ffi::String(std::move(src_scope));
  node->dst_scope = ffi::String(std::move(dst_scope));
  node->src_extents = std::move(src_extents);
  node->dst_extents = std::move(dst_extents);
  node->program = std::move(program);
  return VSimdCopyPlan(std::move(node));
}

VSimdReducePlan MakeVSimdReducePlan(std::string reduce_kind, int64_t dim,
                                    bool clear,
                                    ffi::Optional<VectorProgram> init,
                                    VectorProgram program) {
  auto node = ffi::make_object<VSimdReducePlanNode>();
  node->reduce_kind = ffi::String(std::move(reduce_kind));
  node->dim = dim;
  node->clear = clear;
  node->init = std::move(init);
  node->program = std::move(program);
  return VSimdReducePlan(std::move(node));
}

VSimdItemPlan MakeVSimdRegionItemPlan(int64_t item_index, int64_t span_begin,
                                      int64_t span_end, std::string kind,
                                      VectorProgram program) {
  auto node = ffi::make_object<VSimdItemPlanNode>();
  node->item_index = item_index;
  node->span_begin = span_begin;
  node->span_end = span_end;
  node->kind = ffi::String(std::move(kind));
  node->plan_kind = ffi::String("REGION");
  node->region_plan = ffi::Optional<VectorProgram>(std::move(program));
  return VSimdItemPlan(std::move(node));
}

VSimdItemPlan MakeVSimdCopyItemPlan(int64_t item_index, int64_t span_begin,
                                    int64_t span_end, VSimdCopyPlan plan) {
  auto node = ffi::make_object<VSimdItemPlanNode>();
  node->item_index = item_index;
  node->span_begin = span_begin;
  node->span_end = span_end;
  node->kind = ffi::String("COPY");
  node->plan_kind = ffi::String("COPY");
  node->copy_plan = ffi::Optional<VSimdCopyPlan>(std::move(plan));
  return VSimdItemPlan(std::move(node));
}

VSimdItemPlan MakeVSimdReduceItemPlan(int64_t item_index, int64_t span_begin,
                                      int64_t span_end, VSimdReducePlan plan) {
  auto node = ffi::make_object<VSimdItemPlanNode>();
  node->item_index = item_index;
  node->span_begin = span_begin;
  node->span_end = span_end;
  node->kind = ffi::String("REDUCE");
  node->plan_kind = ffi::String("REDUCE");
  node->reduce_plan = ffi::Optional<VSimdReducePlan>(std::move(plan));
  return VSimdItemPlan(std::move(node));
}

VSimdGroupPlan MakeVSimdGroupPlan(int64_t group_id, std::string status) {
  auto node = ffi::make_object<VSimdGroupPlanNode>();
  node->group_id = group_id;
  node->status = ffi::String(std::move(status));
  return VSimdGroupPlan(std::move(node));
}

VSimdGroupPlan MakeRejectedGroupPlan(int64_t group_id, std::string reason,
                                     std::vector<std::string> diagnostics) {
  auto node = ffi::make_object<VSimdGroupPlanNode>();
  node->group_id = group_id;
  node->status = ffi::String("REJECTED");
  node->reason = ffi::String(std::move(reason));
  std::vector<ffi::String> notes;
  notes.reserve(diagnostics.size());
  for (std::string& note : diagnostics)
    notes.push_back(ffi::String(std::move(note)));
  node->diagnostics = ffi::Array<ffi::String>(notes.begin(), notes.end());
  return VSimdGroupPlan(std::move(node));
}

VSimdGroupPlanSet MakeVSimdGroupPlanSet(int64_t vf_index,
                                        std::vector<VSimdGroupPlan> groups) {
  auto node = ffi::make_object<VSimdGroupPlanSetNode>();
  node->version = 1;
  node->vf_index = vf_index;
  node->groups = ffi::Array<VSimdGroupPlan>(groups.begin(), groups.end());
  node->fragment_mode = ffi::String("none");
  return VSimdGroupPlanSet(std::move(node));
}

namespace {

using script::printer::AccessPath;
using script::printer::DictDoc;
using script::printer::Doc;
using script::printer::ExprDoc;
using script::printer::IRDocsifier;
using script::printer::ListDoc;
using script::printer::LiteralDoc;

/*
 * Without these printers the script printer treats the typed carrier as an
 * opaque object and prints ``metadata["tl.polyhedral.VSimdGroupPlanSet"][0]``;
 * the rejection reason then never shows up in an IR dump. Rendering the plan
 * as a plain dict keeps ``tl.vsimd.group_plan`` readable in place, exactly like
 * ``tl.vsimd.program``. This is diagnostic only: P3 always consumes the typed
 * object, never the printed text.
 */

ExprDoc StrDoc(const ffi::String& value) {
  return LiteralDoc::Str(value, std::nullopt);
}

ExprDoc IntDoc(int64_t value) {
  return LiteralDoc::Int(value, std::nullopt);
}

ExprDoc DictOf(std::vector<std::pair<const char*, ExprDoc>> fields) {
  ffi::Array<ExprDoc> keys;
  ffi::Array<ExprDoc> values;
  for (auto& field : fields) {
    keys.push_back(StrDoc(field.first));
    values.push_back(std::move(field.second));
  }
  return DictDoc(keys, values);
}

ExprDoc StrListDoc(const ffi::Array<ffi::String>& items) {
  ffi::Array<ExprDoc> elements;
  for (const ffi::String& item : items) elements.push_back(StrDoc(item));
  return ListDoc(elements);
}

/*! \brief Compact, bounded rendering of a ``VectorProgram``. */
ExprDoc ProgramDoc(const VectorProgram& program) {
  if (!program.defined()) return LiteralDoc::None(std::nullopt);
  ffi::Array<ExprDoc> segments;
  for (const VectorPlan& segment : program->segments) {
    std::string axis = segment->vector_axis.defined()
                           ? std::string(segment->vector_axis->name)
                           : std::string();
    segments.push_back(DictOf({
        {"lanes", IntDoc(segment->lanes)},
        {"vector_axis", StrDoc(axis)},
        {"stores", IntDoc(static_cast<int64_t>(segment->stores.size()))},
        {"pre_stores", IntDoc(static_cast<int64_t>(segment->pre_stores.size()))},
        {"post_stores", IntDoc(static_cast<int64_t>(segment->post_stores.size()))},
    }));
  }
  return DictOf({
      {"segments", ListDoc(segments)},
      {"diagnostics", StrListDoc(program->diagnostics)},
  });
}

ExprDoc ExtentsDoc(const ffi::Array<PrimExpr>& extents, const AccessPath& path,
                   const IRDocsifier& d) {
  ffi::Array<ExprDoc> elements;
  for (size_t i = 0; i < extents.size(); ++i) {
    elements.push_back(d->AsDoc<ExprDoc>(extents[i], path->ArrayItem(i)));
  }
  return ListDoc(elements);
}

ExprDoc CopyPlanDoc(const VSimdCopyPlan& plan, const AccessPath& path,
                    const IRDocsifier& d) {
  return DictOf({
      {"kind", StrDoc(plan->kind)},
      {"src_scope", StrDoc(plan->src_scope)},
      {"dst_scope", StrDoc(plan->dst_scope)},
      {"src_extents", ExtentsDoc(plan->src_extents, path->Attr("src_extents"), d)},
      {"dst_extents", ExtentsDoc(plan->dst_extents, path->Attr("dst_extents"), d)},
      {"program", ProgramDoc(plan->program)},
  });
}

ExprDoc ReducePlanDoc(const VSimdReducePlan& plan) {
  return DictOf({
      {"reduce_kind", StrDoc(plan->reduce_kind)},
      {"dim", IntDoc(plan->dim)},
      {"clear", LiteralDoc::Boolean(plan->clear, std::nullopt)},
      {"init", plan->init.has_value() ? ProgramDoc(*plan->init)
                                      : LiteralDoc::None(std::nullopt)},
      {"program", ProgramDoc(plan->program)},
  });
}

ExprDoc ItemPlanDoc(const VSimdItemPlan& plan, const AccessPath& path,
                    const IRDocsifier& d) {
  std::vector<std::pair<const char*, ExprDoc>> fields = {
      {"item_index", IntDoc(plan->item_index)},
      {"span_begin", IntDoc(plan->span_begin)},
      {"span_end", IntDoc(plan->span_end)},
      {"kind", StrDoc(plan->kind)},
      {"plan_kind", StrDoc(plan->plan_kind)},
  };
  if (plan->region_plan.has_value())
    fields.emplace_back("region_plan", ProgramDoc(*plan->region_plan));
  if (plan->copy_plan.has_value())
    fields.emplace_back("copy_plan",
                        CopyPlanDoc(*plan->copy_plan, path->Attr("copy_plan"), d));
  if (plan->reduce_plan.has_value())
    fields.emplace_back("reduce_plan", ReducePlanDoc(*plan->reduce_plan));
  return DictOf(std::move(fields));
}

ExprDoc GroupPlanDoc(const VSimdGroupPlan& plan, const AccessPath& path,
                     const IRDocsifier& d) {
  ffi::Array<ExprDoc> items;
  for (size_t i = 0; i < plan->items.size(); ++i) {
    items.push_back(
        d->AsDoc<ExprDoc>(plan->items[i], path->Attr("items")->ArrayItem(i)));
  }
  return DictOf({
      {"group_id", IntDoc(plan->group_id)},
      {"status", StrDoc(plan->status)},
      {"reason", StrDoc(plan->reason)},
      {"diagnostics", StrListDoc(plan->diagnostics)},
      {"items", ListDoc(items)},
  });
}

ExprDoc FragmentRegsDoc(const FragmentRegisterMap& regs) {
  ffi::Array<ExprDoc> entries;
  for (const FragmentReg& reg : regs->entries) {
    entries.push_back(DictOf({
        {"name", StrDoc(reg->name)},
        {"slots", IntDoc(reg->slots)},
        {"lanes", IntDoc(reg->lanes)},
        {"dtype", StrDoc(reg->dtype)},
    }));
  }
  return ListDoc(entries);
}

ExprDoc GroupPlanSetDoc(const VSimdGroupPlanSet& plan, const AccessPath& path,
                        const IRDocsifier& d) {
  ffi::Array<ExprDoc> groups;
  for (size_t i = 0; i < plan->groups.size(); ++i) {
    groups.push_back(
        d->AsDoc<ExprDoc>(plan->groups[i], path->Attr("groups")->ArrayItem(i)));
  }
  std::vector<std::pair<const char*, ExprDoc>> fields = {
      {"version", IntDoc(plan->version)},
      {"vf_index", IntDoc(plan->vf_index)},
      {"groups", ListDoc(groups)},
      {"fragment_mode", StrDoc(plan->fragment_mode)},
  };
  if (plan->fragment_regs.has_value())
    fields.emplace_back("fragment_regs", FragmentRegsDoc(*plan->fragment_regs));
  if (plan->fragment_reason.size() != 0)
    fields.emplace_back("fragment_reason", StrDoc(plan->fragment_reason));
  return DictOf(std::move(fields));
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<VSimdGroupPlanSet>(
        "", [](VSimdGroupPlanSet plan, AccessPath path, IRDocsifier d) -> Doc {
          return GroupPlanSetDoc(plan, path, d);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<VSimdGroupPlan>(
        "", [](VSimdGroupPlan plan, AccessPath path, IRDocsifier d) -> Doc {
          return GroupPlanDoc(plan, path, d);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<VSimdItemPlan>(
        "", [](VSimdItemPlan plan, AccessPath path, IRDocsifier d) -> Doc {
          return ItemPlanDoc(plan, path, d);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<VSimdCopyPlan>(
        "", [](VSimdCopyPlan plan, AccessPath path, IRDocsifier d) -> Doc {
          return CopyPlanDoc(plan, path, d);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<VSimdReducePlan>(
        "", [](VSimdReducePlan plan, AccessPath, IRDocsifier) -> Doc {
          return ReducePlanDoc(plan);
        });

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  VSimdCopyPlanNode::RegisterReflection();
  VSimdReducePlanNode::RegisterReflection();
  VSimdItemPlanNode::RegisterReflection();
  VSimdGroupPlanNode::RegisterReflection();
  VSimdGroupPlanSetNode::RegisterReflection();
}

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm
