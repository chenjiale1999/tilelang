/*!
 * \file group_plan.cc
 * \brief Builders and FFI reflection registration of the typed group plan.
 */
#include "group_plan.h"

#include <tvm/ffi/reflection/registry.h>

#include <utility>

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
