/*!
 * \file model.cc
 * \brief Plan data structure builders, FFI registration and the accumulator criterion.
 */
#include "model.h"

#include <tvm/ffi/reflection/registry.h>

#include <string>
#include <vector>

#include "primitives.h"

namespace tvm {
namespace tl {
namespace polyhedral {

StorePlan MakeStorePlan(StoreSpec spec, AccessInfo access, ValueNode value,
                        ffi::Optional<ValueNode> mask, AccessClass kind,
                        ffi::Optional<ValueNode> index) {
  auto node = ffi::make_object<StorePlanNode>();
  node->spec = std::move(spec);
  node->access = std::move(access);
  node->value = std::move(value);
  node->mask = std::move(mask);
  node->kind_value = AccessClassName(kind);
  node->index = std::move(index);
  return StorePlan(std::move(node));
}

ReduceInfo MakeReduceInfo(Axis axis, std::string op, std::string recipe,
                          ValueNode varying, bool accumulate) {
  auto node = ffi::make_object<ReduceInfoNode>();
  node->axis = std::move(axis);
  node->op = std::move(op);
  node->recipe = std::move(recipe);
  if (varying.defined()) node->varying = ffi::Optional<ValueNode>(varying);
  node->accumulate = accumulate;
  return ReduceInfo(std::move(node));
}

VectorPlan MakeVectorPlan(ffi::Array<Axis> axes, Axis vector_axis, int64_t lanes,
                          ffi::Optional<int64_t> chunks, ffi::Optional<int64_t> tail) {
  auto node = ffi::make_object<VectorPlanNode>();
  node->axes = std::move(axes);
  node->vector_axis = std::move(vector_axis);
  node->lanes = lanes;
  node->chunks = std::move(chunks);
  node->tail = std::move(tail);
  return VectorPlan(std::move(node));
}

VectorProgram MakeVectorProgram(ffi::Array<Axis> axes, ffi::Array<VectorPlan> segments) {
  auto node = ffi::make_object<VectorProgramNode>();
  node->axes = std::move(axes);
  node->segments = std::move(segments);
  return VectorProgram(std::move(node));
}

ReduceInfo ReduceOf(const StorePlan& store) {
  if (!store.defined() || !store->reduce.has_value()) return ReduceInfo();
  return Downcast<ReduceInfo>(*store->reduce);
}

ReduceInfo ReduceOf(const VectorPlan& plan) {
  if (!plan.defined() || !plan->reduce.has_value()) return ReduceInfo();
  return Downcast<ReduceInfo>(*plan->reduce);
}

bool IsPrivateScope(const tirx::Buffer& buffer) {
  std::string scope = buffer.scope().operator std::string();
  return scope == "local" || scope.compare(0, 6, "local.") == 0;
}

ValueNode AccumulatorCandidate(const StorePlan& store, const VectorPlan& plan,
                               arith::Analyzer* analyzer) {
  if (store->mask.has_value()) return ValueNode();
  const tirx::Buffer& buffer = store->spec->buffer;
  bool is_private = IsPrivateScope(buffer);
  std::string kind = store->kind_value;
  if (kind == AccessClassName(AccessClass::kUniform)) {
    if (!is_private) return ValueNode();
  } else if (kind != AccessClassName(AccessClass::kContiguous)) {
    return ValueNode();
  }

  std::vector<Axis> inner;
  for (const Axis& axis : plan->axes) {
    if (axis->index > plan->vector_axis->index) inner.push_back(axis);
  }
  if (inner.empty()) return ValueNode();
  for (const Axis& axis : inner) {
    std::optional<int64_t> delta = ConstInt(
        FiniteDifference(store->access->offset, axis->var, analyzer), analyzer);
    if (!delta.has_value() || *delta != 0) return ValueNode();
  }

  ValueNode old = SameAddressLoad(store->value, buffer, store->spec->indices);
  if (!old.defined()) return ValueNode();
  int64_t same_buffer = 0;
  for (const StorePlan& other : plan->stores) {
    if (SameBuffer(other->spec->buffer, buffer)) ++same_buffer;
  }
  if (same_buffer != 1) return ValueNode();
  for (const StorePlan& other : plan->stores) {
    if (other.get() == store.get()) continue;
    if (ReadsOtherAddress(other->value, buffer, store->spec->indices)) return ValueNode();
  }
  for (const StorePlan& pre : plan->pre_stores) {
    if (pre->mask.has_value() && SameBuffer(pre->spec->buffer, buffer) &&
        SameIndices(pre->spec->indices, store->spec->indices)) {
      return ValueNode();
    }
  }
  if (!is_private) return old;

  // A private scalar also needs a per-lane consumer after the loop: without one,
  // the final value`s visibility depends on which lane writes last.
  std::vector<StorePlan> consumers;
  for (const StorePlan& other : plan->stores) {
    if (other.get() != store.get() && ReadsBuffer(other->value, buffer)) {
      consumers.push_back(other);
    }
  }
  for (const StorePlan& other : plan->post_stores) {
    if (other.get() != store.get() && ReadsBuffer(other->value, buffer)) {
      consumers.push_back(other);
    }
  }
  if (consumers.empty()) return ValueNode();
  for (const StorePlan& other : consumers) {
    if (other->kind_value == AccessClassName(AccessClass::kUniform)) return ValueNode();
  }
  for (const StorePlan& post : plan->post_stores) {
    if (post->kind_value == AccessClassName(AccessClass::kUniform) &&
        !post->value->is_uniform && SameBuffer(post->spec->buffer, buffer)) {
      return ValueNode();
    }
  }
  return old;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  AxisNode::RegisterReflection();
  AccessInfoNode::RegisterReflection();
  StoreSpecNode::RegisterReflection();
  ValueNodeNode::RegisterReflection();
  StorePlanNode::RegisterReflection();
  ReduceInfoNode::RegisterReflection();
  VectorPlanNode::RegisterReflection();
  VectorProgramNode::RegisterReflection();
}

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm
