/*!
 * \file group_plan.h
 * \brief Typed P2 -> P3 carrier: the planned groups of one ``SIMD_VF``.
 *
 * P2 writes one :class:`VSimdGroupPlanSet` into
 * ``SBlock("SIMD_VF").annotations["tl.vsimd.group_plan"]``. Every planned item
 * carries exactly one typed sub-plan (region / copy / reduce), so P3 never
 * re-parses a ``Map<String, Any>`` and tests can assert on real fields.
 *
 * The carrier lives in the polyhedral namespace because its payload is the
 * planner's own :class:`VectorProgram`; the P1 reader (``vsimd_program.h``)
 * stays independent of it.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_GROUP_PLAN_H_
#define TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_GROUP_PLAN_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>

#include <string>
#include <vector>

#include "model.h"

namespace tvm {
namespace tl {
namespace polyhedral {

/*!
 * \brief Phase B copy plan: one UB<->UB elementwise / broadcast copy.
 *
 * ``program`` is the plan of the synthesized
 * ``for idx in T.Parallel(dst_extents): dst[dst_indices + idx] =
 * src[src_indices + idx]`` nest; ``kind`` records which shape it was
 * (``ND`` or ``BROADCAST``).
 */
class VSimdCopyPlanNode : public ffi::Object {
 public:
  ffi::String kind;
  ffi::String src_scope;
  ffi::String dst_scope;
  ffi::Array<PrimExpr> src_extents;
  ffi::Array<PrimExpr> dst_extents;
  VectorProgram program;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.polyhedral.VSimdCopyPlan",
                                    VSimdCopyPlanNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VSimdCopyPlanNode>()
        .def_ro("kind", &VSimdCopyPlanNode::kind)
        .def_ro("src_scope", &VSimdCopyPlanNode::src_scope)
        .def_ro("dst_scope", &VSimdCopyPlanNode::dst_scope)
        .def_ro("src_extents", &VSimdCopyPlanNode::src_extents)
        .def_ro("dst_extents", &VSimdCopyPlanNode::dst_extents)
        .def_ro("program", &VSimdCopyPlanNode::program);
  }
};

class VSimdCopyPlan : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(VSimdCopyPlan, ffi::ObjectRef,
                                             VSimdCopyPlanNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VSimdCopyPlanNode);
};

/*!
 * \brief Phase B reduce plan: one ``T.reduce_<kind>`` over a single dimension.
 *
 * ``program`` reduces the source into the destination. ``clear == false``
 * accumulates into the destination's old value (``dst = op(dst, src)``);
 * ``clear == true`` first writes ``init`` (``dst = src[k = 0]``) and then
 * accumulates the remaining elements, so no dtype-specific identity value is
 * ever synthesized.
 */
class VSimdReducePlanNode : public ffi::Object {
 public:
  ffi::String reduce_kind;
  int64_t dim{0};
  bool clear{true};
  ffi::Optional<VectorProgram> init;
  VectorProgram program;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.polyhedral.VSimdReducePlan",
                                    VSimdReducePlanNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VSimdReducePlanNode>()
        .def_ro("reduce_kind", &VSimdReducePlanNode::reduce_kind)
        .def_ro("dim", &VSimdReducePlanNode::dim)
        .def_ro("clear", &VSimdReducePlanNode::clear)
        .def_ro("init", &VSimdReducePlanNode::init)
        .def_ro("program", &VSimdReducePlanNode::program);
  }
};

class VSimdReducePlan : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(VSimdReducePlan, ffi::ObjectRef,
                                             VSimdReducePlanNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VSimdReducePlanNode);
};

/*! \brief Plan of one P1 item (exactly one of the three plans is defined). */
class VSimdItemPlanNode : public ffi::Object {
 public:
  int64_t item_index{-1};
  int64_t span_begin{-1};
  int64_t span_end{-1};
  ffi::String kind;       // REGION / COPY / REDUCE
  ffi::String plan_kind;  // REGION / COPY / REDUCE
  ffi::Optional<VectorProgram> region_plan;
  ffi::Optional<VSimdCopyPlan> copy_plan;
  ffi::Optional<VSimdReducePlan> reduce_plan;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.polyhedral.VSimdItemPlan",
                                    VSimdItemPlanNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VSimdItemPlanNode>()
        .def_ro("item_index", &VSimdItemPlanNode::item_index)
        .def_ro("span_begin", &VSimdItemPlanNode::span_begin)
        .def_ro("span_end", &VSimdItemPlanNode::span_end)
        .def_ro("kind", &VSimdItemPlanNode::kind)
        .def_ro("plan_kind", &VSimdItemPlanNode::plan_kind)
        .def_ro("region_plan", &VSimdItemPlanNode::region_plan)
        .def_ro("copy_plan", &VSimdItemPlanNode::copy_plan)
        .def_ro("reduce_plan", &VSimdItemPlanNode::reduce_plan);
  }
};

class VSimdItemPlan : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(VSimdItemPlan, ffi::ObjectRef,
                                             VSimdItemPlanNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VSimdItemPlanNode);
};

/*! \brief One P1 group's atomic plan. */
class VSimdGroupPlanNode : public ffi::Object {
 public:
  int64_t group_id{-1};
  ffi::String status;  // PLANNED / REJECTED
  ffi::String reason;
  ffi::Array<ffi::String> diagnostics;
  ffi::Array<VSimdItemPlan> items;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.polyhedral.VSimdGroupPlan",
                                    VSimdGroupPlanNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VSimdGroupPlanNode>()
        .def_ro("group_id", &VSimdGroupPlanNode::group_id)
        .def_ro("status", &VSimdGroupPlanNode::status)
        .def_ro("reason", &VSimdGroupPlanNode::reason)
        .def_ro("diagnostics", &VSimdGroupPlanNode::diagnostics)
        .def_ro("items", &VSimdGroupPlanNode::items);
  }
};

class VSimdGroupPlan : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(VSimdGroupPlan, ffi::ObjectRef,
                                             VSimdGroupPlanNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VSimdGroupPlanNode);
};

/*!
 * \brief ``tl.vsimd.group_plan`` payload: every planned group of one VF.
 *
 * ``fragment_regs`` is the VF-level register map decided by P2 (empty when the
 * VF stays in memory mode). P3 hands the same file to every item emission.
 */
class VSimdGroupPlanSetNode : public ffi::Object {
 public:
  int64_t version{1};
  int64_t vf_index{-1};
  ffi::Array<VSimdGroupPlan> groups;
  ffi::Optional<FragmentRegisterMap> fragment_regs;
  ffi::String fragment_mode;  // none / register / rejected
  ffi::String fragment_reason;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.polyhedral.VSimdGroupPlanSet",
                                    VSimdGroupPlanSetNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VSimdGroupPlanSetNode>()
        .def_ro("version", &VSimdGroupPlanSetNode::version)
        .def_ro("vf_index", &VSimdGroupPlanSetNode::vf_index)
        .def_ro("groups", &VSimdGroupPlanSetNode::groups)
        .def_ro("fragment_regs", &VSimdGroupPlanSetNode::fragment_regs)
        .def_ro("fragment_mode", &VSimdGroupPlanSetNode::fragment_mode)
        .def_ro("fragment_reason", &VSimdGroupPlanSetNode::fragment_reason);
  }
};

class VSimdGroupPlanSet : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(VSimdGroupPlanSet, ffi::ObjectRef,
                                             VSimdGroupPlanSetNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VSimdGroupPlanSetNode);
};

/*! \brief Builders (used by P2). */
TVM_DLL VSimdCopyPlan MakeVSimdCopyPlan(std::string kind, std::string src_scope,
                                        std::string dst_scope,
                                        ffi::Array<PrimExpr> src_extents,
                                        ffi::Array<PrimExpr> dst_extents,
                                        VectorProgram program);
TVM_DLL VSimdReducePlan MakeVSimdReducePlan(std::string reduce_kind, int64_t dim,
                                            bool clear,
                                            ffi::Optional<VectorProgram> init,
                                            VectorProgram program);
TVM_DLL VSimdItemPlan MakeVSimdRegionItemPlan(int64_t item_index,
                                              int64_t span_begin,
                                              int64_t span_end, std::string kind,
                                              VectorProgram program);
TVM_DLL VSimdItemPlan MakeVSimdCopyItemPlan(int64_t item_index,
                                            int64_t span_begin,
                                            int64_t span_end,
                                            VSimdCopyPlan plan);
TVM_DLL VSimdItemPlan MakeVSimdReduceItemPlan(int64_t item_index,
                                              int64_t span_begin,
                                              int64_t span_end,
                                              VSimdReducePlan plan);
TVM_DLL VSimdGroupPlan MakeVSimdGroupPlan(int64_t group_id, std::string status);
TVM_DLL VSimdGroupPlan MakeRejectedGroupPlan(
    int64_t group_id, std::string reason,
    std::vector<std::string> diagnostics = {});
TVM_DLL VSimdGroupPlanSet MakeVSimdGroupPlanSet(int64_t vf_index,
                                                std::vector<VSimdGroupPlan> groups);

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_GROUP_PLAN_H_
