/*!
 * \file vsimd_payload_read.h
 * \brief P2-only decoders for the kind-specific payloads of a P1 item.
 *
 * P1 owns the annotation schema (``../vsimd_program.h``); these readers turn the
 * REGION/COPY/REDUCE payload maps into the planner's typed inputs. They live on
 * the P2 side so ``vsimd_program.*`` stays the producer schema plus its
 * structural verifier.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_VSIMD_PAYLOAD_READ_H_
#define TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_VSIMD_PAYLOAD_READ_H_

#include <string>
#include <vector>

#include "../vsimd_program.h"
#include "../vsimd_region.h"

namespace tvm {
namespace tl {
namespace polyhedral {

/*! \brief Decoded logical source/destination of a COPY or REDUCE item. */
struct VSimdCopyReducePayload {
  tirx::Buffer src;
  tirx::Buffer dst;
  std::string src_scope;
  std::string dst_scope;
  /*! \brief Buffer indices of each tile region (empty when not recorded). */
  std::vector<PrimExpr> src_indices;
  std::vector<PrimExpr> src_extents;
  std::vector<PrimExpr> dst_indices;
  std::vector<PrimExpr> dst_extents;
};

/*! \brief Decoded ``REDUCE`` item: the logical payload plus reduce semantics. */
struct VSimdReducePayload : VSimdCopyReducePayload {
  std::string reduce_kind;
  int dim{-1};
  bool clear{true};
};

/*!
 * \brief Decode one REGION item's ``region`` payload into the existing
 *        hardware-independent ``VSimdRegion`` struct.
 *
 * The returned region has vf_index/nest_index filled from the program/item.
 */
TVM_DLL bool ReadVSimdRegion(const ffi::Map<ffi::String, ffi::Any> &item_map,
                             int vf_index, VSimdRegion *out,
                             std::string *error);

/*!
 * \brief Decode one COPY item (``src/dst/src_region/dst_region``).
 *
 * Fails when the payload is incomplete, so the caller rejects the whole group
 * instead of planning a partial copy.
 */
TVM_DLL bool ReadVSimdCopy(const VSimdProgramItem &item,
                           VSimdCopyReducePayload *out, std::string *error);

/*!
 * \brief Decode one REDUCE item (``src/dst/src_region/dst_region`` + semantics).
 */
TVM_DLL bool ReadVSimdReduce(const VSimdProgramItem &item,
                             VSimdReducePayload *out, std::string *error);

} // namespace polyhedral
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_VSIMD_PAYLOAD_READ_H_
