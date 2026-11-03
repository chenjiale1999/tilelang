/*!
 * \file vsimd_handoff.h
 * \brief Shared VF-body flattening / fingerprint / span replacement for P1-P2-P3.
 *
 * P1, P2 and P3 must agree on what an "item" is. This header is the single
 * implementation of that rule: transparent wrappers (tl.simdvf_scope,
 * tl.assume, tl.buffer_version, tl.manual_multi_buffer) and SeqStmt do not
 * occupy items; every other top-level statement occupies exactly one item.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_VSIMD_HANDOFF_H_
#define TVM_TL_ASCEND_TRANSFORM_VSIMD_HANDOFF_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/tirx/stmt.h>

#include <cstdint>
#include <map>
#include <vector>

namespace tvm {
namespace tl {

/*! \brief Flattened top-level items of one SIMD_VF body. */
struct VSimdFlattenedBody {
  std::vector<tirx::Stmt> stmts;
  std::vector<ffi::Map<ffi::String, ffi::Any>> assumptions;
  std::vector<ffi::Map<ffi::String, ffi::Any>> attrs;
  /*! \brief Stable in-process fingerprint of `stmts` (order-sensitive). */
  uint64_t fingerprint{0};
};

/*! \brief Flatten a SIMD_VF body using the canonical P1 item rule. */
TVM_DLL VSimdFlattenedBody FlattenVSimdBody(const tirx::Stmt& body);

/*! \brief Fingerprint a flattened statement sequence. */
TVM_DLL uint64_t FingerprintVSimdBody(const std::vector<tirx::Stmt>& stmts);

/*!
 * \brief Replace the i-th canonical top-level item with replacements[i].
 *
 * Transparent wrappers are preserved; a replacement is applied to the child
 * statement inside the wrapper, not to the wrapper itself. Items not present in
 * `replacements` are returned verbatim.
 */
TVM_DLL tirx::Stmt ReplaceVSimdSpans(
    const tirx::Stmt& body, const std::map<int, tirx::Stmt>& replacements);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_VSIMD_HANDOFF_H_
