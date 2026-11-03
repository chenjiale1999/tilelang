/*!
 * \file vsimd_program.h
 * \brief Typed C++ reader and invariant checker for the P1 VSimdProgram.
 *
 * P1 writes the annotation as a structured Map<String, Any>; P2 should consume
 * this reader instead of re-parsing raw TIR or raw annotation maps.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_VSIMD_PROGRAM_H_
#define TVM_TL_ASCEND_TRANSFORM_VSIMD_PROGRAM_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/tirx/stmt.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

struct VSimdRegion;

/*! \brief One VSimdProgram item as consumed by P2/P3. */
struct VSimdProgramItem {
  int index{-1};
  int span_begin{-1};
  int span_end{-1};
  int group{-1};
  std::string kind;
  std::string analysis_status;
  std::string status;
  std::string group_status;
  std::string reason;
  ffi::Array<tirx::Buffer> read_buffers;
  ffi::Array<tirx::Buffer> write_buffers;
  /*! \brief Full item map, including kind-specific payload. */
  ffi::Map<ffi::String, ffi::Any> meta;
};

struct VSimdProgramGroup {
  int id{-1};
  std::string status;
  std::vector<int> items;
};

struct VSimdProgramCapture {
  tirx::Buffer buffer;
  std::string name;
  std::string storage;
};

struct VSimdProgramFragment {
  tirx::Buffer buffer;
  std::string name;
  int definition_vf{-1};
  std::string definition_scope;
  std::vector<int> use_vfs;
  bool cross_vf{false};
  bool escapes_vf{false};
  bool uses_outside_vf{false};
};

/*! \brief Typed view of one `tl.vsimd.program` annotation. */
struct VSimdProgram {
  int version{-1};
  int vf_index{-1};
  std::string provenance;
  /*! \brief Number of canonical VF-body items; -1 when absent (older P1). */
  int item_count{-1};
  /*! \brief Canonical VF-body fingerprint; valid when has_body_fingerprint. */
  uint64_t body_fingerprint{0};
  bool has_body_fingerprint{false};
  std::vector<VSimdProgramItem> items;
  std::vector<VSimdProgramGroup> groups;
  std::vector<VSimdProgramCapture> captures;
  std::vector<VSimdProgramFragment> fragments;
  std::vector<ffi::Map<ffi::String, ffi::Any>> assumptions;
  std::vector<ffi::Map<ffi::String, ffi::Any>> attrs;
  ffi::Map<ffi::String, ffi::Any> raw;
};

/*!
 * \brief Read and verify one `tl.vsimd.program` map.
 *
 * Returns false and fills `error` on malformed/inconsistent input. P2 should
 * reject the whole VF on a read error rather than partially consuming it.
 */
TVM_DLL bool ReadVSimdProgram(
    const ffi::Map<ffi::String, ffi::Any> &map, VSimdProgram *out,
    std::string *error);

/*! \brief Verify invariants of a parsed program. */
TVM_DLL bool VerifyVSimdProgram(const VSimdProgram &program,
                                std::string *error);

/*!
 * \brief Decode one REGION item's ``region`` payload into the existing
 *        hardware-independent ``VSimdRegion`` struct.
 *
 * The returned region has vf_index/nest_index filled from the program/item.
 */
TVM_DLL bool ReadVSimdRegion(const ffi::Map<ffi::String, ffi::Any> &item_map,
                             int vf_index, VSimdRegion *out,
                             std::string *error);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VSIMD_PROGRAM_H_
