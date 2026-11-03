/*!
 * \file emit.h
 * \brief Emission layer: ``VectorPlan`` / ``VectorProgram`` -> ``T.simd.*`` TIR.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_EMIT_H_
#define TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_EMIT_H_

#include <tvm/tirx/stmt.h>

#include <map>
#include <string>
#include <vector>

#include "model.h"

namespace tvm {
namespace tl {
namespace polyhedral {

/*!
 * \brief Per-VF fragment register file (design §2.6.5).
 *
 * One ``T.alloc_fragment`` becomes ``ceil(size / lanes)`` ``local.var`` vector
 * buffers, keyed by the fragment's **data Var** (never its name). The file lives
 * for one ``SIMD_VF`` block, so values written by one nest are visible to the
 * next one without any memory round-trip.
 */
class FragmentRegisterFile {
 public:
  struct Entry {
    tirx::Var data;
    std::string name;
    int64_t slots{0};
    int64_t lanes{0};
    std::string dtype;
    std::vector<tirx::Buffer> regs;  // one ``local.var`` vector per slot
  };

  /*! \brief Build a file from plan metadata (``frag_<name>_<id>_r<k>``). */
  static FragmentRegisterFile Build(const ffi::Array<FragmentReg>& regs);

  bool empty() const { return entries_.empty(); }
  const std::vector<Entry>& entries() const { return entries_; }
  /*! \brief Entry of a fragment data Var (nullptr when not registered). */
  const Entry* Find(const tirx::Var& data) const;
  /*! \brief Register buffer of one slot (nullptr when out of range). */
  const tirx::Buffer* Slot(const tirx::Var& data, int64_t slot) const;
  /*! \brief All register buffers, in slot order (the VF allocations). */
  std::vector<tirx::Buffer> Buffers() const;

 private:
  std::vector<Entry> entries_;
  std::map<const tirx::VarNode*, size_t> index_;
};

/*!
 * \brief Emit a plan (or program) as ``T.simd.*`` TIR.
 *
 * The plan is read-only; every node / ordering rule matches the Python emitter
 * node for node (SSA names included). ``registers`` is the enclosing VF's
 * register file; when it is null the plan is standalone (offline ``emit_plan``)
 * and the file is rebuilt from the plan metadata, with the register
 * declarations placed in the emitted prologue.
 */
TVM_DLL tirx::Stmt EmitPlan(const ffi::ObjectRef& plan,
                            FragmentRegisterFile* registers = nullptr);

}  // namespace polyhedral
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_POLYHEDRAL_EMIT_H_
