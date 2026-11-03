#include "vsimd_handoff.h"

#include <tvm/ffi/any.h>
#include <tvm/tirx/op.h>

#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Any;
using ffi::Map;
using ffi::String;

namespace {

constexpr const char* kScopeKey = "tl.simdvf_scope";

static bool IsTransparentAttr(const std::string& key) {
  return key == kScopeKey || key == tirx::attr::tilelang_assume ||
         key == "tl.assume" || key == "tl.buffer_version" ||
         key == "tl.manual_multi_buffer";
}

static void FlattenInto(
    const Stmt& stmt, std::vector<Stmt>* out,
    std::vector<Map<String, Any>>* assumption_meta,
    std::vector<Map<String, Any>>* metadata_attrs) {
  if (const auto* seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt& child : seq->seq)
      FlattenInto(child, out, assumption_meta, metadata_attrs);
    return;
  }
  if (const auto* attr = stmt.as<AttrStmtNode>()) {
    if (attr->attr_key == kScopeKey) {
      FlattenInto(attr->body, out, assumption_meta, metadata_attrs);
      return;
    }
    if (attr->attr_key == tirx::attr::tilelang_assume ||
        attr->attr_key == "tl.assume") {
      size_t meta_index = 0;
      if (assumption_meta != nullptr) {
        Map<String, Any> meta;
        meta.Set("condition", Any(Downcast<PrimExpr>(attr->node)));
        meta.Set("anchor", Any(Integer(out->size())));
        meta_index = assumption_meta->size();
        assumption_meta->push_back(meta);
      }
      if (attr->body.defined())
        FlattenInto(attr->body, out, assumption_meta, metadata_attrs);
      if (assumption_meta != nullptr)
        (*assumption_meta)[meta_index].Set("anchor_end",
                                           Any(Integer(out->size())));
      return;
    }
    if (attr->attr_key == "tl.buffer_version" ||
        attr->attr_key == "tl.manual_multi_buffer") {
      size_t meta_index = 0;
      if (metadata_attrs != nullptr) {
        Map<String, Any> meta;
        meta.Set("key", Any(String(attr->attr_key)));
        meta.Set("node", Any(attr->node));
        meta.Set("value", Any(attr->value));
        meta.Set("anchor", Any(Integer(out->size())));
        meta_index = metadata_attrs->size();
        metadata_attrs->push_back(meta);
      }
      if (attr->body.defined())
        FlattenInto(attr->body, out, assumption_meta, metadata_attrs);
      if (metadata_attrs != nullptr)
        (*metadata_attrs)[meta_index].Set("anchor_end",
                                          Any(Integer(out->size())));
      return;
    }
  }
  out->push_back(stmt);
}

static Stmt ReplaceInto(const Stmt& stmt,
                        const std::map<int, Stmt>& replacements,
                        int* cursor) {
  if (const auto* seq = stmt.as<SeqStmtNode>()) {
    ffi::Array<Stmt> children;
    for (const Stmt& child : seq->seq)
      children.push_back(ReplaceInto(child, replacements, cursor));
    return SeqStmt(children);
  }
  if (const auto* attr = stmt.as<AttrStmtNode>()) {
    if (IsTransparentAttr(attr->attr_key)) {
      Stmt body = ReplaceInto(attr->body, replacements, cursor);
      return AttrStmt(attr->node, attr->attr_key, attr->value, std::move(body));
    }
  }
  const int index = (*cursor)++;
  auto it = replacements.find(index);
  if (it != replacements.end())
    return it->second;
  return stmt;
}

}  // namespace

uint64_t FingerprintVSimdBody(const std::vector<Stmt>& stmts) {
  // FNV-1a over the structural hash of every canonical item. This is only used
  // to detect in-process body changes between P1, P2 and P3.
  uint64_t hash = 1469598103934665603ull;
  for (const Stmt& stmt : stmts) {
    hash ^= static_cast<uint64_t>(ffi::StructuralHash()(stmt));
    hash *= 1099511628211ull;
  }
  return hash;
}

VSimdFlattenedBody FlattenVSimdBody(const Stmt& body) {
  VSimdFlattenedBody flat;
  FlattenInto(body, &flat.stmts, &flat.assumptions, &flat.attrs);
  flat.fingerprint = FingerprintVSimdBody(flat.stmts);
  return flat;
}

Stmt ReplaceVSimdSpans(const Stmt& body,
                       const std::map<int, Stmt>& replacements) {
  int cursor = 0;
  return ReplaceInto(body, replacements, &cursor);
}

}  // namespace tl
}  // namespace tvm
