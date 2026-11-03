#include "vsimd_program.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

static bool Fail(std::string *error, const std::string &message) {
  if (error != nullptr)
    *error = message;
  return false;
}

static bool ReadInt(const Map<String, Any> &map, const char *key, int *out,
                    std::string *error) {
  auto value = map.Get(key);
  if (!value)
    return Fail(error, std::string("missing '") + key + "'");
  auto imm = value.value().as<IntImm>();
  if (!imm)
    return Fail(error, std::string("'") + key + "' is not an integer");
  *out = static_cast<int>(imm.value()->value);
  return true;
}

static bool ReadString(const Map<String, Any> &map, const char *key,
                       std::string *out, std::string *error) {
  auto value = map.Get(key);
  if (!value)
    return Fail(error, std::string("missing '") + key + "'");
  auto str = value.value().as<String>();
  if (!str)
    return Fail(error, std::string("'") + key + "' is not a string");
  *out = std::string(str.value());
  return true;
}

static bool ReadBool(const Map<String, Any> &map, const char *key, bool *out,
                     std::string *error) {
  auto value = map.Get(key);
  if (!value)
    return Fail(error, std::string("missing '") + key + "'");
  auto imm = value.value().as<IntImm>();
  if (!imm)
    return Fail(error, std::string("'") + key + "' is not a bool");
  *out = imm.value()->value != 0;
  return true;
}

static bool ReadArrayOfMaps(const Map<String, Any> &map, const char *key,
                            std::vector<Map<String, Any>> *out,
                            std::string *error) {
  auto value = map.Get(key);
  if (!value)
    return Fail(error, std::string("missing '") + key + "'");
  auto arr = value.value().as<Array<Map<String, Any>>>();
  if (!arr)
    return Fail(error, std::string("'") + key + "' is not an array of maps");
  out->assign(arr.value().begin(), arr.value().end());
  return true;
}

static bool ReadIntArray(const Map<String, Any> &map, const char *key,
                         std::vector<int> *out, std::string *error) {
  auto value = map.Get(key);
  if (!value)
    return Fail(error, std::string("missing '") + key + "'");
  auto arr = value.value().as<Array<Integer>>();
  if (!arr)
    return Fail(error, std::string("'") + key + "' is not an integer array");
  out->clear();
  for (const Integer &item : arr.value())
    out->push_back(static_cast<int>(item->value));
  return true;
}

static bool ReadBuffers(const Map<String, Any> &map, const char *key,
                        Array<tirx::Buffer> *out, std::string *error) {
  auto value = map.Get(key);
  if (!value)
    return Fail(error, std::string("missing '") + key + "'");
  auto arr = value.value().as<Array<tirx::Buffer>>();
  if (!arr)
    return Fail(error, std::string("'") + key + "' is not a buffer array");
  *out = arr.value();
  return true;
}

static bool IsMetadataKind(const std::string &kind) {
  return kind == "BIND" || kind == "ALLOC" || kind == "DECL";
}

} // namespace

bool ReadVSimdProgram(const Map<String, Any> &map, VSimdProgram *out,
                      std::string *error) {
  if (out == nullptr)
    return Fail(error, "null output");
  VSimdProgram program;
  program.raw = map;
  if (!ReadInt(map, "version", &program.version, error) ||
      !ReadInt(map, "vf_index", &program.vf_index, error) ||
      !ReadString(map, "provenance", &program.provenance, error))
    return false;

  // Optional body identity fields. They are absent in annotations produced
  // before the handoff fingerprint was introduced.
  if (auto value = map.Get("item_count")) {
    auto imm = value.value().as<IntImm>();
    if (!imm)
      return Fail(error, "'item_count' is not an integer");
    program.item_count = static_cast<int>(imm.value()->value);
  }
  if (auto value = map.Get("body_fingerprint")) {
    auto imm = value.value().as<IntImm>();
    if (!imm)
      return Fail(error, "'body_fingerprint' is not an integer");
    program.body_fingerprint = static_cast<uint64_t>(imm.value()->value);
    program.has_body_fingerprint = true;
  }

  std::vector<Map<String, Any>> item_maps;
  if (!ReadArrayOfMaps(map, "items", &item_maps, error))
    return false;
  for (size_t i = 0; i < item_maps.size(); ++i) {
    VSimdProgramItem item;
    item.index = static_cast<int>(i);
    item.meta = item_maps[i];
    if (!ReadInt(item.meta, "span_begin", &item.span_begin, error) ||
        !ReadInt(item.meta, "span_end", &item.span_end, error) ||
        !ReadInt(item.meta, "group", &item.group, error) ||
        !ReadString(item.meta, "kind", &item.kind, error) ||
        !ReadString(item.meta, "analysis_status", &item.analysis_status,
                    error) ||
        !ReadString(item.meta, "status", &item.status, error) ||
        !ReadString(item.meta, "group_status", &item.group_status, error) ||
        !ReadBuffers(item.meta, "read_buffers", &item.read_buffers, error) ||
        !ReadBuffers(item.meta, "write_buffers", &item.write_buffers, error))
      return false;
    auto reason = item.meta.Get("reason");
    if (reason)
      if (auto str = reason.value().as<String>())
        item.reason = std::string(str.value());
    program.items.push_back(std::move(item));
  }

  std::vector<Map<String, Any>> group_maps;
  if (!ReadArrayOfMaps(map, "groups", &group_maps, error))
    return false;
  for (const Map<String, Any> &group_map : group_maps) {
    VSimdProgramGroup group;
    if (!ReadInt(group_map, "id", &group.id, error) ||
        !ReadString(group_map, "status", &group.status, error) ||
        !ReadIntArray(group_map, "items", &group.items, error))
      return false;
    program.groups.push_back(std::move(group));
  }

  std::vector<Map<String, Any>> capture_maps;
  if (!ReadArrayOfMaps(map, "captures", &capture_maps, error))
    return false;
  for (const Map<String, Any> &capture_map : capture_maps) {
    VSimdProgramCapture capture;
    auto buffer = capture_map.Get("buffer");
    if (!buffer || !buffer.value().as<tirx::Buffer>())
      return Fail(error, "capture.buffer is missing or invalid");
    capture.buffer = buffer.value().as<tirx::Buffer>().value();
    if (!ReadString(capture_map, "name", &capture.name, error) ||
        !ReadString(capture_map, "storage", &capture.storage, error))
      return false;
    program.captures.push_back(std::move(capture));
  }

  std::vector<Map<String, Any>> fragment_maps;
  if (!ReadArrayOfMaps(map, "fragments", &fragment_maps, error))
    return false;
  for (const Map<String, Any> &fragment_map : fragment_maps) {
    VSimdProgramFragment fragment;
    auto buffer = fragment_map.Get("buffer");
    if (!buffer || !buffer.value().as<tirx::Buffer>())
      return Fail(error, "fragment.buffer is missing or invalid");
    fragment.buffer = buffer.value().as<tirx::Buffer>().value();
    if (!ReadString(fragment_map, "name", &fragment.name, error) ||
        !ReadInt(fragment_map, "definition_vf", &fragment.definition_vf,
                 error) ||
        !ReadString(fragment_map, "definition_scope",
                    &fragment.definition_scope, error) ||
        !ReadIntArray(fragment_map, "use_vfs", &fragment.use_vfs, error) ||
        !ReadBool(fragment_map, "cross_vf", &fragment.cross_vf, error) ||
        !ReadBool(fragment_map, "escapes_vf", &fragment.escapes_vf, error) ||
        !ReadBool(fragment_map, "uses_outside_vf", &fragment.uses_outside_vf,
                  error))
      return false;
    program.fragments.push_back(std::move(fragment));
  }

  if (!ReadArrayOfMaps(map, "assumptions", &program.assumptions, error) ||
      !ReadArrayOfMaps(map, "attrs", &program.attrs, error))
    return false;

  if (!VerifyVSimdProgram(program, error))
    return false;
  *out = std::move(program);
  return true;
}

bool VerifyVSimdProgram(const VSimdProgram &program, std::string *error) {
  if (program.version != 1)
    return Fail(error, "unsupported VSimdProgram version");
  if (program.vf_index < 0)
    return Fail(error, "negative vf_index");

  const int item_count = static_cast<int>(program.items.size());
  if (program.item_count >= 0 && program.item_count != item_count)
    return Fail(error, "item_count does not match items");
  if (item_count == 0)
    return Fail(error, "empty items");
  if (static_cast<int>(program.groups.size()) == 0)
    return Fail(error, "empty groups");

  std::vector<bool> seen(item_count, false);
  std::vector<int> group_sizes(program.groups.size(), 0);
  for (size_t i = 0; i < program.items.size(); ++i) {
    const VSimdProgramItem &item = program.items[i];
    if (item.span_begin != static_cast<int>(i) ||
        item.span_end != static_cast<int>(i) + 1)
      return Fail(error, "items are not dense: item " + std::to_string(i));
    if (item.group < 0 ||
        item.group >= static_cast<int>(program.groups.size()))
      return Fail(error, "item has invalid group id");
    if (item.group_status != program.groups[item.group].status)
      return Fail(error, "item/group status mismatch");
    if (item.analysis_status != "LIFTED" && item.analysis_status != "TYPED" &&
        item.analysis_status != "PRESERVE" &&
        item.analysis_status != "OPAQUE")
      return Fail(error, "invalid analysis_status");
    if (item.status != "LIFTED" && item.status != "TYPED" &&
        item.status != "PRESERVE" && item.status != "OPAQUE")
      return Fail(error, "invalid effective status");
    bool metadata = IsMetadataKind(item.kind);
    if (metadata && item.analysis_status != "PRESERVE")
      return Fail(error, "metadata item is not PRESERVE");
    if (!metadata && item.analysis_status == "PRESERVE")
      return Fail(error, "PRESERVE item is not metadata");
    if (item.group_status == "P1_LIFTABLE") {
      if (metadata || item.analysis_status == "OPAQUE" ||
          item.status == "OPAQUE")
        return Fail(error, "P1_LIFTABLE group contains non-liftable item");
    } else if (item.group_status == "PRESERVE") {
      if (!metadata || item.status != "PRESERVE")
        return Fail(error, "PRESERVE group contains non-metadata item");
    } else if (item.group_status == "OPAQUE") {
      if (item.status != "OPAQUE")
        return Fail(error, "OPAQUE group item is not effective OPAQUE");
    } else {
      return Fail(error, "unknown group_status");
    }
    group_sizes[item.group]++;
    seen[i] = true;
  }

  for (size_t g = 0; g < program.groups.size(); ++g) {
    const VSimdProgramGroup &group = program.groups[g];
    if (group.id != static_cast<int>(g))
      return Fail(error, "groups are not ordered by id");
    if (static_cast<int>(group.items.size()) != group_sizes[g])
      return Fail(error, "group item list is incomplete");
    int previous = -1;
    for (int item_index : group.items) {
      if (item_index <= previous || item_index < 0 || item_index >= item_count)
        return Fail(error, "group item list is not sorted/in range");
      if (program.items[item_index].group != static_cast<int>(g))
        return Fail(error, "group item points to another group");
      previous = item_index;
    }
  }

  for (size_t i = 0; i < program.assumptions.size(); ++i) {
    int anchor = -1;
    int anchor_end = -1;
    if (!ReadInt(program.assumptions[i], "anchor", &anchor, error) ||
        !ReadInt(program.assumptions[i], "anchor_end", &anchor_end, error))
      return false;
    if (anchor < 0 || anchor > anchor_end || anchor_end > item_count)
      return Fail(error, "assumption anchor out of range");
  }
  for (size_t i = 0; i < program.attrs.size(); ++i) {
    int anchor = -1;
    int anchor_end = -1;
    if (!ReadInt(program.attrs[i], "anchor", &anchor, error) ||
        !ReadInt(program.attrs[i], "anchor_end", &anchor_end, error))
      return false;
    if (anchor < 0 || anchor > anchor_end || anchor_end > item_count)
      return Fail(error, "attr anchor out of range");
  }

  for (const VSimdProgramCapture &capture : program.captures) {
    if (capture.storage != "GLOBAL" && capture.storage != "SHARED" &&
        capture.storage != "FRAGMENT" && capture.storage != "UNKNOWN")
      return Fail(error, "invalid capture storage");
  }
  for (const VSimdProgramFragment &fragment : program.fragments) {
    if (fragment.definition_scope != "IN_VF" &&
        fragment.definition_scope != "OUTSIDE_VF")
      return Fail(error, "invalid fragment definition_scope");
    if (fragment.definition_scope == "OUTSIDE_VF" &&
        fragment.definition_vf != -1)
      return Fail(error, "outside fragment has a definition VF");
    if (fragment.definition_scope == "OUTSIDE_VF" && fragment.escapes_vf)
      return Fail(error, "outside fragment cannot escape a VF");
    if (fragment.definition_scope == "OUTSIDE_VF" &&
        fragment.use_vfs.size() <= 1 && fragment.cross_vf)
      return Fail(error, "single-use outside fragment marked cross_vf");
  }
  return true;
}

} // namespace tl
} // namespace tvm
