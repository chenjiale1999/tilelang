#include "vsimd_payload_read.h"

namespace tvm {
namespace tl {
namespace polyhedral {

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

static bool ReadBoolField(const Map<String, Any> &map, const char *key,
                          bool *out, std::string *error) {
  auto value = map.Get(key);
  if (!value)
    return Fail(error, std::string("missing '") + key + "'");
  auto imm = value.value().as<IntImm>();
  if (!imm)
    return Fail(error, std::string("'") + key + "' is not a bool");
  *out = imm.value()->value != 0;
  return true;
}

static bool GetPrimExpr(const Map<String, Any> &map, const char *key,
                        PrimExpr *out, std::string *error) {
  auto value = map.Get(key);
  if (!value)
    return Fail(error, std::string("missing '") + key + "'");
  auto expr = value.value().as<PrimExpr>();
  if (!expr)
    return Fail(error, std::string("'") + key + "' is not a PrimExpr");
  *out = expr.value();
  return true;
}

static bool GetBuffer(const Map<String, Any> &map, const char *key,
                      tirx::Buffer *out, std::string *error) {
  auto value = map.Get(key);
  if (!value)
    return Fail(error, std::string("missing '") + key + "'");
  auto buffer = value.value().as<tirx::Buffer>();
  if (!buffer)
    return Fail(error, std::string("'") + key + "' is not a Buffer");
  *out = buffer.value();
  return true;
}

static bool GetVar(const Map<String, Any> &map, const char *key, Var *out,
                   std::string *error) {
  auto value = map.Get(key);
  if (!value)
    return Fail(error, std::string("missing '") + key + "'");
  auto var = value.value().as<Var>();
  if (!var)
    return Fail(error, std::string("'") + key + "' is not a Var");
  *out = var.value();
  return true;
}

static bool ReadOptionalInt(const Map<String, Any> &map, const char *key,
                            int *out, std::string *error) {
  auto value = map.Get(key);
  if (!value)
    return true;
  auto imm = value.value().as<IntImm>();
  if (!imm)
    return Fail(error, std::string("'") + key + "' is not an integer");
  *out = static_cast<int>(imm.value()->value);
  return true;
}

} // namespace

bool ReadVSimdRegion(const Map<String, Any> &item_map, int vf_index,
                     VSimdRegion *out, std::string *error) {
  if (out == nullptr)
    return Fail(error, "null region output");
  auto region_value = item_map.Get("region");
  if (!region_value)
    return Fail(error, "missing 'region'");
  auto region_opt = region_value.value().as<Map<String, Any>>();
  if (!region_opt)
    return Fail(error, "'region' is not a map");
  const Map<String, Any> &region_map = region_opt.value();

  VSimdRegion region;
  region.vf_index = vf_index;
  int nest_index = -1;
  if (!ReadOptionalInt(item_map, "parallel_nest_index", &nest_index, error) ||
      !ReadOptionalInt(item_map, "nest_index", &nest_index, error))
    return false;
  region.nest_index = nest_index;

  std::vector<Map<String, Any>> domain_maps;
  if (!ReadArrayOfMaps(region_map, "domain", &domain_maps, error))
    return false;
  for (const Map<String, Any> &axis_map : domain_maps) {
    vsimd::LoopInfo loop;
    if (!GetVar(axis_map, "var", &loop.var, error) ||
        !GetPrimExpr(axis_map, "min", &loop.min, error) ||
        !GetPrimExpr(axis_map, "extent", &loop.extent, error))
      return false;
    region.domain.push_back(std::move(loop));
  }

  std::vector<Map<String, Any>> access_maps;
  if (!ReadArrayOfMaps(region_map, "accesses", &access_maps, error))
    return false;
  for (const Map<String, Any> &access_map : access_maps) {
    VSimdAccess wrapped;
    if (!GetBuffer(access_map, "buffer", &wrapped.access.buffer, error))
      return false;
    auto indices = access_map.Get("indices");
    if (!indices)
      return Fail(error, "missing 'indices'");
    auto indices_arr = indices.value().as<Array<PrimExpr>>();
    if (!indices_arr)
      return Fail(error, "'indices' is not an expression array");
    wrapped.access.indices = indices_arr.value();
    if (!ReadBoolField(access_map, "is_write", &wrapped.access.is_write, error))
      return false;
    if (auto value = access_map.Get("value")) {
      if (auto expr = value.value().as<PrimExpr>())
        wrapped.access.value = expr.value();
    }
    if (!ReadIntArray(access_map, "dep_axes", &wrapped.dep_axes, error) ||
        !ReadIntArray(access_map, "value_axes", &wrapped.value_axes, error))
      return false;
    auto value_shape = access_map.Get("value_shape");
    if (!value_shape)
      return Fail(error, "missing 'value_shape'");
    auto value_shape_arr = value_shape.value().as<Array<PrimExpr>>();
    if (!value_shape_arr)
      return Fail(error, "'value_shape' is not an expression array");
    wrapped.value_shape.assign(value_shape_arr.value().begin(),
                               value_shape_arr.value().end());
    if (!ReadBoolField(access_map, "has_layout", &wrapped.has_layout, error))
      return false;
    if (wrapped.has_layout) {
      auto layout = access_map.Get("layout");
      if (!layout)
        return Fail(error, "missing 'layout' on a layout-aware access");
      auto layout_obj = layout.value().as<Layout>();
      if (!layout_obj)
        return Fail(error, "'layout' is not a Layout");
      wrapped.layout = layout_obj.value();
    }
    region.accesses.push_back(std::move(wrapped));
  }

  std::vector<Map<String, Any>> capture_maps;
  if (!ReadArrayOfMaps(region_map, "captures", &capture_maps, error))
    return false;
  for (const Map<String, Any> &capture_map : capture_maps) {
    Buffer buffer;
    std::string kind;
    if (!GetBuffer(capture_map, "buffer", &buffer, error) ||
        !ReadString(capture_map, "kind", &kind, error))
      return false;
    region.captures.emplace_back(buffer, kind);
  }

  std::vector<Map<String, Any>> outer_maps;
  if (!ReadArrayOfMaps(region_map, "outer_vars", &outer_maps, error))
    return false;
  for (const Map<String, Any> &outer_map : outer_maps) {
    Var var;
    std::string kind;
    if (!GetVar(outer_map, "var", &var, error) ||
        !ReadString(outer_map, "kind", &kind, error))
      return false;
    region.outer_vars.emplace_back(var, kind);
  }

  auto shape_syms = region_map.Get("shape_syms");
  if (!shape_syms)
    return Fail(error, "missing 'shape_syms'");
  auto shape_syms_arr = shape_syms.value().as<Array<Var>>();
  if (!shape_syms_arr)
    return Fail(error, "'shape_syms' is not a var array");
  region.shape_syms.assign(shape_syms_arr.value().begin(),
                           shape_syms_arr.value().end());

  std::vector<Map<String, Any>> view_maps;
  if (!ReadArrayOfMaps(region_map, "local_views", &view_maps, error))
    return false;
  for (const Map<String, Any> &view_map : view_maps) {
    Buffer view;
    Buffer base;
    if (!GetBuffer(view_map, "view", &view, error))
      return false;
    // A view with an unresolved base is still a valid local view; keep the
    // pair nullable rather than rejecting the whole region.
    if (auto base_value = view_map.Get("base")) {
      if (auto base_obj = base_value.value().as<Buffer>())
        base = base_obj.value();
    }
    region.local_views.emplace_back(view, base);
  }

  std::vector<Map<String, Any>> layout_maps;
  if (!ReadArrayOfMaps(region_map, "layouts", &layout_maps, error))
    return false;
  for (const Map<String, Any> &layout_map : layout_maps) {
    Buffer buffer;
    if (!GetBuffer(layout_map, "buffer", &buffer, error))
      return false;
    auto layout = layout_map.Get("layout");
    if (!layout)
      return Fail(error, "missing 'layout'");
    auto layout_obj = layout.value().as<Layout>();
    if (!layout_obj)
      return Fail(error, "'layout' is not a Layout");
    region.layouts.emplace_back(buffer, layout_obj.value());
  }

  auto assumptions = region_map.Get("assumptions");
  if (!assumptions)
    return Fail(error, "missing 'assumptions'");
  auto assumptions_arr = assumptions.value().as<Array<PrimExpr>>();
  if (!assumptions_arr)
    return Fail(error, "'assumptions' is not an expression array");
  region.assumptions.assign(assumptions_arr.value().begin(),
                            assumptions_arr.value().end());

  std::vector<Map<String, Any>> binding_maps;
  if (!ReadArrayOfMaps(region_map, "bindings", &binding_maps, error))
    return false;
  for (const Map<String, Any> &binding_map : binding_maps) {
    Var var;
    PrimExpr value;
    if (!GetVar(binding_map, "var", &var, error) ||
        !GetPrimExpr(binding_map, "value", &value, error))
      return false;
    region.bindings.emplace_back(var, value);
  }

  *out = std::move(region);
  return true;
}

namespace {

static bool ReadExprArray(const Map<String, Any> &map, const char *key,
                          std::vector<PrimExpr> *out, std::string *error) {
  auto value = map.Get(key);
  if (!value)
    return Fail(error, std::string("missing '") + key + "'");
  auto arr = value.value().as<Array<PrimExpr>>();
  if (!arr)
    return Fail(error, std::string("'") + key + "' is not an expression array");
  out->assign(arr.value().begin(), arr.value().end());
  return true;
}

/*! \brief Decode one ``{buffer, indices, extents, access_type}`` tile region. */
static bool ReadTileRegion(const Map<String, Any> &owner, const char *key,
                           Buffer *buffer, std::vector<PrimExpr> *indices,
                           std::vector<PrimExpr> *extents, std::string *error) {
  auto value = owner.Get(key);
  if (!value)
    return Fail(error, std::string("missing '") + key + "'");
  auto region = value.value().as<Map<String, Any>>();
  if (!region)
    return Fail(error, std::string("'") + key + "' is not a map");
  if (!GetBuffer(region.value(), "buffer", buffer, error) ||
      !ReadExprArray(region.value(), "indices", indices, error) ||
      !ReadExprArray(region.value(), "extents", extents, error))
    return false;
  if (indices->size() != extents->size())
    return Fail(error, std::string("'") + key +
                           "' has mismatched indices/extents ranks");
  return true;
}

/*! \brief Decode one ``{buffer, name, dtype, shape, scope}`` endpoint. */
static bool ReadEndpoint(const Map<String, Any> &meta, const char *key,
                         Buffer *buffer, std::string *scope,
                         std::string *error) {
  auto value = meta.Get(key);
  if (!value)
    return Fail(error, std::string("missing '") + key + "'");
  auto info = value.value().as<Map<String, Any>>();
  if (!info)
    return Fail(error, std::string("'") + key + "' is not a map");
  return GetBuffer(info.value(), "buffer", buffer, error) &&
         ReadString(info.value(), "scope", scope, error);
}

/*! \brief Decode ``src``/``dst`` (buffer + storage scope) of a typed item. */
static bool ReadEndpoints(const VSimdProgramItem &item, Buffer *src,
                          Buffer *dst, std::string *src_scope,
                          std::string *dst_scope, std::string *error) {
  return ReadEndpoint(item.meta, "src", src, src_scope, error) &&
         ReadEndpoint(item.meta, "dst", dst, dst_scope, error);
}

} // namespace

bool ReadVSimdCopy(const VSimdProgramItem &item, VSimdCopyReducePayload *out,
                   std::string *error) {
  if (out == nullptr)
    return Fail(error, "null copy output");
  VSimdCopyReducePayload payload;
  if (!ReadEndpoints(item, &payload.src, &payload.dst, &payload.src_scope,
                     &payload.dst_scope, error) ||
      !ReadTileRegion(item.meta, "src_region", &payload.src,
                      &payload.src_indices, &payload.src_extents, error) ||
      !ReadTileRegion(item.meta, "dst_region", &payload.dst,
                      &payload.dst_indices, &payload.dst_extents, error))
    return false;
  *out = std::move(payload);
  return true;
}

bool ReadVSimdReduce(const VSimdProgramItem &item, VSimdReducePayload *out,
                     std::string *error) {
  if (out == nullptr)
    return Fail(error, "null reduce output");
  VSimdReducePayload payload;
  if (!ReadEndpoints(item, &payload.src, &payload.dst, &payload.src_scope,
                     &payload.dst_scope, error) ||
      !ReadTileRegion(item.meta, "src_region", &payload.src,
                      &payload.src_indices, &payload.src_extents, error) ||
      !ReadTileRegion(item.meta, "dst_region", &payload.dst,
                      &payload.dst_indices, &payload.dst_extents, error) ||
      !ReadString(item.meta, "reduce_kind", &payload.reduce_kind, error))
    return false;
  if (!ReadInt(item.meta, "reduce_dim", &payload.dim, error))
    return false;
  auto clear = item.meta.Get("clear");
  if (!clear)
    return Fail(error, "missing 'clear'");
  auto clear_imm = clear.value().as<IntImm>();
  if (!clear_imm)
    return Fail(error, "'clear' is not a bool");
  payload.clear = clear_imm.value()->value != 0;
  *out = std::move(payload);
  return true;
}

} // namespace polyhedral
} // namespace tl
} // namespace tvm
