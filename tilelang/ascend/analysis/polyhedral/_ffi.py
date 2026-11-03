"""FFI bindings and value types of the polyhedral P2/P3 implementation.

The C++ port (``src/ascend/transform/polyhedral``) registers its plan objects as
TVM FFI objects whose reflected field names match the Python dataclasses, so
tests (and the emit layer) read ``program.segments[0].stores[0].value.recipe``
without any re-construction. Only two derived members need a Python-side helper:

* :attr:`StorePlan.kind` — the C++ field is ``kind_value`` (a string), because a
  reflected field would shadow a Python property of the same name;
* :meth:`AccessInfo.axis_class` — a method, forwarded to the C++ side.

Rejections arrive as ``{"ok": False, "message": ..., "diagnostics": [...]}`` and
are re-raised as :class:`Unsupported` with the exact same text and diagnostic
order (D3).

The two value types below are the only Python-owned definitions: an enum and an
exception cannot cross the FFI boundary as objects, so they mirror the C++
``AccessClass`` strings and the ``polyhedral::Unsupported`` payload.
"""

from __future__ import annotations

import enum

import tvm_ffi
from tvm.ir.base import Node

from tilelang.transform import _ffi_api


class AccessClass(str, enum.Enum):
    """沿某根轴的访存形态。

    常数 Δ 的形态由 ``classify_delta`` 唯一决定；**非常数 Δ** 再经
    ``classify_access`` 做结构细化（改进 4），把原先含糊的 ``NON_AFFINE``
    拆成三个可区分的标签：

    * ``PIECEWISE`` —— Δ 非常数，但元素偏移可以在「每 lane 索引向量」通道里
      构造出来（仿射 + 2 的幂 ``FloorDiv`` / ``FloorMod`` + 整数算术）→ 走
      ``vgather2`` / ``vscatter``；
    * ``PERMUTE`` —— Δ == -1，即沿该轴的**双射逆序**（置换而非步长访问）→ 同样
      走索引向量通道；
    * ``TRUE_RANDOM`` —— Δ 非常数且索引向量**构造不出来**（如非 2 的幂常量除法、
      数据相关的下标）→ 明确拒绝，不做静默降级。
    """

    CONTIGUOUS = "contiguous"  # Δ == 1
    UNIFORM = "uniform"  # Δ == 0
    STRIDED = "strided"  # Δ == k, |k| > 1（常量步长访问）
    PERMUTE = "permute"  # Δ == -1（双射逆序，置换）
    PIECEWISE = "piecewise"  # Δ 非常数，但可在索引向量通道构造
    INDIRECT = "indirect"  # 下标本身含 BufferLoad（gather/scatter）
    TRUE_RANDOM = "true_random"  # Δ 非常数且不可构造 → 拒绝


class Unsupported(Exception):
    """Raised when a nest cannot be vectorized; carries the diagnostic reason(s).

    ``diagnostics`` 收集本次规划过程中**全部卡点**（每根候选轴的分类摘要与失败
    原因），这样上层能一次性看到所有阻碍，而不是只拿到第一个失败原因（改进 7）。
    ``build_plan`` 会填充它；直接 ``str(err)`` 仍然是单行摘要。
    """

    def __init__(self, message: str, *, diagnostics: list[str] | None = None):
        super().__init__(message)
        self.diagnostics: list[str] = list(diagnostics or [])


# ---------------------------------------------------------------------------
# FFI object types (type keys registered by the C++ port)
# ---------------------------------------------------------------------------


@tvm_ffi.register_object("tl.polyhedral.Axis")
class Axis(Node):
    """One iteration axis (``0 <= var < extent``)."""


@tvm_ffi.register_object("tl.polyhedral.AccessInfo")
class AccessInfo(Node):
    """One buffer access plus its per-axis classification."""

    def axis_class(self, axis):
        """Return ``(AccessClass, stride)`` for ``axis`` (``stride`` may be None)."""
        kind, stride = _ffi_api.AscendPolyhedralAxisClass(self, axis)
        return (AccessClass(kind), stride)


@tvm_ffi.register_object("tl.polyhedral.StoreSpec")
class StoreSpec(Node):
    """Where to write, what to write, and an optional predicate."""


@tvm_ffi.register_object("tl.polyhedral.ValueNode")
class ValueNode(Node):
    """A value DAG node (instruction-selection agnostic recipe)."""


@tvm_ffi.register_object("tl.polyhedral.ReduceInfo")
class ReduceInfo(Node):
    """Reduction description of one store."""


@tvm_ffi.register_object("tl.polyhedral.StorePlan")
class StorePlan(Node):
    """Full plan of one write."""

    @property
    def kind(self):
        # ``kind_value`` is the reflected string; ``kind`` is the enum the tests
        # and the emitter compare against.
        return AccessClass(self.kind_value)


@tvm_ffi.register_object("tl.polyhedral.VectorPlan")
class VectorPlan(Node):
    """Segment-level plan (one body + optional pre/post siblings)."""


@tvm_ffi.register_object("tl.polyhedral.VectorProgram")
class VectorProgram(Node):
    """``build_plan`` result: 1..N segments."""


# ---------------------------------------------------------------------------
# API dispatch helpers
# ---------------------------------------------------------------------------


def _raise(result) -> None:
    raise Unsupported(result["message"], diagnostics=list(result["diagnostics"]))


def build_plan(stmt, lanes=None, layouts=None, vf_alloc_scope=None):
    """C++ ``build_plan``: ``T.Parallel`` nest -> :class:`VectorProgram`.

    ``layouts`` is the explicit ``{Buffer: Layout}`` map of the nest's
    destination buffers (the P1 region hand-off); a buffer without an entry
    keeps the default ND store, an Ascend NZ entry scatters 32B blocks.

    ``vf_alloc_scope`` is the enclosing ``SIMD_VF`` allocation set. A fragment
    access without it is rejected (G0b) instead of planning an un-compilable
    ``vld``/``vsts``; passing the fragment buffer enables register mode.
    """
    result = _ffi_api.AscendPolyhedralBuildPlan(stmt, lanes, layouts, vf_alloc_scope)
    if not result["ok"]:
        _raise(result)
    return result["program"]


def emit_plan(plan):
    """C++ ``emit_plan``: plan -> ``T.simd.*`` TIR."""
    result = _ffi_api.AscendPolyhedralEmitPlan(plan)
    if not result["ok"]:
        _raise(result)
    return result["stmt"]


def render_plan(plan):
    """C++ ``render_plan``: plan -> human-readable text."""
    result = _ffi_api.AscendPolyhedralRenderPlan(plan)
    if not result["ok"]:
        _raise(result)
    return result["text"]


def finite_difference(offset, var, analyzer):
    """C++ ``finite_difference``.

    The Python analyzer object is not transferable over FFI, so the C++ side
    creates a fresh :class:`arith.Analyzer` (identical for every caller that does
    not pre-bind variables; the parity harness covers this).
    """
    return _ffi_api.AscendPolyhedralFiniteDifference(offset, var, analyzer is not None)


def classify_delta(delta, analyzer=None):
    """C++ ``classify_delta``: ``(AccessClass, stride_or_None)``."""
    kind, stride = _ffi_api.AscendPolyhedralClassifyDelta(delta, analyzer is not None)
    return (AccessClass(kind), stride)
