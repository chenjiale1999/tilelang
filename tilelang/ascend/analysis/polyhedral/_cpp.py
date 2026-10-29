"""FFI-backed implementation of the polyhedral P2/P3 API (``impl = "cpp"``).

The C++ port registers its plan objects as TVM FFI objects whose reflected field
names match the Python dataclasses, so tests (and the emit layer) read
``program.segments[0].stores[0].value.recipe`` without any re-construction.
Only two derived members need a Python-side helper:

* :attr:`StorePlan.kind` — the C++ field is ``kind_value`` (a string), because a
  reflected field would shadow a Python property of the same name;
* :meth:`AccessInfo.axis_class` — a method, forwarded to the C++ side.

Rejections arrive as ``{"ok": False, "message": ..., "diagnostics": [...]}`` and
are re-raised as :class:`Unsupported` with the exact same text and diagnostic
order (D3).
"""

from __future__ import annotations

from tvm.ir.base import Node
from tilelang.transform import _ffi_api
import tvm_ffi

from ._contract import AccessClass, Unsupported


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


def build_plan(stmt, lanes=None):
    """C++ ``build_plan``: ``T.Parallel`` nest -> :class:`VectorProgram`."""
    result = _ffi_api.AscendPolyhedralBuildPlan(stmt, lanes)
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


def vectorize_parallel_nests(stmt, *, lanes=None):
    """C++ driver: replace every vectorizable ``T.Parallel`` nest in-place."""
    result = _ffi_api.AscendPolyhedralVectorizeNests(stmt, lanes)
    if not result["ok"]:
        _raise(result)
    return result["stmt"]


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
