"""Shared value types of the polyhedral P2/P3 contract.

Only the two types that cross the C++/Python boundary live here, so the C++-only
facade (:mod:`tilelang.ascend.analysis.polyhedral`) never has to import the
Python reference implementation (:mod:`tilelang.ascend.analysis.polyhedral_reference`):

* :class:`AccessClass` — the access classification tag. The C++ implementation
  returns it as a string; this enum is the typed Python view of the same value;
* :class:`Unsupported` — the rejection exception raised by both implementations
  with the identical ``message`` + ``diagnostics`` payload.

This module imports nothing from tilelang, so it is safe to load on its own.
"""

from __future__ import annotations

import enum


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
