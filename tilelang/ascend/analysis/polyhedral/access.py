"""SCoP 建模：迭代域、仿射原语与访存分类。

核心模型（推导见 :func:`finite_difference` 与 :func:`classify_delta`）：

* **迭代域** ``D = { 0 <= x_k < E_k }``，每个 ``x_k`` 是一根轴（:class:`Axis`）；
* **访存函数** 把一次 ``Buffer`` 访问的多元下标做行主序线性化，得到
  ``off(x) = elem_offset + Σ_d idx_d * stride_d``（:func:`linear_element_offset`）；
* **分类** 对候选向量轴 ``v`` 求有限差分 ``Δ_v(off)``，按取值落到
  ``{1, 0, k, -1, 非常数}`` 上（:func:`classify_delta` / :func:`classify_access`）。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from tvm import arith, tirx

from .primitives import _INDEX_VALUE_OPS, Unsupported


@dataclass(frozen=True)
class Axis:
    """一根迭代轴：``0 <= var < extent``。

    ``kind`` 区分 ``T.Parallel`` 轴与 ``T.serial`` 轴：**直接内嵌且干净**的 serial
    循环也会被吸收进迭代域（见 :func:`~tilelang.ascend.analysis.polyhedral.plan._split_nest` /
    :func:`~tilelang.ascend.analysis.polyhedral.plan._serial_is_clean`），
    因此它同样是一根候选向量轴。
    """

    var: tirx.Var
    extent: tirx.PrimExpr
    index: int = 0  # 轴在嵌套中的位置，0 为最外层
    kind: tirx.ForKind = tirx.ForKind.PARALLEL  # PARALLEL 轴 / 被吸收的 SERIAL 轴

    @property
    def name(self) -> str:
        return self.var.name

    @property
    def is_parallel(self) -> bool:
        return self.kind == tirx.ForKind.PARALLEL


class AccessClass(str, enum.Enum):
    """沿某根轴的访存形态。

    常数 Δ 的形态由 :func:`classify_delta` 唯一决定；**非常数 Δ** 再经
    :func:`classify_access` 做结构细化（改进 4），把原先含糊的 ``NON_AFFINE``
    拆成三个可区分的标签：

    * :attr:`PIECEWISE` —— Δ 非常数，但元素偏移可以在「每 lane 索引向量」通道里
      构造出来（仿射 + 2 的幂 ``FloorDiv`` / ``FloorMod`` + 整数算术）→ 走
      ``vgather2`` / ``vscatter``；
    * :attr:`PERMUTE` —— Δ == -1，即沿该轴的**双射逆序**（置换而非步长访问）→ 同样
      走索引向量通道；
    * :attr:`TRUE_RANDOM` —— Δ 非常数且索引向量**构造不出来**（如非 2 的幂常量除法、
      数据相关的下标）→ 明确拒绝，不做静默降级。
    """

    CONTIGUOUS = "contiguous"  # Δ == 1
    UNIFORM = "uniform"  # Δ == 0
    STRIDED = "strided"  # Δ == k, |k| > 1（常量步长访问）
    PERMUTE = "permute"  # Δ == -1（双射逆序，置换）
    PIECEWISE = "piecewise"  # Δ 非常数，但可在索引向量通道构造
    INDIRECT = "indirect"  # 下标本身含 BufferLoad（gather/scatter）
    TRUE_RANDOM = "true_random"  # Δ 非常数且不可构造 → 拒绝


def _const_int(expr, analyzer: arith.Analyzer | None = None) -> int | None:
    """把表达式识别成整型常数。

    先吃显式常量（``IntImm`` / Python ``int``）。给了 ``analyzer`` 时再退到
    符号判定：先 ``simplify`` 折叠（能折成 ``IntImm`` 即可），否则用
    ``const_int_bound`` —— 它的区间是可靠的，因此**上下界相等即证明为常数**。
    这就把「化简后仍是常量表达式」的输入也认成常数，而不只认字面量。
    """
    if isinstance(expr, tirx.IntImm):
        return int(expr.value)
    if isinstance(expr, int) and not isinstance(expr, bool):
        return int(expr)
    if analyzer is not None:
        simplified = analyzer.simplify(expr)
        if isinstance(simplified, tirx.IntImm):
            return int(simplified.value)
        bound = analyzer.const_int_bound(simplified)
        if bound.min_value == bound.max_value:
            return int(bound.min_value)
    return None


def _compact_strides(shape) -> list:
    """行主序紧凑步长（buffer.strides 为空时使用）。"""
    strides = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc = acc * shape[i]
    return strides


def _buffer_strides(buffer) -> list:
    strides = list(buffer.strides) if len(buffer.strides) else _compact_strides(buffer.shape)
    return strides


def linear_element_offset(buffer, indices, analyzer: arith.Analyzer) -> tirx.PrimExpr:
    """把一次多元访问线性化成元素偏移 ``off = Σ idx_d * stride_d + elem_offset``。

    这是「访存函数」的构造步骤：后续所有连续性判断都在这个标量上做，
    因此 ``C[i//mx, j//my, i%mx, j%my]`` 这种高维摊平在化简后会被识别成
    ``i*my + j``（Δ == 1），不需要任何模板匹配。
    """
    offsets = _buffer_strides(buffer)
    if len(offsets) != len(indices):
        raise Unsupported(f"access rank {len(indices)} does not match buffer rank {len(offsets)}")
    expr = buffer.elem_offset
    for index, stride in zip(indices, offsets):
        term = index * stride
        expr = term if _is_zero(expr) else expr + term
    return analyzer.simplify(expr)


def _is_zero(expr) -> bool:
    c = _const_int(expr)
    return c == 0


def finite_difference(offset: tirx.PrimExpr, var: tirx.Var, analyzer: arith.Analyzer) -> tirx.PrimExpr:
    """沿轴 ``var`` 的有限差分 ``Δ(off) = off(var+1) - off(var)``（符号化简后）。

    这是整个模块的数学核心：**连续性/广播/步长/非仿射四类判定都由它给出**，
    与代码写法无关。例如

    * ``off = x``        → ``Δ = 1``   （连续）
    * ``off = 2*x``      → ``Δ = 2``   （步长 2）
    * ``off = x // 4 * 4 + x % 4`` → ``Δ = 1``（化简后连续）
    * ``off = a[b[x]]``  → 化简不成常数（间接）

    对 ``x//m*m + x%m`` 这类恒等式，TVM 的 ``arith.Analyzer`` 会先做规范化
    再化简，因此高维摊平场景不需要额外规则。
    """
    shifted = tirx.stmt_functor.substitute(offset, {var: var + 1})
    return analyzer.simplify(shifted - offset)


def classify_delta(delta: tirx.PrimExpr, analyzer: arith.Analyzer | None = None) -> tuple[AccessClass, int | None]:
    """把差分值映射成 :class:`AccessClass`（Δ 层的分类规则）。

    传入 ``analyzer`` 时，"是否常数" 的判定交给 :func:`_const_int` 做符号证明，
    而不再只认字面量 ``IntImm``。

    Δ 层只能给出常数情形（``UNIFORM`` / ``CONTIGUOUS`` / ``STRIDED`` / ``PERMUTE``）；
    **非常数** Δ 在这里统一落到 :attr:`AccessClass.PIECEWISE`（"待结构细化"），
    真正的 ``PIECEWISE`` / ``INDIRECT`` / ``TRUE_RANDOM`` 三选一由
    :func:`classify_access` 依据元素偏移能否在索引向量通道构造来定（改进 4）。
    """
    value = _const_int(delta, analyzer)
    if value is None:
        return AccessClass.PIECEWISE, None
    if value == 0:
        return AccessClass.UNIFORM, 0
    if value == 1:
        return AccessClass.CONTIGUOUS, 1
    if value == -1:
        return AccessClass.PERMUTE, -1
    return AccessClass.STRIDED, value


def index_vector_reason(expr, analyzer: arith.Analyzer | None = None) -> str | None:
    """判断元素偏移 ``expr`` 能否在「每 lane 索引向量」通道里构造出来。

    能构造则返回 ``None``；否则返回一句诊断原因（改进 4 的 ``TRUE_RANDOM`` 依据）。
    通道的构造能力 = 索引值 DAG 的构造能力（``build_value`` 的整数子集）：

    * 只允许整数算术（仿射 + ``Min`` / ``Max`` / 位运算）；
    * ``FloorDiv`` / ``FloorMod`` 只允许 **正的 2 的幂常量** 除数（与值算子
      strength-reduction 一致，见 ``value._floordiv_mod``）；
    * 不允许任何 ``BufferLoad``（那是 :attr:`AccessClass.INDIRECT`，另走一条路）。
    """
    reason = []

    def visit(node):
        name = type(node).__name__
        if isinstance(node, tirx.BufferLoad):
            reason.append("index contains a BufferLoad")
            return
        if name in ("FloorDiv", "FloorMod"):
            divisor = _const_int(node.b, analyzer)
            if divisor is None:
                reason.append(f"{name} needs a compile-time constant divisor (got {node.b})")
            elif divisor <= 0 or (divisor & (divisor - 1)):
                reason.append(f"{name} by {divisor} is not a power of two")
            return
        if name not in _INDEX_VALUE_OPS:
            reason.append(f"unsupported index operator {name}")

    tirx.stmt_functor.post_order_visit(expr, visit)
    return reason[0] if reason else None


def _contains_buffer_load(expr) -> bool:
    found = []

    def visit(node):
        if isinstance(node, tirx.BufferLoad):
            found.append(node)

    tirx.stmt_functor.post_order_visit(expr, visit)
    return bool(found)


def _free_vars(expr) -> list[tirx.Var]:
    variables: list[tirx.Var] = []

    def visit(node):
        if isinstance(node, tirx.Var):
            variables.append(node)

    tirx.stmt_functor.post_order_visit(expr, visit)
    return variables


@dataclass
class AccessInfo:
    """一次 buffer 访存 + 它沿每根平行轴的分类结果。"""

    buffer: object
    indices: tuple
    offset: tirx.PrimExpr
    is_store: bool
    per_axis: dict = field(default_factory=dict)  # axis.index -> (AccessClass, stride)
    indirect: bool = False
    in_global: bool = False
    reason: str = ""  # 非常数 Δ 无法构造索引向量时的诊断原因（改进 4）

    def axis_class(self, axis: Axis) -> tuple[AccessClass, int | None]:
        # 以 ``axis.index``（轴在本次迭代域中的位置）为键：同一 nest 内恒唯一，且不依赖
        # ``axis.var`` 的名字——两个不同 serial 循环重名（或在多段 program 的不同段里重名）
        # 时不会互相串味。
        return self.per_axis.get(axis.index, (AccessClass.TRUE_RANDOM, None))


def classify_access(
    buffer,
    indices,
    axes: list[Axis],
    analyzer: arith.Analyzer,
    *,
    is_store: bool,
) -> AccessInfo:
    """构造访问的仿射模型并逐轴分类（改进 1 + 4）。

    Δ 层给出常数形态；非常数 Δ 再做一次**结构判定**：

    * 下标含 ``BufferLoad``（``A[B[i]]``）且沿该轴**变化** → :attr:`AccessClass.INDIRECT`，
      索引向量来自被装载的寄存器；沿该轴**不变**（``Δ == 0``，如 ``c[b[0]]``）时它是
      lane 内均匀的标量地址，仍归 :attr:`AccessClass.UNIFORM`（pre/post 语句因此可以
      只发一次单元素写，而不是整段拒绝）；
    * 否则尝试在索引向量通道构造元素偏移（:func:`index_vector_reason`）：
      可构造 → :attr:`AccessClass.PIECEWISE`（分块置换/高维摊平等）；不可构造 →
      :attr:`AccessClass.TRUE_RANDOM`（如非 2 的幂常量除法、数据相关下标）。

    三类都由同一个「每 lane 索引向量」通道发射成 ``vgather2`` / ``vscatter``，
    区别只在诊断标签。
    """
    scales = tuple(indices)
    indirect = any(_contains_buffer_load(index) for index in scales)
    offset = linear_element_offset(buffer, scales, analyzer)
    info = AccessInfo(
        buffer=buffer,
        indices=scales,
        offset=offset,
        is_store=is_store,
        indirect=indirect,
        in_global=buffer.scope() == "global",
    )
    piecewise_reason = None if indirect else index_vector_reason(offset, analyzer)
    for axis in axes:
        delta = finite_difference(offset, axis.var, analyzer)
        kind, stride = classify_delta(delta, analyzer)
        if indirect:
            # 间接下标（含 ``BufferLoad``）不必然是「每 lane 索引向量」：地址沿该轴 **不变**
            # （Δ==0）时它就是 lane 内均匀的标量地址，例如 pre/post 语句 ``c[b[0]] = 1.0``。
            # 只有沿轴变化的间接地址才必须走索引向量通道（``vgather2`` / ``vscatter``）。
            if kind != AccessClass.UNIFORM:
                kind, stride = AccessClass.INDIRECT, None
            info.per_axis[axis.index] = (kind, stride)
            continue
        if kind == AccessClass.PIECEWISE and piecewise_reason is not None:
            # Δ 非常数，但索引向量通道构造不出来 → 真正的"无结构"访问。
            kind, stride = AccessClass.TRUE_RANDOM, None
            info.reason = piecewise_reason
        elif kind == AccessClass.STRIDED and stride is not None and stride < 0 and piecewise_reason is not None:
            # 负步长**必须**走索引向量通道（无符号 lane 斜坡乘负步长没有意义，缺陷 D8）；
            # 该通道构造不出来时明确拒绝，而不是落到 ``vload_strided``。
            kind, stride = AccessClass.TRUE_RANDOM, None
            info.reason = piecewise_reason
        info.per_axis[axis.index] = (kind, stride)
    return info
