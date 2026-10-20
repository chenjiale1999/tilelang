"""Unified polyhedral (affine) analysis for ``T.Parallel`` vectorization.

This module replaces the "one branch per code shape" style of recognition used
by ``vectorize_parallel_to_pto.cc`` with a single mathematical analysis:

========================== ====================================================
经典做法（case-by-case）     本模块（统一数学分析）
========================== ====================================================
逐条匹配节点形状             建立 SCoP：迭代域 + 访存仿射函数
对每个算子判断维度/stride    对每个访问算「沿轴的有限差分」Δ
按代码形态分支               按 Δ 的取值分类，一条规则覆盖所有形态
========================== ====================================================

核心模型
--------
给定 ``T.Parallel`` 嵌套：``for x0 in Parallel(E0): ... for xn in Parallel(En): body``

* **迭代域** ``D = { 0 <= x_k < E_k }``，每个 ``x_k`` 是一根轴（:class:`Axis`）。
* **访存函数** 把一次 ``Buffer`` 访问的多元下标做行主序线性化，得到
  ``off(x) = elem_offset + Σ_d idx_d * stride_d``（:func:`linear_element_offset`）。
* **调度/向量化决策** 不靠匹配节点形状，而是对候选向量轴 ``v`` 求
  **有限差分** ``Δ_v(off) = off(..., x_v + 1, ...) - off(...)``
  （:func:`finite_difference`）。符号化简后：

  ================ ==================================================
  ``Δ_v(off)``     含义 / 后续动作
  ================ ==================================================
  ``1``            连续 → ``vld`` / ``vsts``（NORM）
  ``0``            lane 内均匀 → 标量读一次 + ``vdup`` 广播
  ``k, |k| > 1``   常量步长 → ``vgather2`` / ``vscatter``（步长斜坡）
  ``-1``           逆序置换（PERMUTE）→ 每 lane 索引向量
  非常数           PIECEWISE / INDIRECT → 每 lane 索引向量（gather/scatter）
  非常数且不可构造   TRUE_RANDOM → 明确拒绝
  ================ ==================================================

  这条规则同时吃掉了几何上完全不同的场景，无需分别写分支：

  * ``c[i] = a[i]`` / ``c[i] = a[2*i]`` / ``c[i] = a[0]``
  * ``g[1, 8+i_s, 3]``（§4 表达式下标）
  * ``h[i//16, i%16]``（§4 高维摊平——化简后 ``Δ == 1``）
  * ``d[i,j] = g[i] - g[j]``（§2 维度不一致：``g[i]`` 沿 j 是 ``Δ == 0``）
  * ``cs[i//4, j//4, i%4, j%4]``（§4 分块置换：非常数 Δ → ``PIECEWISE`` 索引向量）
  * ``a[b[i]]``（下标含 ``BufferLoad`` → ``INDIRECT`` → 索引向量来自寄存器）

  推导见 ``finite_difference`` 与 ``classify_delta`` 的注释。

边界的处理
----------
* **归约**（§2 Case II）：若 store 沿 ``v`` 的 ``Δ == 0`` 而取值沿 ``v`` 变化，
  该轴是归约轴 → :class:`ReduceInfo`，映射到 ``vcadd`` 类指令。
* **分支**（§1）：``if c: store(a) else: store(b)`` 统一降为
  ``store(select(c, a, b))``；单臂 ``if`` 降为**谓词化 store**（mask）。
  选择条件的来源不影响框架——随向量轴变化时是逐 lane 谓词（``vcmp``），
  沿向量轴均匀（循环不变，§1 Case II）时广播成全 lane 谓词
  ``update_mask(cond ? lanes : 0)``，两条都汇入同一个 ``vsel``。
* **多语句**（§3）：body 被规范化成有序的 :class:`StoreSpec` 列表，天然支持。
* **原子操作**（§5）/ **复杂控制流**（§6）：明确拒绝并给出诊断，
  不做静默降级。

本模块只做分析与规划（对应分层方案里的 P1 + P2 的分析部分）；
:func:`emit_plan` 把计划发射成 **``T.simd.*``**（``ascend/language/simd.py``，
CCE 向量原语的前端封装），而不是任何自定义 ABI。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from tvm import arith, tirx

from tilelang.ascend.language import simd
from tilelang.ascend.language.simd import _mask_bits, _vec_dtype, _vreg_lanes

__all__ = [
    "Axis",
    "AccessClass",
    "AccessInfo",
    "StoreSpec",
    "ValueNode",
    "StorePlan",
    "ReduceInfo",
    "VectorPlan",
    "VectorProgram",
    "Unsupported",
    "finite_difference",
    "classify_delta",
    "linear_element_offset",
    "build_plan",
    "emit_plan",
    "render_plan",
    "vectorize_parallel_nests",
]

# ---------------------------------------------------------------------------
# 0. 原语事实表 —— 唯一保留的 per-op 事实（原语级，不含维度/分块/索引形态）
# ---------------------------------------------------------------------------
# 名字采用 TIR 节点/算子名的尾段，避免依赖具体的类对象是否被导出。
_ELEMENTWISE_BINARY = {
    "Add": "vadd",
    "Sub": "vsub",
    "Mul": "vmul",
    "Div": "vdiv",
    "Max": "vmax",
    "Min": "vmin",
    "BitwiseAnd": "vand",
    "BitwiseOr": "vor",
    "BitwiseXor": "vxor",
    "ShiftLeft": "vshl",
    "ShiftRight": "vshr",
}

_ELEMENTWISE_UNARY = {
    "Neg": "vneg",
    "abs": "vabs",
    "exp": "vexp",
    "sqrt": "vsqrt",
    "log": "vln",
}

# 标量-向量（一个操作数是 lane 内均匀值）的专用指令；没有专用变体的算子
# 通过 vdup 广播后走向量-向量指令。
_VEC_SCALAR_BINARY = {"vadd": "vadds", "vmul": "vmuls", "vmax": "vmaxs", "vmin": "vmins"}

# 比较算子 → 谓词。含一个 lane 均匀操作数时走 ``vcmps``（标量形式）。
_COMPARE = {
    "LT": "lt",
    "LE": "le",
    "GT": "gt",
    "GE": "ge",
    "EQ": "eq",
    "NE": "ne",
}

# 布尔组合（谓词化时合并 mask 用）；``And``/``Or`` 由 normalize_body 产生。
# 目标指令是 ``T.simd.pand/por/pnot``（谓词寄存器运算），与数据向量位的
# ``T.simd.vand/vor/vnot`` 区分开。
_BOOL_BINARY = {"And": "pand", "Or": "por"}
_BOOL_UNARY = {"Not": "pnot"}

# 比较谓词的逆：交换操作数（标量形式要求 (vector, scalar)）时用于保持语义。
_COMPARE_FLIP = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le", "eq": "eq", "ne": "ne"}

# 大小写无关的查表：TIR 节点名是 ``Add``/``LT``，而 intrinsic call 的算子尾段是
# ``add``/``lt``，两者共用同一张 recipe 表。
_BINARY_RECIPE = {k.lower(): v for k, v in _ELEMENTWISE_BINARY.items()}
_UNARY_RECIPE = {k.lower(): v for k, v in _ELEMENTWISE_UNARY.items()}
_COMPARE_PRED = {k.lower(): v for k, v in _COMPARE.items()}
_BOOL_BINARY_RECIPE = {k.lower(): v for k, v in _BOOL_BINARY.items()}
_BOOL_UNARY_RECIPE = {k.lower(): v for k, v in _BOOL_UNARY.items()}


def _is_atomic_name(tail: str) -> bool:
    """原子算子识别（``tl.atomic_add`` / ``tl.atomic_add_elem_op`` ...）。"""
    return "atomic" in tail


# 谓词寄存器的 TIR 类型：``T.simd.*`` 的谓词一律是 256 位的 ``vector_bool``。
_MASK_DTYPE = "boolx256"

# lane 内均匀、无需向量寄存器的值（标量即可）。
_UNIFORM_RECIPES = ("imm", "uniform_load", "uniform_expr")


def _for_kind_name(kind) -> str:
    """``For.kind`` 在不同路径下可能是 ``ForKind`` 枚举，也可能退化成 ``int``。"""
    name = getattr(kind, "name", None)
    if name is not None:
        return str(name).lower()
    try:
        return tirx.ForKind(int(kind)).name.lower()
    except Exception:  # pragma: no cover - defensive
        return str(kind)


class Unsupported(Exception):
    """Raised when a nest cannot be vectorized; carries the diagnostic reason(s).

    ``diagnostics`` 收集本次规划过程中**全部卡点**（每根候选轴的分类摘要与失败
    原因），这样上层能一次性看到所有阻碍，而不是只拿到第一个失败原因（改进 7）。
    :func:`build_plan` 会填充它；直接 ``str(err)`` 仍然是单行摘要。
    """

    def __init__(self, message: str, *, diagnostics: list[str] | None = None):
        super().__init__(message)
        self.diagnostics: list[str] = list(diagnostics or [])


# ---------------------------------------------------------------------------
# 1. SCoP：迭代域
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    """一根迭代轴：``0 <= var < extent``。

    ``kind`` 区分 ``T.Parallel`` 轴与 ``T.serial`` 轴：**直接内嵌且干净**的 serial
    循环也会被吸收进迭代域（见 :func:`_split_nest` / :func:`_serial_is_clean`），
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

    @property
    def is_index_vector(self) -> bool:
        """是否必须由「每 lane 索引向量」通道发射（gather/scatter）。"""
        return self in (
            AccessClass.STRIDED,
            AccessClass.PERMUTE,
            AccessClass.PIECEWISE,
            AccessClass.INDIRECT,
        )


# ---------------------------------------------------------------------------
# 2. 仿射原语
# ---------------------------------------------------------------------------


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


# 索引向量通道允许的整数算子（``FloorDiv`` / ``FloorMod`` 另有「2 的幂」约束）。
_INDEX_VALUE_OPS = {
    "Var",
    "IntImm",
    "Add",
    "Sub",
    "Mul",
    "FloorDiv",
    "FloorMod",
    "Cast",
    "Min",
    "Max",
    "Neg",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "ShiftLeft",
    "ShiftRight",
}


def index_vector_reason(expr, analyzer: arith.Analyzer | None = None) -> str | None:
    """判断元素偏移 ``expr`` 能否在「每 lane 索引向量」通道里构造出来。

    能构造则返回 ``None``；否则返回一句诊断原因（改进 4 的 ``TRUE_RANDOM`` 依据）。
    通道的构造能力 = 索引值 DAG 的构造能力（:func:`build_value` 的整数子集）：

    * 只允许整数算术（仿射 + ``Min`` / ``Max`` / 位运算）；
    * ``FloorDiv`` / ``FloorMod`` 只允许 **正的 2 的幂常量** 除数（与值算子
      strength-reduction 一致，见 :func:`_floordiv_mod`）；
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


# ---------------------------------------------------------------------------
# 3. 访问分类
# ---------------------------------------------------------------------------


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

    * 下标含 ``BufferLoad``（``A[B[i]]``）→ :attr:`AccessClass.INDIRECT`，
      索引向量来自被装载的寄存器；
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
        if indirect:
            info.per_axis[axis.index] = (AccessClass.INDIRECT, None)
            continue
        delta = finite_difference(offset, axis.var, analyzer)
        kind, stride = classify_delta(delta, analyzer)
        if kind == AccessClass.PIECEWISE and piecewise_reason is not None:
            # Δ 非常数，但索引向量通道构造不出来 → 真正的"无结构"访问。
            kind, stride = AccessClass.TRUE_RANDOM, None
            info.reason = piecewise_reason
        info.per_axis[axis.index] = (kind, stride)
    return info


# ---------------------------------------------------------------------------
# 4. body 规范化 —— 多语句 / 分支 → 有序的写列表
# ---------------------------------------------------------------------------


@dataclass
class StoreSpec:
    """一次写入：写到哪、写什么、以及可选的谓词（谓词化 store）。"""

    buffer: object
    indices: tuple
    value: tirx.PrimExpr
    mask: tirx.PrimExpr | None = None


def _collect_aliases(stmt: tirx.Stmt) -> dict:
    """收集 ``Bind`` 引入的标量别名（如 ``acc = a[i,j] * 2``）。

    buffer 声明用的 Bind（值是 Buffer / 类型构造 Call）不参与替换，
    只把真正的「值绑定」内联掉，这样 ``§3 多行语句`` 里的标量中间量会
    自然折叠进最终表达式。
    """
    aliases: dict = {}

    def visit(node):
        if type(node).__name__ != "Bind":
            return
        value = node.value
        if not isinstance(value, tirx.PrimExpr) or isinstance(value, tirx.Call):
            return
        aliases[node.var] = value

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return aliases


def _apply_aliases(expr: tirx.PrimExpr, aliases: dict) -> tirx.PrimExpr:
    if not aliases:
        return expr
    for _ in range(8):  # 别名链通常很短，限定迭代次数避免环
        replaced = tirx.stmt_functor.substitute(expr, aliases)
        if replaced.same_as(expr):
            break
        expr = replaced
    return expr


def _is_truthy_else(case) -> bool:
    return case is None or (isinstance(case, tirx.Evaluate) and _is_zero(case.value))


def normalize_body(
    stmt: tirx.Stmt,
    aliases: dict,
    out: list | None = None,
    mask: tirx.PrimExpr | None = None,
) -> list[StoreSpec]:
    """把 body 规范化成有序 :class:`StoreSpec` 列表。

    规则（与算子无关）：

    * ``SeqStmt`` → 顺序展开（§3 多行语句天然支持）；
    * ``BufferStore`` → 一条写；
    * ``IfThenElse`` 单臂 → 谓词化写（``mask &= cond``）；
    * ``IfThenElse`` 双臂且两臂写同一位置 → 合并成 ``select(cond, a, b)``；
    * 其它（**非直接内嵌 / 不干净**的内层 serial loop、原子调用、跨迭代标量状态）
      → 明确拒绝（干净的 serial 循环已在 :func:`_split_nest` 里被吸收成轴）。
    """
    result: list[StoreSpec] = [] if out is None else out

    if isinstance(stmt, tirx.SeqStmt):
        for child in stmt:
            normalize_body(child, aliases, result, mask)
        return result

    if isinstance(stmt, tirx.BufferStore):
        value = _apply_aliases(stmt.value, aliases)
        result.append(StoreSpec(stmt.buffer, tuple(stmt.indices), value, mask))
        return result

    if isinstance(stmt, tirx.BufferLoad):
        # 裸读不求值：语义上无用，直接丢弃。
        return result

    if isinstance(stmt, tirx.Evaluate):
        call = stmt.value
        if isinstance(call, tirx.Call) and _is_atomic_name(_call_tail(call)):
            raise Unsupported("atomic op (T.atomic_add) is not supported yet: no vector+scalar atomic recipe")
        return result

    if isinstance(stmt, tirx.IfThenElse):
        condition = _apply_aliases(stmt.condition, aliases)
        then_stores: list[StoreSpec] = []
        normalize_body(stmt.then_case, aliases, then_stores, None)
        else_stores: list[StoreSpec] = []
        if not _is_truthy_else(stmt.else_case):
            normalize_body(stmt.else_case, aliases, else_stores, None)

        if not else_stores:
            # 单臂 if → 谓词化：每个写都带上 (cond & mask)
            combined = condition if mask is None else tirx.And(mask, condition)
            for spec in then_stores:
                spec.mask = combined if spec.mask is None else tirx.And(combined, spec.mask)
                result.append(spec)
            return result

        if len(then_stores) == 1 and len(else_stores) == 1:
            then_spec, else_spec = then_stores[0], else_stores[0]
            if then_spec.buffer.same_as(else_spec.buffer) and _same_indices(then_spec.indices, else_spec.indices):
                merged = tirx.Select(condition, then_spec.value, else_spec.value)
                for spec in (then_spec, else_spec):
                    if spec.mask is not None:
                        raise Unsupported("masked branch merging is not supported")
                result.append(StoreSpec(then_spec.buffer, then_spec.indices, merged, mask))
                return result

        raise Unsupported("divergent branch arms: an if/else whose arms write different locations cannot be vectorized (方案 §8.1)")

    if isinstance(stmt, tirx.For):
        raise Unsupported(
            f"nested {_for_kind_name(stmt.kind)} loop inside T.Parallel is not affine-vectorizable "
            "(方案 §6): only directly-nested, dependence-free serial loops are absorbed as axes"
        )

    if type(stmt).__name__ == "Bind":
        return result

    raise Unsupported(f"unsupported statement in T.Parallel body: {type(stmt).__name__}")


def _same_indices(lhs: tuple, rhs: tuple) -> bool:
    # 结构相等：``same_as`` 是指针身份比较，对 ``IntImm(0)`` 之类的常量下标会
    # 因对象不同而误判为不等（``acc[0] = acc[0] + x`` 的自读判定因此失效）。
    return len(lhs) == len(rhs) and all(tirx.analysis.expr_deep_equal(a, b) for a, b in zip(lhs, rhs))


def _seq(stmts: list) -> tirx.Stmt | None:
    """把语句列表折成 ``SeqStmt``（空 → ``None``，单元素 → 元素本身）。"""
    if not stmts:
        return None
    if len(stmts) == 1:
        return stmts[0]
    return tirx.SeqStmt(stmts)


def _extern_name(arg) -> str | None:
    """``call_extern`` 的符号名参数可能是 ``str`` 或 ``StringImm``。"""
    if isinstance(arg, str):
        return arg
    value = getattr(arg, "value", None)
    return value if isinstance(value, str) else None


def _call_tail(call: tirx.Call) -> str:
    op = getattr(call, "op", None)
    name = getattr(op, "name", "") or ""
    # ``call_extern`` 把真实符号名放在第一个参数里（如 ``tl.atomic_add``）。
    if "call_extern" in name and call.args:
        extern = _extern_name(call.args[0])
        if extern:
            return extern.split(".")[-1].lower()
    return name.split(".")[-1].lower()


# ---------------------------------------------------------------------------
# 5. 值 DAG
# ---------------------------------------------------------------------------


@dataclass
class ValueNode:
    """一个值节点：recipe 是原语名，args 是操作数。"""

    recipe: str
    dtype: str
    args: list = field(default_factory=list)
    access: AccessInfo | None = None
    scalar: tirx.PrimExpr | None = None
    detail: str = ""

    @property
    def is_uniform(self) -> bool:
        """整个子表达式是否 lane 内均匀（不需要向量寄存器）。

        ``all([])`` 为真，所以**必须**在递归之前先把所有「无 args 的叶子」挡掉：
        ``lane_index`` / ``vbrc``（有 args，显式覆盖）之外，``vload`` / ``vload_strided`` /
        ``gather`` 都是无 args 叶子，它们承载的是**逐 lane 变化**的向量值，一旦被真空判为
        均匀就会喂进 ``vadds``/``vmuls`` 之类的标量-向量变体，在发射期报
        ``cannot scalarize vector value``。
        """
        if self.recipe in ("lane_index", "vbrc"):
            return False
        if self.recipe in _UNIFORM_RECIPES:
            return True
        if not self.args:  # 其余无 args 叶子（vload / vload_strided / gather）恒非均匀
            return False
        return all(arg.is_uniform for arg in self.args)


def _dtype_of(expr) -> str:
    dtype = expr.dtype
    if isinstance(dtype, str):
        return dtype
    return str(dtype)


@dataclass
class _ValueContext:
    axes: list[Axis]
    vector_axis: Axis
    lanes: int
    analyzer: arith.Analyzer
    vector_dtype: str


def build_value(expr, ctx: _ValueContext, is_global_ok: bool = False) -> ValueNode:
    """把标量表达式 lift 成向量值节点（与代码结构无关的递归）。

    决策仍然只依赖「沿向量轴的差分」：

    * ``BufferLoad`` 的 ``Δ_v == 1`` → ``vload``（连续）；
    * ``Δ_v == 0`` → ``uniform_load``（标量读一次 + ``vdup`` 广播）；
    * ``Δ_v == k`` → ``vload_strided``（发射成 ``vgather2``）；
    * 间接（下标含 ``BufferLoad``）→ 明确拒绝；
    * 循环变量本身 → ``lane_index``（``vci``）。
    """
    if isinstance(expr, tirx.BufferLoad):
        info = classify_access(expr.buffer, tuple(expr.indices), ctx.axes, ctx.analyzer, is_store=False)
        return load_node(info, ctx, is_global_ok)
    if isinstance(expr, tirx.Var):
        if ctx.vector_axis.var.same_as(expr):
            # 向量轴变量 == lane id：``vci``（元素类型取轴变量自身的整型）。
            return ValueNode("lane_index", _dtype_of(expr), scalar=expr, detail="vci(offset=1)")
        if any(axis.var.same_as(expr) for axis in ctx.axes):
            # 非向量轴的平行变量在 lane 内是常量（外层循环变量）。
            return ValueNode("imm", _dtype_of(expr), scalar=expr, detail="outer-axis-var")
        return ValueNode("imm", _dtype_of(expr), scalar=expr, detail="capture")
    if isinstance(expr, (tirx.IntImm, tirx.FloatImm)):
        return ValueNode("imm", _dtype_of(expr), scalar=expr)

    if isinstance(expr, tirx.Call):
        return _build_call(expr, ctx, is_global_ok)

    node_name = type(expr).__name__
    name_l = node_name.lower()
    if node_name in ("FloorDiv", "FloorMod") and _arity(expr) == 2:
        return _floordiv_mod(node_name, expr, ctx, is_global_ok)
    if name_l in _COMPARE_PRED and _arity(expr) == 2:
        args = [build_value(expr.a, ctx, is_global_ok), build_value(expr.b, ctx, is_global_ok)]
        return _compare(args, _COMPARE_PRED[name_l], ctx, expr)
    if name_l in _BINARY_RECIPE and _arity(expr) == 2:
        args = [build_value(expr.a, ctx, is_global_ok), build_value(expr.b, ctx, is_global_ok)]
        return _elementwise(_BINARY_RECIPE[name_l], expr, args, ctx)
    if name_l in _UNARY_RECIPE and _arity(expr) == 1:
        args = [build_value(expr.a, ctx, is_global_ok)]
        return _elementwise(_UNARY_RECIPE[name_l], expr, args, ctx)
    if name_l in _BOOL_BINARY_RECIPE and _arity(expr) == 2:
        args = [build_value(expr.a, ctx, is_global_ok), build_value(expr.b, ctx, is_global_ok)]
        return _bool_node(_BOOL_BINARY_RECIPE[name_l], expr, args)
    if name_l in _BOOL_UNARY_RECIPE and _arity(expr) == 1:
        args = [build_value(expr.a, ctx, is_global_ok)]
        return _bool_node(_BOOL_UNARY_RECIPE[name_l], expr, args)

    if isinstance(expr, tirx.Select):
        args = [
            build_value(expr.condition, ctx, is_global_ok),
            build_value(expr.true_value, ctx, is_global_ok),
            build_value(expr.false_value, ctx, is_global_ok),
        ]
        if all(arg.is_uniform for arg in args):
            return _uniform(expr)
        return _select_node(args, ctx)

    if node_name == "Cast":
        arg = build_value(expr.value, ctx, is_global_ok)
        if arg.is_uniform:
            return _uniform(expr)
        # 目标元素类型（去掉 lane 维），发射时交给 ``T.simd.vcvt`` 决定返回 lane 数。
        return ValueNode("vcvt", ctx.vector_dtype, [arg], detail=str(_dtype_of(expr)).split("x", 1)[0])

    raise Unsupported(f"unsupported value expression: {node_name} ({expr})")


def _uniform(expr) -> ValueNode:
    """lane 内均匀的子表达式：直接留用原始标量表达式，发射时一次求值 + ``vdup``。"""
    return ValueNode("uniform_expr", _dtype_of(expr), scalar=expr, detail="lane-uniform")


def _select_node(args: list, ctx: _ValueContext) -> ValueNode:
    """``select(cond, a, b)`` → ``vsel``。

    条件有两种来源，发射端分别处理（见 :meth:`_Emitter._build` 的 ``vsel`` 分支）：

    * **逐 lane 谓词**（``cond`` 随向量轴变化，如 ``vcmp``）：直接作为 ``vsel`` 的 mask；
    * **循环不变条件**（``cond`` 沿向量轴均匀）：在发射端广播成全 lane 谓词
      ``update_mask(cond * lanes)``——整个 chunk 同真同假，因此不需要把分支提升到
      标量层，也不需要逐 lane 比较。这正是 §1 Case II 的三元写法。
    """
    detail = "uniform-cond select" if args[0].is_uniform else "predicated select"
    return ValueNode("vsel", ctx.vector_dtype, args, detail=detail)


def _bool_node(recipe: str, expr, args: list) -> ValueNode:
    if all(arg.is_uniform for arg in args):
        return _uniform(expr)
    return ValueNode(recipe, _MASK_DTYPE, args)


def _compare(args: list, predicate: str, ctx: _ValueContext, expr) -> ValueNode:
    """比较：一侧 lane 均匀 → ``vcmps``，否则 ``vcmp``。结果类型是谓词。

    标量形式要求操作数顺序为 ``(vector, scalar)``；交换操作数时谓词必须取反，
    否则 ``i < j``（i 均匀）会被错误地发成 ``j < i``。
    """
    lhs_uniform, rhs_uniform = args[0].is_uniform, args[1].is_uniform
    if lhs_uniform and rhs_uniform:
        return _uniform(expr)
    if lhs_uniform:
        return ValueNode("vcmps", _MASK_DTYPE, [args[1], args[0]], detail=_COMPARE_FLIP[predicate])
    if rhs_uniform:
        return ValueNode("vcmps", _MASK_DTYPE, [args[0], args[1]], detail=predicate)
    return ValueNode("vcmp", _MASK_DTYPE, args, detail=predicate)


def _arity(expr) -> int:
    return len([f for f in ("a", "b") if hasattr(expr, f) and getattr(expr, f) is not None])


def _elementwise(recipe: str, expr, args: list, ctx: _ValueContext) -> ValueNode:
    """二元/一元逐元素算子：若一侧 lane 均匀，优先用标量-向量指令。"""
    if all(arg.is_uniform for arg in args):
        return _uniform(expr)
    if len(args) == 2:
        lhs_uniform, rhs_uniform = args[0].is_uniform, args[1].is_uniform
        if lhs_uniform or rhs_uniform:
            scalar_recipe = _VEC_SCALAR_BINARY.get(recipe)
            if scalar_recipe is not None:
                ordered = [args[1], args[0]] if lhs_uniform else [args[0], args[1]]
                return ValueNode(scalar_recipe, ctx.vector_dtype, ordered, detail=f"scalar {recipe}->{scalar_recipe}")
            if lhs_uniform:
                return ValueNode(recipe, ctx.vector_dtype, [broadcast(args[0], ctx), args[1]])
            return ValueNode(recipe, ctx.vector_dtype, [args[0], broadcast(args[1], ctx)])
    return ValueNode(recipe, ctx.vector_dtype, args)


def broadcast(node: ValueNode, ctx: _ValueContext) -> ValueNode:
    if node.recipe == "vbrc":
        return node
    return ValueNode("vbrc", ctx.vector_dtype, [node], detail="broadcast uniform")


def _floordiv_mod(node_name: str, expr, ctx: _ValueContext, is_global_ok: bool) -> ValueNode:
    """``FloorDiv`` / ``FloorMod`` 作*值*算子（方案 §5.4(a)）。

    SIMD 层没有整型 ``vdiv`` / ``vmod``（``vdiv`` 是浮点 SFU），所以对**2 的幂**
    常量除数做精确的 strength-reduction：

    * ``x // 2^k`` → ``vshrs(x, k)``：算术右移对负数即向下取整，与 ``FloorDiv`` 同义；
    * ``x % 2^k`` → ``vand(x, 2^k - 1)``：补码按位与正好给出非负余数，与 ``FloorMod`` 同义。

    非 2 的幂常量除数尚未做 magic-number 化简（需 64 位中间量 ``vmull``），
    非常量除数需要硬件整型除法内建——两者都**明确拒绝**而不静默降级。
    两侧都 lane 均匀时整式退化成一个标量，不产生向量指令。
    """
    numer = build_value(expr.a, ctx, is_global_ok)
    denom = build_value(expr.b, ctx, is_global_ok)
    if numer.is_uniform and denom.is_uniform:
        return _uniform(expr)

    divisor = _const_int(expr.b, ctx.analyzer)
    if divisor is None or divisor <= 0 or not denom.is_uniform:
        raise Unsupported(f"{node_name} as a value operator needs a positive compile-time constant divisor (got {expr.b})")
    if divisor & (divisor - 1):
        raise Unsupported(
            f"{node_name} by non-power-of-two constant {divisor} is not strength-reduced yet: "
            "the SIMD layer has no integer vdiv/vmod (方案 §5.4(a))"
        )

    elem = _elem_dtype(numer.dtype)
    if divisor == 1:
        if node_name == "FloorDiv":
            return numer
        return ValueNode("imm", elem, scalar=tirx.IntImm(elem, 0))
    if node_name == "FloorDiv":
        shift = divisor.bit_length() - 1
        return ValueNode("vshrs", _vec_dtype(elem), [numer], scalar=tirx.IntImm("int32", shift), detail=f"//{divisor}")
    remainder = ValueNode("imm", elem, scalar=tirx.IntImm(elem, divisor - 1))
    mask_vec = ValueNode("vbrc", _vec_dtype(elem), [remainder], detail="broadcast uniform")
    return ValueNode("vand", _vec_dtype(elem), [numer, mask_vec], detail=f"%{divisor}")


def index_value(info: AccessInfo, ctx: _ValueContext) -> ValueNode:
    """把一次访存的**绝对元素偏移**构造成整数索引向量（改进 1 的通道）。

    直接在整数上下文里对 ``info.offset`` 做 :func:`build_value`：向量轴变量会落到
    ``lane_index``（``vci(lane_base)`` = 该 chunk 的 lane 绝对值），外层轴变量落到
    lane 内均匀的 ``imm``。于是得到的向量就是 ``offset(..., v = lane_base + lane, ...)``，
    即每 lane 的元素偏移——正好是 ``vgather2`` / ``vscatter`` 需要的索引寄存器。

    ``info.offset`` 已经过 :func:`index_vector_reason` 校验（非常数 Δ 且可构造），
    因此这里不会撞上非 2 的幂除法或数据相关下标。
    """
    index_ctx = _ValueContext(
        axes=ctx.axes,
        vector_axis=ctx.vector_axis,
        lanes=ctx.lanes,
        analyzer=ctx.analyzer,
        vector_dtype=_vec_dtype("int32"),
    )
    return build_value(info.offset, index_ctx)


def load_node(info: AccessInfo, ctx: _ValueContext, is_global_ok: bool) -> ValueNode:
    """把一次已分类的访问变成值节点（连续/广播/步长/置换/分片/间接）。

    常数 Δ 走连续 / 广播 / 步长；非常数 Δ 与间接下标统一走**每 lane 索引向量**
    通道（:attr:`AccessClass.is_index_vector`），发射成 ``vgather2``。真正无结构的
    ``TRUE_RANDOM`` 明确拒绝并带上 :func:`index_vector_reason` 的诊断。
    """
    if info.in_global and not is_global_ok:
        raise Unsupported(
            "global-memory operand inside T.Parallel (方案 §0 Case II): needs an explicit UB staging copy before the parallel nest"
        )
    kind, stride = info.axis_class(ctx.vector_axis)
    if kind == AccessClass.CONTIGUOUS:
        return ValueNode("vload", ctx.vector_dtype, access=info)
    if kind == AccessClass.UNIFORM:
        return ValueNode("uniform_load", _dtype_of_scalar(info), access=info, detail="lane-uniform")
    if kind == AccessClass.STRIDED:
        return ValueNode("vload_strided", ctx.vector_dtype, access=info, detail=f"stride={stride}")
    if kind in (AccessClass.PERMUTE, AccessClass.PIECEWISE, AccessClass.INDIRECT):
        return ValueNode("gather", ctx.vector_dtype, [index_value(info, ctx)], access=info, detail=kind.value)
    raise Unsupported(
        f"non-affine access with respect to axis '{ctx.vector_axis.name}': "
        f"the per-lane index vector is not constructible ({info.reason or 'unknown reason'})"
    )


def _dtype_of_scalar(info: AccessInfo) -> str:
    return str(info.buffer.dtype)


def _build_call(expr: tirx.Call, ctx: _ValueContext, is_global_ok: bool) -> ValueNode:
    tail = _call_tail(expr)
    if _is_atomic_name(tail):
        raise Unsupported("atomic op inside T.Parallel is not supported (方案 §5)")
    if tail in ("if_then_else", "select"):
        args = [build_value(expr.args[i], ctx, is_global_ok) for i in range(3)]
        if all(arg.is_uniform for arg in args):
            return _uniform(expr)
        return _select_node(args, ctx)
    if tail in _COMPARE_PRED and len(expr.args) == 2:
        args = [build_value(arg, ctx, is_global_ok) for arg in expr.args]
        return _compare(args, _COMPARE_PRED[tail], ctx, expr)
    if tail in _BOOL_BINARY_RECIPE and len(expr.args) == 2:
        args = [build_value(arg, ctx, is_global_ok) for arg in expr.args]
        return _bool_node(_BOOL_BINARY_RECIPE[tail], expr, args)
    if tail in _BOOL_UNARY_RECIPE and len(expr.args) == 1:
        args = [build_value(expr.args[0], ctx, is_global_ok)]
        return _bool_node(_BOOL_UNARY_RECIPE[tail], expr, args)
    if tail in _BINARY_RECIPE and len(expr.args) == 2:
        args = [build_value(arg, ctx, is_global_ok) for arg in expr.args]
        return _elementwise(_BINARY_RECIPE[tail], expr, args, ctx)
    if tail in _UNARY_RECIPE and len(expr.args) == 1:
        args = [build_value(expr.args[0], ctx, is_global_ok)]
        return _elementwise(_UNARY_RECIPE[tail], expr, args, ctx)
    raise Unsupported(f"unsupported intrinsic call in T.Parallel body: {tail}")


# ---------------------------------------------------------------------------
# 6. 规划
# ---------------------------------------------------------------------------


@dataclass
class StorePlan:
    spec: StoreSpec
    access: AccessInfo
    value: ValueNode
    mask: ValueNode | None = None
    kind: AccessClass = AccessClass.CONTIGUOUS
    index: ValueNode | None = None  # 分片/置换/间接写的每 lane 索引向量（scatter）
    reduce: ReduceInfo | None = None  # 归约段里**逐 store** 的归约描述（方案 §2 Case II）


@dataclass
class ReduceInfo:
    """归约：store 沿某轴 Δ==0 而取值沿该轴变化（方案 §2 Case II）。

    ``op``/``recipe`` 决定用哪种跨 lane 原语；``varying`` 是**已剥离旧值**的逐 lane
    取值（lane 均匀加数保留在内——跨 lane 归约天然正确处理）；``accumulate`` 表示
    这是累加写 ``dst op= x``，发射时末尾并入目标里原有的值。

    ``VectorPlan.reduce`` 只是「本段是归约段、代表元是哪条 store」的段级标记；真正
    逐条发射用的是 :attr:`StorePlan.reduce`（一个段内可有多条归约，缺陷 D6）。
    """

    axis: Axis
    op: str
    recipe: str = "vcadd"
    varying: ValueNode | None = None
    accumulate: bool = False


# 逐元素算子 → 跨 lane 归约原语（§2 Case II 把内层轴压成标量）。
_REDUCE_RECIPE = {"vadd": "vcadd", "vmax": "vcmax", "vmin": "vcmin"}
# 归约原语 → 累加/合并用的逐元素算子（与上表互逆）。
_REDUCE_ELEMENT = {recipe: op for op, recipe in _REDUCE_RECIPE.items()}


@dataclass
class VectorPlan:
    """**段级**硬件无关向量执行计划（≈ 方案里的中间层 2 的分析部分）。

    一个「段」= 一个主体 + 可选 ``pre``/``post``（方案 §6 第二步）。:func:`build_plan`
    **恒**把它包成 :class:`VectorProgram` 的一段返回；只有等价性护栏 / 外部调用方才会
    直接持有它。
    """

    axes: list[Axis]
    vector_axis: Axis
    lanes: int
    chunks: int
    tail: int
    stores: list[StorePlan] = field(default_factory=list)
    # 方案 §6 第二步：与 serial 循环**同级**的前置/后置语句（沿向量轴均匀），
    # 每个外层迭代各执行一次，发射在 chunk 循环**之外**（顺序：pre → 主体 → post）。
    pre_stores: list[StorePlan] = field(default_factory=list)
    post_stores: list[StorePlan] = field(default_factory=list)
    reduce: ReduceInfo | None = None
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class VectorProgram:
    """``build_plan`` 的**统一返回类型**（恒返回，1..N 段）。

    单段（``T.Parallel`` 体只有一个主体，或一个主体 + 可选 ``pre``/``post``）时是
    「1 段 program」：``axes`` 只含提到最外层包一次的**并行轴前缀**（:func:`_outer_axes`），
    向量轴与其余轴仍由 ``segments[0]`` 自己包裹，因此调用方只需处理一种计划类型。

    体内出现**多个并列干净 ``T.serial`` 循环**时按源码顺序切成若干段，每段是一个
    :class:`VectorPlan`（各有自己的 serial 轴、向量轴、``pre``/``post`` 与归约）；各段
    共享 ``axes`` 里的外层并行轴（只在最外层包裹一次），段内发射保持源码顺序
    （``pre`` → 主体 → ``post``）。

    这是一个「逐段独立向量化 + 源序发射」模型：段间若存在依赖，该依赖必须沿各段向量轴
    均匀（如 ``m[i]``、``s[i]``），于是逐段向量化 + 顺序执行即可保持语义；段间出现跨地址
    真依赖（shift / stencil）时应写成单段（一个 serial 体），而不是并列两段。
    """

    axes: list[Axis]
    segments: list[VectorPlan] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


def _split_nest(stmt: tirx.Stmt, analyzer: arith.Analyzer) -> tuple[list[Axis], Axis | None, list[tirx.Stmt], tirx.Stmt, list[tirx.Stmt]]:
    """收集迭代域，并把 body 拆成 ``(pre, main, post)`` 三段。

    迭代域 = 连续的 ``T.Parallel`` 轴 + **直接的、干净的** ``T.serial`` 轴：

    * **直接内嵌链**（``for i in P: for j in serial: body``）—— 一路下行即可吸收；
    * **语句列表分段**（方案 §6 第二步）：当平行链结束后是一个 ``SeqStmt``，其中
      **唯一一个**干净的 ``T.serial`` 循环也可被吸收——它之前的兄弟语句是 ``pre``、
      之后的是 ``post``（都在 serial 轴**外层**、每个并行迭代执行一次），循环体则是
      被向量化的 ``main``。``pre``/``post`` 天然不含 serial 变量，故沿该轴 ``Δ==0``。

    serial 轴只有在 :func:`_serial_is_clean` 通过时才被吸收；否则保持原样，由
    :func:`normalize_body` 以「nested serial loop」明确拒绝——吸收是纯增量，不改变
    任何既有拒绝行为。

    返回 ``(axes, serial_axis, pre_stmts, main_stmt, post_stmts)``；
    ``serial_axis`` 是被吸收的那根 serial 轴（没有则为 ``None``）。
    """
    axes: list[Axis] = []
    serial_axis: Axis | None = None
    current: tirx.Stmt = stmt
    while isinstance(current, tirx.For):
        kind = current.kind
        if kind == tirx.ForKind.SERIAL:
            if not _serial_is_clean(current, analyzer)[0]:
                break
        elif kind != tirx.ForKind.PARALLEL:
            break
        axis = Axis(current.loop_var, current.extent, len(axes), kind)
        axes.append(axis)
        if kind == tirx.ForKind.SERIAL:
            serial_axis = axis
        current = current.body

    pre: list[tirx.Stmt] = []
    post: list[tirx.Stmt] = []
    if isinstance(current, tirx.SeqStmt):
        loop_positions = [k for k, child in enumerate(current) if isinstance(child, tirx.For)]
        # 仅当恰好有一个「干净的 serial 循环」时才分段；否则原样保留，交给
        # normalize_body 以「nested serial loop」拒绝（保持既有行为不变）。
        if len(loop_positions) == 1:
            pos = loop_positions[0]
            loop = current[pos]
            if loop.kind == tirx.ForKind.SERIAL and _serial_is_clean(loop, analyzer)[0]:
                serial_axis = Axis(loop.loop_var, loop.extent, len(axes), tirx.ForKind.SERIAL)
                axes.append(serial_axis)
                pre = list(current[:pos])
                post = list(current[pos + 1 :])
                current = loop.body

    if not any(axis.is_parallel for axis in axes):
        raise Unsupported("no T.Parallel axis found")
    return axes, serial_axis, pre, current, post


def _absorb_serial_chain(loop: tirx.For, analyzer: arith.Analyzer, base_index: int) -> tuple[list[Axis], tirx.Stmt] | None:
    """把 ``for j in serial: for l in serial: ...`` 这条**纯 serial 链**吸收成迭代轴。

    返回 ``(serial_axes, innermost_body)``；若链上任何一层不干净（:func:`_serial_is_clean`），
    或最内层仍是循环（含 ``T.Parallel``）则返回 ``None``——此时交给既有单段路径按原样处理，
    保持「nested serial loop」等既有拒绝行为不变。
    """
    if not (isinstance(loop, tirx.For) and loop.kind == tirx.ForKind.SERIAL):
        return None
    axes: list[Axis] = []
    current: tirx.Stmt = loop
    while isinstance(current, tirx.For) and current.kind == tirx.ForKind.SERIAL:
        if not _serial_is_clean(current, analyzer)[0]:
            return None
        axes.append(Axis(current.loop_var, current.extent, base_index + len(axes), tirx.ForKind.SERIAL))
        current = current.body
    if isinstance(current, tirx.For):
        return None
    return axes, current


@dataclass
class _SegmentSpecs:
    """一段（serial 链 + 其 ``pre``/``post``）**规范化后**的分析输入。

    单段路径（:func:`build_plan`）与多段路径（:func:`_build_segment`）共用这份
    样板（:func:`_prepare_segment`）：规范化 body → 定 lanes → 逐轴分类，之后各自
    决定向量轴。这样两条路径不会各自维护一份「elem/lanes/infos」推导。
    """

    axes: list[Axis]
    serial_axis: Axis | None
    pre_specs: list[StoreSpec]
    pre_infos: list[AccessInfo]
    main_specs: list[StoreSpec]
    main_infos: list[AccessInfo]
    post_specs: list[StoreSpec]
    post_infos: list[AccessInfo]
    elem: str
    lanes: int

    @property
    def all_infos(self) -> list[AccessInfo]:
        return [*self.pre_infos, *self.main_infos, *self.post_infos]

    @property
    def all_specs(self) -> list[StoreSpec]:
        return [*self.pre_specs, *self.main_specs, *self.post_specs]

    def plan_for(self, vector_axis: Axis, analyzer: arith.Analyzer) -> VectorPlan:
        return _build_plan_for_axis(
            self.axes,
            self.serial_axis,
            self.pre_specs,
            self.pre_infos,
            self.main_specs,
            self.main_infos,
            self.post_specs,
            self.post_infos,
            vector_axis,
            self.lanes,
            self.elem,
            analyzer,
        )


def _prepare_segment(
    axes: list[Axis],
    serial_axis: Axis | None,
    pre_stmts: list[tirx.Stmt],
    main_stmt: tirx.Stmt,
    post_stmts: list[tirx.Stmt],
    aliases: dict,
    analyzer: arith.Analyzer,
    lanes: int | None,
) -> _SegmentSpecs:
    """把一段的语句规范化成 :class:`_SegmentSpecs`（单段 / 多段路径共用）。

    ``lanes`` 只在调用方未指定时按元素位宽推导；``main`` 没有任何写语句时抛
    ``Unsupported``（与两条路径原先各自的检查同一处、同一消息）。
    """
    pre_specs = normalize_body(_seq(pre_stmts), aliases) if pre_stmts else []
    main_specs = normalize_body(main_stmt, aliases)
    post_specs = normalize_body(_seq(post_stmts), aliases) if post_stmts else []
    if not main_specs:
        raise Unsupported("T.Parallel body has no store to vectorize")

    elem = _elem_dtype(main_specs[0].buffer.dtype)
    if lanes is None:
        lanes = _vreg_lanes(elem)

    def _infos(specs):
        return [classify_access(spec.buffer, spec.indices, axes, analyzer, is_store=True) for spec in specs]

    return _SegmentSpecs(
        axes=axes,
        serial_axis=serial_axis,
        pre_specs=pre_specs,
        pre_infos=_infos(pre_specs),
        main_specs=main_specs,
        main_infos=_infos(main_specs),
        post_specs=post_specs,
        post_infos=_infos(post_specs),
        elem=elem,
        lanes=lanes,
    )


def _build_segment(
    axes: list[Axis],
    serial_axis: Axis,
    pre_stmts: list[tirx.Stmt],
    main_stmt: tirx.Stmt,
    post_stmts: list[tirx.Stmt],
    aliases: dict,
    analyzer: arith.Analyzer,
    lanes: int | None,
) -> VectorPlan:
    """为一段（一个 serial 链 + 其 ``pre``/``post``）规划出单段 :class:`VectorPlan`。

    向量轴固定为该段的**最内层 serial 轴**（外层并行轴的向量化尚未支持），因此
    :func:`_build_plan_for_axis` 里「``pre``/``post`` 只能沿 serial 轴向量化」的门禁天然满足。
    该段同样携带自己的诊断（改进 7），与单段路径的 ``plan.diagnostics`` 同构。
    """
    specs = _prepare_segment(axes, serial_axis, pre_stmts, main_stmt, post_stmts, aliases, analyzer, lanes)
    diagnostics = _axis_diagnostics(axes, specs.all_infos, _load_infos(axes, specs.all_specs, analyzer))
    plan = specs.plan_for(serial_axis, analyzer)
    diagnostics.append(f"segment on serial axis '{serial_axis.name}': forced vector axis (方案 §6 第三步)")
    plan.diagnostics = diagnostics
    return plan


def _try_build_program(
    stmt: tirx.Stmt,
    analyzer: arith.Analyzer,
    lanes: int | None,
    notes: list[str] | None = None,
) -> VectorProgram | None:
    """尝试把「多个并列干净 ``T.serial`` 循环」的并行体规划成 :class:`VectorProgram`。

    仅当并行链结束后是一个 ``SeqStmt``、其中含 **≥2 个**可吸收的 serial 链时才接手；
    否则返回 ``None``，让既有单段路径（:func:`_split_nest` / :func:`normalize_body`）原样
    处理——因此不改变任何既有拒绝行为（如「nested serial loop」）。段间语句组按源码顺序
    归属：段前语句组 → 该段的 ``pre``；末尾剩余语句组 → 末段的 ``post``。

    ``notes`` 是**段级失败原因**的出口（缺陷 B）：某一段构造失败时不能只 ``return None``
    把原因丢掉，而要记下来交给 :func:`build_plan` 合并进最终 ``Unsupported.diagnostics``
    ——否则用户只能看到回落单段路径后那句泛化的「nested serial loop」。
    """
    notes = notes if notes is not None else []
    parallel_axes: list[Axis] = []
    current: tirx.Stmt = stmt
    while isinstance(current, tirx.For) and current.kind == tirx.ForKind.PARALLEL:
        parallel_axes.append(Axis(current.loop_var, current.extent, len(parallel_axes), tirx.ForKind.PARALLEL))
        current = current.body
    if not parallel_axes or not isinstance(current, tirx.SeqStmt):
        return None

    items: list[tuple[str, object]] = []
    for child in current:
        if isinstance(child, tirx.For):
            chain = _absorb_serial_chain(child, analyzer, len(parallel_axes))
            if chain is None:
                return None  # 不可吸收：保留既有「nested serial loop」拒绝
            items.append(("seg", chain))
        else:
            items.append(("stmt", child))

    seg_positions = [k for k, (kind, _) in enumerate(items) if kind == "seg"]
    if len(seg_positions) < 2:
        return None

    aliases = _collect_aliases(current)
    segments: list[VectorPlan] = []
    previous = -1
    for order, pos in enumerate(seg_positions):
        pre = [items[k][1] for k in range(previous + 1, pos)]
        serial_axes, body = items[pos][1]
        post = [items[k][1] for k in range(pos + 1, len(items))] if order == len(seg_positions) - 1 else []
        try:
            segments.append(_build_segment([*parallel_axes, *serial_axes], serial_axes[-1], pre, body, post, aliases, analyzer, lanes))
        except Unsupported as err:
            notes.append(f"segment {order} (T.serial '{serial_axes[-1].name}'): {err}")
            notes.extend(f"  {note}" for note in err.diagnostics)
            return None
        previous = pos
    program = VectorProgram(axes=parallel_axes, segments=segments)
    program.diagnostics = [f"multi-segment program: {len(segments)} segments, source order preserved"]
    return program


def _is_cumulative_write(store: tirx.BufferStore) -> bool:
    """``dst = dst op x``（累积写）：取值里读到了**同址**的同一个 buffer。"""
    found = []

    def visit(node):
        if (
            isinstance(node, tirx.BufferLoad)
            and _same_buffer(node.buffer, store.buffer)
            and _same_indices(tuple(node.indices), tuple(store.indices))
        ):
            found.append(True)

    tirx.stmt_functor.post_order_visit(store.value, visit)
    return bool(found)


def _serial_is_clean(loop: tirx.For, analyzer: arith.Analyzer) -> tuple[bool, str]:
    """一个直接内嵌的 ``T.serial`` 循环能否安全吸收进迭代域（方案 §6 第一步）。

    吸收意味着把它的迭代当成一根候选轴（可被整体向量化），所以**必须排除循环携带
    依赖**。判据（保守但可靠，宁拒绝勿错发）：

    * **沿循环变量不变的写**（目的地址 ``Δ_var == 0``）：只有当它是**累积归约**
      （``s[i] = s[i] + a[i,j]``，取值同址回读）**且取值确实沿该轴变化**时才算合法；
      否则是 last-write-wins（``cnt[0] = a[i,j]`` / ``cnt[0] += 1``）→ 拒绝；
    * **跨迭代读**：循环体内某 buffer 被写又被读，且该读不是「与某条写同址」的
      累积自读（stencil ``t[i,j] = t[i,j-1] + ...``）→ 拒绝。

    S1（``c[i,j] = a[i,j]``）与 S3（``s[i] += a[i,j]``）都通过；S4/S5（共享标量、
    数据相关控制流）被挡在门外。
    """
    var = loop.loop_var
    stores: list = []
    loads: list = []

    def visit(node):
        if isinstance(node, tirx.BufferStore):
            stores.append(node)
        elif isinstance(node, tirx.BufferLoad):
            loads.append(node)

    tirx.stmt_functor.post_order_visit(loop.body, visit)

    axis = Axis(var, loop.extent, 0, tirx.ForKind.SERIAL)
    for store in stores:
        offset = linear_element_offset(store.buffer, tuple(store.indices), analyzer)
        if _const_int(finite_difference(offset, var, analyzer), analyzer) == 0:
            if not _is_cumulative_write(store):
                return False, f"write to {store.buffer.name} is invariant along '{var.name}' but not a reduction"
            if not _value_varies_along(store.value, axis, analyzer):
                return False, f"reduction into {store.buffer.name} does not vary along '{var.name}'"

    write_sites = [(store.buffer, tuple(store.indices)) for store in stores]
    for load in loads:
        if not any(_same_buffer(load.buffer, buf) for buf, _ in write_sites):
            continue
        if not any(_same_buffer(load.buffer, buf) and _same_indices(tuple(load.indices), idx) for buf, idx in write_sites):
            return False, f"cross-iteration read of {load.buffer.name} along '{var.name}'"
    return True, ""


# 访问形态 → 轴打分权重（改进 2）：写侧权重高于读侧，连续优于分片，分片优于广播。
_AXIS_WEIGHT = {
    AccessClass.CONTIGUOUS: 8,
    AccessClass.PIECEWISE: 4,
    AccessClass.STRIDED: 3,
    AccessClass.PERMUTE: 3,
    AccessClass.INDIRECT: 2,
    AccessClass.UNIFORM: 1,
    AccessClass.TRUE_RANDOM: 0,
}


def _collect_load_exprs(stores: list[StoreSpec]) -> list:
    """收集所有写语句的取值 / 谓词里出现的 ``BufferLoad``（读侧，用于联合打分）。"""
    found: list = []

    def visit(node):
        if isinstance(node, tirx.BufferLoad):
            found.append(node)

    for spec in stores:
        tirx.stmt_functor.post_order_visit(spec.value, visit)
        if spec.mask is not None:
            tirx.stmt_functor.post_order_visit(spec.mask, visit)
    return found


def _load_infos(axes: list[Axis], specs: list[StoreSpec], analyzer: arith.Analyzer) -> list[AccessInfo]:
    """写语句取值 / 谓词里所有读侧访存的逐轴分类（联合打分与诊断共用）。"""
    return [classify_access(load.buffer, tuple(load.indices), axes, analyzer, is_store=False) for load in _collect_load_exprs(specs)]


def rank_vector_axes(
    axes: list[Axis],
    stores: list[StoreSpec],
    store_infos: list[AccessInfo],
    load_infos: list[AccessInfo],
    analyzer: arith.Analyzer,
) -> list[Axis]:
    """按「联合可行性 + 读写加权」给候选向量轴排序（改进 2）。

    与原实现相比有三点变化：

    * **联合可行性**：候选轴只需「没有 store 是
      :attr:`AccessClass.TRUE_RANDOM`」且「至少一个 store 可写」——STRIDED /
      PIECEWISE / PERMUTE / INDIRECT 的 store 现在都能进候选（靠 ``vscatter``）；
    * **读加权**：读侧（:func:`_collect_load_exprs`）也计入打分，避免「写连续、
      读却四处乱跳」的轴被误选；
    * **轴回退**：返回的是**排序后的整条列表**，:func:`build_plan` 会依次尝试，
      某个轴的取值发射失败时自动退到次优轴，而不是直接判整个 nest 不可向量化。

    归约轴（store 沿该轴均匀且取值沿该轴变化）始终排在最前。
    """
    for axis in axes:
        if _is_reduce_axis(axis, stores, store_infos, analyzer):
            return [axis]

    ranked: list[tuple[int, Axis]] = []
    for axis in axes:
        if any(info.axis_class(axis)[0] == AccessClass.TRUE_RANDOM for info in store_infos):
            continue  # 写侧不可构造 → 该轴不可向量化
        if not any(info.axis_class(axis)[0] != AccessClass.TRUE_RANDOM for info in store_infos):
            continue
        ranked.append((_axis_score(store_infos, load_infos, axis), axis))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [axis for _, axis in ranked]


def _axis_weight_sum(infos: list[AccessInfo], axis: Axis) -> int:
    """一根轴上所有访存的形态权重之和（写侧用，见 :data:`_AXIS_WEIGHT`）。"""
    return sum(_AXIS_WEIGHT[info.axis_class(axis)[0]] for info in infos)


def _axis_score(store_infos: list[AccessInfo], load_infos: list[AccessInfo], axis: Axis) -> int:
    """一根候选轴的联合打分：写侧主序（×100）、读侧次之（×10）、同分取更内层。"""
    return 100 * _axis_weight_sum(store_infos, axis) + 10 * _axis_weight_sum(load_infos, axis) + axis.index


def _axis_diagnostics(axes: list[Axis], store_infos: list[AccessInfo], load_infos: list[AccessInfo]) -> list[str]:
    """每根轴的分类摘要（改进 7）：一次性暴露全部卡点，无论最终成功与否。

    成功时它记录「哪些轴是候选、打分多少、比选中的差在哪」；失败时它是失败的
    完整上下文，避免只报告第一个卡点。
    """
    lines: list[str] = []
    for axis in axes:
        label = f"axis '{axis.name}'[{axis.index}]{'' if axis.is_parallel else '(serial)'}"
        store_classes = [info.axis_class(axis)[0].value for info in store_infos]
        if any(info.axis_class(axis)[0] == AccessClass.TRUE_RANDOM for info in store_infos):
            reasons = sorted({info.reason for info in store_infos if info.reason})
            detail = f" ({'; '.join(reasons)})" if reasons else ""
            lines.append(f"{label}: store={store_classes} -> rejected (true-random store){detail}")
            continue
        load_classes = [info.axis_class(axis)[0].value for info in load_infos]
        lines.append(
            f"{label}: store={store_classes} load={load_classes} -> candidate (score={_axis_score(store_infos, load_infos, axis)})"
        )
    return lines


def _value_varies_along(expr, axis: Axis, analyzer: arith.Analyzer) -> bool:
    """取值是否随该轴变化（直接或间接通过 buffer 访问）。"""
    varies = []

    def visit(node):
        if isinstance(node, tirx.BufferLoad):
            info = classify_access(node.buffer, tuple(node.indices), [axis], analyzer, is_store=False)
            if info.axis_class(axis)[0] != AccessClass.UNIFORM:
                varies.append(True)

    tirx.stmt_functor.post_order_visit(expr, visit)
    return any(axis.var.same_as(v) for v in _free_vars(expr)) or bool(varies)


def _is_reduce_axis(
    axis: Axis,
    stores: list[StoreSpec],
    store_infos: list[AccessInfo],
    analyzer: arith.Analyzer,
) -> bool:
    """第一层归约判据（纯 Δ 分类，尚未建值 DAG）：store 全部沿 ``axis`` ``Δ==0``
    且至少一条取值沿该轴变化。

    :func:`rank_vector_axes`（轴排序 / 提前返回）与 :func:`_classify_reduce`（段级归约
    判定）**共用**这一处判据（缺陷 D13：原先两处各写一份、口径可能不一致）；单条 store
    是否真是累加形式再由 :func:`_plan_reduce` 精化。
    """
    return all(info.axis_class(axis)[0] == AccessClass.UNIFORM for info in store_infos) and any(
        _value_varies_along(spec.value, axis, analyzer) for spec in stores
    )


_REDUCE_OPS = {"Add": "vadd", "Max": "vmax", "Min": "vmin"}


def _reduce_op(stores: list[StoreSpec]) -> str:
    """归约算子名：只认 ``Add`` / ``Max`` / ``Min``。

    未识别的算子（如 ``Mul`` 的**乘积归约**）必须显式拒绝——旧实现默认落到 ``vadd``，
    会把它静默发射成求和，产出**错误结果**且不报任何诊断（缺陷 D1）。
    """
    for spec in stores:
        op = _REDUCE_OPS.get(type(spec.value).__name__)
        if op is not None:
            return op
    names = ", ".join(sorted({type(spec.value).__name__ for spec in stores})) or "<empty>"
    raise Unsupported(f"unsupported reduction operator(s) {names} (方案 §2 Case II): only Add/Max/Min reductions are supported")


def build_plan(stmt: tirx.Stmt, *, lanes: int | None = None) -> VectorProgram:
    """主入口：``T.Parallel`` 嵌套 → :class:`VectorProgram`（1..N 段）。

    伪代码（完整版见模块 docstring）::

        axes, serial_axis, pre, main, post = _split_nest(stmt)     # SCoP 迭代域 + 分段
        specs  = _prepare_segment(axes, ..., pre, main, post, ...) # 规范化 body + 逐轴分类
        ranked = rank_vector_axes(axes, specs.main_specs, specs.all_infos, load_infos, a)
        for v in ranked:                                           # 轴回退：逐个尝试
            try:    return specs.plan_for(v, analyzer)             # Δ 分类的递归
            except Unsupported: continue

    当并行体内出现**多个并列干净 ``T.serial`` 循环**时，先用 :func:`_try_build_program`
    走「逐段独立向量化」的多段路径；不适用时回落到上面的单段路径。

    **返回类型统一为** :class:`VectorProgram`：单段路径的结果会被包成「1 段 program」，
    其中 ``program.axes`` 只含**提到最外层的并行轴前缀**（向量轴与其余轴仍由该段自己
    包裹，见 :func:`_outer_axes`）。这样调用方只需处理一种计划类型，而「哪些轴在最外层
    包一次」这条信息与多段路径完全一致。

    ``lanes`` 默认由元素位宽唯一决定（``2048 / elem_bits``，与 ``T.simd`` 的
    ``vreg`` 宽度一致），因此一般不需要显式传入。
    """
    analyzer = arith.Analyzer()
    notes: list[str] = []  # 多段路径的段级失败原因（缺陷 B：不吞掉）
    program = _try_build_program(stmt, analyzer, lanes, notes)
    if program is not None:
        return program
    try:
        axes, serial_axis, pre_stmts, main_stmt, post_stmts = _split_nest(stmt, analyzer)
        pieces = [*pre_stmts, main_stmt, *post_stmts]
        aliases = _collect_aliases(_seq(pieces))
        specs = _prepare_segment(axes, serial_axis, pre_stmts, main_stmt, post_stmts, aliases, analyzer, lanes)
    except Unsupported as err:
        # 回落路径在这一步就拒绝（典型：并列多 serial 循环 → 「nested serial loop」）时，
        # 也要把多段路径试过但失败的段级原因带上（缺陷 B），否则只剩一句泛化消息。
        if notes:
            raise Unsupported(str(err), diagnostics=[*notes, *(err.diagnostics or [])]) from err
        raise
    all_infos = specs.all_infos
    load_infos = _load_infos(axes, specs.all_specs, analyzer)
    ranked = rank_vector_axes(axes, specs.main_specs, all_infos, load_infos, analyzer)
    # 改进 7：先记录每根轴的分类；多段路径试过但失败的原因一并带上（缺陷 B）。
    diagnostics = [*notes, *_axis_diagnostics(axes, all_infos, load_infos)]
    if not ranked:
        raise Unsupported(
            "no vectorizable axis: every store is true-random (its index cannot be built in the index-vector channel) on all axes",
            diagnostics=diagnostics,
        )

    errors: list[str] = []
    for vector_axis in ranked:
        try:
            plan = specs.plan_for(vector_axis, analyzer)
        except Unsupported as err:
            note = f"axis '{vector_axis.name}': {err}"
            errors.append(note)
            diagnostics.append(note)
            continue
        diagnostics.append(f"selected vector axis '{vector_axis.name}' (score={_axis_score(all_infos, load_infos, vector_axis)})")
        plan.diagnostics = diagnostics
        program = VectorProgram(axes=_outer_axes(plan.axes, plan.vector_axis), segments=[plan])
        # program 级诊断对单段/多段**恒非空**（对称）：调用方无需区分两条路径，
        # 也不会出现「多段有摘要、单段恒空」的口径裂缝。
        program.diagnostics = [f"single-segment program: 1 segment, vector axis '{vector_axis.name}'"]
        return program
    raise Unsupported("no vectorizable axis: " + "; ".join(errors), diagnostics=diagnostics)


def _outer_axes(axes: list[Axis], vector_axis: Axis) -> list[Axis]:
    """取出可以「提到 program 层只包一次」的轴。

    这是 ``axes`` 的**前缀**：连续的「并行轴且不是向量轴」的轴。取前缀（而不是所有
    这样的轴）是关键——:func:`_emit_segment_stmts` 会按 ``reversed(axes)`` 包裹剩余的
    轴，前缀 + 剩余拼接后与「一次包完全部非向量轴」的顺序**逐位相同**；若取的是一组
    与剩余轴交错的集合，嵌套顺序就会变，而顺序变了是非法的（见 §3.9）。
    """
    outer: list[Axis] = []
    for axis in axes:
        if axis.var.same_as(vector_axis.var) or not axis.is_parallel:
            break
        outer.append(axis)
    return outer


def _build_plan_for_axis(
    axes: list[Axis],
    serial_axis: Axis | None,
    pre_specs: list[StoreSpec],
    pre_infos: list[AccessInfo],
    main_specs: list[StoreSpec],
    main_infos: list[AccessInfo],
    post_specs: list[StoreSpec],
    post_infos: list[AccessInfo],
    vector_axis: Axis,
    lanes: int,
    elem: str,
    analyzer: arith.Analyzer,
) -> VectorPlan:
    # 方案 §6 第二步：当 serial 循环与兄弟语句（pre/post）共存时，只有沿**这根 serial 轴**
    # 向量化才能把兄弟语句安全地放在 chunk 循环之外；沿外层并行轴向量化尚未支持。
    if (pre_specs or post_specs) and (serial_axis is None or not vector_axis.var.same_as(serial_axis.var)):
        raise Unsupported(
            "a T.Parallel body with sibling statements around a serial loop can only be vectorized along that serial axis (方案 §6 第二步)"
        )

    extent = _const_int(vector_axis.extent)
    chunks = None if extent is None else (extent + lanes - 1) // lanes
    tail = None if extent is None else extent % lanes

    ctx = _ValueContext(
        axes=axes,
        vector_axis=vector_axis,
        lanes=lanes,
        analyzer=analyzer,
        vector_dtype=_vec_dtype(elem),
    )

    plan = VectorPlan(
        axes=axes,
        vector_axis=vector_axis,
        lanes=lanes,
        chunks=chunks,
        tail=tail,
    )

    def _plan_store(spec, info):
        kind = info.axis_class(vector_axis)[0]
        value = build_value(spec.value, ctx)
        mask = build_value(spec.mask, ctx) if spec.mask is not None else None
        index = index_value(info, ctx) if kind in (AccessClass.PERMUTE, AccessClass.PIECEWISE, AccessClass.INDIRECT) else None
        return StorePlan(spec=spec, access=info, value=value, mask=mask, kind=kind, index=index)

    for spec, info in zip(main_specs, main_infos):
        plan.stores.append(_plan_store(spec, info))
    for spec, info in zip(pre_specs, pre_infos):
        plan.pre_stores.append(_plan_store(spec, info))
    for spec, info in zip(post_specs, post_infos):
        plan.post_stores.append(_plan_store(spec, info))
    # 归约判定必须等值 DAG 建好之后（D13 统一判据）：逐 store 剥离旧值，顺带在 build 期
    # 就把不支持的形态拒绝掉，保证 build 成功 ⇒ emit 成功。
    plan.reduce = _classify_reduce(plan, analyzer)
    return plan


def _classify_reduce(plan: VectorPlan, analyzer: arith.Analyzer) -> ReduceInfo | None:
    """归约段判定 + 逐 store 归约拆分（缺陷 D13：单一「累积写」判据）。

    一条 store 是**可向量化的归约**，当且仅当：目标地址沿向量轴 ``Δ==0``（lane 均匀）
    且取值是**累加形式** ``dst[...] = dst[...] <op> x``（``op`` ∈ Add/Max/Min）。标量语义
    下 ``for v: dst[i] = f(v)`` 本是「后写覆盖」，只有累加形式才等价于跨 lane 归约；因此
    「store 均匀 + 取值变化」**不再**单独构成归约（否则 ``dst[i] = a[i,v] + 1.0`` 会被
    静默发成错的和）。

    返回段级代表元（第一条归约 store 的 :class:`ReduceInfo`），并给每条归约 store 填
    :attr:`StorePlan.reduce`。段内取值沿轴均匀的 uniform store 是「广播写」，原样保留
    （每条都独立发射，缺陷 D6 不再丢 store）。若该轴不是归约轴（:func:`_is_reduce_axis`
    为假，即 store 未全部 lane 均匀或取值也均匀），本函数返回 ``None``，走普通逐元素路径。
    """
    stores = plan.stores
    if not stores:
        return None
    if not _is_reduce_axis(plan.vector_axis, [store.spec for store in stores], [store.access for store in stores], analyzer):
        return None  # store 未全部 lane 均匀（与连续/步长写混用）或取值也均匀 → 逐元素路径
    primary: ReduceInfo | None = None
    for store in stores:
        if not _value_varies_along(store.spec.value, plan.vector_axis, analyzer):
            continue  # 取值沿轴均匀 → 一次广播写即可，不参与归约
        store.reduce = _plan_reduce(store, plan.vector_axis)
        if primary is None:
            primary = store.reduce
    return primary


def _plan_reduce(store: StorePlan, vector_axis: Axis) -> ReduceInfo:
    """把一条 lane 均匀的 store 拆成 :class:`ReduceInfo`（不支持的形态在 **build 期** 拒绝）。"""
    buffer = store.spec.buffer
    if not _reads_buffer(store.value, buffer):
        raise Unsupported(
            f"store to '{buffer.name}' is lane-uniform along axis '{vector_axis.name}' but its value does not read "
            f"'{buffer.name}': only the accumulate form 'dst[...] = dst[...] <op> x' (op ∈ Add/Max/Min) is a "
            "reduction (方案 §2 Case II); a lane-varying value written to a lane-uniform address is a "
            "last-write-wins overwrite and has no vector form"
        )
    op = _reduce_op([store.spec])  # Add/Max/Min 之外显式拒绝（缺陷 D1）
    varying, accumulated = _strip_reduce(store.value, op, store.spec, vector_axis)
    if not accumulated or varying is None:
        raise Unsupported(
            f"the old value of '{buffer.name}' must be a direct operand of the '{op}' reduction "
            "(方案 §2 Case II); it cannot be scaled or nested inside another operation"
        )
    return ReduceInfo(axis=vector_axis, op=op, recipe=_REDUCE_RECIPE[op], varying=varying, accumulate=True)


# 归约算子 → 其「一个操作数 lane 均匀」的标量同构指令（脊上同样允许剥离旧值）。
_REDUCE_SCALAR_RECIPE = {"vadd": "vadds", "vmax": "vmaxs", "vmin": "vmins"}


def _strip_reduce(value: ValueNode, op: str, site: StoreSpec, vector_axis: Axis) -> tuple[ValueNode | None, bool]:
    """在归约算子**脊**上剥离目标 buffer 的 lane 均匀旧值读取。

    返回 ``(varying, accumulated)``：``varying`` 是逐 lane 取值（lane 均匀加数**保留在内**，
    因为跨 lane 归约本身就会正确处理它们）；``accumulated`` 表示剥到了旧值。剥离等价于
    沿脊消去该算子的单位元（Add→0 / Max→-inf / Min→+inf），故与 ``dst op= x`` 同义。

    旧值必须与写入地址**同址**（``site.indices``）：``s[i] = s[i] + a[i,j]`` 的 ``s[i]``
    才是可剥离的旧值；``s[i] = s[i-1] + a[i,j]`` 这类 stencil 跨位读不是累加，留在
    ``varying`` 内并触发末尾的显式拒绝。只有脊上（``op`` 及其标量同构）的直接子树的旧值
    才能安全剥离；旧值若被缩放了或藏在别的算子/``uniform_expr`` 里，``varying`` 仍会读到
    dst → 明确拒绝（不静默错发）。
    """
    buffer = site.buffer
    spine = {op, _REDUCE_SCALAR_RECIPE[op]}
    accumulated = False

    def strip(node: ValueNode) -> ValueNode | None:
        nonlocal accumulated
        if (
            node.recipe == "uniform_load"
            and node.access is not None
            and _same_buffer(node.access.buffer, buffer)
            and _same_indices(tuple(node.access.indices), tuple(site.indices))
        ):
            accumulated = True
            return None
        if node.recipe in spine and len(node.args) == 2:
            kept = [child for child in (strip(node.args[0]), strip(node.args[1])) if child is not None]
            if not kept:
                return None
            if len(kept) == 1:
                return kept[0]
            return ValueNode(node.recipe, node.dtype, kept, detail=node.detail)
        return node

    varying = strip(value)
    if varying is not None and _reads_buffer(varying, buffer):
        raise Unsupported(
            f"the reduction body reads the destination buffer '{buffer.name}' outside the accumulate position "
            "(方案 §2 Case II): the old value must be a direct operand of the reduction operator"
        )
    return varying, accumulated


def _reads_buffer(node: ValueNode, buffer) -> bool:
    """值 DAG 是否在**任何位置**读取给定 buffer（含 ``uniform_expr`` 折叠的标量表达式）。"""
    if node.access is not None and _same_buffer(node.access.buffer, buffer):
        return True
    if node.scalar is not None:
        found: list[bool] = []

        def visit(sub):
            if isinstance(sub, tirx.BufferLoad) and _same_buffer(sub.buffer, buffer):
                found.append(True)

        tirx.stmt_functor.post_order_visit(node.scalar, visit)
        if found:
            return True
    return any(_reads_buffer(arg, buffer) for arg in node.args)


# ---------------------------------------------------------------------------
# 7. 发射 —— VectorPlan → TIR（``T.simd.*``）
# ---------------------------------------------------------------------------
# 发射目标全部是 ``T.simd.*``（``tilelang/ascend/language/simd.py``）——CCE 向量
# 原语的前端封装，与 C++ pass ``AscendSimdVFLowerParallel`` 的发射风格一致：
#
#   vld(addr, "NORM")                          -> <elem> x lanes   连续装载
#   vgather2(base, idx, mask)                  -> <elem> x lanes   按每 lane 索引装载
#   vdup(scalar, elem, mask)                   -> <elem> x lanes   lane 内均匀广播
#   vci(base, "int32")                         -> int32 x lanes    lane 序号
#   vadd/vsub/vmul/... (a, b, mask, mode)      -> <elem> x lanes   逐元素
#   vadds/vmuls/vmaxs/vmins (a, s, mask, mode) -> <elem> x lanes   标量-向量
#   vcmp(a, b, mask, op) / vcmps(a, s, mask, op)-> boolx256        比较谓词
#   pand/por/pnot(a, b, mask)                  -> boolx256         谓词运算
#   pset(bits, "PAT_ALL") / update_mask(n,bits)-> boolx256         掩码
#   vsel(t, f, mask)                           -> <elem> x lanes   mask ? t : f
#   vcvt(src, target_elem, mask)               -> <target> x lanes 类型转换
#   vcadd/vcmax/vcmin(v, mask)                 -> <elem> x lanes   跨 lane 归约(lane0)
#   vsts(addr, v, mask, "NORM_B{bits}")        -> void             连续存储
#   vsts(addr, v, mask, "ONEPT_B{bits}")       -> void             单元素存储(归约回写)
#   vscatter(v, base, idx, mask)               -> void             按每 lane 索引存储
#
# 地址统一用 ``BufferLoad(buffer, indices)`` 交给 ``T.simd`` 自动包成
# ``access_ptr``；连续装载/存储把向量轴变量替换成该 chunk 的 lane 基址
# （普通路径 = ``chunk * lanes``；lane 内均匀访问 = ``0``）。**每 lane 索引向量**
# 通道（STRIDED 步长 / PERMUTE 逆序 / PIECEWISE 分片 / INDIRECT 间接）改用
# ``base = BufferLoad(buffer, 全 0)``（缓冲区起点）+ 绝对元素偏移索引向量
# （``index_value`` 由 :func:`build_value` 在整数上下文构造）。
#
# 归约（§2 Case II）按 C++ pass 的定型方案**在编译期展开** chunk 循环，
# 用 SSA 形式串起累加器（跨迭代的寄存器值无法用可变的 ``Var`` 表达）：:
#
#   mask = pset(bits, "PAT_ALL")
#   acc  = vdup(identity, elem, mask)
#   for ck:                                  # 展开
#       red = vcadd(vld(src_addr(ck), "NORM"), mask)
#       acc = vadd(acc, red, mask)
#   if 累加写: old = vld(dst_addr, "BRC_B{bits}"); acc = vadd(acc, old, mask)
#   vsts(dst_addr, acc, mask, "ONEPT_B{bits}")

_MODE = "MODE_ZEROING"

# 谓词运算自身的执行掩码宽度（全 1，与数据元素宽度无关）。
_PREDICATE_MASK_BITS = 32

# 只承载标量（未向量化）的 recipe：作为向量操作数时必须先 ``vdup``。
_SCALAR_ONLY_RECIPES = ("imm", "uniform_expr")


def _int(value: int):
    return tirx.IntImm("int32", int(value))


def _elem_dtype(dtype_like) -> str:
    """去掉 lane 维，取元素类型字符串。"""
    return str(dtype_like).split("x", 1)[0]


def _unsigned_index(reg):
    """把有符号整数索引向量按位重解释为无符号（``vgather2`` / ``vscatter`` 需要）。"""
    dtype = str(reg.dtype)
    if dtype.startswith("int"):
        return tirx.reinterpret(dtype.replace("int", "uint", 1), reg)
    return reg


def _reduce_identity(recipe: str, elem: str):
    """归约原语的单位元（``vadd``→0，``vmax``→最小可表示值，``vmin``→最大可表示值）。

    单位元必须**按元素类型派生**：旧实现把 ``vmax``/``vmin`` 硬编码为 f32 极值
    ``±3.402823e38``，于是 ``int32`` / ``float16`` 的极值归约在发射期抛 ``ValueError``
    （``FloatImm supports only float, but int32 was supplied`` / ``exceeds minimum of float16``，缺陷 D3）。
    """
    if recipe == "vcadd":
        return tirx.FloatImm(elem, 0.0) if elem.startswith("float") else tirx.IntImm(elem, 0)
    return tirx.min_value(elem) if recipe == "vcmax" else tirx.max_value(elem)


class _Emitter:
    """把 :class:`VectorPlan` 的 chunk 体发射成 ``T.simd.*`` SSA 序列。

    SSA 用 ``SeqStmt`` 里的 ``tirx.Bind`` 表达（``Bind`` 是独立语句而非
    表达式，因此不能像 ``tirx.Let`` 那样嵌套）。``lane_base`` 是代入向量轴
    变量的 lane 基址：普通路径是 ``chunk * lanes``，归约路径在展开时逐 chunk
    重设（并清空 ``memo``，因为地址随 chunk 变化）。
    """

    def __init__(self, plan: VectorPlan, chunk: tirx.Var, prefix: str = ""):
        self.plan = plan
        self.chunk = chunk
        self.prefix = prefix  # 多段路径下按段命名（``s0_``/``s1_``...），避免跨段重名
        self.lane_base: tirx.PrimExpr = chunk * _int(plan.lanes)
        self.memo: dict = {}
        self.lets: list = []  # chunk 体内的 Bind / Evaluate
        self.prologue: list = []  # chunk 无关的 Bind（掩码）
        self.counter = 0
        self._mask_vars: dict = {}

    # -- SSA 绑定 -----------------------------------------------------------
    def bind(self, expr, target: list):
        var = tirx.Var(f"v{self.prefix}{self.counter}", expr.dtype)
        self.counter += 1
        target.append(tirx.Bind(var, expr))
        return var

    # -- 掩码 ---------------------------------------------------------------
    def all_mask(self, bits: int):
        """全 lane 生效的谓词（``pset(bits, PAT_ALL)``），按宽度缓存、只绑一次。"""
        cached = self._mask_vars.get(bits)
        if cached is not None:
            return cached
        var = tirx.Var(f"mask{self.prefix}{bits}", _MASK_DTYPE)
        self.prologue.append(tirx.Bind(var, simd.pset(bits, "PAT_ALL")))
        self._mask_vars[bits] = var
        return var

    def predicate_combine(self, lhs, rhs):
        """两个谓词的合取：``pand(lhs, rhs, ALL)``。"""
        return self.bind(simd.pand(lhs, rhs, self.all_mask(_PREDICATE_MASK_BITS)), self.lets)

    def predicate(self, node: ValueNode, data_dtype):
        """条件节点 → 谓词寄存器。

        逐 lane 条件（``vcmp``/``vcmps``/``pand``...）本身就是谓词寄存器，直接返回；
        循环不变条件（沿向量轴均匀，§1 Case II 的三元条件）在 lane 内恒真/恒假，
        广播成全 lane 谓词 ``update_mask(cond ? lanes : 0)``——整 chunk 同开或同关。
        """
        if not node.is_uniform:
            return self.value(node)
        bits = _mask_bits(_elem_dtype(data_dtype))
        lane_count = tirx.Select(self.scalar(node), _int(self.plan.lanes), _int(0))
        return self.bind(simd.update_mask(lane_count, bits), self.lets)

    # -- 地址 ---------------------------------------------------------------
    def addr(self, access: AccessInfo, uniform: bool = False):
        """一次访问的 ``BufferLoad``：向量轴变量代入当前 chunk 的 lane 基址。"""
        base = _int(0) if uniform else self.lane_base
        indices = [tirx.stmt_functor.substitute(index, {self.plan.vector_axis.var: base}) for index in access.indices]
        return tirx.BufferLoad(access.buffer, indices)

    def index_addr(self, access: AccessInfo):
        """索引向量通道的基址：``BufferLoad(buffer, 全 0)`` = 缓冲区起点。

        索引向量（``index_value``）给出的是**绝对**元素偏移，因此基址必须落在
        缓冲区起点，而不是某个 chunk 的 lane 基址。
        """
        return tirx.BufferLoad(access.buffer, [_int(0) for _ in access.indices])

    # -- 标量操作数 ---------------------------------------------------------
    def scalar(self, node: ValueNode):
        if node.recipe in ("imm", "uniform_expr"):
            return node.scalar
        if node.recipe == "uniform_load":
            return self.addr(node.access, uniform=True)
        if node.recipe == "vbrc":
            return self.scalar(node.args[0])
        raise Unsupported(f"cannot scalarize vector value '{node.recipe}'")

    # -- 值 ----------------------------------------------------------------
    def value(self, node: ValueNode):
        key = id(node)
        if key in self.memo:
            return self.memo[key]
        expr = self._build(node)
        self.memo[key] = expr
        return expr

    def vector_operand(self, node: ValueNode):
        """把一个 lane 内均匀的标量提升成向量寄存器（``vdup``），否则走 ``value``。"""
        if node.recipe in _SCALAR_ONLY_RECIPES:
            elem = _elem_dtype(node.dtype)
            return self.bind(simd.vdup(self.scalar(node), elem, self.all_mask(_mask_bits(elem)), _MODE), self.lets)
        return self.value(node)

    def _build(self, node: ValueNode):
        recipe = node.recipe
        args = node.args

        if recipe == "imm":
            return node.scalar
        if recipe == "lane_index":
            # 向量轴的绝对取值 = chunk 基址 + lane id（用于谓词比较，如 ``i < j``）。
            return self.bind(simd.vci(self.lane_base, _elem_dtype(node.dtype)), self.lets)
        if recipe == "vload":
            return self.bind(simd.vld(self.addr(node.access), "NORM"), self.lets)
        if recipe == "vload_strided":
            # simd 没有带步长装载：用 ``vgather2`` + 步长索引斜坡表达。
            _, stride = node.access.axis_class(self.plan.vector_axis)
            bits = _mask_bits(node.dtype)
            mask = self.all_mask(bits)
            index = simd.vmuls(simd.vci(_int(0), "uint32"), _int(stride), mask)
            return self.bind(simd.vgather2(self.addr(node.access), index, mask), self.lets)
        if recipe == "gather":
            # 每 lane 索引向量通道（改进 1）：``args[0]`` 是绝对元素偏移向量。
            bits = _mask_bits(node.dtype)
            mask = self.all_mask(bits)
            index = _unsigned_index(self.value(args[0]))
            return self.bind(simd.vgather2(self.index_addr(node.access), index, mask), self.lets)
        if recipe in ("uniform_load", "uniform_expr", "vbrc"):
            operand = node if recipe != "vbrc" else args[0]
            elem = _elem_dtype(operand.dtype)
            return self.bind(
                simd.vdup(self.scalar(operand), elem, self.all_mask(_mask_bits(elem)), _MODE),
                self.lets,
            )
        if recipe in _VEC_SCALAR_BINARY.values():
            mask = self.all_mask(_mask_bits(node.dtype))
            return self.bind(getattr(simd, recipe)(self.value(args[0]), self.scalar(args[1]), mask, _MODE), self.lets)
        if recipe in ("vshrs", "vshls"):
            # 常量移位量走标量-向量移位（`vshrs` = 算术右移，即整型 FloorDiv by 2^k）。
            mask = self.all_mask(_mask_bits(node.dtype))
            return self.bind(getattr(simd, recipe)(self.value(args[0]), node.scalar, mask, _MODE), self.lets)
        if recipe in _ELEMENTWISE_UNARY.values():
            mask = self.all_mask(_mask_bits(node.dtype))
            return self.bind(getattr(simd, recipe)(self.value(args[0]), mask, _MODE), self.lets)
        if recipe in _ELEMENTWISE_BINARY.values():
            mask = self.all_mask(_mask_bits(node.dtype))
            return self.bind(getattr(simd, recipe)(self.value(args[0]), self.value(args[1]), mask, _MODE), self.lets)
        if recipe == "vcmp":
            mask = self.all_mask(_mask_bits(args[0].dtype))
            return self.bind(simd.vcmp(self.value(args[0]), self.value(args[1]), mask, node.detail), self.lets)
        if recipe == "vcmps":
            mask = self.all_mask(_mask_bits(args[0].dtype))
            return self.bind(simd.vcmps(self.value(args[0]), self.scalar(args[1]), mask, node.detail), self.lets)
        if recipe == "vsel":
            # ``args = [cond, true, false]``；``vsel`` 的顺序是 ``(t, f, mask)``。
            # 循环不变条件（沿向量轴均匀）→ 广播成全 lane 谓词（整个 chunk 同真同假）；
            # 逐 lane 条件（vcmp/vcmps/pand...）本身就是谓词寄存器，直接用。
            true_value = self.vector_operand(args[1])
            false_value = self.vector_operand(args[2])
            return self.bind(simd.vsel(true_value, false_value, self.predicate(args[0], node.dtype)), self.lets)
        if recipe in ("pand", "por"):
            mask = self.all_mask(_PREDICATE_MASK_BITS)
            return self.bind(getattr(simd, recipe)(self.value(args[0]), self.value(args[1]), mask), self.lets)
        if recipe == "pnot":
            mask = self.all_mask(_PREDICATE_MASK_BITS)
            return self.bind(simd.pnot(self.value(args[0]), mask), self.lets)
        if recipe == "vcvt":
            src = self.value(args[0])
            target = node.detail
            mask = self.all_mask(min(_mask_bits(src.dtype), _mask_bits(target)))
            return self.bind(simd.vcvt(src, target, mask), self.lets)
        raise Unsupported(f"no emitter for value recipe '{recipe}'")


def _tail_mask(plan: VectorPlan, chunk: tirx.Var, bits: int):
    """尾块谓词：``update_mask(extent - chunk*L, bits)``（整除时返回 None）。"""
    if not plan.tail:
        return None
    return simd.update_mask(plan.vector_axis.extent - chunk * _int(plan.lanes), bits)


def _store_mask(plan: VectorPlan, store: StorePlan, emitter: _Emitter, bits: int):
    """store 的最终谓词 = 尾块掩码 ∧ 用户谓词（都为空时用全 1）。"""
    mask = None
    if store.mask is not None:
        mask = emitter.value(store.mask)
    tail = _tail_mask(plan, emitter.chunk, bits)
    if tail is not None:
        mask = tail if mask is None else emitter.predicate_combine(mask, tail)
    if mask is None:
        mask = emitter.all_mask(bits)
    return mask


def _emit_store(plan: VectorPlan, store: StorePlan, emitter: _Emitter) -> None:
    """发射一次写入：连续/均匀写走 ``vsts``，其余走每 lane 索引 ``vscatter``。"""
    elem = _elem_dtype(store.spec.buffer.dtype)
    bits = _mask_bits(elem)
    value = emitter.vector_operand(store.value)
    mask = _store_mask(plan, store, emitter, bits)

    if store.kind in (AccessClass.CONTIGUOUS, AccessClass.UNIFORM):
        uniform = store.kind == AccessClass.UNIFORM
        dist = f"ONEPT_B{bits}" if uniform else f"NORM_B{bits}"
        addr = emitter.addr(store.access, uniform=uniform)
        emitter.lets.append(tirx.Evaluate(simd.vsts(addr, value, mask, dist)))
        return

    if store.kind == AccessClass.STRIDED:
        # 常量步长写：chunk lane 基址 + 步长斜坡（与 ``vload_strided`` 对称）。
        _, stride = store.access.axis_class(plan.vector_axis)
        index = emitter.bind(simd.vmuls(simd.vci(_int(0), "uint32"), _int(stride), mask), emitter.lets)
        emitter.lets.append(tirx.Evaluate(simd.vscatter(value, emitter.addr(store.access), index, mask)))
        return

    # PERMUTE / PIECEWISE / INDIRECT：绝对元素偏移索引向量 + 缓冲区起点基址。
    index = _unsigned_index(emitter.value(store.index))
    emitter.lets.append(tirx.Evaluate(simd.vscatter(value, emitter.index_addr(store.access), index, mask)))


def _emit_outer_store(plan: VectorPlan, store: StorePlan, emitter: _Emitter) -> None:
    """发射一条与 serial 循环**同级**的语句（pre/post，方案 §6 第二步）。

    这类语句不含 serial 变量，沿向量轴 ``Δ==0``，因此每个外层迭代只执行一次——
    发射在 chunk 循环**之外**，用 ``ONEPT`` 单点写（值经 ``vdup`` 广播）。
    """
    if store.kind != AccessClass.UNIFORM:
        raise Unsupported(f"a statement outside the serial loop must be uniform along the vector axis (got {store.kind.value})")
    elem = _elem_dtype(store.spec.buffer.dtype)
    bits = _mask_bits(elem)
    value = emitter.vector_operand(store.value)
    mask = emitter.value(store.mask) if store.mask is not None else emitter.all_mask(bits)
    addr = emitter.addr(store.access, uniform=True)
    emitter.lets.append(tirx.Evaluate(simd.vsts(addr, value, mask, f"ONEPT_B{bits}")))


def emit_plan(plan: VectorPlan | VectorProgram) -> tirx.Stmt:
    """把 :class:`VectorPlan` / :class:`VectorProgram` 发射成 ``T.simd.*`` TIR。

    外层非向量轴退化成 serial 循环，向量轴被 strip-mine 成 ``chunk``
    （``for chunk in serial(ceil(E/L))``）；chunk 体内的 SSA 用 ``Bind`` 表达。
    归约（§2 Case II）则把 chunk 循环在编译期展开（见函数级的方案说明）。

    §6 第二步的 ``pre``/``post``（与 serial 循环同级的兄弟语句）沿向量轴均匀，
    发射在 chunk 循环**之外**，顺序为 ``pre → 主体 → post``。

    :class:`VectorProgram`（1..N 段）逐段发射并保持源码顺序：``program.axes`` 里的轴
    在**最外层统一包一次**（``chunk`` 变量按段命名区分），各段只包自己剩下的轴。
    单独发射一个 :class:`VectorPlan`（``hoisted`` 为空）等价于「该段自己包全部轴」，
    与 :func:`build_plan` 恒返回 program 之前的发射结果**逐节点一致**。
    """
    if isinstance(plan, VectorProgram):
        return _emit_program(plan)
    prologue, body = _emit_segment_stmts(plan, tirx.Var("chunk", "int32"), hoisted=[])
    if prologue:
        return tirx.SeqStmt([*prologue, body])
    return body


def _emit_segment_stmts(plan: VectorPlan, chunk: tirx.Var, *, hoisted: list[Axis], prefix: str = "") -> tuple[list, tirx.Stmt]:
    """发射**一段** :class:`VectorPlan` 的语句体，返回 ``(prologue, body)``。

    ``prologue`` 是 chunk 无关的掩码绑定（调用方负责提到最前）；``body`` 已按非向量轴
    包裹成嵌套 serial 循环。``hoisted`` 是**已由调用方提到最外层**的轴（多段 program 的
    共享并行轴前缀，见 :func:`_outer_axes`）：按**变量身份**跳过它们，而不是笼统跳过所有
    并行轴——否则当向量轴就是某根外层并行轴时，夹在其后的并行轴会被漏包。
    """
    emitter = _Emitter(plan, chunk, prefix)

    for store in plan.pre_stores:  # 前置语句：chunk 循环之前，每个外层迭代一次
        _emit_outer_store(plan, store, emitter)
    pre_lets = emitter.lets
    emitter.lets = []

    if plan.reduce is not None:
        _emit_reduce(plan, emitter)  # 填 emitter.lets（编译期展开 chunk）
        main_lets = emitter.lets
        emitter.lets = []
    else:
        for store in plan.stores:
            _emit_store(plan, store, emitter)
        inner = _seq(emitter.lets)
        emitter.lets = []
        main_lets = [tirx.For(chunk, 0, _int(plan.chunks or 1), tirx.ForKind.SERIAL, inner)]

    for store in plan.post_stores:  # 后置语句：chunk 循环之后，每个外层迭代一次
        _emit_outer_store(plan, store, emitter)
    post_lets = emitter.lets

    body = _seq([*pre_lets, *main_lets, *post_lets])
    for axis in reversed(plan.axes):
        if axis.var.same_as(plan.vector_axis.var):
            continue
        if any(axis.var.same_as(h.var) for h in hoisted):
            continue
        body = tirx.For(axis.var, 0, axis.extent, tirx.ForKind.SERIAL, body)
    return emitter.prologue, body


def _emit_program(program: VectorProgram) -> tirx.Stmt:
    """发射 :class:`VectorProgram`：逐段（源序）发射，``program.axes`` 统一包裹一次。

    ``program.axes`` 必须只是**并行轴的提层前缀**（:func:`_outer_axes` / 多段路径的
    共享并行轴）：它们既被这里包一次，又被 :func:`_emit_segment_stmts` 按身份跳过；
    若其中混入了非并行轴或某段的向量轴，就会出现重复包裹 / 漏包，属非法 IR。
    """
    assert all(axis.is_parallel for axis in program.axes), "program.axes may only hold hoisted T.Parallel axes"
    prologue: list = []
    bodies: list = []
    for index, segment in enumerate(program.segments):
        seg_chunk = tirx.Var(f"chunk_s{index}", "int32")
        seg_prologue, body = _emit_segment_stmts(segment, seg_chunk, hoisted=program.axes, prefix=f"s{index}_")
        prologue.extend(seg_prologue)
        bodies.append(body)
    body = _seq(bodies)
    for axis in reversed(program.axes):
        body = tirx.For(axis.var, 0, axis.extent, tirx.ForKind.SERIAL, body)
    if prologue:
        return tirx.SeqStmt([*prologue, body])
    return body


def _emit_reduce(plan: VectorPlan, emitter: _Emitter) -> None:
    """归约段（§2 Case II）：按源码顺序逐条发射段内每个 store。

    段内可能有**多条**归约（缺陷 D6 原先用 ``next(...)`` 只发第一条）以及取值沿向量轴
    均匀的广播写（一次 ``ONEPT`` 写即可）。每条归约各自展开 chunk、各起一条 SSA 累加链、
    各回写一次。
    """
    for store in plan.stores:
        if store.reduce is None:
            _emit_outer_store(plan, store, emitter)  # 取值沿轴均匀 → 与归约同级，发一次
            continue
        _emit_one_reduce(plan, store, emitter)


def _emit_one_reduce(plan: VectorPlan, store: StorePlan, emitter: _Emitter) -> None:
    """发射单条归约 store：展开 chunk、SSA 串累加器、``ONEPT`` 回写。

    先要 ``vcadd/vcmax/vcmin`` 把每个 chunk 压成 lane0 的部分结果，再用
    ``vadd/vmax/vmin`` 累加；累加写（``dst op= x``）末尾再并入目标里原有的值
    （``vld(..., BRC_B{bits})`` 广播）。``extent % lanes != 0`` 时最后一个 chunk 是尾块：
    ``update_mask`` 造尾块谓词，``vsel`` 把无效车道填成归约单位元（改进 6），再按全 lane
    掩码归约。逐 lane 取值与旧值已由 :func:`_plan_reduce` 在 build 期拆好。

    结果写回 ``emitter.lets``；外层轴与 ``pre``/``post`` 的包裹交给 :func:`emit_plan`。
    """
    reduce = store.reduce
    elem = _elem_dtype(store.spec.buffer.dtype)
    bits = _mask_bits(elem)
    mask = emitter.all_mask(bits)
    recipe = reduce.recipe
    element_op = _REDUCE_ELEMENT[recipe]
    identity = _reduce_identity(recipe, elem)
    acc = emitter.bind(simd.vdup(identity, elem, mask, _MODE), emitter.lets)

    # 尾块（``extent % lanes != 0``，改进 6）：最后一个 chunk 只有 ``tail`` 个有效车道，
    # 其余车道在 ``vld`` 里读到的是越界/脏数据。用 ``vsel(loaded, fill, tail_mask)`` 把它们
    # 替换成归约单位元（add→0，max→-inf，min→+inf），再按全 lane 掩码归约，结果正确。
    fill = emitter.bind(simd.vdup(identity, elem, mask, _MODE), emitter.lets) if plan.tail else None

    varying = reduce.varying
    last_chunk = (plan.chunks or 1) - 1
    for ck in range(plan.chunks or 1):
        lane_base = _int(ck * plan.lanes)
        emitter.lane_base = lane_base
        emitter.memo = {}  # 地址随 chunk 变化，缓存必须按 chunk 失效
        loaded = emitter.value(varying)
        if fill is not None and ck == last_chunk:
            tail_mask = emitter.bind(simd.update_mask(plan.vector_axis.extent - lane_base, bits), emitter.lets)
            loaded = emitter.bind(simd.vsel(loaded, fill, tail_mask), emitter.lets)
        reduced = emitter.bind(getattr(simd, recipe)(loaded, mask, _MODE), emitter.lets)
        acc = emitter.bind(getattr(simd, element_op)(acc, reduced, mask, _MODE), emitter.lets)

    if reduce.accumulate:
        # ``dst = dst op x``：末尾并入目标里原有的值（读到的是本 chunk 之前的结果）。
        old = emitter.bind(simd.vld(emitter.addr(store.access, uniform=True), f"BRC_B{bits}"), emitter.lets)
        acc = emitter.bind(getattr(simd, element_op)(acc, old, mask, _MODE), emitter.lets)

    dst = emitter.addr(store.access, uniform=True)
    emitter.lets.append(tirx.Evaluate(simd.vsts(dst, acc, mask, f"ONEPT_B{bits}")))


def _same_buffer(lhs, rhs) -> bool:
    return bool(lhs.data.same_as(rhs.data))


def render_plan(plan: VectorPlan | VectorProgram) -> str:
    """人类可读的计划文本（诊断 / 测试断言用）。

    ``build_plan`` 恒返回 :class:`VectorProgram`，但**1 段 program 与「单段 plan」逐字同形**
    （级别 1 只统一表示层，不改人类可读输出）：不打印 ``program:`` 头与段标记，program 级
    摘要并入同一段 ``diagnostics:``。只有多段 program 才打印 ``program:`` 头与逐段标记。
    """
    if isinstance(plan, VectorPlan):
        return _render_single_plan(plan)
    if len(plan.segments) == 1:
        return _render_single_plan(plan.segments[0], extra_diagnostics=plan.diagnostics)
    lines = [f"program: axes={[axis.name for axis in plan.axes]} segments={len(plan.segments)}"]
    for index, segment in enumerate(plan.segments):
        lines.append(f"-- segment {index} --")
        lines.extend(_render_single_plan(segment).splitlines())
    if plan.diagnostics:
        lines.append("diagnostics:")
        lines.extend(f"  {note}" for note in plan.diagnostics)
    return "\n".join(lines)


def _render_single_plan(plan: VectorPlan, extra_diagnostics: tuple[str, ...] = ()) -> str:
    lines = [
        f"vector_axis={plan.vector_axis.name} lanes={plan.lanes} chunks={plan.chunks} tail={plan.tail}",
    ]
    if plan.reduce is not None:
        lines.append(f"reduce(axis={plan.reduce.axis.name}, op={plan.reduce.op} -> {plan.reduce.recipe})")
    for tag, group in (("pre", plan.pre_stores), ("store", plan.stores), ("post", plan.post_stores)):
        for store in group:
            sink = f"{store.spec.buffer.name}{list(store.spec.indices)}"
            lines.append(f"{tag} {sink} [{store.kind.value}] mask={'yes' if store.mask else 'no'}")
            _render_value(store.value, lines, 1)
    if plan.diagnostics or extra_diagnostics:
        lines.append("diagnostics:")
        lines.extend(f"  {note}" for note in plan.diagnostics)
        lines.extend(f"  {note}" for note in extra_diagnostics)
    return "\n".join(lines)


def _render_value(node: ValueNode, lines: list, depth: int) -> None:
    pad = "  " * depth
    detail = f" ({node.detail})" if node.detail else ""
    lines.append(f"{pad}- {node.recipe}: {node.dtype}{detail}")
    for arg in node.args:
        _render_value(arg, lines, depth + 1)


def vectorize_parallel_nests(stmt: tirx.Stmt, *, lanes: int | None = None) -> tirx.Stmt:
    """把 stmt 里每个 ``T.Parallel`` 嵌套整段替换成 :func:`emit_plan` 的结果。

    自顶向下（pre-order）遍历：一旦某个 ``T.Parallel`` 能整体规划，就把**整个
    连续的 Parallel 嵌套**（:func:`_split_nest` 会一路吃到非 Parallel）换成发射
    结果——外层轴变 serial、向量轴被 strip-mine 成 chunk 循环。若自底向上替换，
    最内层会先被换成串行代码，外层 Parallel 反而再也规划不了（``normalize_body``
    撞到已发射的 ``For``），于是多维嵌套只下降一半、外层 ``T.Parallel`` 残留。

    整体规划失败时才继续下探子节点，让更内层的 Parallel 嵌套仍有机会单独处理。
    无法向量化的嵌套保持原样（明确不做静默降级）。
    """

    def rewrite(node):
        # pre-order 回调：返回非 None 表示「就地替换且不再下探」，返回 None 表示继续递归。
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL:
            try:
                plan = build_plan(node, lanes=lanes)
                return emit_plan(plan)
            except Unsupported:
                return None
        return None

    return tirx.stmt_functor.ir_transform(stmt, rewrite, None, ["tirx.For"])
