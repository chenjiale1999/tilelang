"""原语事实表与通用小工具（polyhedral 包的最底层）。

这里集中管理**ISA/算子事实**与硬件常量，不依赖任何分析模型，
供 :mod:`~tilelang.ascend.analysis.polyhedral` 其余子模块共用：

* 逐元素 / 标量-向量 / 比较 / 布尔组合的 recipe 表；
* **归约原语表**：TIR 算子 → 逐元素指令 → 跨 lane 归约指令 → 单位元
  （:class:`ReducePrimitive` / :data:`_REDUCE_PRIMITIVES`）；
* 索引向量通道可构造的整数算子集（:data:`_INDEX_VALUE_OPS`）与 recipe 分类
  （:data:`_UNIFORM_RECIPES` / :data:`_SCALAR_ONLY_RECIPES`）；
* 硬件常量（:data:`_MASK_DTYPE` / :data:`_PREDICATE_MASK_BITS` / :data:`_MODE`）；
* :class:`Unsupported`（唯一对外的拒绝异常，携带全量诊断）；
* 语句 / 表达式的快速判定（``_same_indices`` / ``_same_buffer`` / ``_seq``）；
* 类型与标量常量的尾段处理（``_elem_dtype`` / ``_int``）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tvm import tirx

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

# 反向表：标量-向量指令 → 逐元素向量指令。标量操作数若已被提升成向量寄存器
# （如 ``local.var`` 累加器的旧值），发射层退回向量-向量形式。
_SCALAR_TO_VECTOR_BINARY = {scalar: vector for vector, scalar in _VEC_SCALAR_BINARY.items()}

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

# 所有 ``T.simd.*`` 向量指令的执行模式参数（零化目标被动掩码外的车道）。
_MODE = "MODE_ZEROING"

# 谓词-谓词运算（pand/por/pnot）自身的执行掩码宽度。取**最细粒度 b8**（256 lane），
# 这样任何元素位宽下都不会截断：谓词位只有在其来源比较/掩码的粒度内有效，超出的位
# 本来是 0（MODE_ZEROING），多算无害；反之用固定 b32（64 lane）会静默截断 f16（128
# lane）/ int8（256 lane）的谓词（R1）。真正需要精确粒度时调用方应传数据元素的 bits。
_PREDICATE_MASK_BITS = 8

# lane 内均匀、无需向量寄存器的值（标量即可）。
_UNIFORM_RECIPES = ("imm", "uniform_load", "uniform_expr")

# 只承载标量（未向量化）的 recipe：作为向量操作数时必须先 ``vdup``。
_SCALAR_ONLY_RECIPES = ("imm", "uniform_expr")

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

# ---------------------------------------------------------------------------
# 归约原语事实表 —— 归约算子白名单、指令映射与单位元的**唯一来源**
# ---------------------------------------------------------------------------


def _zero_identity(elem: str):
    """``Add`` 的单位元 ``0``（浮点用 ``FloatImm``，整型用 ``IntImm``）。"""
    return tirx.FloatImm(elem, 0.0) if elem.startswith("float") else tirx.IntImm(elem, 0)


def _min_identity(elem: str):
    """``Max`` 的单位元：浮点用 ``-inf``（打印干净、语义精确），整型用该类型最小值。"""
    if elem.startswith("float"):
        return tirx.FloatImm(elem, float("-inf"))
    return tirx.min_value(elem)


def _max_identity(elem: str):
    """``Min`` 的单位元：浮点用 ``+inf``（打印干净、语义精确），整型用该类型最大值。"""
    if elem.startswith("float"):
        return tirx.FloatImm(elem, float("inf"))
    return tirx.max_value(elem)


@dataclass(frozen=True)
class ReducePrimitive:
    """一条归约原语事实：TIR 算子 → 逐元素指令 → 跨 lane 归约指令。"""

    tir_op: str  # TIR 节点名（Add / Max / Min）
    op: str  # 逐元素指令（vadd / vmax / vmin）
    recipe: str  # 跨 lane 归约指令（vcadd / vcmax / vcmin）
    scalar: str  # 标量-向量同构指令（vadds / vmaxs / vmins）
    identity: Callable[[str], tirx.PrimExpr]  # 按元素类型派生单位元


_REDUCE_PRIMITIVES: tuple[ReducePrimitive, ...] = (
    ReducePrimitive("Add", "vadd", "vcadd", "vadds", _zero_identity),
    ReducePrimitive("Max", "vmax", "vcmax", "vmaxs", _min_identity),
    ReducePrimitive("Min", "vmin", "vcmin", "vmins", _max_identity),
)

_REDUCE_BY_RECIPE = {primitive.recipe: primitive for primitive in _REDUCE_PRIMITIVES}

# 供规划 / 发射层使用的派生视图（全部源自同一张 _REDUCE_PRIMITIVES）。
_REDUCE_OPS = {primitive.tir_op: primitive.op for primitive in _REDUCE_PRIMITIVES}
_REDUCE_RECIPE = {primitive.op: primitive.recipe for primitive in _REDUCE_PRIMITIVES}
_REDUCE_ELEMENT = {primitive.recipe: primitive.op for primitive in _REDUCE_PRIMITIVES}
_REDUCE_SCALAR_RECIPE = {primitive.op: primitive.scalar for primitive in _REDUCE_PRIMITIVES}


def _reduce_identity(recipe: str, elem: str):
    """归约原语的单位元（``vcadd``→0，``vcmax``→最小可表示值，``vcmin``→最大可表示值）。

    单位元必须**按元素类型派生**：旧实现把 ``vmax``/``vmin`` 硬编码为 f32 极值
    ``±3.402823e38``，于是 ``int32`` / ``float16`` 的极值归约在发射期抛 ``ValueError``
    （``FloatImm supports only float, but int32 was supplied`` / ``exceeds minimum of float16``，缺陷 D3）。
    """
    return _REDUCE_BY_RECIPE[recipe].identity(elem)


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
    :func:`~tilelang.ascend.analysis.polyhedral.build_plan` 会填充它；直接
    ``str(err)`` 仍然是单行摘要。
    """

    def __init__(self, message: str, *, diagnostics: list[str] | None = None):
        super().__init__(message)
        self.diagnostics: list[str] = list(diagnostics or [])


def _int(value: int):
    return tirx.IntImm("int32", int(value))


def _elem_dtype(dtype_like) -> str:
    """去掉 lane 维，取元素类型字符串。"""
    return str(dtype_like).split("x", 1)[0]


def _same_buffer(lhs, rhs) -> bool:
    return bool(lhs.data.same_as(rhs.data))


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
