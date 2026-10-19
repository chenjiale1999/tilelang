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

  ============ ==================================================
  ``Δ_v(off)`` 含义 / 后续动作
  ============ ==================================================
  ``1``        连续 → ``vld`` / ``vsts``（NORM）
  ``0``        lane 内均匀 → 标量读一次 + ``vdup`` 广播
  ``k != 0,1`` 带步长 → ``vgather2`` + 步长索引斜坡
  非常数       非仿射 → 明确拒绝（间接下标同理）
  ============ ==================================================

  这条规则同时吃掉了几何上完全不同的场景，无需分别写分支：

  * ``c[i] = a[i]`` / ``c[i] = a[2*i]`` / ``c[i] = a[0]``
  * ``g[1, 8+i_s, 3]``（§4 表达式下标）
  * ``h[i//16, i%16]``（§4 高维摊平——化简后 ``Δ == 1``）
  * ``d[i,j] = g[i] - g[j]``（§2 维度不一致：``g[i]`` 沿 j 是 ``Δ == 0``）
  * ``a[b[i]]``（下标含 ``BufferLoad`` → 非常数 → gather）

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
    """Raised when a nest cannot be vectorized; carries the diagnostic reason."""


# ---------------------------------------------------------------------------
# 1. SCoP：迭代域
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    """一根 ``T.Parallel`` 轴：``0 <= var < extent``。"""

    var: tirx.Var
    extent: tirx.PrimExpr
    index: int = 0  # 轴在嵌套中的位置，0 为最外层

    @property
    def name(self) -> str:
        return self.var.name


class AccessClass(str, enum.Enum):
    """沿某根轴的访存形态，由 :func:`classify_delta` 唯一决定。"""

    CONTIGUOUS = "contiguous"  # Δ == 1
    UNIFORM = "uniform"  # Δ == 0
    STRIDED = "strided"  # Δ == k, k != 0, 1
    NON_AFFINE = "non_affine"  # Δ 不是常数（含间接下标）
    INDIRECT = "indirect"  # 下标本身含 BufferLoad（gather/scatter）


# ---------------------------------------------------------------------------
# 2. 仿射原语
# ---------------------------------------------------------------------------


def _const_int(expr) -> int | None:
    if isinstance(expr, tirx.IntImm):
        return int(expr.value)
    if isinstance(expr, int) and not isinstance(expr, bool):
        return int(expr)
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


def classify_delta(delta: tirx.PrimExpr) -> tuple[AccessClass, int | None]:
    """把差分值映射成 :class:`AccessClass`（唯一的分类规则）。"""
    value = _const_int(delta)
    if value is None:
        return AccessClass.NON_AFFINE, None
    if value == 0:
        return AccessClass.UNIFORM, 0
    if value == 1:
        return AccessClass.CONTIGUOUS, 1
    return AccessClass.STRIDED, value


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
    per_axis: dict = field(default_factory=dict)  # var.name -> (AccessClass, stride)
    indirect: bool = False
    in_global: bool = False

    def axis_class(self, axis: Axis) -> tuple[AccessClass, int | None]:
        return self.per_axis.get(axis.name, (AccessClass.NON_AFFINE, None))


def classify_access(
    buffer,
    indices,
    axes: list[Axis],
    analyzer: arith.Analyzer,
    *,
    is_store: bool,
) -> AccessInfo:
    """构造访问的仿射模型并逐轴分类。

    分歧点只有一个：``Δ_v`` 的取值。间接下标（``A[B[i]]``）在下标含
    ``BufferLoad`` 时直接判为 INDIRECT —— 它的地址依赖「被装载的值」，
    多面体模型无法表达成 ``x`` 的仿射函数（见方案 §8.2：数组层可表达、
    硬件层明确拒绝）。
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
    for axis in axes:
        if indirect:
            info.per_axis[axis.name] = (AccessClass.INDIRECT, None)
            continue
        info.per_axis[axis.name] = classify_delta(finite_difference(offset, axis.var, analyzer))
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
    * 其它（内层 serial loop、原子调用、跨迭代标量状态）→ 明确拒绝。
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
        raise Unsupported(f"nested {_for_kind_name(stmt.kind)} loop inside T.Parallel is not affine-vectorizable (方案 §6)")

    if type(stmt).__name__ == "Bind":
        return result

    raise Unsupported(f"unsupported statement in T.Parallel body: {type(stmt).__name__}")


def _same_indices(lhs: tuple, rhs: tuple) -> bool:
    return len(lhs) == len(rhs) and all(a.same_as(b) for a, b in zip(lhs, rhs))


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
        """整个子表达式是否 lane 内均匀（不需要向量寄存器）。"""
        if self.recipe in ("lane_index", "vbrc"):
            return False
        if self.recipe in _UNIFORM_RECIPES:
            return True
        if self.recipe == "vload":
            return self.access is not None and self.access._uniform
        return all(arg.is_uniform for arg in self.args)


def _dtype_of(expr) -> str:
    dtype = expr.dtype
    if isinstance(dtype, str):
        return dtype
    return str(dtype)


def _is_scalar_dtype(expr) -> bool:
    dtype = expr.dtype
    lanes = getattr(dtype, "lanes", 1)
    return int(lanes) == 1


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


def load_node(info: AccessInfo, ctx: _ValueContext, is_global_ok: bool) -> ValueNode:
    """把一次已分类的访问变成值节点（连续/广播/步长/间接）。"""
    if info.in_global and not is_global_ok:
        raise Unsupported(
            "global-memory operand inside T.Parallel (方案 §0 Case II): needs an explicit UB staging copy before the parallel nest"
        )
    kind, stride = info.axis_class(ctx.vector_axis)
    info._uniform = kind == AccessClass.UNIFORM  # type: ignore[attr-defined]
    if kind == AccessClass.CONTIGUOUS:
        return ValueNode("vload", ctx.vector_dtype, access=info)
    if kind == AccessClass.UNIFORM:
        return ValueNode("uniform_load", _dtype_of_scalar(info), access=info, detail="lane-uniform")
    if kind == AccessClass.STRIDED:
        return ValueNode("vload_strided", ctx.vector_dtype, access=info, detail=f"stride={stride}")
    if kind == AccessClass.INDIRECT:
        raise Unsupported(
            "indirect (gather) access A[B[i]] (方案 §8.2): expressed in the array IR but "
            "rejected on the hardware path — no gather recipe at this stage"
        )
    raise Unsupported(
        f"non-affine access with respect to axis '{ctx.vector_axis.name}': "
        f"Δ(offset)={finite_difference(info.offset, ctx.vector_axis.var, ctx.analyzer)} is not a constant"
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


@dataclass
class ReduceInfo:
    """归约：store 沿某轴 Δ==0 而取值沿该轴变化（方案 §2 Case II）。"""

    axis: Axis
    op: str
    recipe: str = "vcadd"


# 逐元素算子 → 跨 lane 归约原语（§2 Case II 把内层轴压成标量）。
_REDUCE_RECIPE = {"vadd": "vcadd", "vmax": "vcmax", "vmin": "vcmin"}
# 归约原语 → 累加/合并用的逐元素算子（与上表互逆）。
_REDUCE_ELEMENT = {recipe: op for op, recipe in _REDUCE_RECIPE.items()}


@dataclass
class VectorPlan:
    """与硬件无关的向量执行计划（≈ 方案里的中间层 2 的分析部分）。"""

    axes: list[Axis]
    vector_axis: Axis
    lanes: int
    chunks: int
    tail: int
    stores: list[StorePlan] = field(default_factory=list)
    reduce: ReduceInfo | None = None
    diagnostics: list[str] = field(default_factory=list)


def _split_nest(stmt: tirx.Stmt) -> tuple[list[Axis], tirx.Stmt]:
    axes: list[Axis] = []
    current = stmt
    while isinstance(current, tirx.For) and current.kind == tirx.ForKind.PARALLEL:
        axes.append(Axis(current.loop_var, current.extent, len(axes)))
        current = current.body
    if not axes:
        raise Unsupported("no T.Parallel axis found")
    return axes, current


def select_vector_axis(
    axes: list[Axis], stores: list[StoreSpec], analyzer: arith.Analyzer
) -> tuple[Axis, list[AccessInfo], ReduceInfo | None]:
    """选向量轴：在 store 可行的轴里，优先「能读出连续性」的、最内层的轴。

    * 候选轴：所有 store 在该轴上都不是 NON_AFFINE/INDIRECT；
    * 打分：该轴上 CONTIGUOUS 的访存数量（含读），越高越好；同分取更内层。

    这里**没有**「按代码形状分支」——打分完全来自 Δ 的统计。
    """
    store_infos: list[AccessInfo] = []
    for spec in stores:
        store_infos.append(classify_access(spec.buffer, spec.indices, axes, analyzer, is_store=True))

    best: Axis | None = None
    best_score = -1
    reduce_info: ReduceInfo | None = None
    best_reduce_axis: Axis | None = None

    for axis in axes:
        if any(info.axis_class(axis)[0] in (AccessClass.NON_AFFINE, AccessClass.INDIRECT) for info in store_infos):
            continue
        score = sum(1 for info in store_infos if info.axis_class(axis)[0] == AccessClass.CONTIGUOUS)
        if score == 0:
            continue
        score = score * 10 + axis.index  # 同分取更内层
        if score > best_score:
            best, best_score = axis, score

    if best is None:
        # 没有 store 沿任何轴连续：可能全是 lane 内均匀写（§0 广播写）或归约。
        for axis in axes:
            if all(info.axis_class(axis)[0] == AccessClass.UNIFORM for info in store_infos):
                best = axis
                break
    if best is None:
        raise Unsupported("no vectorizable axis: every store is non-affine or strided on all axes")

    # 归约检测：store 沿 best 轴均匀，但取值沿该轴变化。
    for axis in axes:
        if all(info.axis_class(axis)[0] == AccessClass.UNIFORM for info in store_infos) and any(
            _value_varies_along(spec.value, axis, analyzer) for spec in stores
        ):
            op = _reduce_op(stores)
            reduce_info = ReduceInfo(axis=axis, op=op, recipe=_REDUCE_RECIPE[op])
            best_reduce_axis = axis
            break
    if reduce_info is not None and best_reduce_axis is not None:
        best = best_reduce_axis

    return best, store_infos, reduce_info


def _value_varies_along(expr, axis: Axis, analyzer: arith.Analyzer) -> bool:
    """取值是否随该轴变化（直接或间接通过 buffer 访问）。"""
    varies = []

    def visit(node):
        if isinstance(node, tirx.BufferLoad):
            info = classify_access(node.buffer, tuple(node.indices), [axis], analyzer, is_store=False)
            if info.axis_class(axis)[0] in (AccessClass.CONTIGUOUS, AccessClass.STRIDED):
                varies.append(True)

    tirx.stmt_functor.post_order_visit(expr, visit)
    return any(axis.var.same_as(v) for v in _free_vars(expr)) or bool(varies)


def _reduce_op(stores: list[StoreSpec]) -> str:
    for spec in stores:
        node_name = type(spec.value).__name__
        if node_name == "Add":
            return "vadd"
        if node_name == "Max":
            return "vmax"
        if node_name == "Min":
            return "vmin"
    return "vadd"


def build_plan(stmt: tirx.Stmt, *, lanes: int | None = None) -> VectorPlan:
    """主入口：``T.Parallel`` 嵌套 → :class:`VectorPlan`。

    伪代码（完整版见模块 docstring）::

        axes, body       = split_nest(stmt)                # SCoP 迭代域
        aliases          = collect_aliases(body)           # 标量中间量内联
        stores           = normalize_body(body, aliases)   # 多语句 / 分支规范化
        v, store_infos, reduce = select_vector_axis(axes, stores)
        for spec, info in zip(stores, store_infos):
            value = build_value(spec.value, ctx)           # Δ 分类的递归
            mask  = build_value(spec.mask)  if spec.mask
        return VectorPlan(axes, v, lanes, chunks, tail, stores, reduce)

    ``lanes`` 默认由元素位宽唯一决定（``2048 / elem_bits``，与 ``T.simd`` 的
    ``vreg`` 宽度一致），因此一般不需要显式传入。
    """
    analyzer = arith.Analyzer()
    axes, body = _split_nest(stmt)
    aliases = _collect_aliases(body)
    stores = normalize_body(body, aliases)
    if not stores:
        raise Unsupported("T.Parallel body has no store to vectorize")

    elem = _buffer_elem_dtype(stores[0].buffer)
    if lanes is None:
        lanes = _vreg_lanes(elem)

    vector_axis, store_infos, reduce_info = select_vector_axis(axes, stores, analyzer)

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
        reduce=reduce_info,
    )

    for spec, info in zip(stores, store_infos):
        value = build_value(spec.value, ctx)
        mask = build_value(spec.mask, ctx) if spec.mask is not None else None
        plan.stores.append(StorePlan(spec=spec, access=info, value=value, mask=mask, kind=info.axis_class(vector_axis)[0]))
    return plan


def _buffer_elem_dtype(buffer) -> str:
    dtype = buffer.dtype
    return str(dtype).split("x", 1)[0]


# ---------------------------------------------------------------------------
# 7. 发射 —— VectorPlan → TIR（``T.simd.*``）
# ---------------------------------------------------------------------------
# 发射目标全部是 ``T.simd.*``（``tilelang/ascend/language/simd.py``）——CCE 向量
# 原语的前端封装，与 C++ pass ``AscendSimdVFLowerParallel`` 的发射风格一致：
#
#   vld(addr, "NORM")                          -> <elem> x lanes   连续装载
#   vgather2(addr, vmuls(vci(0,u32), k), m)    -> <elem> x lanes   步长装载
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
#
# 地址统一用 ``BufferLoad(buffer, indices)`` 交给 ``T.simd`` 自动包成
# ``access_ptr``；indices 里把向量轴变量替换成该 chunk 的 lane 基址
# （普通路径 = ``chunk * lanes``；lane 内均匀访问 = ``0``）。
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


def _reduce_identity(recipe: str, elem: str):
    """归约原语的单位元（``vadd``→0，``vmax``→-inf，``vmin``→+inf）。"""
    if recipe == "vcadd":
        return tirx.FloatImm(elem, 0.0) if elem.startswith("float") else tirx.IntImm(elem, 0)
    sign = -1.0 if recipe == "vcmax" else 1.0
    return tirx.FloatImm(elem, sign * 3.402823e38)


class _Emitter:
    """把 :class:`VectorPlan` 的 chunk 体发射成 ``T.simd.*`` SSA 序列。

    SSA 用 ``SeqStmt`` 里的 ``tirx.Bind`` 表达（``Bind`` 是独立语句而非
    表达式，因此不能像 ``tirx.Let`` 那样嵌套）。``lane_base`` 是代入向量轴
    变量的 lane 基址：普通路径是 ``chunk * lanes``，归约路径在展开时逐 chunk
    重设（并清空 ``memo``，因为地址随 chunk 变化）。
    """

    def __init__(self, plan: VectorPlan, chunk: tirx.Var):
        self.plan = plan
        self.chunk = chunk
        self.lane_base: tirx.PrimExpr = chunk * _int(plan.lanes)
        self.memo: dict = {}
        self.lets: list = []  # chunk 体内的 Bind / Evaluate
        self.prologue: list = []  # chunk 无关的 Bind（掩码）
        self.counter = 0
        self._mask_vars: dict = {}

    # -- SSA 绑定 -----------------------------------------------------------
    def bind(self, expr, target: list):
        var = tirx.Var(f"v{self.counter}", expr.dtype)
        self.counter += 1
        target.append(tirx.Bind(var, expr))
        return var

    # -- 掩码 ---------------------------------------------------------------
    def all_mask(self, bits: int):
        """全 lane 生效的谓词（``pset(bits, PAT_ALL)``），按宽度缓存、只绑一次。"""
        cached = self._mask_vars.get(bits)
        if cached is not None:
            return cached
        var = tirx.Var(f"mask{bits}", _MASK_DTYPE)
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
    elem = _elem_dtype(store.spec.buffer.dtype)
    bits = _mask_bits(elem)
    uniform = store.kind == AccessClass.UNIFORM
    value = emitter.vector_operand(store.value)
    mask = _store_mask(plan, store, emitter, bits)
    dist = f"ONEPT_B{bits}" if uniform else f"NORM_B{bits}"
    addr = emitter.addr(store.access, uniform=uniform)
    emitter.lets.append(tirx.Evaluate(simd.vsts(addr, value, mask, dist)))


def emit_plan(plan: VectorPlan) -> tirx.Stmt:
    """把 :class:`VectorPlan` 发射成 ``T.simd.*`` TIR。

    外层非向量轴退化成 serial 循环，向量轴被 strip-mine 成 ``chunk``
    （``for chunk in serial(ceil(E/L))``）；chunk 体内的 SSA 用 ``Bind`` 表达。
    归约（§2 Case II）则把 chunk 循环在编译期展开（见函数级的方案说明）。
    """
    chunk = tirx.Var("chunk", "int32")
    if plan.reduce is not None:
        emitter = _emit_reduce(plan, chunk)
    else:
        emitter = _Emitter(plan, chunk)
        for store in plan.stores:
            _emit_store(plan, store, emitter)
        inner = emitter.lets[0] if len(emitter.lets) == 1 else tirx.SeqStmt(emitter.lets)
        body = tirx.For(chunk, 0, _int(plan.chunks or 1), tirx.ForKind.SERIAL, inner)
        for axis in reversed(plan.axes):
            if axis.var.same_as(plan.vector_axis.var):
                continue
            body = tirx.For(axis.var, 0, axis.extent, tirx.ForKind.SERIAL, body)
        emitter.lets = [body]

    if emitter.prologue:
        return tirx.SeqStmt([*emitter.prologue, *emitter.lets])
    return emitter.lets[0] if len(emitter.lets) == 1 else tirx.SeqStmt(emitter.lets)


def _emit_reduce(plan: VectorPlan, chunk: tirx.Var) -> _Emitter:
    """归约 store（§2 Case II）：展开 chunk、SSA 串累加器、``ONEPT`` 回写。

    先要 ``vcadd/vcmax/vcmin`` 把每个 chunk 压成 lane0 的部分结果，再用
    ``vadd/vmax/vmin`` 累加；若写成累加形式 ``dst = dst op x``，末尾再并入目标
    里原有的值（``vld(..., BRC_B{bits})`` 广播）。
    """
    store = next(s for s in plan.stores if s.kind == AccessClass.UNIFORM)
    if plan.tail:
        raise Unsupported("reduction requires the vector axis extent to be a multiple of the vector width")
    emitter = _Emitter(plan, chunk)
    elem = _elem_dtype(store.spec.buffer.dtype)
    bits = _mask_bits(elem)
    mask = emitter.all_mask(bits)
    recipe = plan.reduce.recipe
    element_op = _REDUCE_ELEMENT[recipe]
    acc = emitter.bind(simd.vdup(_reduce_identity(recipe, elem), elem, mask, _MODE), emitter.lets)

    varying = _varying_operand(store.value)
    for ck in range(plan.chunks or 1):
        emitter.lane_base = _int(ck * plan.lanes)
        emitter.memo = {}  # 地址随 chunk 变化，缓存必须按 chunk 失效
        reduced = emitter.bind(getattr(simd, recipe)(emitter.value(varying), mask, _MODE), emitter.lets)
        acc = emitter.bind(getattr(simd, element_op)(acc, reduced, mask, _MODE), emitter.lets)

    if _is_accumulate(store):
        # ``dst = dst op x``：末尾并入目标里原有的值（读到的是本 chunk 之前的结果）。
        old = emitter.bind(simd.vld(emitter.addr(store.access, uniform=True), f"BRC_B{bits}"), emitter.lets)
        acc = emitter.bind(getattr(simd, element_op)(acc, old, mask, _MODE), emitter.lets)

    dst = emitter.addr(store.access, uniform=True)
    emitter.lets.append(tirx.Evaluate(simd.vsts(dst, acc, mask, f"ONEPT_B{bits}")))

    for axis in reversed(plan.axes):
        if axis.var.same_as(plan.vector_axis.var):
            continue
        body = emitter.lets[0] if len(emitter.lets) == 1 else tirx.SeqStmt(emitter.lets)
        emitter.lets = [tirx.For(axis.var, 0, axis.extent, tirx.ForKind.SERIAL, body)]
    return emitter


def _is_accumulate(store: StorePlan) -> bool:
    """``dst = dst op x``（累加写）：存在一个读同一目标 buffer 的 lane 均匀操作数。"""
    for arg in store.value.args:
        if arg.recipe == "uniform_load" and arg.access is not None and _same_buffer(arg.access.buffer, store.spec.buffer):
            return True
    return False


def _same_buffer(lhs, rhs) -> bool:
    return bool(lhs.data.same_as(rhs.data))


def _varying_operand(node: ValueNode) -> ValueNode:
    """从 ``op(uniform, varying)`` 里挑出随 lane 变化的那一侧。"""
    if len(node.args) == 2:
        lhs, rhs = node.args
        if rhs.is_uniform and not lhs.is_uniform:
            return lhs
        if lhs.is_uniform and not rhs.is_uniform:
            return rhs
    return node


def render_plan(plan: VectorPlan) -> str:
    """人类可读的计划文本（诊断 / 测试断言用）。"""
    lines = [
        f"vector_axis={plan.vector_axis.name} lanes={plan.lanes} chunks={plan.chunks} tail={plan.tail}",
    ]
    if plan.reduce is not None:
        lines.append(f"reduce(axis={plan.reduce.axis.name}, op={plan.reduce.op} -> {plan.reduce.recipe})")
    for store in plan.stores:
        sink = f"{store.spec.buffer.name}{list(store.spec.indices)}"
        lines.append(f"store {sink} [{store.kind.value}] mask={'yes' if store.mask else 'no'}")
        _render_value(store.value, lines, 1)
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
                plan_text = render_plan(plan)
                print(plan_text, "*************************\n\n")
                return emit_plan(plan)
            except Unsupported:
                return None
        return None

    return tirx.stmt_functor.ir_transform(stmt, rewrite, None, ["tirx.For"])
