"""值 DAG：把标量表达式 lift 成「沿向量轴」的向量值节点。

决策仍然只依赖「沿向量轴的差分」：

* ``BufferLoad`` 的 ``Δ_v == 1`` → ``vload``（连续）；
* ``Δ_v == 0`` → ``uniform_load``（标量读一次 + ``vdup`` 广播）；
* ``Δ_v == k`` → ``vload_strided``（发射成 ``vgather2``）；
* 间接（下标含 ``BufferLoad``）→ 每 lane 索引向量（``gather``）；
* 循环变量本身 → ``lane_index``（``vci``）。

:class:`ValueNode` 是**指令选择无关**的：``recipe`` 只是原语名，真正的 ``T.simd.*``
指令选择推迟到发射层（:mod:`~tilelang.ascend.analysis.polyhedral.emit`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tvm import arith, tirx

from tilelang.ascend.language.simd import _vec_dtype

from .access import AccessClass, AccessInfo, Axis, _const_int, classify_access
from .primitives import (
    _BINARY_RECIPE,
    _BOOL_BINARY_RECIPE,
    _BOOL_UNARY_RECIPE,
    _COMPARE_FLIP,
    _COMPARE_PRED,
    _MASK_DTYPE,
    _UNARY_RECIPE,
    _UNIFORM_RECIPES,
    _VEC_SCALAR_BINARY,
    Unsupported,
    _call_tail,
    _elem_dtype,
    _is_atomic_name,
)


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
    * 间接（下标含 ``BufferLoad``）→ ``gather``（每 lane 索引向量）；
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

    条件有两种来源，发射端分别处理（见 ``emit._Emitter._build`` 的 ``vsel`` 分支）：

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


def _negate_uniform(node: ValueNode) -> ValueNode | None:
    """把 lane 均匀标量取负（用于 ``uniform - vector`` → ``vadds(vector, -uniform)``）。

    SIMD 层没有 ``vsubs``，但标量取负是标量开销：``g[i] - g[j]`` 发成
    ``vadds(vld(g[j]), -g[i])`` 比 ``vdup(g[i]) + vsub`` 少一条向量指令。
    """
    if node.recipe in ("imm", "uniform_expr"):
        scalar = node.scalar
    elif node.recipe == "uniform_load" and node.access is not None:
        scalar = tirx.BufferLoad(node.access.buffer, list(node.access.indices))
    elif node.recipe == "vbrc" and node.args:
        return _negate_uniform(node.args[0])
    else:
        return None
    if scalar is None:
        return None
    return ValueNode("uniform_expr", node.dtype, scalar=-scalar, detail="negated uniform")


def _elementwise(recipe: str, expr, args: list, ctx: _ValueContext) -> ValueNode:
    """二元/一元逐元素算子：若一侧 lane 均匀，优先用标量-向量指令。"""
    if all(arg.is_uniform for arg in args):
        return _uniform(expr)
    if len(args) == 2:
        lhs_uniform, rhs_uniform = args[0].is_uniform, args[1].is_uniform
        if lhs_uniform and not rhs_uniform and recipe == "vsub":
            negated = _negate_uniform(args[0])
            if negated is not None:
                return ValueNode("vadds", ctx.vector_dtype, [args[1], negated], detail="scalar vsub->vadds(-)")
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

    ``info.offset`` 已经过 ``access.index_vector_reason`` 校验（非常数 Δ 且可构造），
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
    ``TRUE_RANDOM`` 明确拒绝并带上 ``access.index_vector_reason`` 的诊断。
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
        if stride is not None and stride < 0:
            # 负步长（Δ ≤ -2）：无符号 lane 斜坡乘负步长表达不了负方向（缺陷 D8），改走
            # ``vgather2`` 的每 lane 索引向量——它的补码表示本就支持负偏移。
            return ValueNode("gather", ctx.vector_dtype, [index_value(info, ctx)], access=info, detail=f"stride={stride} (index vector)")
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
