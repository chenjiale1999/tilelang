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
from tvm.ir import Range

from tilelang.ascend.language.simd import _vec_dtype

from .access import AccessClass, AccessInfo, Axis, _const_int, _free_vars, classify_access
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
    _same_buffer,
    _same_indices,
)


@dataclass
class ValueNode:
    """一个值节点：recipe 是原语名，args 是操作数。

    ``predicate`` 是**装载守卫**：该值只在谓词为真的 lane 上有效（P0-3）。它只出现在
    ``if_then_else`` 这类惰性条件的分支里——分支里的访存只有在条件成立时才应当发生；
    发射层把这个谓词合进 ``vgather2`` 的掩码，未选中 lane 不访问内存。

    ``predicate_free_until`` / ``predicate_free_from`` 给出该守卫**恒为真**的首/尾
    chunk 区间（见 :func:`_guard_free_bounds`）：``chunk < predicate_free_until`` 或
    ``chunk >= predicate_free_from`` 的块可以省掉掩码，把带谓词的 ``vgather2``
    退回普通 ``vld``。
    """

    recipe: str
    dtype: str
    args: list = field(default_factory=list)
    access: AccessInfo | None = None
    scalar: tirx.PrimExpr | None = None
    detail: str = ""
    predicate: ValueNode | None = None
    predicate_free_until: int = 0
    predicate_free_from: int | None = None

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


def _vec_of(dtype_like) -> str:
    """元素类型（或带 ``.dtype`` 的表达式）→ 向量寄存器类型。

    lane 数由**该元素类型自身**的位宽唯一决定，而不是整段共享一个 ``vector_dtype``：
    同一段里 f32 / int32 混用时（位宽相同、元素类型不同），每个值节点必须是自己的
    元素类型，否则掩码宽度/向量类型与实际寄存器不一致（P0-2）。
    """
    dtype = dtype_like if isinstance(dtype_like, str) else _dtype_of(dtype_like)
    return _vec_dtype(_elem_dtype(dtype))


# -- 值 DAG 上的「同址 / 跨址读」查询（plan 与 emit 共用，累加器判据的唯一来源） ----


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


def same_address_load(node: ValueNode, buffer, indices):
    """在值 DAG 里找「读同一 buffer、同一地址」的装载节点（累积写的旧值来源）。"""
    if (
        node.recipe in ("vload", "uniform_load")
        and node.access is not None
        and _same_buffer(node.access.buffer, buffer)
        and _same_indices(tuple(node.access.indices), tuple(indices))
    ):
        return node
    for arg in node.args or ():
        found = same_address_load(arg, buffer, indices)
        if found is not None:
            return found
    return None


def reads_other_address(node: ValueNode, buffer, indices) -> bool:
    """值 DAG 里是否读了 ``buffer`` 上**不同于** ``indices`` 的地址。

    累加器把「同址读改写」搬进寄存器后，同层其它语句若读该 buffer 的**别的**地址，
    会读到尚未回写的旧内存值（如 ``for j: s[0] += a[i,j]; c[i] = s[1]``）。同址读由
    发射层 memo 转发到寄存器，是安全的。
    """
    if (
        node.recipe in ("vload", "uniform_load")
        and node.access is not None
        and _same_buffer(node.access.buffer, buffer)
        and not _same_indices(tuple(node.access.indices), tuple(indices))
    ):
        return True
    return any(reads_other_address(arg, buffer, indices) for arg in node.args or ())


@dataclass
class _ValueContext:
    axes: list[Axis]
    vector_axis: Axis
    analyzer: arith.Analyzer
    # 惰性条件分支的装载守卫（P0-3）：``if_then_else`` 的每个分支只在条件成立时求值，
    # 分支里的访存必须挂上这个谓词，发射层才能把未选中 lane 从 gather 里屏蔽掉。
    guard: ValueNode | None = None
    # 该守卫恒为真的首/尾 chunk 区间：``[0, guard_free_until)`` 与
    # ``[guard_free_from, chunks)``。``until == 0`` / ``from is None`` 表示对应方向证不出。
    guard_free_until: int = 0
    guard_free_from: int | None = None
    # 向量寄存器 lane 数：按 chunk 边界做守卫证明时用来换算迭代域。
    lanes: int = 0


def _with_guard(
    ctx: _ValueContext,
    guard: ValueNode | None,
    free_until: int = 0,
    free_from: int | None = None,
) -> _ValueContext:
    if guard is None:
        return ctx
    return _ValueContext(
        axes=ctx.axes,
        vector_axis=ctx.vector_axis,
        analyzer=ctx.analyzer,
        guard=guard,
        guard_free_until=free_until,
        guard_free_from=free_from,
        lanes=ctx.lanes,
    )


def _combine_free_until(lhs: int, rhs: int) -> int:
    """两个守卫恒真区间的**前缀**合取：取较小者。"""
    return min(lhs, rhs)


def _combine_free_from(lhs: int | None, rhs: int | None) -> int | None:
    """两个守卫恒真区间的**后缀**合取：任一证不出就整体证不出。"""
    if lhs is None or rhs is None:
        return None
    return max(lhs, rhs)


def _prove_guard(expr, ctx: _ValueContext, lo: int, hi: int) -> bool:
    probe = arith.Analyzer()
    probe.bind(ctx.vector_axis.var, Range(lo, hi))
    return probe.can_prove(expr)


def _guard_free_bounds(expr, ctx: _ValueContext) -> tuple[int, int | None]:
    """守卫 ``expr`` 恒真的首/尾 chunk 区间 ``([0, until), [from, chunks))``。

    只做充分证明：在 ``axis.var ∈ [0, u*lanes)`` / ``[c*lanes, extent)`` 上 ``can_prove``。
    整段都恒真的情形不参与（``until`` 最多到 ``chunks-1``、``from`` 从 ``chunks-1`` 起），
    因为那时 :func:`_access_provably_in_bounds` 会直接丢掉守卫，不需要分块。
    """
    extent = _const_int(ctx.vector_axis.extent, ctx.analyzer)
    lanes = ctx.lanes
    if extent is None or lanes <= 0 or extent <= lanes:
        return 0, None
    chunks = (extent + lanes - 1) // lanes
    until = 0
    while until + 1 < chunks and _prove_guard(expr, ctx, 0, (until + 1) * lanes):
        until += 1
    frm: int | None = None
    for c in range(chunks - 1, 0, -1):
        if not _prove_guard(expr, ctx, c * lanes, extent):
            break
        frm = c
    return until, frm


def _uniform_guarded(expr, ctx: _ValueContext) -> ValueNode:
    """全均匀子表达式 → 标量求值一次（``vdup``）。

    若当前处在惰性条件分支里（``ctx.guard`` 非空），这个标量表达式可能包含访存；
    标量读**没有掩码形式**，无法保证「条件不成立就不访问」。此时逐个校验地址能否
    证明在所有 lane 内：能证就照常折叠成标量；证不了就明确拒绝（P0-3）。
    """
    if ctx.guard is not None:
        unpredicable: list[str] = []

        def visit(node):
            if not isinstance(node, tirx.BufferLoad) or unpredicable:
                return
            info = classify_access(node.buffer, tuple(node.indices), ctx.axes, ctx.analyzer, is_store=False)
            if not _access_provably_in_bounds(info, ctx):
                unpredicable.append(node.buffer.name)

        tirx.stmt_functor.post_order_visit(expr, visit)
        if unpredicable:
            raise Unsupported(
                f"lane-uniform load of '{unpredicable[0]}' under a lane-varying lazy conditional cannot be "
                "predicated: a scalar load has no masked form (方案 §1 Case II / P0-3)"
            )
    return _uniform(expr)


def _negate_predicate(node: ValueNode) -> ValueNode:
    """谓词取反：均匀条件折叠成标量 ``Not``，逐 lane 谓词发 ``pnot``。"""
    if node.is_uniform:
        return ValueNode("uniform_expr", node.dtype, scalar=tirx.Not(node.scalar), detail="negated predicate")
    if node.recipe == "pnot":
        return node.args[0]
    return ValueNode("pnot", _MASK_DTYPE, [node])


def _and_predicate(lhs: ValueNode | None, rhs: ValueNode | None) -> ValueNode | None:
    """两个条件谓词的合取（``None`` 表示无守卫）。"""
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    if lhs.is_uniform and rhs.is_uniform:
        return ValueNode("uniform_expr", lhs.dtype, scalar=tirx.And(lhs.scalar, rhs.scalar), detail="predicate conjunction")
    return ValueNode("pand", _MASK_DTYPE, [lhs, rhs])


def _element_count(buffer) -> int | None:
    total = 1
    for dim in buffer.shape:
        value = _const_int(dim)
        if value is None:
            return None
        total *= value
    return total


def _access_provably_in_bounds(info: AccessInfo, ctx: _ValueContext, lo: int = 0, hi: int | None = None) -> bool:
    """该访问在所有 lane / 所有外层迭代上是否**可证**落在 buffer 内。

    只用来决定「惰性条件分支里的 load 要不要谓词化」：能证安全就照常发 ``vld``，
    避免把常见的 ``select(cond, a[i], b[i])`` 退化成带谓词的 gather；证不了就必须
    带守卫。证不了 ≠ 越界——非条件上下文里的普通 load 仍按设计前提（源程序地址
    安全）发射。

    ``lo`` / ``hi`` 把向量轴的迭代域收缩到 ``[lo, hi)``（chunk 边界守卫证明用：只有
    守卫恒真且地址可证安全的 block 才能回到 ``vld``）。``hi`` 为空时取轴 extent。
    """
    if info.indirect:
        return False
    numel = _element_count(info.buffer)
    base = _const_int(info.buffer.elem_offset, ctx.analyzer)
    if numel is None or base is None:
        return False

    def axis_of(var):
        # 用 ``same_as`` 而不是对象身份：TVM 对同一个底层 Var 可能给出不同的 Python
        # 包装对象（与 simd.py 里 ``id`` vs ``same_as`` 的坑同源）。
        for axis in ctx.axes:
            if axis.var.same_as(var):
                return axis
        return None

    analyzer = arith.Analyzer()
    for var in _free_vars(info.offset):
        axis = axis_of(var)
        if axis is None:
            return False
        extent = _const_int(axis.extent, ctx.analyzer)
        if extent is None:
            return False
        if axis.var.same_as(ctx.vector_axis.var):
            start = lo if lo > 0 else 0
            end = extent if hi is None else min(hi, extent)
            if end <= start:
                return False
            analyzer.bind(var, Range(start, end))
        else:
            analyzer.bind(var, Range(0, extent))
    bound = analyzer.const_int_bound(analyzer.simplify(info.offset))
    return bound.min_value >= base and bound.max_value < base + numel


def build_value(expr, ctx: _ValueContext) -> ValueNode:
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
        return load_node(info, ctx)
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
        return _build_call(expr, ctx)

    node_name = type(expr).__name__
    name_l = node_name.lower()
    if node_name in ("FloorDiv", "FloorMod") and _arity(expr) == 2:
        return _floordiv_mod(node_name, expr, ctx)
    if name_l in _COMPARE_PRED and _arity(expr) == 2:
        args = [build_value(expr.a, ctx), build_value(expr.b, ctx)]
        return _compare(args, _COMPARE_PRED[name_l], ctx, expr)
    if name_l in _BINARY_RECIPE and _arity(expr) == 2:
        args = [build_value(expr.a, ctx), build_value(expr.b, ctx)]
        return _elementwise(_BINARY_RECIPE[name_l], expr, args, ctx)
    if name_l in _UNARY_RECIPE and _arity(expr) == 1:
        args = [build_value(expr.a, ctx)]
        return _elementwise(_UNARY_RECIPE[name_l], expr, args, ctx)
    if name_l in _BOOL_BINARY_RECIPE and _arity(expr) == 2:
        args = [build_value(expr.a, ctx), build_value(expr.b, ctx)]
        return _bool_node(_BOOL_BINARY_RECIPE[name_l], expr, args, ctx)
    if name_l in _BOOL_UNARY_RECIPE and _arity(expr) == 1:
        args = [build_value(expr.a, ctx)]
        return _bool_node(_BOOL_UNARY_RECIPE[name_l], expr, args, ctx)

    if isinstance(expr, tirx.Select):
        # ``tirx.Select`` 是**急切**求值（两臂都算）：条件不限制臂的合法性，
        # 只把当前守卫（外层惰性条件）传下去。
        args = [build_value(expr.condition, ctx), build_value(expr.true_value, ctx), build_value(expr.false_value, ctx)]
        if all(arg.is_uniform for arg in args):
            return _uniform_guarded(expr, ctx)
        return _select_node(args, ctx, expr)

    if node_name == "Cast":
        arg = build_value(expr.value, ctx)
        if arg.is_uniform:
            return _uniform_guarded(expr, ctx)
        # 目标元素类型（去掉 lane 维），发射时交给 ``T.simd.vcvt`` 决定返回 lane 数。
        return ValueNode("vcvt", _vec_of(expr), [arg], detail=str(_dtype_of(expr)).split("x", 1)[0])

    raise Unsupported(f"unsupported value expression: {node_name} ({expr})")


def _uniform(expr) -> ValueNode:
    """lane 内均匀的子表达式：直接留用原始标量表达式，发射时一次求值 + ``vdup``。"""
    return ValueNode("uniform_expr", _dtype_of(expr), scalar=expr, detail="lane-uniform")


def _select_node(args: list, ctx: _ValueContext, expr) -> ValueNode:
    """``select(cond, a, b)`` → ``vsel``。

    条件有两种来源，发射端分别处理（见 ``emit._Emitter._build`` 的 ``vsel`` 分支）：

    * **逐 lane 谓词**（``cond`` 随向量轴变化，如 ``vcmp``）：直接作为 ``vsel`` 的 mask；
    * **循环不变条件**（``cond`` 沿向量轴均匀）：在发射端广播成全 lane 谓词
      ``update_mask(cond * lanes)``——整个 chunk 同真同假，因此不需要把分支提升到
      标量层，也不需要逐 lane 比较。这正是 §1 Case II 的三元写法。
    """
    detail = "uniform-cond select" if args[0].is_uniform else "predicated select"
    return ValueNode("vsel", _vec_of(expr), args, detail=detail)


def _bool_node(recipe: str, expr, args: list, ctx: _ValueContext) -> ValueNode:
    if all(arg.is_uniform for arg in args):
        return _uniform_guarded(expr, ctx)
    return ValueNode(recipe, _MASK_DTYPE, args)


def _compare(args: list, predicate: str, ctx: _ValueContext, expr) -> ValueNode:
    """比较：一侧 lane 均匀 → ``vcmps``，否则 ``vcmp``。结果类型是谓词。

    标量形式要求操作数顺序为 ``(vector, scalar)``；交换操作数时谓词必须取反，
    否则 ``i < j``（i 均匀）会被错误地发成 ``j < i``。
    """
    lhs_uniform, rhs_uniform = args[0].is_uniform, args[1].is_uniform
    if lhs_uniform and rhs_uniform:
        return _uniform_guarded(expr, ctx)
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
        return _uniform_guarded(expr, ctx)
    if len(args) == 2:
        lhs_uniform, rhs_uniform = args[0].is_uniform, args[1].is_uniform
        # ``vector - scalar`` → ``vadds(vector, -scalar)``：标量取负是标量开销，省掉
        # 一条 ``vdup``。**反向不能这样折叠**——``vadds`` 是 ``vector + scalar``，
        # 用它表达 ``scalar - vector`` 会得到 ``vector - scalar``（符号相反）。
        # ``scalar - vector`` 走下面的通用回退：``vsub(vdup(scalar), vector)``。
        if recipe == "vsub" and rhs_uniform and not lhs_uniform:
            negated = _negate_uniform(args[1])
            if negated is not None:
                return ValueNode("vadds", _vec_of(expr), [args[0], negated], detail="vector - scalar -> vadds(-)")
        if lhs_uniform or rhs_uniform:
            scalar_recipe = _VEC_SCALAR_BINARY.get(recipe)
            if scalar_recipe is not None:
                ordered = [args[1], args[0]] if lhs_uniform else [args[0], args[1]]
                return ValueNode(scalar_recipe, _vec_of(expr), ordered, detail=f"scalar {recipe}->{scalar_recipe}")
            if lhs_uniform:
                return ValueNode(recipe, _vec_of(expr), [broadcast(args[0], ctx), args[1]])
            return ValueNode(recipe, _vec_of(expr), [args[0], broadcast(args[1], ctx)])
    return ValueNode(recipe, _vec_of(expr), args)


def broadcast(node: ValueNode, ctx: _ValueContext) -> ValueNode:
    if node.recipe == "vbrc":
        return node
    return ValueNode("vbrc", _vec_of(node.dtype), [node], detail="broadcast uniform")


def _floordiv_mod(node_name: str, expr, ctx: _ValueContext) -> ValueNode:
    """``FloorDiv`` / ``FloorMod`` 作*值*算子（方案 §5.4(a)）。

    SIMD 层没有整型 ``vdiv`` / ``vmod``（``vdiv`` 是浮点 SFU），所以对**2 的幂**
    常量除数做精确的 strength-reduction：

    * ``x // 2^k`` → ``vshrs(x, k)``：算术右移对负数即向下取整，与 ``FloorDiv`` 同义；
    * ``x % 2^k`` → ``vand(x, 2^k - 1)``：补码按位与正好给出非负余数，与 ``FloorMod`` 同义。

    非 2 的幂常量除数尚未做 magic-number 化简（需 64 位中间量 ``vmull``），
    非常量除数需要硬件整型除法内建——两者都**明确拒绝**而不静默降级。
    两侧都 lane 均匀时整式退化成一个标量，不产生向量指令。
    """
    numer = build_value(expr.a, ctx)
    denom = build_value(expr.b, ctx)
    if numer.is_uniform and denom.is_uniform:
        return _uniform_guarded(expr, ctx)

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
        analyzer=ctx.analyzer,
        guard=ctx.guard,
        guard_free_until=ctx.guard_free_until,
        guard_free_from=ctx.guard_free_from,
        lanes=ctx.lanes,
    )
    return build_value(info.offset, index_ctx)


def load_node(info: AccessInfo, ctx: _ValueContext) -> ValueNode:
    """把一次已分类的访问变成值节点（连续/广播/步长/置换/分片/间接）。

    常数 Δ 走连续 / 广播 / 步长；非常数 Δ 与间接下标统一走**每 lane 索引向量**
    通道（``PERMUTE`` / ``PIECEWISE`` / ``INDIRECT``，以及负步长 ``STRIDED``），
    发射成 ``vgather2``。真正无结构的
    ``TRUE_RANDOM`` 明确拒绝并带上 ``access.index_vector_reason`` 的诊断。

    节点 dtype 取**被访问 buffer 自身的元素类型**（而不是整段共享的那个），这样
    同段内 f32 / int32 混用时掩码宽度与寄存器类型仍与访问一致（P0-2）。
    """
    if info.in_global:
        raise Unsupported(
            "global-memory operand inside T.Parallel (方案 §0 Case II): needs an explicit UB staging copy before the parallel nest"
        )
    # 惰性条件分支里的访问只有在条件成立时才合法：能证安全就不必带守卫（保持常见的
    # ``select(cond, a[i], b[i])`` 仍发 vld），否则把条件挂到装载节点上（P0-3）。
    # ``free_until`` / ``free_from`` 是守卫恒真的首/尾 chunk 区间；只有**守卫恒真且该段
    # 地址同样可证安全**时才能在这段里省掉掩码（发射层据此回退普通 ``vld``）。
    guard = ctx.guard
    free_until = ctx.guard_free_until
    free_from = ctx.guard_free_from
    if guard is not None:
        if _access_provably_in_bounds(info, ctx):
            guard = None
            free_until, free_from = 0, None
        else:
            if free_until <= 0 or not _access_provably_in_bounds(info, ctx, hi=free_until * ctx.lanes):
                free_until = 0
            if free_from is None or not _access_provably_in_bounds(info, ctx, lo=free_from * ctx.lanes):
                free_from = None
    vec_dtype = _vec_of(info.buffer.dtype)
    kind, stride = info.axis_class(ctx.vector_axis)
    if kind == AccessClass.CONTIGUOUS:
        return ValueNode(
            "vload",
            vec_dtype,
            access=info,
            predicate=guard,
            predicate_free_until=free_until,
            predicate_free_from=free_from,
        )
    if kind == AccessClass.UNIFORM:
        if guard is not None:
            raise Unsupported(
                f"lane-uniform load of '{info.buffer.name}' under a lane-varying lazy conditional cannot be "
                "predicated: a scalar load has no masked form (方案 §1 Case II / P0-3)"
            )
        return ValueNode("uniform_load", _dtype_of_scalar(info), access=info, detail="lane-uniform")
    if kind == AccessClass.STRIDED:
        if stride is not None and stride < 0:
            # 负步长（Δ ≤ -2）：无符号 lane 斜坡乘负步长表达不了负方向（缺陷 D8），改走
            # ``vgather2`` 的每 lane 索引向量——它的补码表示本就支持负偏移。
            return ValueNode(
                "gather",
                vec_dtype,
                [index_value(info, ctx)],
                access=info,
                detail=f"stride={stride} (index vector)",
                predicate=guard,
                predicate_free_until=free_until,
                predicate_free_from=free_from,
            )
        return ValueNode(
            "vload_strided",
            vec_dtype,
            access=info,
            detail=f"stride={stride}",
            predicate=guard,
            predicate_free_until=free_until,
            predicate_free_from=free_from,
        )
    if kind in (AccessClass.PERMUTE, AccessClass.PIECEWISE, AccessClass.INDIRECT):
        return ValueNode(
            "gather",
            vec_dtype,
            [index_value(info, ctx)],
            access=info,
            detail=kind.value,
            predicate=guard,
            predicate_free_until=free_until,
            predicate_free_from=free_from,
        )
    raise Unsupported(
        f"non-affine access with respect to axis '{ctx.vector_axis.name}': "
        f"the per-lane index vector is not constructible ({info.reason or 'unknown reason'})"
    )


def _dtype_of_scalar(info: AccessInfo) -> str:
    return str(info.buffer.dtype)


def _build_call(expr: tirx.Call, ctx: _ValueContext) -> ValueNode:
    tail = _call_tail(expr)
    if _is_atomic_name(tail):
        raise Unsupported("atomic op inside T.Parallel is not supported (方案 §5)")
    if tail in ("if_then_else", "select"):
        condition = build_value(expr.args[0], ctx)
        # ``T.if_then_else`` 是**惰性**内建：臂只在条件成立时求值，因此臂里的访存
        # 必须带条件谓词（P0-3）。``select`` 内建是急切求值，不额外加守卫。
        if tail == "if_then_else":
            cond_until, cond_from = _guard_free_bounds(expr.args[0], ctx)
            neg_until, neg_from = _guard_free_bounds(tirx.Not(expr.args[0]), ctx)
            # 没有外层守卫时本条件就是完整守卫；有外层守卫时两个方向分别取交。
            if ctx.guard is None:
                then_until, then_from = cond_until, cond_from
                else_until, else_from = neg_until, neg_from
            else:
                then_until = _combine_free_until(ctx.guard_free_until, cond_until)
                then_from = _combine_free_from(ctx.guard_free_from, cond_from)
                else_until = _combine_free_until(ctx.guard_free_until, neg_until)
                else_from = _combine_free_from(ctx.guard_free_from, neg_from)
            then_ctx = _with_guard(ctx, _and_predicate(ctx.guard, condition), then_until, then_from)
            else_ctx = _with_guard(ctx, _and_predicate(ctx.guard, _negate_predicate(condition)), else_until, else_from)
        else:
            then_ctx = else_ctx = ctx
        args = [condition, build_value(expr.args[1], then_ctx), build_value(expr.args[2], else_ctx)]
        if all(arg.is_uniform for arg in args):
            # 整式沿向量轴均匀：保留原始标量表达式（``?:`` 的惰性由 codegen 保证）。
            return _uniform_guarded(expr, ctx)
        return _select_node(args, ctx, expr)
    if tail in _COMPARE_PRED and len(expr.args) == 2:
        args = [build_value(arg, ctx) for arg in expr.args]
        return _compare(args, _COMPARE_PRED[tail], ctx, expr)
    if tail in _BOOL_BINARY_RECIPE and len(expr.args) == 2:
        args = [build_value(arg, ctx) for arg in expr.args]
        return _bool_node(_BOOL_BINARY_RECIPE[tail], expr, args)
    if tail in _BOOL_UNARY_RECIPE and len(expr.args) == 1:
        args = [build_value(expr.args[0], ctx)]
        return _bool_node(_BOOL_UNARY_RECIPE[tail], expr, args)
    if tail in _BINARY_RECIPE and len(expr.args) == 2:
        args = [build_value(arg, ctx) for arg in expr.args]
        return _elementwise(_BINARY_RECIPE[tail], expr, args, ctx)
    if tail in _UNARY_RECIPE and len(expr.args) == 1:
        args = [build_value(expr.args[0], ctx)]
        return _elementwise(_UNARY_RECIPE[tail], expr, args, ctx)
    raise Unsupported(f"unsupported intrinsic call in T.Parallel body: {tail}")
