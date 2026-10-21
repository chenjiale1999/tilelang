"""规划层：迭代域拆分、轴选择、归约判定与 ``VectorPlan`` / ``VectorProgram`` 构造。

本层负责产出计划（数据结构本身在 :mod:`model`）：计划里只有轴、lane 数、chunk、
每个 store 的值 DAG，**没有任何指令名**；指令选择推迟到
:mod:`~tilelang.ascend.analysis.polyhedral.emit`。

``build_plan`` 恒返回 :class:`VectorProgram`（1..N 段）：

* 单段路径（:func:`_split_nest` + :func:`_prepare_segment` + :func:`_build_plan_for_axis`）
  处理「一个主体 + 可选 ``pre``/``post``」；
* 多段路径（:func:`_try_build_program`）处理 ``T.Parallel`` 体内**多个并列干净
  ``T.serial`` 循环**，逐段独立向量化并按源码顺序发射。
"""

from __future__ import annotations

from dataclasses import dataclass

from tvm import arith, tirx

from tilelang.ascend.language.simd import _mask_bits, _vec_dtype, _vreg_lanes

from .access import AccessClass, AccessInfo, Axis, _const_int, _free_vars, classify_access, finite_difference, linear_element_offset
from .body import StoreSpec, _collect_aliases, normalize_body
from .model import ReduceInfo, StorePlan, VectorPlan, VectorProgram
from .primitives import (
    _REDUCE_OPS,
    _REDUCE_RECIPE,
    _REDUCE_SCALAR_RECIPE,
    Unsupported,
    _elem_dtype,
    _same_buffer,
    _same_indices,
    _seq,
)
from .value import ValueNode, _ValueContext, build_value, index_value


def _split_nest(stmt: tirx.Stmt, analyzer: arith.Analyzer) -> tuple[list[Axis], Axis | None, list[tirx.Stmt], tirx.Stmt, list[tirx.Stmt]]:
    """收集迭代域，并把 body 拆成 ``(pre, main, post)`` 三段。

    迭代域 = 连续的 ``T.Parallel`` 轴 + **直接的、干净的** ``T.serial`` 轴：

    * **直接内嵌链**（``for i in P: for j in serial: body``）—— 一路下行即可吸收；
    * **语句列表分段**（方案 §6 第二步）：当平行链结束后是一个 ``SeqStmt``，其中
      **唯一一个**干净的 ``T.serial`` 循环也可被吸收——它之前的兄弟语句是 ``pre``、
      之后的是 ``post``（都在 serial 轴**外层**、每个并行迭代执行一次），循环体则是
      被向量化的 ``main``。``pre``/``post`` 天然不含 serial 变量，故沿该轴 ``Δ==0``。

    serial 轴只有在 :func:`_serial_is_clean` 通过时才被吸收；否则保持原样，由
    :func:`~tilelang.ascend.analysis.polyhedral.body.normalize_body` 以「nested serial
    loop」明确拒绝——吸收是纯增量，不改变任何既有拒绝行为。

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
    for spec in (*pre_specs, *main_specs, *post_specs):
        _require_supported_element(spec.buffer)

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
    否则返回 ``None``，让既有单段路径（:func:`_split_nest` / ``normalize_body``）原样
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
    这样的轴）是关键——``emit._emit_segment_stmts`` 会按 ``reversed(axes)`` 包裹剩余的
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

    extent = _const_int(vector_axis.extent, analyzer)
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
        kind, stride = info.axis_class(vector_axis)
        value = build_value(spec.value, ctx)
        mask = build_value(spec.mask, ctx) if spec.mask is not None else None
        # 负步长与 PERMUTE / PIECEWISE / INDIRECT 共用「每 lane 索引向量」通道（缺陷 D8）。
        needs_index = kind in (AccessClass.PERMUTE, AccessClass.PIECEWISE, AccessClass.INDIRECT) or (
            kind == AccessClass.STRIDED and stride is not None and stride < 0
        )
        index = index_value(info, ctx) if needs_index else None
        return StorePlan(spec=spec, access=info, value=value, mask=mask, kind=kind, index=index)

    for spec, info in zip(main_specs, main_infos):
        plan.stores.append(_plan_store(spec, info))
    for spec, info in zip(pre_specs, pre_infos):
        plan.pre_stores.append(_plan_store(spec, info))
    for spec, info in zip(post_specs, post_infos):
        plan.post_stores.append(_plan_store(spec, info))
    # pre/post 必须是沿向量轴均匀的语句（与 serial 循环同级的语句天然满足；间接下标若沿轴
    # 不变也归 UNIFORM，见 :func:`classify_access`）。这是**防御性**门禁：把原先只在发射期
    # （``emit._emit_outer_store``）的检查前移到 build 期，保证 build 成功 ⇒ emit 成功。
    for where, stores in (("before", plan.pre_stores), ("after", plan.post_stores)):
        for store in stores:
            if store.kind != AccessClass.UNIFORM:
                raise Unsupported(
                    f"a statement {where} the serial loop must be uniform along vector axis "
                    f"'{vector_axis.name}' (got {store.kind.value}) (方案 §6 第二步)"
                )
    # 归约判定必须等值 DAG 建好之后（D13 统一判据）：逐 store 剥离旧值，顺带在 build 期
    # 就把不支持的形态拒绝掉，保证 build 成功 ⇒ emit 成功。
    plan.reduce = _classify_reduce(plan, analyzer)
    # 缺陷 A0：吸收的 serial 轴上，「部分归约」没有向量形式。lane 均匀的地址配逐 lane 变化的
    # 取值，只有**全段归约**（:func:`_classify_reduce` 成功）才能表达；否则逐元素路径会用
    # ``ONEPT`` 只保留 lane 0，把其余迭代的累加静默丢掉（结果错误且无诊断）。
    if plan.reduce is None and not vector_axis.is_parallel:
        for store in plan.stores:
            if store.kind == AccessClass.UNIFORM and not store.value.is_uniform:
                raise Unsupported(
                    f"store to '{store.spec.buffer.name}' is lane-uniform along the absorbed serial axis "
                    f"'{vector_axis.name}' but its value varies with it: a partial reduction mixed with "
                    "element-wise stores has no vector form (方案 §2 Case II)"
                )
    # 缺陷 D9（归约分支）：归约的 chunk 循环在编译期展开，符号 extent 无法展开；宁可拒绝，
    # 也不能像旧实现那样只展开 1 个 chunk（静默少算）。
    if plan.reduce is not None and plan.chunks is None:
        raise Unsupported(
            "a reduction needs a compile-time vector-axis extent (the chunk loop is unrolled at build time): "
            f"axis '{vector_axis.name}' has symbolic extent {vector_axis.extent}"
        )
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


def _require_supported_element(buffer) -> None:
    """store 的元素类型必须能推出谓词宽度（``_mask_bits`` 的能力边界）。

    ``bool`` 是典型反例：``_vreg_lanes("bool")`` 给出 256 而不报错，但 ``_mask_bits("bool")``
    在发射期抛 ``ValueError``——于是出现「build 成功、emit 崩溃」（与缺陷 D11 同族）。
    在 build 期统一转成 ``Unsupported``，保持「拒绝一律走 Unsupported」的约定。
    """
    elem = _elem_dtype(buffer.dtype)
    try:
        _mask_bits(elem)
    except ValueError as err:
        raise Unsupported(f"store element dtype '{elem}' of buffer '{buffer.name}' is not supported: {err}") from err
