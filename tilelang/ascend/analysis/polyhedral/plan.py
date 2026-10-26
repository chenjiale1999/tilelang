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

from tilelang.ascend.language.simd import _mask_bits, _vreg_lanes

from .access import AccessClass, AccessInfo, Axis, _const_int, _free_vars, classify_access, finite_difference, linear_element_offset
from .body import StoreSpec, _collect_aliases, normalize_body
from .model import ReduceInfo, StorePlan, VectorPlan, VectorProgram, accumulator_candidate, is_private_scope
from .primitives import (
    _REDUCE_OPS,
    _REDUCE_RECIPE,
    _REDUCE_SCALAR_RECIPE,
    Unsupported,
    _elem_dtype,
    _for_kind_name,
    _int,
    _reduce_identity,
    _same_buffer,
    _same_indices,
    _seq,
)
from .value import (
    ValueNode,
    _ValueContext,
    _and_predicate,
    _combine_free_from,
    _combine_free_until,
    _guard_free_bounds,
    _reads_buffer,
    _with_guard,
    build_value,
    index_value,
)


def _require_zero_min(loop: tirx.For, analyzer: arith.Analyzer) -> None:
    """迭代域建模成 ``0 <= x < extent``，因此循环下界必须**可证为 0**。

    ``Axis`` 只记录 ``(var, extent)``，发射时按 ``For(var, 0, extent)`` 重建；源程序的
    非零下界（``T.serial(1, K)`` / ``T.Parallel(1, K)``）会被静默丢掉，变成从 0 开始
    的**错位下标**（R5）。无法证明下界为 0 时明确拒绝。
    """
    if _const_int(loop.min, analyzer) != 0:
        raise Unsupported(f"{_for_kind_name(loop.kind)} loop '{loop.loop_var.name}' must start at 0 to be vectorized (min={loop.min})")


def _absorb_parallel_chain(stmt: tirx.Stmt, analyzer: arith.Analyzer) -> tuple[list[Axis], tirx.Stmt]:
    """吸收**连续**的 ``T.Parallel`` 轴（单段路径与多段路径共用）。"""
    axes: list[Axis] = []
    current: tirx.Stmt = stmt
    while isinstance(current, tirx.For) and current.kind == tirx.ForKind.PARALLEL:
        _require_zero_min(current, analyzer)
        axes.append(Axis(current.loop_var, current.extent, len(axes), tirx.ForKind.PARALLEL))
        current = current.body
    return axes, current


def _split_nest(
    stmt: tirx.Stmt, analyzer: arith.Analyzer, notes: list[str] | None = None
) -> tuple[list[Axis], Axis | None, list[tirx.Stmt], tirx.Stmt, list[tirx.Stmt]]:
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
    axes, current = _absorb_parallel_chain(current, analyzer)
    while isinstance(current, tirx.For):
        kind = current.kind
        _require_zero_min(current, analyzer)
        if kind == tirx.ForKind.SERIAL:
            clean, reason = _serial_is_clean(current, analyzer)
            if not clean:
                _record_serial_reason(notes, current, reason)
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
            if loop.kind == tirx.ForKind.SERIAL:
                clean, reason = _serial_is_clean(loop, analyzer)
                if clean:
                    serial_axis = Axis(loop.loop_var, loop.extent, len(axes), tirx.ForKind.SERIAL)
                    axes.append(serial_axis)
                    pre = list(current[:pos])
                    post = list(current[pos + 1 :])
                    current = loop.body
                else:
                    _record_serial_reason(notes, loop, reason)

    if not any(axis.is_parallel for axis in axes):
        raise Unsupported("no T.Parallel axis found")
    return axes, serial_axis, pre, current, post


def _record_serial_reason(notes: list[str] | None, loop: tirx.For, reason: str) -> None:
    """把 ``_serial_is_clean`` 的具体原因记进诊断出口（去重、带循环名）。

    顶层消息原本只有一句泛化的「nested serial loop」；把「跨迭代读 ``c`` 沿 ``j``」
    这类真正原因带上，用户才知道该改哪里。
    """
    if notes is None:
        return
    note = f"serial loop '{loop.loop_var.name}': {reason or 'loop-carried dependence'}"
    if note not in notes:
        notes.append(note)


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
        if _const_int(current.min, analyzer) != 0:
            return None  # 非零下界：交给单段路径给出精确拒绝（_require_zero_min）
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
    _require_uniform_element_width((*pre_specs, *main_specs, *post_specs))

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
        lanes=lanes,
    )


def _plan_segment(
    axes: list[Axis],
    serial_axis: Axis | None,
    pre_stmts: list[tirx.Stmt],
    main_stmt: tirx.Stmt,
    post_stmts: list[tirx.Stmt],
    aliases: dict,
    analyzer: arith.Analyzer,
    lanes: int | None,
    *,
    single: bool,
    base_notes: list[str],
) -> tuple[VectorPlan, bool]:
    """一段的规划主体：**先试展平通道**，再按轴打分逐个回退。

    单段路径（:func:`build_plan`）与多段路径（:func:`_build_segment`）共用这一份实现
    ——两条路径原先各维护一遍「展平 → 轴排序 → 逐个尝试」，诊断措辞与失败语义容易
    漂移。``single`` 只影响诊断措辞（"nest"/"segment"）；``base_notes`` 是调用方已经
    收集到的前置说明（如多段路径试过但失败的段级原因）。

    返回 ``(plan, flattened)``；``flattened`` 表示合成轴已经把并行前缀吃掉，program
    层不能再包一次。失败抛 ``Unsupported``（带全量逐轴诊断）。
    """
    flatten_word = "nest" if single else "segment"
    specs = _prepare_segment(axes, serial_axis, pre_stmts, main_stmt, post_stmts, aliases, analyzer, lanes)
    load_infos = _load_infos(axes, specs.all_specs, analyzer)
    diagnostics = [*base_notes, *_axis_diagnostics(axes, specs.all_infos, load_infos, analyzer, specs.lanes)]

    flattened, flatten_note = _try_flatten(axes, pre_stmts, main_stmt, post_stmts, specs.lanes, analyzer)
    if flatten_note:
        diagnostics.append(flatten_note)
    if flattened is not None:
        flat_axis = flattened.axes[0]
        try:
            plan = flattened.plan_for(flat_axis, analyzer)
        except Unsupported as err:
            diagnostics.append(f"flattened axis '{flat_axis.name}': {err}")
        else:
            plan.diagnostics = [
                f"flattened {flatten_word} [{', '.join(axis.name for axis in axes)}] -> '{flat_axis.name}' "
                f"(extent {flat_axis.extent}, full-lane chunks)",
                *diagnostics,
            ]
            _note_guard_split(plan)
            return plan, True

    errors: list[str] = []
    ranked = rank_vector_axes(axes, specs.all_infos, load_infos, analyzer, specs.lanes)
    if not ranked:
        raise Unsupported(
            "no vectorizable axis: every store is true-random (its index cannot be built in the index-vector channel) on all axes",
            diagnostics=diagnostics,
        )
    for axis in ranked:
        try:
            plan = specs.plan_for(axis, analyzer)
        except Unsupported as err:
            # 逐轴失败原因**成功时也要保留**（改进 7）：否则最终计划上看不出"为什么
            # 没选分数最高的轴"，只剩另一根轴的 selected 行。
            errors.append(f"axis '{axis.name}': {err}")
            diagnostics.append(f"axis '{axis.name}': {err} -> fallback")
            diagnostics.extend(f"  {note}" for note in (err.diagnostics or []))
            continue
        if single:
            diagnostics.append(
                f"selected vector axis '{axis.name}' (score={_axis_score(specs.all_infos, load_infos, axis, analyzer, specs.lanes)})"
            )
        else:
            diagnostics.append(
                f"segment vector axis '{axis.name}' (active={_active_lanes(axis, analyzer, specs.lanes)}/{specs.lanes}, 方案 §6 第三步)"
            )
        plan.diagnostics = diagnostics
        _note_guard_split(plan)
        return plan, False
    raise Unsupported("no vectorizable axis: " + "; ".join(errors), diagnostics=diagnostics)


def _note_guard_split(plan: VectorPlan) -> None:
    """把守卫恒真区间的 chunk 切点记进诊断（成功路径也要保留，改进 7）。"""
    if plan.guard_boundaries:
        plan.diagnostics.append(f"guard-free chunk cuts at {plan.guard_boundaries}: masked gathers revert to plain vld")


def _build_segment(
    axes: list[Axis],
    serial_axis: Axis,
    pre_stmts: list[tirx.Stmt],
    main_stmt: tirx.Stmt,
    post_stmts: list[tirx.Stmt],
    aliases: dict,
    analyzer: arith.Analyzer,
    lanes: int | None,
) -> tuple[VectorPlan, bool]:
    """为一段（一个 serial 链 + 其 ``pre``/``post``）规划出单段 :class:`VectorPlan`。

    与单段路径共用 :func:`_plan_segment`（展平优先 + 轴打分回退）；该段携带自己的
    诊断（改进 7），与单段路径的 ``plan.diagnostics`` 同构。
    """
    return _plan_segment(
        axes,
        serial_axis,
        pre_stmts,
        main_stmt,
        post_stmts,
        aliases,
        analyzer,
        lanes,
        single=False,
        base_notes=[],
    )


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
    parallel_axes, current = _absorb_parallel_chain(stmt, analyzer)
    if not parallel_axes or not isinstance(current, tirx.SeqStmt):
        return None

    items: list[tuple[str, object]] = []
    for child in current:
        if isinstance(child, tirx.For):
            chain = _absorb_serial_chain(child, analyzer, len(parallel_axes))
            if chain is None:
                # 不吸收：把具体原因留下（顶层消息不再只有一句泛化的 nested serial loop）。
                _record_serial_reason(notes, child, _serial_is_clean(child, analyzer)[1])
                return None  # 不可吸收：保留既有「nested serial loop」拒绝
            items.append(("seg", chain))
        else:
            items.append(("stmt", child))

    seg_positions = [k for k, (kind, _) in enumerate(items) if kind == "seg"]
    if len(seg_positions) < 2:
        return None

    aliases = _collect_aliases(current)
    segments: list[VectorPlan] = []
    flattened_any = False
    previous = -1
    for order, pos in enumerate(seg_positions):
        pre = [items[k][1] for k in range(previous + 1, pos)]
        serial_axes, body = items[pos][1]
        post = [items[k][1] for k in range(pos + 1, len(items))] if order == len(seg_positions) - 1 else []
        try:
            segment, flattened = _build_segment([*parallel_axes, *serial_axes], serial_axes[-1], pre, body, post, aliases, analyzer, lanes)
        except Unsupported as err:
            notes.append(f"segment {order} (T.serial '{serial_axes[-1].name}'): {err}")
            notes.extend(f"  {note}" for note in err.diagnostics)
            return None
        segments.append(segment)
        flattened_any = flattened_any or flattened
        previous = pos
    # program 层只能提「第一根被某段用作**向量轴**的并行轴」之前的轴：
    # 向量轴已经吃成 lane 维，若再把它（或它内层的轴）包在最外层，整段向量化会被
    # 重复执行 `extent/lanes` 次（p_serial_reduce_then_use 的 64 次空转就是这么来的）。
    # 展平段同理：合成轴已经把并行前缀吃掉了。
    consumed = [segment.vector_axis.index for segment in segments if segment.vector_axis.is_parallel]
    first_consumed = min(consumed) if consumed else None
    hoisted = [] if flattened_any else [axis for axis in parallel_axes if first_consumed is None or axis.index < first_consumed]
    program = VectorProgram(axes=hoisted, segments=segments)
    program.diagnostics = [
        f"multi-segment program: {len(segments)} segments, source order preserved"
        + (", flattened/consumed parallel axes are wrapped by each segment" if not hoisted else "")
    ]
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
    * **跨迭代写写别名**：同一 buffer 被多条 store 写、且目的地址互不相同时，不同迭代
      可能写到同一个地址（``c[i,2*j]`` 与 ``c[i,j]``、或 ``c[i,j]`` 与 ``c[i,j+1]``），
      标量顺序下的"后写覆盖"在向量化后不成立 → 拒绝；
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
    offsets = [(store, linear_element_offset(store.buffer, tuple(store.indices), analyzer)) for store in stores]
    for store, offset in offsets:
        if _const_int(finite_difference(offset, var, analyzer), analyzer) == 0:
            if not _is_cumulative_write(store):
                return False, f"write to {store.buffer.name} is invariant along '{var.name}' but not a reduction"
            if not _value_varies_along(store.value, axis, analyzer):
                return False, f"reduction into {store.buffer.name} does not vary along '{var.name}'"

    for index, (first, first_offset) in enumerate(offsets):
        for second, second_offset in offsets[index + 1 :]:
            if _same_buffer(first.buffer, second.buffer) and not tirx.analysis.expr_deep_equal(first_offset, second_offset):
                return False, (
                    f"multiple writes to {first.buffer.name} at different addresses along '{var.name}' may alias "
                    "across iterations (last-write-wins is not preserved by vectorization, R6)"
                )

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
    store_infos: list[AccessInfo],
    load_infos: list[AccessInfo],
    analyzer: arith.Analyzer,
    lanes: int,
) -> list[Axis]:
    """按「联合可行性 + 读写加权 + lane 利用率」给候选向量轴排序（改进 2）。

    与原实现相比有三点变化：

    * **联合可行性**：候选轴只需「没有 store 是
      :attr:`AccessClass.TRUE_RANDOM`」且「至少一个 store 可写」——STRIDED /
      PIECEWISE / PERMUTE / INDIRECT 的 store 现在都能进候选（靠 ``vscatter``）；
    * **读加权**：读侧（:func:`_collect_load_exprs`）也计入打分，避免「写连续、
      读却四处乱跳」的轴被误选；
    * **轴回退**：返回的是**排序后的整条列表**，:func:`build_plan` 会依次尝试，
      某个轴的取值发射失败时自动退到次优轴，而不是直接判整个 nest 不可向量化。

    归约轴不再**无条件**排在最前：``vcadd`` 这类跨 lane 归约只有在轴能填满一个
    向量寄存器时才划算；extent 小于 lanes 的轴（典型是 §6 吸收的 ``T.serial`` 轴）
    要按 ``active/lanes`` 折算分数，否则 ``(64, 8)`` 形态会一直选 8/64 lane 的
    serial 轴，把并行轴上的满 lane 向量化漏掉。
    """
    ranked: list[tuple[int, Axis]] = []
    for axis in axes:
        if any(info.axis_class(axis)[0] == AccessClass.TRUE_RANDOM for info in store_infos):
            continue  # 写侧不可构造 → 该轴不可向量化
        ranked.append((_axis_score(store_infos, load_infos, axis, analyzer, lanes), axis))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [axis for _, axis in ranked]


def _axis_weight_sum(infos: list[AccessInfo], axis: Axis) -> int:
    """一根轴上所有访存的形态权重之和（写侧用，见 :data:`_AXIS_WEIGHT`）。"""
    return sum(_AXIS_WEIGHT[info.axis_class(axis)[0]] for info in infos)


def _active_lanes(axis: Axis, analyzer: arith.Analyzer, lanes: int) -> int:
    """轴每个 chunk 的有效 lane 数（常量 extent 取 ``min(extent, lanes)``）。

    符号 extent 假定可以填满（``chunks`` 运行时才确定）；常量小 extent（如吸收的
    serial 轴 extent=8）如实给出 8，供打分折算利用率。
    """
    extent = _const_int(axis.extent, analyzer)
    if extent is None:
        return lanes
    return max(min(extent, lanes), 1)


def _axis_score(store_infos: list[AccessInfo], load_infos: list[AccessInfo], axis: Axis, analyzer: arith.Analyzer, lanes: int) -> int:
    """一根候选轴的联合打分：写侧主序（×100）、读侧次之（×10）、再乘 lane 利用率。"""
    weight = 100 * _axis_weight_sum(store_infos, axis) + 10 * _axis_weight_sum(load_infos, axis)
    return weight * _active_lanes(axis, analyzer, lanes) // lanes + axis.index


def _axis_diagnostics(
    axes: list[Axis], store_infos: list[AccessInfo], load_infos: list[AccessInfo], analyzer: arith.Analyzer, lanes: int
) -> list[str]:
    """每根轴的分类摘要（改进 7）：一次性暴露全部卡点，无论最终成功与否。

    成功时它记录「哪些轴是候选、打分多少、比哪根轴差」；失败时它是失败的
    完整上下文，避免只报告第一个卡点。分数里已含 lane 利用率，``active`` 字段
    直接给出该轴每个 chunk 的有效 lane 数，便于定位「8/64 lane」这类浪费。
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
        active = _active_lanes(axis, analyzer, lanes)
        lines.append(
            f"{label}: store={store_classes} load={load_classes} -> candidate "
            f"(score={_axis_score(store_infos, load_infos, axis, analyzer, lanes)}, active={active}/{lanes})"
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


def _expr_reads_buffers(expr, buffers: list) -> bool:
    """表达式里是否读到了给定 buffer 之一（用于展平通道的写冲突门禁）。"""
    found: list = []

    def visit(node):
        if isinstance(node, tirx.BufferLoad) and any(_same_buffer(node.buffer, buffer) for buffer in buffers):
            found.append(True)

    tirx.stmt_functor.post_order_visit(expr, visit)
    return bool(found)


def _flatten_axes(axes: list[Axis], analyzer: arith.Analyzer) -> tuple[Axis, dict, int] | None:
    """把整条 nesting 展平成一个合成轴 ``k``（P1）。

    展平是「最内层轴不足一个向量寄存器」场景（``(64, 8)`` + 64 lane）的正解：单轴
    向量化只能用到 8/64 个 lane，还要用带谓词的 ``vgather2`` 装载；展平后是一次
    **满 lane 连续** ``vld``/``vsts``。

    每个成员轴 ``v_m`` 用 ``k`` 的分解式替换（行主序）::

        v_inner = k % E_inner
        v_mid   = (k // suffix_m) % E_m
        v_outer = k // suffix_outer          # 最外层不取模：k < Π E 恒成立

    最外层**不能**写成 ``(k // suffix) % E``：TVM 会把它化简成 ``k % Π E``，进而
    ``Δ(k)`` 不再是常数 1，连续性就丢了（实测）。

    只接受 2 的幂 extent：分解式的除数因此都是 2 的幂，chunk 基址代入后能被
    ``arith.Analyzer`` 化简成 ``chunk * <每 chunk 行数>``（否则会在生成的 C 代码里
    留下运行期整数除法）。

    返回 ``(合成轴, {成员轴变量: 分解式}, 总 extent)``；不适用时返回 ``None``。
    """
    if len(axes) < 2:
        return None
    extents = []
    for axis in axes:
        extent = _const_int(axis.extent, analyzer)
        if extent is None or extent < 2 or (extent & (extent - 1)):
            return None
        extents.append(extent)
    total = 1
    for extent in extents:
        total *= extent
    var = tirx.Var("k_" + "_".join(axis.name for axis in axes), "int32")
    mapping: dict = {}
    suffix = 1
    for position, (axis, extent) in enumerate(zip(reversed(axes), reversed(extents))):
        if position == 0:
            mapping[axis.var] = var % extent
        elif position == len(axes) - 1:
            mapping[axis.var] = var // suffix
        else:
            mapping[axis.var] = (var // suffix) % extent
        suffix *= extent
    return Axis(var, _int(total), 0, tirx.ForKind.PARALLEL), mapping, total


def _try_flatten(
    axes: list[Axis],
    pre_stmts: list[tirx.Stmt],
    main_stmt: tirx.Stmt,
    post_stmts: list[tirx.Stmt],
    lanes: int,
    analyzer: arith.Analyzer,
) -> tuple[_SegmentSpecs | None, str]:
    """尝试「合成连续轴」通道，返回 ``(specs, 拒绝原因)``。

    适用条件（全部满足才生成候选）：

    * 没有 ``pre``/``post`` 兄弟语句（它们引用外层并行变量，展平后不再是 lane 均匀的；
      保留原路径即可）；
    * 最内层轴的 extent **小于** lanes——这正是唯一需要展平的场景；``extent == lanes``
      或更大时现有单轴实现本就发出满 lane ``vld``，不做改动；
    * 全部 extent 是常量且为 2 的幂（见 :func:`_flatten_axes`）；
    * **写侧门禁**：每条 store 沿合成轴都是 ``CONTIGUOUS``（``Δ == 1``）——组内每个 k
      的写地址两两不同，故展平成并行轴不会引入写冲突，也天然排除归约/覆盖写；
    * **读侧门禁 1**：所有读只能是 ``CONTIGUOUS`` 或 ``UNIFORM``——展平只是把下标换成
      ``k`` 的分解式，**消不掉** gather（分片/置换/间接读取）；
    * **读侧门禁 2**：被写的 buffer 不得在体内被读（排除 stencil、累加、跨 k 依赖）。

    任何一步不满足都返回 ``(None, reason)``，由调用方记录诊断后走原有路径——**纯增量**，
    不改变任何既有行为。
    """
    if pre_stmts or post_stmts:
        return None, ""
    if len(axes) < 2:
        return None, ""
    innermost = _const_int(axes[-1].extent, analyzer)
    if innermost is None or innermost >= lanes:
        return None, ""  # 最内层已能填满一个向量寄存器：不是本次要解决的问题
    flat = _flatten_axes(axes, analyzer)
    if flat is None:
        return None, f"flatten rejected: extent product of [{', '.join(axis.name for axis in axes)}] is not a power-of-two chain"
    flat_axis, mapping, total = flat
    body = tirx.stmt_functor.substitute(main_stmt, mapping)
    try:
        specs = _prepare_segment([flat_axis], None, [], body, [], _collect_aliases(body), analyzer, lanes)
    except Unsupported as err:
        return None, f"flatten rejected: {err}"
    if any(info.axis_class(flat_axis)[0] != AccessClass.CONTIGUOUS for info in specs.main_infos):
        return None, (f"flatten rejected: not every store is contiguous along the synthesized axis '{flat_axis.name}' (extent {total})")
    # 读侧门禁：展平只负责「连续 + 广播」；若还有访存需要每 lane 索引向量（gather），
    # 展平并不能消掉 gather（只是换个轴算下标），就不接手，保持原单轴路径。
    load_infos = _load_infos([flat_axis], specs.all_specs, analyzer)
    if any(info.axis_class(flat_axis)[0] not in (AccessClass.CONTIGUOUS, AccessClass.UNIFORM) for info in load_infos):
        return None, (
            f"flatten rejected: a load along the synthesized axis '{flat_axis.name}' still needs an "
            f"index vector (gather) — flattening would not remove it"
        )
    targets = [spec.buffer for spec in specs.main_specs]
    for index, buffer in enumerate(targets):
        if any(_same_buffer(buffer, other) for other in targets[:index]):
            return None, (
                f"flatten rejected: two statements write the same buffer '{buffer.name}' along the synthesized "
                f"axis '{flat_axis.name}' (their addresses may overlap across k) — extent {total}"
            )
    for spec in specs.main_specs:
        for expr in (spec.value, spec.mask):
            if expr is not None and _expr_reads_buffers(expr, targets):
                return None, (
                    f"flatten rejected: a written buffer is also read inside the nest (no write-conflict-free flattening) — extent {total}"
                )
    return specs, ""


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
        axes, serial_axis, pre_stmts, main_stmt, post_stmts = _split_nest(stmt, analyzer, notes)
        pieces = [*pre_stmts, main_stmt, *post_stmts]
        aliases = _collect_aliases(_seq(pieces))
        plan, flattened = _plan_segment(
            axes,
            serial_axis,
            pre_stmts,
            main_stmt,
            post_stmts,
            aliases,
            analyzer,
            lanes,
            single=True,
            base_notes=notes,
        )
    except Unsupported as err:
        # 回落路径在这一步就拒绝（典型：并列多 serial 循环 → 「nested serial loop」）时，
        # 也要把多段路径试过但失败的段级原因 / serial 不干净的具体原因带上（缺陷 B）。
        # ``_plan_segment`` 已经把 ``notes`` 放进 diagnostics，避免重复合并。
        if notes and not (err.diagnostics and err.diagnostics[: len(notes)] == notes):
            raise Unsupported(f"{err} [{'; '.join(notes)}]", diagnostics=[*notes, *(err.diagnostics or [])]) from err
        raise
    program = VectorProgram(axes=[] if flattened else _outer_axes(plan.axes, plan.vector_axis), segments=[plan])
    # program 级诊断对单段/多段**恒非空**（对称）：调用方无需区分两条路径，
    # 也不会出现「多段有摘要、单段恒空」的口径裂缝。
    detail = f"flattened axis '{plan.vector_axis.name}'" if flattened else f"vector axis '{plan.vector_axis.name}'"
    program.diagnostics = [f"single-segment program: 1 segment, {detail}"]
    return program


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
    analyzer: arith.Analyzer,
) -> VectorPlan:
    # 方案 §6 第二步：``pre``/``post`` 是 serial 循环的兄弟语句，只有两种合法向量化：
    #   * 沿该 serial 轴（现有路径）：pre/post 在 chunk 循环外、每个外层迭代一次；
    #   * 沿**外层并行轴**（本次新增）：pre/post 变成与 serial 循环同级的一次向量写，
    #     发射结构为 ``for chunk: pre; for serial: main; post``。
    # 后者只在 serial 链恰好一层时成立（多层 serial 的 pre/post 归属会错位）。
    along_serial = serial_axis is not None and vector_axis.var.same_as(serial_axis.var)
    if pre_specs or post_specs:
        serial_count = sum(1 for axis in axes if not axis.is_parallel)
        if not along_serial and not (vector_axis.is_parallel and serial_count == 1):
            raise Unsupported(
                "a T.Parallel body with sibling statements around a serial loop can only be vectorized "
                "along that serial axis or a single outer parallel axis (方案 §6 第二步)"
            )

    extent = _const_int(vector_axis.extent, analyzer)
    chunks = None if extent is None else (extent + lanes - 1) // lanes
    tail = None if extent is None else extent % lanes

    ctx = _ValueContext(
        axes=axes,
        vector_axis=vector_axis,
        analyzer=analyzer,
        lanes=lanes,
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
        mask = build_value(spec.mask, ctx) if spec.mask is not None else None
        # 语句级条件分支的臂内访存是**惰性**的：store 谓词同时也是取值表达式的装载守卫。
        # 否则 ``if i >= 32: c[i] = a[i-32]`` 的值会在所有 lane 上急切求值，发成无掩码的
        # ``vld a[-32]``（D16）。守卫进入 ``load_node`` 后仍由
        # ``_access_provably_in_bounds`` 过滤：能证安全的普通访问照常发 ``vld``。
        # ``mask_free_until/from`` 进一步给出守卫恒真的首/尾 chunk 区间：发射层在这两段
        # 块里把带谓词的 ``vgather2`` 退回普通 ``vld``（前缀/后缀两种方向共用这一套写法）。
        mask_free_until, mask_free_from = _guard_free_bounds(spec.mask, ctx) if spec.mask is not None else (0, None)
        if mask is None:
            value_ctx = ctx
        elif ctx.guard is None:
            # 没有外层守卫：本语句的条件就是完整守卫。
            value_ctx = _with_guard(ctx, mask, mask_free_until, mask_free_from)
        else:
            value_ctx = _with_guard(
                ctx,
                _and_predicate(ctx.guard, mask),
                _combine_free_until(ctx.guard_free_until, mask_free_until),
                _combine_free_from(ctx.guard_free_from, mask_free_from),
            )
        value = build_value(spec.value, value_ctx)
        # 负步长与 PERMUTE / PIECEWISE / INDIRECT 共用「每 lane 索引向量」通道（缺陷 D8）。
        needs_index = kind in (AccessClass.PERMUTE, AccessClass.PIECEWISE, AccessClass.INDIRECT) or (
            kind == AccessClass.STRIDED and stride is not None and stride < 0
        )
        index = index_value(info, ctx) if needs_index else None
        return StorePlan(
            spec=spec,
            access=info,
            value=value,
            mask=mask,
            kind=kind,
            index=index,
            mask_free_until=mask_free_until,
            mask_free_from=mask_free_from,
        )

    for spec, info in zip(main_specs, main_infos):
        plan.stores.append(_plan_store(spec, info))
    for spec, info in zip(pre_specs, pre_infos):
        plan.pre_stores.append(_plan_store(spec, info))
    for spec, info in zip(post_specs, post_infos):
        plan.post_stores.append(_plan_store(spec, info))
    # 沿 serial 轴向量化时，pre/post 与 chunk 循环同级、必须是 lane 均匀的语句；沿并行轴
    # 向量化时它们落在 chunk 循环体内，按普通向量 store 发射（可以有连续/步长等形态）。
    if along_serial:
        for where, stores in (("before", plan.pre_stores), ("after", plan.post_stores)):
            for store in stores:
                if store.kind != AccessClass.UNIFORM:
                    raise Unsupported(
                        f"a statement {where} the serial loop must be uniform along vector axis "
                        f"'{vector_axis.name}' (got {store.kind.value}) (方案 §6 第二步)"
                    )
    # 归约判定必须等值 DAG 建好之后（D13 统一判据）：逐 store 剥离旧值，顺带在 build 期
    # 就把不支持的形态拒绝掉，保证 build 成功 ⇒ emit 成功。
    if vector_axis.is_parallel:
        # 跨位自读（stencil）沿并行轴不是逐 lane 独立的读改写：向量化会让所有 lane
        # 读到写之前的旧向量，与标量顺序不同 → 明确拒绝。
        stencil = _cross_address_read([*pre_specs, *main_specs, *post_specs], vector_axis, analyzer)
        if stencil is not None:
            raise Unsupported(stencil)
        # 累加器提升是**计划期**的单一决策（与归约判据 D13 同一原则）：发射层只消费
        # ``StorePlan.promote``，不再自带 analyzer 重算一遍（避免两处口径漂移）。
        for store in plan.stores:
            store.promote = accumulator_candidate(store, plan, analyzer)
        # 并行轴上的 lane 均匀写若是**循环私有标量**（``T.alloc_var`` → ``local.var``），
        # 源程序里每个并行迭代各有一份；沿并行轴向量化会让所有 lane 共享同一个标量。
        # 只有「同址读改写 + 逐 lane 消费」的累加器形态能提升成向量寄存器（S2），
        # 其余仍明确拒绝并回退到 serial 轴。
        for store in plan.stores:
            if (
                store.kind == AccessClass.UNIFORM
                and not store.value.is_uniform
                and is_private_scope(store.spec.buffer)
                and store.promote is None
            ):
                raise Unsupported(
                    f"store to private scalar '{store.spec.buffer.name}' varies along vector axis "
                    f"'{vector_axis.name}': vectorizing the outer axis needs a promotable accumulator "
                    "(a per-iteration private scalar cannot be shared by all lanes) (方案 §2 Case II)"
                )
    plan.reduce = _classify_reduce(plan, analyzer)
    if plan.reduce is not None:
        # 归约路径把 pre/post 交给 ``_emit_outer_store``（chunk 循环外的一次单点写），
        # 它要求 lane 均匀。沿并行轴向量化时上面的门禁并不检查这一点 → 会把
        # 「build OK、emit Unsupported」的裂缝留给调用方（R7），这里补上。
        for where, stores in (("before", plan.pre_stores), ("after", plan.post_stores)):
            for store in stores:
                if store.kind != AccessClass.UNIFORM:
                    raise Unsupported(
                        f"a statement {where} the serial loop must be uniform along the reduction vector axis "
                        f"'{vector_axis.name}' (got {store.kind.value}) (方案 §6 第二步)"
                    )
        _fold_identity_init(plan)
    _drop_dead_pre_stores(plan, analyzer)
    # 缺陷 A0：吸收的 serial 轴上，「部分归约」没有向量形式。lane 均匀的地址配逐 lane 变化的
    # 取值，只有**全段归约**（:func:`_classify_reduce` 成功）才能表达；否则逐元素路径会用
    # ``ONEPT`` 只保留 lane 0，把其余迭代的累加静默丢掉（结果错误且无诊断）。
    if plan.reduce is None:
        for store in plan.stores:
            if store.kind != AccessClass.UNIFORM or store.value.is_uniform:
                continue
            if vector_axis.is_parallel and store.promote is not None:
                continue  # S2：逐 lane 私有的累加器，发射层提升成 local.var 向量寄存器
            if vector_axis.is_parallel:
                raise Unsupported(
                    f"store to '{store.spec.buffer.name}' is lane-uniform along vector axis "
                    f"'{vector_axis.name}' but its value varies with it: a lane-varying value written to a "
                    "lane-uniform address has no vector form (方案 §2 Case II)"
                )
            raise Unsupported(
                f"store to '{store.spec.buffer.name}' is lane-uniform along the absorbed serial axis "
                f"'{vector_axis.name}' but its value varies with it: a partial reduction mixed with "
                "element-wise stores has no vector form (方案 §2 Case II)"
            )
    plan.guard_boundaries = _guard_cuts(plan)
    return plan


# chunk 切点上限：每个守卫最多贡献两个切点。超过上限时放弃分块（保持单一循环），
# 避免为极多条件分支膨胀代码。
_GUARD_CUT_LIMIT = 6


def _guard_cuts(plan: VectorPlan) -> list[int]:
    """守卫恒真区间的 chunk 切点（升序，只保留内部切点）。

    每个守卫给出两段恒真区间 ``[0, free_until)`` 与 ``[free_from, chunks)``；把两个
    端点都作为切点，发射层在相邻切点之间独立发射一段，段内再用
    ``[chunk_min, chunk_max)`` 判定守卫是否已恒真。这样前缀/后缀、多条语句的多种
    边界可以**共用同一套分段写法**，互不干扰。
    """
    if plan.chunks is None or plan.chunks <= 1:
        return []
    chunks = plan.chunks
    cuts: set[int] = set()
    stack: list[ValueNode] = []
    stores = list(plan.stores)
    if plan.vector_axis.is_parallel:
        # 沿并行轴向量化时 pre/post 落在 chunk 循环体内，同样参与守卫边界计算；
        # 沿 serial 轴时它们在循环外发射，chunk 边界对它们没有意义。
        stores.extend(plan.pre_stores)
        stores.extend(plan.post_stores)
    for store in stores:
        if store.mask is not None:
            cuts.add(store.mask_free_until)
            if store.mask_free_from is not None:
                cuts.add(store.mask_free_from)
        for root in (store.value, store.mask, store.index):
            if root is not None:
                stack.append(root)
    while stack:
        node = stack.pop()
        if node.predicate is not None:
            cuts.add(node.predicate_free_until)
            if node.predicate_free_from is not None:
                cuts.add(node.predicate_free_from)
        stack.extend(node.args or ())
    interior = sorted(cut for cut in cuts if 0 < cut < chunks)
    if len(interior) > _GUARD_CUT_LIMIT:
        return []
    return interior


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
    if plan.vector_axis.is_parallel:
        # 私有标量（``T.alloc_var`` / ``local``）每个并行迭代各一份：沿并行轴的
        # 「lane 均匀写」不是跨 lane 归约，而是逐 lane 独立的累加器（发射层提升成
        # ``local.var`` 寄存器，见 :func:`~.model.accumulator_candidate`）。
        # 沿 serial 轴时相反——同一个私有标量被多个迭代累加，是真归约。
        stores = [store for store in stores if not is_private_scope(store.spec.buffer)]
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


def _is_same_const(expr, value) -> bool:
    """两个常量表达式是否数值相等（``0.0`` / ``0`` / ``4286578688`` 之类）。"""
    if isinstance(expr, (tirx.IntImm, tirx.FloatImm)) and isinstance(value, (tirx.IntImm, tirx.FloatImm)):
        return float(expr.value) == float(value.value)
    return False


def _fold_identity_init(plan: VectorPlan) -> None:
    """``pre`` 把目标位置写成归约单位元时，折叠成「累加器初值」并删掉这条 pre 写。

    典型：S2 的 ``acc = 0; for j: acc += a[i,j]`` —— 初值就是 Add 的单位元，
    ``_emit_reduce_tail`` 里的「并入旧值」是多余的，pre 的 ``ONEPT``/标量写也白做。
    """
    if not plan.pre_stores:
        return
    kept: list[StorePlan] = []
    for pre in plan.pre_stores:
        folded = False
        if pre.kind == AccessClass.UNIFORM and pre.mask is None:
            for store in plan.stores:
                reduce = store.reduce
                if reduce is None or not reduce.accumulate:
                    continue
                if not (
                    _same_buffer(pre.spec.buffer, store.spec.buffer) and _same_indices(tuple(pre.spec.indices), tuple(store.spec.indices))
                ):
                    continue
                identity = _reduce_identity(reduce.recipe, _elem_dtype(store.spec.buffer.dtype))
                if _is_same_const(pre.spec.value, identity):
                    reduce.accumulate = False
                    folded = True
                    break
        if not folded:
            kept.append(pre)
    plan.pre_stores = kept


def _drop_dead_pre_stores(plan: VectorPlan, analyzer: arith.Analyzer) -> None:
    """删除被主体**必然覆写**的 pre 写。

    典型 M1：``c[i,0] = 1.0; for j: c[i,j] = a[i,j]*2`` —— serial 循环从 0 开始，
    ``j=0`` 那一轮写的就是 ``c[i,0]``，pre 写是死写（保留也不影响语义，但白做一次写）。
    判据：同 buffer、主体写无谓词、主体地址在 serial 轴取 0 时与 pre 地址完全相等。
    """
    if not plan.pre_stores:
        return
    serial_vars = [axis.var for axis in plan.axes if not axis.is_parallel]
    if not serial_vars:
        return
    # 零迭代的 serial 循环不会执行主体：此时 pre 写仍然有效，不能当死写。
    if any(_const_int(axis.extent, analyzer) == 0 for axis in plan.axes if not axis.is_parallel):
        return
    kept: list[StorePlan] = []
    for pre in plan.pre_stores:
        dead = False
        for store in plan.stores:
            if store.mask is not None or not _same_buffer(pre.spec.buffer, store.spec.buffer):
                continue
            # 读改写（``s[i] += x``）依赖 pre 写下的初值，不能当死写删掉。
            if _reads_buffer(store.value, store.spec.buffer):
                continue
            offset = store.access.offset
            for var in serial_vars:
                offset = tirx.stmt_functor.substitute(offset, {var: _int(0)})
            diff = analyzer.simplify(offset - pre.access.offset)
            if _const_int(diff, analyzer) == 0:
                dead = True
                break
        if not dead:
            kept.append(pre)
    plan.pre_stores = kept


def _plan_reduce(store: StorePlan, vector_axis: Axis) -> ReduceInfo:
    """把一条 lane 均匀的 store 拆成 :class:`ReduceInfo`（不支持的形态在 **build 期** 拒绝）。"""
    buffer = store.spec.buffer
    if store.mask is not None:
        # 条件归约（``if cond: s[0] += a[i]``）目前没有向量形式：``vcadd/vcmax/vcmin``
        # 的掩码只承载尾块谓词，用户谓词会在发射期被**静默丢弃**（对全部 lane 求和）。
        # 在 build 期显式拒绝，保证「build 成功 ⇒ 语义正确」；未来可用
        # ``vsel(varying, identity, cond)`` + 合并掩码实现（见设计文档）。
        raise Unsupported(
            f"conditional reduction into '{buffer.name}' is not supported yet: the cross-lane reduction mask only "
            "carries the tail predicate, so the guard would be silently dropped (方案 §2 Case II / R3)"
        )
    if not _reads_buffer(store.value, buffer):
        raise Unsupported(
            f"store to '{buffer.name}' is lane-uniform along axis '{vector_axis.name}' but its value does not read "
            f"'{buffer.name}': only the accumulate form 'dst[...] = dst[...] <op> x' (op ∈ Add/Max/Min) is a "
            "reduction (方案 §2 Case II); a lane-varying value written to a lane-uniform address is a "
            "last-write-wins overwrite and has no vector form"
        )
    op = _reduce_op([store.spec])  # Add/Max/Min 之外显式拒绝（缺陷 D1）
    varying, accumulated = _strip_reduce(store.value, op, store.spec)
    if not accumulated or varying is None:
        raise Unsupported(
            f"the old value of '{buffer.name}' must be a direct operand of the '{op}' reduction "
            "(方案 §2 Case II); it cannot be scaled or nested inside another operation"
        )
    return ReduceInfo(axis=vector_axis, op=op, recipe=_REDUCE_RECIPE[op], varying=varying, accumulate=True)


def _strip_reduce(value: ValueNode, op: str, site: StoreSpec) -> tuple[ValueNode | None, int]:
    """在归约算子**脊**上剥离目标 buffer 的 lane 均匀旧值读取。

    返回 ``(varying, accumulated)``：``varying`` 是逐 lane 取值（lane 均匀加数**保留在内**，
    因为跨 lane 归约本身就会正确处理它们）；``accumulated`` 是旧值被剥离的**次数**
    （正常累加写恰好 1 次；出现多次说明单个初值表达不了，调用方拒绝）。剥离等价于
    沿脊消去该算子的单位元（Add→0 / Max→-inf / Min→+inf），故与 ``dst op= x`` 同义。

    旧值必须与写入地址**同址**（``site.indices``）：``s[i] = s[i] + a[i,j]`` 的 ``s[i]``
    才是可剥离的旧值；``s[i] = s[i-1] + a[i,j]`` 这类 stencil 跨位读不是累加，留在
    ``varying`` 内并触发末尾的显式拒绝。只有脊上（``op`` 及其标量同构）的直接子树的旧值
    才能安全剥离；旧值若被缩放了或藏在别的算子/``uniform_expr`` 里，``varying`` 仍会读到
    dst → 明确拒绝（不静默错发）。
    """
    buffer = site.buffer
    spine = {op, _REDUCE_SCALAR_RECIPE[op]}
    accumulated = 0  # 旧值被剥离的**次数**：必须恰好 1 次才能折叠成一个初值
    repeated: list[str] = []

    def strip(node: ValueNode) -> ValueNode | None:
        nonlocal accumulated
        if (
            node.recipe == "uniform_load"
            and node.access is not None
            and _same_buffer(node.access.buffer, buffer)
            and _same_indices(tuple(node.access.indices), tuple(site.indices))
        ):
            accumulated += 1
            if accumulated > 1:
                # ``s[i] = s[i] + (s[i] + a[i,j])``：旧值出现两次，任何「一个初值 + 一次
                # 跨 lane 归约」的发射都表达不了（会丢掉一份旧值）→ 拒绝（R4）。
                repeated.append("old value appears more than once")
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
    if repeated:
        raise Unsupported(
            f"the old value of '{buffer.name}' appears more than once inside the reduction expression "
            "(方案 §2 Case II): only a single same-address old-value operand can be folded into the "
            "reduction's initial value"
        )
    if varying is not None and _reads_buffer(varying, buffer):
        raise Unsupported(
            f"the reduction body reads the destination buffer '{buffer.name}' outside the accumulate position "
            "(方案 §2 Case II): the old value must be a direct operand of the reduction operator"
        )
    return varying, accumulated


def _spec_cross_address_read(spec: StoreSpec, sites: list[tuple], vector_axis: Axis, analyzer: arith.Analyzer) -> str | None:
    """单个写语句的取值里，是否有「本段写过、但地址不同」的读沿向量轴变化。"""
    bad: list[str] = []

    def visit(node):
        if not isinstance(node, tirx.BufferLoad):
            return
        if not any(_same_buffer(node.buffer, buf) for buf, _ in sites):
            return
        if any(_same_buffer(node.buffer, buf) and _same_indices(tuple(node.indices), idx) for buf, idx in sites):
            return
        info = classify_access(node.buffer, tuple(node.indices), [vector_axis], analyzer, is_store=False)
        if info.axis_class(vector_axis)[0] != AccessClass.UNIFORM:
            bad.append(node.buffer.name)

    tirx.stmt_functor.post_order_visit(spec.value, visit)
    return bad[0] if bad else None


def _cross_address_read(specs: list[StoreSpec], vector_axis: Axis, analyzer: arith.Analyzer) -> str | None:
    """并行向量轴上，本段写过的 buffer 是否在**不同地址**被读（stencil）。

    ``s[i] = s[i-1] + a[i,j]`` 的 ``s[i-1]`` 与写入地址不同：标量顺序下后者会看到
    前一次迭代写过的值，而向量化后所有 lane 读的是写之前的同一条向量，结果不同。
    同址回读（``s[i] += a[i,j]``）是逐 lane 独立的读改写，不受影响。
    """
    sites = [(spec.buffer, tuple(spec.indices)) for spec in specs]
    for spec in specs:
        name = _spec_cross_address_read(spec, sites, vector_axis, analyzer)
        if name is not None:
            return (
                f"read of '{name}' at a different address than any write in the same nest is a stencil along "
                f"vector axis '{vector_axis.name}': vectorizing would read a different element than the scalar order"
            )
    return None


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


def _require_uniform_element_width(specs) -> None:
    """一段共享一个 lane 数，因此所有写必须是**同一元素位宽**。

    段级 ``lanes`` 由首个 store 的元素位宽派生，chunk 地址按 ``chunk * lanes`` 步进；
    若同段里混入不同位宽的 store（如 f16 + int32），后者每个 chunk 只能覆盖
    ``2048/位宽`` 个元素，会静默漏写一半或越界（P0-2）。这类写法应拆成两段，
    或在 nest 之外显式做类型转换。
    """
    widths = {_mask_bits(_elem_dtype(spec.buffer.dtype)): spec.buffer for spec in specs}
    if len(widths) <= 1:
        return
    detail = ", ".join(f"{buffer.name}:{_elem_dtype(buffer.dtype)} ({bits}bit)" for bits, buffer in sorted(widths.items()))
    raise Unsupported(
        f"mixed element widths in one vector segment ({detail}): every store of a segment must have the "
        "same element bit width because the chunk stride is shared; split the nest or stage an explicit cast"
    )
