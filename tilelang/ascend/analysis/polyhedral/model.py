"""指令选择无关的计划数据结构。

本模块只定义分析与发射之间的**共享类型**：``StorePlan`` / ``ReduceInfo`` /
``VectorPlan`` / ``VectorProgram``。它刻意不依赖 :mod:`plan`（规划算法），
这样发射层（:mod:`emit`）与渲染层（:mod:`render`）只需要计划类型，而不必依赖
整个规划模块——依赖方向保持严格向下::

    primitives → access → {body, value} → model → plan → {emit, render} → driver

注意：计划里**没有任何 ``T.simd.*`` 指令名**（指令选择推迟到发射层），但
``lanes`` / ``chunks`` / 掩码粒度仍由目标向量寄存器宽度派生，因此准确说法是
「指令选择无关」，而不是「完全硬件无关」。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tvm import arith

from .access import AccessClass, AccessInfo, Axis, _const_int, finite_difference
from .body import StoreSpec
from .primitives import _same_buffer, _same_indices
from .value import ValueNode, _reads_buffer, reads_other_address, same_address_load


@dataclass
class StorePlan:
    """一条写的完整计划：写什么、写在哪、用什么谓词/索引向量、是否归约。

    ``promote`` 是**计划期**给出的累加器提升决策：非空表示这条同址读改写可以在发射期
    提升成 ``local.var`` 向量寄存器，值是它读取的「同址旧值」节点。发射层只消费这个
    字段，不再自己调用 :func:`accumulator_candidate` 重算——判据只有一处（与归约判据
    D13 同一原则），且不需要在发射期再造一个 ``arith.Analyzer``。
    """

    spec: StoreSpec
    access: AccessInfo
    value: ValueNode
    mask: ValueNode | None = None
    kind: AccessClass = AccessClass.CONTIGUOUS
    index: ValueNode | None = None  # 分片/置换/间接写的每 lane 索引向量（scatter）
    reduce: ReduceInfo | None = None  # 归约段里**逐 store** 的归约描述（方案 §2 Case II）
    promote: ValueNode | None = None  # 累加器提升（同址旧值）；None = 不提升
    # 用户谓词恒真的首/尾 chunk 区间下界：发射层在这两段块里用全 lane 掩码，省掉 `vcmps`。
    mask_free_until: int = 0
    mask_free_from: int | None = None


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


@dataclass
class VectorPlan:
    """**段级**指令选择无关的执行计划（≈ 方案里的中间层 2 的分析部分）。

    一个「段」= 一个主体 + 可选 ``pre``/``post``（方案 §6 第二步）。``build_plan``
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
    # 守卫恒真区间的 chunk 切点（升序，见 ``plan._guard_cuts``）：发射层在相邻切点之间
    # 分段发射，每段用 ``[chunk_min, chunk_max)`` 判定哪些守卫已恒真、可退回普通 ``vld``。
    # 空列表 = 不切分。
    guard_boundaries: list[int] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class VectorProgram:
    """``build_plan`` 的**统一返回类型**（恒返回，1..N 段）。

    单段（``T.Parallel`` 体只有一个主体，或一个主体 + 可选 ``pre``/``post``）时是
    「1 段 program」：``axes`` 只含提到最外层包一次的**并行轴前缀**（``plan._outer_axes``），
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


# ---------------------------------------------------------------------------
# 累加器判据 —— plan（**允许**私有标量）与 emit（**提升**成寄存器）的唯一来源
# ---------------------------------------------------------------------------


def is_private_scope(buffer) -> bool:
    """buffer 是否是**每次迭代私有**的标量/局部存储（``local`` / ``local.var``）。

    这类存储（``T.alloc_var`` / ``T.alloc_local``）在源程序里每个并行迭代各有一份；
    沿并行轴向量化会让所有 lane 共享同一份，必须先提升成向量寄存器。
    """
    scope = buffer.scope()
    return scope == "local" or scope.startswith("local.")


def accumulator_candidate(store: StorePlan, plan: VectorPlan, analyzer: arith.Analyzer) -> ValueNode | None:
    """该写能否提升成 ``local.var`` 向量寄存器累加器；能则返回同址旧值节点。

    这是「读改写沿内层轴不变」的唯一判据，plan 与 emit 共用：

    * plan 用它**允许**私有标量（``T.alloc_var``）沿并行轴向量化（S2）；
    * emit 用它把共享/私有 RMW 从每轮 ``vld``+``vsts`` 提升成寄存器更新。

    条件：无谓词；目标连续（共享）或私有标量；地址沿向量轴之后的所有 inner 轴不变；
    取值同址回读（RMW）；该 buffer 在本段只有这一条主写；同层其它写不读它的其它地址；
    同址 pre 写不能带谓词（带谓词就无法折叠成寄存器初值）。

    私有标量还要求循环后有**逐 lane** 消费者：没有消费者时末值可见性由"哪条 lane
    最后写内存"决定，与标量逐迭代语义不同。
    """
    if store.mask is not None:
        return None
    buffer = store.spec.buffer
    private = is_private_scope(buffer)
    if store.kind == AccessClass.UNIFORM:
        if not private:
            return None
    elif store.kind != AccessClass.CONTIGUOUS:
        return None
    inner = [axis for axis in plan.axes if axis.index > plan.vector_axis.index]
    if not inner:
        return None
    if any(_const_int(finite_difference(store.access.offset, axis.var, analyzer), analyzer) != 0 for axis in inner):
        return None
    old = same_address_load(store.value, buffer, store.spec.indices)
    if old is None:
        return None
    if sum(1 for other in plan.stores if _same_buffer(other.spec.buffer, buffer)) != 1:
        return None
    if any(other is not store and reads_other_address(other.value, buffer, store.spec.indices) for other in plan.stores):
        return None
    if any(
        pre.mask is not None and _same_buffer(pre.spec.buffer, buffer) and _same_indices(tuple(pre.spec.indices), tuple(store.spec.indices))
        for pre in plan.pre_stores
    ):
        return None
    if not private:
        return old
    consumers = [other for other in (*plan.stores, *plan.post_stores) if other is not store and _reads_buffer(other.value, buffer)]
    if not consumers or any(other.kind == AccessClass.UNIFORM for other in consumers):
        return None
    if any(
        post.kind == AccessClass.UNIFORM and not post.value.is_uniform
        for post in plan.post_stores
        if _same_buffer(post.spec.buffer, buffer)
    ):
        return None
    return old
