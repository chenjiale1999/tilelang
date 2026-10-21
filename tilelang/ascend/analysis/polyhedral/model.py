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

from .access import AccessClass, AccessInfo, Axis
from .body import StoreSpec
from .value import ValueNode


@dataclass
class StorePlan:
    """一条写的完整计划：写什么、写在哪、用什么谓词/索引向量、是否归约。"""

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
