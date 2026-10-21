"""Unified polyhedral (affine) analysis for ``T.Parallel`` vectorization.

This package replaces the "one branch per code shape" style of recognition used
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

  推导见 :func:`finite_difference` 与 :func:`classify_delta` 的注释。

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

包内分层（依赖只向下）
----------------------
============================ ==================================================
模块                          职责
============================ ==================================================
:mod:`primitives`             原语事实表（算子 → ``T.simd.*``）、``Unsupported``、
                              语句/表达式小工具
:mod:`access`                 SCoP：``Axis`` / ``AccessClass`` / 有限差分 /
                              ``AccessInfo`` / ``classify_access``
:mod:`body`                   body 规范化：多语句 / 分支 → 有序 ``StoreSpec``
:mod:`value`                  值 DAG：``ValueNode`` / ``build_value`` / ``load_node``
:mod:`model`                  计划数据结构：``StorePlan`` / ``ReduceInfo`` /
                              ``VectorPlan`` / ``VectorProgram``
:mod:`plan`                   规划：轴排序、归约判定、``build_plan``
:mod:`emit`                   发射：``_Emitter`` / ``emit_plan``（``T.simd.*``）
:mod:`render`                 可读计划渲染：``render_plan``
:mod:`driver`                 语句树变换：``vectorize_parallel_nests``
============================ ==================================================

包只做分析与规划（对应分层方案里的 P1 + P2 的分析部分）；:func:`emit_plan`
把计划发射成 **``T.simd.*``**（``ascend/language/simd.py``，CCE 向量原语的前端
封装），而不是任何自定义 ABI。
"""

from __future__ import annotations

from .access import AccessClass, AccessInfo, Axis, classify_delta, finite_difference, linear_element_offset
from .body import StoreSpec
from .driver import vectorize_parallel_nests
from .emit import emit_plan
from .model import ReduceInfo, StorePlan, VectorPlan, VectorProgram
from .plan import build_plan
from .primitives import Unsupported
from .render import render_plan
from .value import ValueNode

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
