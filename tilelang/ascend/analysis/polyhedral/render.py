"""人类可读的计划渲染（诊断 / 测试断言用）。

``build_plan`` 恒返回 :class:`VectorProgram`，但**1 段 program 与「单段 plan」逐字同形**
（级别 1 只统一表示层，不改人类可读输出）：不打印 ``program:`` 头与段标记，program 级
摘要并入同一段 ``diagnostics:``。只有多段 program 才打印 ``program:`` 头与逐段标记。
"""

from __future__ import annotations

from collections.abc import Sequence

from .model import VectorPlan, VectorProgram
from .value import ValueNode


def render_plan(plan: VectorPlan | VectorProgram) -> str:
    """渲染计划文本（单段 / 多段两种形态，见模块 docstring）。"""
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


def _render_single_plan(plan: VectorPlan, extra_diagnostics: Sequence[str] = ()) -> str:
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
