"""语句树驱动：把整棵 IR 里的 ``T.Parallel`` 嵌套替换成向量化发射结果。"""

from __future__ import annotations

from tvm import tirx

from .emit import emit_plan
from .plan import build_plan
from .primitives import Unsupported


def vectorize_parallel_nests(stmt: tirx.Stmt, *, lanes: int | None = None) -> tirx.Stmt:
    """把 stmt 里每个 ``T.Parallel`` 嵌套整段替换成 :func:`emit_plan` 的结果。

    自顶向下（pre-order）遍历：一旦某个 ``T.Parallel`` 能整体规划，就把**整个
    连续的 Parallel 嵌套**（``plan._split_nest`` 会一路吃到非 Parallel）换成发射
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
                return emit_plan(plan)
            except Unsupported:
                return None
        return None

    return tirx.stmt_functor.ir_transform(stmt, rewrite, None, ["tirx.For"])
