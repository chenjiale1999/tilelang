"""body 规范化：多语句 / 分支 → 有序的写列表。

与算子无关的规范化规则：

* ``SeqStmt`` → 顺序展开（§3 多行语句天然支持）；
* ``BufferStore`` → 一条写；
* ``IfThenElse`` 单臂 → 谓词化写（``mask &= cond``）；
* ``IfThenElse`` 双臂且两臂写同一位置 → 合并成 ``select(cond, a, b)``；
* 其它（**非直接内嵌 / 不干净**的内层 serial loop、原子调用、跨迭代标量状态）
  → 明确拒绝（干净的 serial 循环已在规划层被吸收成轴）。
"""

from __future__ import annotations

from dataclasses import dataclass

from tvm import tirx

from .primitives import Unsupported, _call_tail, _for_kind_name, _is_atomic_name, _same_indices


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


def _is_zero(expr) -> bool:
    """字面量零判定（语句规范化不引入符号分析依赖）。"""
    if isinstance(expr, tirx.IntImm):
        return int(expr.value) == 0
    return isinstance(expr, int) and not isinstance(expr, bool) and expr == 0


def _is_truthy_else(case) -> bool:
    return case is None or (isinstance(case, tirx.Evaluate) and _is_zero(case.value))


def normalize_body(
    stmt: tirx.Stmt,
    aliases: dict,
    out: list | None = None,
    mask: tirx.PrimExpr | None = None,
) -> list[StoreSpec]:
    """把 body 规范化成有序 :class:`StoreSpec` 列表。

    规则见模块 docstring；不支持的形态一律 ``Unsupported``，不做静默降级。
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
                        raise Unsupported(
                            "masked branch merging is not supported: one arm of this same-address if/else is itself "
                            "predicated (e.g. a nested else-if), so there is no single select condition; "
                            "rewrite it as two consecutive masked stores"
                        )
                result.append(StoreSpec(then_spec.buffer, then_spec.indices, merged, mask))
                return result

        raise Unsupported("divergent branch arms: an if/else whose arms write different locations cannot be vectorized (方案 §8.1)")

    if isinstance(stmt, tirx.For):
        raise Unsupported(
            f"nested {_for_kind_name(stmt.kind)} loop inside T.Parallel is not affine-vectorizable "
            "(方案 §6): only directly-nested, dependence-free serial loops are absorbed as axes"
        )

    if type(stmt).__name__ == "Bind":
        return result

    raise Unsupported(f"unsupported statement in T.Parallel body: {type(stmt).__name__}")
