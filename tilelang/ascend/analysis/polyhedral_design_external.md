# Ascend 统一多面体向量化 —— 设计说明

> **文档定位（对外分享版）**
>
> 本文说明一套把 `T.Parallel` 标量循环**统一向量化**的编译分析与发射方案。为了让没有代码仓库访问权限的同事也能完整读懂，本文**自包含**：所有关键算法都附上**伪代码**或**真实实现片段**，无需打开任何源文件。
>
> 对应实现是一个约 1300 行的独立 Python 分析模块（下称 `polyhedral.py`）与一组 36 个用例的离线测试。它目前是一个**离线分析原型**：只被自身的测试调用，尚未接入生产编译管线（接入路径见 §5.6）。

---

## 0. 阅读指南

* 标有 **「伪代码」** 的代码块是算法骨架，用来说明步骤之间的逻辑关系；
* 标有 **「实现」** 的代码块是从真实模块中摘录（或略作删节的）Python 源码，可直接对照理解；
* 「支持 / 部分支持 / 拒绝」表示某个场景当前的处理结果，**拒绝即明确报错并给出诊断**，本项目不存在「静默降级」。

---

## 1. 背景与动机

### 1.1 现状：case-by-case 识别

Ascend 现有的 `T.Parallel` 向量化在编译器侧按「代码形状」逐条分支识别：

* 只接受 `BufferStore` 及其顺序组合，不认 `if` / `select`；
* 对每个算子单独判断维度 / stride；
* 归约、广播、多语句各有专门逻辑。

问题在于：**几何上等价的场景，写法千差万别**。每新增一种写法就要加一个分支，分支之间还会互相干扰。

### 1.2 目标：单一数学判据

把「识别代码形状」替换成「对访存地址沿候选向量轴求**有限差分**」：

| 经典做法（case-by-case） | 本方案（统一数学分析） |
|---|---|
| 逐条匹配节点形状 | 建立 SCoP：迭代域 + 访存仿射函数 |
| 对每个算子判断维度 / stride | 对每个访问算「沿轴的有限差分」Δ |
| 按代码形态分支 | 按 Δ 的取值分类，一条规则覆盖所有形态 |

**核心洞察**：下列几何上完全不同的写法，沿轴求差分后会落到同一个标量集合 `{1, 0, k, 非常数}` 上：

```text
c[i]   = a[i]          →  Δ = 1      （连续）
c[i]   = a[2*i]        →  Δ = 2      （步长 2）
c[i]   = a[0]          →  Δ = 0      （广播）
h[i//16, i%16]         →  Δ = 1      （高维摊平，化简后）
d[i,j] = g[i] - g[j]   →  g[i] 沿 j 的 Δ = 0（维度不一致 → 广播）
```

因此，**一套基于 Δ 的规则就能同时覆盖所有这些形态**，不需要为每种写法写分支。

---

## 2. 总体架构

### 2.1 在编译流水线中的位置

该方案对应一个三层分工中的后两层：

```text
┌───────────────────────────────────────────────────────────────────┐
│ 分层方案                                                           │
│                                                                   │
│  P1  LiftParallelToArray   数组化：把 T.Parallel 降成数组 IR        │
│  P2  AscendArrayExecPlan   规划：连续轴 / 向量化策略                │
│  P3  AscendLowerArrayPlan  发射：T.simd.* 指令                      │
└───────────────────────────────────────────────────────────────────┘

   P1 输出 (数组 IR)
        │
        ▼
  ┌──────────────────────────────┐
  │  build_plan   ─── 分析 = P2  │   T.Parallel 嵌套 ─▶ VectorPlan
  └──────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────┐
  │  emit_plan    ─── 发射 = P3  │   VectorPlan ─▶ T.simd.* TIR
  └──────────────────────────────┘
        │
        ▼
  vectorize_parallel_nests(...)   把整棵语句树里的 Parallel 嵌套整段替换
```

关键点：`VectorPlan`（中间计划）里**没有任何硬件指令名**——只有轴、lane 数、chunk、每个 store 的值 DAG。**指令选择推迟到 `emit_plan`**。这条「硬件无关中间表示」的设计，使得更换发射目标（例如从某套自定义内建改为 `T.simd.*`）只需改发射层，分析层完全不动。

### 2.2 模块分层

```text
polyhedral.py
├─ 0. 原语事实表        _ELEMENTWISE_BINARY/_UNARY, _VEC_SCALAR_BINARY, _COMPARE,
│                       _BOOL_BINARY/_UNARY  —— 唯一保留的 per-op 事实（原语级）
├─ 1. SCoP：迭代域       Axis, AccessClass
├─ 2. 仿射原语           linear_element_offset, finite_difference, classify_delta
├─ 3. 访问分类           AccessInfo, classify_access
├─ 4. body 规范化        StoreSpec, normalize_body, _collect_aliases/_apply_aliases
├─ 5. 值 DAG             ValueNode, build_value, broadcast, load_node, _build_call
├─ 6. 规划               StorePlan, ReduceInfo, VectorPlan, select_vector_axis, build_plan
├─ 7. 发射               _Emitter, emit_plan, _emit_reduce, render_plan
└─ 8. 语句树变换         vectorize_parallel_nests
```

### 2.3 数据结构总览

| 结构 | 含义 | 关键字段 | 产出者 |
|---|---|---|---|
| `Axis` | 一根 `T.Parallel` 轴 `0<=var<extent` | `var, extent, index` | `_split_nest` |
| `AccessClass` | 沿某轴的访存形态（枚举） | `CONTIGUOUS/UNIFORM/STRIDED/NON_AFFINE/INDIRECT` | `classify_delta` |
| `AccessInfo` | 一次访存 + 逐轴分类 | `buffer, indices, offset, is_store, per_axis, indirect, in_global` | `classify_access` |
| `StoreSpec` | 一条写 | `buffer, indices, value, mask` | `normalize_body` |
| `ValueNode` | 值 DAG 节点 | `recipe, dtype, args, access, scalar, detail` | `build_value` |
| `StorePlan` | 一条写的完整计划 | `spec, access, value, mask, kind` | `build_plan` |
| `ReduceInfo` | 归约信息 | `axis, op, recipe` | `select_vector_axis` |
| `VectorPlan` | 最终硬件无关计划 | `axes, vector_axis, lanes, chunks, tail, stores, reduce` | `build_plan` |

其中 `AccessClass` 由 Δ 唯一决定：

| `Δ_v(off)` | `AccessClass` | 后续动作 |
|---|---|---|
| `1` | `CONTIGUOUS` | `vld` / `vsts`（NORM） |
| `0` | `UNIFORM` | 标量读一次 + `vdup` 广播 |
| `k != 0,1` | `STRIDED` | `vgather2` + 步长索引斜坡 |
| 非常数 | `NON_AFFINE` | 明确拒绝 |
| 下标含 `BufferLoad` | `INDIRECT` | 明确拒绝（gather） |

### 2.4 主数据流

**伪代码**

```text
build_plan(stmt)
  ├─ _split_nest    ▶ axes[], body                    # SCoP 迭代域
  ├─ _collect_aliases(body)          ▶ {var: expr}    # 标量中间量
  ├─ normalize_body(body, aliases)   ▶ StoreSpec[]    # 控制流 → 线性写列表
  ├─ select_vector_axis(axes, stores)▶ (vector_axis, store_infos, reduce)
  ├─ 构造 _ValueContext(axes, vector_axis, lanes, analyzer, vector_dtype)
  └─ for each store: build_value(spec.value) ▶ ValueNode
  ▶ VectorPlan

emit_plan(plan)
  ├─ reduce is None → _Emitter 逐 store 发射；重建 chunk 循环 + 外层 serial 循环
  └─ reduce != None → _emit_reduce：编译期展开 chunk，SSA 串累加器
  ▶ TIR (SeqStmt / For)
```

---

## 3. 算法设计（含伪代码与实现）

### 3.1 SCoP 建模

**(a) 迭代域** —— `_split_nest` 从某个 `T.Parallel` For 出发，**连续吃掉**所有 `T.Parallel`，一旦遇到非 Parallel 就停：

**实现**

```python
def _split_nest(stmt: tirx.Stmt) -> tuple[list[Axis], tirx.Stmt]:
    axes: list[Axis] = []
    current = stmt
    while isinstance(current, tirx.For) and current.kind == tirx.ForKind.PARALLEL:
        axes.append(Axis(current.loop_var, current.extent, len(axes)))
        current = current.body
    if not axes:
        raise Unsupported("no T.Parallel axis found")
    return axes, current
```

> **推论**：**必须从嵌套头调用**才能拿到全部轴。这正是语句树变换采用「自顶向下遍历」、测试 helper 取「最外层 `T.Parallel`」的原因（见 §3.9）。

**(b) 访存函数** —— `linear_element_offset` 把多元下标线性化成元素偏移：

**实现**

```python
def linear_element_offset(buffer, indices, analyzer: arith.Analyzer) -> tirx.PrimExpr:
    offsets = _buffer_strides(buffer)          # 行主序紧凑 stride，或 buffer.strides
    if len(offsets) != len(indices):
        raise Unsupported(f"access rank {len(indices)} does not match buffer rank {len(offsets)}")
    expr = buffer.elem_offset
    for index, stride in zip(indices, offsets):
        term = index * stride
        expr = term if _is_zero(expr) else expr + term
    return analyzer.simplify(expr)             # 立刻化简，交给符号化简器
```

两个要点：

* **行主序紧凑 stride** 藏在 `_compact_strides` 里；`buffer.strides` 非空时优先，以兼容下游 padding / alias。
* **立刻 `simplify`**。`h[i//16, i%16]` 线性化得 `(i//16)*16 + (i%16)`，用恒等式 `x//m*m + x%m == x` 折成 `i`。**高维摊平不需要专门规则，靠的就是符号化简器。**

### 3.2 核心数学：有限差分判据

**实现**

```python
def finite_difference(offset, var, analyzer: arith.Analyzer) -> tirx.PrimExpr:
    """沿轴 var 的有限差分 Δ(off) = off(var+1) - off(var)（符号化简后）。"""
    shifted = tirx.stmt_functor.substitute(offset, {var: var + 1})  # off(var+1)
    return analyzer.simplify(shifted - offset)                      # 再化简一次

def classify_delta(delta) -> tuple[AccessClass, int | None]:
    """把差分值映射成 AccessClass（唯一的分类规则）。"""
    value = _const_int(delta)          # 能否求成常数？
    if value is None: return AccessClass.NON_AFFINE, None   # 折不动 → 非仿射
    if value == 0:    return AccessClass.UNIFORM, 0
    if value == 1:    return AccessClass.CONTIGUOUS, 1
    return AccessClass.STRIDED, value
```

**差分之后必须再化简一次**：`off` 已化简，但 `off(v+1) - off(v)` 会重新产生 `(v+1)//m*m + (v+1)%m - v` 这类式子，只有再化简才能压回常数。

> **设计精髓**：`_const_int(delta) is None` 把「非仿射」的判定**外包给符号化简器的能力边界**——化简器能折成常数就是仿射，折不动就是非仿射。这是「把语义问题转成符号计算问题」的典型手法，既优雅也有风险：**化简器的行为就是接口契约**（因此 `finite_difference` 前后各 `simplify` 一次）。

**INDIRECT 是单独的快速分支**：`classify_access` 先检查下标里有没有 `BufferLoad`，有则每根轴都标 `INDIRECT`。因为 `A[B[i]]` 的地址**依赖被装载的值**，写不成 `x` 的仿射函数——即使 `B[i]` 沿 `i` 的 `Δ=1` 也不行。这是多面体模型「表达不了 data-dependent 地址」的硬边界。

**实现**（访问分类的入口）

```python
def classify_access(buffer, indices, axes, analyzer, *, is_store) -> AccessInfo:
    scales = tuple(indices)
    indirect = any(_contains_buffer_load(index) for index in scales)
    offset = linear_element_offset(buffer, scales, analyzer)
    info = AccessInfo(buffer=buffer, indices=scales, offset=offset,
                      is_store=is_store, indirect=indirect,
                      in_global=buffer.scope() == "global")
    for axis in axes:
        if indirect:
            info.per_axis[axis.name] = (AccessClass.INDIRECT, None)
            continue
        info.per_axis[axis.name] = classify_delta(
            finite_difference(offset, axis.var, analyzer))
    return info
```

### 3.3 body 规范化

`normalize_body` 把任意语句结构压成**有序 `StoreSpec` 列表**，从而「多语句天然支持」：

| 输入结构 | 处理 |
|---|---|
| `SeqStmt` | 顺序展开 |
| `BufferStore` | 一条写 |
| `IfThenElse` 单臂 | 谓词化：`mask &= cond` |
| `IfThenElse` 双臂、**同址写** | 合并成 `Select(cond, a, b)` |
| `IfThenElse` 双臂、异址写 | `Unsupported`（发散分支） |
| 裸 `BufferLoad` | 丢弃（无副作用） |
| `Evaluate(atomic_*)` | `Unsupported` |
| 内层 `For` | `Unsupported` |
| `Bind`（标量别名） | 丢弃（已内联） |

**实现（节选）**

```python
def normalize_body(stmt, aliases, out=None, mask=None) -> list[StoreSpec]:
    result = [] if out is None else out

    if isinstance(stmt, tirx.SeqStmt):
        for child in stmt:
            normalize_body(child, aliases, result, mask)
        return result

    if isinstance(stmt, tirx.BufferStore):
        value = _apply_aliases(stmt.value, aliases)
        result.append(StoreSpec(stmt.buffer, tuple(stmt.indices), value, mask))
        return result

    if isinstance(stmt, tirx.Evaluate):                       # 原子操作 → 拒绝
        call = stmt.value
        if isinstance(call, tirx.Call) and _is_atomic_name(_call_tail(call)):
            raise Unsupported("atomic op (T.atomic_add) is not supported yet: ...")
        return result

    if isinstance(stmt, tirx.IfThenElse):
        condition = _apply_aliases(stmt.condition, aliases)
        then_stores: list[StoreSpec] = []
        normalize_body(stmt.then_case, aliases, then_stores, None)
        else_stores: list[StoreSpec] = []
        if not _is_truthy_else(stmt.else_case):
            normalize_body(stmt.else_case, aliases, else_stores, None)

        if not else_stores:                                   # 单臂 if → 谓词化
            combined = condition if mask is None else tirx.And(mask, condition)
            for spec in then_stores:
                spec.mask = combined if spec.mask is None else tirx.And(combined, spec.mask)
                result.append(spec)
            return result

        if len(then_stores) == 1 and len(else_stores) == 1:   # 双臂同址 → select
            then_spec, else_spec = then_stores[0], else_stores[0]
            if then_spec.buffer.same_as(else_spec.buffer) and _same_indices(...):
                merged = tirx.Select(condition, then_spec.value, else_spec.value)
                result.append(StoreSpec(then_spec.buffer, then_spec.indices, merged, mask))
                return result

        raise Unsupported("divergent branch arms: ...")       # 异址 → 拒绝

    if isinstance(stmt, tirx.For):                            # 内层循环 → 拒绝
        raise Unsupported(f"nested {_for_kind_name(stmt.kind)} loop inside T.Parallel ...")

    if type(stmt).__name__ == "Bind":
        return result

    raise Unsupported(f"unsupported statement in T.Parallel body: {type(stmt).__name__}")
```

配套**标量别名内联**：`_collect_aliases` 收集 `Bind` 引入的真值绑定（排除 buffer 声明用的 Bind），`_apply_aliases` 迭代替换（**限定 8 次**防别名环），把 `k = k_iter*BK + j` 这类中间量折叠进最终表达式。这是「多行语句」收敛成单表达式再向量化的关键。

### 3.4 向量轴选择

`select_vector_axis` 是**纯打分**，无一处匹配代码形状：

1. **候选过滤**：某轴上只要有 store 是 `NON_AFFINE/INDIRECT`，该轴出局。
2. **打分**：`score = (该轴上 CONTIGUOUS 的访存数) * 10 + axis.index`
   * `*10` 让「连续访存数」成为主序；
   * `+axis.index` 让同分取**更内层**（更内层通常更连续）。
3. **兜底**：全轴无 store 连续时，若某轴所有 store 都 `UNIFORM`，选它（广播写）。

**实现**

```python
def select_vector_axis(axes, stores, analyzer):
    store_infos = [classify_access(s.buffer, s.indices, axes, analyzer, is_store=True)
                   for s in stores]
    best, best_score = None, -1

    for axis in axes:                                   # 候选过滤 + 打分
        if any(info.axis_class(axis)[0] in (AccessClass.NON_AFFINE, AccessClass.INDIRECT)
               for info in store_infos):
            continue
        score = sum(1 for info in store_infos
                    if info.axis_class(axis)[0] == AccessClass.CONTIGUOUS)
        if score == 0:
            continue
        score = score * 10 + axis.index                 # 同分取更内层
        if score > best_score:
            best, best_score = axis, score

    if best is None:                                    # 兜底：全均匀
        for axis in axes:
            if all(info.axis_class(axis)[0] == AccessClass.UNIFORM for info in store_infos):
                best = axis
                break
    if best is None:
        raise Unsupported("no vectorizable axis: every store is non-affine or strided on all axes")

    # 归约检测（见 §3.5），命中时用归约轴覆盖 best
    ...
    return best, store_infos, reduce_info
```

> 对比：C++ 参考实现里「识别归约」是专门逻辑；这里它只是打分表的一个正交维度——**先选轴，再问这个轴是不是归约轴**。

### 3.5 归约识别

条件（同时满足）：

* 该轴所有 store 都 `UNIFORM`（`Δ_store = 0`）；
* 取值沿该轴**变化**（自由变量出现 **或** 被访问 buffer 沿轴 `CONTIGUOUS/STRIDED`）。

命中则构造 `ReduceInfo(axis, op, recipe)`，其中 `op` 由取值最外层算子挑（`Add→vadd`、`Max→vmax`、`Min→vmin`），`recipe ∈ {vcadd, vcmax, vcmin}`，并用归约轴覆盖 `vector_axis`。

**实现**

```python
_REDUCE_RECIPE = {"vadd": "vcadd", "vmax": "vcmax", "vmin": "vcmin"}

# 在 select_vector_axis 内：
for axis in axes:
    if all(info.axis_class(axis)[0] == AccessClass.UNIFORM for info in store_infos) and any(
            _value_varies_along(spec.value, axis, analyzer) for spec in stores):
        op = _reduce_op(stores)
        reduce_info = ReduceInfo(axis=axis, op=op, recipe=_REDUCE_RECIPE[op])
        best = axis
        break
```

### 3.6 值 DAG 构造与 `is_uniform`

`build_value` 是**结构无关的递归**，决策仍只用 Δ：

**伪代码**

```text
BufferLoad  → classify_access → load_node：
    Δ=1            → vload（连续装载）
    Δ=0            → uniform_load（标量读一次 + vdup）
    Δ=k            → vload_strided（发射成 vgather2）
    INDIRECT/非常数 → 拒绝
Var         → 向量轴变量 ? lane_index(vci) : imm（外层轴变量 / capture）
IntImm/FloatImm → imm
Call        → _build_call（比较 / 布尔 / 二元 / 一元 / if_then_else）
Select      → _select_node（vsel）
Cast        → vcvt 或 uniform
其它        → Unsupported
```

**第二核心谓词 `ValueNode.is_uniform`**（是否 lane 内均匀、不需要向量寄存器）：

**实现**

```python
@property
def is_uniform(self) -> bool:
    """整个子表达式是否 lane 内均匀（不需要向量寄存器）。"""
    if self.recipe in ("lane_index", "vbrc"):
        return False
    if self.recipe in _UNIFORM_RECIPES:      # ("imm", "uniform_load", "uniform_expr")
        return True
    if self.recipe == "vload":
        return self.access is not None and self.access._uniform
    return all(arg.is_uniform for arg in self.args)   # 递归传播
```

它驱动：

* **标量-向量特化**：`vadd→vadds`、`vmul→vmuls`、`vmax→vmaxs`、`vmin→vmins`；无专用变体则先 `broadcast(vdup)` 再走向量-向量。
* **比较特化**：一侧均匀 → `vcmps`；`vcmps` 要求 `(vector, scalar)` 顺序，交换操作数时谓词**必须翻转**（`lt↔gt`、`le↔ge`）。
* 是否需要 `vdup`；归约的累加写判定。

> ⚠️ **必须理解的坑**：`all([]) == True`。叶子节点若不在 `_UNIFORM_RECIPES` 里会被**误判为均匀**，所以 `lane_index` / `vbrc` / `vload` 必须显式返回 `False`。

`Select` 的处理有两种条件来源，汇入同一 `vsel`：

| 条件来源 | 判定 | 发射 |
|---|---|---|
| **逐 lane**（随向量轴变化） | `args[0].is_uniform == False` | 直接作 `vsel` 的 mask（谓词寄存器） |
| **循环不变**（沿向量轴均匀） | `args[0].is_uniform == True` | 广播成全 lane 谓词 `update_mask(cond ? lanes : 0)` |

若条件与两臂**全部均匀**，整个 select 折叠成一次标量求值 + `vdup`，不产生 `vsel`。

**实现**

```python
def _select_node(args, ctx) -> ValueNode:
    detail = "uniform-cond select" if args[0].is_uniform else "predicated select"
    return ValueNode("vsel", ctx.vector_dtype, args, detail=detail)
```

### 3.7 发射算法

发射目标全部是 `T.simd.*`（CCE 向量原语的前端封装）。**地址统一用 `BufferLoad(buffer, indices)`**，交给 `T.simd` 自动包成 `access_ptr`；indices 里把向量轴变量替换成该 chunk 的 lane 基址（普通路径 `chunk*lanes`；lane 内均匀访问 `0`）。

指令映射（`_Emitter._build` 的分派表）：

| DAG recipe | `T.simd.*` | 说明 |
|---|---|---|
| `imm` | （标量表达式） | 直接用 |
| `lane_index` | `vci(lane_base, "int32")` | lane 序号（谓词比较用） |
| `vload` | `vld(addr, "NORM")` | 连续装载 |
| `vload_strided` | `vgather2(addr, vmuls(vci(0,u32), k, m), m)` | 步长装载（ramp 当索引） |
| `uniform_load`/`uniform_expr`/`vbrc` | `vdup(scalar, elem, mask, MODE_ZEROING)` | 广播 |
| 数据二元 / 一元 | `vadd/vsub/vmul/... / vneg/vabs/...` | `(a, b, mask, mode)` |
| 标量-向量 | `vadds/vmuls/vmaxs/vmins` | `(vec, scalar, mask, mode)` |
| `vcmp` / `vcmps` | `vcmp(a,b,mask,op)` / `vcmps(a,s,mask,op)` | 谓词 |
| `vsel` | `vsel(t, f, mask)` | mask 由 `predicate()` 给出 |
| `pand`/`por`/`pnot` | `pand/por/pnot` | 谓词运算 |
| `vcvt` | `vcvt(src, target, mask)` | mask bits = `min(src, target)` |

**实现（`_Emitter._build` 的代表性分支）**

```python
def _build(self, node: ValueNode):
    recipe, args = node.recipe, node.args

    if recipe == "vload":
        return self.bind(simd.vld(self.addr(node.access), "NORM"), self.lets)

    if recipe == "vload_strided":          # 无带步长装载：用 vgather2 + ramp 索引
        _, stride = node.access.axis_class(self.plan.vector_axis)
        mask = self.all_mask(_mask_bits(node.dtype))
        index = simd.vmuls(simd.vci(_int(0), "uint32"), _int(stride), mask)
        return self.bind(simd.vgather2(self.addr(node.access), index, mask), self.lets)

    if recipe in ("uniform_load", "uniform_expr", "vbrc"):   # 广播
        operand = node if recipe != "vbrc" else args[0]
        elem = _elem_dtype(operand.dtype)
        return self.bind(simd.vdup(self.scalar(operand), elem,
                                   self.all_mask(_mask_bits(elem)), _MODE), self.lets)

    if recipe == "vcmps":                  # 标量比较，要求 (vector, scalar)
        mask = self.all_mask(_mask_bits(args[0].dtype))
        return self.bind(simd.vcmps(self.value(args[0]), self.scalar(args[1]),
                                    mask, node.detail), self.lets)

    if recipe == "vsel":                   # args = [cond, true, false]
        true_value = self.vector_operand(args[1])
        false_value = self.vector_operand(args[2])
        return self.bind(simd.vsel(true_value, false_value,
                                   self.predicate(args[0], node.dtype)), self.lets)
    # ... 其余 recipe 按 _VEC_SCALAR_BINARY / _ELEMENTWISE_* 表分派
```

**SSA 与 `Bind`**：每个中间值用一条独立语句 `Bind(var, expr)` 表示，追加进 `lets` 列表。不能用表达式式的 `Let`，因为 SSA 中间值是一条条独立语句，只能摊平成 `SeqStmt`。

`_Emitter` 的三个状态容器：

| 容器 | 作用 |
|---|---|
| `prologue` | **chunk 无关**的绑定，如全 lane 掩码 `pset(bits, "PAT_ALL")`（按 width 缓存，只绑一次） |
| `lets` | chunk 体内的 SSA 语句 |
| `memo` | `id(node) → 变量`，同一子表达式去重 |

**掩码三层叠加**：`store 的最终掩码 = 用户谓词 ∧ 尾块掩码`，都为空则全 1。尾块掩码 `update_mask(extent - chunk*lanes, bits)`，整除时为 `None`。

**系统常量**：

* `lanes = 2048 / elem_bits`：float32→64、f16/bf16→128、int8→256；向量寄存器 2048 位。
* 谓词类型一律 `boolx256`（256 位 `vector_bool`）。
* `_mask_bits(dtype) = max(elem_bits, 8)`：CCE 谓词粒度 b8/b16/b32/b64（无 b4）。
* `_MODE = "MODE_ZEROING"`。

### 3.8 归约发射（最特别的部分）

**伪代码**

```text
mask = pset(bits, "PAT_ALL")
acc  = vdup(identity, elem, mask)               # vadd→0, vmax→-inf, vmin→+inf
for ck in range(chunks):                        # 编译期展开，不是运行时 for！
    red = vcadd(vld(src_addr(ck), "NORM"), mask) # 跨 lane 归约，结果只在 lane0
    acc = vadd(acc, red, mask)
if 累加写: old = vld(dst_addr, "BRC_B{bits}"); acc = vadd(acc, old, mask)
vsts(dst_addr, acc, mask, "ONEPT_B{bits}")
```

**为什么必须编译期展开？**「跨迭代的寄存器值」在纯 SSA（`Bind`，无 PHI / 可变变量）里无法表达——不能写 `acc = acc ...` 引用上一轮。于是把 chunk 循环**在 Python 层展开**，用不同名字的 `vadd` 串起来。这与 C++ pass 的定型方案一致。

**实现**

```python
def _emit_reduce(plan: VectorPlan, chunk: tirx.Var) -> _Emitter:
    store = next(s for s in plan.stores if s.kind == AccessClass.UNIFORM)
    if plan.tail:
        raise Unsupported("reduction requires the vector axis extent to be a "
                          "multiple of the vector width")
    emitter = _Emitter(plan, chunk)
    elem = _elem_dtype(store.spec.buffer.dtype)
    bits = _mask_bits(elem)
    mask = emitter.all_mask(bits)
    recipe = plan.reduce.recipe
    element_op = _REDUCE_ELEMENT[recipe]

    acc = emitter.bind(simd.vdup(_reduce_identity(recipe, elem), elem, mask, _MODE),
                       emitter.lets)
    varying = _varying_operand(store.value)

    for ck in range(plan.chunks or 1):          # 编译期展开
        emitter.lane_base = _int(ck * plan.lanes)
        emitter.memo = {}                       # 地址随 chunk 变化，缓存按 chunk 失效
        reduced = emitter.bind(getattr(simd, recipe)(emitter.value(varying), mask, _MODE),
                               emitter.lets)
        acc = emitter.bind(getattr(simd, element_op)(acc, reduced, mask, _MODE),
                           emitter.lets)

    if _is_accumulate(store):                   # dst = dst op x：并入目标旧值
        old = emitter.bind(simd.vld(emitter.addr(store.access, uniform=True),
                                    f"BRC_B{bits}"), emitter.lets)
        acc = emitter.bind(getattr(simd, element_op)(acc, old, mask, _MODE), emitter.lets)

    dst = emitter.addr(store.access, uniform=True)
    emitter.lets.append(tirx.Evaluate(simd.vsts(dst, acc, mask, f"ONEPT_B{bits}")))
    # ... 再逆序包上外层 serial 循环
    return emitter
```

配套细节：

* `_reduce_identity`：`vmax→-FLT_MAX`、`vmin→+FLT_MAX`、`vadd→0`。
* `_is_accumulate`：DAG 里存在读**同一 buffer** 的 `uniform_load` → 判定为 `dst = dst op x`，末尾用 `BRC` 广播读并入旧值。
* 归约路径逐 chunk 改 `lane_base` 并**清空 `memo`**（地址随 chunk 变，缓存必须失效）。
* 归约要求 `tail == 0`（否则 `Unsupported`）。

### 3.9 循环重建与遍历策略

`emit_plan`：先把 chunk 体包成 `for chunk in serial(ceil(E/L))`，再**逆序**把非向量轴包成 serial 循环（跳过向量轴本身）。于是 `T.Parallel(M,K)` → `for i in serial(M): for chunk in serial(K/L): ...`。

`vectorize_parallel_nests`：**自顶向下（pre-order）**遍历语句树，命中 `T.Parallel` 就整体 `emit_plan(build_plan(...))` 替换（`_split_nest` 会一路吃到非 Parallel，故整个 nest 被一次性下降）；整体规划失败才返回 `None` 继续下探子节点。

**实现**

```python
def vectorize_parallel_nests(stmt: tirx.Stmt, *, lanes: int | None = None) -> tirx.Stmt:
    def rewrite(node):
        # 返回非 None 表示「就地替换且不再下探」，返回 None 表示继续递归。
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL:
            try:
                return emit_plan(build_plan(node, lanes=lanes))
            except Unsupported:
                return None
        return None

    return tirx.stmt_functor.ir_transform(stmt, rewrite, None, ["tirx.For"])
```

> 若自底向上替换，最内层会先被换成串行代码，外层 `T.Parallel` 反而再也规划不了（`normalize_body` 撞到已发射的 `For`），导致多维嵌套只下降一半、外层 `T.Parallel` 残留。

---

## 4. 场景支持矩阵

图例：**支持** = 分析 + 发射均支持 · **部分支持** = 依赖具体形状 / 某一步受限 · **拒绝** = 明确报错，不静默降级。

| 编号 | 场景 | 状态 | 机制 / 拒绝原因 |
|---|---|---|---|
| §0-I | Copy / 广播写 | 支持 | Δ 分类为 CONTIGUOUS / UNIFORM |
| §0-II | GM（全局内存）操作数 | 拒绝 | `global-memory operand ... needs an explicit UB staging copy` |
| §1-I-a | 单臂 `if` | 支持 | 谓词化 store（`mask &= cond`） |
| §1-I-b | `if`/`else` 同址写 | 支持 | 合并 `select` → `vsel` |
| §1-II | 三元：**循环不变**条件 | 支持 | 广播全 lane 谓词 `update_mask(cond?lanes:0)` |
| §1-II | 三元：逐 lane 条件 | 支持 | `vcmp` 逐 lane 谓词 |
| §2-I | 操作数维度不一致 | 支持 | Δ=0 一侧广播（`vbrc`/`vdup`） |
| §2-II | 规约（`+=` / `max`） | 支持 | `ReduceInfo` → `vcadd/vcmax/vcmin` |
| §2-II | 规约 + 尾块（`extent % lanes != 0`） | 部分支持 | 分析通过，**发射**要求整除 |
| §3-I | 多行语句（连续） | 支持 | `normalize_body` 产出多条 `StoreSpec` |
| §3-I | 多行语句（末维固定 `[.., k]`） | 部分支持 | 末维固定 → 沿唯一轴 STRIDED → `no vectorizable axis` |
| §3-II | 复杂标量 + `if_then_else`（im2col） | 拒绝 | 首卡点：值算子 `FloorDiv/FloorMod` 未支持；其次非仿射下标 / GM |
| §4-I | 表达式下标 | 支持（UB）/ 拒绝（GM） | 线性化+化简后连续 / GM 需 staging |
| §4-II-a | 高维摊平（一维化） | 支持 | 恒等式化简后 Δ=1 |
| §4-II-a | 步长访问（`a[2*i]`） | 支持 | STRIDED → `vgather2` |
| §4-II-b | 高维→低维（block-tiled 置换） | 拒绝 | Δ 非常数 → `non-affine access` |
| §4-gather | 间接下标 `A[B[i]]` | 拒绝 | data-dependent 地址 → `indirect (gather)` |
| §5 | `T.atomic_add` | 拒绝 | 无「向量+标量」原子配方 |
| §6 | 嵌套 serial / 跨迭代标量状态 | 拒绝 | `nested ... loop ... is not affine-vectorizable` |
| — | 数学核心 Δ 分类 | 支持 | 5 个直接单测 |
| — | 语句树变换（整段下降） | 支持 | 自顶向下替换 |

### 4.1 可复现性

一个场景扫描脚本逐条编码上述场景并打印 `OK/REJECT + 原因`（只做 `build_plan`，不发射）。实测输出（节选）：

```text
§1-II tern(unif)   OK   axes=['i_s2','i_v'] chunks=2 tail=0 vaxis=i_v kinds=['contiguous']
§2-II reduce       OK   axes=['i','j'] chunks=4 tail=0 reduce=vcadd@j
§3-I multi         REJECT no vectorizable axis: every store is non-affine or strided on all axes
§3-II complex(GM)  REJECT unsupported value expression: FloorDiv (m % 16 // 4)
§4-II-b tiled      REJECT non-affine access with respect to axis 'j': Δ(offset)=... is not a constant
§6 nested          REJECT nested serial loop inside T.Parallel is not affine-vectorizable
```

---

## 5. 扩展性分析

### 5.1 新增一个逐元素算子

* 目标指令已在 `simd.py`：
  * 有专用标量-向量变体 → 加进 `_VEC_SCALAR_BINARY`（如 `vsub→vsubs`）；
  * 普通二元/一元 → 加进 `_ELEMENTWISE_BINARY/_UNARY`（键取 TIR 节点名尾段小写）。
  * **发射层无需改动**（`_build` 按 recipe 查表 dispatch）。
* 目标指令是新的 `T.simd.*`：需先在 `simd.py` 注册内建，再在 `_build` 加一个 `recipe` 分支。
* 比较 / 布尔类：改 `_COMPARE` / `_BOOL_BINARY` / `_BOOL_UNARY`（并注意 `_COMPARE_FLIP`）。
* **成本**：通常 1–5 行 + 1 个测试。

### 5.2 新增一个几何场景

绝大多数情况下**不需要改代码**——只要新场景的访存沿某轴线性化后 Δ 是常数，就会自动落进现有分类。此时**只需补一个测试**固化边界（这正是支持矩阵的产出方式）。

只有两种情况需要动代码：

1. **引入新的 `AccessClass`**（例如 stride 为负 / 分段仿射）→ 改 `classify_delta`（分类）+ `select_vector_axis`（候选/打分）+ `load_node`（建节点）+ `_build`（发射）。
2. **引入新的语句形态** → 在 `normalize_body` 加一个分支。

### 5.3 换发射目标 / 新增硬件指令

因为 `VectorPlan` 硬件无关，**换 target 只改发射层**：`emit_plan` / `_Emitter._build` / `_tail_mask` / `_store_mask` 里的指令名。分析层（§3 的规划部分）完全不动。

**成本**：与目标指令集规模成正比，与场景数量无关。

### 5.4 路线图：im2col 场景的三段拆分

im2col 本质是**三个耦合特性**，建议分段推进：

| 子特性 | 内容 | 落点 | 成本 / 风险 |
|---|---|---|---|
| (a) | `FloorDiv`/`FloorMod` 作**值算子** | `build_value` + `_build`；但 `simd.py` **无整型 `vdiv/vmod`**（`vdiv` 是浮点 SFU，`vcvt` 只有 `ROUND_F`），需新增内建或对常量除数做 strength-reduction（`vmull`+`vshr` 模拟） | 中；精度/边界敏感 |
| (b) | 非仿射下标 → **gather** | `load_node` 新增 NON_AFFINE 分支 + 逐 lane 索引向量构造（依赖 (a)）。内建已有 `vgather2/vgatherb` | 中偏大 |
| (c) | 非仿射下标的 **UB staging** | 需**新建一个 pass**：识别非仿射/GM 操作数 → 其前插入 staging copy（UB 分配）→ 改写 nest 内操作数读 staging。会牵动 buffer 分配 / 别名 / 同步 | 大；**接入管线后高风险** |

> (a)(b) 仍在离线原型内，风险低；(c) 一旦接入生产管线风险高——且 im2col 是 **gather**（`vgather2` 成本高），建议先做收益评估。

### 5.5 新增 reduction 算子

`_reduce_op`（算子→`vadd/vmax/vmin`）+ `_REDUCE_RECIPE`（→`vcadd/vcmax/vcmin`）+ `_reduce_identity`（单位元）三处同步扩展；发射端 `_emit_reduce` 无需改（按 recipe 查表）。

### 5.6 与生产管线接线

当前模块是**离线原型**（仅测试引用）。接入路径建议：

1. 在 `--convert-tilelang-to-npu` 之后新增一个 Python pass，对目标 `T.Parallel` 嵌套调用 `vectorize_parallel_nests`；
2. 与 C++ 侧的并行向量化 pass 做**能力对齐或替换**时，先对齐 §4 矩阵的边界；
3. 接线后才需要考虑 pass 排序、与 UB 分配 / 同步 / alias pass 的先后关系。

---

## 6. 边界与已知局限

| 类别 | 局限 | 说明 / 潜在改进 |
|---|---|---|
| 归约 | `tail != 0` 不可发射 | 展开式需要整 chunk；尾块归约待支持 |
| 内存域 | GM 操作数拒绝 | 需 UB staging（§5.4(c)） |
| 值算子 | `FloorDiv`/`FloorMod` 仅作下标 | 作值算子拒绝（§5.4(a)） |
| 地址 | 非仿射 / 间接拒绝 | 明确不静默降级 |
| 语句 | 发散分支、嵌套 For、atomic 拒绝 | 见 §3.3 |
| 代码 | `vload.is_uniform` 分支等价于 `False` | `all([])` 坑的防御性写法，可简化 |
| 代码 | `AccessInfo._uniform` 动态挂属性 | 可提升为 dataclass 字段 |
| 判定 | `_const_int` 只认 `IntImm` | 不认「化简后仍是常量表达式」的多项式，限制更复杂仿射判定 |
| 模型 | 只做连续性判定，无完整依赖分析 / 调度 | 迭代域已建模，可扩展到依赖 / 调度 |

---

## 7. 测试策略

* **离线、与硬件/编译器无关**：每个用例构造 TileLang IR，直接跑 `build_plan` / `emit_plan`，检查 `VectorPlan` 与发射出的 `tl.simd.*` 调用序列。不涉及设备执行。
* **分组**：数学核心 5 项 + §0–§6 场景 + 语句树变换，共 **36** 个用例。
* **辅助工具**：
  * `_first_parallel`：取**最外层** `T.Parallel`（后序遍历先出最内层，故取最后一个）。
  * `_emitted_ops`：抽取有序的 `tl.simd.*` op 名。
  * `_recipes`：把 `ValueNode` DAG 展平成 recipe 名列表。
  * `_for_kinds`：统计变换后的循环种类（验证 `PARALLEL` 是否消除）。
* **运行**：`pytest` + `ruff check` / `ruff format --check`（需绕过 editable-install 的 finder 重定向）。

---

## 8. 术语表

| 术语 | 含义 |
|---|---|
| SCoP | Static Control Part，多面体模型里的「静态控制域」（迭代域 + 访存函数） |
| 有限差分 Δ | `Δ_v(off) = off(v+1) - off(v)`，连续性的唯一判据 |
| `AccessClass` | 沿某轴的访存形态（连续/均匀/步长/非仿射/间接） |
| `recipe` | 值 DAG 节点的原语名（`vload`/`vmuls`/`vsel` ...），发射时查表成 `T.simd.*` |
| `is_uniform` | 子表达式是否 lane 内均匀（决定标量-向量特化与 `vdup`） |
| lane | 向量寄存器中的一个元素槽位 |
| `vreg` | 向量寄存器，2048 位；`lanes = 2048 / elem_bits` |
| 谓词 / mask | `boolx256` 类型，CCE 的 P 寄存器，`bits ∈ {b8,b16,b32,b64}` |
| NORM / ONEPT / BRC | store 距离：普通连续 / 单元素 / 广播 |
