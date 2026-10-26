# Ascend 统一多面体向量化 —— 设计说明

>
> 本文说明一套把 `T.Parallel` 标量循环**统一向量化**的编译分析与发射方案。为了让没有代码仓库访问权限的同事也能完整读懂，本文**自包含**：所有关键算法都附上**伪代码**或**真实实现片段**，无需打开任何源文件。
>
> 对应实现是一个约 4162 行的独立 Python 分析包 `polyhedral/`（10 个模块）与一组 108 个用例的离线测试。它目前是一个**离线分析原型**：只被自身的测试调用，尚未接入生产编译管线（接入路径见 §5.6）。

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
  ┌──────────────────────────────────────┐
  │  build_plan   ─── 分析 = P2           │   T.Parallel 嵌套 ─▶ VectorProgram（1..N 段）
  └──────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────┐
  │  emit_plan    ─── 发射 = P3           │   VectorPlan / VectorProgram ─▶ T.simd.* TIR
  └──────────────────────────────────────┘
        │
        ▼
  vectorize_parallel_nests(...)   把整棵语句树里的 Parallel 嵌套整段替换
```

关键点：`VectorPlan` / `VectorProgram`（中间计划）里**没有任何硬件指令名**——只有轴、lane 数、chunk、每个 store 的值 DAG。**指令选择推迟到 `emit_plan`**。这条「指令选择无关中间表示」的设计，使得更换发射目标（例如从某套自定义内建改为 `T.simd.*`）只需改发射层，分析层完全不动。

`build_plan` **恒返回** `VectorProgram`：单段（一个主体 + 可选 `pre`/`post`）时它是「1 段 program」（`program.axes` 只含提到最外层包一次的**并行轴前缀**，其余轴由该段自己包裹）；`T.Parallel` 体内出现**多个并列的干净 `T.serial` 循环**时含多段（每段一个 `VectorPlan`，见 §3.11）。`VectorPlan` 因此是**段级**中间表示。

### 2.2 模块分层

```text
polyhedral/（包，公共 API 由 __init__.py 再导出）
├─ primitives.py   0. 原语事实表   _ELEMENTWISE_BINARY/_UNARY, _VEC_SCALAR_BINARY, _COMPARE,
│                                  _BOOL_BINARY/_UNARY, _REDUCE_PRIMITIVES（含单位元）,
│                                  _MASK_DTYPE/_MODE...
│                                  —— ISA/算子事实与硬件常量的唯一来源
├─ access.py       1. SCoP：迭代域  Axis, AccessClass
│                  2. 仿射原语      linear_element_offset, finite_difference, classify_delta
│                  3. 访问分类      AccessInfo, classify_access
├─ body.py         4. body 规范化   StoreSpec, normalize_body, _collect_aliases/_apply_aliases
├─ value.py        5. 值 DAG        ValueNode, build_value, broadcast, load_node, _build_call
├─ model.py        6. 计划数据结构  StorePlan, ReduceInfo, VectorPlan, VectorProgram
├─ plan.py         7. 规划          _AXIS_WEIGHT, rank_vector_axes, build_plan
├─ emit.py         8. 发射          _Emitter, emit_plan, _emit_reduce
├─ render.py       9. 可读渲染      render_plan
└─ driver.py       10. 语句树变换   vectorize_parallel_nests
```

> 依赖方向：`primitives ← access ← {body, value} ← model ← plan ← {emit, render} ← driver`。
> `model` 只放共享数据结构，`emit` / `render` 因此不依赖规划算法。

### 2.3 数据结构总览

| 结构 | 含义 | 关键字段 | 产出者 |
|---|---|---|---|
| `Axis` | 一根迭代轴 `0<=var<extent`（并行轴或被吸收的 serial 轴） | `var, extent, index, kind` | `_split_nest` / `_absorb_serial_chain` |
| `AccessClass` | 沿某轴的访存形态（枚举） | `CONTIGUOUS/UNIFORM/STRIDED/PERMUTE/PIECEWISE/INDIRECT/TRUE_RANDOM` | `classify_delta` + `classify_access` |
| `AccessInfo` | 一次访存 + 逐轴分类 | `buffer, indices, offset, is_store, per_axis, indirect, in_global, reason` | `classify_access` |
| `StoreSpec` | 一条写 | `buffer, indices, value, mask` | `normalize_body` |
| `ValueNode` | 值 DAG 节点 | `recipe, dtype, args, access, scalar, detail` | `build_value` |
| `StorePlan` | 一条写的完整计划 | `spec, access, value, mask, kind, index, reduce, promote` | `_build_plan_for_axis` |
| `ReduceInfo` | 归约信息（逐 store） | `axis, op, recipe, varying, accumulate` | `_classify_reduce` → `_plan_reduce` |
| `VectorPlan` | **段级**指令选择无关计划（段 = 主体 + 可选 `pre`/`post`） | `axes, vector_axis, lanes, chunks, tail, stores, pre_stores, post_stores, reduce, diagnostics` | `_build_plan_for_axis`（每段一个） |
| `VectorProgram` | 统一计划（1..N 段，`build_plan` 的返回类型） | `axes`（提层的并行轴前缀）, `segments`, `diagnostics` | `build_plan` |

其中 `AccessClass` 分两层判定：Δ 层给常数形态，非常数再做**结构细化**（改进 4）：

| `Δ_v(off)` | `AccessClass` | 后续动作 |
|---|---|---|
| `1` | `CONTIGUOUS` | `vld` / `vsts`（NORM） |
| `0` | `UNIFORM` | 标量读一次 + `vdup` 广播 |
| `k,\ \|k\|>1` | `STRIDED` | `vgather2` / `vscatter` + 步长索引斜坡 |
| `-1` | `PERMUTE` | 每 lane 索引向量 → `vgather2` / `vscatter` |
| 非常数且索引向量可构造 | `PIECEWISE` | 每 lane 索引向量 → `vgather2` / `vscatter` |
| 下标含 `BufferLoad` | `INDIRECT` | 每 lane 索引向量（来自寄存器）→ `vgather2` / `vscatter` |
| 非常数且不可构造 | `TRUE_RANDOM` | 明确拒绝 |

### 2.4 主数据流

**伪代码**

```text
build_plan(stmt)
  ├─ _try_build_program(stmt, analyzer, lanes, notes) ▶ VectorProgram? # 多个并列 serial 段（§6 第三步）
  │     任一段构造失败 → 段级原因写入 notes 并 return None（缺陷 B：不吞掉）
  ├─ _split_nest    ▶ axes[], body                    # SCoP 迭代域（单段路径，回落）
  ├─ _collect_aliases(body)          ▶ {var: expr}    # 标量中间量
  ├─ normalize_body(body, aliases)   ▶ StoreSpec[]    # 控制流 → 线性写列表
  ├─ rank_vector_axes(axes, stores, store_infos, load_infos) ▶ 轴排行（联合可行性 + 读写加权）
  ├─ for vector_axis in ranked:                       # 轴回退：逐个尝试
  │     try: 构造 _ValueContext(axes, vector_axis, lanes, analyzer, vector_dtype)
  │           for each store: build_value(spec.value) ▶ ValueNode
  │           ▶ VectorPlan
  │     except Unsupported: continue
  ▶ VectorProgram(axes=_outer_axes(...), segments=[plan])   # 单段包成 1 段 program

emit_plan(plan)
  ├─ VectorProgram → 逐段 _emit_segment_stmts（源序），program.axes 统一包一次
  ├─ reduce is None → _Emitter 逐 store 发射；重建 chunk 循环 + 外层 serial 循环
  └─ reduce != None → _emit_reduce：短归约（≤4 chunk）展开 + 多路累加器；否则真实循环 + 可变寄存器累加器
  ▶ TIR (SeqStmt / For)
```

---

## 3. 算法设计（含伪代码与实现）

### 3.1 SCoP 建模

**(a) 迭代域** —— `_split_nest` 从某个 `T.Parallel` For 出发，**连续吃掉**并行轴；遇到**直接内嵌且干净**的 serial 轴（`_serial_is_clean`）也一并吸收：

**实现**

```python
def _split_nest(stmt, analyzer) -> tuple[list[Axis], Axis | None,
                                         list[tirx.Stmt], tirx.Stmt, list[tirx.Stmt]]:
    axes: list[Axis] = []
    serial_axis: Axis | None = None
    current = stmt
    while isinstance(current, tirx.For):
        if current.kind == tirx.ForKind.SERIAL:
            if not _serial_is_clean(current, analyzer)[0]:
                break                       # 循环携带依赖 → 不吸收
        elif current.kind != tirx.ForKind.PARALLEL:
            break
        axis = Axis(current.loop_var, current.extent, len(axes), current.kind)
        axes.append(axis)
        if current.kind == tirx.ForKind.SERIAL:
            serial_axis = axis
        current = current.body

    # 平行链结束后若是 SeqStmt，且其中恰好一个干净 serial 循环：
    # 前兄弟语句 → pre、循环体 → main、后兄弟语句 → post（§3.11 第二步）
    return axes, serial_axis, pre, current, post
```

> **推论**：**必须从嵌套头调用**才能拿到全部轴。这正是语句树变换采用「自顶向下遍历」、测试 helper 取「最外层 `T.Parallel`」的原因（见 §3.9）。
>
> **下界必须为 0**：迭代域被建模成 `0 <= x < extent`，发射时按 `For(var, 0, extent)` 重建。
> `T.serial(1, K)` / `T.Parallel(1, K)` 的非零下界会被静默丢掉、变成错位下标，因此
> `_require_zero_min` 在吸收前拒绝无法证明下界为 0 的循环。

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

def classify_delta(delta, analyzer=None) -> tuple[AccessClass, int | None]:
    """Δ 层：只分常数 / 非常数（非常数交结构细化层）。"""
    value = _const_int(delta, analyzer)  # 能否（符号）证明成常数？
    if value is None: return AccessClass.PIECEWISE, None   # 折不动 → 待结构细化
    if value == 0:    return AccessClass.UNIFORM, 0
    if value == 1:    return AccessClass.CONTIGUOUS, 1
    if value == -1:   return AccessClass.PERMUTE, -1
    return AccessClass.STRIDED, value
```

**差分之后必须再化简一次**：`off` 已化简，但 `off(v+1) - off(v)` 会重新产生 `(v+1)//m*m + (v+1)%m - v` 这类式子，只有再化简才能压回常数。

> **设计精髓**：`_const_int(delta) is None` 把「非仿射」的判定**外包给符号化简器的能力边界**——化简器能折成常数就是仿射，折不动就是非常数。这是「把语义问题转成符号计算问题」的典型手法，既优雅也有风险：**化简器的行为就是接口契约**（因此 `finite_difference` 前后各 `simplify` 一次）。
>
> `_const_int` 本身也不只认字面量 `IntImm`：给了 `analyzer` 时先 `simplify`，再用 `const_int_bound`——区间是可靠的，因此**上下界相等即证明为常数**，这样「化简后仍是常量表达式」的 Δ 也会被判成常数。

**非常数 Δ 的结构细化（改进 4）**：下标含 `BufferLoad`（`A[B[i]]`）→ `INDIRECT`（索引向量来自被装载的寄存器）；否则用 `index_vector_reason(offset)` 检查元素偏移能否在索引向量通道构造（只允许整数算术；`FloorDiv`/`FloorMod` 除数须为**正的 2 的幂常量**）——可构造 → `PIECEWISE`，不可构造 → `TRUE_RANDOM`（带诊断原因）。`Δ == -1`（`A[N-1-i]`）单独标 `PERMUTE`。三类（`PERMUTE`/`PIECEWISE`/`INDIRECT`）共用同一个「每 lane 索引向量」通道，区别只在诊断标签。

**实现**（访问分类的入口）

```python
def classify_access(buffer, indices, axes, analyzer, *, is_store) -> AccessInfo:
    scales = tuple(indices)
    indirect = any(_contains_buffer_load(index) for index in scales)
    offset = linear_element_offset(buffer, scales, analyzer)
    info = AccessInfo(buffer=buffer, indices=scales, offset=offset,
                      is_store=is_store, indirect=indirect,
                      in_global=buffer.scope() == "global")
    reason = None if indirect else index_vector_reason(offset, analyzer)
    for axis in axes:
        kind, stride = classify_delta(finite_difference(offset, axis.var, analyzer), analyzer)
        if indirect:
            # 沿轴不变（Δ==0）→ lane 内均匀标量地址，仍是 UNIFORM（D11）
            if kind != AccessClass.UNIFORM:
                kind, stride = AccessClass.INDIRECT, None
        elif kind == AccessClass.PIECEWISE and reason is not None:
            kind, stride = AccessClass.TRUE_RANDOM, None   # 索引向量构造不出来 → 拒绝
            info.reason = reason
        elif kind == AccessClass.STRIDED and stride < 0 and reason is not None:
            kind, stride = AccessClass.TRUE_RANDOM, None   # 负步长必须走索引向量（D8）
            info.reason = reason
        info.per_axis[axis.index] = (kind, stride)         # 键 = 轴在迭代域中的位置
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
| 内层 `For` | `Unsupported`（见 §3.11 / §6 场景） |
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

> 注意：`normalize_body` 遇到内层 `For` 才拒绝——但**直接内嵌、链式嵌套、以及并列的干净 `T.serial` 循环，在进入 `normalize_body` 之前就已被吸收成迭代轴**（见 §3.11）。因此「含 serial 的并行体」大多根本走不到这条拒绝分支。

### 3.4 向量轴选择

`rank_vector_axes` 是**纯打分 + 轴回退**，无一处匹配代码形状：

1. **联合可行性**：某轴上只要有 store 是 `TRUE_RANDOM`，或**所有** store 都写不进去（无候选），该轴出局；`STRIDED`/`PERMUTE`/`PIECEWISE`/`INDIRECT` 的 store 都能进候选（靠 `vscatter`）。
2. **读写加权**：`score = 100 * Σ store权重 + 10 * Σ load权重 + axis.index`，权重 `CONTIGUOUS=8 > PIECEWISE=4 > STRIDED=PERMUTE=3 > INDIRECT=2 > UNIFORM=1 > TRUE_RANDOM=0`。
   * `*100 / *10` 让写侧成为主序、读侧次之（避免「写连续、读四处乱跳」的轴被误选）；
   * `+axis.index` 让同分取**更内层**（更内层通常更连续）。
3. **轴回退**：返回**排序后的整条轴列表**，`build_plan` 依次尝试；某轴发射失败时自动退到次优轴。
4. **归约不是独立的选轴入口**（见 §3.5）：打分只看「写侧是否 true-random」与加权分；归约在**选定轴之后**由 `_classify_reduce` 判定。归约轴若过不了 build 期门禁，仍会**回退到下一根轴**。

**为什么"打分"和"回退"分两阶段**：打分只能基于**分析期的形态**（Δ 的分类），但一个轴最终能否发射还取决于**发射期**才能确定的细节——元素偏移能否在「每 lane 索引向量」通道构造出来（如非 2 的幂常量除法就构造不出）、值算子是否有 strength-reduction 等。所以打分只作"优先级"，真正的可行性用 `try/except` 在发射期兜底。

**例：一次真实的轴回退** —— nest 为 `for m in parallel(16): for kk in parallel(12)`，写 `data_shared[m, kk] = select(in_bound, data[m//64, access_h, access_w, kk%4], 0)`，其中 `access_h = m%64//8 + kk//12 - 1`：

| 轴 | 写 `data_shared[m, kk]` | 读 `data[...]` | 打分 |
|---|---|---|---|
| `kk`（内） | Δ=1 → `CONTIGUOUS`(8) | 下标含 `kk//12`（÷12 非 2 的幂）→ `TRUE_RANDOM`(0) | 100·8 + 10·0 + 1 = **801** |
| `m`（外） | Δ=12 → `STRIDED`(3) | `m//8`/`m%8` 可构造 → `STRIDED`(3) | 100·3 + 10·3 + 0 = **330** |

`kk` 因"写连续"拿最高分、被优先尝试；但发射它的读侧需要构造每 lane 索引向量，`kk//12` 的除数 12 不是 2 的幂、构造不出，于是抛 `Unsupported`。规划器捕获后**回退到次优轴 `m`**：写侧 `vscatter`（步长 12 的绝对偏移索引向量）＋读侧步长 4 装载＋谓词 `vsel`＋尾块掩码 `update_mask(16)`，成功发射。诊断完整记录了"分析期排序 → 发射期回退 → 最终结果"：

```text
axis 'm'[0]:  store=['strided']  load=['strided']      -> candidate (score=330)
axis 'kk'[1]: store=['contiguous'] load=['true_random'] -> candidate (score=801)
axis 'kk': FloorDiv by non-power-of-two constant 12 is not strength-reduced yet (§5.4(a)) -> fallback
selected vector axis 'm' (score=330)
```

> 这就是"不静默降级、也不轻易整体放弃"：最高分的轴发不出时退到次优轴，只有**所有**轴都发不出才整体拒绝。注意**成功路径也保留失败轴的 `-> fallback` 行**，否则最终计划上看不出"为什么没选最高分轴"。

**实现**

```python
def rank_vector_axes(axes, store_infos, load_infos, analyzer, lanes):
    ranked = []
    for axis in axes:
        if any(info.axis_class(axis)[0] == AccessClass.TRUE_RANDOM for info in store_infos):
            continue                                    # 写侧不可构造 → 出局
        # 联合打分：写侧主序（×100）、读侧次之（×10）、再乘 lane 利用率；同分取更内层。
        weight = 100 * sum(_AXIS_WEIGHT[info.axis_class(axis)[0]] for info in store_infos)
        weight += 10 * sum(_AXIS_WEIGHT[info.axis_class(axis)[0]] for info in load_infos)
        score = weight * _active_lanes(axis, analyzer, lanes) // lanes + axis.index
        ranked.append((score, axis))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [axis for _, axis in ranked]                 # 整条列表 → 轴回退
```

> 对比：C++ 参考实现里「识别归约」是专门逻辑；这里它是**选轴之后**的正交判定——**先选轴，再问这个轴是不是归约轴**。

### 3.5 归约识别

条件（同时满足）：

* 该轴所有 store 都 `UNIFORM`（`Δ_store = 0`）——第一层判据 `_is_reduce_axis`；
* 取值沿该轴**变化**（自由变量出现 **或** 被访问 buffer 沿轴 `CONTIGUOUS/STRIDED`）；
* 单条 store 还须是**累加形式** `dst[...] = dst[...] <op> x`（`op` ∈ Add/Max/Min），旧值作为归约算子的**直接操作数**且与写入地址**同址**——第二层判据 `_plan_reduce`。

命中则构造 `ReduceInfo(axis, op, recipe, varying, accumulate)`，其中 `op` 由取值脊上的归约算子挑（表源是 `primitives._REDUCE_PRIMITIVES`：`Add→vadd`、`Max→vmax`、`Min→vmin`），`recipe ∈ {vcadd, vcmax, vcmin}`，`varying` 是剥离旧值后的逐 lane 取值。归约判定发生在**选轴之后**；该轴过不了下列门禁时，规划器**回退到下一根轴**（不再是"独占返回"）。

> ⚠️ **使用者须知（归约算子的支持范围）**：
> 1. **只识别 `Add`(`+=`) / `T.max` / `T.min`**：其它算子（如 `Mul` 的乘积归约 `s[i] *= a[i,j]`）会被**显式拒绝**（`unsupported reduction operator(s) Mul`）——不会再静默发成求和，但也无法向量化。
> 2. **必须有累加形式**：`s[i] = a[i,j]`（last-write-wins 覆写）会被**显式拒绝**（`does not read 's' … accumulate form`）；旧值被缩放（`s[i] = s[i]*2.0 + a[i,j]`）或跨位（stencil）读（`s[i] = s[i-1] + a[i,j]`）同样拒绝。
> 3. **旧值只能出现一次**（R4）：`s[0] = s[0] + (s[0] + a[i,j])` 会被拒绝——单个累加初值表达不了两份旧值。
> 4. **归约不能带用户条件**（R3）：`if cond: s[0] += a[i]` 会被显式拒绝。跨 lane 归约的掩码当前只承载尾块谓词，用户谓词会被静默丢弃；后续可用 `vsel(varying, identity, cond)` 填单位元实现。
> 5. **归约段的 `pre`/`post` 必须 lane 均匀**（R7）：沿并行轴归约时，`t[i] = 1.0` 这类连续 pre 会让该轴在 build 期失败并回退到 serial 轴。
>
> 因此**写归约时请只用 `+=`（Add）/ `T.max` / `T.min` 三种形态，旧值直接作为操作数，且不要给归约语句加 `if`**。

**实现**

```python
# primitives.py：归约原语事实表（白名单 + 指令映射 + 单位元，唯一来源）
_REDUCE_PRIMITIVES = (
    ReducePrimitive("Add", "vadd", "vcadd", "vadds", _zero_identity),
    ReducePrimitive("Max", "vmax", "vcmax", "vmaxs", _min_identity),
    ReducePrimitive("Min", "vmin", "vcmin", "vmins", _max_identity),
)
_REDUCE_OPS     = {p.tir_op: p.op for p in _REDUCE_PRIMITIVES}       # Add → vadd
_REDUCE_RECIPE  = {p.op: p.recipe for p in _REDUCE_PRIMITIVES}       # vadd → vcadd
_REDUCE_SCALAR_RECIPE = {p.op: p.scalar for p in _REDUCE_PRIMITIVES}  # vadd → vadds

# 第一层：轴是否为归约轴（由 _classify_reduce 使用；选轴阶段不特判归约）
def _is_reduce_axis(axis, stores, store_infos, analyzer):
    return all(info.axis_class(axis)[0] == AccessClass.UNIFORM for info in store_infos) and any(
        _value_varies_along(spec.value, axis, analyzer) for spec in stores)

# 第二层：逐 store 精化为累加形式（build 期；不支持即拒绝）
def _plan_reduce(store, vector_axis):
    buffer = store.spec.buffer
    if store.mask is not None:                         # 条件归约（R3）→ 拒绝
        raise Unsupported("conditional reduction ... guard would be silently dropped")
    if not _reads_buffer(store.value, buffer):         # 覆写（last-write-wins）→ 拒绝
        raise Unsupported("... does not read ... accumulate form")
    op = _reduce_op([store.spec])                      # Add/Max/Min 之外显式拒绝
    varying, stripped = _strip_reduce(store.value, op, store.spec)   # 剥离次数必须恰好 1（R4）
    if stripped != 1 or varying is None:
        raise Unsupported("... must be a direct operand ... accumulate position")
    return ReduceInfo(axis=vector_axis, op=op, recipe=_REDUCE_RECIPE[op],
                      varying=varying, accumulate=True)
```

### 3.6 值 DAG 构造与 `is_uniform`

`build_value` 是**结构无关的递归**，决策仍只用 Δ：

**伪代码**

```text
BufferLoad  → classify_access → load_node：
    Δ=1            → vload（连续装载）
    Δ=0            → uniform_load（标量读一次 + vdup）
    Δ=k            → vload_strided（发射成 vgather2）
    PERMUTE/PIECEWISE/INDIRECT → gather（每 lane 索引向量 → vgather2）
    TRUE_RANDOM    → 拒绝（带 index_vector_reason 诊断）
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
    if not self.args:                        # 其余无 args 叶子（vload/vload_strided/gather）恒非均匀
        return False
    return all(arg.is_uniform for arg in self.args)   # 递归传播
```

它驱动：

* **标量-向量特化**：`vadd→vadds`、`vmul→vmuls`、`vmax→vmaxs`、`vmin→vmins`；无专用变体则先 `broadcast(vdup)` 再走向量-向量。
* **比较特化**：一侧均匀 → `vcmps`；`vcmps` 要求 `(vector, scalar)` 顺序，交换操作数时谓词**必须翻转**（`lt↔gt`、`le↔ge`）。
* 是否需要 `vdup`；归约的累加写判定。

> ⚠️ **必须理解的坑**：`all([]) == True`。叶子节点若不在 `_UNIFORM_RECIPES` 里会被**误判为均匀**。实现已统一在递归之前挡掉**所有无 args 叶子**（`lane_index`/`vbrc`/`vload`/`vload_strided`/`gather`），所以步长/索引向量装载不会再被当成标量喂进标量-向量指令。若仍遇到发射期 `cannot scalarize vector value ...`，请按新用例上报。

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
| `gather` | `vgather2(base(BufferLoad 全 0), reinterpret(u32, idx), m)` | 每 lane 索引向量装载（`PERMUTE`/`PIECEWISE`/`INDIRECT`） |
| `uniform_load`/`uniform_expr`/`vbrc` | `vdup(scalar, elem, mask, MODE_ZEROING)` | 广播 |
| 数据二元 / 一元 | `vadd/vsub/vmul/... / vneg/vabs/...` | `(a, b, mask, mode)` |
| 标量-向量 | `vadds/vmuls/vmaxs/vmins` | `(vec, scalar, mask, mode)` |
| 常量移位 | `vshrs` / `vshls` | 整型 `//` / `%` 2 的幂常量除数做 strength-reduction |
| `vcmp` / `vcmps` | `vcmp(a,b,mask,op)` / `vcmps(a,s,mask,op)` | 谓词 |
| `vsel` | `vsel(t, f, mask)` | mask 由 `predicate()` 给出 |
| `pand`/`por`/`pnot` | `pand/por/pnot` | 谓词运算；执行掩码**跟随数据元素宽度**（R1） |
| `vcvt` | `vcvt(src, target, mask)` | mask bits = `min(src, target)` |

**写入映射（`_emit_store`）**：`CONTIGUOUS`/`UNIFORM` → `vsts(addr, v, mask, "NORM_B{bits}"/"ONEPT_B{bits}")`；`STRIDED` → `vscatter(v, addr(chunk 基址), vmuls(vci(0,u32), k, m), m)`；`PERMUTE`/`PIECEWISE`/`INDIRECT` → `vscatter(v, index_addr（缓冲区起点）, reinterpret(u32, idx), m)`。

**实现（`_Emitter._build` 的代表性分支）**

```python
def _build(self, node: ValueNode):
    recipe, args = node.recipe, node.args

    if recipe == "vload":                  # 连续装载；尾块/守卫退化成带谓词的 vgather2（D10）
        mask = self.load_guard(node, _mask_bits(node.dtype))
        if mask is None:
            return self.bind(simd.vld(self.addr(node.access), "NORM"), self.lets)
        return self.bind(simd.vgather2(self.addr(node.access),
                                       simd.vci(_int(0), "uint32"), mask), self.lets)

    if recipe == "vload_strided":          # 无带步长装载：vgather2 + ramp 索引
        _, stride = node.access.axis_class(self.plan.vector_axis)
        bits = _mask_bits(node.dtype)
        guard = self.load_guard(node, bits) or self.all_mask(bits)
        index = self.ramp(stride, self.ramp_mask(bits))   # 与 chunk 无关，可提到 prologue
        return self.bind(simd.vgather2(self.addr(node.access), index, guard), self.lets)

    if recipe in ("uniform_load", "uniform_expr", "vbrc"):   # 广播
        operand = node if recipe != "vbrc" else args[0]
        elem = _elem_dtype(operand.dtype)
        return self.bind(simd.vdup(self.scalar(operand), elem,
                                   self.data_mask(elem), _MODE), self.lets)

    if recipe == "vcmps":                  # 标量比较，要求 (vector, scalar)
        mask = self.data_mask(args[0].dtype)
        return self.bind(simd.vcmps(self.value(args[0]), self.scalar(args[1]),
                                    mask, node.detail), self.lets)

    if recipe == "vsel":                   # args = [cond, true, false]
        true_value = self.vector_operand(args[1])
        false_value = self.vector_operand(args[2])
        bits = _mask_bits(_elem_dtype(node.dtype))
        return self.bind(simd.vsel(true_value, false_value,
                                   self.predicate(args[0], bits)), self.lets)
    # ... 其余 recipe 按 _VEC_SCALAR_BINARY / _ELEMENTWISE_* 表分派
```

**谓词掩码的两条硬约束**（R1 / R2）：

1. `pand/por/pnot` 的第三个参数是**执行掩码**，按 `pset_b<width>` 的粒度覆盖 `256 / width`
   个 lane。它必须与**数据元素**的谓词宽度一致（f32→b32/64 lane、f16→b16/128 lane）；
   固定 b32 会把 f16/int8 的 lane 64 之后清零（尾块只写一半）。
2. store 的用户条件必须走 `predicate()`：均匀条件广播成 `update_mask(cond ? lanes : 0)`，
   逐 lane 条件直接使用比较谓词。把 bool 条件当数据 `vdup` 会在发射期因
   `Cannot infer default mask width for dtype 'bool'` 崩溃。

**SSA 与 `Bind`**：每个中间值用一条独立语句 `Bind(var, expr)` 表示，追加进 `lets` 列表。不能用表达式式的 `Let`，因为 SSA 中间值是一条条独立语句，只能摊平成 `SeqStmt`。

`_Emitter` 的主要状态：

| 状态 | 作用 |
|---|---|
| `prologue` | **chunk 无关**的绑定，如全 lane 掩码 `pset(bits, "PAT_ALL")`（按 width 缓存，只绑一次） |
| `lets` | chunk 体内的 SSA 语句 |
| `memo` | `id(node) → 变量`，同一子表达式去重 |
| `_mask_vars` | 掩码按 width / 尾块键缓存，避免重复 `pset` / `update_mask` |
| `counter` | SSA 变量命名序号（跨段再加 `prefix` 区分） |

**掩码三层叠加**：`store 的最终掩码 = 用户谓词 ∧ 尾块掩码`，都为空则全 1。

* 常量 extent 且 `tail != 0` 时，发射拆成「整 chunk 块（全 lane 掩码）+ 尾块块（`update_mask(tail)`）」，
  尾块块只有一次迭代；整 chunk 不再背一个恒真的 `update_mask`。
* 符号 extent 只发一个运行时 chunk 循环 `ceildiv(extent, lanes)`，尾块车道数在运行时
  **夹到 lanes**：`update_mask(min(extent - chunk*lanes, lanes))`。

> ✅ **原「三条已知缺陷」均已修复**（详见 §6）：
> 1. **尾块装载不再越界**：尾块内的连续装载退化成**带谓词的 `vgather2`**（`vld` 本身没有谓词形参），
>    步长/间接装载直接使用尾块谓词；归约尾块也先带谓词装载再用 `vsel` 填单位元（D10）。
> 2. **负步长已支持**：`Δ ≤ -2` 改走「每 lane 索引向量」通道（`vgather2`/`vscatter`，负偏移以补码表达），
>    读、写两侧都覆盖（D8）。
> 3. **符号 extent 已支持**：非归约与归约都由运行时 chunk 循环驱动，尾块掩码运行时夹紧；
>    归约在符号 extent 下走「真实循环 + 可变寄存器累加器」，不再在规划期拒绝（D9）。

**系统常量**：

* `lanes = 2048 / elem_bits`：float32→64、f16/bf16→128、int8→256；向量寄存器 2048 位。
* 谓词类型一律 `boolx256`（256 位 `vector_bool`）。
* `_mask_bits(dtype) = max(elem_bits, 8)`：CCE 谓词粒度 b8/b16/b32/b64（无 b4）。
* `_MODE = "MODE_ZEROING"`。

> 放置原则：`primitives.py` 只放 ISA/算子事实与硬件常量；`AccessClass` 留在 `access.py`，
> 可调策略 `_AXIS_WEIGHT` 留在 `plan.py`——各表跟随其语义归属层，而非物理集中。

### 3.8 归约发射（最特别的部分）

归约按 chunk 数分**两种策略**：短归约（`chunks <= 4`）编译期展开并拆多条累加链；
长归约与符号 extent 走「真实 chunk 循环 + 可变寄存器累加器」。两条路径的尾块处理一致。

**A. 短归约：编译期展开 + 每 chunk 一条链 + 树形合并**

```text
for ck in range(chunks):                        # 编译期展开，不是运行时 for！
    if tail and ck == last:                     # 尾块：带谓词装载（D10）
        x = vgather2(src_addr(ck), vci(0), tail_mask)
        x = vsel(x, fill, tail_mask)            # 无效车道填单位元
    else:
        x = vld(src_addr(ck), "NORM")           # 整 chunk
    red[ck] = vcadd(x, mask)                    # 每个 chunk 一条独立链，不需要单位元
acc = red[0..chunks) 树形合并                    # 两两 vadd（lane 0）
if 累加写: old = 标量读 dst; acc = vadds/vmaxs/vmins(acc, old, lane0_mask)
vsts(dst_addr, acc, lane0_mask, "ONEPT_B{bits}")
```

**B. 长归约 / 符号 extent：真实循环 + `local.var` 累加器**

```text
acc = AllocBuffer(1, elem x lanes, scope="local.var")   # 可写向量寄存器
acc[0] = vdup(identity, elem, mask)
for ck in range(full or ceildiv(extent, lanes)):        # 真实运行时循环
    x = vld(src_addr(ck), "NORM")                       # 符号 extent 时带谓词装载 + vsel 填单位元
    acc[0] = vadd(acc[0], vcadd(x, mask), lane0_mask)
final = acc[0]
if 累加写: old = 标量读 dst; final = vadds/vmaxs/vmins(final, old, lane0_mask)
vsts(dst_addr, final, lane0_mask, "ONEPT_B{bits}")
```

**为什么要分档？** 展开去掉循环控制、给后端跨块调度空间，收益集中在 chunk 数少时；
chunk 数大时指令体积随 `chunks` 线性膨胀而收益占比下降，因此改用循环 + 可变寄存器
（`tirx.AllocBuffer` + `BufferStore`/`BufferLoad`，即 `T.simd.alloc_var` 的语义）。
**树形合并**（短归约）把 `vcadd` 的串行依赖从 `chunks` 级压到 `ceil(log2 chunks)` 级，
指令条数不变；每个链首直接取该 chunk 的部分结果，因此不需要 `vdup` 单位元。
**归约后的算术收窄到 lane 0**：`vcadd/vcmax/vcmin` 只有 lane 0 有效（硬件模板用 `PAT_VL1`
合并），链合并/累加/回写统一用 `PAT_VL1` 谓词；累加写的旧值按标量读一次直接进
`vadds/vmaxs/vmins`，省掉 `BRC` 广播装载。

> 早期实现的 `_REDUCE_CHAINS`（=4）与展开阈值相等，`min(4, chunks)` 个槽位在
> `chunks <= 4` 时每槽恰好只用一次，「链内累加」分支不可达；已删除该常量与分支，
> 发射结果逐指令不变。

**尾块（改进 6）**：`extent % lanes != 0` 时最后一个 chunk 只有 `tail` 个有效车道，其余车道的 `vld` 读到越界/脏数据；用 `update_mask(extent - ck*lanes, bits)` 造尾块谓词、`vsel(x, fill, tail_mask)` 把无效车道替换成**归约单位元**，再按全 lane 掩码归约（`vadd` 填 0、`vmax` 填 -inf、`vmin` 填 +inf 均不影响结果）。这样 `tail != 0` 也能正确发射，不再需要整除前提。

> ✅ **使用者须知（归约相关，D4/D5/D6/D13 已修）**：
> 1. **归约体可以夹 lane 均匀加数**：`s[i] += a[i,j] + 1.0` 现按正确语义发射（均匀加数保留在归约体内，跨 lane 归约天然正确处理）。
> 2. **累加读须与写入地址同址**：`s[i] = s[i] + a[i,j] + 1.0` 会正确识别为累加；但 `s[i] = s[i] * 2.0 + a[i,j]`（旧值被缩放）或 `s[i] = s[i-1] + a[i,j]`（跨位 stencil 读）会被**显式拒绝**（旧值必须作为归约算子的直接操作数且同址），请改写为 `s[i] += a[i,j]`。
> 3. **一个 nest 内支持多条归约**：`s1[i]+=a[i,j]` / `s2[i]+=b[i,j]` 乃至求和+求最大混合，各自独立发射（求和用 `vcadd`、求最大用 `vcmax`），不再丢写。
>
> 注：单位元已按元素类型派生，`int32` / `float16` 的 `max`/`min` 归约不会再抛 `ValueError`（缺陷 D3 已修）。

**实现**

```python
def _emit_reduce(plan: VectorPlan, emitter: _Emitter) -> None:
    for store in plan.stores:                       # D6：逐 store 发射，不再只取第一条
        if store.reduce is None:
            _emit_outer_store(store, emitter)       # 取值沿轴均匀 → 一次广播写
            continue
        _emit_one_reduce(plan, store, emitter)


def _emit_one_reduce(plan, store, emitter) -> None:
    reduce = store.reduce                            # D13：归约描述由 _classify_reduce 在 build 期填好
    chunks = plan.chunks
    if chunks == 0:                                  # 零次迭代：目标保持原值
        return
    if chunks is not None and chunks <= _REDUCE_UNROLL_LIMIT:
        _emit_reduce_unrolled(plan, store, emitter, reduce, chunks)   # A：展开 + 每 chunk 一条链
    else:
        _emit_reduce_loop(plan, store, emitter, reduce)               # B：真实循环 + local.var


def _emit_reduce_unrolled(plan, store, emitter, reduce, chunks) -> None:
    elem, bits = _elem_dtype(store.spec.buffer.dtype), _mask_bits(...)
    mask = emitter.all_mask(bits)
    fill = vdup(identity) if plan.tail and reduce.recipe != "vcadd" else None   # Max/Min 尾块填充
    chains = []
    for ck in range(chunks):                         # 编译期展开
        emitter.lane_base = _int(ck * plan.lanes)
        emitter.memo, emitter._predicates = {}, {}   # 地址/谓词随 chunk 失效
        emitter.masked_tail = bool(plan.tail) and ck == chunks - 1
        chains.append(_emit_reduce_chunk(plan, store, emitter, reduce, bits, mask, fill))
    emitter.masked_tail = False
    while len(chains) > 1:                           # 树形合并（只算 lane 0）
        lane0 = emitter.lane0_mask(bits)
        chains = [vadd(chains[k], chains[k + 1], lane0) for k in range(0, len(chains) - 1, 2)]
                  + ([chains[-1]] if len(chains) % 2 else [])
    _emit_reduce_tail(store, emitter, reduce, elem, bits, chains[0])


def _emit_reduce_chunk(plan, store, body, reduce, bits, mask, fill):
    loaded = body.value(reduce.varying)              # D4/D5：已剥离旧值，均匀加数在内
    tail = body.tail_mask(bits) if body.masked_tail else None
    if tail is not None:
        if reduce.recipe == "vcadd":                 # Add 单位元是 0：零化掩码天然正确
            return body.bind(simd.vcadd(loaded, tail, _MODE), body.lets)
        loaded = body.bind(simd.vsel(loaded, fill, tail), body.lets)   # Max/Min 先填单位元
    return body.bind(getattr(simd, reduce.recipe)(loaded, mask, _MODE), body.lets)


def _emit_reduce_tail(store, emitter, reduce, elem, bits, acc):
    lane0 = emitter.lane0_mask(bits)                 # vcadd 结果只有 lane 0 有效
    if reduce.accumulate:                            # 旧值按标量读一次 → vadds/vmaxs/vmins
        old = emitter.addr(store.access, uniform=True)
        acc = getattr(simd, _VEC_SCALAR_BINARY[element_op])(acc, old, lane0, _MODE)
    vsts(emitter.addr(store.access, uniform=True), acc, lane0, f"ONEPT_B{bits}")
```

配套细节：

* `_reduce_identity`：按元素类型派生——`vcadd→0`、`vcmax→tirx.min_value(elem)`、`vmin→tirx.max_value(elem)`。
* `_strip_reduce`：在归约算子**脊**（`op` 及其标量同构 `vadds/vmaxs/vmins`）上递归剥离与写入地址**同址**的旧值读取，并统计**剥离次数**（必须恰好 1 次）；`accumulate=True` 时末尾把旧值按标量并入。旧值被缩放、跨位（stencil）读或重复出现则显式拒绝。
* 归约路径逐 chunk 改 `lane_base` 并**清空 `memo` / `_predicates`**（地址随 chunk 变，缓存必须失效）。
* 尾块（改进 6）：`tail != 0` 时用 `update_mask` + `vsel` 把无效车道填成归约单位元；Add 直接复用尾块掩码（零化=单位元 0）。

### 3.9 循环重建与遍历策略

`emit_plan`：先把 chunk 体包成 `for chunk in serial(ceil(E/L))`，再**逆序**把非向量轴包成 serial 循环（跳过向量轴本身）。于是 `T.Parallel(M,K)` → `for i in serial(M): for chunk in serial(K/L): ...`。

对 `VectorProgram`（多个并列 serial 段，§3.11），`emit_plan` 逐段发射（每段一个 `chunk_s<k>` 循环 + 该段自身的非向量 serial 轴），**共享外层并行轴只在最外层包一次**，各段掩码 prologue 提到最前，语句顺序即源码顺序。

`vectorize_parallel_nests`：**自顶向下（pre-order）**遍历语句树，命中 `T.Parallel` 就整体 `emit_plan(build_plan(...))` 替换（`_split_nest` 会把连续的并行轴与可吸收的干净 serial 轴一并吃掉，故整个 nest 被一次性下降）；整体规划失败才返回 `None` 继续下探子节点。

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

### 3.10 诊断体系（改进 7）

规划过程**无论成败都收集全部卡点**，而不是只抛第一个原因：

* `_axis_diagnostics` 为每根轴生成一行分类摘要——候选轴给出 `store=/load=` 形态与联合打分；`TRUE_RANDOM` 轴给出 `rejected (true-random store)` 并附带原因（如「非 2 的幂常量除法」）。
* 成功时这些行连同 `selected vector axis '<name>' (score=...)` 一起写入 `VectorPlan.diagnostics`（可用 `render_plan` 打印）；**失败轴的原因也会保留**（`axis '<name>': <reason> -> fallback`）。失败时把同样的行 + 每根轴的发射失败原因塞进 `Unsupported.diagnostics`，而 `str(err)` 仍是单行摘要。

```python
diagnostics = _axis_diagnostics(axes, store_infos, load_infos)   # 每根轴的分类
for vector_axis in ranked:                                       # 轴回退
    try:    plan = _build_plan_for_axis(...)
    except Unsupported as err:
        diagnostics.append(f"axis '{vector_axis.name}': {err} -> fallback")  # 成功路径也保留
        continue
    diagnostics.append(f"selected vector axis '{vector_axis.name}' (score=...)")
    plan.diagnostics = diagnostics
    return plan
raise Unsupported("no vectorizable axis: ...", diagnostics=diagnostics)
```

### 3.11 循环吸收与多段规划（§6：`T.Parallel` 体内的 serial 循环）

`T.Parallel` 体内嵌套 `T.serial` 是常见写法（归约、softmax、分块搬运等）。这类场景在**进入 §3.3 的 `normalize_body` 之前**先被吸收成迭代轴。按复杂度分三步：

| 步骤 | 形态 | 处理 |
|---|---|---|
| 第一步 | 并行体**直接是一个**（或链式嵌套的）干净 `T.serial` 循环 | 该 serial 轴吸收为候选项量轴（S1 连续 copy → `vld/vsts`；S3 归约 → `vcadd`） |
| 第二步 | 并行体是 `SeqStmt`，其中**恰好一个**干净 serial 循环，前后还有兄弟语句 | 吸收该 serial 轴；前置兄弟语句作 `pre`、后置作 `post`，在 chunk 循环**之外**按 `pre → 主体 → post` 每迭代一次 |
| 第三步 | 并行体是 `SeqStmt`，含**多个并列的干净 serial 循环** | 走**多段**路径，产出 `VectorProgram`（`N` 段，源序发射） |

「干净」= `_serial_is_clean`，四条保守判据：
1. 沿循环变量地址不变的写必须是同址累积归约（`cnt[0] = a[i,j]` / `cnt[0] += 1` 拒绝）；
2. 归约取值必须真的沿该轴变化；
3. **同一 buffer 的多条写必须目的地址相同**（R6：`c[i,2*j]` 与 `c[i,j]` 会跨迭代相撞）；
4. 不允许跨迭代读（stencil `a[j-1]`、共享标量读取）。
此外循环**下界必须可证为 0**（R5），否则迭代域建模会丢掉下界。

**第三步：多段（`VectorProgram`）**

为什么不「融合成一段」：多段之间常带**顺序语义**（如 softmax 的 `max → sum → 归一化`），融合后第一段必须整段算完才能进第二段，跨段的 store→load forwarding 又需要新的值通道规则。而**逐段独立向量化 + 源序发射**天然保持「段 k 完全先于段 k+1」，不需要新语义，也不动既有发射层。

```python
# build_plan 先试多段；不适用/失败则返回 None，回落到 §3.3–§3.10 的单段路径
def _try_build_program(stmt, analyzer, lanes, notes=None) -> VectorProgram | None:
    parallel_axes, body = _absorb_parallel_chain(stmt, analyzer)   # 连续吸收 T.Parallel
    if not parallel_axes or body 不是 SeqStmt:
        return None                       # 单段路径
    for child in body:                    # 按源序切分
        if child 是 For:
            chain = _absorb_serial_chain(child)   # 连续吸收干净 SERIAL
            if chain is None: return None         # 不干净/最内层仍循环 → 回落单段
            items.append(("seg", chain))
        else:
            items.append(("stmt", child))
    if serial 链少于 2: return None        # 只有 ≥2 段才走多段
    # 语句组按源序归属：段前 → 该段 pre；末尾剩余 → 末段 post
    segments = [_build_segment([*parallel_axes, *serial_axes], pre, body, post) for 每个 seg]
    # 只提「第一根被某段用作向量轴的并行轴」之前的轴；任一段展平时不提层
    consumed = [seg.vector_axis.index for seg in segments if seg.vector_axis.is_parallel]
    hoisted = [] if flattened_any else [a for a in parallel_axes
                                        if not consumed or a.index < min(consumed)]
    return VectorProgram(axes=hoisted, segments=segments)
```

每段的规划（`_build_segment`）与单段**完全同构**：`axes = [*并行轴, *该段 serial 轴]`，向量轴由**打分 + 展平优先**决定（并行轴与 serial 轴都是候选）；每段各自产出 `VectorPlan`（带自己的 `reduce` / `pre_stores` / `post_stores`）。

发射（`_emit_program`）：逐段生成 `chunk_s<k>` 循环，**共享外层并行轴只包一次**，各段掩码 prologue 提到最前，顺序 = 源码顺序。

> **提层前缀（重要修正）**：早期版本把全部共享并行轴提到 program 层，但某些段的向量轴
> 就是外层并行轴（已吃成 lane 维），同一组 64-lane 操作被外层 `for i` 重复执行
> `extent/lanes` 次。现在只提「第一根被用作向量轴的并行轴」之前的轴，`program.axes`
> 可以为空，各段自己包。

**段间共享与寄存器直通**（纯增量优化，不改变源序执行模型）：

1. **共享 prologue**：全 lane / lane0 掩码与步长斜坡只依赖常量，提到 program 最前只绑
   一次（例如 `p_serial_reduce_then_use` 的 2×`pset` + 2×`ramp` 降为 1+1）。
2. **寄存器直通**：段 k 的累加器回写把 `(buffer 键, 下标) → (向量轴变量, 寄存器值)`
   登记到 program 级表；段 k+1 同址连续读直接复用该寄存器，省掉 `vld`。
   仅当**生产段**单 chunk、无尾块、无用户谓词、回写未被 post 接管、向量轴前无未提层
   外层轴，且**消费段**单 chunk、向量轴变量相同、装载无尾块/守卫时成立；任何写都会
   作废登记。生产段的 `vsts` 保留（内存写是否可省需要额外 liveness 信息）。
   多 chunk 段之间不会直通（lane 基址逐 chunk 变化）。

**正确性前提**：每个并行迭代内，各段按源码顺序**整段执行**（段 k 的全部 lane 完成后才进入段 k+1），因此**段间** RAW/WAR 天然成立；真正需要排除的是**段内**的循环携带依赖：

| 形态 | 依赖关系 | 结果 |
|---|---|---|
| 两个互不相关的 serial（各自 element-wise） | 无 | 支持 |
| 归约段 → 广播使用段（`s[i] += …` 后 `c[i,l] = s[i]*2`） | 顺序执行；`s[i]` 在前段已完成 | 支持 |
| element-wise 组合（`c[i,j]=a*2` 后 `d[i,l]=c[i,l]+1`） | 顺序执行；`c` 在前段已完成 | 支持 |
| **跨段**跨地址读（前段写 `c`，后段读 `c[i,N-1-l]`） | 顺序执行；`c` 已完整物化 | 支持（后段按每 lane 索引向量 `gather` 读） |
| **同一段内**跨地址真依赖（`c[i,l] = c[i,l-1] + …`，循环携带 stencil） | 段内沿向量轴非均匀 | 拒绝 |
| **同一段内**跨迭代 WAW 别名（`c[i,2*l]` 与 `c[i,l]`） | 不同迭代写同一地址 | 拒绝（R6） |
| 并列段中**任一段**不干净 | — | 整体回落单段路径，按「nested serial loop」拒绝 |

> **低风险不变式**：多段路径**只新增**接受面。`_try_build_program` 一旦返回 `None`，`build_plan` 走的是与多段支持之前**逐字节相同**的单段代码；因此既有测试（含各拒绝用例）零改动，单段被接受场景的发射结果也不变。`render_plan` / `emit_plan` 同时接受两种计划类型。
>
> **统一返回契约（级别 1）**：无论走哪条路径，`build_plan` 都返回 `VectorProgram`——单段路径把结果包成 1 段（`axes=_outer_axes(...)`，仅含提到最外层包一次的并行轴前缀），调用方只需处理一种计划类型。该包装不改变发射结果：单独发射一个 `VectorPlan`（自己包全部非向量轴）与发射「1 段 program」产出的 TIR **逐节点结构等价**，由护栏测试 `test_single_segment_program_emits_identically_to_single_plan` 覆盖（含「向量轴就是外层并行轴」的关键用例）。
>
> **诊断不吞（缺陷 A/B）**：program 与每段都带 `diagnostics`（段级含逐轴分类与「强制向量轴」说明，program 级含段数摘要）；某段构造失败时，段级原因经 `notes` 合并进最终 `Unsupported.diagnostics`，不会只剩回落单段路径后那句泛化的「nested serial loop」。

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
| §2-II | 规约 + 尾块（`extent % lanes != 0`） | 支持 | 尾块 `update_mask` + `vsel` 单位元填充（改进 6） |
| §3-I | 多行语句（连续） | 支持 | `normalize_body` 产出多条 `StoreSpec` |
| §3-I | 多行语句（末维固定 `[.., k]`） | 支持 | 末维固定 → 沿唯一轴 STRIDED → `vscatter` 索引向量写 |
| §3-II | 复杂标量 + `if_then_else`（im2col） | 部分支持 | 值算子 `FloorDiv/FloorMod` 已对 **2 的幂常量除数**做精确 strength-reduction（`vshrs`/`vand`）；最优（写连续）轴仍可能因**非 2 的幂常量除数**被拒，此时靠**轴回退**（§3.4）退到另一根轴照常向量化 |
| §4-I | 表达式下标 | 支持（UB）/ 拒绝（GM） | 线性化+化简后连续 / GM 需 staging |
| §4-II-a | 高维摊平（一维化） | 支持 | 恒等式化简后 Δ=1 |
| §4-II-a | 步长访问（`a[2*i]`） | 支持 | STRIDED → `vgather2` |
| §4-II-b | 高维→低维（block-tiled 置换） | 支持 | Δ 非常数且可构造 → `PIECEWISE` → `vgather2` 索引向量 |
| §4-II-b | 逆序（`a[N-1-i]`） | 支持 | Δ=-1 → `PERMUTE` → `vgather2` |
| §4-gather | 间接下标 `A[B[i]]` | 支持 | data-dependent 下标 → `INDIRECT` → `vgather2`（索引来自寄存器） |
| §4-gather | 无结构下标（`a[i//3]`，非 2 的幂） | 拒绝 | `TRUE_RANDOM` → `not a power of two` |
| §5 | `T.atomic_add` | 拒绝 | 无「向量+标量」原子配方 |
| §6 第一步 | 直接内嵌、干净的 serial 循环（S1 copy / S3 归约） | 支持 | serial 轴被吸收为候选轴；S1 → 连续 `vld/vsts`，S3 → `vcadd` |
| §6 第二步 | 多条语句夹一个 serial（M1/M2/M3/夹心、S2 私有标量） | 支持 | body 恰好含一个干净 serial 循环时，其前后兄弟语句作为 `pre`/`post` 段，在 chunk 循环外每迭代一次（`pre → 主体 → post`） |
| §6 第三步 | 多个**并列**的干净 serial 循环（≥2 段） | 支持 | 逐段独立向量化 + 源序发射（`VectorProgram`）：每段各自选向量轴/归约/`pre`/`post`，共享外层并行轴只包一次；段间按源码顺序整段执行（段内不得有循环携带依赖） |
| §6 | 嵌套 serial / 跨迭代标量状态 / 并列段中有一段不干净 | 拒绝 | `nested ... loop ... is not affine-vectorizable` |
| §1-I | **双臂 `if` 的某一臂再嵌 `if`**（臂自带谓词） | 拒绝 | `masked branch merging is not supported`（先前文档未声明；缺陷 D12） |
| §2-II | **一个 nest 内多条归约**（`s1+=…` 且 `s2+=…`，或求和+求最大混合） | 支持 | **D6 已修**：逐 store 发射——每条归约各起独立累加链、各写一次 `ONEPT vsts`（求和用 `vcadd`、求最大用 `vcmax`） |
| §2-II | **归约体含 lane 均匀加数**（`s[i] += a[i,j] + 1.0`） | 支持 | **D4/D5 已修**：沿归约算子脊递归剥离旧值，lane 均匀加数保留在归约体内（跨 lane 归约天然正确处理） |
| §2-II | **非 Add/Max/Min 的归约**（乘积归约 `s[i] *= a[i,j]`） | 拒绝 | **D1 已修**：显式拒绝 `unsupported reduction operator(s) Mul`，不再静默发成求和 |
| §2-II | **覆写 `s[i] = a[i,j]`（store 沿轴均匀、取值变化）** | 拒绝 | **D2 已修**（随 D1）：显式拒绝，不再误判为 `vcadd` 归约 |
| §2-II | **`int`/`fp16` 的 `max`/`min` 归约** | 支持 | **D3 已修**：单位元按 dtype 派生，不再抛 `ValueError` |
| §4-II-a | **负步长（`Δ ≤ -2`，如 `a[126-2*i]`）** | 支持 | **D8 已修**：负步长走每 lane 索引向量（`vgather2`/`vscatter`），负偏移以补码表达；读、写两侧都覆盖 |
| §4-I | **符号 extent 的向量化**（`a: (n,)` 且 `n` 为变量） | 支持（非归约 + 归约） | **D9 已修**：非归约发运行时 chunk 循环 `ceildiv(n, lanes)` + 运行时夹紧的尾块掩码；归约走「整块主循环 + peel 尾块」+ `local.var` 累加器 |
| §0-I | **非归约 + 尾块**（向量轴长度非 `lanes` 整数倍） | 支持 | **D10 已修**：尾块单独成块，装载带谓词（连续装载退化成带谓词的 `vgather2`），三条 load 通道与归约尾块都不再越界读 |
| — | **多 store 混合 / 语句外置**（含间接下标等非均匀语句） | 支持 | **D7 已修**（混合 store 现 build+emit 均成功）；**D11 已支持**：间接下标只在**沿轴变化**时走索引向量通道，`Δ==0`（如 `c[b[0]]`）是 lane 内均匀标量地址 → pre/post 发一次 `ONEPT`，serial 主体照常向量化（部分向量化）；`bool` 等不支持的 store 元素类型在规划期拒绝 |
| §2-II | **吸收的 serial 轴上的部分归约**（`s[2*i] += a[i,j]` 与 `u[i,j] = s[2*i]` 混排） | 拒绝（单段路径回退并行轴） | **A0 已修**：逐元素路径会把 lane 均匀写发成 `ONEPT` 只留 lane 0（只累加 2/128 项）；现规划期拒绝，单段路径自动回退到并行轴 |
| §1-I | **谓词合取的掩码宽度**（f16/int8 尾块 + 条件） | 支持 | **R1 已修**：`pand/por/pnot` 的执行掩码跟随数据元素宽度，不再把 lane 64 之后清零 |
| §1-I | **循环不变条件**（`if a[0] > 0: c[i] = a[i]`） | 支持 | **R2 已修**：store 条件统一走 `predicate()`，均匀条件广播成 `update_mask(cond ? lanes : 0)` |
| §2-II | **条件归约**（`if cond: s[0] += a[i]`） | 拒绝 | **R3 已修**：归约掩码只承载尾块谓词，用户谓词会被静默丢弃 → build 期拒绝 |
| §2-II | **归约旧值重复出现**（`s[0] = s[0] + (s[0] + a)`） | 拒绝 | **R4 已修**：剥离次数必须恰好 1 |
| §6 | **非 0 循环下界**（`T.serial(1, K)`） | 拒绝 | **R5 已修**：迭代域建模成 `0 <= x < extent`，下界无法证明为 0 即拒绝 |
| §6 | **跨迭代写写别名**（`c[i,2*j]` 与 `c[i,j]`） | 拒绝 | **R6 已修**：同 buffer 多写必须目的地址相同 |
| §6 | **归约段 + 非均匀 pre/post** | 支持（轴回退） | **R7 已修**：该轴 build 期失败并回退到 serial 轴（不再 build OK / emit Unsupported） |
| — | 数学核心 Δ 分类 | 支持 | 6 个直接单测 |
| — | 语句树变换（整段下降） | 支持 | 自顶向下替换 |
| — | 诊断（改进 7） | 支持 | 成败都收集全部卡点 |

### 4.1 可复现性

`tilelang/ascend/analysis/test_drive_plan.py` 逐条编码上述场景并打印 `OK/REJECT + 原因`（`--tir` 时连带打印发射出的 TIR）。

```text
[OK ] c0_copy  c0_broadcast  c1_ifelse  c1_ternary  c1_single_arm  c2_rank
[OK ] c2_reduce  c2_reduce_tail（改进6: update_mask+vsel）  c2_axis_pick（改进2: 选轴 i）
[OK ] c3_multi  c3_scalars  c3_value_pow2（改进3: vshrs+vand）  c3_topk（改进1: vscatter）
[OK ] c4_flat  c4_strided  c4_reverse（PERMUTE → vgather2）  c4_tiled（PIECEWISE → vgather2）
[OK ] c_gather（INDIRECT → vgather2）  c3_im2col（改进2: 最优轴 kk 被拒 → 回退到 m）
[OK ] s1_serial_copy  s1_serial_predicated  s3_serial_reduce               （§6 第一步）
[OK ] s2_serial_scalar  m_serial_after_init  m_serial_before_post          （§6 第二步）
[OK ] m_serial_middle  m_serial_indep_group                               （§6 第二步）
[OK ] p_two_serial_independent（program: 2 段）  p_serial_reduce_then_use     （§6 第三步）
[OK ] p_serial_compose  p_serial_leading_stmt                             （§6 第三步）
[OK ] r1_predicate_width_f16  r2_uniform_condition  r7_reduce_parallel_pre_fallback
[REJ] c6_complex      nested serial loop inside T.Parallel is not affine-vectorizable
[REJ] s_serial_stencil  nested serial loop inside T.Parallel is not affine-vectorizable
[REJ] p_serial_one_unclean  nested serial loop inside T.Parallel is not affine-vectorizable
[REJ] c_gm            global-memory operand inside T.Parallel ...
[REJ] c4_true_random  ... FloorDiv by 3 is not a power of two
[REJ] r3_masked_reduce / r4_repeated_old_value / r5_serial_nonzero_min / r6_serial_waw
```

脚本共 **64** 个用例：**53 OK / 11 拒绝 / 0 ERR**（拒绝 = 嵌套循环 / 跨迭代 stencil / 并列段含不干净 serial / GM 操作数 / `TRUE_RANDOM` / 条件归约 / 旧值重复 / 非 0 下界 / 跨迭代 WAW）。每个用例的每根轴分类与打分都通过 `plan.diagnostics`（成功，含失败轴 `-> fallback`）或 `Unsupported.diagnostics`（失败）打印。

---

## 5. 扩展性分析

### 5.1 新增一个逐元素算子

* 目标指令已在 `simd.py`：
  * 有专用标量-向量变体 → 加进 `primitives.py` 的 `_VEC_SCALAR_BINARY`（如 `vsub→vsubs`）；
  * 普通二元/一元 → 加进 `primitives.py` 的 `_ELEMENTWISE_BINARY/_UNARY`（键取 TIR 节点名尾段小写）。
  * **发射层无需改动**（`_build` 按 recipe 查表 dispatch）。
* 目标指令是新的 `T.simd.*`：需先在 `simd.py` 注册内建，再在 `emit.py` 的 `_build` 加一个 `recipe` 分支。
* 比较 / 布尔类：改 `primitives.py` 的 `_COMPARE` / `_BOOL_BINARY` / `_BOOL_UNARY`（并注意 `_COMPARE_FLIP`）。
* **成本**：通常 1–5 行 + 1 个测试。

### 5.2 新增一个几何场景

绝大多数情况下**不需要改代码**——只要新场景的访存沿某轴线性化后 Δ 是常数，就会自动落进现有分类。此时**只需补一个测试**固化边界（这正是支持矩阵的产出方式）。

只有两种情况需要动代码：

1. **引入新的 `AccessClass`**（例如 stride 为负 / 分段仿射）→ 改 `access.py` 的 `classify_delta`（分类）+ `plan.py` 的 `rank_vector_axes`（候选/打分）+ `value.py` 的 `load_node`（建节点）+ `emit.py` 的 `_build` / `_emit_store`（发射）。
2. **引入新的语句形态** → 在 `body.py` 的 `normalize_body` 加一个分支。

### 5.3 换发射目标 / 新增硬件指令

因为 `VectorPlan` / `VectorProgram` 指令选择无关，**换 target 只改发射层**：`emit_plan` / `_Emitter._build` / `_Emitter.tail_mask` / `_store_mask` 里的指令名。分析层（§3 的规划部分）完全不动。多段路径只改变**语句编排**（逐段发射、外层轴包一次），不引入任何新的指令选择。

**成本**：与目标指令集规模成正比，与场景数量无关。

### 5.4 路线图：im2col 场景的三段拆分

im2col 本质是**三个耦合特性**，建议分段推进：

| 子特性 | 内容 | 落点 | 成本 / 风险 |
|---|---|---|---|
| (a) | `FloorDiv`/`FloorMod` 作**值算子** | **已实现**（`build_value` + `_build`）：`simd.py` **无整型 `vdiv/vmod`**（`vdiv` 是浮点 SFU），故对 **2 的幂常量除数**做精确 strength-reduction——`x//2^k → vshrs(x,k)`、`x%2^k → vand(x, 2^k-1)`。仍缺：非 2 的幂常量除数（需 `vmull` magic-number）、非常量除数（需整型内建） | 中；精度/边界敏感 |
| (b) | 非仿射 / 间接下标 → **gather / scatter** | **已实现**（改进 1 + 4）：新增「每 lane 索引向量」通道——`index_value` 在整数上下文构造绝对元素偏移，`load_node` 建 `gather` 节点，`_build` 发 `vgather2`、`_emit_store` 发 `vscatter`；`PERMUTE` / `PIECEWISE` / `INDIRECT` 三类共用该通道 | 中；已落地测试 |
| (c) | 非仿射下标的 **UB staging** | 需**新建一个 pass**：识别非仿射/GM 操作数 → 其前插入 staging copy（UB 分配）→ 改写 nest 内操作数读 staging。会牵动 buffer 分配 / 别名 / 同步 | 大；**接入管线后高风险** |

> (a) 的 2 的幂路径、(b) 的索引向量通道均已完成并落地测试（离线原型内，风险低）；仍缺 (a) 的 magic-number 分支（非 2 的幂常量除数，需 `vmull`）与 (c)。(c) 一旦接入生产管线风险高——且 im2col 是 **gather**（`vgather2` 成本高），建议先做收益评估。
>
> 注：im2col 现已能靠**轴回退**（§3.4）沿外层轴 `m` 向量化成功——上面 (a)/(c) 针对的是「沿写连续轴 `kk` 向量化」这条更优但更难的路径，(a) 的 magic-number 分支正是打通该路径的下一步。

### 5.5 新增 reduction 算子

只需在 `primitives.py` 的 `_REDUCE_PRIMITIVES` 里**加一行** `ReducePrimitive(tir_op, op, recipe, scalar, identity)`：算子白名单（`_REDUCE_OPS`）、跨 lane 指令（`_REDUCE_RECIPE`）、标量同构（`_REDUCE_SCALAR_RECIPE`）与单位元（`_reduce_identity`）全部由这张表派生；发射端 `_emit_reduce` 无需改（按 recipe 查表）。

### 5.6 与生产管线接线

当前模块是**离线原型**（仅测试引用）。接入路径建议：

1. 在 `--convert-tilelang-to-npu` 之后新增一个 Python pass，对目标 `T.Parallel` 嵌套调用 `vectorize_parallel_nests`；
2. 与 C++ 侧的并行向量化 pass 做**能力对齐或替换**时，先对齐 §4 矩阵的边界；
3. 接线后才需要考虑 pass 排序、与 UB 分配 / 同步 / alias pass 的先后关系。

---

## 6. 边界与已知局限

### 6.1 已知缺陷（均已有回归基线）

下列问题原本是**当前实现**的确定性行为，**多数为静默错误结果**。现在全部已修复，下表保留「触发场景 → 原现象」作为回归对照；对应复现用例在 `tilelang/ascend/analysis/test_drive_plan.py` 尾部（`d*`）与 `testing/ascend/analysis/test_polyhedral_vectorize.py` 末尾成对固化。

| 编号 | 触发场景 | 原现象 | 现状 |
|---|---|---|---|
| D1 | 乘积归约 `s[i] *= a[i,j]` | 静默发成求和（**结果错误**） | ✅ **已修**：显式拒绝 |
| D2 | 覆写 `s[i] = a[i,j]`（store 均匀 + 取值变化） | 判成归约（**结果错误**） | ✅ **已修**（随 D13）：显式拒绝 |
| D3 | `int32` / `float16` 的 `max`/`min` 归约 | 发射期抛 `ValueError` | ✅ **已修**：单位元按 dtype 派生 |
| D4 | 归约体含 lane 均匀加数 `s[i] += a[i,j] + 1.0` | `1.0` **被静默丢弃** | ✅ **已修**：沿脊递归剥离，均匀加数保留在归约体内 |
| D5 | 累积读不在最外层（`s[i] = s[i] + a[i,j] + 1.0`） | 漏判累加写，旧值可能被覆盖 | ✅ **已修**：递归「累积写」谓词；旧值被缩放/跨位读仍显式拒绝 |
| D6 | 一个 nest 内两条归约 / 求和+求最大混合 | **只发射第一条 store** | ✅ **已修**：逐 store 发射，各写一次 `ONEPT vsts` |
| D7 | store 混合 + 步长/索引向量装载 | 规划成功、**发射期** `cannot scalarize … 'vload_strided'` | ✅ **已修**：`is_uniform` 挡掉无 args 叶子 |
| D8 | 负步长 `Δ ≤ -2`（`a[126-2*i]`） | 无符号 ramp × 负步长 → **结果错误** | ✅ **已修**：负步长走索引向量（`vgather2`/`vscatter`），读+写两侧 |
| D9 | 符号 extent（`a: (n,)`，`n` 为变量） | 只发一个 chunk → **静默少算** | ✅ **已修**：非归约发运行时 `ceildiv` chunk 循环；归约走整块主循环 + peel 尾块 + `local.var` 累加器 |
| D10 | 非归约 + 向量轴长度非 `lanes` 整数倍 | 尾块**整 lane 越界读** | ✅ **已修**：尾块拆块 + 装载带谓词（三条 load 通道 + 归约尾块） |
| D11 | 语句外置且下标含 `BufferLoad`（如 `c[b[0]] = 1.0`） | 原先规划成功、**发射期** `Unsupported`（所有轴一刀切 `INDIRECT`） | ✅ **已支持**：间接下标只在沿轴**变化**时归 `INDIRECT`；`Δ==0` 归 `UNIFORM`，pre/post 发一次 `ONEPT`，serial 主体照常向量化；`bool` 等 store 元素类型在规划期拒绝 |
| A0 | 吸收的 serial 轴上的部分归约（`s[2*i] += a[i,j]` 与 `u[i,j] = s[2*i]` 混排） | `ONEPT` 只留 lane 0 → **只累加 2/128 项** | ✅ **已修**：规划期拒绝；单段路径自动回退到并行轴 |
| D12 | 双臂 `if` 的某一臂再嵌 `if` | `masked branch merging is not supported` | 已文档化（设计边界，非缺陷） |
| D13 | 「归约判据」原先有两套口径（分析期 vs 发射期不一致） | 与 D2/D5 同源 | ✅ **已修**：统一为「lane 均匀地址 + **同址**累加形式」单一判据 |
| R1 | f16/int8 尾块 + 条件（`pand` 固定 b32） | lane 64 之后谓词被清零 → **尾块只写一半** | ✅ **已修**：谓词运算掩码跟随数据元素宽度 |
| R2 | 循环不变 bool 条件（`if a[0] > 0: c[i] = a[i]`） | 规划成功、**发射期** `ValueError: Cannot infer default mask width for dtype 'bool'` | ✅ **已修**：store 条件走 `predicate()` |
| R3 | 条件归约（`if cond: s[0] += a[i]`） | 归约忽略 mask，**静默对全部 lane 求和** | ✅ **已修**：规划期显式拒绝 |
| R4 | 归约旧值重复出现（`s[0] = s[0] + (s[0] + a)`） | 两份旧值被折叠成一个初值 → **少算一份** | ✅ **已修**：剥离次数 ≠ 1 即拒绝 |
| R5 | 非 0 循环下界（`T.serial(1, K)`） | 下界被丢弃 → **下标整体错位** | ✅ **已修**：`_require_zero_min` 拒绝 |
| R6 | 跨迭代写写别名（`c[i,2*j]` 与 `c[i,j]`） | 标量"后写覆盖"在向量化后不成立 → **结果不定** | ✅ **已修**：`_serial_is_clean` 拒绝（展平通道同族门禁） |
| R7 | 归约段 + 非均匀 pre/post（沿并行轴） | 规划成功、**发射期** `Unsupported` | ✅ **已修**：规划期拒绝该轴并回退到 serial 轴 |
| R8 | 写后读转发在 scatter / 标量写 / 回写后仍可命中 | 可能读到写前的陈旧向量值 | ✅ **已修**：每次写都作废该 buffer 的转发项 |

> 上述用例断言**正确契约**（历史上有 `xfail(strict=True)` 标注，修复后 `XPASS` 强制同步摘除；现已全部转为正常 `pass`）。

### 6.2 设计边界与长期局限

| 类别 | 局限 | 说明 / 潜在改进 |
|---|---|---|
| 归约 | 短归约（≤4 个 chunk）展开 + 每 chunk 一条链 + 树形合并、长归约与符号 extent 走真实循环 | `tail != 0` 已支持（尾块带谓词装载 + 单位元填充）；符号 extent 归约也已支持（运行期夹紧尾块掩码）；**条件归约**明确拒绝（R3） |
| 迭代域 | 循环下界必须可证为 0（R5） | 非零下界（`T.serial(1, K)`）目前拒绝；建模 `min` 后即可支持 |
| 依赖 | 只检查吸收的 serial 轴上的携带依赖 | 跨迭代 WAW 别名（R6）按「同 buffer 多写必须同址」保守拒绝；非仿射 scatter 写的地址注入性由使用者保证（多迭代写同一地址本身是未定义行为） |
| 诊断 | 已有结构化 `diagnostics`（改进 7） | 成功 / 失败都收集全部卡点（成功路径也保留 `-> fallback` 原因）；暂无机器可读的 error code（没有消费者，待接入管线时再加） |
| 内存域 | GM 操作数拒绝 | 需 UB staging（§5.4(c)） |
| 值算子 | `FloorDiv`/`FloorMod` 作值算子仅支持 **2 的幂常量除数** | 非 2 的幂常量除数（需 `vmull` magic-number）、非常量除数（需整型内建）仍拒绝（§5.4(a)） |
| 地址 | 仅 `TRUE_RANDOM` 拒绝 | `PERMUTE`/`PIECEWISE`/`INDIRECT` 已走索引向量通道（`vgather2`/`vscatter`）；不可构造的索引明确报错，不静默降级 |
| 语句 | 发散分支、atomic 拒绝；serial 支持面有限 | 直接内嵌、链式嵌套、或「一个/多个并列干净 serial 循环夹在语句序列中」可吸收；含循环携带状态、并列段中有一段不干净、非直接内嵌者拒绝（见 §3.3 / §6） |
| 代码 | ~~`vload.is_uniform` 分支等价于 `False`~~ | ✅ **已删**：`is_uniform` 统一在递归前挡掉所有无 args 叶子（`vload`/`vload_strided`/`gather`），见 D7 |
| 代码 | ~~`AccessInfo._uniform` 动态挂属性~~ | ✅ **已删**：随 `is_uniform` 统一而移除 |
| 代码 | ~~累加器提升在 plan 与 emit 各算一遍~~ | ✅ 已收敛为 `StorePlan.promote`（计划期单一决策） |
| 代码 | ~~`_REDUCE_CHAINS` 与展开阈值相等 → 链内累加分支不可达~~ | ✅ 已删（行为逐指令不变） |
| 代码 | ~~死参数 / 死字段（`is_global_ok`、`_ValueContext.lanes`、`is_index_vector` 等）~~ | ✅ 已删除；`_split_nest` / `_try_build_program` 共用 `_absorb_parallel_chain` |
| 判定 | ~~取 extent 时未传 `analyzer`~~ | ✅ 已补（`_build_plan_for_axis`）；严格符号 extent 仍由 D9 的运行时 chunk 循环处理 |
| 模型 | 只做连续性判定，无完整依赖分析 / 调度 | 迭代域已建模，可扩展到依赖 / 调度 |

---

## 7. 测试策略

* **离线、与硬件/编译器无关**：每个用例构造 TileLang IR，直接跑 `build_plan`（恒返回 `VectorProgram`）/ `emit_plan`，检查 `program.segments[*]`（单段用例取 `segments[0]`）与发射出的 `tl.simd.*` 调用序列。不涉及设备执行。
* **分组**：数学核心 6 项 + §0–§6 场景 + 诊断 + 语句树变换 + 统一契约/等价性护栏 + 已知缺陷回归（D1–D13 / A0） + 设计层审计回归（R1–R7） + 段间共享/寄存器直通门禁 2 项，共 **108** 个用例（其中 D12 为特征化 `pass`）。
* **已知缺陷基线**：缺陷用例断言**正确契约**（历史上有 `xfail(strict=True)`）；现在全部已修、标记已摘除，运行预期 `108 passed`。
* **辅助工具**：
  * `_first_parallel`：取**最外层** `T.Parallel`（后序遍历先出最内层，故取最后一个）。
  * `_single_segment`：断言 `build_plan` 恒返回 `VectorProgram` 且恰 1 段，取回 `program.segments[0]`（单段用例经它访问段级字段）。
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
| `AccessClass` | 沿某轴的访存形态（连续/均匀/步长/逆序/分片/间接/无结构） |
| 轴回退 | 按打分排序候选轴后**逐个尝试**发射，某个轴发不出时自动退到次优轴（§3.4） |
| 段 / `VectorProgram` | `T.Parallel` 体内一个干净 serial 链 + 其 `pre`/`post`（一个 `VectorPlan`）；多个并列 serial 时按源码顺序组成 `VectorProgram`，逐段独立向量化 + 源序发射（§3.11） |
| 每 lane 索引向量 | 元素偏移 `offset(..., v=lane)` 的整数向量，供 `vgather2`/`vscatter` 使用（改进 1） |
| `recipe` | 值 DAG 节点的原语名（`vload`/`vmuls`/`vsel` ...），发射时查表成 `T.simd.*` |
| `is_uniform` | 子表达式是否 lane 内均匀（决定标量-向量特化与 `vdup`） |
| lane | 向量寄存器中的一个元素槽位 |
| `vreg` | 向量寄存器，2048 位；`lanes = 2048 / elem_bits` |
| 谓词 / mask | `boolx256` 类型，CCE 的 P 寄存器，`bits ∈ {b8,b16,b32,b64}` |
| NORM / ONEPT / BRC | store 距离：普通连续 / 单元素 / 广播 |
