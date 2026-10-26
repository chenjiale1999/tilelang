# Ascend 统一多面体向量化 —— 设计文档

> 对应实现：[`polyhedral/`](./polyhedral)（10 个模块，共约 4637 行；最大 `plan.py` 1352 行）
> 对应测试：[`testing/ascend/analysis/test_polyhedral_vectorize.py`](../../../../testing/ascend/analysis/test_polyhedral_vectorize.py)（135 个用例）
> 需求基准：[`Parallel_ascend.md`](../../../../Parallel_ascend.md) §315–484「支持场景」
> 参考实现（发射风格权威）：`src/ascend/transform/ascend_simdvf_lower_parallel.cc`

***

## 0. 文档定位

本文档是 `polyhedral` 包的设计说明，供以下场景使用：

- **学习**：理解模块的数学内核、数据结构与发射逻辑；
- **扩展**：新增算子 / 场景 / 硬件指令时，知道应该改哪一层；
- **评审**：界定「支持 / 拒绝」的边界及其理由，避免静默降级。

本文档描述的是**离线分析原型**：`polyhedral` 包目前只被它自己的测试引用，尚未接入生产编译管线（见 §5.6）。

***

## 1. 背景与动机

### 1.1 现状：case-by-case 识别

Ascend 现有的 `T.Parallel` 向量化在 C++ 侧按「代码形状」逐条分支识别：

- `CanLowerBody` 只接受 `BufferStore` 及其 `SeqStmt`（不认 `IfThenElse`/`Select`）；
- `DecomposeExpr` 逐算子判断维度 / stride；
- 归约、广播、多语句各有专门逻辑。

问题在于：**几何上等价的场景写法千差万别**，每加一种就要加一个分支，分支之间还会互相干扰。

### 1.2 目标：单一数学判据

把「识别代码形状」替换成「对访存地址沿候选向量轴求有限差分」：

| 经典做法（case-by-case） | 本模块（统一数学分析）          |
| ------------------ | -------------------- |
| 逐条匹配节点形状           | 建立 SCoP：迭代域 + 访存仿射函数 |
| 对每个算子判断维度 / stride | 对每个访问算「沿轴的有限差分」Δ     |
| 按代码形态分支            | 按 Δ 的取值分类，一条规则覆盖所有形态 |

核心洞察：`c[i]=a[i]`、`c[i]=a[2*i]`、`c[i]=a[0]`、`h[i//16, i%16]`、`d[i,j]=g[i]-g[j]` 这些几何上完全不同的写法，**沿轴求差分后落到同一个标量集合** `{1, 0, k, 非常数}` 上。

***

## 2. 架构设计

### 2.1 编译流水线中的位置

```
┌───────────────────────────────────────────────────────────────────┐
│ 分层方案（Parallel_ascend.md）                                     │
│                                                                   │
│  P1  LiftParallelToArray      数组化：把 T.Parallel 降成数组 IR    │
│  P2  AscendArrayExecPlan      规划：连续轴 / 向量化策略            │
│  P3  AscendLowerArrayPlan     发射：T.simd.* 指令                  │
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

`VectorPlan` / `VectorProgram` 里**没有任何硬件指令名**——只有轴、lane 数、chunk、每个 store 的值 DAG。**指令选择推迟到** **`emit_plan`**。这条「指令选择无关中间表示」的设计使换发射目标（如从自定义 `tl.vmi.*` 改为 `T.simd.*`）只需改发射层。

`build_plan` **恒返回** `VectorProgram`：单段（一个主体 + 可选 `pre`/`post`）时它是「1 段 program」，`program.axes` 只含提到最外层包一次的**并行轴前缀**（`_outer_axes`），其余轴仍由该段自己包裹；`T.Parallel` 体内出现**多个并列的干净** **`T.serial`** **循环**时，program 含多段（每段一个 `VectorPlan`，§3.11）。`VectorPlan` 因此是**段级**中间表示，`VectorProgram` 才是 `build_plan` 对外的统一返回类型。

### 2.2 模块分层

包内按**依赖单向向下**拆成 9 个职责模块（公共 API 由 `__init__.py` 统一再导出）：

```
polyhedral/
├─ primitives.py   0. 原语事实表   _ELEMENTWISE_BINARY/_UNARY, _VEC_SCALAR_BINARY, _COMPARE,
│                                  _BOOL_BINARY/_UNARY, _REDUCE_PRIMITIVES（含单位元）,
│                                  _MASK_DTYPE/_MODE...
│                                  —— ISA/算子事实与硬件常量的唯一来源
│                                  + Unsupported + 语句/表达式小工具
├─ access.py       1. SCoP：迭代域  Axis, AccessClass
│                  2. 仿射原语      linear_element_offset, finite_difference, classify_delta
│                  3. 访问分类      AccessInfo, classify_access
├─ body.py         4. body 规范化   StoreSpec, normalize_body, _collect_aliases/_apply_aliases
├─ value.py        5. 值 DAG        ValueNode, build_value, broadcast, load_node, _build_call,
│                                  装载守卫 + _guard_free_bounds（守卫恒真首/尾 chunk 区间）
├─ model.py        6. 计划数据结构  StorePlan, ReduceInfo, VectorPlan, VectorProgram
├─ plan.py         7. 规划          _AXIS_WEIGHT, rank_vector_axes, build_plan（含多段 program）,
│                                  _plan_store（谓词→守卫）, _guard_cuts（chunk 切点）
├─ emit.py         8. 发射          _Emitter, emit_plan, _emit_reduce
├─ render.py       9. 可读渲染      render_plan
├─ driver.py       10. 语句树变换   vectorize_parallel_nests
└─ __init__.py      公共 API 再导出 + 整体设计说明
```

> 依赖方向：`primitives ← access ← {body, value} ← model ← plan ← {emit, render} ← driver`。
> `model` 只放共享数据结构，因此 `emit` / `render` 只依赖计划类型而不依赖规划算法；
> 任意模块都**不得**反向导入上层，避免循环依赖。
>
> 改动指引：新增算子配方改 `primitives`；新增访存形态改 `access`；新增控制流形态改
> `body`；新增取值算子改 `value`；新增计划字段改 `model`；新增轴选择 / 归约规则改
> `plan`；换硬件指令目标改 `emit`。

### 2.3 数据结构总览

| 结构              | 含义                                | 关键字段                                                                                           | 产出者                                  |
| --------------- | --------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------ |
| `Axis`          | 一根迭代轴 `0<=var<extent`（并行轴或被吸收的 serial 轴） | `var, extent, index, kind`                                                                     | `_split_nest` / `_absorb_serial_chain` |
| `AccessClass`   | 沿某轴的访存形态（枚举）                      | `CONTIGUOUS/UNIFORM/STRIDED/PERMUTE/PIECEWISE/INDIRECT/TRUE_RANDOM`                            | `classify_delta` + `classify_access` |
| `AccessInfo`    | 一次访存 + 逐轴分类                       | `buffer, indices, offset, is_store, per_axis, indirect, in_global, reason`                     | `classify_access`                    |
| `StoreSpec`     | 一条写                               | `buffer, indices, value, mask`                                                                 | `normalize_body`                     |
| `ValueNode`     | 值 DAG 节点                          | `recipe, dtype, args, access, scalar, detail, predicate, predicate_free_until/from`           | `build_value`                        |
| `StorePlan`     | 一条写的完整计划                          | `spec, access, value, mask, kind, index, reduce, promote, mask_free_until/from`                | `_build_plan_for_axis`               |
| `ReduceInfo`    | 归约信息（逐 store）                       | `axis, op, recipe, varying, accumulate`                                                        | `_classify_reduce` → `_plan_reduce`  |
| `VectorPlan`    | **段级**指令选择无关计划（段 = 主体 + 可选 `pre`/`post`） | `axes, vector_axis, lanes, chunks, tail, stores, pre_stores, post_stores, reduce, guard_boundaries, diagnostics` | `_build_plan_for_axis`（每段一个）      |
| `VectorProgram` | 统一计划（1..N 段，`build_plan` 的返回类型）      | `axes`（提层的并行轴前缀）, `segments`, `diagnostics`                                              | `build_plan`                         |

`AccessClass` 分两层判定：Δ 层给常数形态，非常数再做**结构细化**（改进 4）：

| `Δ_v(off)`       | `AccessClass` | 后续动作                                        |
| ---------------- | ------------- | ------------------------------------------- |
| `1`              | `CONTIGUOUS`  | `vld` / `vsts`（NORM）                        |
| `0`              | `UNIFORM`     | 标量读一次 + `vdup` 广播                           |
| `k,\ \|k\|>1`    | `STRIDED`     | `vgather2` / `vscatter` + 步长索引斜坡            |
| `-1`             | `PERMUTE`     | 每 lane 索引向量 → `vgather2` / `vscatter`       |
| 非常数且索引向量可构造      | `PIECEWISE`   | 每 lane 索引向量 → `vgather2` / `vscatter`       |
| 下标含 `BufferLoad` | `INDIRECT`    | 每 lane 索引向量（来自寄存器）→ `vgather2` / `vscatter` |
| 非常数且不可构造         | `TRUE_RANDOM` | 明确拒绝                                        |

### 2.4 主数据流

```
build_plan(stmt)
  ├─ _try_build_program(stmt, analyzer, lanes, notes) ▶ VectorProgram?   # 多个并列 serial 段（§6 第三步）
  │     └─ 命中 ≥2 个干净 serial 链 → _build_segment（每段一个 VectorPlan）
  │        任一段构造失败 → 段级原因写入 notes 并 return None（缺陷 B：不吞掉）
  ├─ _split_nest ▶ axes[], serial_axis, pre, main, post   # SCoP 迭代域（单段路径，回落）
  ├─ _prepare_segment(axes, ..., pre, main, post) ▶ _SegmentSpecs
  │     = normalize_body + lanes 推导 + 逐轴 classify_access（单段/多段共用）
  ├─ rank_vector_axes(axes, stores, store_infos, load_infos) ▶ 轴排行（联合可行性 + 读写加权）
  ├─ for vector_axis in ranked:      # 轴回退：逐个尝试
  │     try: 构造 _ValueContext(axes, vector_axis, lanes, analyzer)
  │           for each store: build_value(spec.value, guard=store 谓词) ▶ ValueNode
  │           （守卫先按 full-range 可证安全过滤，再算首/尾恒真 chunk 边界）
  │           ▶ VectorPlan（含 guard_boundaries）
  │     except Unsupported: continue
  ▶ VectorProgram(axes=_outer_axes(...), segments=[plan])        # 单段包成 1 段 program

emit_plan(plan)
  ├─ VectorProgram → 逐段 _emit_segment_stmts（源序），program.axes 统一包一次（段内按变量身份跳过）
  ├─ 单独 VectorPlan → 等价的「自己包全部非向量轴」发射路径（护栏测试比对两者逐节点一致）
  ├─ reduce is None → _strip_mine 按 guard_boundaries 重建 chunk 循环/分段 + 外层 serial 循环
  │                   （_Emitter 逐 store 发射；守卫恒真段回退 vld，§3.7.1）
  └─ reduce != None → _emit_reduce：`chunks <= 4` 编译期展开 + 多路累加器；否则真实循环 + `local.var` 累加器
  ▶ TIR (SeqStmt / For)
```

***

## 3. 算法设计

### 3.1 SCoP 建模

**(a) 迭代域** —— `_split_nest(stmt, analyzer)` 从某个 `T.Parallel` For 出发，**连续吃掉**并行轴；遇到**直接内嵌且干净**的 serial 轴（`_serial_is_clean`）也一并吸收。返回 `(axes, serial_axis, pre_stmts, main_stmt, post_stmts)`：

```python
while isinstance(current, tirx.For):
    if current.kind == tirx.ForKind.SERIAL:
        if not _serial_is_clean(current, analyzer)[0]:
            break                                  # 有循环携带依赖 → 不吸收，留给 normalize_body 拒绝
    elif current.kind != tirx.ForKind.PARALLEL:
        break
    axis = Axis(current.loop_var, current.extent, len(axes), current.kind)
    axes.append(axis)
    if current.kind == tirx.ForKind.SERIAL:
        serial_axis = axis
    current = current.body

# 平行链结束后若是 SeqStmt，且其中恰好一个干净 serial 循环：
#   → 前兄弟语句 = pre、循环体 = main、后兄弟语句 = post（§6 第二步）
```

> 推论：**必须从嵌套头调用**才能拿到全部轴。这是 `vectorize_parallel_nests` 采用自顶向下遍历、测试 helper `_first_parallel` 取最外层的原因。
>
> **下界必须为 0**（`_require_zero_min`）：`Axis` 只记录 `(var, extent)`，发射时按
> `For(var, 0, extent)` 重建。`T.serial(1, K)` / `T.Parallel(1, K)` 的非零下界会被静默
> 丢掉、变成从 0 开始的错位下标（R5），因此无法证明下界为 0 时直接拒绝；`_split_nest`
> 与 `_absorb_serial_chain` 在吸收前都做这项检查。
>
> 注意：并行链会**先**交给 `_try_build_program`（§3.11）——若体内含多个并列的干净 serial 循环，则整体走多段路径；只有不适用时才回落到本节这条单段路径（`_split_nest` + `serial_axis` + `pre`/`post`）。两条路径共用 `_absorb_parallel_chain` 吸收并行前缀。

**(b) 访存函数** —— `linear_element_offset` 把多元下标线性化成元素偏移：

```python
off = elem_offset + Σ_d idx_d · stride_d      # 行主序 stride（优先用 buffer.strides）
return analyzer.simplify(off)
```

两个要点：

- **行主序紧凑 stride** 藏在 `_compact_strides` 里；`buffer.strides` 非空时优先，兼容下游 padding/alias。
- **立刻** **`simplify`**。`h[i//16, i%16]` 线性化得 `(i//16)*16 + (i%16)`，用恒等式 `x//m*m + x%m == x` 折成 `i`。**高维摊平不需要专门规则，靠的是符号化简器。**

### 3.2 核心数学：有限差分判据

```python
def finite_difference(offset, var, analyzer):
    shifted = substitute(offset, {var: var + 1})   # off(var+1)
    return analyzer.simplify(shifted - offset)      # 再化简一次
```

**差分之后必须再化简一次**：`off` 已化简，但 `off(v+1) - off(v)` 会重新产生 `(v+1)//m*m + (v+1)%m - v` 这类式子，只有再化简才能压回常数。

```python
def classify_delta(delta, analyzer=None):          # Δ 层：只分常数 / 非常数
    value = _const_int(delta, analyzer)   # 能否（符号）证明成常数？
    if value is None: return PIECEWISE    # 折不动 → 待结构细化
    if value == 0:    return UNIFORM
    if value == 1:    return CONTIGUOUS
    if value == -1:   return PERMUTE
    return STRIDED, value

def classify_access(buffer, indices, axes, analyzer, *, is_store):   # 结构细化层（改进 4）
    offset = linear_element_offset(buffer, tuple(indices), analyzer)
    info   = AccessInfo(buffer=buffer, indices=tuple(indices), offset=offset, is_store=is_store, ...)
    indirect = any(_contains_buffer_load(idx) for idx in indices)
    reason   = None if indirect else index_vector_reason(offset, analyzer)
    for axis in axes:
        kind, stride = classify_delta(finite_difference(offset, axis.var, analyzer), analyzer)
        if indirect:
            # 沿轴不变（Δ==0）的间接地址是 lane 内均匀的标量地址 → 仍归 UNIFORM（D11）
            if kind != AccessClass.UNIFORM:
                kind, stride = AccessClass.INDIRECT, None
        elif kind == AccessClass.PIECEWISE and reason is not None:
            kind, stride = AccessClass.TRUE_RANDOM, None     # 索引向量构造不出来 → 拒绝
            info.reason = reason
        elif kind == AccessClass.STRIDED and stride < 0 and reason is not None:
            kind, stride = AccessClass.TRUE_RANDOM, None     # 负步长必须走索引向量（D8）
            info.reason = reason
        info.per_axis[axis.index] = (kind, stride)           # 键 = 轴在迭代域中的位置
    return info
```

> **设计精髓**：`_const_int(delta) is None` 把「仿射 vs 非仿射」的判定**外包给符号化简器的能力边界**——化简器能折成常数就是仿射，折不动就是非常数。这是「把语义问题转成符号计算问题」的典型手法，既优雅也有风险：**化简器的行为就是接口契约**（因此 `finite_difference` 前后各 `simplify` 一次）。
>
> `_const_int` 本身也不只认字面量 `IntImm`：给了 `analyzer` 时先 `simplify`，再用 `const_int_bound`——区间是可靠的，因此**上下界相等即证明为常数**，这样「化简后仍是常量表达式」的 Δ 也会被判成常数。

**非常数 Δ 的结构细化（改进 4）**：

- 下标含 `BufferLoad`（`A[B[i]]`）→ `INDIRECT`，索引向量来自被装载的寄存器；
- 否则用 `index_vector_reason(offset)` 检查元素偏移能否在索引向量通道构造（只允许整数算术；`FloorDiv`/`FloorMod` 除数须为**正的 2 的幂常量**；不许再含 `BufferLoad`）——可构造 → `PIECEWISE`，不可构造 → `TRUE_RANDOM`（带诊断原因）。
- `Δ == -1`（`A[N-1-i]`）是**双射逆序**，单独标 `PERMUTE`，与 `PIECEWISE` 共用索引向量通道。
- 三类（`PERMUTE` / `PIECEWISE` / `INDIRECT`）都由同一个「每 lane 索引向量」通道发射成 `vgather2` / `vscatter`，区别只在诊断标签。

### 3.3 body 规范化

`normalize_body` 把任意语句结构压成**有序** **`StoreSpec`** **列表**，从而「多语句天然支持」：

| 输入结构                    | 处理                                                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `SeqStmt`               | 顺序展开                                                                                                                           |
| `BufferStore`           | 一条写                                                                                                                            |
| `IfThenElse` 单臂         | 谓词化：`mask &= cond`（谓词同时作为取值表达式的**装载守卫**，§3.12.3）                                                                                                      |
| `IfThenElse` 双臂、**同址写** | 合并成**惰性** `if_then_else(cond, a, b)`（不是急切求值的 `Select`，两臂访存各自带条件守卫）                                                                              |
| `IfThenElse` 双臂、异址写     | `Unsupported`（发散分支，方案 §8.1）                                                                                                    |
| 裸 `BufferLoad`          | 丢弃（无副作用）                                                                                                                       |
| `Evaluate(atomic_*)`    | `Unsupported`（§5）                                                                                                              |
| 内层 `For`                | `Unsupported`（§6）——注意：**直接内嵌 / 链式嵌套 / 并列的干净 serial 循环在进入本函数之前已被吸收成迭代轴**，只有吸收不掉的（跨迭代状态、stencil、并列段中有一段不干净、嵌套 `T.Parallel`）才落到这里 |
| `Bind`（标量别名）            | 丢弃（已内联）                                                                                                                        |

配套**标量别名内联**：`_collect_aliases` 收集 `Bind` 引入的真值绑定（排除 buffer 声明用的 Bind），`_apply_aliases` 迭代替换（**限定 8 次**防别名环），把 `k = k_iter*BK + j`、`access_h = ...` 这类中间量折叠进最终表达式。这是 §3「多行语句」收敛成单表达式再向量化的关键。

### 3.4 向量轴选择

`rank_vector_axes` 是**纯打分 + 轴回退**，无一处匹配代码形状：

1. **联合可行性**：某轴上只要有 store 是 `TRUE_RANDOM`，或**所有** store 都写不进去（无候选），该轴出局；`STRIDED` / `PERMUTE` / `PIECEWISE` / `INDIRECT` 的 store 都能进候选（靠 `vscatter`）。
2. **读写加权**：`score = 100 * Σ store权重 + 10 * Σ load权重 + axis.index`，权重 `CONTIGUOUS=8 > PIECEWISE=4 > STRIDED=PERMUTE=3 > INDIRECT=2 > UNIFORM=1 > TRUE_RANDOM=0`。
   - `*100 / *10` 让写侧成为主序、读侧次之（避免「写连续、读四处乱跳」的轴被误选）；
   - `+axis.index` 让同分取**更内层**（更内层通常更连续）。
3. **轴回退**：返回**排序后的整条轴列表**，`build_plan` 依次尝试；某个轴的取值发射失败时自动退到次优轴，而不是整段判死。
4. **归约不是独立的选轴入口**：`rank_vector_axes` 只看「写侧是否 true-random」与加权分，不做归约特判；归约由 `_classify_reduce`（§3.5）在**选定轴之后**判定。lane 均匀的归约 store 权重只有 1，通常会让位给写连续的轴；只有当所有候选 store 都沿该轴均匀、且取值变化时，该轴才成为归约轴。归约轴若在 build 期过不了「累加形式 / 条件归约 / pre-post 均匀性」门禁，仍会**回退到下一根轴**（R7 即此路径）。

**为什么"打分"和"回退"要分两阶段**：打分只能基于**分析期的形态**（Δ 的分类），但一个轴最终能否发射还取决于**发射期**才能确定的细节——元素偏移能否在「每 lane 索引向量」通道构造出来（如非 2 的幂常量除法就构造不出）、值算子是否有 strength-reduction 等。因此打分只作"优先级"，真正的可行性用 `try/except` 在发射期兜底。**成功路径也保留每根失败轴的诊断**（`axis '<name>': ... -> fallback`），这样"为什么没选分数最高的轴"在最终计划上仍然可见。

**例：一次真实的轴回退（`c3_im2col`** **取数）** —— nest 为 `for m in parallel(16): for kk in parallel(12)`，写 `data_shared[m, kk] = select(in_bound, data[m//64, access_h, access_w, kk%4], 0)`，其中 `access_h = m%64//8 + kk//12 - 1`：

| 轴       | 写 `data_shared[m, kk]` | 读 `data[...]`                              | 打分                         |
| ------- | ---------------------- | ------------------------------------------ | -------------------------- |
| `kk`（内） | Δ=1 → `CONTIGUOUS`(8)  | 下标含 `kk//12`（÷12 非 2 的幂）→ `TRUE_RANDOM`(0) | 100·8 + 10·0 + 1 = **801** |
| `m`（外）  | Δ=12 → `STRIDED`(3)    | `m//8`/`m%8` 可构造 → `STRIDED`(3)            | 100·3 + 10·3 + 0 = **330** |

`kk` 因"写连续"拿最高分、被优先尝试；但发射它的读侧需要构造每 lane 索引向量，`kk//12` 的除数 12 不是 2 的幂、构造不出，`_build_plan_for_axis` 抛 `Unsupported`。`build_plan` 捕获后**回退到次优轴** **`m`**：写侧用 `vscatter`（步长 12 的绝对偏移索引向量）＋读侧步长 4 装载＋谓词 `vsel`＋尾块掩码 `update_mask(16)`，成功发射。对应诊断（改进 7）：

```text
axis 'm'[0]:  store=['strided']  load=['strided']      -> candidate (score=330)
axis 'kk'[1]: store=['contiguous'] load=['true_random'] -> candidate (score=801)
axis 'kk': FloorDiv by non-power-of-two constant 12 is not strength-reduced yet (§5.4(a))
selected vector axis 'm' (score=330)
```

> 对比：C++ 参考实现里「识别归约」是专门逻辑；这里它只是打分表的正交维度——**先选轴，再问这个轴是不是归约轴**。

### 3.5 归约识别

条件（同时满足）：

- 该轴所有 store 都 `UNIFORM`（`Δ_store = 0`）；
- 取值沿该轴**变化**（`_value_varies_along`：自由变量出现 **或** 被访问 buffer 沿轴 `CONTIGUOUS/STRIDED`）。

命中则构造 `ReduceInfo(axis, op, recipe, varying, accumulate)`，其中 `op` 由取值的最外层算子在 `_reduce_op`（`plan.py`）里挑（表源是 `primitives._REDUCE_PRIMITIVES`）：

| 最外层算子 | `op` | `recipe = _REDUCE_RECIPE[op]` |
| --- | --- | --- |
| `Add` | `vadd` | `vcadd` |
| `Max` | `vmax` | `vcmax` |
| `Min` | `vmin` | `vcmin` |
| **其它（含 `Mul`）** | **拒绝**（`Unsupported`） | — |

**归约轴不是独立入口**：`_classify_reduce` 在**选定轴之后**判定归约并填充 `StorePlan.reduce`；
若该轴在 build 期过不了下面的门禁，`build_plan` 照常**回退到下一根轴**（R7）。

> ✅ **条件归约（R3，已修：改为拒绝）**：`if cond: s[0] += a[i]` 的用户谓词旧实现会被
> `_emit_reduce_chunk` **静默丢弃**（`vcadd/vcmax/vcmin` 的掩码只承载尾块谓词），于是对
> 全部 lane 求和——静默错误结果。现在 `_plan_reduce` 在 `store.mask is not None` 时显式
> 拒绝（`conditional reduction ... guard would be silently dropped`）。未来可用
> `vsel(varying, identity, cond)` + 合并掩码实现。
>
> ✅ **旧值重复出现（R4，已修：改为拒绝）**：`s[0] = s[0] + (s[0] + a[i,j])` 里旧值出现
> 两次，`_strip_reduce` 沿归约脊会把两处都剥掉、只留一个初值 → 丢掉一份旧值。现在
> `_strip_reduce` 统计**剥离次数**，`> 1` 即拒绝（`old value ... appears more than once`）。
>
> ✅ **归约段 + 非均匀 pre/post（R7，已修：build 期拒绝该轴）**：沿并行轴归约时，pre/post
> 由 `_emit_outer_store` 发射并要求 lane 均匀；旧实现只在「沿 serial 轴」分支检查这条门禁，
> 于是 build OK、emit 抛 `Unsupported`。现在 `_classify_reduce` 成功后再校验一次，该轴失败
> 后自动回退到 serial 轴。

> ✅ **缺陷 D1（已修）**：`_reduce_op`（`plan.py`）原先把未识别算子一律默认成 `vadd`，于是
> `s[i] = s[i] * a[i,j]` 这类**乘积归约**被静默发射成 `vcadd`（求和），得到**错误的数值结果**。
> 现改为查 `_REDUCE_OPS` 表，未命中即抛
> `unsupported reduction operator(s) <Node> (方案 §2 Case II): only Add/Max/Min reductions are supported`。
> 该拒绝经 `_plan_reduce`（由 `_classify_reduce` 在 build 期调用）抛出，若无其它可向量化轴则整段拒绝。
> 回归：`test_defect_d1_product_reduction_is_not_emitted_as_sum`（已摘 `xfail`）。
>
> ✅ **缺陷 D2（已修，随 D13 统一判据一并修）**：覆写 `s[i] = a[i,j]`（last-write-wins）现在由统一的
> 归约判据拒绝——目标地址 lane 均匀但取值不读 `dst`，不是累加形式 → 抛
> `... does not read 's': only the accumulate form 'dst[...] = dst[...] <op> x' ...`。
> 回归：`test_defect_d2_plain_overwrite_is_not_treated_as_reduction`（已摘 `xfail`）。
>
> ✅ **缺陷 D13（已修）**：原先 `_detect_reduce` 的判据是「store 沿轴均匀 + 取值沿轴变化」，与发射期
> 真正使用的 `_is_accumulate`（靠「DAG 里是否出现同 buffer 的 `uniform_load`」）**口径不一致**，两处
> 可能给出不同结论。现删除 `_detect_reduce`/`_is_accumulate`/`_varying_operand`，改为**单一判据**：
>
> * 第一层 `_is_reduce_axis`（`plan.py`）——「store 全部 lane 均匀 + 取值沿轴变化」，由
>   `_classify_reduce`（建计划）使用（选轴阶段不再特判归约，见 §3.4 第 4 条）；
> * 第二层 `_plan_reduce`（`plan.py`）——把单条 lane 均匀 store 精化为累加形式 `dst[...] = dst[...] <op> x`
>   （`op` ∈ Add/Max/Min），旧值必须作为归约算子的**直接操作数**且**与写入地址同址**（`_strip_reduce`
>   用 `_same_indices` 校验），否则显式拒绝。
>
> 该判据在 **build 期**完成（`_build_plan_for_axis` 末尾调用 `_classify_reduce`），因此「build 成功 ⇒
> emit 成功」不再依赖发射期兜底。回归：`test_defect_d13_lane_varying_overwrite_is_not_a_reduction`、
> `test_defect_d13_stencil_shifted_read_is_not_an_accumulation`、`test_defect_d5_scaled_old_value_is_rejected`。

### 3.6 值 DAG 构造与 `is_uniform`

`build_value` 是**结构无关的递归**，仍只用 Δ 决策：

```
BufferLoad → classify_access → load_node：
    Δ=1 → vload            Δ=0 → uniform_load
    Δ=k → vload_strided    PERMUTE/PIECEWISE/INDIRECT → gather（每 lane 索引向量）
    TRUE_RANDOM → 拒绝（带 index_vector_reason 诊断）
Var → 向量轴? lane_index(vci) : imm（外层轴变量 / capture）
IntImm/FloatImm → imm
Call → _build_call（比较/布尔/二元/一元/if_then_else）
Select → _select_node（vsel）
Cast → vcvt 或 uniform
其它 → Unsupported
```

**第二核心谓词** **`ValueNode.is_uniform`**（是否 lane 内均匀、不需要向量寄存器）：

```python
# 实现（value.py ValueNode.is_uniform，D7 已修）
if recipe in ("lane_index", "vbrc"): return False
if recipe in _UNIFORM_RECIPES:      return True     # imm / uniform_load / uniform_expr
if not self.args:                   return False    # ← D7 修复：所有无 args 叶子（vload/vload_strided/gather）
return all(arg.is_uniform for arg in self.args)     # 递归传播
```

> ✅ **缺陷 D7（已修）**：`all([]) == True`，故**任何没有被上面分支拦下的「无 args 叶子」都会被真空判为均匀**。
> 旧实现的防护清单只覆盖 `lane_index`/`vbrc`/`vload`，遗漏了同样是**无参叶子**的
> `vload_strided`、`gather`（见 §3.6 的 `build_value` 分支：这两类把访问信息放在 `access`/`index_value` 上，`args` 为空），
> 于是 `STRIDED`/`PERMUTE`/`PIECEWISE`/`INDIRECT` 的装载节点被当作标量喂进
> `vadds/vmuls/vmaxs/vmins`，在**发射期**抛 `cannot scalarize vector value 'vload_strided'`——
> 即 **`build_plan` 成功但 `emit_plan` 失败**的契约不一致（探针 M）。
> 现改为「`_UNIFORM_RECIPES` 之后、递归之前」统一挡掉 `not self.args`，
> 顺带删掉了等价的 `vload` 特判与为它服务的 `AccessInfo._uniform` 动态属性（见 §3.6）。
> 回归：`test_defect_d7_build_success_implies_emit_success_for_mixed_stores`（已摘 `xfail`）。


它驱动：

- **标量-向量特化**（`_elementwise`）：`vadd→vadds`、`vmul→vmuls`、`vmax→vmaxs`、`vmin→vmins`；无专用变体则先 `broadcast(vdup)` 再走向量-向量。
- **比较特化**（`_compare`）：一侧均匀 → `vcmps`；`vcmps` 要求 `(vector, scalar)` 顺序，交换操作数时谓词**必须翻转**（`lt↔gt`、`le↔ge`）。
- 是否需要 `vdup`；归约的累加写判定。

> ⚠️ **必须理解的坑**：`all([]) == True`。叶子节点若不在 `_UNIFORM_RECIPES` 里会被**误判为均匀**——这正是上面 D7 的成因，防护必须覆盖**全部**无参叶子（`lane_index`/`vbrc`/`vload`/`vload_strided`/`gather`）。

`Select` 的处理（`_select_node`）有两种条件来源，汇入同一 `vsel`：

| 条件来源                        | 判定                            | 发射                                           |
| --------------------------- | ----------------------------- | -------------------------------------------- |
| **逐 lane**（随向量轴变化）          | `args[0].is_uniform == False` | 直接作 `vsel` 的 mask（谓词寄存器）                     |
| **循环不变**（沿向量轴均匀，§1 Case II） | `args[0].is_uniform == True`  | 广播成全 lane 谓词 `update_mask(cond ? lanes : 0)` |

若条件与两臂**全部均匀**，整个 select 折叠成 `_uniform`（一次标量求值 + `vdup`），不产生 `vsel`。

**急切 / 惰性必须区分**（D16）：

- 源码里的 `tirx.Select`（以及 `T.if_then_else` **内建**）按 TIR 语义处理：`tirx.Select`
  **急切**求值，两臂都构造成值 DAG，不加守卫；`T.if_then_else` 是**惰性**内建，两臂分别在
  `cond` / `¬cond` 守卫下构造。
- **语句级 `if/else`**（`IfThenElse`）是惰性的。`normalize_body` 合并同址两臂时构造的是
  `tirx.op.if_then_else(cond, a, b)`（`Call`，不是 `tirx.Select`），因此走 `_build_call`
  的惰性分支：真/假臂的访存各挂自己的条件守卫。若合并成急切 `Select`，`if i >= 32:
  c[i] = a[i-32]` 会变成无条件 `vld a[-32]`（D16）。

**装载守卫的最终落地**（`load_node` → `_Emitter.load_guard`）：

1. `ctx.guard` 为空 → 普通 `vld`；
2. `_access_provably_in_bounds(info, ctx)` 能在**整个迭代域**上证明地址安全 → 直接丢掉守卫，
   仍发 `vld`（常见的 `select(cond, a[i], b[i])` 不退化）；
3. 否则挂钩守卫，并额外算出守卫**恒真**的首/尾 chunk 区间
   （`_guard_free_bounds` → `ValueNode.predicate_free_until/from`，§3.7.1）：

```text
guard_free_until = 最大 u：axis.var ∈ [0, u*lanes) 上 can_prove(cond)
guard_free_from  = 最小 c：axis.var ∈ [c*lanes, extent) 上 can_prove(cond)
```

  两个端点都还要通过 `_access_provably_in_bounds(info, ctx, hi=u*lanes)` /
   `_access_provably_in_bounds(info, ctx, lo=c*lanes)` 的地址安全复核——守卫恒真且该段地址
  可证安全，才能在这段里省掉掩码。
4. lane 均匀的标量读没有掩码形式：守卫恒真证不出时 build 期直接 `Unsupported`（P0-3）。

### 3.7 发射算法

发射目标全部是 `T.simd.*`（`tilelang/ascend/language/simd.py`，CCE 向量原语的前端封装）。**地址统一用** **`BufferLoad(buffer, indices)`**，交给 `T.simd` 自动包成 `access_ptr`；indices 里把向量轴变量替换成该 chunk 的 lane 基址（普通路径 `chunk*lanes`；lane 内均匀访问 `0`）。

指令映射（`_Emitter._build`）：

| DAG recipe                           | `T.simd.*`                                                 | 说明                                              |
| ------------------------------------ | ---------------------------------------------------------- | ----------------------------------------------- |
| `imm`                                | （标量表达式）                                                    | 直接用                                             |
| `lane_index`                         | `vci(lane_base, "int32")`                                  | lane 序号（谓词比较用）                                  |
| `vload`                              | `vld(addr, "NORM")`                                        | 连续装载                                            |
| `vload_strided`                      | `vgather2(addr, vmuls(vci(0,u32), k, m), m)`               | 步长装载（ramp 当索引）                                  |
| `gather`                             | `vgather2(base(BufferLoad 全 0), reinterpret(u32, idx), m)` | 每 lane 索引向量装载（`PERMUTE`/`PIECEWISE`/`INDIRECT`） |
| `uniform_load`/`uniform_expr`/`vbrc` | `vdup(scalar, elem, mask, MODE_ZEROING)`                   | 广播                                              |
| 数据二元 / 一元                            | `vadd/vsub/vmul/... / vneg/vabs/...`                       | `(a, b, mask, mode)`                            |
| 标量-向量                                | `vadds/vmuls/vmaxs/vmins`                                  | `(vec, scalar, mask, mode)`                     |
| 常量移位                                 | `vshrs` / `vshls`                                          | 整型 `//` / `%` 2 的幂常量除数做 strength-reduction      |
| `vcmp` / `vcmps`                     | `vcmp(a,b,mask,op)` / `vcmps(a,s,mask,op)`                 | 谓词                                              |
| `vsel`                               | `vsel(t, f, mask)`                                         | mask 由 `predicate()` 给出                         |
| `pand`/`por`/`pnot`                  | `pand/por/pnot`                                            | 谓词运算                                            |
| `vcvt`                               | `vcvt(src, target, mask)`                                  | mask bits = `min(src, target)`                  |

**写入映射（`_emit_store`）**：`CONTIGUOUS`/`UNIFORM` → `vsts(addr, v, mask, "NORM_B{bits}"/"ONEPT_B{bits}")`；`STRIDED` → `vscatter(v, addr(chunk 基址), vmuls(vci(0,u32), k, m), m)`；`PERMUTE`/`PIECEWISE`/`INDIRECT` → `vscatter(v, index_addr（缓冲区起点）, reinterpret(u32, idx), m)`（索引向量是**绝对**元素偏移，故基址取缓冲区起点）。

**SSA 与** **`Bind`**：每个中间值用 `tirx.Bind(var, expr)` 表示，追加进 `lets` 列表。不能用 `tirx.Let`，因为后者是**表达式**（body 必须是 PrimExpr），而 SSA 中间值是一条条独立语句，只能摊平成 `SeqStmt`。

`_Emitter` 的主要状态：

| 状态 | 作用 |
| ---------- | ------------------------------------------------------------------- |
| `prologue` | **chunk 无关**的绑定，如全 lane 掩码 `pset(bits, "PAT_ALL")`（按 width 缓存，只绑一次） |
| `lets`     | chunk 体内的 SSA 语句                                                    |
| `memo`     | `id(node) → 变量`，同一子表达式去重                                            |
| `_mask_vars` | 掩码按 width / 尾块键缓存，避免重复 `pset` / `update_mask`                        |
| `counter`  | SSA 变量命名序号（跨段再加 `prefix` 区分）                                        |

**掩码三层叠加**：`store 的最终掩码 = 用户谓词 ∧ 尾块掩码`，都为空则全 1。

- **谓词宽度必须跟随数据元素**（R1）：`pand/por/pnot` 的第三个参数是**执行掩码**，它按
  `pset_b<width>` 的粒度决定覆盖多少个 lane（`256 / width`）。旧实现固定 b32 → 只覆盖
  64 lane，f16（128 lane）/ int8（256 lane）的谓词在 lane 64 之后被清零，尾块只写一半。
  现在 `predicate_combine` / `predicate` 都由调用方传入数据元素的 `bits`；`_build` 的
  防御性谓词分支则用最细粒度 b8（256 lane，覆盖所有元素宽度）。
- **store 条件必须走 `predicate()`**（R2）：`if a[0] > 0: c[i] = a[i]` 的条件是 **bool 标量**，
  `value()` 会把它当数据向量 `vdup`，而 bool 没有谓词宽度（emit 期 `ValueError`）。
  `_store_mask` / `_emit_outer_store` 现在统一交给 `predicate()`：均匀条件广播成
  `update_mask(cond ? lanes : 0)`，逐 lane 条件直接使用比较谓词。
- **尾块掩码只在尾块块里出现**：常量 extent 且 `tail != 0` 时，发射被拆成「前 `chunks-1` 个整
  chunk（全 lane 掩码，走 `vld`）」+「尾块 `for ck in [chunks-1, chunks)`（尾块掩码，
  `update_mask(tail)`）」。整 chunk 不再背一个恒真的 `update_mask`。
- 符号 extent（`chunks is None`）只发一个运行时 chunk 循环 `ceildiv(extent, lanes)`，
  尾块掩码在运行时计算并**夹到 lanes**：`update_mask(min(extent - chunk*lanes, lanes))`
  （`update_mask` 只保证 `[0, lanes]` 内有意义，缺陷 A5）。

> ✅ **缺陷 D10（已修，回归基线见 §4.2）**：旧实现把尾块掩码**只加在 store 侧**，而装载
> `vld(addr, "NORM")` 没有 mask 形参（`simd.py` 的 `vld` 签名里根本没有谓词参数），于是
> **所有 load 通道**（连续 `vld`、步长 `vgather2`、间接 `vgather2`）以及**归约尾块的 `vld`**
> 都在最后一个 chunk 上整 lane 越过 buffer 尾部（`a: (64, 8)` 上
> `vld(access_ptr(a[i, chunk*64], 64, 1))` 多读 56 个元素；一维 `(100,)` 上 `vld(a[64], 64)`
> 多读 28 个元素）。现改为：
> * 尾块内的连续装载**退化成带谓词的 `vgather2`**（`vci(0)` 相对 lane 偏移 + 尾块谓词）；
> * 步长/间接装载直接用 `load_guard()`（尾块谓词 ∧ 惰性条件守卫）；
> * 归约尾块同样先带谓词装载，再用 `vsel` 填单位元（数值正确性 + 访问边界都成立）。
>
> 回归：`test_defect_d10_non_reduction_tail_load_is_bounded`（二维）、
> `test_defect_d10_one_dimensional_tail_load_is_bounded`（一维）。

#### 3.7.1 装载守卫与按 chunk 边界回退 `vld`（D16）

`vld` 没有谓词形式，因此「带守卫的装载」统一退化成
`vgather2(addr, vci(0), mask)`（相对 lane 偏移 + 谓词）。这解决了越界读，但会让
**每一个 chunk** 都付一次 gather。守卫通常只在边界附近才可能为假（`i >= 32`、`i < 96`），
于是计划期再算一次「守卫恒真的首/尾 chunk 区间」，发射期据此**分段**：

```text
VectorPlan.guard_boundaries = 所有守卫的 predicate_free_until / predicate_free_from
                              ∪ store 谓词的 mask_free_until / mask_free_from
                              （升序、去重，仅保留 0 < cut < chunks；上限 6 个）
```

`_strip_mine` 在相邻切点之间各发一段，每段携带 `[chunk_min, chunk_max)`；
`_Emitter.guard_free(free_until, free_from)` 判定：

```text
free_from  <= chunk_min          → 本段在守卫恒真尾段内，去掉装载掩码
free_until >= chunk_max          → 本段在守卫恒真首段内，去掉装载掩码
```

store 谓词同理（`mask_free_until/from`）：恒真段直接用全 lane 掩码，连 `vcmps` 链也省掉。
切点只来自**可证**的仿射条件（`arith.Analyzer.can_prove`）；数据相关条件、符号 extent、
证不出边界的守卫保持原样（安全优先）。示例（`N=128`, `if i >= 32: c[i] = a[i-32]`）：

```text
# [0, 1)：守卫跨边界，保留掩码
vs0_gr0_0 = vci(0); vs0_gr0_1 = vcmps(vs0_gr0_0, 32, mask, "ge")
vs0_gr0_2 = vgather2(access_ptr(a[-32], 64, 1), vci(0), vs0_gr0_1)
vsts(access_ptr(c[0], 64, 2), vs0_gr0_2, vs0_gr0_1, "NORM_B32")
# [1, 2)：守卫恒真 → 普通 vld，store 也不再需要 vcmps
vs0_gr1_0 = vld(access_ptr(a[32], 64, 1), "NORM")
vsts(access_ptr(c[64], 64, 2), vs0_gr1_0, mask_32, "NORM_B32")
```

前缀方向完全对称（`if i < 96: c[i] = a[i+32]` → chunk 0 走 `vld`、chunk 1 走带掩码
`vgather2`）；同段内前缀与后缀守卫共存时，切点会同时保留两边的边界，各段独立判定。

> ✅ **缺陷 D8（已修，回归基线见 §4.2）**：`vload_strided` / `STRIDED` store 的发射式
> `vmuls(vci(0,"uint32"), k, m)` 用**无符号** lane 斜坡乘步长，只对 `k > 0` 成立；`Δ ≤ -2`
> 的负步长会静默发出无意义的无符号偏移。现**负步长改走「每 lane 索引向量」通道**
> （与 `PERMUTE`/`PIECEWISE`/`INDIRECT` 同一条路：绝对偏移 + `vscatter`/`vgather2`，
> 负偏移以补码表达），索引向量构造不出来时在分析期落 `TRUE_RANDOM` 拒绝。
> 读写两侧同时覆盖；复现用例已修正为合法下标（旧用例 `a[N-2-2*i]` 在 `i ≥ 64` 时本身越界）。

> ✅ **缺陷 D9（已修，回归基线见 §4.2）**：符号 extent 时 `chunks = tail = None`，旧发射端
> `range(plan.chunks or 1)` 退化为**只发一个 chunk 且无尾掩码**（`n > lanes` 时静默少算）。
> 现改为：非归约路径发**运行时 chunk 循环** `ceildiv(extent, lanes)` + 运行时夹紧的尾块掩码；
> 归约路径同样发运行时 chunk 循环（累加器落成 `local.var` 可写向量寄存器，见 §3.8），
> 不再有任何「静默伪造成 1 个 chunk」的形态。

**系统常量**：

- `lanes = 2048 / elem_bits`（`_vreg_lanes`）：float32→64、f16/bf16→128、int8→256；向量寄存器 2048 位。
- 谓词类型一律 `boolx256`（256 位 `vector_bool`）。
- `_mask_bits(dtype) = max(elem_bits, 8)`：CCE 谓词粒度 b8/b16/b32/b64（无 b4）。
- `_MODE = "MODE_ZEROING"`。

> 放置原则：`primitives.py` 只放 ISA/算子事实与硬件常量；分析词汇（`AccessClass`）留在
> `access.py`，可调策略（`_AXIS_WEIGHT`）留在 `plan.py`——各表跟随其语义归属层，而非物理集中。

### 3.8 归约发射（最特别的部分）

归约按 chunk 数分**两种策略**（阈值 `emit._REDUCE_UNROLL_LIMIT = 4`）；两条路径的尾块处理
与收尾相同。

**A. 短归约（`chunks <= 4`）：编译期展开 + 每 chunk 一条链 + 树形合并**

```
for ck in range(chunks):                        # 编译期展开，不是运行时 for！
    if tail and ck == last:                     # 尾块：装载带谓词（vgather2），再填单位元
        x = vgather2(src_addr(ck), vci(0), tail_mask)   # 不越过 buffer 尾部（D10）
        x = vsel(x, fill, tail_mask)
    else:
        x = vld(src_addr(ck), "NORM")           # 整 chunk
    red[ck] = vcadd(x, mask)                    # 每个 chunk 一条独立链，不需要单位元
acc = red[0 .. chunks) 树形合并                 # 两两 vadd（lane 0）
if 累加写: old = 标量读 dst; acc = vadds/vmaxs/vmins(acc, old, lane0_mask)   # 无 BRC 广播
vsts(dst_addr, acc, lane0_mask, "ONEPT_B{bits}")
```

- **链首直接取 `vcadd` 结果**，因此不需要 `vdup` 单位元（`chunks=1` 时退化成「一次 vcadd + 回写」）；
- **树形合并**把跨 lane 归约的串行依赖从 `chunks` 级压到 `ceil(log2 chunks)` 级，指令条数不变；
- **归约后的算术一律收窄到 lane 0**：`vcadd/vcmax/vcmin` 只有 lane 0 有效（硬件模板用
  `PAT_VL1` 合并），故链合并/累加/回写都用 `pset(bits, "PAT_VL1")`，不再对 63 个无效车道运算；
- **旧值按标量并入**：累加写 `dst op= x` 的旧值直接标量读一次进 `vadds/vmaxs/vmins`，
  省掉一条 `vld(..., BRC_B*)` 广播装载（仅归约结果被拓宽时退回广播）；
- 直线代码无循环控制开销，后端可把下一块的 `vld` 提前调度。

> 旧实现的 `_REDUCE_CHAINS`（=4）与展开阈值相等，`min(4, chunks)` 个槽位在
> `chunks <= 4` 时每槽恰好只用一次，「链内累加」分支**不可达**；已删除该常量与分支，
> 行为逐指令不变（每 chunk 一条链 + 树形合并）。

**B. 长归约（`chunks > 4`）与符号 extent：真实 chunk 循环 + `local.var` 累加器**

```
acc = AllocBuffer(1, elem x lanes, scope="local.var")   # 可写向量寄存器（codegen 发 vector_f32）
acc[0] = vdup(identity, elem, mask)
for ck in range(full or ceildiv(extent, lanes)):        # 真实运行时循环
    x = vld(src_addr(ck), "NORM")                       # 符号 extent 时改带谓词装载 + vsel 填单位元
    acc[0] = vadd(acc[0], vcadd(x, mask), lane0_mask)
final = acc[0]
if 累加写: old = 标量读 dst; final = vadds/vmaxs/vmins(final, old, lane0_mask)
vsts(dst_addr, final, lane0_mask, "ONEPT_B{bits}")
```

- 常量 extent：主循环跑 `chunks-1` 个整块（全 lane 掩码），尾块单独一个**单次**循环块；
- 符号 extent：单条循环，每轮 `update_mask(min(E - ck*lanes, lanes))` 夹紧 + `vsel` 填单位元；
- `local.var` 的向量 dtype 由 codegen 直接声明为向量寄存器（`vector_f32`），
  `acc[0] = ...` / `acc[0]` 读写即寄存器读改写。

**为什么短归约展开？** 展开去掉循环控制、给后端跨块调度空间，收益集中在 chunk 数少、
单段指令预算充足时；chunk 数大时指令体积线性膨胀而收益占比下降，故改用循环 + 可变寄存器
（`tirx.AllocBuffer` + `BufferStore`/`BufferLoad`，即 `T.simd.alloc_var` 的语义）。
两条路径都由同一判据驱动，因此**符号 extent 不再被拒绝**（D9 归约分支已解除）。

配套细节：

- `_reduce_identity`：`vadd→0`、`vmax→` 元素类型最小可表示值、`vmin→` 元素类型最大可表示值（按 dtype 派生）。
  > ✅ **缺陷 D3（已修）**：`_reduce_identity`（`primitives.py`）原先把单位元**硬编码成 f32 极值 `±3.402823e38`**，
  > 未按 store 的 dtype 派生，于是 `int32` 的 `max` 归约抛
  > `FloatImm supports only float, but int32 was supplied`、`float16` 的 `max` 归约抛
  > `Literal value -3.40282e+38 exceeds minimum of float16`——
  > 即**分析期 `[OK ]`、发射期 `ValueError` 崩溃**（非 `Unsupported`，调用方无法优雅兜底）。
  > 现改为按元素类型派生：`vcadd→0`（float 用 `FloatImm`、整型用 `IntImm`），
  > `vcmax→tirx.min_value(elem)`，`vmin→tirx.max_value(elem)`。
  > 回归：`test_defect_d3_int_max_reduction_does_not_crash`、`test_defect_d3_f16_max_reduction_does_not_crash`（均已摘 `xfail`）。
- 旧值并入（`accumulate`）：`_plan_reduce` 在**归约算子脊**上剥离与写入地址**同址**的旧值读取
  （`_strip_reduce`，认 `op` 及其标量同构 `vadds/vmaxs/vmins`），末尾统一用 `BRC` 广播读并入旧值。
  > ✅ **缺陷 D5（已修，随 D13 一并修）**：原先只**扫顶层 `args`**/只下探一层，对
  > `s[i] = s[i] + a[i,j] + 1.0` 这类嵌套表达式会漏判累加（旧值被当作全新结果写回），
  > 同时 `_varying_operand` 会把 lane 均匀加数 `1.0` **静默丢弃**。现 `_strip_reduce` 沿脊递归剥离，
  > lane 均匀加数保留在 `varying` 内（跨 lane 归约天然正确处理）；旧值被缩放/藏在非脊位置
  > （`s[i] = s[i] * 2.0 + a[i,j]`）则显式拒绝 `... must be a direct operand ... accumulate position`。
  > 回归：`test_defect_d5_scaled_old_value_is_rejected`。
- 归约路径逐 chunk 改 `lane_base` 并**清空** **`memo`**（地址随 chunk 变，缓存必须失效）。
- **尾块（改进 6）**：`extent % lanes != 0` 时最后一个 chunk 只有 `tail` 个有效车道。尾块**先带谓词装载**（`vld` 无谓词形式 → 带谓词的 `vgather2`，缺陷 D10），再用 `vsel(x, fill, tail_mask)` 把无效车道替换成**归约单位元**，最后按全 lane 掩码归约（`vadd` 填 0、`vmax` 填 -inf、`vmin` 填 +inf 均不影响结果）。这样 `tail != 0` 既能正确归约，也不会越过 buffer 尾部。

> ✅ **缺陷 D4（已修，随 D5 一并修）**：原先 `_varying_operand`（只下探一层）会**静默丢弃** lane 均匀
> 常量加数（复现 `d4_uniform_addend`：`s[i] = s[i] + a[i,j] + 1.0` 发射成 `vadds(vld(a), s[i])`，`1.0`
> 不见踪影 → 结果偏小）。现由 `_strip_reduce` 递归处理，均匀加数保留在 `varying` 内。回归：
> `test_defect_d4_uniform_addend_inside_reduction_is_not_dropped`（已摘 `xfail`）。

> ✅ **缺陷 D6（已修，随 D13 一并修）**：原先 `_emit_reduce` 用
> `store = next(s for s in plan.stores if s.kind == AccessClass.UNIFORM)` **只取第一条** `UNIFORM` store 发射，
> 于是同一 nest 里两条归约（`s1[i]+=a[i,j]` / `s2[i]+=b[i,j]`）或混合归约（求和 + 求最大）时第二条被**静默丢弃**。
> 现在归约描述改为**逐 store**（`StorePlan.reduce`，由 `_classify_reduce` 填充），`_emit_reduce` 遍历全部 store：
> 每条归约 store 各起一条独立 SSA 累加链、各写一次 `ONEPT vsts`；取值沿轴均匀的 store 走 `_emit_outer_store`
> 广播写。回归：`test_defect_d6_two_reductions_in_one_nest_are_both_emitted`、
> `test_defect_d6b_mixed_reductions_keep_their_own_recipe`、`test_reduce_segment_with_uniform_broadcast_sibling`（均已摘 `xfail`）。

### 3.9 循环重建与遍历策略

`emit_plan`：先把 chunk 体包成 `for chunk in serial(ceil(E/L))`，再**逆序**把非向量轴包成 serial 循环（跳过向量轴本身）。于是 `T.Parallel(M,K)` → `for i in serial(M): for chunk in serial(K/L): ...`。

`emit_plan` 对 `VectorProgram`（多个并列 serial 段，§6 第三步）：逐段发射（每段一个 `chunk_s<k>` 循环 + 该段自身的非向量 serial 轴），**共享外层并行轴只在最外层包一次**；各段掩码 prologue 提到最前。发射顺序即源码顺序。

`vectorize_parallel_nests`：**自顶向下（pre-order）**遍历语句树，命中 `T.Parallel` 就整体 `emit_plan(build_plan(...))` 替换（`_split_nest` 会把连续的并行轴与可吸收的干净 serial 轴一并吃掉，故整个 nest 被一次性下降）；整体规划失败才返回 `None` 继续下探子节点。

> 若自底向上替换，最内层会先被换成串行代码，外层 `T.Parallel` 反而再也规划不了（`normalize_body` 撞到已发射的 `For`），导致多维嵌套只下降一半、外层 `T.Parallel` 残留。

### 3.10 诊断体系（改进 7）

规划过程**无论成败都收集全部卡点**，而不是只抛第一个原因：

- `_axis_diagnostics` 为每根轴生成一行分类摘要——候选轴给出 `store=/load=` 形态与联合打分；`TRUE_RANDOM` 轴给出 `rejected (true-random store)` 并附带 `AccessInfo.reason`（如「非 2 的幂常量除法」）。
- 成功时这些行连同 `selected vector axis '<name>' (score=...)` 一起写入 `VectorPlan.diagnostics`，可用 `render_plan` 打印；调用方无需重新推导选择过程。
- **失败轴的原因在成功路径也保留**（`axis '<name>': <reason> -> fallback`）——否则“为什么没选分数最高的轴”在最终计划上看不出来。
- 失败时（轴回退逐个尝试全失败 / 无候选轴）`build_plan` 把同样的行 + 每根轴的发射失败原因塞进 `Unsupported.diagnostics`，`str(err)` 仍是单行摘要。

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

### 3.11 多段规划：多个并列 serial 循环（§6 第三步）

§3.3–§3.10 描述的是**单段**模型（一个主体 + 可选 `pre`/`post`）。当 `T.Parallel` 体内出现**多个并列的干净** **`T.serial`** **循环**时，`build_plan` 先用 `_try_build_program` 走**多段**路径，产出 `VectorProgram`。

**为什么不是「融合成一段」**：多段之间常带顺序语义（如 softmax 的 `max → sum → 归一化`），融合后第一段必须整段算完才能进第二段，跨段的 store→load forwarding 又要引入新的值通道规则；而**逐段独立向量化 + 源序发射**天然保持「段 k 完全先于段 k+1」，不需要任何新语义，也不动既有发射层。

**切段规则**（`_try_build_program`）：

```python
# 1. 吃掉并行链；body 必须是 SeqStmt，否则交回单段路径
parallel_axes = 连续吸收的 T.Parallel 轴
if body 不是 SeqStmt: return None               # 单段（含「链式嵌套 serial」，交给 _split_nest）
# 2. 按源序把子语句切成「语句组」/「serial 链」
for child in body:
    if child 是 For:
        chain = _absorb_serial_chain(child)     # 连续吸收干净 SERIAL；不干净/最内层仍循环 → None
        if chain is None: return None           # 回落单段 → 保持既有「nested serial loop」拒绝
        items.append(("seg", chain))
    else:
        items.append(("stmt", child))
# 3. 只有 ≥2 个 serial 链才走多段
if len(seg_positions) < 2: return None
# 4. 语句组按源序归属：段前 → 该段 pre；末尾剩余 → 末段 post
segments = [ _build_segment([*parallel_axes, *serial_axes], pre, body, post) for 每个 seg（源序） ]
# 5. 只提「第一根被某段用作**向量轴**的并行轴」之前的并行轴前缀；
#    任一段走展平通道时 program 层不再提层（各段自包）。
consumed = [seg.vector_axis.index for seg in segments if seg.vector_axis.is_parallel]
hoisted = [] if flattened_any else [axis for axis in parallel_axes
                                    if not consumed or axis.index < min(consumed)]
return VectorProgram(axes=hoisted, segments=segments)
```

> **提层前缀（重要修正）**：早期版本把全部共享并行轴提到 program 层，但某些段的向量轴
> 就是外层并行轴（已吃成 lane 维），于是同一组 64-lane 操作被外层 `for i` 重复执行
> `extent/lanes` 次。现在只提「第一根被用作向量轴的并行轴」之前的轴——`program.axes`
> 因此可以是空列表，各段自己包。

**每段的规划**（`_build_segment`）与单段完全同构：`axes = [*并行轴, *该段 serial 轴]`，向量轴由**打分 + 展平优先**决定（并行轴与 serial 轴都是候选）；每段独立产出 `VectorPlan`，各带自己的 `vector_axis` / `reduce` / `pre_stores` / `post_stores` / `diagnostics`。

**发射**（`_emit_program`）：逐段调用 `_emit_segment_stmts(segment, chunk_s<k>, hoisted=program.axes, prefix=f"s<k>_")`——每段一个 `chunk_s<k>` 循环 + 该段自身的非向量 serial 轴；`hoisted` 里的**共享外层并行轴只在最外层包一次**（段内按变量身份跳过）；各段掩码 prologue 提到最前；语句顺序 = 源码顺序。`_Emitter` 按段加名字前缀（`s0_`/`s1_`）避免跨段 SSA 重名。

**正确性前提**：每个并行迭代内，各段按源码顺序**整段执行**（段 k 的全部 lane 完成后才进入段 k+1），因此**段间** RAW/WAR 天然成立；真正需要排除的是**段内**的循环携带依赖（由 `_serial_is_clean` 检查）：

| 形态                                                | 依赖关系                | 是否支持                               |
| ------------------------------------------------- | ------------------- | ---------------------------------- |
| 两个互不相关的 serial（各自 element-wise）                   | 无                   | ✅                                  |
| 归约段 → 广播使用段（`s[i] += …` 后 `c[i,l] = s[i]*2`）      | 顺序执行；`s[i]` 在前段已完成 | ✅                                  |
| element-wise 组合（`c[i,j]=a*2` 后 `d[i,l]=c[i,l]+1`） | 顺序执行；`c` 在前段已完成     | ✅                                  |
| **跨段**跨地址读（前段写 `c`，后段读 `c[i,N-1-l]`）             | 顺序执行；`c` 已完整物化      | ✅（后段按每 lane 索引向量 `gather` 读）        |
| **同一段内**跨地址真依赖（`c[i,l] = c[i,l-1] + …`，循环携带 stencil） | 段内沿向量轴非均匀          | ❌ 拒绝                               |
| 并列段中**任一段**不干净                                    | —                   | ❌ 整体回落单段路径，按「nested serial loop」拒绝 |

> **不变式（低风险设计）**：多段路径**只新增**接受面，不改动既有行为——`_try_build_program` 一旦返回 `None`（不适用 / 无 ≥2 段 / 任一段构造失败），`build_plan` 走的是与多段支持之前**逐字节相同**的单段代码。这保证既有测试（含各拒绝用例）零 churn，单段被接受场景的发射结果也不变。`render_plan` / `emit_plan` 同时接受两种计划类型。
>
> **统一返回契约（级别 1）**：无论走哪条路径，`build_plan` 都返回 `VectorProgram`——单段路径把结果包成 1 段（`axes=_outer_axes(...)`，仅含提到最外层包一次的并行轴前缀），因而调用方只需处理一种计划类型。该包装不改变发射结果：单独发射一个 `VectorPlan`（自己包全部非向量轴）与发射「1 段 program」产出的 TIR **逐节点结构等价**，由护栏测试 `test_single_segment_program_emits_identically_to_single_plan` 覆盖（含「向量轴就是外层并行轴」的关键用例）。
>
> **诊断不吞（缺陷 A/B）**：多段路径的 program 与每段都带 `diagnostics`（段级含逐轴分类与「强制向量轴」说明，program 级含段数摘要）；某一段构造失败时，段级原因经 `notes` 合并进最终 `Unsupported.diagnostics`，不会只剩回落单段路径后那句泛化的「nested serial loop」。

#### 3.11.1 段间共享与寄存器直通

多段独立发射会带来两类**纯冗余**（不涉及语义，只是同一事实被算了两遍）：

1. **重复的 chunk 无关绑定**：每段各绑一份 `pset(bits, "PAT_ALL")` 与
   `vmuls(vci(0), stride, mask)`。它们只依赖常量与彼此，可以放到 program 最前只算一次。
2. **「回写后立刻重读」的往返**：归约/累加段把结果写回共享内存（`vsts NORM`），
   下一段第一件事就是 `vld` 同一地址。寄存器里的值在回写后仍然有效，可以直接复用。

对应两个发射层机制（都只在 `VectorProgram` 路径启用，单段 `emit_plan(VectorPlan)` 行为不变）：

| 机制 | 范围 | 实现 | 约束 |
|---|---|---|---|
| `_ProgramPrologue` | 掩码 / 步长斜坡 | 段间共享 `statements`，各段 `_mask_vars` / `_ramps` 命中即复用 | **只共享纯值**；依赖段内 prologue 掩码（常量尾块 `update_mask`）的斜坡仍绑在段内，避免「先用后定义」 |
| `carry`（寄存器直通） | 整向量连续写 → 同址连续读 | 段 k 的累加器回写把 `(buffer 键, 下标) -> (向量轴变量, 寄存器值)` 登记进 program 级表；段 k+1 的 `vload` 命中即省掉 `vld` | 见下 |

**寄存器直通的判定（宁缺勿错）**：

* **生产段**：该段 `chunks == 1` 且 `tail == 0`（lane 基址恒 0、所有 lane 有效）；回写
  无用户谓词、是 `CONTIGUOUS` 整向量写；地址沿 inner 轴不变（累加器判据已保证）；
  回写没有被 `post` 接管（`superseded`）；向量轴之前没有**未提层**的外层轴
  （否则值只对应最后一次外层迭代）。
* **消费段**：`chunks == 1`（否则第二个 chunk 会错误复用同一寄存器）；向量轴变量与
  登记项**同一对象**（`same_as`，lane k 才对应同一迭代）；装载无尾块/守卫（带谓词的
  `vgather2` 有零化语义，不能直接换成寄存器）。
* **失效**：任何段对同一 buffer 的写都会经 `note_write` 作废对应项；跨段的写序因此
  自动生效。

> 该机制**不删除**生产段的 `vsts`：内存写是否可省需要整段 liveness 信息（`s` 可能在
> `T.Parallel` 之后仍被使用）。因此收益是「省一条 `vld` + 去掉重复的 pset/ramp」，
> 而不是「消除内存往返」。
>
> 示例（`p_serial_reduce_then_use`，修复前 → 修复后）：prologue 由「2×`pset` + 2×ramp」
> 降为「1×`pset` + 1×ramp」；段 1 的 `vld s` 消失，直接 `vmuls(acc, 2.0)`。
> 多 chunk 的段间读（`p_serial_compose`）**不会**直通——由
> `test_multi_segment_carry_is_not_used_across_chunks` 固定住。
>
> **门禁实测**：多 chunk（`Q=8`、`Q=2`）、`extent < lanes` 的尾块、段内中间覆写、链式
> 归约四类探针均符合预期——前两类不直通（重新 `vld` 或掩码 `vgather2`），中间覆写作废
> 旧登记，链式归约只复用**最新**生产者的寄存器。对应用例：
> `test_multi_segment_carry_is_not_used_across_chunks`、
> `test_carry_is_invalidated_by_an_intermediate_write`、
> `test_carry_tracks_the_newest_producer`。

### 3.12 条件分支：`if` 的降级与谓词化

分支全部在**规范化层**（`body.normalize_body`）与**值 DAG 层**（`value._select_node`）消化；发射层只看到「带 mask 的 store」或「`vsel` 节点」，不再有控制流。因此「带 `if` 怎么向量化」等价于「谓词从哪来、加在谁身上」。

#### 3.12.1 语句级 `if` 的三种结局

| TIR 形态 | 判定（`body.normalize_body`） | 产物 | 发射 |
|---|---|---|---|
| `if c: body`（单臂；`else` 缺省或为 `Evaluate(0)`） | 递归规范化 `body`，把 `c` 与已有 mask 取合取 | 谓词化 store（单臂可含**多条**写，全部带谓词）；`c` 同时作为取值表达式的装载守卫 | `vsts(..., mask)` + 带守卫的装载 |
| `if c: storeA else: storeB`，两臂各**恰好 1 条**写、`buffer.same_as`、下标 `expr_deep_equal` | 合并 | `StoreSpec(value=tirx.op.if_then_else(c, A, B), mask=外层 mask)`（**惰性** `Call`） | `vsel`；两臂装载各带 `c` / `¬c` 守卫 |
| 有 `else` 且双臂不满足上述条件（异址 / 臂内多条写 / 同址但某臂已有 mask） | 无法合并 | `Unsupported` | — |

关键点：

1. **单臂是「谓词化 store」**：`mask = And(外层 mask, cond)` 会一路累积，因此嵌套单臂 `if` 自然合成复合谓词。
2. **同址判定用结构化相等**：`_same_indices` 用 `tirx.analysis.expr_deep_equal`，`IntImm(0)` 这类「不同对象、同值」的下标也能识别。
3. **有 `else` 时两臂必须各只有一条写**（单臂分支不限制写条数）；否则不是同一位置的 `select` → `divergent branch arms`（§4 矩阵 §1-I-b）。
4. **`masked branch merging`**：两臂同址但某臂内部已经是谓词化写（如 `if c1: if c2: c[i]=x else: c[i]=y`），一个 `if_then_else` 表达不了两份 mask → 显式拒绝（D12，已知边界）。
5. `else` 为字面量 `0`（TIR 的 `Evaluate(0)`）视作无 `else`（`_is_truthy_else`）。
6. **合并结果必须是惰性的**（D16）：用 `tirx.op.if_then_else(...)`（`Call`）而不是 `tirx.Select`。语句级 `if/else` 的两臂只在条件成立时执行；`Select` 是急切求值，会让越界臂的地址被无条件装载。

实现骨架（`body.py`）：

```python
if not else_stores:                         # 单臂 → 谓词化（谓词也会传给取值表达式）
    combined = condition if mask is None else tirx.And(mask, condition)
    for spec in then_stores:
        spec.mask = combined if spec.mask is None else tirx.And(combined, spec.mask)
elif len(then_stores) == 1 and len(else_stores) == 1:
    if 同 buffer and _same_indices(...) and 两臂都没有 mask:
        return [StoreSpec(buf, idx, tirx.op.if_then_else(condition, t, f), mask)]
    raise Unsupported("divergent branch arms ...")
else:
    raise Unsupported("divergent branch arms ...")
```

#### 3.12.2 `Select` 表达式 → `vsel`

`T.if_then_else` / `tirx.Select` 走 `build_value`：

```text
条件 / 真臂 / 假臂全部 is_uniform → _uniform(expr)            # 标量求值一次 + vdup
否则                              → ValueNode("vsel", [cond, t, f])
```

`T.if_then_else` 是**惰性**内建：两臂分别在 `cond` / `¬cond` 守卫下构造（§3.6）；
`tirx.Select` 是**急切**求值，两臂都算、不挂守卫——语句级 `if/else` 的合并结果因此用
`tirx.op.if_then_else` 承载，而不是 `tirx.Select`（D16）。

条件有两种来源，发射端 `_Emitter.predicate` 统一处理：

| 条件 | 判定 | 发射 |
|---|---|---|
| 逐 lane（随向量轴变化） | `cond.is_uniform == False` | 条件本身即谓词寄存器（`vcmp` / `vcmps` / `pand` / `por` / `pnot`），直接作 `vsel` 的 mask |
| 循环不变（沿向量轴 Δ=0） | `cond.is_uniform == True` | `update_mask(Select(cond, lanes, 0), bits)` 广播成全 lane 谓词，整个 chunk 同真同假 |

布尔组合由 `_bool_node` 建 `pand` / `por` / `pnot`；比较由 `_compare` 建 `vcmp`（两侧都随 lane 变化）或 `vcmps`（一侧 lane 均匀，交换操作数时翻转谓词）。

#### 3.12.3 访存怎么处理（关键语义）

**语句级 `if` 与 `T.if_then_else` 的臂都是惰性的：store 谓词同时也是取值表达式的装载守卫**
（D16 已修）。

- `if c: c[i] = a[i]` → 装载守卫 `c` 在全域可证安全 → `vld a` + `vsts c`（mask = `c`）；
- `if i >= 32: c[i] = a[i-32]` → `a[i-32]` 在第 0 个 chunk 会读到 `a[-32]`，守卫不可省：
  chunk 0 发带掩码的 `vgather2`，chunk 1 守卫恒真 → 回到普通 `vld`（§3.7.1）；
- `if c: c[i] = a[i] else: c[i] = b[i]` → 合并成**惰性** `if_then_else`：真臂的 `a` 挂
  `c`、假臂的 `b` 挂 `¬c`；两个地址都可证安全时仍各自发普通 `vld` + `vsel`；
- `c[i] = T.if_then_else(cond, a[i - 1], 0)`（惰性内建）→ 真臂的 `a[i-1]` 在 lane 0 越界，
  谓词 `cond` 合进装载掩码，发带谓词的 `vgather2`；
- 能**证明**地址在所有 lane 内（`_access_provably_in_bounds`）时省掉守卫，
  常见的 `select(cond, a[i], b[i])` / `if a[i] != 0: c[i] = a[i]` 仍发普通 `vld`。

语义要点：

> **语句级 `if` 的臂只在条件成立时求值**。因此把 store 谓词挂到臂内访存上不是「额外优化」，
> 而是**保持源语义**：条件为假时那些地址本来就不该被访问。合并同址 `if/else` 时必须用惰性
> `if_then_else` 承载两臂，否则急切 `Select` 会把未选中臂的地址也读一遍（D16）。
>
> 守卫只保证「条件为真时地址合法」；若源程序在条件为真时本身就越界（如 512 元素上
> `if i <= 499: c[i] = a[i+32]`），那不是向量化能修复的问题。

chunk 内 mask 的叠加（`_store_mask` / `_Emitter.load_guard`）：

```text
store 最终 mask = 用户谓词 ∧ 尾块谓词（都为空 → 全 lane 谓词）
load  最终 mask = 尾块谓词 ∧ 惰性条件守卫（都为空 → 全 lane 谓词）
                 其中「守卫恒真的首/尾 chunk 段」直接去掉守卫，退回 vld（§3.7.1）
pre/post store = 用户谓词（不叠加尾块谓词，它们不随 chunk 变化）
```

#### 3.12.4 曾经的缺口（R2 / R3，均已修）

| 形态 | 旧现象 | 修复 |
|---|---|---|
| 单臂 `if` + **循环不变条件**（`if a[0] > 0: c[i] = a[i]`） | build OK，emit 抛 `ValueError: Cannot infer default mask width for dtype 'bool'`（`_store_mask` 把 bool 条件当数据 `vdup`） | **R2 已修**：`_store_mask` / `_emit_outer_store` 统一走 `_Emitter.predicate(node, bits)`，均匀条件广播成 `update_mask(cond ? lanes : 0)` |
| **被条件包裹的归约**（`if i < N//2: s[0] += a[i]`） | build OK，归约忽略 mask，**静默对全部 lane 求和** | **R3 已修（改为拒绝）**：`_plan_reduce` 对 `store.mask is not None` 显式拒绝；该轴失败后规划器回退到其它轴。支持方式：`vsel(varying, identity, cond)` 填单位元 + 合并掩码（后续工作） |
| **语句级分支的臂内越界访存**（`if i >= 32: c[i] = a[i-32]`） | build OK，臂内 `a[i-32]` 被无条件发成 `vld a[-32]`（越界 32 元素）；同址 `if/else` 合并成急切 `Select`，未选中臂同样被读 | **D16 已修**：① store 谓词作为取值表达式的装载守卫；② 同址 `if/else` 合并为惰性 `if_then_else`；③ 守卫恒真的首/尾 chunk 段回退普通 `vld`（§3.7.1） |

> 这些缺口与 ana.md 的 A1（bool 数据操作数）同族：都是契约裂缝。写条件归约时仍建议改写为
> `c[i] = T.if_then_else(cond, a[i], c[i])` 形式，或不要在条件内做归约。
>
> **R1（谓词宽度）** 也属于这一族：`pand/por/pnot` 的执行掩码旧实现固定 b32（64 lane），
> f16/int8 的谓词在 lane 64 之后被清零，尾块只写一半；现在按数据元素宽度取掩码（§3.7）。
>
> **R8（写后读转发失效）**：`_Emitter.note_write` 现在会作废该 buffer 上的
> `_forward` 转发项。旧实现只在连续写时更新转发、从不清除，于是 scatter / 标量写 /
> 累加器回写之后仍可能命中"写前"的陈旧向量值（一条潜在的静默错误路径）。

### 3.13 拒绝判据（REJ）总表

#### 3.13.1 判定层级与回退语义

| 层级 | 触发点 | 影响范围 | 能否回退 |
|---|---|---|---|
| **轴级** | `rank_vector_axes` 过滤候选；或 `plan_for` 抛 `Unsupported` | 一根候选向量轴 | ✅ 换下一根轴（`build_plan` 的轴回退） |
| **段级** | `_build_segment` / `_try_build_program` | 一个 serial 段 | ✅ 回落单段路径（`_try_build_program` 返回 `None`，原因写入 `notes`） |
| **Nest 级** | `_split_nest` / `_prepare_segment`；或所有候选轴都失败 | 整个 `T.Parallel` | ❌ 整段拒绝；`vectorize_parallel_nests` 捕获后保留原 IR |
| **Build 门禁** | `_require_supported_element`、归约精化、pre/post 均匀性 | 整个 nest | ❌（保证 build 成功 ⇒ emit 成功） |
| **Emit 防御** | `_Emitter.scalar` / `_build` / `_emit_store` | — | 理论上不可达；触发即说明 build 门禁有缺口 |

**轴级 vs Nest 级的分界**很关键：

- **写侧**是 `TRUE_RANDOM` 时，该轴在 `rank_vector_axes` 阶段就被淘汰（不进入候选）；
- **读侧 / 值算子 / 内存域**的问题只有在 `plan_for` 真正建值 DAG 时才暴露，属轴级失败，可以回退到次优轴；
- 所有轴都失败才抛 nest 级 `no vectorizable axis: ...`；两种消息分别对应「无候选轴」与「候选轴逐个失败」。

#### 3.13.2 A. 模型 / 边界类拒绝

| 判据 | 触发条件 | 典型消息 | 代码位置 |
|---|---|---|---|
| 无并行轴 | 入口不是 `T.Parallel` 嵌套 | `no T.Parallel axis found` | `plan._split_nest` |
| 无写语句 | `main` 规范化为空 | `T.Parallel body has no store to vectorize` | `plan._prepare_segment` |
| 元素类型不支持 | store 元素类型推不出谓词宽度（如 `bool`） | `store element dtype 'bool' ... is not supported` | `plan._require_supported_element` |
| GM 操作数 | 访存 `buffer.scope() == "global"` 且未 staging | `global-memory operand ... needs an explicit UB staging copy` | `value.load_node` |
| 原子操作 | `Evaluate(atomic_*)` 或值表达式含原子调用 | `atomic op ... no vector+scalar atomic recipe` | `body.normalize_body` / `value._build_call` |
| 无法识别的语句 | 非 `SeqStmt`/`BufferStore`/`BufferLoad`/`Evaluate`/`IfThenElse`/`For`/`Bind` | `unsupported statement in T.Parallel body: ...` | `body.normalize_body` |
| 访问秩不匹配 | 下标个数 ≠ buffer 秩 | `access rank ... does not match buffer rank ...` | `access.linear_element_offset` |

#### 3.13.3 B. 访存 / 仿射类拒绝

| 判据 | 触发条件 | 典型消息 |
|---|---|---|
| 真无结构（写侧） | 某 store 沿轴为 `TRUE_RANDOM` | 该轴**不进入候选**；全轴如此 → `every store is true-random ... on all axes` |
| 真无结构（读侧/值） | 元素偏移无法在索引向量通道构造 | 轴级：`non-affine access ... the per-lane index vector is not constructible (...)` |
| 非 2 的幂常量除法 | 索引含 `FloorDiv/FloorMod`，除数不是正 2 的幂 | `FloorDiv by 3 is not a power of two`（`index_vector_reason`） |
| 非常量除法 | 索引的 `FloorDiv/FloorMod` 除数不是编译期常量 | `needs a compile-time constant divisor` |
| 不支持的索引算子 | 索引 DAG 含 `index_vector_reason` 不认识的算子 | `unsupported index operator <Name>` |
| 负步长不可构造 | `STRIDED` 且 `stride < 0`，但索引向量构造不出 | `FloorDiv/... is not a power of two` 等（落 `TRUE_RANDOM`，D8） |

#### 3.13.4 C. 值 DAG 类拒绝

| 判据 | 触发条件 | 典型消息 |
|---|---|---|
| 不支持的值表达式 | 非 BufferLoad/Var/Imm/Call/Select/Cast/比较/算术/布尔 | `unsupported value expression: <Node>` |
| 不支持的内建调用 | intrinsic 尾段不在 recipe 表 | `unsupported intrinsic call in T.Parallel body: <tail>` |
| `FloorDiv/FloorMod` 作值算子 | 除数非正、非常量或非 2 的幂 | `... needs a positive compile-time constant divisor` / `not strength-reduced yet` |
| 无法标量化 | 把向量值当标量操作数（emit 防御） | `cannot scalarize vector value '<recipe>'` |
| 无发射 recipe | 值节点 recipe 没有 `_build` 分支 | `no emitter for value recipe '<recipe>'` |

#### 3.13.5 D. 归约类拒绝

| 判据 | 触发条件 | 典型消息 |
|---|---|---|
| 非 Add/Max/Min | 最外层归约算子为 `Mul` 等 | `unsupported reduction operator(s) Mul ... only Add/Max/Min ...` |
| 覆写（last-write-wins） | lane 均匀地址 + 取值不读 dst | `... does not read 's': only the accumulate form ...` |
| 旧值不在累加位置 | 旧值被缩放 / 嵌套 / 跨址（stencil） | `must be a direct operand ... accumulate position` |
| 归约体在非累加位置读 dst | 剥离后 `varying` 仍读目标 buffer | `the reduction body reads the destination buffer ... outside the accumulate position` |
| 部分归约 | 吸收的 serial 轴上，lane 均匀写 + 取值逐 lane 变化，但整段不是归约 | `a partial reduction mixed with element-wise stores has no vector form` |
| 条件归约（R3） | 归约 store 带 mask（`if cond: s[0] += ...`） | `conditional reduction into 's' is not supported yet ... (R3)` |
| 旧值重复出现（R4） | 归约表达式里同址旧值出现两次（`s[0] = s[0] + (s[0] + a[i,j])`） | `the old value of 's' appears more than once ...` |
| 归约段 + 非均匀 pre/post（R7） | 沿并行轴归约，但 pre/post 沿该轴不是 `UNIFORM` | `a statement before/after the serial loop must be uniform along the reduction vector axis ...`（该轴失败后回退） |

#### 3.13.6 E. 语句结构 / serial 类拒绝

| 判据 | 触发条件 | 典型消息 |
|---|---|---|
| 嵌套循环 | 进入 `normalize_body` 的 `For`（不干净 serial / 嵌套 `T.Parallel` / 非直接内嵌） | `nested <kind> loop inside T.Parallel is not affine-vectorizable` |
| 非 0 循环下界（R5） | 被吸收/待吸收的 `For` 下界无法证明为 0（`T.serial(1, K)`） | `<kind> loop 'j' must start at 0 to be vectorized (min=1)` |
| 兄弟语句只能沿 serial 轴 | 有 `pre`/`post` 但向量轴不是该 serial 轴 | `... can only be vectorized along that serial axis` |
| pre/post 非均匀 | `pre`/`post` 的 store 沿向量轴不是 `UNIFORM` | `a statement before/after the serial loop must be uniform ...` |
| 发散分支 | 双臂写不同地址 / 某一臂多条写 | `divergent branch arms ...` |
| 带 mask 的臂合并 | 两臂同址但臂内已有谓词 | `masked branch merging is not supported`（D12） |
| 多段中任一段不干净 | `_absorb_serial_chain` 返回 `None` | 回落单段 → `nested serial loop ...`；段级原因在 `diagnostics` |

#### 3.13.7 F. 数据依赖类拒绝（serial 吸收的判据）

`T.Parallel` 的迭代按定义相互独立；真正需要依赖分析的是**被吸收的 `T.serial` 轴**。`plan._serial_is_clean` 用四条保守判据排除循环携带依赖：

| # | 判据 | 拒绝条件 | 典型消息 |
|---|---|---|---|
| 1 | 沿循环变量**地址不变**的写必须是归约 | 目的地址 `Δ==0`，但取值不是 `dst[...] = dst[...] <op> x`（同址自读） | `write to X is invariant along 'j' but not a reduction` |
| 2 | 归约必须真的随循环变量变化 | 是累积写，但取值沿该轴 `Δ==0`（如 `cnt[0] += 1`） | `reduction into X does not vary along 'j'` |
| 3 | 禁止跨迭代写写别名（R6） | 同一 buffer 被多条写覆盖，且目的地址互不相同（`c[i,2*j]` 与 `c[i,j]`、`c[i,j]` 与 `c[i,j+1]`） | `multiple writes to X at different addresses along 'j' may alias across iterations` |
| 4 | 禁止跨迭代读 | 循环内某 buffer 被写，又被读，且读地址与**所有**写地址都不同（stencil） | `cross-iteration read of X along 'j'` |

补充说明：

- 判据 1、2 共同覆盖「共享标量」：`cnt[0] = a[i,j]`（覆盖写）与 `cnt[0] += 1`（取值不变化）都拒绝；
- 判据 3 是保守的充分条件：同一地址的多条写（如两条 `s[i] += ...`）仍允许（顺序发射保持语义），不同地址则可能跨迭代相撞；
- 判据 4 的等价表述：读地址必须与某条写的地址**同址**（`_same_indices`），才允许「读自己刚写的值」；
- 通过者即 S1（`c[i,j] = a[i,j]`）与 S3（`s[i] += a[i,j]`）；
- 吸收失败不会立即报错：`_split_nest` 停止吸收，`normalize_body` 最终以 `nested serial loop` 拒绝（多段路径则先回落单段，段级原因保留在 `diagnostics`）。

**没有做的依赖分析**：

- 不做通用的跨语句 / 跨段依赖求解；多段路径依赖**源序顺序执行**保证段间 RAW/WAR 正确，不做重排；
- 同一 `T.Parallel` 内**多迭代写同一地址**的语义未定义（数据竞争），模型不拒绝也不保证 last-write-wins（ana.md A3）。同一条规则覆盖 `PIECEWISE` / `INDIRECT` **scatter 写**的地址碰撞（`c[b[i]]`、`c[i//2]`）：使用者需保证每 lane 写地址两两不同；
- 归约的旧值必须同址，这条规则既是正确性要求，也是当前唯一建模的「循环携带依赖」形式；
- **展平通道**（`_try_flatten`）额外要求每条 store 连续、读写不重叠、且**同一 buffer 只能有一条写**（否则展平后不同 k 的写地址可能重叠，见 R6 同族判据）。

#### 3.13.8 与 `test_drive_plan.py` 的对应

`drive_plan` 当前 **66** 个用例、**11** 个 REJ，与上述判据的对应关系：

| 用例 | 判据类别 | 具体原因 |
|---|---|---|
| `c6_complex` | E. 嵌套循环 | 共享标量 + 非直接内嵌 serial → `nested serial loop` |
| `s_serial_stencil` | F. 跨迭代读 | stencil 读地址 ≠ 写地址 |
| `p_serial_one_unclean` | E. 多段不干净 | 一段不可吸收 → 回落单段 → nested loop |
| `c_gm` | A. 内存域 | GM 操作数需 staging |
| `c4_true_random` | B. 索引不可构造 | `i//3` 非 2 的幂 |
| `d14_partial_reduce` | D. 部分归约 | serial 轴上 lane 均匀写 + 变化取值 |
| `d15_bool_store` | A. 元素类型 | `bool` 无谓词宽度 |
| `d1_product_reduce` | D. 归约算子 | `Mul` 不在白名单 |
| `d2_plain_overwrite` | D. 覆写 | 取值不读 dst |
| `d12_masked_branch` | E. 分支形态 | 臂内已有 mask，无法合并 |
| `r3_masked_reduce` | D. 条件归约 | 归约掩码只承载尾块谓词（R3） |
| `r4_repeated_old_value` | D. 旧值重复 | 单一初值表达不了两次旧值（R4） |
| `r5_serial_nonzero_min` | E. 非 0 下界 | `T.serial(1, K)`（R5） |
| `r6_serial_waw` | F. 跨迭代写写别名 | `c[i,2*j]` 与 `c[i,j]`（R6） |

> 统一约定：**所有拒绝都抛 `Unsupported`**（不抛 `ValueError` / `AssertionError`），并尽可能携带 `diagnostics`。历史例外（§3.12.4 的两个 `if` 缺口、ana.md A1）现已分别由 R2 / R3 修复。

***

## 4. 场景支持矩阵

图例：✅ 分析+发射均支持 · ⚠️ 部分支持 / 依赖具体形状 · ❌ 明确拒绝（不静默降级）

| 编号        | 场景                                  | 状态           | 机制 / 拒绝原因                                                                                                                              | 对应测试                                                                                                                                                                                                                                                                 |
| --------- | ----------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| §0-I      | Copy / Brc                          | ✅            | Δ 分类为 CONTIGUOUS / UNIFORM                                                                                                             | `test_copy`、`test_scalar_broadcast`                                                                                                                                                                                                                                  |
| §0-II     | GM 操作数                              | ❌            | `global-memory operand ... needs an explicit UB staging copy`                                                                          | `test_global_operand_rejected`                                                                                                                                                                                                                                       |
| §1-I-a    | 单臂 `if`                             | ✅            | 谓词化 store（`mask &= cond`）；谓词同时作为臂内访存的装载守卫（D16）                                                                      | `test_if_single_arm_is_predicated_store`、`test_statement_if_arm_load_is_guarded`                                                                                                                                                                                     |
| §1-I-b    | `if`/`else` 同址写                     | ✅            | 合并成**惰性** `if_then_else` → `vsel`；两臂装载各挂 `c`/`¬c` 守卫                                                                               | `test_if_else_merges_into_select`、`test_statement_if_else_merge_keeps_branch_laziness`                                                                                                                                                                              |
| §1-II     | 三元：**循环不变**条件                       | ✅            | 广播全 lane 谓词 `update_mask(cond?lanes:0)`                                                                                                | `test_ternary_with_loop_invariant_condition_broadcasts_predicate`、`test_select_node_with_loop_invariant_condition_broadcasts_predicate`、`test_loop_invariant_condition_with_uniform_arm_broadcasts_and_dedups`、`test_select_that_is_uniform_everywhere_stays_scalar` |
| §1-II     | 三元：逐 lane 条件                        | ✅            | `vcmp` 逐 lane 谓词                                                                                                                       | `test_ternary`                                                                                                                                                                                                                                                       |
| §1-I/II   | 语句级分支的臂内**越界**访存                    | ✅            | D16：装载守卫 + `guard_boundaries` 切分；只有跨边界的 chunk 付 `vgather2`，恒真段回退 `vld`                                                                     | `test_statement_if_arm_load_is_guarded`、`test_statement_if_prefix_guard_is_dropped_before_boundary`、`test_guard_boundaries_cut_prefix_and_suffix_independently`、`test_statement_if_safe_load_stays_unmasked`                                                         |
| §2-I      | 操作数维度不一致                            | ✅            | Δ=0 一侧广播（`vbrc`/`vdup`）                                                                                                                | `test_operand_rank_mismatch_broadcasts_outer_load`                                                                                                                                                                                                                   |
| §2-II     | 规约（`+=` / `max`）                    | ✅            | `ReduceInfo` → `vcadd/vcmax/vcmin`                                                                                                     | `test_reduction_detected`、`test_reduction_max_maps_to_vcmax`                                                                                                                                                                                                         |
| §2-II     | 规约 + 尾块（`extent % lanes != 0`）      | ✅            | 尾块带谓词装载（`vgather2`）+ `vsel` 单位元填充（改进 6 / D10）                                                                                                  | `test_reduction_with_tail_fills_identity`                                                                                                                                                                                                                            |
| §3-I      | 多行语句（连续）                            | ✅            | `normalize_body` 产出多条 `StoreSpec`                                                                                                      | `test_multi_store`、`test_multistore_with_unit_trailing_dim_is_contiguous`                                                                                                                                                                                            |
| §3-I      | 多行语句（文档原型：末维固定 `[.., k]`）           | ✅            | 末维固定 → 沿唯一轴 STRIDED → `vscatter` 索引向量写                                                                                                 | `test_store_strided_on_the_only_axis_is_scattered`                                                                                                                                                                                                                   |
| §3-II     | 复杂标量 + `if_then_else`（im2col）       | ⚠️           | 值算子 `FloorDiv/FloorMod` 已对 **2 的幂常量除数**做精确 strength-reduction（`vshrs`/`vand`）；最优（写连续）轴仍可能因**非 2 的幂常量除数**被拒，此时靠**轴回退**（§3.4）退到另一根轴照常向量化 | `test_floordiv_value_operator_strength_reduced`、`test_floormod_value_operator_strength_reduced`、`test_floordiv_by_non_power_of_two_is_rejected`                                                                                                                      |
| §4-I      | 表达式下标                               | ✅（UB）/ ❌（GM） | 线性化+化简后连续 / GM 需 staging                                                                                                               | `test_expression_subscript`、`test_expression_subscript_reads_shared_operand_as_uniform`、`test_expression_subscript_rejected_on_global_operand`                                                                                                                       |
| §4-II-a   | 高维摊平（一维化）                           | ✅            | 恒等式化简后 Δ=1                                                                                                                             | `test_highdim_flatten`、`test_finite_difference_absorbs_flatten_identity`                                                                                                                                                                                             |
| §4-II-a   | 步长访问（`a[2*i]`）                      | ✅            | STRIDED → `vgather2`                                                                                                                   | `test_strided_access`                                                                                                                                                                                                                                                |
| §4-II-b   | 高维→低维（block-tiled 置换）               | ✅            | Δ 非常数且可构造 → `PIECEWISE` → `vgather2` 索引向量                                                                                              | `test_block_tiled_fragment_layout_is_piecewise_gather`                                                                                                                                                                                                               |
| §4-II-b   | 逆序（`a[N-1-i]`）                      | ✅            | Δ=-1 → `PERMUTE` → `vgather2`                                                                                                          | `test_reversal_is_permute_gather`                                                                                                                                                                                                                                    |
| §4-gather | 间接下标 `A[B[i]]`                      | ✅            | data-dependent 下标 → `INDIRECT` → `vgather2`（索引来自寄存器）                                                                                   | `test_indirect_gather_is_expressed`                                                                                                                                                                                                                                  |
| §4-gather | 无结构下标（`a[i//3]`，非 2 的幂）             | ❌            | `TRUE_RANDOM` → `not a power of two`                                                                                                   | `test_true_random_access_is_rejected`                                                                                                                                                                                                                                |
| §5        | `T.atomic_add`                      | ❌            | 无「向量+标量」原子配方                                                                                                                           | `test_atomic_rejected`                                                                                                                                                                                                                                               |
| §6 第一步    | 直接内嵌、干净的 serial 循环（S1 copy / S3 归约） | ✅            | serial 轴被吸收为候选轴：S1 → 连续 `vld/vsts`，S3 → `vcadd`；含循环携带状态（共享标量 / stencil）仍拒                                                              | `test_direct_serial_copy_absorbed`、`test_direct_serial_predicated_absorbed`、`test_direct_serial_reduction_absorbed`                                                                                                                                                  |
| §6 第二步    | 多条语句夹一个 serial（M1/M2/M3/夹心、S2 私有标量） | ✅            | body 为 `SeqStmt` 且**恰好含一个干净 serial 循环**时吸收该轴：其前置/后置兄弟语句成为 `pre`/`post` 段，发射在 chunk 循环**之外**（顺序 `pre → 主体 → post`）                      | `test_serial_pre_statement_absorbed`、`test_serial_post_statement_absorbed`、`test_serial_sandwiched_statements_absorbed`、`test_serial_independent_store_group_absorbed`、`test_direct_serial_private_scalar_absorbed`                                                  |
| §6 第三步    | 多个**并列**的干净 serial 循环（≥2 段）         | ✅            | 逐段独立向量化 + 源序发射（`VectorProgram`）：每段各自选向量轴/归约/`pre`/`post`，共享外层并行轴只包一次；段间按源码顺序整段执行（段内不得有循环携带依赖）                                                       | `test_multi_serial_independent_segments`、`test_multi_serial_reduce_then_use`、`test_multi_serial_compose_keeps_source_order`、`test_multi_serial_leading_statement_is_first_segment_pre`、`test_vectorize_parallel_nests_lowers_multi_serial_program`                   |
| §6        | 嵌套 serial / 跨迭代标量状态 / 并列段中有一段不干净    | ❌            | `nested ... loop ... is not affine-vectorizable`                                                                                       | `test_serial_shared_scalar_rejected`、`test_serial_stencil_rejected`、`test_multi_serial_one_unclean_is_rejected`                                                                                                                                                      |
| §1-I      | **双臂 `if` 的某一臂再嵌 `if`**（臂自带谓词）    | ❌            | `normalize_body`（`body.py`）`masked branch merging is not supported`——**两份文档先前均未声明**（缺陷 D12）                                                 | `test_masked_branch_merging_is_rejected`（特征化）                                                                                                                                                                                                                        |
| §2-II     | **一个 nest 内多条归约**（`s1+=…` 且 `s2+=…`，或求和+求最大混合） | ✅            | 归约描述改为**逐 store**（`StorePlan.reduce`），`_emit_reduce` 遍历全部 store，各起独立累加链 + 各写一次 `ONEPT vsts`（**D6 已修**）                                                             | `test_defect_d6_two_reductions_in_one_nest_are_both_emitted`、`test_defect_d6b_mixed_reductions_keep_their_own_recipe`、`test_reduce_segment_with_uniform_broadcast_sibling`（**已摘 xfail**）                                                                                                                                                                                                  |
| §2-II     | **归约体含 lane 均匀加数**（`s[i] += a[i,j] + 1.0`） | ✅            | `_strip_reduce` 沿归约算子脊递归剥离旧值 → 均匀加数保留在 `varying` 内（**D4/D5 已修**）                                                                                    | `test_defect_d4_uniform_addend_inside_reduction_is_not_dropped`、`test_defect_d5_scaled_old_value_is_rejected`（**已摘 xfail**）                                                                                                                                                                                               |
| §2-II     | **非 Add/Max/Min 的归约**（乘积归约 `s[i] *= a[i,j]`） | ✅（拒绝）        | `_reduce_op`（`plan.py`）**已改**：未识别算子显式拒绝 → 乘积归约不再被发成求和（**D1 已修**）                                                                                    | `test_defect_d1_product_reduction_is_not_emitted_as_sum`（**已摘 xfail**）                                                                                                                                                                                                     |
| §2-II     | **覆写 `s[i] = a[i,j]`（store 沿轴均匀、取值变化）** | ✅（拒绝）        | 统一归约判据：lane 均匀但取值不读 `dst` → 显式拒绝（精确诊断 `does not read 's' … accumulate form`），不再误判成 `vcadd`（**D2 已修**，随 D13）                                                                                    | `test_defect_d2_plain_overwrite_is_not_treated_as_reduction`（**已摘 xfail**）                                                                                                                                                                                           |
| §2-II     | **`int`/`fp16` 的 `max`/`min` 归约**  | ✅            | `_reduce_identity` 按 dtype 派生单位元（`tirx.min_value/max_value`）→ 不再抛 `ValueError`（**D3 已修**）                                                                                    | `test_defect_d3_int_max_reduction_does_not_crash`、`test_defect_d3_f16_max_reduction_does_not_crash`（**均已摘 xfail**）                                                                                                                                                    |
| §1-I      | **谓词合取的掩码宽度**（f16/int8 尾块 + 条件）      | ✅            | `pand/por/pnot` 的执行掩码跟随**数据元素**宽度；固定 b32 只会覆盖 64 lane（**R1 已修**）                                                                     | `test_predicate_combine_follows_the_data_element_width`                                                                                                                                                                                                              |
| §1-I      | **循环不变条件**（`if a[0] > 0: c[i] = a[i]`）    | ✅            | store 条件统一走 `predicate()`：均匀条件 → `update_mask(cond ? lanes : 0)`（**R2 已修**，不再 emit 崩溃）                                                             | `test_lane_uniform_store_condition_is_broadcast_as_a_predicate`                                                                                                                                                                                                      |
| §2-II     | **条件归约**（`if cond: s[0] += a[i]`）      | ❌（拒绝）         | 跨 lane 归约掩码只承载尾块谓词 → build 期显式拒绝（**R3 已修**，不再静默全 lane 求和）                                                                                  | `test_conditional_reduction_is_rejected`                                                                                                                                                                                                                             |
| §2-II     | **归约旧值重复出现**（`s[0] = s[0] + (s[0] + a)`） | ❌（拒绝）         | 剥离次数必须恰好 1（**R4 已修**）                                                                                                             | `test_repeated_old_value_inside_a_reduction_is_rejected`                                                                                                                                                                                                             |
| §6        | **非 0 循环下界**（`T.serial(1, K)`）        | ❌（拒绝）         | 迭代域建模成 `0 <= x < extent`，下界无法证明为 0 即拒绝（**R5 已修**）                                                                                  | `test_serial_loop_with_non_zero_lower_bound_is_rejected`                                                                                                                                                                                                             |
| §6        | **跨迭代写写别名**（`c[i,2*j]` 与 `c[i,j]`）   | ❌（拒绝）         | 同 buffer 多写必须目的地址相同，否则拒绝吸收 serial 轴（**R6 已修**）                                                                                       | `test_multiple_writes_to_one_buffer_at_different_addresses_are_rejected`                                                                                                                                                                                             |
| §6        | **归约段 + 非均匀 pre/post**                | ✅（轴回退）        | 沿并行轴归约时 pre/post 必须 lane 均匀；不满足则该轴失败并回退（**R7 已修**，build⇒emit 契约恢复）                                                                     | `test_parallel_axis_reduction_with_non_uniform_pre_falls_back`                                                                                                                                                                                                       |
| —         | 数学核心 Δ 分类                           | ✅            | 6 个直接单测                                                                                                                                | `test_finite_difference_classifies_*`、`test_finite_difference_classifies_non_constant_as_piecewise`、`test_const_int_proves_constant_beyond_literals`                                                                                                                 |
| —         | 语句树变换（整段下降）                         | ✅            | 自顶向下替换                                                                                                                                 | `test_vectorize_parallel_nests_*`、`test_multidim_nest_keeps_every_axis`                                                                                                                                                                                              |
| —         | 诊断（改进 7）                            | ✅            | 成败都收集全部卡点                                                                                                                              | `test_diagnostics_list_every_candidate_axis_on_success`、`test_diagnostics_are_attached_to_a_rejection`                                                                                                                                                               |

### 4.1 矩阵的可复现性

`tilelang/ascend/analysis/test_drive_plan.py` 逐条编码上述场景并打印 `OK/REJECT + 原因`（`--tir` 时连带打印发射出的 TIR）。脚本的 `first_parallel` 取**最外层** `T.Parallel`（`post_order_visit` 先出最内层，故取最后一个），与 `build_plan` 需要「嵌套头」一致（§3.1 推论）。代表性输出：

```text
[OK ] c0_copy  c0_broadcast  c1_ifelse  c1_ternary  c1_single_arm  c2_rank
[OK ] c2_reduce  c2_reduce_tail(改进6: update_mask+vsel)  c2_axis_pick(改进2: 选轴 i)
[OK ] c3_multi  c3_scalars  c3_value_pow2(改进3: vshrs+vand)  c3_topk(改进1: vscatter)
[OK ] c4_flat  c4_strided  c4_reverse(改进1+4: PERMUTE→vgather2)  c4_tiled(PIECEWISE→vgather2)
[OK ] c_gather(INDIRECT→vgather2)  c3_im2col(改进2: 最优轴 kk 被拒 → 回退到 m)
[OK ] s1_serial_copy  s1_serial_predicated  s3_serial_reduce            (§6 第一步)
[OK ] s2_serial_scalar  m_serial_after_init  m_serial_before_post         (§6 第二步)
[OK ] m_serial_middle  m_serial_indep_group                              (§6 第二步)
[OK ] p_two_serial_independent(program: 2 段)  p_serial_reduce_then_use    (§6 第三步)
[OK ] p_serial_compose  p_serial_leading_stmt                            (§6 第三步)
[OK ] r1_predicate_width_f16  r2_uniform_condition  r7_reduce_parallel_pre_fallback
[OK ] d16_branch_load_oob(chunk 0 掩码 gather → chunk 1 回退 vld)
[OK ] d16b_prefix_branch_load(chunk 0 回退 vld → chunk 1 掩码 gather)
[REJ] c6_complex      nested serial loop inside T.Parallel is not affine-vectorizable (方案 §6)
[REJ] s_serial_stencil  nested serial loop inside T.Parallel is not affine-vectorizable (方案 §6)
[REJ] p_serial_one_unclean  nested serial loop inside T.Parallel is not affine-vectorizable (方案 §6)
[REJ] c_gm            no vectorizable axis: axis 'i': global-memory operand inside T.Parallel ...
[REJ] c4_true_random  no vectorizable axis: axis 'i': ... FloorDiv by 3 is not a power of two
[REJ] r3_masked_reduce / r4_repeated_old_value / r5_serial_nonzero_min / r6_serial_waw
```

脚本共 **66** 个用例：**55 OK / 11 REJ / 0 ERR**。每个用例的每根轴分类与打分都通过 `plan.diagnostics`（成功，含失败轴的 `-> fallback` 行）或 `Unsupported.diagnostics`（失败）打印。REJ 用例统一描述详见 §3.13.8。

> 用例分组：基线场景（A–F 类）+ `d*`（D1–D16 / A0 已知缺陷复现）+ `r*`（R1–R7 设计层审计回归）。

### 4.2 已知缺陷回归基线（D1–D16 与 A0 均已修，D12 为特征化边界）

`tilelang/ascend/analysis/test_drive_plan.py` 尾部与 `testing/ascend/analysis/test_polyhedral_vectorize.py` 末尾「已知缺陷回归基线（探针 A–M）」小节**成对固化**了下列用例。测试每个都断言**正确契约**：缺陷修复前用 `xfail(strict=True)` 标注（修复后 `XPASS` 强制摘标记），或以「拒绝也接受」的容错形式固化；现在全部转为正常 `pass`。

| 缺陷 | 用例（`drive_plan.py` / pytest） | 现象（修复前） | 类别 | 状态 |
| --- | --- | --- | --- | --- |
| D1 | `d1_product_reduce` / `test_defect_d1_product_reduction_is_not_emitted_as_sum` | `[OK ]`，`reduce.recipe='vcadd'`（乘积→求和） | 静默错误结果 | ✅ **已修**（改为拒绝，`[REJ] unsupported reduction operator(s) Mul`） |
| D2 | `d2_plain_overwrite` / `test_defect_d2_plain_overwrite_is_not_treated_as_reduction` | `[OK ]`，覆写被判归约 | 静默错误结果 | ✅ **已修**（随 D13 改为拒绝，精确诊断 `does not read 's' … accumulate form`） |
| D3 | `d3_int_max_reduce` / `test_defect_d3_int_max_reduction_does_not_crash` | `[ERR]` `ValueError: FloatImm supports only float, but int32` | 发射期崩溃 | ✅ **已修**（`tirx.min_value`，现 `[OK ]`） |
| D3 | `d3_f16_max_reduce` / `test_defect_d3_f16_max_reduction_does_not_crash` | `[ERR]` `ValueError: -3.40282e+38 exceeds minimum of float16` | 发射期崩溃 | ✅ **已修**（现 `[OK ]`） |
| D7 | `d7_uniform_store_mixed` / `test_defect_d7_build_success_implies_emit_success_for_mixed_stores` | `[OK ]` build → `[REJ]` `cannot scalarize vector value 'vload_strided'` | build/emit 契约不一致 | ✅ **已修**（`is_uniform` 挡掉无 args 叶子，现 build+emit 均 `[OK ]`） |
| D4/D5 | `d4_uniform_addend` / `test_defect_d4_uniform_addend_inside_reduction_is_not_dropped` | `[OK ]`，`1.0` 被丢弃 | 静默错误结果 | ✅ **已修**（`_strip_reduce` 沿脊递归，均匀加数保留在 `varying`） |
| D6 | `d6_two_reduces`、`d6b_sum_max_reduce` / `test_defect_d6_two_reductions_in_one_nest_are_both_emitted`、`test_defect_d6b_mixed_reductions_keep_their_own_recipe` | `[OK ]`，只发射第一条归约 | 静默丢写 | ✅ **已修**（逐 store `ReduceInfo`，各自发射累加链 + 一次 `ONEPT vsts`） |
| D8 | `d8_negative_stride`、`d8b_negative_stride_store` / `test_defect_d8_negative_stride_below_minus_one`、`test_defect_d8_negative_stride_store_is_scattered` | `[OK ]`，`vmuls(<uint32 ramp>, -2)` | 静默错误结果 | ✅ **已修**（负步长 → 索引向量 `vgather2`/`vscatter`；读+写两侧） |
| D9 | `d9_symbolic_extent`、`d9b_symbolic_extent_reduce` / `test_defect_d9_symbolic_extent_is_not_silently_truncated`、`test_symbolic_extent_reduction_uses_a_runtime_chunk_loop` | `[OK ]`，`chunks=None` → `range(1)` | 静默少算 | ✅ **已修**（非归约与归约都发运行时 `ceildiv` chunk 循环；归约用 `local.var` 累加器，尾块掩码运行时夹紧） |
| D10 | `d10_one_dimensional_tail` / `test_defect_d10_non_reduction_tail_load_is_bounded`、`test_defect_d10_one_dimensional_tail_load_is_bounded` | `[OK ]`，`vld(access_ptr(a[i,0],64))` 越界（基线 L1322） | 越界读 | ✅ **已修**（尾块拆块 + 装载带谓词，三条 load 通道 + 归约尾块全覆盖） |
| D11 | `d11_indirect_pre`、`d11b_lane_invariant_indirect_load` / `test_defect_d11_build_success_implies_emit_success_for_sibling_statement`、`test_lane_invariant_indirect_load_is_uniform`、`test_bool_store_is_rejected_at_build_time` | `[REJ]` 「outside the serial loop must be uniform … (got indirect)」（每 lane 索引向量通道）；`bool` store → `ValueError` | build/emit 契约不一致 + 能力缺失 | ✅ **已修**：间接下标只在**沿轴变化**时归 `INDIRECT`；`Δ==0`（如 `c[b[0]]`）归 `UNIFORM`，pre/post 发一次 `ONEPT`，serial 主体照常向量化（部分向量化）。不支持的 store 元素类型在 build 期转 `Unsupported` |
| A0 | `d14_partial_reduce`、`d14b_partial_reduce_fallback` / `test_partial_reduction_on_absorbed_serial_axis_is_rejected`、`test_partial_reduction_falls_back_to_the_parallel_axis` | `[OK ]`，`s` 只累加 2/128 项（`ONEPT` 只留 lane 0） | 静默错误结果 | ✅ **已修**（吸收 serial 轴上的部分归约 build 期拒绝；单段路径回退到并行轴） |
| D12 | `d12_masked_branch` / `test_masked_branch_merging_is_rejected` | `[REJ] masked branch merging is not supported` | 未声明路径（特征化） | 已文档化（非缺陷） |
| D13 | 见 D2（判据口径不一致） | — | 设计一致性 | ✅ **已修**（单一判据 `_is_reduce_axis` + `_plan_reduce`，`_detect_reduce`/`_is_accumulate`/`_varying_operand` 已删） |
| R1 | `r1_predicate_width_f16` / `test_predicate_combine_follows_the_data_element_width` | f16 尾块 + 条件：`pand` 用固定 b32，只覆盖 64/128 lane，尾块写一半 | 静默错误结果 | ✅ **已修**（谓词运算掩码跟随数据元素宽度） |
| R2 | `r2_uniform_condition` / `test_lane_uniform_store_condition_is_broadcast_as_a_predicate` | 循环不变 bool 条件被 `vdup` → emit 期 `ValueError` | build/emit 契约不一致 | ✅ **已修**（store 条件走 `predicate()`） |
| R3 | `r3_masked_reduce` / `test_conditional_reduction_is_rejected` | `if cond: s[0] += a[i]` 静默对全部 lane 求和 | 静默错误结果 | ✅ **已修**（build 期拒绝） |
| R4 | `r4_repeated_old_value` / `test_repeated_old_value_inside_a_reduction_is_rejected` | 旧值出现两次被折叠成单个初值，丢一份旧值 | 静默错误结果 | ✅ **已修**（剥离次数 ≠ 1 即拒绝） |
| R5 | `r5_serial_nonzero_min` / `test_serial_loop_with_non_zero_lower_bound_is_rejected` | `T.serial(1, K)` 下界被丢弃 → 下标整体错位 | 静默错误结果 | ✅ **已修**（`_require_zero_min` 拒绝） |
| R6 | `r6_serial_waw` / `test_multiple_writes_to_one_buffer_at_different_addresses_are_rejected` | `c[i,2*j]` 与 `c[i,j]` 跨迭代写同一地址，后写覆盖不成立 | 静默错误结果 | ✅ **已修**（`_serial_is_clean` 拒绝；展平通道同族门禁） |
| R7 | `r7_reduce_parallel_pre_fallback` / `test_parallel_axis_reduction_with_non_uniform_pre_falls_back` | 并行轴归约 + 非均匀 pre：build OK、emit `Unsupported` | build/emit 契约不一致 | ✅ **已修**（build 期拒绝该轴并回退） |
| R8 | （无独立用例，由既有用例覆盖） | 写后读转发在 scatter / 标量写 / 回写后仍可命中陈旧值 | 潜在静默错误路径 | ✅ **已修**（`note_write` 作废该 buffer 的转发项） |
| D16 | `d16_branch_load_oob`、`d16b_prefix_branch_load` / `test_statement_if_arm_load_is_guarded`、`test_statement_if_else_merge_keeps_branch_laziness`、`test_statement_if_prefix_guard_is_dropped_before_boundary`、`test_guard_boundaries_cut_prefix_and_suffix_independently`、`test_statement_if_safe_load_stays_unmasked` | `if i >= 32: c[i] = a[i-32]` 发成无掩码 `vld a[-32]`（越界 32 元素）；同址 `if/else` 合并成急切 `Select`，未选中臂同样被读 | 越界读 | ✅ **已修**（store 谓词 → 装载守卫；惰性 `if_then_else`；`guard_boundaries` 分块，恒真段回退 `vld`） |

> **A5（随 D10 一并修）**：尾块车道数原先直接用 `extent - chunk*lanes`，整 chunk 上会超过向量宽度
> （如 `update_mask(100)` / 64 lane）。现在常量尾块只在尾块块里出现，符号路径用
> `min(extent - chunk*lanes, lanes)` 夹紧。

> **D13 附带新增**：`test_defect_d13_stencil_shifted_read_is_not_an_accumulation`（`s[i] = s[i-1] + a[i,j]`
> 的跨位读与写入地址不同址 → 拒绝）、`test_reduce_segment_with_uniform_broadcast_sibling`（归约段内
> 取值沿轴均匀的 store 走广播写，不参与归约也不被丢弃）。

> **设计层冗余清理（与 R1–R8 同轮）**：
> * 累加器提升决策上移到计划期（`StorePlan.promote`），发射层不再自带 `arith.Analyzer` 重算；
> * 删除死参数 / 死字段：`build_value(..., is_global_ok)`、`_ValueContext.lanes`、
>   `AccessClass.is_index_vector`、`rank_vector_axes(..., stores)`、`_strip_reduce(..., vector_axis)`；
> * 删除不可达的「链内累加」分支与 `_REDUCE_CHAINS`（行为不变）；
> * `_split_nest` / `_try_build_program` 共用 `_absorb_parallel_chain`；
> * `_plan_segment` 的两个字符串开关（`flatten_word` / `axis_note`）合并为 `single: bool`。

复现命令：

```bash
python tilelang/ascend/analysis/test_drive_plan.py --tir &> log_poly.log     # 55 OK / 11 REJ / 0 ERR（66 例）
python tmp/run_poly_test.py testing/ascend/analysis/test_polyhedral_vectorize.py -q
# 期望：135 passed（0 xfailed；D12 为正常 pass 的特征化用例）
```

***

## 5. 扩展性分析

### 5.1 新增一个逐元素算子

- 目标指令已在 `simd.py`：
  - 有专用标量-向量变体 → 加进 `primitives.py` 的 `_VEC_SCALAR_BINARY`（如 `vsub→vsubs`）；
  - 普通二元/一元 → 加进 `primitives.py` 的 `_ELEMENTWISE_BINARY/_UNARY`（键取 TIR 节点名尾段小写）。
  - **发射层无需改动**（`_build` 按 recipe 查表 dispatch）。
- 目标指令是新的 `T.simd.*`：需先在 `simd.py` 注册内建，再在 `emit.py` 的 `_build` 加一个 `recipe` 分支。
- 比较 / 布尔类：改 `primitives.py` 的 `_COMPARE`/`_BOOL_BINARY`/`_BOOL_UNARY`（并注意 `_COMPARE_FLIP`）。
- **成本**：通常 1–5 行 + 1 个测试。

### 5.2 新增一个几何场景

绝大多数情况下**不需要改代码**——只要新场景的访存沿某轴线性化后 Δ 是常数，就会自动落进现有分类。此时**只需补一个测试**固化边界（这正是支持矩阵的产出方式）。

只有两种情况需要动代码：

1. **引入新的 `AccessClass`**（例如 stride 为负 / 分段仿射）→ 改 `access.py` 的 `classify_delta`（分类）+ `plan.py` 的 `rank_vector_axes`（候选/打分）+ `value.py` 的 `load_node`（建节点）+ `emit.py` 的 `_build` / `_emit_store`（发射）。
2. **引入新的语句形态** → 在 `body.py` 的 `normalize_body` 加一个分支。

### 5.3 换发射目标 / 新增硬件指令

因为 `VectorPlan` / `VectorProgram` 指令选择无关，**换 target 只改发射层**：`emit_plan` / `_Emitter._build` / `_Emitter.tail_mask` / `_store_mask` 里的指令名。分析层（§2–§6 的规划部分）完全不动。多段路径只改变**语句编排**（逐段发射、外层轴包一次），不引入任何新的指令选择。

**成本**：与目标指令集规模成正比，与场景数量无关。

### 5.4 路线图：任务 2（§3-II）的三段拆分

§3-II（im2col）本质是**三个耦合特性**，建议分段推进：

| 子特性 | 内容                                | 落点                                                                                                                                                                                                                           | 成本 / 风险        |
| --- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| (a) | `FloorDiv`/`FloorMod` 作**值算子**    | **已实现**（`build_value` + `_build`）：`simd.py` **无整型** **`vdiv/vmod`**（`vdiv` 是浮点 SFU），故对 **2 的幂常量除数**做精确 strength-reduction——`x//2^k → vshrs(x,k)`、`x%2^k → vand(x, 2^k-1)`。仍缺：非 2 的幂常量除数（需 `vmull` magic-number）、非常量除数（需整型内建） | 中；精度/边界敏感      |
| (b) | 非仿射 / 间接下标 → **gather / scatter** | **已实现**（改进 1 + 4）：新增「每 lane 索引向量」通道——`index_value` 在整数上下文构造绝对元素偏移，`load_node` 建 `gather` 节点，`_build` 发 `vgather2`、`_emit_store` 发 `vscatter`；`PERMUTE` / `PIECEWISE` / `INDIRECT` 三类共用该通道                                    | 中；已落地测试        |
| (c) | 非仿射下标的 **UB staging**             | 需**新建一个 pass**：识别非仿射/GM 操作数 → 其前插入 staging copy（UB 分配）→ 改写 nest 内操作数读 staging。会牵动 buffer 分配/别名/同步                                                                                                                            | 大；**接入管线后高风险** |

> (a) 的 2 的幂路径、(b) 的索引向量通道均已完成并落地测试（离线原型内，风险低）；仍缺 (a) 的 magic-number 分支（非 2 的幂常量除数，需 `vmull`）与 (c)。(c) 一旦接入生产管线风险高——且 im2col 是 **gather**（`vgather2` 成本高），文档本身也标注「不确定能否支持」，建议先做收益评估。
>
> 注：im2col 现已能靠**轴回退**（§3.4）沿外层轴 `m` 向量化成功——上面 (a)/(c) 针对的是「沿写连续轴 `kk` 向量化」这条更优但更难的路径，(a) 的 magic-number 分支正是打通该路径的下一步。

### 5.5 新增 reduction 算子

只需在 `primitives.py` 的 `_REDUCE_PRIMITIVES` 里**加一行** `ReducePrimitive(tir_op, op, recipe, scalar, identity)`：算子白名单（`_REDUCE_OPS`）、跨 lane 指令（`_REDUCE_RECIPE`）、标量同构（`_REDUCE_SCALAR_RECIPE`）与单位元（`_reduce_identity`）全部由这张表派生；发射端 `_emit_reduce` 无需改（按 recipe 查表）。

### 5.6 与生产管线接线

当前 `polyhedral` 包是**离线原型**（仅测试引用）。接入路径建议：

1. 在 `--convert-tilelang-to-npu` 之后新增一个 Python pass，对目标 `T.Parallel` 嵌套调用 `vectorize_parallel_nests`；
2. 与 C++ 的 `vectorize_parallel_to_pto.cc` / `ascend_simdvf_lower_parallel.cc` 做**能力对齐或替换**时，先对齐 §4 矩阵的边界；
3. 接线后才需要考虑 pass 排序、与 UB 分配/同步/alias pass 的先后关系。

***

## 6. 边界与已知局限

| 类别  | 局限                                         | 说明 / 潜在改进                                                                          |
| --- | ------------------------------------------ | ---------------------------------------------------------------------------------- |
| 归约  | 短归约（`chunks <= 4`）展开 + 每 chunk 一条链 + 树形合并；长归约 / 符号 extent 走真实循环 + `local.var` 累加器。尾块带谓词装载 + 单位元填充 | `tail != 0` 已支持；符号 extent 也已支持（运行期夹紧掩码）。**条件归约**（`if cond: s += ...`）明确拒绝（R3，可用 `vsel` 填单位元实现） |
| 迭代域 | 循环下界必须可证为 0（R5）                        | 非零下界（`T.serial(1, K)`）目前直接拒绝；建模 `min` 后即可支持                                                  |
| 依赖  | 只检查「吸收的 serial 轴」上的携带依赖              | 跨迭代 WAW 别名（R6）按「同 buffer 多写必须同址」保守拒绝；完整的地址注入性/别名分析未做 |
| 诊断  | 已有结构化 `diagnostics`（改进 7）                  | 成功/失败都收集全部卡点（成功路径也保留 `-> fallback` 原因）；尚无机器可读的 error code                                                    |
| 内存域 | GM 操作数拒绝                                   | 需 UB staging（§5.4(c)）                                                              |
| 值算子 | `FloorDiv`/`FloorMod` 作值算子仅支持 **2 的幂常量除数** | 非 2 的幂常量除数（需 `vmull` magic-number）、非常量除数（需整型内建）仍拒绝（§5.4(a)）                        |
| 地址  | 仅 `TRUE_RANDOM` 拒绝                         | `PERMUTE`/`PIECEWISE`/`INDIRECT` 已走索引向量通道（`vgather2`/`vscatter`）；不可构造的索引明确报错，不静默降级 |
| 守卫  | 装载掩码按 chunk 边界回退 `vld`（D16）              | 只对**仿射且可证**的守卫边界生效（`arith.Analyzer.can_prove`）；数据相关条件、符号 extent 保持整段带掩码 `vgather2`；切点上限 6 个，超出则完全不分段（安全回落）。lane 均匀标量读无掩码形式时仍按 P0-3 在 build 期拒绝 |
| 语句  | 发散分支、atomic 拒绝；serial 支持面有限                | 直接内嵌、链式嵌套、或「一个/多个并列干净 serial 循环夹在语句序列中」可吸收（§6 第一/二/三步）；含循环携带状态、并列段中有一段不干净、非直接内嵌者拒绝。语句级 `if/else` 的臂按**惰性**语义处理：store 谓词即臂内访存的装载守卫，同址两臂合并成惰性 `if_then_else`（D16）。**双臂 `if` 的臂内再嵌 `if`** 走未声明路径 `masked branch merging is not supported`（D12） |
| 缺陷  | **已知设计/实现缺陷（D1–D16 / A0 / R1–R8）**                  | 逐条现象、触发条件与复现用例见 **§4.2**，现已全部修复：D1（非 Add/Max/Min 归约显式拒绝）、D2/D13（归约判据统一为「lane 均匀地址 + 同址累加形式」）、D3（单位元按 dtype 派生）、D4/D5（归约脊上递归剥离旧值、保留均匀加数）、D6（多归约逐 store 发射）、D7（`is_uniform` 挡掉无 args 叶子）、D8（负步长走索引向量）、D9（符号 extent 运行时 chunk 循环 / 归约拒绝）、D10（尾块装载带谓词）、D11（间接地址沿轴不变 → `UNIFORM`，支持部分向量化）、D16（语句级分支臂内访存挂装载守卫 + 惰性 `if_then_else` + chunk 边界回退 `vld`）、A0（部分归约拒绝）；R1（谓词运算掩码跟随元素宽度）、R2（bool 条件走 `predicate`）、R3（条件归约拒绝）、R4（旧值重复出现拒绝）、R5（非 0 下界拒绝）、R6（跨迭代 WAW 拒绝）、R7（归约段 pre/post 均匀性门禁）、R8（写后读转发随写失效） |
| 代码  | ~~累加器提升在 plan 与 emit 各算一遍（emit 另带一个 analyzer）~~ | ✅ 已收敛为 `StorePlan.promote`（计划期单一决策，与 D13 同原则） |
| 代码  | ~~`_REDUCE_CHAINS` 与展开阈值相等 → 「链内累加」分支不可达~~ | ✅ 已删（行为逐指令不变） |
| 代码  | ~~`_split_nest` / `_try_build_program` 各写一遍并行链吸收；`_plan_segment` 用两个字符串开关~~ | ✅ 已抽 `_absorb_parallel_chain`；开关合并为 `single: bool` |
| 代码  | ~~死参数 / 死字段：`is_global_ok`、`is_index_vector`、`rank_vector_axes(stores)`、`_strip_reduce(vector_axis)`~~ | ✅ 已删除（内部 API，无外部消费者）。注：`_ValueContext.lanes` 曾随该轮清理删除，D16 的守卫边界证明与 `[lo,hi)` 地址复核需要按 chunk 换算迭代域，已作为**活跃字段**重新引入（`_guard_free_bounds` / `_access_provably_in_bounds` 使用） |
| 代码  | ~~`_emit_outer_store` 标量快路径直接用未代入的原始下标~~ | ✅ 已改走 `_Emitter.addr_indices(..., uniform=True)`（与向量通道同源） |
| 代码  | ~~`vload.is_uniform` 分支等价于 `False`~~       | ✅ 已随 D7 一并删除（该分支与为它服务的 `AccessInfo._uniform` 动态属性均已移除）                                    |
| 代码  | ~~`AccessInfo._uniform` 动态挂属性~~             | ✅ 已删除（`vload` 特判移除后无引用者）                                                                 |
| 代码  | ~~`_buffer_elem_dtype` 与 `_elem_dtype` 重复~~       | ✅ 已合并（前者删除，改调 `_elem_dtype(buffer.dtype)`）                                                 |
| 代码  | ~~`_is_scalar_dtype` 无任何调用者~~                    | ✅ 已删除                                                                                 |
| 代码  | ~~`vectorize_parallel_nests` 内注释掉的调试打印~~       | ✅ 已删除                                                                                 |
| 代码  | ~~`_const_int` 在取 extent 时未传 `analyzer`~~                     | ✅ 已补（`_build_plan_for_axis`）；影响仍有限：TIR 构造期已折叠常量表达式，严格符号 extent 走 D9 的运行时 chunk 循环 |
| 代码  | ~~`_emit_store`（CONTIGUOUS/UNIFORM 分支）与 `_emit_outer_store` 主体重复~~ | ✅ 已抽 `_emit_vector_write(store, emitter, bits, mask)`；掩码来源仍由调用方区分（chunk 内用 `_store_mask`，pre/post 只用用户谓词） |
| 代码  | ~~`_build` 中 `mask = self.all_mask(_mask_bits(...))` 重复~~ | ✅ 已收敛为 `_Emitter.data_mask(dtype)`（按宽度缓存；`vcvt` 的取小宽度仍特殊处理） |
| 代码  | `Unsupported` 无机器可读 error code | 暂不做：当前没有消费者，加 `code` 要改 20+ 处 `raise` 站点；等接入生产管线、driver 需要区分「回退 / 硬失败」时再加 |
| 模型  | 只做连续性判定，无完整依赖分析/调度                         | 迭代域已建模，可扩展到依赖/调度                                                                   |

***

## 7. 测试策略

- **离线、与硬件/编译器无关**：每个用例构造 TileLang IR，直接跑 `build_plan`（恒返回 `VectorProgram`）/ `emit_plan`，检查 `program.segments[*]`（单段用例取 `segments[0]`）与发射出的 `tl.simd.*` 调用序列。不涉及 PTOAS / Bisheng / 设备执行。
- **分组**：数学核心 6 项 + §0–§6 场景 + 诊断 + 语句树变换 + 统一契约/等价性护栏 + 已知缺陷回归（D1–D16 / A0） + 设计层审计回归（R1–R7） + 段间共享/寄存器直通门禁 2 项 + 装载守卫/分块边界（D16）5 项，共 **135** 个用例（其中 D12 为特征化 `pass`）。
- **已知缺陷基线**：缺陷用例断言**正确契约**（`xfail(strict=True)` 在修复后 `XPASS` 报错，强制同步摘标记；现全部已摘）；对「改为明确拒绝也算修复」的缺陷，用例已把 `Unsupported` 纳入可接受路径。运行预期 `135 passed`。
- **守卫分块回归**：`test_statement_if_arm_load_is_guarded`（后缀：chunk 0 gather / chunk 1 `vld`）、`test_statement_if_prefix_guard_is_dropped_before_boundary`（前缀镜像）、`test_guard_boundaries_cut_prefix_and_suffix_independently`（同段前后缀两个切点 `[1,7]`）、`test_statement_if_else_merge_keeps_branch_laziness`（惰性 `if_then_else`）、`test_statement_if_safe_load_stays_unmasked`（可证安全时零退化）。
- **辅助**：
  - `_first_parallel`：取**最外层** `T.Parallel`（`post_order_visit` 先出最内层，故取最后一个）。
  - `_single_segment`：断言 `build_plan` 恒返回 `VectorProgram` 且恰 1 段，取回 `program.segments[0]`（单段用例经它访问段级字段）。
  - `_emitted_ops`：抽取有序的 `tl.simd.*` op 名。
  - `_recipes`：把 `ValueNode` DAG 展平成 recipe 名列表。
  - `_for_kinds`：统计变换后的循环种类（验证 PARALLEL 是否消除）。
- **运行**（需绕过 editable-install 的 finder 重定向）：
  ```bash
  python tmp/run_poly_test.py          # 内部先移除 ScikitBuildRedirectingFinder
  ruff check --no-cache  <files> && ruff format --check --no-cache <files>
  ```

***

## 8. 术语表

| 术语                  | 含义                                                                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| SCoP                | Static Control Part，多面体模型里的「静态控制域」（迭代域 + 访存函数）                                                                            |
| 有限差分 Δ              | `Δ_v(off) = off(v+1) - off(v)`，连续性的唯一判据                                                                                   |
| `AccessClass`       | 沿某轴的访存形态（连续/均匀/步长/逆序/分片/间接/无结构）                                                                                           |
| 轴回退                 | 按打分排序候选轴后**逐个尝试**发射，某个轴发不出时自动退到次优轴（§3.4）                                                                                  |
| 段 / `VectorProgram` | `T.Parallel` 体内一个干净 serial 链 + 其 `pre`/`post`（一个 `VectorPlan`）；多个并列 serial 时按源码顺序组成 `VectorProgram`，逐段独立向量化 + 源序发射（§3.11） |
| 每 lane 索引向量         | 元素偏移 `offset(..., v=lane)` 的整数向量，供 `vgather2`/`vscatter` 使用（改进 1）                                                         |
| 装载守卫 / `predicate`   | 惰性条件分支里访存的附加掩码：只有条件成立的 lane 才访问内存（P0-3 / D16）                                                       |
| 守卫恒真区间             | `predicate_free_until/from`：守卫对所有 lane 恒为真的首/尾 chunk 段；段内装载可回退普通 `vld`（§3.7.1）                                    |
| `guard_boundaries`  | 所有守卫恒真区间的端点集合；`_strip_mine` 在相邻切点之间分段发射，逐段判定守卫是否可省（§3.7.1）                                            |
| `recipe`            | 值 DAG 节点的原语名（`vload`/`vmuls`/`vsel` ...），发射时查表成 `T.simd.*`                                                                |
| `is_uniform`        | 子表达式是否 lane 内均匀（决定标量-向量特化与 `vdup`）                                                                                        |
| lane                | 向量寄存器中的一个元素槽位                                                                                                             |
| `vreg`              | 向量寄存器，2048 位；`lanes = 2048 / elem_bits`                                                                                   |
| 谓词 / mask           | `boolx256` 类型，CCE 的 P 寄存器，`bits ∈ {b8,b16,b32,b64}`                                                                       |
| NORM / ONEPT / BRC  | store 距离：普通连续 / 单元素 / 广播                                                                                                  |

***

## 9. 参考

- 需求基准：`Parallel_ascend.md` §315–484
- 发射风格权威：`src/ascend/transform/ascend_simdvf_lower_parallel.cc`
- 向量内建：`tilelang/ascend/language/simd.py`
- 场景扫描脚本：`tmp/scan_cases.py`；计划打印脚本：`tilelang/ascend/analysis/test_drive_plan.py`
- 测试套件：`testing/ascend/analysis/test_polyhedral_vectorize.py`

