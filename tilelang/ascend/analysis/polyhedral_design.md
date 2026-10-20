# Ascend 统一多面体向量化 —— 设计文档

> 对应实现：[`polyhedral.py`](./polyhedral.py)（约 2304 行）
> 对应测试：[`testing/ascend/analysis/test_polyhedral_vectorize.py`](../../../../testing/ascend/analysis/test_polyhedral_vectorize.py)（80 个用例）
> 需求基准：[`Parallel_ascend.md`](../../../../Parallel_ascend.md) §315–484「支持场景」
> 参考实现（发射风格权威）：`src/ascend/transform/ascend_simdvf_lower_parallel.cc`

***

## 0. 文档定位

本文档是 `polyhedral.py` 的设计说明，供以下场景使用：

- **学习**：理解模块的数学内核、数据结构与发射逻辑；
- **扩展**：新增算子 / 场景 / 硬件指令时，知道应该改哪一层；
- **评审**：界定「支持 / 拒绝」的边界及其理由，避免静默降级。

本文档描述的是**离线分析原型**：`polyhedral.py` 目前只被它自己的测试引用，尚未接入生产编译管线（见 §5.6）。

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

`VectorPlan` / `VectorProgram` 里**没有任何硬件指令名**——只有轴、lane 数、chunk、每个 store 的值 DAG。**指令选择推迟到** **`emit_plan`**。这条「硬件无关中间表示」的设计使换发射目标（如从自定义 `tl.vmi.*` 改为 `T.simd.*`）只需改发射层。

`build_plan` **恒返回** `VectorProgram`：单段（一个主体 + 可选 `pre`/`post`）时它是「1 段 program」，`program.axes` 只含提到最外层包一次的**并行轴前缀**（`_outer_axes`），其余轴仍由该段自己包裹；`T.Parallel` 体内出现**多个并列的干净** **`T.serial`** **循环**时，program 含多段（每段一个 `VectorPlan`，§3.11）。`VectorPlan` 因此是**段级**中间表示，`VectorProgram` 才是 `build_plan` 对外的统一返回类型。

### 2.2 模块分层

```
polyhedral.py
├─ 0. 原语事实表        _ELEMENTWISE_BINARY/_UNARY, _VEC_SCALAR_BINARY, _COMPARE,
│                       _BOOL_BINARY/_UNARY  —— 唯一保留的 per-op 事实（原语级）
├─ 1. SCoP：迭代域       Axis, AccessClass
├─ 2. 仿射原语           linear_element_offset, finite_difference, classify_delta
├─ 3. 访问分类           AccessInfo, classify_access
├─ 4. body 规范化        StoreSpec, normalize_body, _collect_aliases/_apply_aliases
├─ 5. 值 DAG             ValueNode, build_value, broadcast, load_node, _build_call
├─ 6. 规划               StorePlan, ReduceInfo, VectorPlan, VectorProgram, rank_vector_axes, build_plan
├─ 7. 发射               _Emitter, emit_plan, _emit_reduce, render_plan
└─ 8. 语句树变换         vectorize_parallel_nests
```

### 2.3 数据结构总览

| 结构              | 含义                                | 关键字段                                                                                           | 产出者                                  |
| --------------- | --------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------ |
| `Axis`          | 一根 `T.Parallel` 轴 `0<=var<extent` | `var, extent, index`                                                                           | `_split_nest`                        |
| `AccessClass`   | 沿某轴的访存形态（枚举）                      | `CONTIGUOUS/UNIFORM/STRIDED/PERMUTE/PIECEWISE/INDIRECT/TRUE_RANDOM`                            | `classify_delta` + `classify_access` |
| `AccessInfo`    | 一次访存 + 逐轴分类                       | `buffer, indices, offset, is_store, per_axis, indirect, in_global, reason`                     | `classify_access`                    |
| `StoreSpec`     | 一条写                               | `buffer, indices, value, mask`                                                                 | `normalize_body`                     |
| `ValueNode`     | 值 DAG 节点                          | `recipe, dtype, args, access, scalar, detail`                                                  | `build_value`                        |
| `StorePlan`     | 一条写的完整计划                          | `spec, access, value, mask, kind, index, reduce`                                               | `_build_plan_for_axis`               |
| `ReduceInfo`    | 归约信息（逐 store）                       | `axis, op, recipe, varying, accumulate`                                                        | `_classify_reduce` → `_plan_reduce`  |
| `VectorPlan`    | **段级**硬件无关计划（段 = 主体 + 可选 `pre`/`post`） | `axes, vector_axis, lanes, chunks, tail, stores, pre_stores, post_stores, reduce, diagnostics` | `_build_plan_for_axis`（每段一个）      |
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
  ├─ _try_build_program(stmt, notes)  ▶ VectorProgram?          # 多个并列 serial 段（§6 第三步）
  │     └─ 命中 ≥2 个干净 serial 链 → _build_segment（每段一个 VectorPlan）
  │        任一段构造失败 → 段级原因写入 notes 并 return None（缺陷 B：不吞掉）
  ├─ _split_nest    ▶ axes[], body                # 单段路径（回落）
  ├─ _collect_aliases(body)          ▶ {var: expr}            # 标量中间量
  ├─ normalize_body(body, aliases)   ▶ StoreSpec[]            # 控制流 → 线性写列表
  ├─ rank_vector_axes(axes, stores, store_infos, load_infos) ▶ 轴排行（联合可行性 + 读写加权）
  ├─ for vector_axis in ranked:      # 轴回退：逐个尝试
  │     try: 构造 _ValueContext(axes, vector_axis, lanes, analyzer, vector_dtype)
  │           for each store: build_value(spec.value) ▶ ValueNode
  │           ▶ VectorPlan
  │     except Unsupported: continue
  ▶ VectorProgram(axes=_outer_axes(...), segments=[plan])        # 单段包成 1 段 program

emit_plan(plan)
  ├─ VectorProgram → 逐段 _emit_segment_stmts（源序），program.axes 统一包一次（段内按变量身份跳过）
  ├─ 单独 VectorPlan → 等价的「自己包全部非向量轴」发射路径（护栏测试比对两者逐节点一致）
  ├─ reduce is None → _Emitter 逐 store 发射；重建 chunk 循环 + 外层 serial 循环
  └─ reduce != None → _emit_reduce：编译期展开 chunk，SSA 串累加器
  ▶ TIR (SeqStmt / For)
```

***

## 3. 算法设计

### 3.1 SCoP 建模

**(a) 迭代域** —— `_split_nest` 从某个 `T.Parallel` For 出发，**连续吃掉**所有 `T.Parallel`，一旦遇到非 Parallel 就停：

```python
while isinstance(current, tirx.For) and current.kind == tirx.ForKind.PARALLEL:
    axes.append(Axis(current.loop_var, current.extent, len(axes)))
    current = current.body
```

> 推论：**必须从嵌套头调用**才能拿到全部轴。这是 `vectorize_parallel_nests` 采用自顶向下遍历、测试 helper `_first_parallel` 取最外层的原因。
>
> 注意：并行链会**先**交给 `_try_build_program`（§3.11）——若体内含多个并列的干净 serial 循环，则整体走多段路径；只有不适用时才回落到本节这条单段路径（`_split_nest` + `serial_axis` + `pre`/`post`）。

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

def classify_access(...):                          # 结构细化层（改进 4）
    indirect = any(含 BufferLoad for idx in indices)
    reason   = None if indirect else index_vector_reason(offset, analyzer)
    for axis in axes:
        kind, stride = classify_delta(finite_difference(offset, axis.var, analyzer), analyzer)
        if kind == PIECEWISE and reason is not None:
            kind = TRUE_RANDOM                     # 索引向量构造不出来 → 拒绝
        info.per_axis[axis.name] = (kind, stride)
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
| `IfThenElse` 单臂         | 谓词化：`mask &= cond`                                                                                                             |
| `IfThenElse` 双臂、**同址写** | 合并成 `Select(cond, a, b)`                                                                                                       |
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
4. **归约轴**（store 沿该轴**全部**均匀且**存在**取值沿该轴变化的 store）经 `_is_reduce_axis` 识别后**独占**返回，单独走 `_classify_reduce` → `_plan_reduce`，不再回退到别的轴。

**为什么"打分"和"回退"要分两阶段**：打分只能基于**分析期的形态**（Δ 的分类），但一个轴最终能否发射还取决于**发射期**才能确定的细节——元素偏移能否在「每 lane 索引向量」通道构造出来（如非 2 的幂常量除法就构造不出）、值算子是否有 strength-reduction 等。因此打分只作"优先级"，真正的可行性用 `try/except` 在发射期兜底。

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

命中则构造 `ReduceInfo(axis, op, recipe)`，其中 `op` 由取值的最外层算子在 `_reduce_op`（L1472–1494）里挑：

| 最外层算子 | `op` | `recipe = _REDUCE_RECIPE[op]` |
| --- | --- | --- |
| `Add` | `vadd` | `vcadd` |
| `Max` | `vmax` | `vcmax` |
| `Min` | `vmin` | `vcmin` |
| **其它（含 `Mul`）** | **拒绝**（`Unsupported`） | — |

并用归约轴覆盖 `vector_axis`。

> ✅ **缺陷 D1（已修）**：`_reduce_op`（L1472–1494）原先把未识别算子一律默认成 `vadd`，于是
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
> * 第一层 `_is_reduce_axis`（L1481–1496）——「store 全部 lane 均匀 + 取值沿轴变化」，由
>   `rank_vector_axes`（选轴）与 `_classify_reduce`（建计划）**共用**；
> * 第二层 `_plan_reduce`（L1690）——把单条 lane 均匀 store 精化为累加形式 `dst[...] = dst[...] <op> x`
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
# 实现（polyhedral.py ValueNode.is_uniform，L685–700，D7 已修）
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
| `vcmp` / `vcmps`                     | `vcmp(a,b,mask,op)` / `vcmps(a,s,mask,op)`                 | 谓词                                              |
| `vsel`                               | `vsel(t, f, mask)`                                         | mask 由 `predicate()` 给出                         |
| `pand`/`por`/`pnot`                  | `pand/por/pnot`                                            | 谓词运算                                            |
| `vcvt`                               | `vcvt(src, target, mask)`                                  | mask bits = `min(src, target)`                  |

**写入映射（`_emit_store`）**：`CONTIGUOUS`/`UNIFORM` → `vsts(addr, v, mask, "NORM_B{bits}"/"ONEPT_B{bits}")`；`STRIDED` → `vscatter(v, addr(chunk 基址), vmuls(vci(0,u32), k, m), m)`；`PERMUTE`/`PIECEWISE`/`INDIRECT` → `vscatter(v, index_addr（缓冲区起点）, reinterpret(u32, idx), m)`（索引向量是**绝对**元素偏移，故基址取缓冲区起点）。

**SSA 与** **`Bind`**：每个中间值用 `tirx.Bind(var, expr)` 表示，追加进 `lets` 列表。不能用 `tirx.Let`，因为后者是**表达式**（body 必须是 PrimExpr），而 SSA 中间值是一条条独立语句，只能摊平成 `SeqStmt`。

`_Emitter` 的三个状态容器：

| 容器         | 作用                                                                  |
| ---------- | ------------------------------------------------------------------- |
| `prologue` | **chunk 无关**的绑定，如全 lane 掩码 `pset(bits, "PAT_ALL")`（按 width 缓存，只绑一次） |
| `lets`     | chunk 体内的 SSA 语句                                                    |
| `memo`     | `id(node) → 变量`，同一子表达式去重                                            |

**掩码三层叠加**：`store 的最终掩码 = 用户谓词 ∧ 尾块掩码`，都为空则全 1。尾块掩码 `update_mask(extent - chunk*lanes, bits)`，整除时为 `None`。

> ⚠️ **缺陷 D10（未修，回归基线见 §4.2）**：上述尾块掩码**只加在 store 侧**（`vsts` 带 `update_mask`），
> 而装载 `vld(addr, "NORM")` **没有 mask 形参**（`simd.py` 的 `vld` 签名里根本没有谓词参数）。
> 于是**非归约**的连续读写路径在最后一个 chunk 上会**整 lane 越界读**：例如
> `a: (64, 8)` 上发射 `vld(access_ptr(a[i, chunk*64], 64, 1))`，每行多读 56 个元素
> （`log_poly.log` 第 1322 行 / `p_two_serial_independent`；复现 `d10_non_reduction_tail_overread`）。
> 归约路径靠 `vsel(x, fill, tail_mask)` 兜住了越界车道（§3.8），**非归约路径没有对应兜底**。
> 修法：装载后同样用 `vsel` 把越界车道清零，或要求非归约路径的 `extent` 是 `lanes` 的整数倍。

> ⚠️ **缺陷 D8（未修，回归基线见 §4.2）**：`vload_strided` 的发射式
> `vgather2(addr, vmuls(vci(0,"uint32"), k, m), m)` 用**无符号** lane 斜坡乘步长 `k`，
> 只对 `k > 0` 成立。对 `Δ ≤ -2` 的**负步长**（如 `a[N-2-2*i]`，`Δ=-2`）会静默发出
> 无意义的无符号偏移（复现 `d8_negative_stride`，`vmuls(<uint32 ramp>, -2)`），**结果错误且不报错**。
> 修法：负步长走 `PERMUTE`/索引向量通道（绝对偏移），或显式拒绝 `k < 0`。

> ⚠️ **缺陷 D9（未修，回归基线见 §4.2）**：`extent` 取自 `_const_int`（L1595–1597），
> 只有**编译期可证常量**才会算出 `chunks/tail`；符号 extent 时 `chunks = tail = None`，
> 发射端 `range(plan.chunks or 1)`（L2001 / L2067）退化为**只发一个 chunk 且无尾掩码**，
> 对 `n > lanes` 的符号形状**静默少算**（复现 `d9_symbolic_extent`）。
> 修法：符号 extent 应驱动运行时 chunk 循环，或明确拒绝。

**系统常量**：

- `lanes = 2048 / elem_bits`（`_vreg_lanes`）：float32→64、f16/bf16→128、int8→256；向量寄存器 2048 位。
- 谓词类型一律 `boolx256`（256 位 `vector_bool`）。
- `_mask_bits(dtype) = max(elem_bits, 8)`：CCE 谓词粒度 b8/b16/b32/b64（无 b4）。
- `_MODE = "MODE_ZEROING"`。

### 3.8 归约发射（最特别的部分）

```
mask = pset(bits, "PAT_ALL")
acc  = vdup(identity, elem, mask)               # vadd→0, vmax→-inf, vmin→+inf
fill = tail ? vdup(identity, elem, mask) : None # 尾块填充寄存器（改进 6）
for ck in range(chunks):                        # 编译期展开，不是运行时 for！
    x = vld(src_addr(ck), "NORM")
    if tail and ck == last:                                  # 尾块：越界车道填单位元
        x = vsel(x, fill, update_mask(extent - ck*lanes, bits))
    red = vcadd(x, mask)                        # 跨 lane 归约，结果只在 lane0
    acc = vadd(acc, red, mask)
if 累加写: old = vld(dst_addr, "BRC_B{bits}"); acc = vadd(acc, old, mask)
vsts(dst_addr, acc, mask, "ONEPT_B{bits}")
```

**为什么必须编译期展开？**「跨迭代的寄存器值」在纯 SSA（`Bind`，无 PHI / 可变变量）里无法表达——不能写 `acc = acc ...` 引用上一轮。于是把 chunk 循环**在 Python 层展开**，用不同名字的 `vadd` 串起来。这与 C++ pass 的定型方案一致。

配套细节：

- `_reduce_identity`：`vadd→0`、`vmax→` 元素类型最小可表示值、`vmin→` 元素类型最大可表示值（按 dtype 派生）。
  > ✅ **缺陷 D3（已修）**：`_reduce_identity`（L1713–1730）原先把单位元**硬编码成 f32 极值 `±3.402823e38`**，
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
- **尾块（改进 6）**：`extent % lanes != 0` 时最后一个 chunk 只有 `tail` 个有效车道，其余车道的 `vld` 读到越界/脏数据；用 `update_mask(extent - ck*lanes, bits)` 造尾块谓词、`vsel(x, fill, tail_mask)` 把无效车道替换成**归约单位元**，再按全 lane 掩码归约（`vadd` 填 0、`vmax` 填 -inf、`vmin` 填 +inf 均不影响结果）。这样 `tail != 0` 也能正确发射，不再需要整除前提。

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

`vectorize_parallel_nests`：\*\*自顶向下（pre-order）\*\*遍历语句树，命中 `T.Parallel` 就整体 `emit_plan(build_plan(...))` 替换（`_split_nest` 会一路吃到非 Parallel，故整个 nest 被一次性下降）；整体规划失败才返回 `None` 继续下探子节点。

> 若自底向上替换，最内层会先被换成串行代码，外层 `T.Parallel` 反而再也规划不了（`normalize_body` 撞到已发射的 `For`），导致多维嵌套只下降一半、外层 `T.Parallel` 残留。

### 3.10 诊断体系（改进 7）

规划过程**无论成败都收集全部卡点**，而不是只抛第一个原因：

- `_axis_diagnostics` 为每根轴生成一行分类摘要——候选轴给出 `store=/load=` 形态与联合打分；`TRUE_RANDOM` 轴给出 `rejected (true-random store)` 并附带 `AccessInfo.reason`（如「非 2 的幂常量除法」）。
- 成功时这些行连同 `selected vector axis '<name>' (score=...)` 一起写入 `VectorPlan.diagnostics`，可用 `render_plan` 打印；调用方无需重新推导选择过程。
- 失败时（轴回退逐个尝试全失败 / 无候选轴）`build_plan` 把同样的行 + 每根轴的发射失败原因塞进 `Unsupported.diagnostics`，`str(err)` 仍是单行摘要。

```python
diagnostics = _axis_diagnostics(axes, store_infos, load_infos)   # 每根轴的分类
for vector_axis in ranked:                                       # 轴回退
    try:    plan = _build_plan_for_axis(...)
    except Unsupported as err:
        diagnostics.append(f"axis '{vector_axis.name}': {err}")
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
return VectorProgram(axes=parallel_axes, segments=segments)
```

**每段的规划**（`_build_segment`）与单段完全同构：`axes = [*并行轴, *该段 serial 轴]`，**向量轴强制取该段最内层 serial 轴**（外层并行轴的向量化尚未支持），于是「`pre`/`post` 只能沿 serial 轴向量化」的门禁（§3.3 第二步）天然满足。每段独立产出 `VectorPlan`，各带自己的 `vector_axis` / `reduce` / `pre_stores` / `post_stores` / `diagnostics`。

**发射**（`_emit_program`）：逐段调用 `_emit_segment_stmts(..., wrap_parallel=False)`——每段一个 `chunk_s<k>` 循环 + 该段自身的非向量 serial 轴；**共享外层并行轴只在最外层包一次**；各段掩码 prologue 提到最前；语句顺序 = 源码顺序。`_Emitter` 按段加名字前缀（`s0_`/`s1_`）避免跨段 SSA 重名。

**正确性前提**：段间依赖必须**沿各段向量轴均匀**（Δ=0）。此时每个并行迭代内的段间次序与标量程序一致，逐段向量化不改变结果：

| 形态                                                | 段间依赖                | 是否支持                               |
| ------------------------------------------------- | ------------------- | ---------------------------------- |
| 两个互不相关的 serial（各自 element-wise）                   | 无                   | ✅                                  |
| 归约段 → 广播使用段（`s[i] += …` 后 `c[i,l] = s[i]*2`）      | `s[i]` 沿第二段向量轴 l 均匀 | ✅                                  |
| element-wise 组合（`c[i,j]=a*2` 后 `d[i,l]=c[i,l]+1`） | 同址、沿向量轴等值           | ✅                                  |
| 跨地址真依赖（`c[i,l] = c[i,l-1] + …`，stencil）           | 沿向量轴非均匀             | ❌ 拒绝                               |
| 并列段中**任一段**不干净                                    | —                   | ❌ 整体回落单段路径，按「nested serial loop」拒绝 |

> **不变式（低风险设计）**：多段路径**只新增**接受面，不改动既有行为——`_try_build_program` 一旦返回 `None`（不适用 / 无 ≥2 段 / 任一段构造失败），`build_plan` 走的是与多段支持之前**逐字节相同**的单段代码。这保证既有测试（含各拒绝用例）零 churn，单段被接受场景的发射结果也不变。`render_plan` / `emit_plan` 同时接受两种计划类型。
>
> **统一返回契约（级别 1）**：无论走哪条路径，`build_plan` 都返回 `VectorProgram`——单段路径把结果包成 1 段（`axes=_outer_axes(...)`，仅含提到最外层包一次的并行轴前缀），因而调用方只需处理一种计划类型。该包装不改变发射结果：单独发射一个 `VectorPlan`（自己包全部非向量轴）与发射「1 段 program」产出的 TIR **逐节点结构等价**，由护栏测试 `test_single_segment_program_emits_identically_to_single_plan` 覆盖（含「向量轴就是外层并行轴」的关键用例）。
>
> **诊断不吞（缺陷 A/B）**：多段路径的 program 与每段都带 `diagnostics`（段级含逐轴分类与「强制向量轴」说明，program 级含段数摘要）；某一段构造失败时，段级原因经 `notes` 合并进最终 `Unsupported.diagnostics`，不会只剩回落单段路径后那句泛化的「nested serial loop」。

***

## 4. 场景支持矩阵

图例：✅ 分析+发射均支持 · ⚠️ 部分支持 / 依赖具体形状 · ❌ 明确拒绝（不静默降级）

| 编号        | 场景                                  | 状态           | 机制 / 拒绝原因                                                                                                                              | 对应测试                                                                                                                                                                                                                                                                 |
| --------- | ----------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| §0-I      | Copy / Brc                          | ✅            | Δ 分类为 CONTIGUOUS / UNIFORM                                                                                                             | `test_copy`、`test_scalar_broadcast`                                                                                                                                                                                                                                  |
| §0-II     | GM 操作数                              | ❌            | `global-memory operand ... needs an explicit UB staging copy`                                                                          | `test_global_operand_rejected`                                                                                                                                                                                                                                       |
| §1-I-a    | 单臂 `if`                             | ✅            | 谓词化 store（`mask &= cond`）                                                                                                              | `test_if_single_arm_is_predicated_store`                                                                                                                                                                                                                             |
| §1-I-b    | `if`/`else` 同址写                     | ✅            | 合并 `select` → `vsel`                                                                                                                   | `test_if_else_merges_into_select`                                                                                                                                                                                                                                    |
| §1-II     | 三元：**循环不变**条件                       | ✅            | 广播全 lane 谓词 `update_mask(cond?lanes:0)`                                                                                                | `test_ternary_with_loop_invariant_condition_broadcasts_predicate`、`test_select_node_with_loop_invariant_condition_broadcasts_predicate`、`test_loop_invariant_condition_with_uniform_arm_broadcasts_and_dedups`、`test_select_that_is_uniform_everywhere_stays_scalar` |
| §1-II     | 三元：逐 lane 条件                        | ✅            | `vcmp` 逐 lane 谓词                                                                                                                       | `test_ternary`                                                                                                                                                                                                                                                       |
| §2-I      | 操作数维度不一致                            | ✅            | Δ=0 一侧广播（`vbrc`/`vdup`）                                                                                                                | `test_operand_rank_mismatch_broadcasts_outer_load`                                                                                                                                                                                                                   |
| §2-II     | 规约（`+=` / `max`）                    | ✅            | `ReduceInfo` → `vcadd/vcmax/vcmin`                                                                                                     | `test_reduction_detected`、`test_reduction_max_maps_to_vcmax`                                                                                                                                                                                                         |
| §2-II     | 规约 + 尾块（`extent % lanes != 0`）      | ✅            | 尾块 `update_mask` + `vsel` 单位元填充（改进 6）                                                                                                  | `test_reduction_with_tail_fills_identity`                                                                                                                                                                                                                            |
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
| §6 第三步    | 多个**并列**的干净 serial 循环（≥2 段）         | ✅            | 逐段独立向量化 + 源序发射（`VectorProgram`）：每段各自选向量轴/归约/`pre`/`post`，共享外层并行轴只包一次；段间依赖须沿向量轴均匀                                                       | `test_multi_serial_independent_segments`、`test_multi_serial_reduce_then_use`、`test_multi_serial_compose_keeps_source_order`、`test_multi_serial_leading_statement_is_first_segment_pre`、`test_vectorize_parallel_nests_lowers_multi_serial_program`                   |
| §6        | 嵌套 serial / 跨迭代标量状态 / 并列段中有一段不干净    | ❌            | `nested ... loop ... is not affine-vectorizable`                                                                                       | `test_serial_shared_scalar_rejected`、`test_serial_stencil_rejected`、`test_multi_serial_one_unclean_is_rejected`                                                                                                                                                      |
| §1-I      | **双臂 `if` 的某一臂再嵌 `if`**（臂自带谓词）    | ❌            | `normalize_body`（L616）`masked branch merging is not supported`——**两份文档先前均未声明**（缺陷 D12）                                                 | `test_masked_branch_merging_is_rejected`（特征化）                                                                                                                                                                                                                        |
| §2-II     | **一个 nest 内多条归约**（`s1+=…` 且 `s2+=…`，或求和+求最大混合） | ✅            | 归约描述改为**逐 store**（`StorePlan.reduce`），`_emit_reduce` 遍历全部 store，各起独立累加链 + 各写一次 `ONEPT vsts`（**D6 已修**）                                                             | `test_defect_d6_two_reductions_in_one_nest_are_both_emitted`、`test_defect_d6b_mixed_reductions_keep_their_own_recipe`、`test_reduce_segment_with_uniform_broadcast_sibling`（**已摘 xfail**）                                                                                                                                                                                                  |
| §2-II     | **归约体含 lane 均匀加数**（`s[i] += a[i,j] + 1.0`） | ✅            | `_strip_reduce` 沿归约算子脊递归剥离旧值 → 均匀加数保留在 `varying` 内（**D4/D5 已修**）                                                                                    | `test_defect_d4_uniform_addend_inside_reduction_is_not_dropped`、`test_defect_d5_scaled_old_value_is_rejected`（**已摘 xfail**）                                                                                                                                                                                               |
| §2-II     | **非 Add/Max/Min 的归约**（乘积归约 `s[i] *= a[i,j]`） | ✅（拒绝）        | `_reduce_op`（L1472–1494）**已改**：未识别算子显式拒绝 → 乘积归约不再被发成求和（**D1 已修**）                                                                                    | `test_defect_d1_product_reduction_is_not_emitted_as_sum`（**已摘 xfail**）                                                                                                                                                                                                     |
| §2-II     | **覆写 `s[i] = a[i,j]`（store 沿轴均匀、取值变化）** | ✅（拒绝）        | 统一归约判据：lane 均匀但取值不读 `dst` → 显式拒绝（精确诊断 `does not read 's' … accumulate form`），不再误判成 `vcadd`（**D2 已修**，随 D13）                                                                                    | `test_defect_d2_plain_overwrite_is_not_treated_as_reduction`（**已摘 xfail**）                                                                                                                                                                                           |
| §2-II     | **`int`/`fp16` 的 `max`/`min` 归约**  | ✅            | `_reduce_identity` 按 dtype 派生单位元（`tirx.min_value/max_value`）→ 不再抛 `ValueError`（**D3 已修**）                                                                                    | `test_defect_d3_int_max_reduction_does_not_crash`、`test_defect_d3_f16_max_reduction_does_not_crash`（**均已摘 xfail**）                                                                                                                                                    |
| —         | 数学核心 Δ 分类                           | ✅            | 6 个直接单测                                                                                                                                | `test_finite_difference_classifies_*`、`test_finite_difference_classifies_non_constant_as_piecewise`、`test_const_int_proves_constant_beyond_literals`                                                                                                                 |
| —         | 语句树变换（整段下降）                         | ✅            | 自顶向下替换                                                                                                                                 | `test_vectorize_parallel_nests_*`、`test_multidim_nest_keeps_every_axis`                                                                                                                                                                                              |
| —         | 诊断（改进 7）                            | ✅            | 成败都收集全部卡点                                                                                                                              | `test_diagnostics_list_every_candidate_axis_on_success`、`test_diagnostics_are_attached_to_a_rejection`                                                                                                                                                               |

### 4.1 矩阵的可复现性

`tmp/drive_plan.py` 逐条编码上述场景并打印 `OK/REJECT + 原因`（`--tir` 时连带打印发射出的 TIR）。脚本的 `first_parallel` 取**最外层** `T.Parallel`（`post_order_visit` 先出最内层，故取最后一个），与 `build_plan` 需要「嵌套头」一致（§3.1 推论）。实测输出（改进 1+2+3+4+6+7 全部落地后）：

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
[REJ] c6_complex      nested serial loop inside T.Parallel is not affine-vectorizable (方案 §6)
[REJ] s_serial_stencil  nested serial loop inside T.Parallel is not affine-vectorizable (方案 §6)
[REJ] p_serial_one_unclean  nested serial loop inside T.Parallel is not affine-vectorizable (方案 §6)
[REJ] c_gm            no vectorizable axis: axis 'i': global-memory operand inside T.Parallel ...
[REJ] c4_true_random  no vectorizable axis: axis 'i': ... FloorDiv by 3 is not a power of two
```

共 **36** 个场景：**31 OK**、**5 REJ**（嵌套循环 / 跨迭代 stencil / 并列段含不干净 serial / GM 操作数 / `TRUE_RANDOM`），REJ 用例统一置于列表尾部。每个用例的每根轴分类与打分都通过 `plan.diagnostics`（成功）或 `Unsupported.diagnostics`（失败）打印。

> 上表对应脚本中**基线 36 例**；其后的 `d1`–`d12`（12 例）为下节固化的**已知缺陷复现**，不改变上表结论（其输出顶部打印 `[OK ]` / `[REJ]` / `[ERR]`，与基线 OK/REJ 分列统计）。

### 4.2 已知缺陷回归基线（D1/D2/D3/D4/D5/D6/D7/D13 已修，其余待跟进）

`tmp/drive_plan.py` 尾部与 `testing/ascend/analysis/test_polyhedral_vectorize.py` 末尾「已知缺陷回归基线（探针 A–M）」小节**成对固化**了下列用例。测试每个都断言**正确契约**并挂 `xfail(strict=True)`：缺陷修复后测试会 `XPASS` 直接报错，**强制顺手摘掉 `xfail` 标记**；若缺陷被「以另一种合理方式」修复（如从静默错发改为明确拒绝），用例里已把**拒绝**当作可接受结果一并接受。

| 缺陷 | 用例（`drive_plan.py` / pytest） | 现象（修复前） | 类别 | 状态 |
| --- | --- | --- | --- | --- |
| D1 | `d1_product_reduce` / `test_defect_d1_product_reduction_is_not_emitted_as_sum` | `[OK ]`，`reduce.recipe='vcadd'`（乘积→求和） | 静默错误结果 | ✅ **已修**（改为拒绝，`[REJ] unsupported reduction operator(s) Mul`） |
| D2 | `d2_plain_overwrite` / `test_defect_d2_plain_overwrite_is_not_treated_as_reduction` | `[OK ]`，覆写被判归约 | 静默错误结果 | ✅ **已修**（随 D13 改为拒绝，精确诊断 `does not read 's' … accumulate form`） |
| D3 | `d3_int_max_reduce` / `test_defect_d3_int_max_reduction_does_not_crash` | `[ERR]` `ValueError: FloatImm supports only float, but int32` | 发射期崩溃 | ✅ **已修**（`tirx.min_value`，现 `[OK ]`） |
| D3 | `d3_f16_max_reduce` / `test_defect_d3_f16_max_reduction_does_not_crash` | `[ERR]` `ValueError: -3.40282e+38 exceeds minimum of float16` | 发射期崩溃 | ✅ **已修**（现 `[OK ]`） |
| D7 | `d7_uniform_store_mixed` / `test_defect_d7_build_success_implies_emit_success_for_mixed_stores` | `[OK ]` build → `[REJ]` `cannot scalarize vector value 'vload_strided'` | build/emit 契约不一致 | ✅ **已修**（`is_uniform` 挡掉无 args 叶子，现 build+emit 均 `[OK ]`） |
| D4/D5 | `d4_uniform_addend` / `test_defect_d4_uniform_addend_inside_reduction_is_not_dropped` | `[OK ]`，`1.0` 被丢弃 | 静默错误结果 | ✅ **已修**（`_strip_reduce` 沿脊递归，均匀加数保留在 `varying`） |
| D6 | `d6_two_reduces`、`d6b_sum_max_reduce` / `test_defect_d6_two_reductions_in_one_nest_are_both_emitted`、`test_defect_d6b_mixed_reductions_keep_their_own_recipe` | `[OK ]`，只发射第一条归约 | 静默丢写 | ✅ **已修**（逐 store `ReduceInfo`，各自发射累加链 + 一次 `ONEPT vsts`） |
| D8 | `d8_negative_stride` / `test_defect_d8_negative_stride_below_minus_one` | `[OK ]`，`vmuls(<uint32 ramp>, -2)` | 静默错误结果 | ⏳ 待修 |
| D9 | `d9_symbolic_extent` / `test_defect_d9_symbolic_extent_is_not_silently_truncated` | `[OK ]`，`chunks=None` → `range(1)` | 静默少算 | ⏳ 待修 |
| D10 | `d10_non_reduction_tail_overread` / `test_defect_d10_non_reduction_tail_load_is_bounded` | `[OK ]`，`vld(access_ptr(a[i,0],64))` 越界（基线 L1322） | 越界读 | ⏳ 待修 |
| D11 | `d11_indirect_pre` / `test_defect_d11_build_success_implies_emit_success_for_sibling_statement` | `[OK ]` build → `[REJ]` 「outside the serial loop must be uniform … (got indirect)」 | build/emit 契约不一致 | ⏳ 待修 |
| D12 | `d12_masked_branch` / `test_masked_branch_merging_is_rejected` | `[REJ] masked branch merging is not supported` | 未声明路径（特征化） | 已文档化（非缺陷） |
| D13 | 见 D2（判据口径不一致） | — | 设计一致性 | ✅ **已修**（单一判据 `_is_reduce_axis` + `_plan_reduce`，`_detect_reduce`/`_is_accumulate`/`_varying_operand` 已删） |

> **D13 附带新增**：`test_defect_d13_stencil_shifted_read_is_not_an_accumulation`（`s[i] = s[i-1] + a[i,j]`
> 的跨位读与写入地址不同址 → 拒绝）、`test_reduce_segment_with_uniform_broadcast_sibling`（归约段内
> 取值沿轴均匀的 store 走广播写，不参与归约也不被丢弃）。

复现命令：

```bash
python tmp/drive_plan.py --tir &> log_poly.log     # d* 用例输出见日志 L1651–L2182
python tmp/run_poly_test.py testing/ascend/analysis/test_polyhedral_vectorize.py -q
# 期望：76 passed, 4 xfailed（4 = D8、D9、D10、D11；D12 为正常 pass 的特征化用例）
```

***

## 5. 扩展性分析

### 5.1 新增一个逐元素算子

- 目标指令已在 `simd.py`：
  - 有专用标量-向量变体 → 加进 `_VEC_SCALAR_BINARY`（如 `vsub→vsubs`）；
  - 普通二元/一元 → 加进 `_ELEMENTWISE_BINARY/_UNARY`（键取 TIR 节点名尾段小写）。
  - **发射层无需改动**（`_build` 按 recipe 查表 dispatch）。
- 目标指令是新的 `T.simd.*`：需先在 `simd.py` 注册内建，再在 `_build` 加一个 `recipe` 分支。
- 比较 / 布尔类：改 `_COMPARE`/`_BOOL_BINARY`/`_BOOL_UNARY`（并注意 `_COMPARE_FLIP`）。
- **成本**：通常 1–5 行 + 1 个测试。

### 5.2 新增一个几何场景

绝大多数情况下**不需要改代码**——只要新场景的访存沿某轴线性化后 Δ 是常数，就会自动落进现有分类。此时**只需补一个测试**固化边界（这正是支持矩阵的产出方式）。

只有两种情况需要动代码：

1. **引入新的** **`AccessClass`**（例如 stride 为负 / 分段仿射）→ 改 `classify_delta`（分类）+ `rank_vector_axes`（候选/打分）+ `load_node`（建节点）+ `_build` / `_emit_store`（发射）。
2. **引入新的语句形态** → 在 `normalize_body` 加一个分支。

### 5.3 换发射目标 / 新增硬件指令

因为 `VectorPlan` / `VectorProgram` 硬件无关，**换 target 只改发射层**：`emit_plan` / `_Emitter._build` / `_tail_mask` / `_store_mask` 里的指令名。分析层（§2–§6 的规划部分）完全不动。多段路径只改变**语句编排**（逐段发射、外层轴包一次），不引入任何新的指令选择。

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

`_reduce_op`（算子→`vadd/vmax/vmin`）+ `_REDUCE_RECIPE`（→`vcadd/vcmax/vcmin`）+ `_reduce_identity`（单位元）三处同步扩展；发射端 `_emit_reduce` 无需改（按 recipe 查表）。

### 5.6 与生产管线接线

当前 `polyhedral.py` 是**离线原型**（仅测试引用）。接入路径建议：

1. 在 `--convert-tilelang-to-npu` 之后新增一个 Python pass，对目标 `T.Parallel` 嵌套调用 `vectorize_parallel_nests`；
2. 与 C++ 的 `vectorize_parallel_to_pto.cc` / `ascend_simdvf_lower_parallel.cc` 做**能力对齐或替换**时，先对齐 §4 矩阵的边界；
3. 接线后才需要考虑 pass 排序、与 UB 分配/同步/alias pass 的先后关系。

***

## 6. 边界与已知局限

| 类别  | 局限                                         | 说明 / 潜在改进                                                                          |
| --- | ------------------------------------------ | ---------------------------------------------------------------------------------- |
| 归约  | 展开式需整 chunk，尾块已用单位元填充                      | `tail != 0` 已支持（改进 6）；仍无通用 masked-load                                             |
| 诊断  | 已有结构化 `diagnostics`（改进 7）                  | 成功/失败都收集全部卡点；尚无机器可读的 error code                                                    |
| 内存域 | GM 操作数拒绝                                   | 需 UB staging（§5.4(c)）                                                              |
| 值算子 | `FloorDiv`/`FloorMod` 作值算子仅支持 **2 的幂常量除数** | 非 2 的幂常量除数（需 `vmull` magic-number）、非常量除数（需整型内建）仍拒绝（§5.4(a)）                        |
| 地址  | 仅 `TRUE_RANDOM` 拒绝                         | `PERMUTE`/`PIECEWISE`/`INDIRECT` 已走索引向量通道（`vgather2`/`vscatter`）；不可构造的索引明确报错，不静默降级 |
| 语句  | 发散分支、atomic 拒绝；serial 支持面有限                | 直接内嵌、链式嵌套、或「一个/多个并列干净 serial 循环夹在语句序列中」可吸收（§6 第一/二/三步）；含循环携带状态、并列段中有一段不干净、非直接内嵌者拒绝。**双臂 `if` 的臂内再嵌 `if`** 走未声明路径 `masked branch merging is not supported`（D12） |
| 缺陷  | **已知设计/实现缺陷（D1–D13）**                  | 逐条现象、触发条件与复现用例见 **§4.2**。**已修**：D1（非 Add/Max/Min 归约显式拒绝）、D2/D13（归约判据统一为「lane 均匀地址 + 同址累加形式」）、D3（单位元按 dtype 派生）、D4/D5（归约脊上递归剥离旧值、保留均匀加数）、D6（多归约逐 store 发射）、D7（`is_uniform` 挡掉无 args 叶子）。**待修**：D8（负步长）、D9（符号 extent 少算）、D10（非归约尾块越界读）、D11（build/emit 契约不一致） |
| 代码  | ~~`vload.is_uniform` 分支等价于 `False`~~       | ✅ 已随 D7 一并删除（该分支与为它服务的 `AccessInfo._uniform` 动态属性均已移除）                                    |
| 代码  | ~~`AccessInfo._uniform` 动态挂属性~~             | ✅ 已删除（`vload` 特判移除后无引用者）                                                                 |
| 代码  | `_buffer_elem_dtype` 与 `_elem_dtype` 重复       | ✅ 已合并（前者删除，改调 `_elem_dtype(buffer.dtype)`）                                                 |
| 代码  | `_is_scalar_dtype` 无任何调用者                    | ✅ 已删除                                                                                 |
| 代码  | `vectorize_parallel_nests` 内注释掉的调试打印       | ✅ 已删除                                                                                 |
| 代码  | `_const_int` 只认编译期可证常量                     | 符号 extent 时 `chunks/tail=None` → 静默单 chunk（D9）；传 `analyzer` 时用 `simplify` + `const_int_bound`（上下界相等即常数） |
| 代码  | `_emit_store`（CONTIGUOUS/UNIFORM 分支）与 `_emit_outer_store` 主体近似重复 | 掩码语义不同（后者不含尾块掩码、且发在 chunk 循环外），合并需先统一掩码来源；低优先 |
| 代码  | `_build` 长 if-chain 中 `mask = self.all_mask(_mask_bits(...))` 重复出现 | 风格一致、分支独立，抽取收益有限；低优先 |
| 模型  | 只做连续性判定，无完整依赖分析/调度                         | 迭代域已建模，可扩展到依赖/调度                                                                   |

***

## 7. 测试策略

- **离线、与硬件/编译器无关**：每个用例构造 TileLang IR，直接跑 `build_plan`（恒返回 `VectorProgram`）/ `emit_plan`，检查 `program.segments[*]`（单段用例取 `segments[0]`）与发射出的 `tl.simd.*` 调用序列。不涉及 PTOAS / Bisheng / 设备执行。
- **分组**：数学核心 6 项 + §0–§6 场景 + 诊断 + 语句树变换 + 统一契约/等价性护栏，共 **63** 个用例；另有 **§4.2 已知缺陷回归基线** 17 个用例（D1/D2/D3×2/D4/D6×2/D7/D13×2/广播兄弟/无符号同址等修好后由 `xfail` 转为 `pass`，余 4 个仍 `xfail(strict=True)`，1 个特征化 `pass`），合计 **80** 个。
- **已知缺陷基线**：缺陷用例断言**正确契约**并挂 `xfail(strict=True)`——修复后会 `XPASS` 报错，从而强制同步摘除标记；对「改为明确拒绝也算修复」的缺陷，用例已把 `Unsupported` 纳入可接受路径。运行预期 `76 passed, 4 xfailed`。
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
- 场景扫描脚本：`tmp/scan_cases.py`；计划打印脚本：`tmp/drive_plan.py`
- 测试套件：`testing/ascend/analysis/test_polyhedral_vectorize.py`

