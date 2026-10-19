# Ascend `T.Parallel` 分层向量化方案评审报告

---

## 0. 摘要（30 秒版）

今天，`T.SimdVF` 里的 `T.Parallel` 循环由**一个 pass、一个大 if 链**逐个算子、逐种 TileOp 特判地向量化，维度上限 2D、长度必须整除，加一个算子或一个维度都要动这一坨。

方案把它拆成**三层 + 两个中间层**：

- **P1** 把循环 lift 成硬件无关的数组 IR；
- **P2** 把它计划成 256B 分块方案；
- **P3** 按 recipe 发射真实指令。

case-by-case **不会消失**，但从「组合式膨胀」收敛为「原语级局部规则」（第 5 节正面回答）。本次只做 **Stage1**（能向量化则向量化、无 cost model、单 plan），Stage2/3 仅预留。

---

## 1. 背景：我们在向量化什么

### 1.1 两个名词

```python
with T.SimdVF():                          # 进入「寄存器级 SIMD」区域
    for i, j, k in T.Parallel(2, 3, N):   # 声明：这三个维度无依赖，可以向量化
        x = A[i, j, k] * alpha
        C[i, j, k] = x + B[i, j, k]
```

- `T.SimdVF()`：一段跑在昇腾向量单元（AIV）上的代码块，`T.simd.*` 这类寄存器指令只能在这里面用。
- `T.Parallel(...)`：声明「这些维度之间没有依赖，可以并行/向量化」。它**不指定怎么向量化**——那是编译器的工作。
- 我们的目标：把上面这段，变成一条条真实向量指令（`vld` 一次读 64 个 fp32、`vmul`/`vadd` 逐 lane 算、`vsts` 写回）。

### 1.2 今天的问题：一个「大 if 链」

现状代码（`ascend_simdvf_lower_parallel.cc`，简化）大致长这样：

```cpp
// 每个算子硬编码一段
if (是 Add) return vadd(...);
if (是 Sub) return vsub(...);
if (是 Mul) return vmul(...);
if (是 exp) return vexp(...);
// 还有 TileOp 特判：
if (是 fill)   return LowerFill(...);     // 要求满 buffer、长度整除
if (是 copy)   return LowerCopy(...);
if (是 reduce) return LowerReduce(...);
```

它把四件不相干的事揉在一个 pass 里，还带着硬限制：

| 问题 | 具体表现 |
|---|---|
| 每算子一个 case | 加 `abs` 就得在 if 链里再加一段 |
| 维度写死 | `3D+ parallel not supported`（只能 1D/2D） |
| 长度写死 | `extent % 64 == 0`，否则直接报错 |
| 复合表达式爆炸 | `exp(add(mul(...)))` 这种嵌套要补整树 pattern |
| 分析重复 | 维度、别名、连续性的判断在每个 emitter 里各写一遍 |

这就是「诟病 case-by-case」的由来。**但请注意：我们要消灭的不是「per-op 规则」本身，而是「组合式的 per-op 规则」**——这个区别是第 5 节的核心，也是本方案最该被审视的地方。

---

## 2. 核心思想：三层 + 两个中间层

```
  T.Parallel（标量循环）
        │  P1  LiftParallelToArray
        ▼
  ┌─────────────────────────────┐
  │  LogicalArrayRegion         │  ← 中间层 1：硬件无关的数组语义
  └─────────────────────────────┘
        │  P2  AscendPlanArray
        ▼
  ┌─────────────────────────────┐
  │  AscendArrayExecPlan        │  ← 中间层 2：硬件相关的执行计划
  └─────────────────────────────┘
        │  P3  AscendLowerArrayPlan
        ▼
  TIRX + tl.simd.*（真实指令）
```

四句口诀（判断一个新需求落在哪层）：

1. **这是向量化的什么** → P1（lift 语义，唯一）。
2. **这段访问/运算是哪种模式** → 共享分析（硬件无关）。
3. **有没有 recipe 承接它、怎么切块** → P2（查表 + 分块）。
4. **发哪条指令、是否回退** → P3（发射 + 决策）。

---

## 3. 三个 Pass 的责任边界

| Pass | 只做这一件事 | 明确不做 |
|---|---|---|
| **P1** `LiftParallelToArray` | 把 `T.SimdVF` 内的 `T.Parallel` 循环 lift 成数组 IR（迭代域 + 索引映射 + 逐元素值 + 写） | 不碰 256B / lane / 指令名 / target；不碰 TileOp（gemm/copy/reduce） |
| **P2** `AscendPlanArray` | 决定分块（哪个轴向量化、切几块、尾块 mask）并给每个节点绑 recipe id | 不发 `tl.simd.*`；不做模式分类（共享分析的事）；不按表达式形状发射 |
| **P3** `AscendLowerArrayPlan` | 把 plan 里的 recipe 逐节点发射成真实指令 + serial loop | 不重新选轴、不改分块、不重新推断连续性 |

**范围边界（处理什么 / 不处理什么）**：

- ✅ 只处理 `T.SimdVF` 里的 `T.Parallel`。
- ❌ 不处理手写 `T.simd.*`、SIMT（`T.SimtVF`）、TileOp（gemm/copy/reduce 走既有路径）。
- ❌ 不做前置/后置融合：多个 Parallel 之间的 store→load 消去、跨 nest 转发，属于 buffer-versioning 层的职责，不进 P1–P3（见 §8.4）。
- ❌ 不引入 cost model（Stage1）、不跨 VF 融合、不改 `DataType`、不引 MLIR、不加 TIR Stmt 子类。

**P2/P3 为什么分开**：「怎么把逻辑元素分布到寄存器、读多少字节」和「op 的 ABI/dtype/predicate 怎么映射指令」是两件正交的事——前者是分块，后者是 ABI。合在一起，会重新出现「改一条指令要动分块、改分块要重新识别表达式」的纠缠（VPlan 佐证见 §5.3）。

---

## 4. 两个中间层各包含什么（简版）

**`LogicalArrayRegion`（P1→P2）**——只回答「这段循环在算什么」，与硬件无关：

- `domain`：迭代域（每轴的循环变量、下界、范围、步长）。
- `captures`：域外引用的标量/缓冲（如 `alpha`、`A`）。
- `nodes`：拓扑排序的值 DAG——`Index`（坐标）/ `Uniform`（标量广播）/ `Read`（读，含间接下标的依赖）/ `Elementwise`（逐元素算子，含 `Select`）。
- `stores`：有序的写（写哪个 buffer、写到哪、写哪个值）。
- `assumptions`：分析依赖的约束及来源（如 `noalias(A,C)`）。

**关键**：里面**没有任何** 256B、lane、指令名、成本信息——它甚至能反推回等价标量循环。

**`AscendArrayExecPlan`（P2→P3）**——只回答「这段在 950 上怎么落地」：

- `vector_axis` / 分块：哪个轴向量化、切几块、尾块剩多少。
- `value_plans` / `access_plans`：每个值 / 每次访问绑定的 recipe（**是 recipe 标识，不是指令本身**）。
- `predicate_plans`：mask 的物理宽度、有效位置。
- `tail_ub_status`：尾块满读是否被证明在 UB 内。

**关键**：这里有 256B 和分块，但**没有** `tl.simd.*` 调用——指令是 P3 才发的。

---

## 5. 正面回答：P1 还有 case-by-case，必要吗？

这是本方案最该被质疑的一点。正面回答：**必要，而且它是「另一种 case-by-case」——原语级，不是组合级。**

### 5.1 两种 case-by-case 的区别

| | 被诟病的（旧） | 保留的（新） |
|---|---|---|
| 粒度 | 整条复合表达式 / 整棵树 | 单个标量算子原语 |
| 内容 | `exp(add(mul(...)))` 这种整树 pattern | `add` 这一条：arity=2、纯、fp32 语义 |
| 增长 | **组合式**：op × 维度 × 分块 × 索引形状 全交叉 | **线性**：加一个 op = 加一条规则 + 一行 recipe |
| 位置 | 一个 pass 里到处散落 | P1 的 `ElementwiseRule` 表（无硬件名） |

举个具体例子——给系统加一个 `abs` 算子：

- **旧**：要在 emitter 的 if 链里加一段 `abs → vabs`，还要处理 `abs` 和 `mul`/`add` 嵌套时要不要整树 pattern、2D 下 `abs` 的 stride 怎么算……牵一发动全身。
- **新**：P1 的规则表加一条「abs = 一元纯算子」，P2/P3 的 recipe 表加一行「abs fp32 → `simd_vabs`」。**维度、分块、别名分析一行都不用改**——因为它们是 rank-N、且与具体算子无关。

### 5.2 为什么 P1 的 per-op 规则不可消除

1. **原语语义无法推导**。`add` 是二元、纯、遵守 IEEE fp32 舍入——这是「事实」，不是能从别处推出来的。任何编译器都得有地方记下它。
2. **「自动信任所有纯调用」更危险**。若把每个 `kPure` 调用都自动当逐元素算子，就会静默接受语义未知的 op，把不保证的数值行为悄悄带进 SIMD。显式注册是刻意的保守。

### 5.3 前沿做法佐证

LLVM / MLIR / VPlan 都没有「消灭 per-op 规则」，它们做的是**把 per-op 规则限定为原语级、把组合逻辑泛化**：

- **LLVM SelectionDAG / ISel**：每个 opcode 一个 `visitADD/visitSUB/...`，目标侧用 TableGen `.td` 写「一条 opcode → 一条指令」的 pattern。规则是 per-op 的，但**类型合法化（LegalizeTypes）、指令调度、寄存器分配都是与 op 无关的泛化 pass**。
- **MLIR Linalg → Vector → 硬件**：逐 op 的 lowering pattern（`linalg.generic` 的每条 op → vector op → 目标指令），但 tile/pad/distribute 这些变换与具体 op 无关。
- **LLVM VPlan**：先把「要做什么」建成 plan，再执行改写。它把「规划」和「执行」分开，正是本方案 P2/P3 分离的同类思路。

结论一句话：**case-by-case 不消失，但从「组合式膨胀」收敛成「原语级局部规则」。** H3 就是用实验去验证这个收敛是否真的发生（加 op / 加维度 / 改索引形态时，是否只改各自的一类规则、不为组合新增整树 pattern）。

---

## 6. Stage 目标边界

| Stage | 目标 | 关键取舍 |
|---|---|---|
| **Stage 1（本次）** | 覆盖尽量多的 op（add/sub/mul/neg/abs/exp/cast/select…），**不保证全覆盖**；无 cost model；P2 只给一种 plan（第一个合法轴）；P3 简单发射 | 覆盖不到 → 明确 unsupported / 回退标量，不静默。只处理 `T.SimdVF` 内 `T.Parallel` |
| **Stage 2（预留，不保证）** | P2 产出**多个候选 plan** + 选择策略（cost model 或固定算法）；支持 per-op 回退（某 load 标量、其余仍向量） | 数据结构已预留逐节点 `value_plans`/`access_plans` + `enumerate_candidates` 缝，不改 schema |
| **Stage 3（更远期，不保证）** | 把非 `T.SimdVF` 下的 Parallel 纳入，可能与 Simt Parallel 统一 | 标量回退最终统一成 Simt VF；本次不考虑 |

后两个 Stage 只做「预留接口 + 数据结构不阻碍」，不承诺实施。

---

## 7. 完整例子走查（happy path）

拿 §1.1 的三维例子走一遍，看三层各自产出什么。

```python
with T.SimdVF():
    for i, j, k in T.Parallel(2, 3, N):
        x = A[i, j, k] * alpha
        C[i, j, k] = x + B[i, j, k]
```

**P1 输出（数组 IR，与硬件无关）**：

```text
domain   = [(i,0,2), (j,0,3), (k,0,N)]     # 三个轴，N 是符号量
captures = { alpha, A, B, C }
nodes:
  a = read A[(i,j,k)]          : f32
  s = uniform alpha (depends={})
  x = mul(a, s)
  b = read B[(i,j,k)]
  y = add(x, b)
stores:
  C[(i,j,k)] <- y
```

注意：**没有 256B、没有 lane、没有指令名**。它还能反推回等价标量循环。

**P2 输出（执行计划）**：

```text
vector_axis = k, 剩余轴 = [i, j]          # 选最内层连续轴
packet = 64 元素（fp32 满寄存器）
Q = ceildiv(N, 64),  R = N % 64          # 分块数、尾块
value_plans:
  a,b -> recipe=contig_fp32_load (NORM)
  s   -> recipe=scalar_replicate (vdup)
  x,y -> recipe=mul_fp32 / add_fp32
access_plans:
  A/B/C -> Contiguous, footprint=64 元素, 对齐 32B
predicate: 满块 mask=全 64；尾块 mask=[0,R)
```

注意：**有 256B 和分块，但没有 `vld`/`vmul` 这些调用**。

**P3 输出（真实指令）**：

```text
for i in serial(2):
  for j in serial(3):
    for q in serial(ceildiv(N,64)):
      valid = (q*64 + lane < N)
      va = vld(A[i,j,q*64], NORM)
      vb = vld(B[i,j,q*64], NORM)
      vs = vdup(alpha)
      vx = vmul(va, vs, valid)
      vy = vadd(vx, vb, valid)
      vsts(C[i,j,q*64], vy, valid, NORM_B32)
```

这个例子串起了三层：P1 只关心「算什么」，P2 只关心「怎么切块」，P3 只关心「发什么指令」。

---

## 8. 边界情况怎么处理（例子为主）

### 8.1 分支 = 谓词化，不是控制流

```python
A[i] = B[i] if i < n else 0        # i < n 是「对循环变量仿射」的条件
```

`i < n` 不是控制流分支，而是 **mask**：

```text
P1:  v = select(pred=(i<n), B[i], 0)      # 值级 Select
P2:  predicate_plans: (i<n) -> 有效 lane 集
P3:  valid = update_mask(n) 或 pge(...)
     va = vld(B[...], NORM)
     vv = vsel(va, vdup(0), valid)
```

三条规则：仿射谓词 → mask；数据相关谓词（`B[i]>0`）→ `vcmp`+`vsel`；两臂结构发散 → P1 拒绝。

### 8.2 gather：数组层能表达，硬件层明确拒绝

```python
C[i] = A[B[i]]        # A 的下标是 B[i] 的「值」
```

```text
P1:  b = read B[i]
     a = read A[b]          # 下标依赖 b 的值，gather 用 id 显式连边
P2:  strategy=Unsupported，diagnostics="无 gather recipe"
P3:  不执行；参照循环可跑
```

这是刻意分层：数组 IR 不因为「没有硬件 gather」就拒绝表达，但硬件路径诚实报 unsupported。

### 8.3 strided：可表达，但不算「成组」

```python
C[i] = A[2 * i]        # stride=2
```

P2 判成 `PointwiseSIMD`（逐元素），并**明确报告「没有成组向量化」**，不计入成组覆盖率。

### 8.4 reduce / 多 Parallel / TileOp：不进 P1–P3

```python
A[i] += B[i, j]          # reduce：走既有 TileOp 路径，不 lift
```

多 Parallel 之间、以及 Parallel 与 `T.gemm`/`T.copy`/`T.reduce_*` 之间的依赖（如 topk+attention 场景），P1–P3 **每次只处理一个 Parallel nest**，TileOp 是硬屏障，跨 nest 的 store→load 消去归 buffer-versioning。

---

## 9. 风险与已知限制

- **尾块满 256B 读的 UB 物理末端**：尚未在设备上验证，保留 warning，不冒充已证明安全。
- **annotation carrier**：三个 pass 必须连续执行，中间不能插普通优化（集成折中，实验失败可整体回退）。
- **per-op 回退**：Stage1 不做，但数据结构已逐节点化，Stage2 接入不改 schema。
- **H3 是可证伪的假设**：本方案不承诺「代码量必然减少」，只承诺「新增能力影响范围局部」，用组合测试去证伪。

---

## 10. 请评审确认

1. 三层（lift / plan / emit）+ 两个中间层的边界是否成立？
2. P1 保留的原语级 case-by-case 是否可接受？（见 §5）
3. Stage1 的「覆盖尽量多 op、不保证、无 cost model、单 plan」范围是否合理？
4. 分支按「谓词化」处理、gather 明确拒绝、reduce 走 TileOp，这三个边界是否正确？



#  支持场景：
# `T.Prallel` 需求识别.txt
# `T.Prallel` 需求识别

## 0. 基本功能补齐

### Case I

Copy/Brc操作，即Parallel中未进行任何计算的场景

分析：增加无运算符场景的支持

### Case II

直接GM访问类操作，即操作数不是buffer而是GM的场景

分析：可以自动插入缓存申请，但语义上是否应当支持？

## 1. 嵌套分支语句

### Case I

```python
for i_s1, i_s2 in T.Parallel(block_S, block_S):
    if i_s1 < i_s2:
        A_fragment[i_s1, i_s2] = 0
```

```python
for i in T.Parallel(block_M):
    if m_prev[i] == -T.infinity(accum_dtype):
        scale_factor[i] = 0.0
    else:
        scale_factor[i] = T.exp(m_prev[i] - m_i[i])
```

分析：显式的分支结构，对应IfThenElseNode，基本可以迁移if_then_else，但没有else的场景需重新分析

### Case II

```python
for i_s2, i_v in T.Parallel(block_S, block_DV):
    dv_fragment[i_s2, i_v] = (
        dv_fragment[i_s2, i_v] * G_fragment_post[i_s2] if G_last_local - G_fragment[i_s2] <= 0 else 0
    )
```

分析：三元运算结构，暂不确定对应节点，不过同上处理

## 2. 操作数维度不一致

### Case I

```python
for i_s1, i_s2 in T.Parallel(block_S, block_S):
    G_diff_local[i_s1, i_s2] = G_shared[i_s1] - G_shared[i_s2]
```

分析：基于dst的shape分析，需要对操作数进行view操作后再执行运算

### Case II

```python
for i1_m, i1_n in T.Parallel(block_M, block_N):
    o_reducer[i1_m] += a_frag[i1_m, i1_n] * x_frag[i1_n]
```

分析：规约场景，在dst维数小于src，且出现原地运算符时会出现规约场景，应翻译出reduce类操作

## 3. 多行语句

### Case I

```python
for i in T.Parallel(blk_m):
    topk_gates[bx * blk_m + i, k] = max_val[i]
    topk_indices[bx * blk_m + i, k] = max_idx[i]
```

分析：仅仅是写了不止一行，当前被拦截，正确判断并允许即可

### Case II

```python
for i, j in T.Parallel(block_M, block_K):
    k = k_iter * block_K + j
    m = by * block_M + i
    access_h = m % (OH * OW) // OW * S + k // (KW * C) * D - P
    access_w = m % OW * S + k // C % KW * D - P
    in_bound = (access_h >= 0) and (access_w >= 0) and (access_h < H) and (access_w < W)
    data_shared[i, j] = T.if_then_else(in_bound, data[m // (OH * OW), access_h, access_w, k % C], 0)
```

分析：由复杂标量运算+实际指令组成的多行语句，不确定能否支持，可能需要依赖TVM现有能力，待分析

## 4. 表达式下标

### Case I

```python
for i_s, i_k in T.Parallel(block_S, block_DK):
    G_shared[i_s, i_k] = G[bb, bs * block_S + i_s, bh]
```

分析：最典型的表达式下标情景，常见于在tile为单位的copy操作中。当前的思路是设计一个能够分析下标连续性的方法，来因对这种情况，但这个方法逻辑较为复杂，仍需预估工作量。

### Case II

```python
for i_kv in T.Parallel(block_DK * block_DV):
    dg_last_fragment[i_kv] = h_shared[i_kv // block_DV, i_kv % block_DV] * dh_shared[i_kv // block_DV, i_kv % block_DV]
```

```python
for i, j in T.Parallel(block_M, block_N):
    C[by * block_M + i, bx * block_N + j] = C_shared[
        i // micro_size_x,
        j // micro_size_y,
        i % micro_size_x,
        j % micro_size_y,
    ]
```

分析：高维计算结果摊平到低维，该情况下较难分析高维下标的连续性，目前考虑的方案有二：

1. 识别到循环体大于下标范围的这类情况进行特殊处理，分析其高维做特殊的带stride的连续性判定
2. 通过类似于模板匹配的方法处理这种特殊情形

## 5. 未支持的运算符

### Case I

```python
for i, j in T.Parallel(block_M, block_N):
    T.atomic_add(C[by * block_M + i, bx * block_N + j], C_shared[i, j])
```

分析：暂未开放atomic_add系列操作，需要确定此类操作是否已同时支持向量和标量操作？

## 6. 复杂场景

```python
for tm in T.Parallel(block_M):
    for g_i in range(0, block_K // group):
        a_k = g_i * group
        non_zero_cnt[0] = 0
        for i in range(elem):
            non_zero_elt_log_idx[i] = 0
        for i in range(group):
            val = A_shared[tm, a_k + i]
            if val != 0.0:
                non_zero_elt_log_idx[non_zero_cnt[0]] = i
                A_sp_shared[tm, a_k // 2 + non_zero_cnt[0]] = val
                non_zero_cnt[0] += 1
        # TODO: use T.device_assert(non_zero_cnt <= 2) after rebasing main
        if non_zero_cnt[0] == 1 and non_zero_elt_log_idx[0] == 3:
            non_zero_elt_log_idx[0] = 0
            non_zero_elt_log_idx[1] = 3
            A_sp_shared[tm, a_k // 2 + 1] = A_sp_shared[tm, a_k // 2]
            A_sp_shared[tm, a_k // 2] = 0.0
        elif non_zero_cnt[0] == 1:
            A_sp_shared[tm, a_k // 2 + 1] = 0
            non_zero_elt_log_idx[1] = 3
        for i in T.serial(elem):
            val = non_zero_elt_log_idx[i]
            E_shared[tm, a_k // e_factor] |= T.shift_left(val, 4 * (g_i % (e_factor // group)) + 2 * i)
```

分析：太复杂了，暂不支持