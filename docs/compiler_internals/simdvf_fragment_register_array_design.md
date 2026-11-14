# Ascend SimdVF `alloc_fragment` → 单寄存器数组降级设计（v2）

> **历史文档**：本文记录写作时的实现/设计状态。后续改动已删除或重命名部分
> 文件、测试与行号；当前实现见 `src/ascend/transform/parallel_vectorize/` 与
> `src/ascend/transform/vsimd_*`，以代码为准。


> **⚠️ 已废弃（`p_123_parallel_new_p1` 分支）**：本设计针对的"一个 fragment 拆成
> k 个独立寄存器 buffer"的 fragment 寄存器文件实现已**全部删除**（`vsimd_fragment.{h,cc}`、
> `polyhedral/fragment_plan.{h,cc}`、`FragmentRegisterFile`、`FragmentReg` 等），其前置
> pass `AscendSimdVFallocFragment`（`src/ascend/transform/simdvf_alloc_fragment.cc`）也
> 已连同 FFI 注册与测试一并删除。
> 当前行为：P1 按 Parallel 粒度把触碰到 `local.fragment` 的 Parallel 标记为 OPAQUE，
> 该 nest 走 legacy 标量路径；P2/P3 无 fragment 代码。本文仅作为历史设计记录保留。
> 现行说明见 [vsimd_p1_analysis.md](vsimd_p1_analysis.md)。

> **文档定位**：把当前"一个 fragment 拆成 k 个独立寄存器 buffer"的实现，改为
> "一个 fragment = 一个寄存器数组 buffer"的设计方案与场景分析。
> 本文只做设计论证，不包含代码实现。
>
> **当时的现状实现**：`src/ascend/transform/simdvf_alloc_fragment.cc`
> （每个 chunk 一个 `(lanes,) local.fragment` buffer；该文件已删除）。
> **目标形态来源**：`tilelang/ascend/language/simd.py:243-253`
> （`T.simd.alloc_local`，`(k,) elemXlanes local`）。
>
> **相关既有机制**：
>
> - P2/P3 fragment register mode：`src/ascend/transform/parallel_vectorize/{plan,emit}.cc`
> - P1 访问分类：`src/ascend/transform/vsimd_analysis_common.h`、`vsimd_region.cc`
> - P2 存储分类：`src/ascend/transform/parallel_vectorize/access.cc:89-90`
> - 寄存器预算：`src/ascend/transform/parallel_vectorize/primitives.h:97-101`
> - codegen：`src/ascend/codegen/codegen_ascend.cc`

---

## 0. 结论速览

1. **表示**：`frag (m, n) elem` → `frag_reg (k,) "<elem>x<lanes>" scope="local"`，
   其中 `lanes = 2048 / elem_bits`、`k = prod(shape) / lanes`。这是唯一一个 buffer，
   不是 k 个。
2. **索引映射**：元素线性偏移 `off` → 寄存器下标 `off / lanes`、lane `off % lanes`；
   任意维度按行主序线性化，v1 要求 fragment layout 恒等、`prod(shape) % lanes == 0`。
3. **使用点改写**：把每个使用改写成"寄存器-通道规范形"：
   外层寄存器循环（`k` 次，pass 内展开成常量下标）+ 内层 lane 循环（长度 `lanes`，
   作为后续向量化轴）；fragment 访问变为 `frag_reg[r]`（整寄存器），其它 buffer
   的下标变为 `r*lanes + lane`。
4. **本 pass 不做向量化**：`tl.simd.*` 指令仍由 P3 Emitter（或专门的向量化 pass）
   发射。因此 pass 输出是**中间形态**，必须配套改造消费方；方案对比见 §7。
   进一步地，"使用点改写"与 parallel/tileop 的向量化是**结构性耦合**（向量轴选择、
   部分写 RMW、窄化/NZ、预算都在向量化侧决定），无法用"纯索引映射"切开；
   职责边界与三种划分方案见 §7.5。
5. **预算**：单个 VF 内用户手动寄存器总数 `Σk ≤ 20`（预留 8 个自动临时寄存器，
   与 `kMaxVRegsPerVf=32` / `kManualVRegPool=24` 兼容）；超限的 fragment 不降级。
6. **原子回退**：任一使用不满足映射规则（stride、间接下标、部分 tile、带谓词、
   嵌套控制流、reduce、非恒等 layout）→ 该 fragment 完全不降级，行为与现状一致。
7. **推荐落地形态（§7.6）**：`frag` 保留为逻辑视图（用户写法、元素级使用不动），
   pass 只新增物理寄存器数组 `frag_reg (k,) elemXlanes local` 与映射元数据；
   parallel/tileop 在各自降级时用该数组做向量化发射。索引映射由向量化链路计算，
   pass 不复制。

---

## 1. 需求拆解

| # | 需求 | 设计回应 |
| --- | --- | --- |
| 1 | 单寄存器 256 B = 2048 bit，`lanes = 2048 / dtype_bits` | f32=64、f16/bf16=128、i8=256；与 codegen `ValidateSimdVectorType` 的 2048-bit 约束一致 |
| 2 | v1 要求 `prod(shape) = k * lanes`，不做尾部 mask | 不整除直接不降级；尾部 mask 留待 v2（可用谓词寄存器 + `vsel`） |
| 3 | 用户手动寄存器总量有限制（如 20 个） | 每个 VF 统计 `Σk`，超过阈值则整体不降级（§6） |
| 4 | 任意维度 fragment → 一个寄存器数组；使用点做索引映射 | 行主序线性化 + `(reg, lane)` 映射 + 寄存器-通道规范形（§3/§4） |
| 5 | 该 pass 只负责寄存器定义 + 索引映射，不负责向量化 | 规范形保留 lane 循环，`tl.simd.*` 由消费方发射（§7） |

---

## 2. 现状与硬约束（设计必须满足的事实）

| # | 事实 | 证据 | 对设计的影响 |
| --- | --- | --- | --- |
| F1 | 一颗 V 核向量寄存器固定 2048 bit | `codegen_ascend.cc:413-437` `ValidateSimdVectorType` | `lanes` 唯一确定 |
| F2 | `scope="local"` + 向量 dtype 的 alloc 打印为 `vector_f32 v[k];` | `codegen_ascend.cc:3066-3112` 未拦截，落 `CodeGenC::VisitStmt_(AllocBufferNode)`（`codegen_c.cc:1085-1109`） | 单数组可直接落地，无需改 codegen |
| F3 | `local.var` + 向量 dtype 打印为 `vector_f32 v;`（单个，忽略 shape） | `codegen_ascend.cc:3114-3129` | 现状 k 个 buffer = k 个 C 变量，无法表达数组 |
| F4 | 向量 buffer 的标量下标读写按整寄存器处理 | `GetBufferRef` 对 `local.var` 直接返回变量名（`codegen_ascend.cc:3684-3686`）；`local` 走通用 `buf[i]` | `frag_reg[r]` 是整寄存器语义 |
| F5 | SimdVF 内禁止 generic 向量运算，必须 `tl.simd.*` | `RejectGenericVectorInSimd`（`codegen_ascend.cc:440-447`），调用点含 BufferLoad/BufferStore/binary/select/cast | **向量化只能由 P3 Emitter 之类发 `tl.simd.*` 的组件完成** |
| F6 | P1 只拒绝 `local.var` 访问，`local` 访问可 lift | `vsimd_analysis_common.h:206-230`、`vsimd_region.cc:104-115` | `local` 数组不会被 P1 直接拒 |
| F7 | P2 把 `local` 当"每迭代私有标量" | `access.cc:89-90`、`model.cc:114-120` | 直接降级成 `local` 数组会被 P2 以 "store to private scalar ... varies along vector axis" 拒绝（`plan.cc:2133-2141`） |
| F8 | P2/P3 已有完整 fragment register 发射链路 | `plan.cc` `ClassifyFragmentRegisters`、`emit.h:28-54`、`emit.cc:615-669` | 应复用，不要重写发射器 |
| F9 | P3 当前为每个 slot 建一个 `local.var (1,)` buffer | `emit.cc:615-644` | v2 改为"一个数组 + slot 下标"的主要改动点 |

**关键结论**：把 fragment 直接换成 `scope="local"` 的数组后，**P1 能 lift，但 P2 会把它
当私有标量拒绝**。因此 v2 不是"只写一个 pass"就能闭环，必须同时确定消费方改造（§7）。

---

## 3. 寄存器数组表示与索引映射

### 3.1 定义

```
frag_reg = T.sblock_alloc_buffer((k,), "<elem>x<lanes>", scope="local")
```

| 项 | 规则 |
| --- | --- |
| `lanes` | `2048 / elem_bits` |
| `k` | `prod(shape) / lanes`，要求整除、`k ≥ 1` |
| dtype | 元素类型保持（f32/f16/bf16/i8/i16/i32/u8/u16/u32） |
| layout | v1 要求恒等 `Fragment`（无 padding/replicate/permute），否则不降级 |
| 命名 | `<fragment 名>_reg`（与 P3 现在的 `frag_<name>_<id>_r<slot>` 区分，避免同名） |

### 3.2 线性化与映射

对 fragment 的 N 维下标 `(i0, …, i_{d-1})`：

```
off = ((i0 * n1 + i1) * n2 + i2) …          # 行主序，恒等 layout
reg = off / lanes                            # 0 ≤ reg < k
lane = off % lanes                           # 0 ≤ lane < lanes
```

因此"fragment 的一个值"= "第 `reg` 个寄存器的第 `lane` 条 lane"。
一个访问是**整寄存器可表达**的充要条件是：沿最内层单位步长轴连续遍历 `lanes`
个元素，且这 `lanes` 个元素的 `reg` 相同、`lane` 从 0 到 `lanes-1`。

### 3.3 例子

| fragment | dtype | lanes | k | 说明 |
| --- | --- | --- | --- | --- |
| `(64,)` | f32 | 64 | 1 | 一个寄存器；用户例子 |
| `(128,)` | f32 | 64 | 2 | 两个寄存器 |
| `(8, 64)` | f32 | 64 | 8 | 每行一个寄存器 |
| `(8, 16)` | f16 | 128 | 1 | 128 个 f16 恰一个寄存器 |
| `(128, 128)` | f32 | 64 | 256 | 超预算 → 不降级 |
| `(100,)` | f32 | 64 | — | 不整除 → 不降级 |

---

## 4. 使用点改写规则

### 4.1 寄存器-通道规范形（Canonical Form）

对每个使用 fragment 的 elementwise nest，改写成：

```
for r in <serial|parallel>(k):                 # 寄存器维（pass 内展开为常量）
    for lane in T.Parallel(lanes):             # 通道维（后续向量化轴）
        <body: frag[off] → frag_reg[r]; 其它 buffer 下标 off' → r*lanes + lane>
```

约定（**这是 pass 与向量化 pass 之间的契约**）：

1. 内层 `lane` 循环是唯一的向量化轴，长度恒为 `lanes`；
2. 对 fragment 的访问在规范形里只出现 `frag_reg[r]`，其值语义是"整个寄存器的值"；
3. 内层 body 中 `lane` 只出现在非 fragment 操作数上；
4. 外层 `r` 在 pass 内按常量展开（`r = 0 … k-1`），保证数组下标都是编译期常量。

> **规范形的两种落地方式**：§7 方案 B 由 pass 做文本改写、直接产出上述结构；
> §7 方案 A 保留 element-level 访问，由 P2/P3 的既有 slot 计算完成同一映射
> （slot = off/lanes、lane = off%lanes），本 pass 只保证映射合法并补声明。
> 两种方式共享本节定义的映射契约。

> **为什么不在本 pass 直接发 `tl.simd.*`**：F5 决定 SimdVF 内所有向量运算必须走
> `tl.simd.*`，而把标量表达式映射到正确 recipe（含 mask、窄化、NZ、跨 lane）
> 是 P3 Emitter 的职责；重复实现会引入两套不一致的发射规则（见 §7）。

### 4.2 各类使用的改写

| 类别 | 原形态 | 是否支持 | 改写 |
| --- | --- | --- | --- |
| U1 | `T.Parallel`，最内轴单位步长遍历整个 tile | ✅ | 规范形；最内轴成为 `lane` 轴，其余轴吸收进 `r` |
| U2 | 顶层 `T.serial`，body 全是对该 fragment 的自索引 store | ✅ | 同 U1，循环种类改为 `T.Parallel`（无跨迭代依赖） |
| U3 | `T.copy(frag, ub)` / `T.copy(ub, frag)`，整块 1-D 连续 | ✅ | 每个 `r` 一个 `T.Parallel(lanes)` copy nest |
| U4 | `T.copy(frag, frag2)`（两者都降级） | ✅ | `frag2_reg[r] = frag_reg[r]`（需同 `lanes`、同总长） |
| U5 | `T.reduce_*(frag, …)` | ❌ v1 | 跨 lane/跨寄存器归约，需要新的 recipe；保持不降级 |
| U6 | `T.Parallel` body 内的 tileop（`T.copy`/`T.reduce_*`） | ❌ v1 | 组合语义不明确；保持不降级 |
| U7 | 标量/部分 tile/带谓词/间接下标/stride | ❌ | 保持不降级 |
| U8 | 同一 nest 内多个 fragment | ⚠️ | 要求总长相同、`lanes` 相同、且都被降级；否则全部不降级 |

### 4.3 循环调整规则

设最内层并行轴为 `v`（单位步长轴），其余轴为 outer。

1. 若 `extent(v) == lanes` 且 `off(v=0) % lanes == 0`：`v` 直接作为 lane 轴。
2. 若 `extent(v) > lanes`：`v` 先按 `lanes` 拆分，`v // lanes` 并入寄存器维 `r`。
3. 若 `off` 沿某个 outer 轴变化导致 `r` 变化：该轴保留为外层循环（serial 或 parallel），
   `r` 写成该轴下标的表达式；pass 展开后必须成为常量。
4. 若 `off` 沿 outer 轴变化导致 `lane` 变化（即 `r` 不唯一）：不支持。
5. 若 `extent(v) < lanes` 或不对齐：不支持（v1 不做尾部 mask）。

---

## 5. 场景分析（含 IR）

### 5.1 C1：Parallel 整块填充（`(64,) f32`，k=1）

```python
# before
frag = T.alloc_fragment((64,), "float32")
for i in T.Parallel(64):
    frag[i] = a_ub[i + 32]
```

```python
# after
frag_reg = T.sblock_alloc_buffer((1,), "float32x64", scope="local")
for i in T.Parallel(64):
    frag_reg[0] = a_ub[i + 32]        # 契约：整寄存器写，value 沿 i 向量化
```

### 5.2 C2：Parallel 消费到 UB（`(128,) f32`，k=2）

```python
# before
for i in T.Parallel(128):
    c_ub[i] = frag[i] * 2.0

# after（r 展开）
for i in T.Parallel(64):
    c_ub[i] = frag_reg[0] * 2.0
for i in T.Parallel(64):
    c_ub[64 + i] = frag_reg[1] * 2.0
```

### 5.3 C3：顶层 elementwise serial

```python
# before
for i in T.serial(128):
    frag[i] = frag[i] * 2.0

# after（前提：body 只有该 fragment 的自索引 store）
for i in T.Parallel(64):
    frag_reg[0] = frag_reg[0] * 2.0
for i in T.Parallel(64):
    frag_reg[1] = frag_reg[1] * 2.0
```

### 5.4 C4/C5：整块 copy（tileop）

```python
# before
T.copy(frag, e_ub)         # frag: (128,) f32
T.copy(a_ub, frag)

# after
for i in T.Parallel(64): e_ub[i]        = frag_reg[0]
for i in T.Parallel(64): e_ub[64 + i]   = frag_reg[1]
for i in T.Parallel(64): frag_reg[0]    = a_ub[i]
for i in T.Parallel(64): frag_reg[1]    = a_ub[64 + i]
```

### 5.5 C6：多轴的任意维 fragment（`(8, 64) f32`，k=8）

```python
# before
for i, j in T.Parallel(8, 64):
    frag[i, j] = a_ub[i, j] + 1.0

# after：outer 轴 i 保留为循环，寄存器下标 = i
for i in T.Parallel(8):
    for j in T.Parallel(64):
        frag_reg[i] = a_ub[i, j] + 1.0
```

注意：`frag_reg[i]` 的下标不是编译期常量，需要后续把 `i` 展开（若 `i` 是并行轴，
P2 规划时 slot = `i` 随外层并行轴变化，需要消费方支持"寄存器下标随外层轴"；
v1 建议限制为 **outer 轴是常量展开** 或 **k 较小且在 pass 内展开**，否则不降级）。

### 5.6 C7：reduce（不支持）

```python
# before
T.reduce_max(frag, m_new, dim=0, clear=False)
```

fragment 降级后需要"跨 lane + 跨寄存器"归约，属于新 recipe（`vcmax` + 跨寄存器合并）。
v1 保持不降级，走既有 UB 版本。

### 5.7 C8：Parallel 内 tileop（不支持）

```python
for i in T.Parallel(64):
    T.copy(frag[i], x_ub[i])       # body 内 tileop
```

body 内 tileop 的语义是"每个迭代一次 copy"，与寄存器数组的整寄存器语义冲突；
v1 直接否决。

### 5.8 C9：stride / 部分访问（不支持）

```python
for i in T.Parallel(32):
    frag[2 * i] = a_ub[2 * i]      # lane 不连续
T.copy(frag[0:32], e_ub)            # 部分 tile
```

均不降级。

### 5.9 C10：非恒等 layout

fragment 带 padding / NZ layout（`layout_map` 非 identity）时，`off` 不再等于行主序
线性下标，寄存器/lane 映射不成立；v1 要求 `FragmentLayoutIsIdentity`（与 P2 G6 同判据），
否则不降级。

### 5.10 C11：预算超限

```python
frag_a = T.alloc_fragment((128, 128), "float32")   # k=256
frag_b = T.alloc_fragment((64,), "float32")        # k=1
```

`Σk = 257 > 20` → 两个 fragment 都不降级（或按"逐 fragment"只降级 `frag_b`，
推荐前者：整 VF 原子，避免混合语义）。

---

## 6. 预算与合法性

| 项 | 规则 |
| --- | --- |
| 单寄存器 | 2048 bit |
| 单 VF 用户寄存器上限 | `kUserRegArrayBudget = 20`（可配置），需 `Σk ≤ 20` |
| 自动临时预留 | 保持 `kAutoTempVRegReserve = 8`（`kMaxVRegsPerVf = 32`） |
| 单 fragment 上限 | `k ≤ 20`（且必须能展开） |
| 对齐 | `prod(shape) % lanes == 0`；`off(v=0) % lanes == 0` |
| dtype | 整数 8/16/32、f16/bf16/f32；bool/fp8/fp4 不支持 |
| 回退粒度 | 整 VF 原子：任一 fragment 不满足 → 该 VF 全部 fragment 不降级（避免寄存器/内存混用） |

> 与现有 P2 预算的关系：P2 的 `kManualVRegPool = 24` 是"fragment slot 总数"上限；
> v2 用数组表示后，slot 数 = 数组长度 `k`，二者可以直接对齐。建议 v2 的 20 与
> `kManualVRegPool` 保持独立常量，便于后续按硬件标定。

---

## 7. 与 P1/P2/P3 的衔接（关键决策）

### 7.1 三种候选方案

| 方案 | 做法 | 优点 | 缺点 |
| --- | --- | --- | --- |
| **A：数组作为 P3 寄存器文件载体（推荐分两步）** | pass 只声明数组 + 写映射元数据；fragment 仍保留给 P1/P2 分析；P3 的 `FragmentRegisterFile` 改为"数组 + slot 下标" | 复用全部 P2/P3 规则与发射器；改动局限在 `emit.*` + 元数据 | pass 输出仍含 fragment（与"fragment 消失"的字面目标不同） |
| **B：fragment 完全消失，body 直接访问数组** | pass 把 alloc 换成数组并改写所有使用为规范形；P1/P2 新增 `kRegisterArray` 存储类与判定；P3 直接发数组 | 完全符合字面目标；IR 更干净 | 需要改 P1（存储标签/捕获）、P2（新存储类 + slot 判定）、P3；回归面大 |
| **C：pass 直接发 `tl.simd.*`** | pass 内实现标量→SIMD recipe 映射 | 一步到位 | 与 P3 Emitter 重复，两套规则必然漂移；不推荐 |

### 7.2 方案 A 的改动点（最小闭环）

1. **pass**：为每个符合条件的 fragment 在 VF block 内新增
   `frag_reg (k,) elemXlanes local`，并写 VF annotation：
   `tl.simd.reg_array = [{fragment: frag.data, array: frag_reg.data, lanes, slots}]`；
   fragment 的 `alloc_buffers` 条目保留（供 P1/P2 分析），不删除。
2. **P3 `FragmentRegisterFile`**（`emit.h:28-54`、`emit.cc:615-669`）：
   - `Build` 若 entry 带 array：`regs = {array}`，`Slot(data, slot)` 返回
     `(array, slot)`；
   - `RegisterRead`（`emit.cc:1056-1076`）改为 `BufferLoad(array, {slot})`；
   - `EmitVectorWrite` 的寄存器分支改为 `BufferStore(array, value, {slot})`；
   - `RewriteFragmentAllocations`（`vsimd_group_passes.cc:710-733`）删除 fragment
     alloc，保留（去重）数组 alloc。
3. **P2**：仅需把 annotation 中的数组信息带进 `FragmentReg`（新增可选字段
   `array`）；slot/lane 计算完全不变（仍按元素 offset / `lanes`）。
4. **P1**：无需改动（数组未参与任何 body 访问；作为 alloc capture 的 UNKNOWN
   storage 不影响判定）。但需确认 `vf_alloc_scope` 里的数组 buffer 不会被 G0 误用。

### 7.3 方案 B 的改动点（字面目标）

1. **pass**：fragment alloc → 数组 alloc；所有使用改写为 §4 的规范形；
   删除 `layout_map` 中 fragment 的条目。
2. **P1**：
   - `StorageKind` 增加 `REGISTER_ARRAY`（`local` + 向量 dtype + 静态 shape）；
   - 访问分类允许 `frag_reg[r]` 形式（整寄存器读写）；
   - 记录数组的 `lanes/k`（可从 buffer dtype/shape 推出）。
3. **P2**：
   - `StorageClass` 增加 `kRegisterArray`；`IsPrivateScope` 排除它；
   - `ClassifyFragmentRegisters` 增加数组分支：向量轴长度必须为 `lanes`，
     slot = 数组下标（相对向量轴恒定），`store_lane_mode = full`；
   - `FragmentReg` 直接指向数组 buffer，`slots = k`。
4. **P3**：`RegisterRead`/`EmitVectorWrite` 直接对数组 + slot 发射；
   `RewriteFragmentAllocations` 不删除数组（pass 已声明）。
5. **回退**：P2 拒绝时数组变成 dead alloc（`local` scope 不参与 UB 合并），
   需要 pass 侧保守判定尽量避免，或在 P3 清理。

### 7.4 推荐

- **推荐 §7.6 的"逻辑视图 + 物理寄存器数组"**：用户写法与元素级 IR 完全不变，
  物理分配由本 pass 给出，parallel/tileop 在向量化时消费该分配。它同时满足
  "只有一个寄存器数组""使用点由向量化链路负责映射"两个目标，且不复制向量化逻辑。
- 若后续确实要求 fragment 从 IR 中消失，再按方案 B 演进；方案 C 不采纳。

---

### 7.5 职责边界与耦合分析（"使用改写" vs "并行/tiop 向量化"）

**结论：耦合是结构性的，不是实现细节。** 只要 fragment 的存储变成"整寄存器数组"，
它的每一处使用都必然要被提升为整寄存器操作；而"把标量访问提升成整寄存器操作"
正是 parallel / tileop 两条 lowering 链路各自在做的事（向量轴选择、chunk 拆分、
掩码、窄化、NZ、归约配方）。把这些改写提前到一个 pre-P1 pass 里，等于复制了
向量化决策，后续两边必须始终一致，否则会生成语义不同的两套结果。

#### 为什么不能"只改定义 + 纯索引映射"

1. 寄存器没有单 lane 写语义。`frag[i]` 映射成 `frag_reg[i / L]` 之后，若 RHS 仍是
   标量，IR 就是"标量写向量元素"：codegen 会打印 `frag_reg[i / 64] = value;`
   （`float` 赋给 `vector_f32`），非法。
2. 要让它良构，只有两条路：
   - 把 RHS 变成向量（`frag_reg[r] = <vector expr>`）——这就是向量化；
   - 引入标量视图（同一 data var 的 `(k*L,)` 标量 buffer）——但 data var 的指针
     类型是 `float32x64*`，标量视图的 `frag[i]` 在 codegen 里仍按 `frag[i]`
     打印，越界且类型不符；除非改 codegen 增加"视图=寄存器 lane 插入/提取"语义，
     这同样是向量化能力。
3. 因此 pass 不可能输出"良构但未向量化"的 IR；所谓"只做索引映射"在类型层面
   就已经越过了向量化边界。

#### 具体耦合点

| 决策 | 当前归属 | 若 pass 提前改写使用 | 后果 |
| --- | --- | --- | --- |
| 哪个轴是向量轴 | P2 `RankVectorAxes` / `AxisScore`（按访问模式打分，**不保证最内轴**，`plan.cc:466-497`） | pass 必须自己选（通常只能假定最内轴） | 与 P2/P3 选择不一致，映射错位 |
| chunk 拆分与对齐 | P2 `ClassifyFragmentRegisters` G2（`plan.cc`） | pass 复刻 | 两套对齐规则 |
| 部分写 / 未初始化写 | P2 G8 + P3 `vsel` RMW | pass 需自己判定 | 语义错误（丢旧 lane） |
| 窄化 / pack2 / NZ store | P2 `store_lane_mode`（G5） | pass 需理解 | 与 NZ/pack2 发射冲突 |
| 寄存器预算 | P2.5 `PlanVfFragmentRegisters` + `kManualVRegPool`（`fragment_plan.cc:243`） | pass 再算一遍 | 双重预算、阈值漂移 |
| `T.copy` 读/写 fragment | `PlanCopyItem` **要求两侧都是 UB**（`vsimd_group_passes.cc:236-246`），fragment 源/目的整组拒绝 | pass 改写成 parallel copy nest | 与 tileop 的 copy 语义/广播/stride 规则重叠 |
| `T.reduce_*` 读 fragment | P2 显式拒绝（"reduction into a fragment register has no register form"，`plan.cc:2144-2148`） | pass 需发明跨寄存器归约 recipe | 属于发射器能力，不应在 pass 里做 |
| 并行体内 tileop | `ExpandParallelTileOps`（`body.cc:547`）先展开成循环 | pass 需处理组合语义 | 职责重叠 |

#### 三种职责划分

| 方案 | pass 职责 | 使用点归属 | 优点 | 代价 |
| --- | --- | --- | --- | --- |
| **D1（推荐先做）** | 只做寄存器数组定义（含命名/预算），**不改写元素级访问** | P2/P3 既有 slot/lane 逻辑与 EmitPlan；tileop 仍走 UB/拒绝 | 零重复；复用全部已验证规则 | fragment 在 P1/P2 期间仍存在（与"定义消失"的字面目标不同） |
| **D2** | pass 成为唯一向量化者：定义 + 使用 + `tl.simd.*` 发射 | 删除 P2/P3 fragment 分支、tileop 对 fragment 的分支 | 架构单一 | 等于把 P2/P3 搬进 pass，回归面巨大 |
| **D3（两阶段契约）** | pass 产出"lane 显式"中间表示（`frag_reg[off/L]` + lane 轴） | P1/P2/P3 与 tileop lowering 改造为消费该表示 | 长期干净、fragment 可完全消失 | 需要同时改四条链路并冻结契约，落地周期长 |

#### 推荐边界（可直接作为实现约束）

- **本 pass 拥有**：寄存器数组的声明、命名、`lanes/k` 计算、预算（`Σk ≤ 20`）、
  以及把"P2 不可规划的 elementwise 用法"归一化成可规划形态
  （`T.serial` → `T.Parallel`、整块 `T.copy` → parallel copy nest）。
  这些改动不改变访问的元素语义。
- **parallel 链路拥有**：向量轴选择、chunk 拆分、mask/RMW、`store_lane_mode`、
  寄存器 slot 计算与 `tl.simd.*` 发射（P1/P2/P3）。
- **tileop 链路拥有**：`T.copy`/`T.reduce_*` 的展开与配方；fragment 参与的 tileop
  在 v1 一律不降级（回退），不在本 pass 里发明新 recipe。
- **禁止**：pass 里出现"因为要发 `vadd/vsel/vsts` 所以这样改写"的逻辑；一旦出现，
  就说明它已经侵入向量化职责（D2），应改为 D3 的显式契约另行推进。

### 7.6 推荐落地：逻辑视图 + 物理寄存器数组

> 本节是 §7.5 中 D1 的具体化，也是当前推荐的最终形态：**用户写法不变，pass 只补
> 物理分配，向量化链路在发射时使用该分配**。

#### 7.6.1 IR 形态

```python
with T.sblock("SIMD_VF", no_realize=True):
    # 逻辑视图：用户写法，元素级语义，所有使用保持不变
    frag = T.sblock_alloc_buffer((64,), scope="local.fragment")
    # 物理分配：pass 新增，一个 fragment 一个数组
    frag_reg = T.sblock_alloc_buffer((1,), "float32x64", scope="local")
    T.sblock_attr({"tl.simd.reg_alloc": [
        {"fragment": frag.data, "array": frag_reg, "lanes": 64, "slots": 1},
    ]})
    T.attr("simdvf", "tl.simdvf_scope", 1)
    for i in T.Parallel(64):
        frag[i] = a_ub[i + 32]                     # 不改写
    for i in T.serial(128 // 64 * 64):             # 仅循环种类归一化（见 7.6.3）
        ...
```

要点：

1. **逻辑 fragment 不消失**：P1/P2/P3 的 fragment 捕获、G0–G8、V1–V4、布局推断
   全部继续按 `local.fragment` 工作。
2. **物理数组只被发射层使用**：P1/P2 阶段它没有任何 body 访问（只是 alloc +
   元数据），因此不参与访问分类，也不会改变 slot 判定。
3. **映射不入 body**：元素→(slot, lane) 仍由既有规则计算
   （`RegisterSlotOf`、tileop 的 vreg 分块），只是写入目标从 k 个 `local.var`
   换成 `array[slot]`。

#### 7.6.2 pass 的职责（只有 4 项）

| # | 职责 | 说明 |
| --- | --- | --- |
| 1 | 计算 `lanes = 2048 / elem_bits`、`slots = prod(shape) / lanes` | 静态 shape、整除、dtype 合法性 |
| 2 | 校验 layout 恒等（G6 同判据） | 非恒等不分配 |
| 3 | 预算 `Σslots ≤ 20`（每 VF） | 超限不分配 |
| 4 | 声明物理数组并写 `tl.simd.reg_alloc` 元数据 | 命名、`lanes/slots` |

可选的第 5 项：把 P2 无法规划的 elementwise 用法做**循环种类归一化**
（`T.serial` → `T.Parallel`、`T.parallel(1)` wrapper 透明化），不改访问索引。
不属于"索引映射"的改写一律不做。

#### 7.6.3 parallel 链路如何消费

| 阶段 | 改动 |
| --- | --- |
| P1 | 无（fragment 与元素级访问都不变） |
| P2 | `FragmentReg` 增加 `array` 字段（或 VF 级 `fragment_data → Buffer` 映射）；`slots/lanes` 直接取元数据；slot 计算复用 `RegisterSlotOf` |
| P3 | `FragmentRegisterFile` 由"k 个 `local.var`"改为"一个数组 + slot 下标"：`RegisterRead` → `BufferLoad(array, {slot})`、寄存器写 → `BufferStore(array, value, {slot})`；`RewriteFragmentAllocations` 删除 fragment alloc 并保留数组 |
| codegen | 无（`local` + 向量 dtype → `vector_f32 frag_reg[k];`） |

**关键点**：整条链路不需要新的映射数学，只换"寄存器载体的表示"。

#### 7.6.4 tileop 链路如何消费

| tileop | 现状 | 需要的改动 |
| --- | --- | --- |
| `T.copy(frag, ub)` / `T.copy(ub, frag)` | `PlanCopyItem` 要求两侧都是 UB（`vsimd_group_passes.cc:236-246`），fragment 源/目的整组拒绝 | 当 fragment 有物理数组时放开该限制：合成的 copy nest 仍以元素级 fragment 访问，直接命中 P2 的 fragment register 模式，P3 发射 `frag_reg[slot]` 读 + UB 侧 `vld/vsts` |
| `T.copy(frag, frag2)` | 同上（UB-only） | 两侧都有物理数组时，发射为 `frag2_reg[slot] = frag_reg[slot]`（要求同 lanes、同总长） |
| `T.reduce_*(frag, dst)` | P2 显式拒绝（`plan.cc:2144-2148`） | v1 仅支持"归约轴完全落在单个寄存器内"（k=1 或轴内跨 lane），复用现有 `vcadd/vcmax` recipe；跨寄存器归约（k>1 的整维归约）需要 k-1 次寄存器合并，列为后续能力，其余回退 |
| Parallel 体内的 tileop | `ExpandParallelTileOps` 先展开成循环 | 展开后若访问 fragment，天然命中 register 模式，无需额外改动 |

#### 7.6.5 回退与预算

- 任一 fragment 不满足 §7.6.2 → 该 VF 不分配物理数组，全部走现有路径。
- P2 拒绝该 VF（例如手写 `T.simd.*` 逃逸、layout 非恒等）→ 逻辑 fragment 走 legacy
  或标量路径；物理数组若已 alloc 则成为 dead alloc（`local` 不参与 UB 合并），
  建议在 P3 对"无 register 消费者的 VF"做一次清理，或让 pass 只登记元数据、
  由 P3 在采纳 register 模式时物化 alloc。
- 预算元数据是唯一真相：P2.5 不再重复计算，只校验 `Σslots ≤ kManualVRegPool`。

#### 7.6.6 与当前 v1 实现的差异

| 项 | v1（已实现） | 本方案 |
| --- | --- | --- |
| fragment 定义 | 拆成 k 个 `(lanes,) local.fragment` | 保留原 fragment + 1 个物理数组 |
| 使用改写 | 重写为 k 个 chunk 循环（元素→chunk） | 不改写（仅循环种类归一化） |
| 索引映射 | pass 内 `ChunkRewriter` 文本改写 | P2/P3/tileop 在向量化时计算 |
| 向量化 | P2/P3 把每个子 fragment 当 1 个 slot | P2/P3 把物理数组当 k 个 slot |
| pass 复杂度 | 高（循环拆分 + copy 拆分 + DeclBuffer/layout 清理） | 低（分配 + 元数据 + 预算） |

#### 7.6.7 风险与开放问题

1. **映射实现必须单点**：parallel 与 tileop 两条链路都要用 `off → (slot, lane)`，
   建议提取一个公共头（`RegisterArrayMap`），避免两套规则漂移。
2. **死数组**：P2 拒绝时物理数组无消费者；需要清理或延迟物化。
3. **reduce 跨寄存器**：属于新发射能力，不在本 pass 范围。
4. **手写 `T.simd.*` 逃逸**：fragment 同时被手写 SIMD 使用时，P2 V1 会拒绝整个 VF，
   物理数组失去意义；pass 无法预知，需接受回退。
5. **元数据存活**：P1/P2/P3 都会重写 VF block 的 annotations，需要确认写入
   `tl.vsimd.program` / `tl.vsimd.group_plan` 时保留 `tl.simd.reg_alloc`
   （P1 是增量 Set，已验证；P3 需在替换时保留）。
6. **与 `AddWrapperForSingleBufStore` 的先后**：该 pass 在管线更早位置运行，
   归一化它的 `T.parallel(1)` wrapper 属于 v1 已处理的细节。

## 8. 实施步骤（建议）

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| S0 | 常量与元数据：`kUserRegArrayBudget=20`、`tl.simd.reg_alloc` schema、公共映射头 `RegisterArrayMap` | 单测：元数据可读、映射函数单点 |
| S1 | pass v2：**只声明物理数组 + 元数据 + 预算校验**（不改 body） | `frag (64,)` → `frag_reg (1,) f32x64 local`；IR 结构断言 |
| S2 | P3 `FragmentRegisterFile` 支持数组载体；`FragmentReg` 透传数组 | 生成 `vector_f32 frag_reg[k];`，无 `local.var` slot 声明 |
| S3 | tileop 放开：`PlanCopyItem` 接受有物理数组的 fragment；reduce 限"单寄存器内归约" | 现有 `p_simdvf_alloc_fragment.py` 完整向量化（含 `T.copy(frag2, e_ub)`） |
| S4 | 回退清理：无 register 消费者的 VF 清理 dead 数组 | fallback 场景无冗余分配 |
| S5 | （可选）方案 B：P1/P2 数组存储类 + 访问改写 | fragment 从 IR 消失，数组直接承载 |

---

## 9. 测试与验收

1. **结构用例**（离线，`testing/ascend/transform/`）：
   - `(64,)` → k=1；`(128,)` → k=2；`(8,64)` → k=8；
   - 不整除 / 超预算 / 非恒等 layout / stride / reduce / Parallel 内 tileop → 不降级；
   - 多 fragment 同 nest：一致时降级，不一致时整体不降级。
2. **端到端**：`p_simdvf_alloc_fragment.py` 生成
   `vector_f32 frag_reg[2];` 且 fragment 侧无 `vld/vsts`；与 v1（k 个独立变量）逐值一致。
3. **回归**：`testing/ascend/transform` 全绿；`testing/ascend` 中与设备无关的用例全绿。
4. **约束验证**：`Σk > 20` 时不生成数组；`m*n % lanes != 0` 时不生成数组。

---

## 10. 风险与开放问题

1. **ccec 对寄存器数组的支持**：`vector_f32 v[k];` 是否要求所有下标为常量？若不支持
   动态下标，需要在 pass 内展开 `r`（§4.1 已按此设计）。
2. **方案 A 的 IR 冗余**：数组与 fragment 同时存在，需确保 P3 一定会删除 fragment
   alloc；若 P2 拒绝，数组成为 dead alloc，需要清理或在 pass 里避免。
3. **任意维度的寄存器下标**：当 outer 轴映射到 `r` 且不是常量时（C6），需要消费方支持
   "slot 是外层轴表达式"；v1 建议限制为可展开的常量。
4. **与 `AddWrapperForSingleBufStore` 的交互**：`T.parallel(1)` wrapper 需要在映射时
   透明穿透（v1 已处理）。
5. **尾部 mask**：v2 可用 `pset`/`vsel` 实现非整除尾块；当前不设计。
6. **reduce**：跨寄存器归约需要跨寄存器合并指令，属于独立能力项。
7. **预算标定**：20 是保守初始值，需按 dav-3510 实测寄存器文件大小校准。

---

## 11. 代码索引

| 主题 | 位置 |
| --- | --- |
| 当前 v1 实现（k 个独立 buffer） | `src/ascend/transform/simdvf_alloc_fragment.cc` |
| 2048-bit 校验 | `src/ascend/codegen/codegen_ascend.cc:413-437` |
| `local` / `local.var` alloc codegen | `src/ascend/codegen/codegen_ascend.cc:3066-3162` |
| generic 向量运算拒绝 | `src/ascend/codegen/codegen_ascend.cc:440-447` |
| P1 `local.var` 拒绝 | `src/ascend/transform/vsimd_analysis_common.h:206-230`、`vsimd_region.cc:104-115` |
| P2 存储分类 | `src/ascend/transform/parallel_vectorize/access.cc:89-90` |
| P2 私有标量判定 | `src/ascend/transform/parallel_vectorize/model.cc:114-120` |
| P2 fragment register 规划 | `src/ascend/transform/parallel_vectorize/plan.cc`（`ClassifyFragmentRegisters`） |
| P3 寄存器文件 | `src/ascend/transform/parallel_vectorize/emit.h:28-54`、`emit.cc:615-669` |
| P3 寄存器读 | `src/ascend/transform/parallel_vectorize/emit.cc:1056-1076` |
| P3 寄存器分配替换 | `src/ascend/transform/parallel_vectorize/vsimd_group_passes.cc:710-733` |
| 预算常量 | `src/ascend/transform/parallel_vectorize/primitives.h:97-101` |
| 管线插入点 | `tilelang/ascend/pipeline.py:126-130` |
| 现有实现文档 | `docs/compiler_internals/simdvf_alloc_fragment_lowering.md` |
