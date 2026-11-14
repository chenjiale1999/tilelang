# Ascend SimdVF `alloc_fragment` 降级：寄存器数组（`AscendSimdVFallocFragment`）

> **⚠️ 已删除（`p_123_parallel_new_p1` 分支）**：本 pass 已从代码库中**完全移除**：
>
> - `src/ascend/transform/simdvf_alloc_fragment.cc`（880 行）已删除；
> - `tilelang/ascend/transform/__init__.py` 的 `AscendSimdVFallocFragment()` 包装与
>   `__all__` 条目已删除，`tl.transform.AscendSimdVFallocFragment` 的 FFI 注册已不存在；
> - `pipeline.py` 中被注释的调用已删除；
> - `testing/ascend/transform/test_ascend_SimdVF_alloc_fragment.py` 已删除。
>
> 删除原因：本分支不支持 fragment 向量化——P1 按 Parallel 粒度把触碰到
> `local.fragment` 的 Parallel 判为 OPAQUE 并走 legacy 标量路径，因此“把 fragment
> 切成寄存器大小再逐寄存器向量化”的下游链路已不存在，这个前置 pass 也就没有消费者。
> 本文仅作为历史实现记录保留。现行说明见
> [vsimd_p1_analysis.md](vsimd_p1_analysis.md) 与
> [p1_region_p2_p3_integration_report.md](p1_region_p2_p3_integration_report.md)。

> **文档定位**：记录 `T.SimdVF` 内 `T.alloc_fragment`（scope `local.fragment`）
> 在 P1 之前被降级为"按硬件向量寄存器切分的寄存器数组"的 pass 设计、实现、
> 支持边界与验证结果。
>
> **代码基线**：
>
> - 实现：`src/ascend/transform/simdvf_alloc_fragment.cc`
> - 管线：`tilelang/ascend/pipeline.py:126-130`
> - FFI 包装：`tilelang/ascend/transform/__init__.py:60-70`
> - 相关既有机制：`src/ascend/transform/polyhedral/{plan,emit}.cc`（P2/P3 fragment register mode）、
>   `src/ascend/codegen/codegen_ascend.cc`（`vector_<T>` 声明）、
>   `tilelang/transform/add_bufstore_wrapper.py`（`T.parallel(1)` store wrapper）

---

## 0. 结论速览

1. `T.alloc_fragment` 在 SimdVF 内是**逻辑上的 V 核寄存器 tile**，而一颗硬件向量
   寄存器固定为 **2048 bit（256 B）**：f32 64 lane、f16/bf16 128 lane、i8 256 lane。
   一个 `(128,) float32` fragment 需要 2 个寄存器，但现有 codegen 对
   `local.fragment` 只声明**一个** `vector_f32`，无法表达。
2. 新 pass `AscendSimdVFallocFragment` 位于 `AscendLiftVSimd`（P1）之前，把
   `R = prod(shape) / lanes > 1` 的 fragment 拆成 `R` 个寄存器大小的 fragment
   （`frag_r0 … frag_r(R-1)`，shape `(lanes,)`，scope 仍为 `local.fragment`）。
3. 仅拆分分配是不够的：fragment 的 elementwise 使用必须同步改写，否则索引会越过
   新的 shape。pass 把 `T.Parallel` / 顶层 elementwise `T.serial` / 整块
   `T.copy` 改写成 `R` 个 `T.Parallel(lanes)` chunk nest，每个 nest 只触碰一个寄存器。
4. 改写按**整个 fragment 原子**生效：只要存在任何一个不支持的用法（stride、间接
   下标、2-D 索引、嵌套控制流、reduce、部分区域 copy、带 annotation 的循环……），
   该 fragment 完全不降级，VF 行为与改动前逐字节一致。
5. 降级后每个 nest 的向量轴恰好是一个完整寄存器（chunks = 1），P2 的
   fragment-register mode 直接把它规划成 `local.var` 向量寄存器，P3 发射
   `vector_f32` 声明与纯寄存器读写，**fragment 侧没有任何 `vld`/`vsts`**。
6. 回归：`testing/ascend/transform` **594 passed**；`p_simdvf_alloc_fragment.py`
   （含 `T.serial` 与 `T.copy(frag2, e_ub)`）由"legacy 报错"变为完整向量化。

---

## 1. 问题背景

### 1.1 fragment 的语义与 codegen 现状

| 环节 | 现状 |
| --- | --- |
| 前端 | `T.alloc_fragment` → `local.fragment`；`T.SimdVF` 退出时被 steal 进 VF block 的 `alloc_buffers`（`src/ascend/ir.cc`） |
| P1 | 把 VF block 的 `alloc_buffers` 标记为 `FRAGMENT` capture；body 只做 metadata |
| P2 | `FragmentMode` 逐 nest 判定（G0–G8）：要求每个 chunk 是一个完整寄存器 |
| P3 | 通过判定后写 `fragment_regs`，把 fragment 替换为 `ceil(size/lanes)` 个 `local.var` 向量寄存器 |
| codegen | `local.fragment` 分配在 SimdVF 内直接发一个 `vector_f32 <name>;`（`codegen_ascend.cc:3076-3086`），**忽略 shape** |

因此"一个 fragment = 多个寄存器"只能由 P2/P3 在**可规划的 nest** 上完成。

### 1.2 触发本次改动的问题

`p_simdvf_alloc_fragment.py` 这样的 VF：

```python
with T.SimdVF():
    frag  = T.alloc_fragment((128,), "float32")
    frag2 = T.alloc_fragment((128,), "float32")
    for i in T.Parallel(128):        # 1. 并行填充
        frag[i] = a_ub[i + 32]
    for i in T.serial(128):          # 2. elementwise serial
        frag[i] = frag[i] * 2.0
    for i in T.Parallel(128):        # 3. 并行使用
        frag[i]  = frag[i] + d_ub[i]
        c_ub[i]  = frag[i]
        frag2[i] = frag[i] - 1.0
    T.copy(frag2, e_ub)              # 4. 整块 copy 回 UB
```

- P1 把 item 2（非 `T.Parallel`）判为 `OPAQUE`，于是整组 `OPAQUE`，P2/P3 不规划；
- 回退到 legacy `AscendSimdVFLowerParallel` 后，item 2 被
  `AddWrapperForSingleBufStore` 包成 `for _ in T.parallel(1)`，legacy 直接失败：
  `T.Parallel inside SimdVF requires VReg extent divisible by VReg size (64), got 1`。

也就是说：**只要 fragment 的用法里出现 serial 或 tile-op copy，现有两条路径都无法
把它向量化到寄存器**。本 pass 在 P1 之前先把 fragment 拆到寄存器粒度，并把这类
elementwise 用法统一成 `T.Parallel(lanes)`，让 P2/P3 成为唯一路径。

---

## 2. 输入与输出 IR

### 2.1 输入（P1 之前的 VF block）

```python
with T.sblock("SIMD_VF", no_realize=True):
    frag  = T.sblock_alloc_buffer((128,), data=frag.data,  scope="local.fragment")
    frag2 = T.sblock_alloc_buffer((128,), data=frag2.data, scope="local.fragment")
    T.attr("simdvf", "tl.simdvf_scope", 1)
    for i in T.parallel(128):
        frag[i] = a_ub[i + 32]
    for i in range(128):
        for _ in T.parallel(1):              # AddWrapperForSingleBufStore
            frag[i] = frag[i] * T.float32(2.0)
    for i in T.parallel(128):
        frag[i] = frag[i] + d_ub[i]
        c_ub[i] = frag[i]
        frag2[i] = frag[i] - T.float32(1.0)
    T.ascend_copy(T.region(frag2[0], 1, 128), T.region(e_ub[0], 2, 128))
```

### 2.2 输出

```python
with T.sblock("SIMD_VF", no_realize=True):
    frag_r0  = T.sblock_alloc_buffer((64,), scope="local.fragment")
    frag_r1  = T.sblock_alloc_buffer((64,), scope="local.fragment")
    frag2_r0 = T.sblock_alloc_buffer((64,), scope="local.fragment")
    frag2_r1 = T.sblock_alloc_buffer((64,), scope="local.fragment")
    T.attr("simdvf", "tl.simdvf_scope", 1)
    for i_c0 in T.parallel(64):
        frag_r0[i_c0] = a_ub[i_c0 + 32]
    for i_c1 in T.parallel(64):
        frag_r1[i_c1] = a_ub[64 + i_c1 + 32]
    for i_c0 in T.parallel(64):
        frag_r0[i_c0] = frag_r0[i_c0] * T.float32(2.0)
    for i_c1 in T.parallel(64):
        frag_r1[i_c1] = frag_r1[i_c1] * T.float32(2.0)
    for i_c0 in T.parallel(64):
        frag_r0[i_c0] = frag_r0[i_c0] + d_ub[i_c0]
        c_ub[i_c0] = frag_r0[i_c0]
        frag2_r0[i_c0] = frag_r0[i_c0] - T.float32(1.0)
    for i_c1 in T.parallel(64):
        frag_r1[i_c1] = frag_r1[i_c1] + d_ub[64 + i_c1]
        c_ub[64 + i_c1] = frag_r1[i_c1]
        frag2_r1[i_c1] = frag_r1[i_c1] - T.float32(1.0)
    for i in T.parallel(64):
        e_ub[i] = frag2_r0[i]
    for i in T.parallel(64):
        e_ub[64 + i] = frag2_r1[i]
```

同时：

- 原 `frag`/`frag2` 的 `alloc_buffers` 条目被替换为寄存器条目；
- 原 fragment 的 `DeclBuffer` 被替换为各寄存器的 `DeclBuffer`；
- 各 block 的 `layout_map` 中指向原 fragment 的条目被删除（寄存器 buffer 无 layout，
  P2 的 `FragmentLayoutIsIdentity` 对无 layout 的 buffer 返回 identity）。

---

## 3. 规则

### 3.1 寄存器宽度与数量

| 项 | 规则 |
| --- | --- |
| 寄存器宽度 | `kVectorRegisterBits = 2048`（bit），即 256 B，与 `ValidateSimdVectorType` 的 2048-bit 约束一致 |
| lane 数 | `lanes = 2048 / dtype.bits()`：f32=64、f16/bf16=128、int8/uint8=256 |
| 切分前提 | shape 全静态、`prod(shape) > 0`、`prod(shape) % lanes == 0` |
| 寄存器数 | `R = prod(shape) / lanes`，只有 `2 <= R <= 16`（`kMaxRegistersPerFragment`）才降级 |
| 元素类型 | int8/16/32、uint8/16/32、float16/32、bfloat16（与 P2 `FragmentDtypeSupported` 对齐）；fp8/bf16 以外的特殊类型不处理 |
| 命名 | `<原 buffer 名>_r<k>`，scope 仍为 `local.fragment`，shape `(lanes,)` |

说明：

- `R == 1` 时 fragment 本身就是一个寄存器，**不拆分**（保持现有已验证路径）；
- `R > 16` 不拆分，避免 IR/寄存器膨胀（P2 的 `kFragmentUnrollLimit` 也是 16）。

### 3.2 支持的使用模式

一个 fragment 只有在**所有**使用都是下列模式之一时才会被降级。

| # | 模式 | 判定条件 | 改写 |
| --- | --- | --- | --- |
| U1 | `T.Parallel(N)` elementwise | `N == prod(shape)`；body 无嵌套 `For`/`IfThenElse`；该 fragment 的每次 load/store 都是单下标且**恰好等于循环变量**；无 store predicate | 拆成 `R` 个 `T.Parallel(lanes)`，第 k 个循环用 `frag_rk`，其它 buffer 的下标替换为 `k*lanes + i` |
| U2 | 顶层 elementwise `T.serial(N)` | 在 U1 基础上要求：body（穿透 `T.parallel(1)` 后）全部是 `BufferStore`，且所有被写 buffer 都是待降级 fragment（读写只改自身元素） | 同上，循环种类改为 `T.Parallel`（保证无跨迭代依赖才允许） |
| U3 | 整块 `T.copy(frag, other)` / `T.copy(other, frag)` | `tl.tileop.copy` / `tl.tileop.ascend_copy`；两侧都是 1-D region、extent 相等且等于 fragment 总元素数；dtype 相同；fragment 侧 base 为 0；两侧可同为待降级 fragment | 拆成 `R` 个 `T.Parallel(lanes)` copy nest：`dst[k*lanes+i] = frag_rk[i]` 或反向 |
| U4 | `T.parallel(1)` store wrapper | `AddWrapperForSingleBufStore` 产生的恒等循环 | 分析时透明穿透；改写时删除（否则会成为 P1 拒绝的嵌套 parallel） |

U1/U2 的"恰好等于循环变量"意味着 v1 只接受 `Δ=1`、chunk 对齐、无 base 偏移的
fragment 访问；`frag[i + 32]`、`frag[2*i]`、`frag[b[i]]` 都属于不支持。

### 3.3 回退条件（fragment 保持原样）

| 类别 | 例子 |
| --- | --- |
| 分配不可降级 | 动态 shape；`prod(shape) % lanes != 0`；`R == 1`；`R > 16`；不支持的 dtype |
| 访问形态 | 非单下标（2-D `frag[i, j]`）；下标不是循环变量；带 predicate |
| 控制流 | 循环体内有嵌套 `For` / `IfThenElse`（`T.parallel(1)` 除外） |
| 循环属性 | `T.unroll` / `T.vectorized`；有 `thread_binding`；`annotations` 非空 |
| 其他用法 | `T.reduce_*` 等 tile op；部分区域 copy；fragment 作为 copy 的非常量/非 1-D region |
| 多 fragment 混用 | 同一语句里被触碰的 fragment 的 `lanes` / 总元素数不一致，或其中一个已被别的语句否决 |
| 跨 VF 共享 | 同一 data Var 已由另一个 VF 降级（防御性拒绝） |

回退是**逐 fragment** 的：同一 VF 中满足条件的 fragment 仍会降级，不满足的保持原样。

---

## 4. 实现结构

文件：`src/ascend/transform/simdvf_alloc_fragment.cc`（约 880 行），整体分三个阶段，
由 `LowerSimdVFAllocFragment`（`:847`）串起来：

| 阶段 | 类 / 函数 | 职责 |
| --- | --- | --- |
| 1 | `PlanCollector`（`:701`） + `PlanBlock`（`:513`） | 扫描每个 `SIMD_VF` block，算出要降级的 fragment 集合与每个 unit 的改写方案（**不修改 IR**） |
| 2 | `PlanApplier`（`:732`） + `RewriteUnit`（`:588`） + `ChunkRewriter`（`:406`） | 替换 `alloc_buffers`、改写 body 中受影响的 unit |
| 3 | `DeclCleanup`（`:798`） | 替换原 fragment 的 `DeclBuffer`、删除失效的 `layout_map` 条目 |

阶段 1 只读，保证阶段 2 可以按 `const SBlockNode*` 精确命中同一个 block。

### 4.1 数据结构

```cpp
struct RegisterArray {          // 一个待降级 fragment
  Buffer source;
  int64_t total;                // prod(shape)
  int64_t lanes;                // 2048 / dtype.bits()
  std::vector<Buffer> regs;     // R 个 (lanes,) local.fragment buffer
};

struct UnitScan {               // VF body 的一条顶层语句
  Stmt stmt;
  std::set<const VarNode*> touched;   // 触碰到的候选 fragment（data Var 身份）
  bool supported;                     // 是否匹配 U1–U3
  int64_t extent;                     // 循环/拷贝的迭代域
};
```

身份一律用 `buffer->data`（data Var）而不是 buffer 名字：同名 fragment 是两个不同的
data Var，必须各自独立成组（与 `p2_p3_manual_fragment_register_design.md` §2.8 的
结论一致）。

### 4.2 判定：不动点收缩

`PlanBlock` 先由 `MakeRegisterArray`（`:475`）筛出候选，然后：

1. 用 `CollectUnits` 把 VF body 拆成顶层 unit（穿透 `SeqStmt`、`tl.simdvf_scope`
   AttrStmt，跳过 `DeclBuffer`）；
2. 每个 unit 用 `UnitScanner`（`:243`）扫描：得到 `touched` 与 `supported`；
   不支持的 unit 也会记录它触碰的候选 fragment，用于否决；
3. 反复迭代直到稳定：
   - 若 unit 不支持 → 移除它触碰的全部 fragment；
   - 若 unit 触碰的某个 fragment 已被移除 → 本 unit 也失败，移除其余 fragment；
   - 若 unit 内 fragment 的 `total != extent` 或 `lanes` 不一致 → 同样移除。

只有最终仍在集合里的 fragment 才会生成 `RegisterArray`。这一步保证"某条语句既
引用已降级 fragment、又引用未降级 fragment"的情况不会出现。

### 4.3 改写：`ChunkRewriter`

对第 k 个 chunk（`k = 0 … R-1`）：

- 新建 chunk 循环变量 `i_ck`，把原循环变量在**所有非 fragment 访问**中替换为
  `k*lanes + i_ck`；
- 对被降级的 fragment，`frag[loop_var]` 直接替换为 `frag_rk[i_ck]`
  （`BufferLoad` / `BufferStore` 各一条分支，并用 `TVM_FFI_ICHECK` 守住
  "索引必须是循环变量"的契约）；
- 恒等 `T.parallel(1)` wrapper 在改写时被剥离，避免出现嵌套 parallel。

`T.copy` 的改写不经过 `ChunkRewriter`：`RewriteUnit` 解析两侧 `tl.region`
（`DecodeRegion`，`:217`），逐 chunk 生成 `BufferLoad`/`BufferStore` 的
`T.Parallel(lanes)` nest。

`RewriteBody`（`:675`）负责在原位置把旧 unit 替换为 `R` 条新语句，并保持
`SeqStmt` 扁平、`tl.simdvf_scope` AttrStmt 不变。

### 4.4 收尾：`DeclCleanup`

- `DeclBuffer(frag)` → `R` 个 `DeclBuffer(frag_rk)`；
- 任意 block 的 `layout_map` 删除 key 为原 fragment 的条目；若 map 因此为空则删除
  整个 annotation（与 P3 `RewriteFragmentAllocations` 的做法一致）。

---

## 5. 管线位置与开关

```python
if pass_ctx.config.get(tilelang.PassConfigKey.TL_ASCEND_SIMD_LIFT, True):
    mod = ascend_transform.AscendSimdVFallocFragment()(mod)   # 新增
    mod = ascend_transform.AscendLiftVSimd()(mod)             # P1
    mod = ascend_transform.AscendVectorizeVSimd()(mod)        # P2 + P3
```

- 位置：`AutoSchedule → LowerScheduledTIR → RestoreWhileLoops → Simplify →
  NormalizeBufferVersion` 之后、P1 之前（`pipeline.py:126-130`）；
- 与 P1/P2/P3 共用 `TL_ASCEND_SIMD_LIFT` 开关：关闭该开关时 pass 不运行；
- 加在 P1 之前的原因：P1 只读 body、不改 body，P2/P3 以 P1 记录的 span 为准；
  若在 P1 之后改 body 会让 span 失效。

---

## 6. 与 P2/P3 的衔接

降级后每个 nest 的向量轴长度等于 `lanes`，正好是 P2 的
`chunks = ceil(extent / lanes) = 1`：

- P2 `ClassifyFragmentRegisters`（`polyhedral/plan.cc`）为每个 nest 生成一条
  `FragmentReg`：`slots=1`、`lanes=lanes`；
- P3 `FragmentRegisterFile::Build`（`polyhedral/emit.cc:615`）为每个 slot 生成一个
  `local.var` 向量 buffer `<name>_r0`；
- codegen 在 `VisitStmt_(AllocBufferNode)` 的 `local.var` 分支发
  `vector_f32 <name>;`，`GetBufferRef` 对 `local.var` 直接返回变量名，
  因此读写就是寄存器赋值。

`p_simdvf_alloc_fragment.py` 的最终源码（节选）：

```c
__simd_vf__ inline void main_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_f32 frag_frag_r0_0_r0;
  vector_f32 frag_frag2_r0_1_r0;
  vector_f32 frag_frag_r1_2_r0;
  vector_f32 frag_frag2_r1_3_r0;
  frag_frag_r0_0_r0 = simd_inst::vlds_norm<float>(... [32] ...);
  frag_frag_r1_2_r0 = simd_inst::vlds_norm<float>(... [96] ...);
  frag_frag_r0_0_r0 = simd_inst::vmuls(frag_frag_r0_0_r0, 2.0f, mask, MODE_ZEROING);
  frag_frag_r1_2_r0 = simd_inst::vmuls(frag_frag_r1_2_r0, 2.0f, mask, MODE_ZEROING);
  ...
  simd_inst::vsts_norm(frag_frag_r0_0_r0, ... [288] ...);   // c_ub
  simd_inst::vsts_norm(frag_frag_r1_2_r0, ... [352] ...);   // c_ub
  simd_inst::vsts_norm(frag_frag2_r0_1_r0, ... [416] ...);  // e_ub
  simd_inst::vsts_norm(frag_frag2_r1_3_r0, ... [480] ...);  // e_ub
}
```

fragment 侧只有寄存器声明与寄存器读写；`vlds/vsts` 只作用于 UB。

---

## 7. 验证

### 7.1 离线结构用例

| 用例 | 结果 |
| --- | --- |
| `(256,) f32` + 1 个 parallel + `T.copy(frag, ub)` | 4 个 `(64,)` fragment；4 个 parallel fill + 4 个 parallel copy |
| `T.copy(ub, frag)` 反向 | 4 个 copy nest + 4 个 consume nest |
| `frag[2*i]` stride 访问 | 不降级，原 `(256,) local.fragment` 保持不变 |
| `p_simdvf_alloc_fragment.py`（serial + copy） | 8 个 chunk nest，2 个 fragment → 4 个寄存器 |

### 7.2 回归

| 套件 | 结果 |
| --- | --- |
| `pytest testing/ascend/transform` | 594 passed（含 `test_ascend_SimdVF_alloc_fragment.py`） |
| `pytest testing/ascend`（其余） | 210 failed / 892 passed / 170 skipped，失败均为环境问题：bisheng 在 glibc `floatn.h` 报 `unsupported machine mode '__TC__'`、缺 `torch_npu`、无 npu device；与本次改动无关 |

---

## 8. 已知限制与后续方向

1. **只处理 1-D 元素访问**：`frag[i, j]` 这类 2-D fragment 即使展平后连续，也不降级。
   后续可以在 `UnitScanner` 里把多下标线性化后再判断。
2. **不支持带 base 偏移的整块访问**：`frag[i + 32]`、partial region copy 都回退。
3. **serial 转 parallel 有严格前提**：只有 "body 全部是对该 fragment 的自索引 store"
   才转换；带其它写、条件、嵌套循环的 serial 一律回退（仍会走 legacy）。
4. **`R > 16` 不拆分**：更大的 tile 目前留给既有 fragment 路径或用户改写成 UB。
5. **fragment 逃逸**（被 `T.reduce_*`、手写 `T.simd.*`、标量语句使用）一律否决，
   这与 P2 V1 escape 规则一致。
6. **与 `AddWrapperForSingleBufStore` 的耦合**：pass 依赖"`T.parallel(1)` 是恒等
   wrapper"这一事实；若该 pass 的行为变化（例如不再包 wrapper），此处逻辑仍然成立
   （只是不再需要剥离）。

---

## 9. 关键代码索引

| 符号 | 位置 |
| --- | --- |
| `kVectorRegisterBits` / `kMaxRegistersPerFragment` | `simdvf_alloc_fragment.cc:86` / `:89` |
| `IsIdentityParallel` / `FlattenElementwiseStmts` | `:142` / `:151` |
| `RegisterArray` / `UnitScan` | `:168` / `:176` |
| `DecodeRegion` | `:217` |
| `UnitScanner`（U1/U2/U3 判定） | `:243` |
| `ChunkRewriter` | `:406` |
| `MakeRegisterArray` | `:475` |
| `PlanBlock`（候选 + 不动点） | `:513` |
| `RewriteUnit`（循环 / copy 拆分） | `:588` |
| `RewriteBody` | `:675` |
| `PlanCollector` / `PlanApplier` / `DeclCleanup` | `:701` / `:732` / `:798` |
| `LowerSimdVFAllocFragment` / pass 注册 | `:847` / `:862` |
| Python 包装 | `tilelang/ascend/transform/__init__.py:60-70` |
| 管线插入点 | `tilelang/ascend/pipeline.py:126-130` |
