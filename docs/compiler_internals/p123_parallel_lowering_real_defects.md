# P1/P2/P3 `T.Parallel` 降级：真实缺陷修复清单

> **历史文档**：本文记录写作时的实现/设计状态。后续改动已删除或重命名部分
> 文件、测试与行号；当前实现见 `src/ascend/transform/parallel_vectorize/` 与
> `src/ascend/transform/vsimd_*`，以代码为准。


> **用途**：本文只记录在 Ascend 950PR 上**实测确认的 3 个真实实现缺陷**，
> 供在另一台机器上照着改代码。测试文件自身的断言错误、CANN 限制、
> 既有 `Simplify` 语义问题**不在本文范围**，见
> `p123_simdvf_parallel_device_test_report.md`。
>
> | 项 | 内容 |
> | --- | --- |
> | 分支 | `p_123_parallel_new_p1` |
> | 设备 | Ascend 950PR（NPU 0） |
> | CANN | 9.3.0.weekly.20260930（`bisheng` clang 15.0.5，`--npu-arch=dav-3510`） |
> | 验证文件 | `testing/ascend/language/test_tilelang_ascend_simdvf_parallel.py` |
> | 结论来源 | 40 passed / 37 failed 中的 5 个失败（3 类根因） |
>
> 三条缺陷都满足"手写 `T.simd.*` 或参数扫描可复现、改一处即正确"，
> 因此判定为发射/规划侧实现问题，不是硬件或 CANN 不支持。

---

## 0. 缺陷总表

| # | 缺陷 | 代码位置 | 影响用例 | 严重度 |
| --- | --- | --- | --- | --- |
| **D1** | `vsstb` 步长低 16 位写成 `1`（应为 `0`） | `emit.cc` 1625-1626、1732-1733 | `compact_nz_store`、`paired_nz_store_with_an_nd_sibling`、`mixed_recipe`(C) | **P0**：NZ 直存全错，含 FA softmax 路径 |
| **D2** | 串行 shell 的读改写缺 `mem_bar(VST_VLD)` | `emit.cc` 2353-2357、2749-2751 | `under_a_serial_shell` | **P1**：跨迭代竞争 |
| **D3** | 归约段把"未被使用的并行轴"当成外层循环，重复累加 | `plan.cc` `AssembleProgram` / `OuterAxes` 语义 | `mixed_recipe`(D)、2D nest 归约 | **P1**：结果按 `N` 倍放大 |

---

## D1. `vsstb` 步长低 16 位必须是 0

### 现状代码

`src/ascend/transform/parallel_vectorize/emit.cc:1624-1627`（`EmitVectorWrite`，普通 NZ 直存）：

```cpp
int64_t block = store->nz_block_stride.has_value() ? *store->nz_block_stride : 0;
int64_t elem = store->nz_elem_stride.has_value() ? *store->nz_elem_stride : 1;
PrimExpr stride = IntImm(DataType::Int(32), (block << 16) | (elem & 0xffff));
```

`src/ascend/transform/parallel_vectorize/emit.cc:1731-1733`（`EmitPackedNzStore`，配对 pack2 形态）：

```cpp
int64_t block = store->nz_block_stride.has_value() ? *store->nz_block_stride : 0;
int64_t elem_stride = store->nz_elem_stride.has_value() ? *store->nz_elem_stride : 1;
PrimExpr stride = IntImm(DataType::Int(32), (block << 16) | (elem_stride & 0xffff));
```

`src/ascend/transform/parallel_vectorize/plan.cc:2001` 把 `nz_elem_stride` 恒设为 `1`：

```cpp
store.CopyOnWrite()->nz_elem_stride = ffi::Optional<int64_t>(1);
```

于是实际发出的是 `524289 = (8 << 16) | 1`。

### 设备实测（决定性证据）

在 950PR 上直接手写 `T.simd.vsstb`，扫低 16 位。f32 / bf16 结果一致：

| 形态 | 低 16 位 | 结果 |
| --- | --- | --- |
| 普通 store（`vsstb(src, c[...], stride, mask)`） | `0` | **与 compact-NZ 物理序逐位相等** |
| 普通 store | `1` | 每个 32B 块在 `base + 1 块` 处重复写，输出错乱 |
| post-update（`update=True`） | `0` | 错（指针不前进） |
| post-update | `1` | 对（每次前进 1 块） |

完整矩阵（`R` 行、单 chunk、`c0` = 32B 块元素数）：

```
float32 R=8  post=False low=0: exact=True      float32 R=8  post=True low=0: exact=False
float32 R=8  post=False low=1: exact=False     float32 R=8  post=True low=1: exact=True
float32 R=16 post=False low=0: exact=True      float32 R=16 post=True low=0: exact=False
float32 R=16 post=False low=1: exact=False     float32 R=16 post=True low=1: exact=True
bfloat16 R=8  post=False low=0: exact=True     bfloat16 R=8  post=True low=0: exact=False
bfloat16 R=8  post=False low=1: exact=False    bfloat16 R=8  post=True low=1: exact=True
bfloat16 R=16 post=False low=0: exact=True     bfloat16 R=16 post=True low=0: exact=False
bfloat16 R=16 post=False low=1: exact=False    bfloat16 R=16 post=True low=1: exact=True
```

**结论**：
- 低 16 位不是"块内元素步长"，而是 **POST_UPDATE 的指针前进量（以 32B 块计）**。
- 普通 store（P3 当前用的形态）**必须填 0**。
- post-update 形态才需要填前进量；P3 目前**不发** post-update（`emit.cc` 里没有
  `POST_UPDATE`），所以 P3 只需改 0。

CANN 侧印证：`asc_store_align_config` 是 `{ repeat_stride : 16; block_stride : 16 }`
（`x86_64-linux/asc/impl/c_api/reg_base_impl/storealign_intf_impl.h:28-34`），
`repeat_stride` 在非 post-update 指令里没有"块内步长"含义。

### 修复

**改 1** — `emit.cc:1624-1626`：

```cpp
    // ``vsstb`` 的 stride 是 ``(block_stride << 16) | repeat_stride``；
    // repeat_stride 只对 POST_UPDATE 有意义（指针前进多少个 32B 块）。
    // 普通 store 必须为 0，否则每个块会在 base + repeat 块处重复写。
    int64_t block = store->nz_block_stride.has_value() ? *store->nz_block_stride : 0;
    PrimExpr stride = IntImm(DataType::Int(32), block << 16);
```

**改 2** — `emit.cc:1731-1733`：

```cpp
  int64_t block = store->nz_block_stride.has_value() ? *store->nz_block_stride : 0;
  PrimExpr stride = IntImm(DataType::Int(32), block << 16);
```

**改 3（可选，清理）** — `plan.cc:2001` 删除 `nz_elem_stride` 赋值；
`model.h:97-101` 的字段与注释同步删除或标注"仅 post-update 用，普通 store 恒 0"。
若保留字段，注释必须改掉"element step inside a block, normally 1"这句错误描述。

### 连带需要改的 pass 级断言

`testing/ascend/transform/test_ascend_parallel_vectorize_*.py` 有 5 处断言写死了 `| 1`：

| 行 | 断言 |
| --- | --- |
| 3891 | `assert store.nz_elem_stride == 1` |
| 3901-3902 | `# (block_stride << 16) \| elem_stride = (128 << 16) \| 1` / `assert str((_WIDE << 16) \| 1) in text` |
| 3914 | `assert store.nz_elem_stride == 1` |
| 4015 | `assert str((_WIDE << 16) \| 1) in text` |
| 4063 | `assert str((rows << 16) \| 1) in text` |
| 4195-4196 | `# 块步长 = (129 << 16) \| 1` / `assert str((129 << 16) \| 1) in out` |

全部改为 `(_WIDE << 16)` / `(rows << 16)` / `(129 << 16)`，
并删除 `nz_elem_stride == 1` 断言（或改断言其为 `0`/`None`，取决于改 3 的取舍）。

### 文档需同步修正

`docs/compiler_internals/p2_p3_design_implementation_report.md` §F.5.2
把 `(129 << 16) | 1` 当作"已验证配方"，该结论错误，应改为 `(129 << 16)`。
§F.6 的 FA 示例同样要改。

---

## D2. 串行 shell 的读改写缺 `mem_bar(VST_VLD)`

### 触发形态

```python
with T.SimdVF():
    for k in T.serial(steps):          # 标量 shell
        for i in T.Parallel(rows):     # 向量体
            c[i] = c[i] + a[k, i]      # 读改写同一地址
```

### 现状生成（`rows=128, steps=4`）

```c
for (int32_t k = 0; k < 4; ++k) {
  for (int32_t chunk_s0 = 0; chunk_s0 < 2; ++chunk_s0) {
    auto vs0_0 = simd_inst::vlds_norm<float>(... c + chunk_s0*64 ...);
    auto vs0_1 = simd_inst::vlds_norm<float>(... a + k*128 + chunk_s0*64 ...);
    auto vs0_2 = simd_inst::vadd(vs0_0, vs0_1, mask_32, MODE_ZEROING);
    simd_inst::vsts_norm(vs0_2, ... c + chunk_s0*64 ...);
  }
}
```

相邻 `vsts` 与下一轮 `vlds` 之间**没有** `simd_inst::mem_bar(VST_VLD)`，
读改写跨迭代竞争。

### 设备实测

| rows | steps | 结果 |
| --- | --- | --- |
| 64 | 4 | ✅（单 chunk，恰好保序） |
| 128 | 1 / 2 | ✅ |
| 128 | 3 / 4 / 8 | ❌ |
| 192 / 256 | 4（纯写，非 RMW） | ✅ |

具体（`a[k,i] % 5`，`rows=128, steps=4`）：

```
got[0:6]  = [4, 0, 1, 2, 3, 4]      ← 只累加了部分迭代
exp[0:6]  = [8, 7, 6, 10, 9, 8]
```

**手工在每次 shell 迭代末尾插入 `T.simd.mem_bar("VST_VLD")` 后逐位正确**：

```
with mem_bar exact: True  got=[8,7,6,10,9,8]  exp=[8,7,6,10,9,8]
```

> 这条推翻了第一轮"bisheng -O2 误编译"的旧结论。判据是
> "单 chunk 对、多 chunk 错"——编译器误编译不会这样依赖 chunk 数。

### 修复

shell 包裹的向量体，**每次迭代末尾**补一条 `mem_bar(VST_VLD)`。
两个发射点：

**改 1** — `emit.cc:2352-2357`（`EmitSegmentStmts` 的 shell 包裹）：

```cpp
  std::vector<Stmt> main_group = flat_main;
  for (auto it = plan->shells.rbegin(); it != plan->shells.rend(); ++it) {
    if (main_group.empty()) break;
    Stmt inner = SeqOfVec(main_group)[0];
    main_group = {tirx::For((*it)->var, IntImm32(0), (*it)->extent, ForKind::kSerial, inner)};
  }
```

改为在包 `For` 之前，把屏障追加到 `main_group` 尾部：

```cpp
  std::vector<Stmt> main_group = flat_main;
  if (!plan->shells.empty() && !main_group.empty()) {
    // 一个 shell 迭代内的向量体可能对同一地址读改写；相邻迭代之间必须
    // 有 VST->VLD 屏障，否则 vld 会读到上一轮 vsts 之前的旧值。
    main_group.push_back(tirx::Evaluate(
        SimdCall(simd_mem_bar(), DataType::Void(), {StrImm("VST_VLD")})));
  }
  for (auto it = plan->shells.rbegin(); it != plan->shells.rend(); ++it) {
    ...
  }
```

**改 2** — `emit.cc:2748-2751`（归约路径的 shell 包裹）：

```cpp
    Stmt main_stmt = SeqOfVec(loop_main)[0];
    for (auto it = plan->shells.rbegin(); it != plan->shells.rend(); ++it) {
      main_stmt = tirx::For((*it)->var, IntImm32(0), (*it)->extent, ForKind::kSerial, main_stmt);
    }
```

同样在 `main_stmt` 外层包 `For` 之前，先把屏障与 `main_stmt` 组成 `SeqStmt`：

```cpp
    Stmt main_stmt = SeqOfVec(loop_main)[0];
    if (!plan->shells.empty()) {
      main_stmt = tirx::SeqStmt({main_stmt, tirx::Evaluate(
          SimdCall(simd_mem_bar(), DataType::Void(), {StrImm("VST_VLD")}))});
    }
    for (auto it = plan->shells.rbegin(); it != plan->shells.rend(); ++it) { ... }
```

### 可用的现成设施

- `simd_mem_bar()` 已在 `src/ascend/op/builtin.h:143` 声明，
  `builtin.cc:301` 注册（1 个参数）。
- codegen 已支持：`src/ascend/codegen/codegen_ascend.cc:2537-2540`
  把 `tl.simd.mem_bar` 打成 `simd_inst::mem_bar(<str>);`。
- `emit.cc` 已 `#include "ascend/op/builtin.h"`（第 26 行），无需新增 include。
- 模板：`src/tl_templates/ascend/simd_inst.h:1046-1047`
  `mem_bar(T) → ::asc_mem_bar(T)`。

### 注意

- 只在 `!plan->shells.empty()` 时加，避免给无 shell 的普通段引入额外指令。
- 若想更精细：只在 shell 体内存在"对同一 buffer 的读写"时加。
  保守做法（无条件加）已验证正确，且 shell 本身罕见，优先选保守。

---

## D3. 归约段把"未被使用的并行轴"当成外层循环，重复累加

### 触发形态

```python
with T.SimdVF():
    for i, j in T.Parallel(M, N):      # 2D 并行 nest
        for k in T.serial(N):          # 内层串行归约
            d[i] += s[i, k]            # 只依赖 i，与 j 无关
```

### 现状生成（`M=8, N=64`）

```c
for (int32_t i = 0; i < 8; ++i) {
  for (int32_t j = 0; j < 64; ++j) {     // ← 多余的 64 次外层循环
    auto vs0_0 = simd_inst::vlds_norm<float>(... s + i*64 ...);   // 不依赖 j
    auto vs0_1 = simd_inst::vcadd(vs0_0, mask_32, MODE_ZEROING);
    auto vs0_2 = simd_inst::vadds(vs0_1, ... d + i ...);
    simd_inst::vsts_1st(vs0_2, ... d + i ...);
  }
}
```

`j` 完全不出现在体内，但被当成外层串行循环包了 64 次，
于是每行被累加 64 遍。

### 设备实测

| M | N | 期望（rowsum） | 实测 | 期望（N×rowsum） |
| --- | --- | --- | --- | --- |
| 2 | 8 | `[21, 22]` | `[168, 176]` | `[168, 176]` |
| 2 | 16 | `[43, 47]` | `[258, 282]` | `[688, 752]` |
| 4 | 16 | `[43,47,51,48]` | `[688,752,816,768]` | 同左 |
| 8 | 128（全 1） | `128` | `4096` | — |

对照：**1D 形式（`for i in T.Parallel(M)` + 内层 serial）完全正确**
（全 1 输入得 128）。所以问题只在"多出一个不参与计算的并行轴"。

### 根因

`plan.cc` 的 `AssembleProgram`（约 2650-2680 行）构造 `program.axes`（外层串行包裹）：

```cpp
  if (single) {
    if (!planned.flattened_any) {
      for (const Axis& axis : OuterAxes(segments[0]->axes, segments[0]->vector_axis)) {
        hoisted.push_back(axis);
      }
    }
  }
```

`OuterAxes`（`plan.cc:1414-1422`）取"vector_axis 之前的所有并行轴"：

```cpp
std::vector<Axis> OuterAxes(const ffi::Array<Axis>& axes, const Axis& vector_axis) {
  std::vector<Axis> outer;
  for (const Axis& axis : axes) {
    if (axis->var.same_as(vector_axis->var) || !axis->is_parallel) break;
    outer.push_back(axis);
  }
  return outer;
}
```

对 `for i,j in T.Parallel(M,N)` + 内层 `k`，`seg.axes = [j(parallel,0), k(serial,1)]`，
`vector_axis = k`，于是 `OuterAxes` 返回 `[j]`，`program.axes = [j]`。
`EmitProgram`（`emit.cc:2805-2807`）把它们包成串行 `For`，
`i` 则由 `EmitSegmentStmts` 的末尾循环（`emit.cc:2363-2369`）包上：

```cpp
  for (auto it = plan->axes.rbegin(); it != plan->axes.rend(); ++it) {
    const Axis& axis = *it;
    if (axis->var.same_as(plan->vector_axis->var)) continue;
    if (HasHoisted(hoisted, axis)) continue;
    body = tirx::For(axis->var, IntImm32(0), axis->extent, ForKind::kSerial, body);
  }
```

注意 `seg.axes` 里**根本没有 `i`**（只有 `j` 和 `k`）——`i` 是 `T.Parallel` 的外层轴，
被 `AbsorbParallelChain` 拆掉后只留在 store 的索引里，作为自由变量。
`i` 的循环来自 `EmitProgram` 对 `program.axes` 的包裹，而 `program.axes` 只有 `j`。
所以 `i` 的绑定其实来自更外层的 `T.Kernel` 展开（`i` 是 kernel 的循环变量），
`j` 才是被错误地当成"每个 i 都要跑 64 次"的那一个。

即：**`j` 在归约段里是死轴**（不参与任何索引/值），但 `OuterAxes` 只看
"位置在 vector_axis 之前的并行轴"，不看"是否真的被使用"。

### 修复（二选一）

#### 方案 A（推荐，最小改动）：规划期拒绝/剔除死并行轴

在 `AssembleProgram` 构造 `hoisted` 时，跳过"在该段所有 store 索引与值里都不出现"的并行轴。
判据可直接复用 `plan.cc` 已有的 `ValueVariesAlong` / 索引自由变量检查：

```cpp
/*! \brief 该段是否真的用到 axis 的循环变量（store 索引或值里出现）。 */
bool SegmentUsesAxis(const VectorPlan& plan, const Axis& axis) {
  for (const StorePlan& store : plan->stores) {
    for (const PrimExpr& index : store->spec->indices) {
      if (!tirx::UndefinedVars(index).empty()) {
        for (const tirx::Var& used : tirx::UndefinedVars(index)) {
          if (used.same_as(axis->var)) return true;
        }
      }
    }
    if (ValueVariesAlong(store->spec->value, axis, /*analyzer=*/nullptr)) return true;
  }
  return false;
}
```

然后在 `single` 分支里：

```cpp
      for (const Axis& axis : OuterAxes(segments[0]->axes, segments[0]->vector_axis)) {
        if (!SegmentUsesAxis(segments[0], axis)) continue;   // 死轴：不包外层循环
        hoisted.push_back(axis);
      }
```

**但注意**：这样只是"不包循环"，`j` 的变量仍然没有任何绑定。
如果 `j` 在 kernel 层已经是 `T.Kernel` 展开的循环变量，那没问题；
如果 `j` 只存在于这个 Parallel nest，则必须在 P1 阶段就把它从 nest 里剔除，
或者直接**拒绝**该 nest（更保守）。

**保守版本（建议先用这个）**：在 `PlanSegment` 里对归约段做一次检查，
发现"位置在 vector_axis 之前、但整段未使用的并行轴"就抛 `Unsupported`：

```cpp
  if (reduce.defined()) {
    for (const Axis& axis : OuterAxes(axes, vector_axis)) {
      if (!SegmentUsesAxis(plan, axis)) {
        throw Unsupported(
            "a reduction nest has an unused outer parallel axis '" +
            axis->name.operator std::string() +
            "': it would be emitted as a redundant outer loop and multiply the "
            "accumulation (drop the axis or split the nest)",
            {}, "reduce.dead_parallel_axis");
      }
    }
  }
```

这样至少不会产出错误数值，而是回退到 legacy 标量路径。

#### 方案 B（更彻底）：P1 阶段剔除死并行轴

在 P1（`lift_vsimd.cc` / `vsimd_analysis_common.h` 的 `NestCollector`）里，
把"整个 nest 体里从未出现的并行轴"从 region 的 axis 列表里删掉。
这样 P2 根本看不到 `j`，`OuterAxes` 自然返回空。

代价：要改 P1 的 region 描述与 axis 编号，影响面比方案 A 大。
**建议先用方案 A 的保守版本止血，再评估 B。**

### 回归用例

修复后应新增/保留这两个用例（当前 `mixed_recipe` 的 D 就是它）：

```python
# 1D 形式（已正确，作为对照）
for i in T.Parallel(M):
    for k in T.serial(N):
        d[i] += s[i, k]

# 2D nest 死轴（当前错误）
for i, j in T.Parallel(M, N):
    for k in T.serial(N):
        d[i] += s[i, k]      # 与 j 无关
```

期望两者结果一致且等于 `s.sum(dim=1)`。

---

## 4. 修复顺序与验证

### 顺序

1. **D1**（`vsstb` 低 16 位）—— 影响面最大，且 FA softmax 路径也走它。
   改 `emit.cc` 两处 + pass 级断言 + 文档。
2. **D2**（shell 屏障）—— 一处改动，风险低。
3. **D3**（死并行轴）—— 先用保守拒绝版本，避免错误数值流出。

### 每步的验证命令

```bash
cd <repo>
export PYTHONPATH=.
export ASCEND_RT_VISIBLE_DEVICES=<idle id>
export TILELANG_CACHE_DIR=/tmp/tlcache_$(date +%s)

# D1：NZ 直存三例
python -m pytest testing/ascend/language/test_tilelang_ascend_simdvf_parallel.py \
  -q -p no:cacheprovider -k "compact_nz or paired_nz or mixed_recipe"

# D2：串行 shell
python -m pytest testing/ascend/language/test_tilelang_ascend_simdvf_parallel.py \
  -q -p no:cacheprovider -k "under_a_serial_shell"

# D3：归约
python -m pytest testing/ascend/language/test_tilelang_ascend_simdvf_parallel.py \
  -q -p no:cacheprovider -k "row_reduction or sibling_reduction or mixed_recipe"

# pass 级（D1 会改断言）
python -m pytest testing/ascend/transform/test_ascend_parallel_vectorize_*.py \
  -q -p no:cacheprovider
```

### 注意：测试文件本身还有 30 个假失败

跑整文件时，`cast_pair`(24)、`eight_bit_full_chunk`(2)、`fp16_to_fp8`(1)、
多输出 `out_idx`(3) 的失败**不是实现问题**，数值全部正确。
在修完 D1/D2/D3 之前不要被它们干扰；详见
`p123_simdvf_parallel_device_test_report.md` §3 A/B/C。

---

## 5. 附：三处缺陷的最小复现代码

### D1

```python
import torch, tilelang
from tilelang.ascend import language as T
from tilelang.layout import make_ascend_compact_nz_layout
M, N = 8, 128
@T.prim_func
def main(A: T.Tensor((M, N), "float32"), C: T.Tensor((M, N), "float32")):
    with T.Kernel(1):
        a = T.alloc_shared((M, N), "float32")
        c = T.alloc_shared((M, N), "float32")
        T.annotate_layout({c: make_ascend_compact_nz_layout(c)})
        T.copy(A, a)
        with T.SimdVF():
            for i, j in T.Parallel(M, N):
                c[i, j] = a[i, j]
        T.copy(c, C)
k = tilelang.compile(main, target="ascend", out_idx=-1)
a = torch.arange(M * N, dtype=torch.float32).reshape(M, N)
got = k(a.npu()).cpu().flatten()
exp = a.reshape(M, N // 8, 8).permute(1, 0, 2).reshape(-1)   # compact NZ 物理序
print(torch.equal(got, exp))   # 现状 False；修 D1 后 True
# 生成代码里应看到 stride = 524288 (8<<16)，而不是 524289 ((8<<16)|1)
```

### D2

```python
rows, steps = 128, 4
@T.prim_func
def main(A: T.Tensor((steps, rows), "float32"), C: T.Tensor((rows,), "float32")):
    with T.Kernel(1):
        a = T.alloc_shared((steps, rows), "float32")
        c = T.alloc_shared((rows,), "float32")
        T.copy(A, a); T.fill(c, 0.0)
        with T.SimdVF():
            for k in T.serial(steps):
                for i in T.Parallel(rows):
                    c[i] = c[i] + a[k, i]
        T.copy(c, C)
k = tilelang.compile(main, target="ascend", out_idx=-1)
a = (torch.arange(steps * rows, dtype=torch.int32).reshape(steps, rows) % 5).to(torch.float32)
print(torch.equal(k(a.npu()).cpu(), a.sum(0)))   # 现状 False；修 D2 后 True
# 生成代码的 shell 循环体末尾应出现 simd_inst::mem_bar(VST_VLD);
```

### D3

```python
M, N = 8, 128
@T.prim_func
def main(S: T.Tensor((M, N), "float32"), D: T.Tensor((M,), "float32")):
    with T.Kernel(1):
        s = T.alloc_shared((M, N), "float32")
        d = T.alloc_shared((M,), "float32")
        T.copy(S, s); T.fill(d, 0.0)
        with T.SimdVF():
            for i, j in T.Parallel(M, N):
                for k in T.serial(N):
                    d[i] += s[i, k]        # 与 j 无关
        T.copy(d, D)
k = tilelang.compile(main, target="ascend", out_idx=-1)
s = torch.ones(M, N)
print(k(s.npu()).cpu()[:3])   # 现状 [4096, 4096, 4096]；正确 [128, 128, 128]
# 生成代码里不应出现 "for (j = 0; j < 128; ++j)" 这层冗余循环
```

---

## 6. 本工作树的落地记录

> 本节记录代码侧实际改了什么，供另一台机器对照与复跑。

### 6.1 D1：`vsstb` 步长低 16 位改为 0

| 文件 | 改动 |
| --- | --- |
| `emit.cc::EmitVectorWrite` | `stride = block << 16`（删掉 `elem` 位） |
| `emit.cc::EmitPackedNzStore` | 同上 |
| `plan.cc` | 删除 `nz_elem_stride = 1` 赋值 |
| `model.h` | 删除 `nz_elem_stride` 字段与反射（该字段没有正确语义） |
| `analyze_parallel_vectorize.cc` | plan 渲染改为 `layout=nz[<block>]`，不再打印低 16 位 |
| `test_ascend_parallel_vectorize_nz.py` | 断言改为 `(_WIDE << 16)` / `(rows << 16)` / `(129 << 16)`，删除 `nz_elem_stride` 断言 |
| `p2_p3_design_implementation_report.md` | §F.5.2 / §F.6 的配方改为 `(129 << 16)` |

### 6.2 D2：屏障加在 P3 发射，不是 `pipeline.py` 的 `AscendThreadSync("global")`

`pipeline.py:197-198` 的 `AscendThreadSync("global")` **不能**承担这条屏障：

- `thread_storage_sync.cc:2046-2049`：`sync_scope.rank == kGlobal` 时**直接返回原函数**，
  所以该调用本身就是 no-op；
- 它只处理 `SIMT_VF` block 内的线程级 storage 依赖（`thread_storage_sync.cc:2067-2093`），
  语义是"多线程对 shared 的读写先后"，不是 AIV 向量流水线内的 VST→VLD 相关；
- 这条依赖只存在于 P3 生成的"shell 迭代 → chunk 程序"结构里，只有发射侧知道迭代边界。

落地（三个发射点，都在 `!shells.empty()` / 存在外层壳时发）：

1. `emit.cc::VstVldBarrier()`：`tl.simd.mem_bar("VST_VLD")` 的唯一构造入口（emit.h 导出）；
2. elementwise 路径：在 `main_group` 尾部追加后再包 `plan->shells`；
3. 归约路径：把 `main_stmt` 与屏障组成 `SeqStmt` 后再包 `plan->shells`；
4. **外层串行壳**（设备用例 `under_a_serial_shell` 的形态）：`vsimd_group_passes.cc::RegionEmitter`
   统计当前串行 For 深度，替换 Parallel 时若 `serial_shells_ > 0` 就在替换结果末尾追加屏障。
   这一条是必须的——`plan->shells` 只覆盖 **Parallel 体内**的 shell，而 D2 触发形态的
   `for k` 在 Parallel **外面**，它由 P1 的 `outer_shells` 记录、P3 原样保留，只有 P3 知道它。

离线断言：`test_non_absorbable_shell_is_kept_as_a_scalar_loop`（体内壳 `mem_bar` 恰好 1 条）、
`test_serial_shell_reduction_is_absorbed_as_an_axis`（无壳 → 无屏障）、
`test_parallel_under_a_serial_shell_closes_each_iteration`（外层壳 → 循环体内 1 条）。

### 6.3 D3：P1 直接报错（不静默"修正" DSL）

`vsimd_analysis_common.h::NestCollector::RejectDeadParallelAxes` 在 nest 发布时执行两条检查：

1. **整段死轴**：某个 Parallel 轴变量在 nest body（bounds/guards/indices/values）里从未出现；
2. **归约忽略轴**：某条累积 store（值里读同一 buffer 同一地址）的索引与值都不含某个 Parallel 轴。

两条都**直接 `LOG(FATAL)` 中止编译**，错误信息给出具体轴名与修法。
不再返回 `OPAQUE` 回退标量：设备实测 `SIMD_VF` 内的标量循环会被 bisheng 以
`Unsupported scalar instruction in AIV loop` 拒绝（`__simd_vf__` 函数体内不允许标量 UB 读写），
所以"拒绝即回退"在这条链路上不可编译。

影响：`mixed_recipe` 这类"elementwise 用到 j、归约语句不用 j"的 nest 现在编译期报错，用户需改 DSL。
设备用例已拆成 `test_parallel_elementwise_nz_and_row_reduction_are_planned_independently`
（合法 DSL：两个 nest 各自向量化）；死轴策略不再保留设备用例（属于用户 DSL 错误）。

离线用例（`test_ascend_vsimd_lift.py`）改为期望抛错：
`test_unused_parallel_axis_is_rejected`、
`test_reduction_over_a_dead_parallel_axis_is_rejected`；
`test_reduction_along_the_vector_axis_is_still_lifted` 仍验证合法归约照常 lift。

---

## 7. 第三轮设备报告：基址 32B 对齐与 `extent == 1`（2026-11-13）

> 这两类**不是**发射器的编码错误，而是 ISA 语义 / DSL 形态的硬约束。
> 本轮按"**暂不支持**"落地：P1/P2 显式拒绝 + 文档标注，不做发射侧修复。

### 7.1 现象（设备实测）

用例 `test_parallel_row_reduction_with_a_tail`
（`for i in Parallel(rows): for j in serial(k): c[i] += a[i, j]`，f32）：

| 形态 | 结果 |
| --- | --- |
| `rows, k = 8, 100`（行距 400B） | 失败：输出 507035（只有第 0 行正确） |
| `rows, k = 8, 104`（行距 416B = 13 × 32B） | 通过 |
| `rows, k = 1, 100` | CANN 报 `Unsupported scalar instruction in AIV loop`（VF 内标量累加） |

`8, 100` 的生成代码：

```cpp
for (int32_t i = 0; i < 8; ++i) {
  auto vs0_0 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(..[i * 100])), 0);
  ...
}
```

`asc_loadalign` 的基址必须 32B 对齐，残余元素偏移只能走 offset 形参
（`__clang_cce_vector_intrinsics.h:1639` 的
`vlds(vector&, __ubuf__ LT* base, int32_t offset /* in unit of element */, T dist)`）。
发射器把 `i * 100` 全部折进基址、offset 恒 0，所以奇数行落在半个 32B 块上。
`rows, k = 1` 的失败与对齐无关：并行轴 `extent == 1` 被 `Simplify` 折成 0，
nest 没有 lane，退化为 VF 内标量。

### 7.2 落地：P2 拒绝未对齐的整块流

`plan.cc::ValidateStreamAlignment`（诊断码 `plan.alignment`）在 P2 检查计划里的
整块 `vlds`/`vsts` 流：

- 要求 `lanes % (32B / elem_size) == 0`（f32=8、f16=16、i8=32 元素一块）；
- 把访问的平坦偏移在 `vector_axis = 0` 处化简，要求 `offset % 块元素数 == 0`
  （chunk 起点是 `chunk * lanes`，lanes 是块大小的整数倍，故只需查 0 号 chunk）；
- 只查连续流（`kVLoad` 且无谓词/不是唯一的掩码尾块）与 `kContiguous` store；
  gather/scatter 按元素寻址、单元素 `ONEPT` store 实测容忍未对齐基址（`8, 104`
  的 `vsts_1st(i + 832)`），都不受限；NZ 的 `vsstb` 原点由 layout 块基址决定，
  不在本轮检查范围。

拒绝是**按轴**的：`PlanSegment` 会继续尝试下一个候选轴。`rows, k = 8, 100`
因此落到"沿行轴 gather"的合法计划（`vgather2` + `ramp(100)`，结果正确、更慢），
只有单轴形态（如 `a[i + 4]`）才会整体拒绝。

设备用例已改为 32B 对齐的 `rows, k = 8, 104`；离线断言：
`test_expression_subscript` 的 `PAD` 由 3 改为 8（32B 整块），新增
`test_misaligned_stream_base_is_rejected`。

### 7.3 后续解除限制的路径（已确认可行，本轮不做）

`src/tl_templates/ascend/simd_inst.h` **无需改动**：
`vlds_norm(src, offset)` 与 `vsts_norm(data, base, offset, mask)` 已经把
offset 透传给 `asc_loadalign` / `asc_storealign`。只需：

1. `emit.cc::Vld`/`Vsts` 增加可选 offset 实参：`epb = 32 * 8 / bits`，
   `base = flat - flat % epb`，`offset = flat % epb`（把残余从最内维下标里减掉）；
2. `codegen_ascend.cc` 的 `simd_vld` / `simd_vsts` 打印该 offset（目前恒 `0`），
   `codegen_pto.cc` 同步（PTO 的 `pto.vlds` 本来就收 offset）。

### 7.4 `extent == 1`：P1 直接报错

`vsimd_analysis_common.h::NestCollector::RejectDeadParallelAxes` 现在先按
`IsOneExtent` 报出真因（"has extent 1: the nest has no lane to vectorize …"），
而不是落到"axis 从未使用"的提示。理由与 D3 相同：VF 内的标量回退不可编译，
所以这类"太小、只能标量"的 nest 必须显式拒绝（"必须能向量化"）。

离线断言：`test_unit_extent_parallel_axis_is_rejected`。

---

## 8. 第四轮设备报告：lane-uniform 谓词在 VF 内落成标量（2026-11-13，已修）

### 8.1 现象（设备实测）

用例 `test_parallel_lane_uniform_condition_broadcasts_the_mask`
（`if b[0] > 0.0: c[i] = a[i] * 2.0`，`b` 是 UB 上的 1 元素 buffer）bisheng 编译失败：

```
fatal error: error in backend: tl_kernel.asc:22:3: Unsupported scalar instruction in AIV loop.
```

生成代码（修复前）：

```cpp
__simd_vf__ inline void main_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  for (int32_t chunk_s0 = 0; chunk_s0 < 2; ++chunk_s0) {
    auto vs0_0 = simd_inst::update_mask_b32(
        ((float(0) < ((__ubuf__ float*)buf_dyn_shmem)[256]) ? 64 : 0));   // 标量读 + 标量 select
    ...
```

第二轮曾把它归类为"CANN 限制"，但那是误判：AIV loop 禁止的是**独立标量指令**，
而"标量作为向量指令的操作数"是合法的——同批上板通过的归约代码里就有
`vadds(vec, ((__ubuf__ float*)buf_dyn_shmem)[i + 832], …)`。

### 8.2 根因

两处把 lane-uniform 谓词折成了标量：

1. `value.cc::Compare` 对"两侧都 lane-uniform"的比较直接 `UniformGuarded`（标量布尔表达式）；
2. `emit.cc::Predicate` 对 `is_uniform` 节点发 `update_mask(Select(cond, lanes, 0))`。

`Select` 是一条标量指令（其操作数还需要一次标量 UB 读），落在 `__simd_vf__` 内即被拒。

### 8.3 修复：谓词改为向量化

- `value.cc::Compare`：两侧 uniform 时**只有编译期常量**（无 buffer load、无自由变量）
  才折成标量；运行期谓词保留向量形式 `vcmps(广播操作数, 标量)`。守卫检查
  （`RequireGuardableScalar`）仍然执行，不放松 lazy-conditional 的安全性。
- `value.cc::BoolNode` / `NegatePredicate` / `AndPredicate`：只有**全部操作数都有标量形式**
  时才折叠；否则用 `pand`/`por`/`pnot` 组合向量掩码（避免对没有 `scalar` 的
  `vcmps` 解引用）。
- `emit.cc::Predicate`：先分派谓词配方（`pnot`/`pand`/`por`/`pxor`、`vcmp`/`vcmps`），
  最后才处理"lane-uniform 标量叶子"；叶子用 `UniformLeafMask`：
  `mask = vcmps(vdup(cast(cond, int<bits>)), 0, ne)`，编译期常量直接折成
  `AllMask` / 空掩码。

修复后生成（设备用例同形）：

```cpp
__simd_vf__ inline void main_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  auto vs0_0 = simd_inst::vdup<float>(0.0f, mask_32, MODE_ZEROING);
  for (int32_t chunk_s0 = 0; chunk_s0 < 2; ++chunk_s0) {
    auto vs0_1 = simd_inst::vcmps_lt(vs0_0, ((__ubuf__ float*)buf_dyn_shmem)[256], mask_32);
    auto vs0_2 = simd_inst::vlds_norm<float>(...);
    auto vs0_3 = simd_inst::vmuls(vs0_2, 2.0f, mask_32, MODE_ZEROING);
    simd_inst::vsts_norm(vs0_3, ..., 0, vs0_1);
```

`vdup` 被提到 chunk 循环之外；每 chunk 只剩一条 `vcmps`（原来是一条 `update_mask`，
指令数不变或更少）。标量 UB 读只作为向量指令的操作数出现。

注意比较方向：条件进入 `Compare` 前已被规范化成 `0.0 < b[0]`（字面量在左），
所以广播的是字面量、谓词是 `lt`。逐 lane 用例里 `Compare` 的 `CompareFlip`
会把方向翻回 `gt`；两侧都 uniform 时无需翻（语义等价）。断言因此只固定
**指令族** `vcmps_`，不固定 `lt`/`gt`。

### 8.4 断言同步

- 设备：`test_parallel_lane_uniform_condition_broadcasts_the_mask` 改为断言
  `vdup<` + `vcmps_` 存在、`update_mask` 不存在。
- 离线（`test_ascend_parallel_vectorize_delta.py`）：三个 lane-uniform `vsel` 用例与
  R2 用例改为断言 `tl.simd.vcmps` / 不再断言 `tl.simd.update_mask`（尾块谓词除外）；
  "整条 fully-uniform select" 用例改为断言 `vsel` + `vcmps` + `vdup`
  （标量 `?:` 同样是 VF 内标量指令）。

### 8.5 新增的宽度约束

`vcmps` 的掩码粒度由**条件**的元素宽度决定（b8/b16/b32），而 `update_mask` 是唯一
能按写入宽度生成掩码的形态。因此"lane-uniform 条件 + 不同宽度的写入目标"（例如
f32 条件守卫 int8 store）现在没有合法形态，`emit.cc::Predicate` 用
`PredicateDataBits` 显式拒绝，避免把 b32 掩码交给 b8 store（原本会只写 1/4 的字节）。
lane-varying 谓词本来就有同样的宽度约束，只是此前没有检查。



