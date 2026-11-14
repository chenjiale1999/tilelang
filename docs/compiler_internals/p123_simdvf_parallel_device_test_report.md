# `T.Parallel` 降级（P1/P2/P3）全功能设备测试报告

> **测试对象**
>
> | 项 | 内容 |
> | --- | --- |
> | 分支 | `p_123_parallel_new_p1` |
> | 测试文件 | `testing/ascend/language/test_tilelang_ascend_simdvf_parallel.py`（77 例，合并后的全功能设备回归） |
> | 实现文档 | `docs/compiler_internals/p1_p2_p3_implementation_report.md` |
> | 对比文档 | `docs/compiler_internals/p123_vs_pto_parallel_comparison.md` |
> | 设备 | Ascend 950PR（NPU 0） |
> | CANN | 9.3.0.weekly.20260930（`bisheng` clang 15.0.5，`--npu-arch=dav-3510`） |
> | 构建 | 本地 `build/lib/libtilelang.so`（`PYTHONPATH=.`，全新 `TILELANG_CACHE_DIR`） |
>
> **更新（2026-11-12）**：本报告是第二轮上板结果。第一轮针对旧的
> `test_tilelang_ascend_vsimd_numeric.py`（31 例）已完成，结论见
> `p123_vs_pto_parallel_comparison.md` §8。本轮以覆盖更全的合并文件为准。

---

## 0. 结论速览

整文件 **40 passed / 37 failed**，耗时 176s。

37 个失败收敛到 **6 类根因**，归因如下：

| 归因 | 用例数 | 说明 |
| --- | --- | --- |
| **实现错误** | **5** | `vsstb` 步长编码 3 例、串行 shell 漏屏障 1 例、混合段归约壳未分块 1 例 |
| **测试文件自身错误** | 30 | 断言字符串 27 例、多输出 `out_idx` 3 例；这些用例的**数值全部正确** |
| **CANN 编译器限制** | 1 | VF 内标量条件触发 `Unsupported scalar instruction in AIV loop` |
| **既有管线语义** | 1 | `not(a > 0)` 在 polyhedral pass 之前被 `Simplify` 折成 `a <= 0` |

其中 `test_parallel_max_min_nan_and_signed_zero` 的 2 个失败单列在
"归因 D"：**实现符合已钉死的 CANN 契约，是测试 golden 写错**。

一句话：**5 个真实缺陷，30 个测试侧问题，1 个平台限制，1 个既有语义**。

> **测试侧修复后的现状（2026-11-12 复跑）**：30 个测试侧问题已全部修好，
> 整文件 **72 passed / 5 failed**。剩余 5 个失败为：
> `compact_nz_store`、`paired_nz_store_with_an_nd_sibling`、`mixed_recipe`（D1，
> 其中 `mixed_recipe` 还叠加 D3）、`under_a_serial_shell`（D2）、
> `lane_uniform`（CANN 限制）。实现侧修复清单见
> `p123_parallel_lowering_real_defects.md`。
>
> **更新（2026-11-13）**：D1/D2/D3 修复后复测
> `test_parallel_row_reduction_with_a_tail`（原 `rows, k = 8, 100`，f32 行距
> 400B）仍失败：行起始地址不是 32B 的整数倍，`vlds_norm` 在奇数行解错 32B 块
> （输出 507035）；`8, 104`（416B）通过。该形态现按**不支持**处理：P2 以
> `plan.alignment` 拒绝未对齐的整块流（nest 还有其它轴时换轴为 gather 形态），
> 设备用例已改为 32B 对齐的 `8, 104`；`rows = 1` 的"只能标量"形态由 P1 显式
> 拒绝。详见 `p123_parallel_lowering_real_defects.md` §7。
>
> 同一轮还复核了 `lane_uniform`：之前归因为"CANN 限制"是误判——AIV loop 禁止的是
> **独立标量指令**，而"标量作为向量指令操作数"合法。根因是 lane-uniform 谓词被
> 折成 `update_mask(Select(...))`（标量读 + 标量 select）。已改为向量谓词
> （`vdup` + `vcmps`），详见 `p123_parallel_lowering_real_defects.md` §8。


---

## 1. 失败行号聚类

`pytest --tb=line` 的失败行号直方图：

| 行号 | 次数 | 位置 |
| --- | --- | --- |
| `1189` | 24 | `test_parallel_cast_pair` 的 `assert "simd_inst::vcvt(" in source` |
| `54` | 4 | `_assert_close`（数值不匹配） |
| `242` | 2 | `test_parallel_eight_bit_full_chunk` 的 `assert "simd_inst::vlds_norm(" in source` |
| `630` | 1 | `test_parallel_condition_and_or_not` 的 `assert "simd_inst::pnot(" in source` |
| `827` | 1 | `test_parallel_under_a_serial_shell` 的 `assert "simd_inst::vlds_norm(" in source` |
| 其余 5 | — | `ValueError: Kernel expected N inputs`（3）、`RuntimeError: Ascend device compilation failed`（1）、`assert vcvt/vsts_pack_b16`（1） |

行号高度集中说明：**多数失败是同一个断言写法造成的**，不是 24 个独立缺陷。

---

## 2. 归因表

| # | 根因 | 用例 | 归因 | 数值是否正确 |
| --- | --- | --- | --- | --- |
| A | 断言 `simd_inst::vcvt(` 缺模板实参 | 24（`cast_pair` 全部） | 测试侧 | ✅ 24/24 逐位正确 |
| B | `vlds_norm(` / `vsts_pack_b16` 断言与实际生成不符 | 3（2×`eight_bit_full_chunk`、`fp16_to_fp8`） | 测试侧 | ✅ 全对 |
| C | 多输出 kernel 用 `out_idx=-1` | 3（`sibling_reduction`、`multiple_independent`、`mixed_recipe`） | 测试侧 | ✅ 前两个全对 |
| D | max/min golden 用 `a>b?a:b` | 2 | 测试侧 | 实现符合 CANN 契约 |
| E | `vsstb` 步长低 16 位写成 1（应为 0） | 3（`compact_nz`、`paired_nz`、`mixed_recipe` 的 C） | **实现错误** | ❌ |
| F | 串行 shell 的 RMW 缺 VST→VLD 屏障 | 1（`under_a_serial_shell`） | **实现错误** | ❌ |
| G | 混合段归约壳 extent 未按 chunk 分块 | 1（`mixed_recipe` 的 D） | **实现错误** | ❌ |
| H | VF 内标量条件 | 1（`lane_uniform`） | CANN 限制 | 未执行 |
| I | `not(a>0)` 被 Simplify 折成 `a<=0` | 1（`condition_and_or_not`） | 测试断言 + 既有管线 | NaN 语义偏离 |

---

## 3. 逐类证据

### A. `cast_pair` 24 例 —— 断言字符串缺模板实参

实际生成的是带模板实参的调用：

```c
simd_inst::vcvt<int32_t>(vs0_0, mask_32, ROUND_R, RS_ENABLE, MODE_ZEROING);
simd_inst::vcvt<float>(vs0_0, mask_16, PART_EVEN, MODE_ZEROING);
simd_inst::vcvt<fp8_e4_t>(vs0_1, mask_8, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING);
```

测试断言的是无模板实参的 `simd_inst::vcvt(`，因此 24 个用例全部在断言处失败。

逐对跑完全部 24 个转换（含 `uint32→int16/uint16/uint8`、`int16→uint32`、
`uint16→uint32`、`bfloat16→float8_e4m3fn` 等），**24/24 数值逐位正确**。
即这 24 个失败与降级实现无关。

**修法**：断言改为 `"simd_inst::vcvt<" in source`。

### B. 8-bit full chunk / fp16→fp8 —— 断言字符串错误

- `test_parallel_eight_bit_full_chunk`：实际生成
  `simd_inst::vlds_norm<uint8_t>(...)` / `simd_inst::vsts_norm(vs0_0, ...)`，
  测试断言无模板实参的 `vlds_norm(`。数值正确。
- `test_parallel_fp16_to_fp8_store`：实际生成 f16→f32→fp8 的两步配方
  （`vcvt<float>(PART_EVEN)` → `vcvt<fp8_e4_t>(PART_P0)` → `vsts_pack_quarter`），
  测试却断言 `vsts_pack_b16`。16 个 e4m3 精确值逐位相等。

**修法**：`vlds_norm(` → `vlds_norm<`；`vsts_pack_b16` → `vsts_pack_quarter`。

### C. 多输出 kernel 的 `out_idx` —— 测试构造错误

`kernel(a.npu(), b.npu())` 只传 2 个实参，但 kernel 有 4 个参数（2 入 2 出）。
`out_idx=-1` 只把最后一个参数当输出，前一个输出仍被当作输入，
于是报 `Kernel expected 3 inputs, but 2 are provided`。

改成 `out_idx=[-2, -1]` 后：

- `sibling_reduction_shells_are_fissioned`：`vcadd` 恰好 2 条，C/D 全对 ✅
- `multiple_independent_stores`：`vadds` + `vmuls`，`vsts_norm` 2 条，C/D 全对 ✅
- `mixed_recipe`：C/D 另有真 bug（见 E / G）

**修法**：`out_idx=[-2, -1]`（或 `[1, 2]`）。

### D. max/min —— 实现符合契约，golden 写错

950PR 实测 `vmax` / `vmin`：

| 输入 | `max` 实测 | `min` 实测 |
| --- | --- | --- |
| `(NaN, 1.0)` | NaN | NaN |
| `(1.0, NaN)` | NaN | NaN |
| `(NaN, NaN)` | NaN | NaN |
| `(+0, -0)` | **+0** | **-0** |
| `(-0, +0)` | +0 | -0 |
| `(+0, +0)` | +0 | +0 |
| `(-0, -0)` | -0 | -0 |
| `(±inf)` | 正常 | 正常 |

即：**NaN 传播**；`max` 仅在两个操作数都是 `-0` 时得 `-0`；
`min` 仅在两个都是 `+0` 时得 `+0`。这与
`p123_vs_pto_parallel_comparison.md` §3.6 已改过的契约完全一致。

测试仍用 `torch.where(a > b, a, b)`（NaN 时取另一侧、`max(+0,-0)` 得 `-0`），
因此失败。注意 NaN 的符号位也不同：硬件返回的是无符号 NaN，
`torch.maximum` 返回的是带符号 NaN，逐位比较时也要按契约处理。

**修法**：golden 按 CANN 契约重写（NaN 传播 + `±0` 规则），
不要用 `torch.where(a>b,a,b)`，也不要直接和 `torch.maximum` 做逐位比较。

### E. `vsstb` 步长低 16 位 —— 实现错误（3 例）

`emit.cc` 的 NZ 直存编码：

```cpp
int64_t block = store->nz_block_stride.value_or(0);
int64_t elem  = store->nz_elem_stride.value_or(1);
PrimExpr stride = IntImm(DataType::Int(32), (block << 16) | (elem & 0xffff));
```

而 `plan.cc` 里 `nz_elem_stride` 恒为 `1`，于是发出 `524289 = (8 << 16) | 1`。

设备实测（f32 / bf16、单 chunk / 多 chunk、普通 store / post-update 全覆盖）：

```
block=8, low=0 → 与 compact-NZ 物理序逐位相等
block=8, low=1 → 每个 32B 块在 base+1 块处重复写入，输出错乱
```

CANN 的 `asc_store_align_config` 是 `{ repeat_stride : 16; block_stride : 16 }`，
`repeat_stride` 并不是"块内元素步长"，填 1 会让硬件按 1 块重复。

影响用例：`test_parallel_compact_nz_store`（93.8% 元素不匹配）、
`test_parallel_paired_nz_store_with_an_nd_sibling`（93.8%，出现 NaN）、
`test_parallel_mixed_recipe_is_split_into_two_segments` 的 C 输出。

> **文档需同步修正**：`p2_p3_design_implementation_report.md` §F.5.2
> 把 `(129 << 16) | 1` 当作已验证配方，该结论按本次实测应改为 `(129 << 16)`。

**修法**：`emit.cc` 两处（`EmitVectorWrite` 与 `EmitPackedNzStore`）去掉
`elem` 位，改为 `block << 16`；`plan.cc` 的 `nz_elem_stride` 字段随之废弃或恒 0。

### F. 串行 shell 的 RMW 缺内存屏障 —— 实现错误

```python
with T.SimdVF():
    for k in T.serial(4):
        for i in T.Parallel(rows):
            c[i] = c[i] + a[k, i]
```

生成：

```c
for (k = 0; k < 4; ++k)
  for (chunk = 0; chunk < 2; ++chunk) {
    vlds_norm(c + chunk*64);
    vlds_norm(a + k*128 + chunk*64);
    vadd;
    vsts_norm(c + chunk*64);
  }
```

相邻 `vsts` 与下一轮 `vlds` 之间没有 `mem_bar(VST_VLD)`，读改写跨迭代竞争。

实测矩阵：

| rows | steps | 结果 |
| --- | --- | --- |
| 64 | 4 | ✅（单 chunk，恰好保序） |
| 128 | 1 / 2 | ✅ |
| 128 | 3 / 4 / 8 | ❌ |
| 192 / 256 | 4（纯写） | ✅ |

手工在每次 shell 迭代末尾插入 `T.simd.mem_bar("VST_VLD")` 后**逐位正确**。

> **推翻旧结论**：第一轮曾把该现象判为 CANN `bisheng -O2` 误编译。
> 本次证据表明是**发射侧漏屏障**，可修，不是平台限制。

**修法**：P3 在"shell 包裹的向量体"每次迭代末尾补 `mem_bar(VST_VLD)`
（或按写后读依赖插入）。

### G. 混合段归约壳 extent 未分块 —— 实现错误

`mixed_recipe` 的 `d[i] += s[i, k]` 在"elementwise + NZ"段被拆分后生成：

```c
for (j = 0; j < 128; ++j) {      // 应为 N / chunk = 2
  vcadd(chunk0); vcadd(chunk1); vadd; vadds; vsts_1st
}
```

循环按整行 128 次迭代，每次累加整行和，D 严重过计
（全 1 输入得 4096，应为 128）。

对照组：**纯归约段（无 elementwise 兄弟）是正确的**——
`for i in T.Parallel(8): for j in T.serial(128): c[i] += a[i,j]`
直接生成 `vcadd` 且无 `j` 循环，实测逐位正确。

**修法**：段拆分后归约壳需要按 chunk 分块（`N / lanes` 次），
而不是保留整行 `for j < N`。

### H. lane-uniform 条件 —— CANN 编译器限制

```c
auto vs0_0 = simd_inst::update_mask_b32(((0.0 < shmem[256]) ? 64 : 0));
```

bisheng 报：

```
fatal error: error in backend: tl_kernel.asc:22:3:
Unsupported scalar instruction in AIV loop.
```

两种形态对照：

- 标量条件表达式写在 `T.SimdVF()` 内 → 编译失败（CANN 限制）
- 把标量条件提到 `T.SimdVF()` 之外再 `update_mask` → **编译通过且数值正确**

即 CANN 不允许在 AIV 循环体内做标量分支/UB 标量读。

**处理**：记录为已知限制；若要让该用例通过，实现侧应把 lane-uniform
标量条件外提到 VF 之外。

### I. `condition_and_or_not` —— 断言 + 既有管线语义

`pand` / `por` 都生成了，但没有 `pnot`：`not(a[i] > 0.0)` 在 polyhedral
pass 之前就被 `Simplify` 折成 `a[i] <= 0.0`，生成 `vcmps_le`。

同样的折叠在**标量路径**上也发生（`__cond_0 = shmem[i] <= 0.0`），
所以是既有管线行为，不是本分支引入。

副作用：NaN 时 `not(a > 0)` 应为真，`a <= 0` 为假 ——
实测 NaN 位置输出 0 而非 3。这属于语义偏离（IEEE 意义上是 bug），
但影响面在 `Simplify`，不在 P1/P2/P3。

**处理**：断言改为接受 `vcmps_le`，或把源改成不会被 Simplify 折叠的形态。

---

## 4. 修复清单

### 实现侧（按优先级）

| 优先级 | 改动 | 位置 | 影响用例 |
| --- | --- | --- | --- |
| P0 | `vsstb` 步长改为 `block << 16`（去掉低 16 位 `elem`） | `src/ascend/transform/polyhedral/emit.cc`（`EmitVectorWrite`、`EmitPackedNzStore`） | 3 |
| P1 | shell 包裹的向量体每次迭代末尾补 `mem_bar(VST_VLD)` | P3 发射 | 1 |
| P2 | 混合段拆分后的归约壳按 chunk 分块 | P2 规划 / P3 发射 | 1 |
| — | 同步修正 `p2_p3_design_implementation_report.md` §F.5.2 的 `(129<<16)\|1` | 文档 | — |

### 测试侧（已落地）

| 改动 | 用例数 | 状态 |
| --- | --- | --- |
| `"simd_inst::vcvt("` → `"simd_inst::vcvt<"`（反向断言同步改为不带 `(`） | 24 | ✅ |
| `"simd_inst::vlds_norm("` → `"simd_inst::vlds_norm<"` | 2 | ✅ |
| `"vsts_pack_b16"` → `"vsts_pack_quarter"` | 1 | ✅ |
| 多输出 kernel 的 `out_idx=-1` → `[-2, -1]` | 3 | ✅ |
| `condition_and_or_not` 三个输出 → `out_idx=[-3, -2, -1]` | 1 | ✅ |
| max/min golden 按 CANN 契约重写 | 2 | ✅ |
| `pnot` 断言改为接受 `vcmps_le`（`not(a>0)` 被 `Simplify` 折成 `a<=0`） | 1 | ✅ |

修完测试侧后整文件为 **72 passed / 5 failed**，剩余失败恰好是
D1（`compact_nz_store`、`paired_nz_store_with_an_nd_sibling`、`mixed_recipe`）、
D2（`under_a_serial_shell`）与 1 个 CANN 限制（`lane_uniform`）。

### 平台限制（记录，不修）

| 限制 | 用例 |
| --- | --- |
| VF 内标量条件 → `Unsupported scalar instruction in AIV loop` | 1 |

---

## 5. 复现方式

```bash
cd /home/c00986984/tilelang
export PYTHONPATH=.
export ASCEND_RT_VISIBLE_DEVICES=0
export TILELANG_CACHE_DIR=/tmp/tlcache_simdvf_$(date +%s)

python -m pytest testing/ascend/language/test_tilelang_ascend_simdvf_parallel.py \
  -q -p no:cacheprovider
```

单用例隔离（区分真实失败与设备状态连带失败）：

```bash
python -m pytest \
  "testing/ascend/language/test_tilelang_ascend_simdvf_parallel.py::test_parallel_compact_nz_store" \
  -q -p no:cacheprovider --tb=short
```

本轮已按用例逐个隔离验证：37 个失败在单跑时**全部复现**，
不存在第一轮那种"设备写坏后连带失败"的情况。

---

## 6. 与第一轮（numeric 文件）的关系

| 项 | 第一轮 | 本轮 |
| --- | --- | --- |
| 文件 | `test_tilelang_ascend_vsimd_numeric.py`（31 例） | `test_tilelang_ascend_simdvf_parallel.py`（77 例，合并） |
| 结果 | 31 passed | 40 passed / 37 failed |
| 串行 RMW | 判为 CANN `-O2` 限制（已改只读形态） | **推翻**：漏 `VST_VLD` 屏障，可修 |
| max/min | 契约改为以 CANN 为基准 | 实现符合契约，测试 golden 未同步 |
| NZ 直存 | 未覆盖 | 首次上板，暴露 `vsstb` 步长错误 |

合并文件新增的覆盖维度（NZ 直存、配对 pack2、混合段拆分、多输出、
加宽/窄化 cast 全矩阵）正好覆盖了第一轮未触及的发射路径，
因此本轮暴露的 3 个实现缺陷都是**新增覆盖带来的真实收益**。
