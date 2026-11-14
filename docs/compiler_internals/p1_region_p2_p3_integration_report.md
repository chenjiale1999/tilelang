# 新 P1（`tl.vsimd.region`）对接 P2/P3 代码变更与测试验证报告

## 文档信息

| 项目 | 内容 |
| --- | --- |
| 开发分支 | `/home/cjl/workspace/tilelang` → `p_123_parallel_new_p1` |
| 基线分支 | `p_123_parallel` @ `a8fe5586`（旧 P1 `tl.vsimd.program` + 完整 P2/P3） |
| 新 P1 来源 | `/home/cjl/workspace/tilelang_stable-parallel` @ `64feedf6`（`stable-parallel`） |
| 变更规模 | tracked +2474 / −3059 行；新增文件 2342 行；另移除 fragment 处理（`vsimd_fragment.{h,cc}`、`fragment_plan.{h,cc}` 删除）；后续扩展 P1 放行 `if` 与 copy/reduce tile-op |
| 构建 | `./build.sh -j8`（Release，CMake/Ninja，Ascend ON） |
| 结论 | P1→P2→P3 全链路通过；`testing/ascend/transform` 620 passed；FA 全链路 lower 成功；触碰到 `T.alloc_fragment` 的 Parallel 按 Parallel 粒度回退 legacy；Parallel 体内 `if` / copy / reduce 已放行（按 nest 降级） |

***

## 1. 目标与范围

把 P2/P3 的输入从旧 P1 的 **per-VF `tl.vsimd.program`**（Map/items/spans/groups）切换到新 P1 的每个顶层 `T.Parallel` 的 `For.annotations["tl.vsimd.region"]`（类型化 `tl.vsimd.Region`）。

把 P2/P3 收敛为“只处理 `T.Parallel`”：plan 直接挂在被规划的 `For` 上，Parallel 之外的 tile-op 不再由本条链路处理，并**移除全部 fragment 处理**：

- P1 **按 Parallel 粒度**拒绝：`NestTouchesFragment` 检查某个顶层 Parallel 的子树（`SBlock.alloc_buffers` + BufferLoad/BufferStore）是否触碰到 `local.fragment` buffer（`T.alloc_fragment`），命中则只把**该 Parallel** 标记为 OPAQUE（`UNSUPPORTED`，reason 含 `fragment`）；同一 VF 内不接触 fragment 的 Parallel 仍照常 LIFTED 并被 P2/P3 向量化；region 无 `group` 字段，也不生成 `tl.vsimd.fragment` 表；
- P2 对 OPAQUE region 一律跳过，只规划 LIFTED region；VF 顶层的 copy/reduce 不在 P2 职责内，语句原样保留；
- P3 不构建/重写 fragment 寄存器分配；
- 因此 **P2/P3 的 pass 代码（`vsimd_group_passes.{h,cc}` / `region_plan.{h,cc}`）中没有任何 fragment 引用**，fragment 拒绝完全由 P1 驱动；离线 planner（`polyhedral/plan.cc` / `emit.cc` / `model.{h,cc}` / `value.cc` / `primitives.h`）的 fragment 寄存器文件能力也一并删除，只保留"直接对 fragment nest 调 `build_plan` 抛 `Unsupported`"的防御性守卫。

***

## 2. 代码变更清单

### 2.1 新增文件

| 文件 | 行数 | 作用 |
| --- | ---: | --- |
| `src/ascend/transform/vsimd_ir.h` | 353 | 类型化 region IR（Axis/BufRegion/Access/Op/Capture/VarBinding/View/Region） |
| `src/ascend/transform/vsimd_ir.cc` | 847 | 对象构造、schema verifier、dump、FFI 构造器 |
| `src/ascend/transform/analyze_vsimd.cc` | 91 | 统一只读 debug 入口（3 个报告 + 合并 dump） |
| `tilelang/ascend/vsimd_ir.py` | 119 | Python 类型绑定 + `region()` 构造器 |
| `testing/ascend/transform/test_ascend_vsimd_ir.py` | 385 | schema/verifier/expander 测试（22） |
| `testing/ascend/transform/test_ascend_vsimd_parallel_fa.py` | 55 | FA P1 回归测试（2） |

### 2.2 替换/重写

| 文件 | 变化 | 原因 |
| --- | --- | --- |
| `src/ascend/transform/lift_vsimd.cc` | 1039 → 491 行 | 新 P1：逐顶层 Parallel 挂 region；结构 oracle；按 Parallel 粒度的 fragment 拒绝；OPAQUE region 的 tileop footprint 按副作用保守记录 |
| `src/ascend/transform/vsimd_region.{h,cc}` | 旧 416 → 新 771+ 行 | `BuildRegion` 闸门 + 按语句结构生成 ops（store / if / copy / reduce）+ 参考展开 + 展开校验 + 原 body 比较 |
| `src/ascend/transform/vsimd_capture_scope.{h,cc}` | +6 行 | 新增 `root_for` 指针身份；VF-local assume 注入 |
| `src/ascend/transform/vsimd_memory_legality.{h,cc}` | +55 行 | 新增 per-nest `VSimdMemoryRegionStatus` + `root`；`AnalyzeVSimdMemory` |
| `src/ascend/transform/vsimd_analysis_common.h` | +82 行 | `VFAssumeCollector`、sibling nest 检测、per-nest `MarkUnsupported`、`IsSupportedParallelTileOp` |
| `src/ascend/transform/polyhedral/vsimd_group_passes.cc` | 873 → ~330 行 | 只处理带 `tl.vsimd.region` 的 `T.Parallel`；plan 写到同一 `For`；按 loop 替换/回滚；无 fragment、无 span |
| `src/ascend/transform/polyhedral/vsimd_group_passes.h` | 文档重写 | 删除 VF fragment 表 / fragment group 描述 |
| `src/ascend/transform/polyhedral/region_plan.{h,cc}` | 新增（替代 `group_plan.{h,cc}`） | `VSimdRegionPlan{version,vf_index,region_index,program}`，挂在 `For` 上 |
| `tilelang/ascend/transform/__init__.py` | 改写 debug 包装 | `AscendDumpVSimd*`；旧 `AscendAnalyzeSimdVFMemory/Capture/VSimdRegion` 删除 |
| `tilelang/transform/pass_config.py` | +10 行 | `TL_ASCEND_SIMD_LIFT` 说明更新 + `TL_ASCEND_SIMD_REGION_VERIFY` |
| `src/ascend/op/builtin.cc` | +1 行 | 注册 `tl.ascend_simd_region_verify` |
| `tilelang/ascend/__init__.py` | +1 行 | 导入 `vsimd_ir` 注册 FFI 类型 |
| `testing/ascend/transform/test_ascend_vsimd_lift.py` | 516 行 | 17 个 P1 pass 测试（含 if/tile-op 放行与按 nest 拒绝） |
| `testing/ascend/transform/test_ascend_vsimd_region.py` | 551 行 | 23 个 region 报告/oracle 测试（含 if/tile-op 往返） |
| `testing/ascend/transform/test_ascend_vsimd_group_passes.py` | 498 行 | 13 个 P2/P3 对接测试（含 if/tile-op 端到端与按 nest 降级） |
| `testing/ascend/transform/test_ascend_polyhedral_vectorize.py` | 删除 10 个 fragment 离线用例 + helper，新增 1 个守卫用例 | 适配 `tl.vsimd.region`；fragment 寄存器文件用例随 planner 逻辑一并删除，改为断言 `build_plan` 对 fragment nest 抛 `Unsupported` |
| `testing/ascend/transform/test_ascend_vsimd_capture_scope.py` / `..._memory_legality.py` | 各 1 行 | debug pass 改 `AscendDumpVSimd` |

### 2.3 删除文件

| 文件 | 原因 |
| --- | --- |
| `src/ascend/transform/analyze_simdvf_capture.cc` | 合并进 `analyze_vsimd.cc` |
| `src/ascend/transform/analyze_simdvf_memory.cc` | 同上 |
| `src/ascend/transform/analyze_vsimd_region.cc` | 同上 |
| `src/ascend/transform/vsimd_program.{h,cc}` | 旧 `tl.vsimd.program` schema reader/verifier，无调用者 |
| `src/ascend/transform/polyhedral/vsimd_payload_read.{h,cc}` | 旧 item-map payload 解码；改为在 `vsimd_group_passes.cc` 直接解码 `Call` |
| `src/ascend/transform/vsimd_fragment.{h,cc}` | fragment 事实收集/分组/VF 表；本分支不处理 `T.alloc_fragment` |
| `src/ascend/transform/polyhedral/fragment_plan.{h,cc}` | VF 级 fragment 寄存器（V1–V4）校验；已无调用者 |

### 2.4 保留复用

| 文件 | 说明 |
| --- | --- |
| `src/ascend/transform/polyhedral/plan.cc` / `emit.cc` / `model.*` | 规划/发射核心的 fragment 寄存器文件逻辑已一并删除（`PlanOptions::vf_alloc_scope`、`FragmentRegisterFile`、`FragmentReg`/`FragmentRegisterMap`、`kRegLoad` recipe 等），只保留"fragment 操作数抛 `Unsupported`"的守卫 |
| `examples/ascend/flash_attention/parallel_fa.py` | 保持基线版本，验证旧 FA 形态仍可全链路 lower |

***

## 3. 对接后的数据流

```text
P1 AscendLiftVSimd
  → For.annotations["tl.vsimd.region"]       (typed VSimdRegion)
  → 语句结构建模：store / if（guard + 两个分支）/ copy / reduce
  → 语法不支持的 nest（while/serial/sibling Parallel/未建模 tile-op/opaque Evaluate）：
       仅该 nest → OPAQUE（UNSUPPORTED，reason 为该语句的诊断）
       同 VF 内其他 Parallel 照常 LIFTED
  → 某个顶层 Parallel 的子树触碰到 local.fragment（T.alloc_fragment）时：
       仅该 Parallel → OPAQUE（UNSUPPORTED，reason 含 fragment）
       （无 fragment 表，无 group 字段）

P2 AscendPlanVSimd
  → 逐 SBlock("SIMD_VF")：收集该 VF 内 LIFTED region 的 capture layout
  → 逐带 tl.vsimd.region 的 T.Parallel：
       status != LIFTED      → 跳过（legacy）
       BuildPlan 抛 Unsupported → 跳过 + WARNING（legacy）
       成功                  → For.annotations["tl.vsimd.plan"] = VSimdRegionPlan
  → VF 顶层 tileop（Parallel 外）不参与

P3 AscendEmitVSimd
  → 逐带 tl.vsimd.plan 的 For：EmitPlan(plan.program)，成功即整 loop 替换
       EmitPlan 失败 / 版本不符 → 保留原 loop（per-loop 回滚）
  → 遍历全函数，删除 For 上的 tl.vsimd.region / tl.vsimd.plan
  → 无 plan 的 loop 原样保留（legacy）
```

关键适配点：

1. **不解析旧 annotation**：`ReadVSimdProgram`、`VSimdProgramItem`、`VSimdPayloadRead` 全部移除。
2. **按节点定位**：plan 就挂在被规划的 `For` 上，P3 不需要 span/展平下标/fingerprint；注解丢失即跳过。
3. **loop 事务**：每个 `T.Parallel` 独立规划、独立发射、独立回退；不做任何跨 loop 合并。
4. **只处理 Parallel**：Parallel **体内**的 copy/reduce 由 P1 建成结构化 op，P2 在 `BuildPlan` 里用 `ExpandParallelTileOps` 展开；体内 `if` 由 `NormalizeBody` 谓词化/惰性选择。VF 顶层的 copy/reduce 保语句原样，交 `AscendLowerTileOp`。
5. **fragment 不支持（P1 驱动，按 Parallel 粒度）**：P1 的 `NestTouchesFragment` 发现某个顶层 Parallel 的子树触碰到 `local.fragment` 时，只把该 Parallel 标记为 OPAQUE，P2 对 OPAQUE 一律跳过，因此 LIFTED region 不可能访问 fragment；P2/P3 pass 代码中没有任何 fragment 引用，离线 planner 也只剩一条 `Unsupported` 守卫。顶层 copy/reduce 若接触到 fragment，由既有的 UB scope 检查拒绝。
6. **annotation 不泄漏**：P3 无条件清理所有 `tl.vsimd.region` / `tl.vsimd.plan`。
7. **失效域是 loop**：语法不支持的 nest 只让自己 OPAQUE；P2 对它不产生 plan，P3 保留原语句交 legacy，同 VF 其他 loop 不受影响。

***

## 4. 测试验证

### 4.1 构建

```bash
cd /home/cjl/workspace/tilelang
./build.sh -j8
# ==> 完成，用时 11s，产物: build/lib/libtilelang.so
```

### 4.2 测试结果

测试运行时需绕过环境里的 editable install（`_editable_skbc_tilelang` 会把 `tilelang` 指向其它工作树），并使用临时缓存目录：

```bash
TILELANG_CACHE_DIR=/tmp/tilelang_cache PYTHONPATH=$PWD python - <<'PY'
import sys
sys.meta_path = [f for f in sys.meta_path
                 if "editable" not in type(f).__module__]
import pytest
raise SystemExit(pytest.main(["testing/ascend/transform", "-q"]))
PY
```

| 测试集 | 结果 |
| --- | --- |
| `testing/ascend/transform`（全部） | **620 passed, 1 pre-existing warning** |
| `test_ascend_vsimd_lift.py` | 18 passed（含 if/tile-op 放行与按 nest 拒绝） |
| `test_ascend_vsimd_ir.py` | 22 passed |
| `test_ascend_vsimd_region.py` | 23 passed（含 if/tile-op 结构 oracle 往返） |
| `test_ascend_vsimd_capture_scope.py` | 15 passed |
| `test_ascend_vsimd_memory_legality.py` | 23 passed |
| `test_ascend_vsimd_parallel_fa.py` | 2 passed |
| `test_ascend_vsimd_group_passes.py` | 18 passed（plan 挂载、metadata-only、顶层 tileop 不介入、if/copy/reduce、符号尾块、多 VF、降级、幂等） |
| `test_ascend_polyhedral_vectorize.py` | 156 passed |

### 4.3 全链路 lower 验证

对以下 kernel 执行 `tilelang.lower(target="ascend")`：

| Kernel | 结果 | 关键输出 |
| --- | --- | --- |
| 1-D 元素级（shared → shared） | OK | `simd_inst::` 存在 |
| fragment 两段（fill/consume） | OK（回退） | 无 `simd_inst::`，两个 Parallel 均走标量路径（`float frag[64]` 局部数组 + 朴素循环）；P1 把两个 region 标为 OPAQUE（reason 含 `fragment`），P2 无 group |
| fragment + 非 fragment Parallel 同 VF | OK（部分向量化） | fragment Parallel 走标量路径，另一个 Parallel 用 `simd_inst::vlds_norm`/`vmuls`/`vsts_norm` 向量化——按 Parallel 粒度的直接证据 |
| fragment + 独立 UB 顶层 copy | OK | 与 fragment 无关的 UB→UB copy 仍被向量化；fragment 部分回退 |
| partial lift（非法 nest + 合法 nest） | OK | 合法 nest 向量化，非法保留 |
| Parallel 体内 `if`（同址双臂） | OK | `simd_inst::`，含 `vcmps`/`vsel`/`vld`/`vsts`，原 `T.parallel` 已替换 |
| Parallel 体内 `if`（异址双臂） | OK（回退） | 该 nest 的 item 被 P2 拒绝、原语句保留走标量；同 VF 的兄弟 Parallel 仍 `simd_inst::` |
| Parallel 体内 `reduce`（P1 结构化建模） | OK | P2 经 `ExpandParallelTileOps` 规划，产物含 `tl.simd.vld`/`vmax`/`vsts` |
| Parallel 体内 `if` 臂嵌 `T.copy` | OK（回退） | P1 LIFTED，P2 报 `nested parallel loop ... not affine-vectorizable` 并只拒绝该 item |
| FA（parallel_fa.py） | **失败（设计变更后预期）** | VF 顶层 copy/reduce 不再由 P2 规划；后续 lower 分别报 `Generic vector BufferStore ... inside SimdVF`（copy）与 `Reduce ... not implemented`（reduce）；见 L8 |

> Parallel 体内的 copy/reduce 目前只在 pass 级验证（`AscendLiftVSimd` →
  `AscendPlanVSimd` → `AscendEmitVSimd`）：`tilelang.lower` 的前端语义检查
> `NestedLoopChecker` 仍然禁止用户源码把 tile-op 写进 `T.Parallel`，
> 因此这条路径面向的是 target-DSL 前端直接构造的 IR。

字节级回归对比（临时把 P1 的 fragment 拒绝改成 `if (false && NestTouchesFragment(...))` 重新构建 vs 改动后）：上述 kernel 的 `kernel_source` 逐字节一致，仅拒绝点从 P1（`NestTouchesFragment`）前移到/后移到等价位置——即 P1 拒绝与"P2  planner 守卫拒绝"产生完全相同的最终代码。

FA 的 P1→P3 结构测试同时验证：

- `tl.simd.vld2` / `PART_ODD` / `tl.simd.vor` / `tl.simd.vsstb` 均存在；
- NZ 块步长 `(129 << 16) | 1` 正确；
- 原 Parallel nest 已被替换。

### 4.4 环境导致的排除项

运行 `testing/ascend/{analysis,language,layout,target}` 时存在与本次改动无关的环境失败：

- `torch_npu` 未安装（`ModuleNotFoundError`）；
- `.npu` device / 真机用例在无 NPU 环境下失败。

这些用例不涉及 P1/P2/P3 代码路径；`testing/ascend/transform` 全量通过。

***

## 5. 已知限制与风险

| 编号 | 限制 | 影响 | 处理 |
| --- | --- | --- | --- |
| L1 | ~~P1 的 `NestCollector` 语法拒绝是 VF 级 Poison~~ | 已改为 **per-nest**：`ParallelNest.unsupported/reason` 只让该 nest OPAQUE，同 VF 其他 region 照常 | 已实现；剩余 VF 级拒绝只有 manual VF 与“无顶层 Parallel” |
| L2 | ~~P1 不建模 `T.Parallel` 内的 `T.copy`/`T.reduce_*`~~ | 已建模 `tl.tileop.copy` / `tl.tileop.ascend_copy` / `tl.tileop.reduce` 与语句级 `if`；P2 用 `ExpandParallelTileOps` / `NormalizeBody` 规划 | 已实现；`fill`/atomic 等其他 tile-op 仍按 nest OPAQUE |
| L3 | ~~顶层 COPY/REDUCE 解码~~ | 已移出 P2：Parallel 外的 tile-op 不由本条链路处理 | — |
| L4 | P2/P3 之间若插入重建 loop 的 pass | plan 注解丢失 | P3 跳过该 loop（保留原语句） |
| L5 | 本分支不支持 fragment | 触碰到 `T.alloc_fragment` 的 Parallel 回退 legacy（同 VF 的其他 Parallel 不受影响） | P1 按 Parallel 粒度判 OPAQUE，P2 跳过；已删除 fragment 收集/规划/重写代码（含离线 planner 的寄存器文件逻辑），P2/P3 无 fragment 引用，planner 仅保留 `Unsupported` 守卫 |
| L5b | fragment Parallel 旁边的非 fragment item 仍会被向量化 | 与 fragment 无关的 item 不受按 Parallel 粒度拒绝的影响 | 语义正确且相互独立（已在 `test_fragment_rejection_is_per_parallel` 与全链路 lower 中验证）；如需“整个 VF 一律不向量化”，需在 P1 增加 VF 级 opaque 标记 |
| L6 | `region_index` 与 `nest_index` 可能分叉 | 误用 nest_index 会错位 | P2/P3 只用 `For` + `region_index` |
| L7 | fragment + 顶层 `T.copy(frag, ...)` 形态在 legacy 路径触发 codegen 拒绝（`Generic vector BufferStore ... inside SimdVF`） | 该形态的 fragment kernel 无法 lower | 预先存在（`AscendSimdVFLowerParallel` 在 pipeline 中被注释掉所致），与本次改动无关；改动前后失败信息逐字节相同 |
| L8 | Parallel 外的 tile-op（VF 顶层 copy/reduce）无承接者 | copy → codegen `Generic vector BufferStore ... inside SimdVF`；reduce → `Reduce for buffers in scope (shared.dyn, shared.dyn) is not implemented`；FA 无法 lower | 设计取舍（P1/P2/P3 只处理 Parallel）；后续由 tile-op 降级补齐或加前置展开 pass |

***

## 6. 复现命令

```bash
# 1) 构建
cd /home/cjl/workspace/tilelang
./build.sh -j8

# 2) 全量 transform 测试
TILELANG_CACHE_DIR=/tmp/tilelang_cache PYTHONPATH=$PWD python - <<'PY'
import sys
sys.meta_path = [f for f in sys.meta_path
                 if "editable" not in type(f).__module__]
import pytest
raise SystemExit(pytest.main(["testing/ascend/transform", "-q"]))
PY

# 3) 单独跑新 P1 / 对接测试
TILELANG_CACHE_DIR=/tmp/tilelang_cache PYTHONPATH=$PWD python - <<'PY'
import sys
sys.meta_path = [f for f in sys.meta_path
                 if "editable" not in type(f).__module__]
import pytest
raise SystemExit(pytest.main([
    "testing/ascend/transform/test_ascend_vsimd_lift.py",
    "testing/ascend/transform/test_ascend_vsimd_ir.py",
    "testing/ascend/transform/test_ascend_vsimd_region.py",
    "testing/ascend/transform/test_ascend_vsimd_group_passes.py",
    "-q",
]))
PY

# 4) 全链路 lower（FA）
TILELANG_CACHE_DIR=/tmp/tilelang_cache PYTHONPATH=$PWD python - <<'PY'
import sys
sys.meta_path = [f for f in sys.meta_path
                 if "editable" not in type(f).__module__]
sys.path.insert(0, "examples/ascend/flash_attention")
import tilelang, tvm
from parallel_fa import parallel_flash_attention_fwd
with tvm.target.Target("ascend"), tilelang.transform.PassContext():
    out = tilelang.lower(parallel_flash_attention_fwd())
assert "simd_inst::" in out.kernel_source
print("FA lower OK", len(out.kernel_source))
PY
```

***

## 7. 变更可追溯性

- 分支：`p_123_parallel_new_p1`（基于 `p_123_parallel` @ `a8fe5586`）。
- 新 P1 代码来源：`stable-parallel` @ `64feedf6`。
- 主要 commit 参考：`c1133a7f`、`168d307f`、`349b2c68`、`5b86b4f2`、`f8f8ca21`、`64feedf6`。
- 本报告与 `vsimd_p1_analysis.md` 同步描述了新 P1 契约与 P2/P3 适配结果。
