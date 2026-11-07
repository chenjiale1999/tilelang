# Ascend `T.Parallel` 三阶段（P1→P2→P3）实现报告

> **对齐对象（唯一事实来源，均对应当前工作树）**
>
> - 分支：`p_123_parallel`
> - P1：`src/ascend/transform/lift_vsimd.cc`、`vsimd_program.{h,cc}`、`vsimd_handoff.{h,cc}`
> - P1 分析库：`vsimd_memory_legality.{h,cc}`、`vsimd_capture_scope.{h,cc}`、`vsimd_region.{h,cc}`、`vsimd_analysis_common.h`
> - P2/P3：`src/ascend/transform/polyhedral/`（23 个文件）
> - 运行时兜底：`src/ascend/transform/ascend_simdvf_lower_parallel.cc`
> - 流水线接入：`tilelang/ascend/pipeline.py:120-128`
> - 测试：`testing/ascend/transform/test_ascend_vsimd_group_passes.py`（11 例）等 6 个文件
> - 关联文档：`p2_p3_vsimd_program_integration_design.md`（设计与状态）、`vsimd_p1_analysis.md`（P1 详解）

本文是**实现报告**，只描述“代码现在实际做了什么”，并在第 11 节列出与旧
`array_region` 实现的变化。旧调研文档 `p2_p3_cpp_port_survey.md` 中关于
「`tl.vsimd.array_region` 载体」「`P2` 反展开 region」「legacy 被注释、无兜底」的表述
均已失效，以本文为准。

---

## 0. 结论速览

1. **三阶段全部是 C++**，并且是流水线中 `T.Parallel` 的**唯一向量化路径**：
   `pipeline.py:125-127` 在同一个开关下执行 `AscendLiftVSimd` → `AscendVectorizeVSimd`（P2+P3）；
   随后 `AscendSimdVFLowerParallel` 作为 legacy 兜底，只处理被拒的 nest。
2. **载体从 `AttrStmt` 换成 SBlock annotation**：
   P1 写 `tl.vsimd.program`，P2 写 `tl.vsimd.group_plan`；**两者都不改 body**，
   因此 P1 记录的 item span 在 P2 规划期间始终有效，失败时也不需要“解载体”。
3. **P1 是 metadata-only**：按 `SIMD_VF` body 的顶层语句稠密编号成 item，
   给出 `REGION/COPY/REDUCE/BIND/ALLOC/DECL/OPAQUE` 七类事实、依赖组、
   captures、fragments、assumptions，并写入 `item_count` 与 `body_fingerprint`。
4. **P2 是 group 事务**：只处理 `P1_LIFTABLE` group；任一 item 失败整组 `REJECTED`，
   **绝不发布部分 item 计划**。规划输入是**原始 span 语句**，region 只作为布局/假设上下文。
5. **COPY/REDUCE Phase B 已落地**：COPY 合成 UB↔UB 的 `T.Parallel` copy nest；
   REDUCE 合成源域 `T.Parallel` 归约 nest（`clear=true` 额外发射单位元 init），
   两者都复用现有 `BuildPlan`/`EmitPlan`。
6. **fragment 采用 VF 原子策略**：group 内用到 `OUTSIDE_VF` fragment → 该 group 直接拒绝；
   其余 fragment 由 `PlanVfFragmentRegisters`（V1 escape / V2 预算 / V3 一致性 / V4 生命周期）
   做 VF 级判定，通过后 P2 写 `fragment_regs`，P3 生成 `local.var` 寄存器文件。
7. **旧路径已删除**：`stage_passes.{h,cc}`（`LiftParallelToArray` / `PlanArray` /
   `LowerArrayPlan` / `VSimdRegionRef` / `array_region` / `array_plan`）及其 FFI、
   Python 包装全部移除；`TL_ASCEND_VSIMD_VECTORIZE` 开关一并删除。
8. **回归结果**：`testing/ascend/transform` **593 例**、`testing/ascend/auto_schedule` 639 例，
   合计 **1232 passed / 31 skipped**；`testing/ascend/target` 19 + `cache` 8 + `analysis` 22
   = 49 passed（总计 **1281 passed / 31 skipped**）。
9. **诊断可见性**：`tl.vsimd.group_plan` 注册了 TVMScript printer，`print(mod)` 会像
   `tl.vsimd.program` 一样把它内联展开（含 `status/reason/diagnostics/items`）；REJECTED
   group 的 `reason` 以 `item N:` 标明失败 item。
10. **能力边界（重要）**：P1 的 `NestCollector` 默认拒绝 `T.Parallel` 体内的
    `T.serial` / `if` 语句，因此 P2 的 pre/main/post、多 segment、serial 吸收、语句级
    `if` 谓词化在标准管线**不可达**，只有离线 `build_plan` 覆盖；详见 §12。
11. **`T.Parallel` 体内 tile-op（`T.copy` / `T.reduce_*`）已有 P2/P3 实现（离线可达）**：
    `ExpandParallelTileOps`（`body.cc:547`）先把 `T.reduce_*` 的前端 fragment staging
    折成 shared→shared 归约，再把 tile-op 调用展开成显式循环；`SplitAllSegmentSources`
    （`plan.cc:2408`）→ `PlanSources`（2675）按语句分段规划并保持源序；P3 用「并行轴三段式」发射，归约退化为
    **寄存器累加器**（每输出向量 1 次 `vld` + 1 次 `vsts`）。见 §4.8 / §5.7。
12. **归约的寄存器化**：归约维被放成**最内层 serial 循环**，因此目标 RMW 命中
    `AccumulatorCandidate` 判据被提升为 `local.var` 寄存器；`clear=true` 的单位元作为
    同段 pre-store 折进累加器初值（`vdup`），不再写 shared。

---

## 1. 代码规模与文件清单

### 1.1 P1 与交接层

| 文件 | 行数 | 职责 | 关键符号 |
| --- | ---: | --- | --- |
| `lift_vsimd.cc` | 1039 | P1 pass：VF 遍历、item 构造、group 划分、annotation 写入与自校验 | `BuildProgram`（728）、`BuildGroups`（635）、`LiftMutator`（973）、FFI `AscendLiftVSimd`（1033） |
| `vsimd_program.h` | 99 | P1 program schema（typed）与 reader 声明 | `VSimdProgram`、`VSimdProgramItem/Group/Capture/Fragment` |
| `vsimd_program.cc` | 326 | annotation 解析与不变量校验 | `ReadVSimdProgram`（102）、`VerifyVSimdProgram`（214） |
| `vsimd_handoff.h/.cc` | 53 + 129 | P1/P2/P3 共用的展平、指纹、span 替换 | `FlattenVSimdBody`（`.cc:115`）、`FingerprintVSimdBody`（104）、`ReplaceVSimdSpans`（122） |
| `vsimd_memory_legality.{h,cc}` | 48 + 1252 | PR-A：per-nest 内存合法性 | `AnalyzeVSimdMemoryStatuses` |
| `vsimd_capture_scope.{h,cc}` | 104 + 614 | PR-B：capture/scope 闭包 | `AnalyzeVSimdCaptureInfo` |
| `vsimd_region.{h,cc}` | 133 + 416 | PR-C/D：最小 region 构造与自校验 | `BuildMinimalRegion`、`VerifyExpandedRegion` |
| `vsimd_analysis_common.h` | 312 | VF/nest 索引与公共类型 | `VSimdCaptureInfo`、`VSimdMemoryRegionStatus` |

### 1.2 P2/P3（`src/ascend/transform/polyhedral/`）

| 文件 | 行数 | 职责 | 关键符号 |
| --- | ---: | --- | --- |
| `emit.cc` | 2737 | P3 发射：value/store/归约/pack2/NZ/寄存器文件/多段 | `Emitter`、`EmitStore`、`EmitElementwiseChunks`、`EmitReduce*`、`EmitSegmentStmts`、`EmitParallelAxisParts`、`FragmentRegisterFile`、`EmitPlan` |
| `plan.cc` | 2827 | P2 规划：nest 切分、轴打分、归约、布局、pack2、多段 / tile-op 分段 | `SegmentBody`、`SplitAllSegmentSources`、`PlanSources`、`AssembleProgram`、`SerialIsClean`、`BuildPlanForAxis`、`ClassifyReduce`、`ClassifyStoreLayout`、`ClassifyFragmentRegisters`、`BuildPlan` |
| `value.cc` | 1032 | value DAG：标量表达式 → 逐 lane 向量值 | `BuildValue`、`LoadNode`、`IndexValue`、`GuardFreeBounds`、`ValueRecipeFromName` |
| `primitives.cc` | 419 | ISA 事实表、`Unsupported`、dtype/掩码工具 | `ReducePrimitives`、`VRegLanes`、`MaskBits`、`ReduceIdentity` |
| `vsimd_group_passes.cc` | 873 | **P2/P3 driver**：group 规划、COPY/REDUCE Phase B、fragment 判定、span 替换 | 见 §4/§5 |
| `fragment_plan.cc` | 291 | VF 级 fragment register 判定（V1–V4） | `PlanVfFragmentRegisters`（172） |
| `group_plan.cc` | 324 | typed group plan 构造器、反射注册、IR 内联打印 | `MakeVSimdGroupPlanSet` 等 |
| `access.cc` | 277 | SCoP 建模：线性化、有限差分、访存分类 | `ClassifyAccess`、`FiniteDifference`、`ClassifyDelta` |
| `model.cc` | 209 | 计划数据结构构造器 + 反射 + 累加器判据 | `MakeStorePlan`、`AccumulatorCandidate` |
| `body.cc` | 665 | body 归一化 + **`T.Parallel` 体内 tile-op 展开** | `ExpandParallelTileOps`（547）、`ReduceStagingFuser`（443）、`TileOpExpander`（343）、`BuildCopyNestFromRegions`（138）、`BuildReduceNestFromRegions`（230）、`NormalizeBodyInto`（561） |
| `analyze_polyhedral.cc` | 298 | **离线唯一入口**：FFI（`build_plan`/`emit_plan`/`render_plan`）+ plan 文本渲染 | `AscendPolyhedralBuildPlan`（247）等、`RenderPlanText`（198） |
| `vsimd_payload_read.cc` | 405 | **P2 专用** payload 解码：REGION/COPY/REDUCE → typed 输入 | `ReadVSimdRegion`（135）、`ReadVSimdCopy`（361）、`ReadVSimdReduce`（377）、`VSimdCopyReducePayload`、`VSimdReducePayload` |
| 头文件 | 1771 | `group_plan.h` 257、`model.h` 365、`primitives.h` 220、`access.h` 220、`value.h` 308、`body.h` 106、`vsimd_payload_read.h` 72、`emit.h` 72、`plan.h` 66、`fragment_plan.h` 54、`vsimd_group_passes.h` 31 | — |

> 规模集中度：`emit.cc` + `plan.cc` + `value.cc` 共 6596 行，占 P2/P3 库（12128 行）的约 54%。

### 1.3 运行时兜底

| 文件 | 行数 | 说明 |
| --- | ---: | --- |
| `ascend_simdvf_lower_parallel.cc` | 999 | `AscendSimdVFLowerParallel`：P2/P3 拒绝的 `T.Parallel` / `T.copy` / `T.reduce_*` 由它按原语义降到 SIMD 指令 |

### 1.4 测试规模

| 文件 | 用例数 | 覆盖 |
| --- | ---: | --- |
| `test_ascend_polyhedral_vectorize.py` | 168 def / 169 例（4905 行） | 离线 `build_plan`/`emit_plan` 全场景 + 新链路 + fragment + FA + **`T.Parallel` 体内 tile-op 8 例** |
| `test_ascend_vsimd_group_passes.py` | 11（370 行） | P1→P2→P3 契约、COPY/REDUCE Phase B、fragment、默认流水线 |
| `test_ascend_vsimd_lift.py` | 15 | P1 annotation 内容与放行场景 |
| `test_ascend_vsimd_memory_legality.py` | 23 | PR-A |
| `test_ascend_vsimd_capture_scope.py` | 15 | PR-B |
| `test_ascend_vsimd_region.py` | 17 | PR-C/D |
| **`testing/ascend/transform` 合计** | **593** | — |

---

## 2. 架构与数据流

### 2.1 模块依赖

```mermaid
flowchart TD
  P["primitives.h/.cc<br/>ISA 事实表 + Unsupported"] --> A["access.h/.cc<br/>Axis/AccessInfo/分类"]
  A --> B["body.h/.cc<br/>StoreSpec/归一化"]
  A --> V["value.h/.cc<br/>ValueNode/守卫/索引向量"]
  A --> M["model.h/.cc<br/>StorePlan/VectorPlan/Program"]
  B --> M
  V --> M
  M --> PL["plan.h/.cc<br/>BuildPlan"]
  M --> E["emit.h/.cc<br/>EmitPlan + FragmentRegisterFile"]
  A --> FP["group_plan.h/.cc<br/>typed P2→P3 载体"]
  PL --> GP["vsimd_group_passes.h/.cc<br/>P2/P3 driver"]
  E --> GP
  FP --> GP
  M --> FPL["fragment_plan.h/.cc<br/>V1–V4"]
  FPL --> GP
  H["../vsimd_handoff.h/.cc<br/>展平/指纹/span 替换"] --> GP
  H --> LV["../lift_vsimd.cc<br/>P1"]
  PR["../vsimd_program.h/.cc<br/>program schema + verifier"] --> GP
  PR --> LV
  PRD["vsimd_payload_read.h/.cc<br/>REGION/COPY/REDUCE 解码"] --> GP
  P1A["../vsimd_memory_legality<br/>../vsimd_capture_scope<br/>../vsimd_region"] --> LV
  GP --> FFI["analyze_polyhedral.cc<br/>离线 FFI"]
```

要点：

- polyhedral 内部依赖仍严格向下（`primitives → access → {body,value} → model → plan/emit`）。
- **P1 交接层不再住在 polyhedral 里**：`vsimd_handoff`、`vsimd_program` 在
  `src/ascend/transform/` 下，polyhedral 通过 `../vsimd_*` 单向引用。
  `vsimd_program.*` 只保留 P1 的 schema 与结构校验；REGION/COPY/REDUCE 的
  payload 解码在 `polyhedral/vsimd_payload_read.*`，由 P2 消费。
- 旧的 region 反展开交接（`vsimd_p2_input.*`）与 `render.*` 已移除：离线入口
  全部收敛到 `analyze_polyhedral.cc`（含 `RenderPlanText`）。
- `group_plan` 是 P2 与 P3 之间唯一的类型化载体，P3 不需要知道 P1 annotation 的结构。

### 2.2 阶段契约

| 阶段 | 入口 | 输入 | 输出 | 载体 | 失败语义 |
| --- | --- | --- | --- | --- | --- |
| P1 | `AscendLiftVSimd` | `PrimFunc` | 同一 `PrimFunc`，body 不变 | `SBlock("SIMD_VF").annotations["tl.vsimd.program"]` | 不抛异常；无 `T.Parallel` 的 VF 不写 annotation；结构自校验失败写 `validation_error` |
| P2 | `AscendPlanVSimd` | 带 `tl.vsimd.program` 的 `PrimFunc` | 同一 body，新增 group plan | `...annotations["tl.vsimd.group_plan"]`（typed `VSimdGroupPlanSet`） | 整组 `REJECTED`；不写部分 item plan；body 不变 |
| P3 | `AscendEmitVSimd` | 带 `tl.vsimd.group_plan` 的 `PrimFunc` | item span 被替换为 `T.simd.*` | 直接改 body；删除两个 annotation | 整组回滚（保留原语句）；其它 group 不受影响 |
| 组合 | `AscendVectorizeVSimd` | `PrimFunc` | `PrimFunc` | — | = P2 + P3 |
| 兜底 | `AscendSimdVFLowerParallel` | 残留的 `T.Parallel` / `T.copy` / `T.reduce_*` | SIMD 指令 | — | 原有语义 |

### 2.3 载体协议

| 载体 | key | 载荷 | body | 生产者 | 消费者 |
| --- | --- | --- | --- | --- | --- |
| program | `tl.vsimd.program`（`lift_vsimd.cc:47`） | `Map<String,Any>`（`version/vf_index/provenance/item_count/body_fingerprint/items/groups/captures/fragments/assumptions/attrs`） | 原 VF body | P1 | P2；P3 在 P2 未规划时负责删除 |
| group plan | `tl.vsimd.group_plan`（`vsimd_group_passes.cc:54`） | `VSimdGroupPlanSet`（typed FFI 对象） | 原 VF body | P2 | P3 |

> 两个 annotation 最终都不会进入 codegen：P3 无条件删除。
>
> `tl.vsimd.group_plan` 的 typed 对象注册了 `IRDocsifier` printer，`print(mod)` 会像
> `tl.vsimd.program` 一样把它内联展开成普通 dict（`version/vf_index/groups/...`、
> group 的 `status/reason/diagnostics/items`），不再落到
> `metadata["tl.polyhedral.VSimdGroupPlanSet"][0]` 占位符。该打印只是诊断视图，
> P3 仍消费 typed 对象，载体协议不变。

### 2.4 共享展平规则（`vsimd_handoff.cc`）

`FlattenVSimdBody` 是 P1/P2/P3 唯一的“item 定义”：

- `SeqStmt` 递归展开；
- 透明 wrapper **不占 item**：`tl.simdvf_scope`、`tilelang_assume` / `tl.assume`、
  `tl.buffer_version`、`tl.manual_multi_buffer`（`.cc:20-24`）；
- 其余顶层语句各占一个 item（`AllocBuffer`、`DeclBuffer`、`Bind`、`For`、`Evaluate`…）。

`FingerprintVSimdBody` 对展平后的语句序列做结构指纹；P1 把 `item_count` 与
`body_fingerprint` 写进 annotation。**P2 在规划前重新展平并逐项比对**：不一致就不写
group plan（P3 随后只删 annotation，nest 交 legacy）。P3 自己不再比对——它只消费
P2 在一致性校验之后写出的 group plan，并按 span 替换；因此 P2 → P3 之间同样不允许
插入改 body 的 pass。这保证了跨 pass 组合时不会静默错位替换。

---

## 3. P1：`AscendLiftVSimd`

### 3.1 入口与契约

- FFI：`tl.transform.AscendLiftVSimd`（`lift_vsimd.cc:1033`）。
- Python：`tilelang.ascend.transform.AscendLiftVSimd`。
- 输入/输出都是同一个 `PrimFunc`；**P1 只读 body**，唯一副作用是给
  `SBlock("SIMD_VF")` 加 annotation（`lift_vsimd.cc:984`）。
- `TL_ASCEND_SIMD_LIFT`（默认 `True`）控制是否运行；关掉即整条新链路停用，
  `T.Parallel` 全部交给 legacy。

### 3.2 处理流程

```mermaid
flowchart TD
  A["LiftMutator 遍历 PrimFunc"] --> B["遇到 SBlock(SIMD_VF)，分配 vf_index"]
  B --> C["ProgramInputs：<br/>PR-A AnalyzeVSimdMemoryStatuses<br/>PR-B AnalyzeVSimdCaptureInfo<br/>fragment 定义/外部使用收集"]
  C --> D["BuildProgram(const SBlock &vf, int vf_index, const ProgramInputs &inputs))<br/>顶层 item 稠密编号 span=0..n-1"]
  D --> E{"逐 item 分类"}
  E -->|"For(Parallel)"| F["REGION：capture info + BuildMinimalRegion<br/>lifted ? LIFTED : OPAQUE"]
  E -->|"Evaluate(Call op 名含 copy)"| G["COPY：TYPED"]
  E -->|"Evaluate(Call op 名含 reduce)"| H["REDUCE：TYPED"]
  E -->|"Bind"| I["BIND：PRESERVE"]
  E -->|"AllocBuffer / DeclBuffer"| J["ALLOC / DECL：PRESERVE"]
  E -->|"其它（if/SBlock/AttrStmt/While/...）"| K["OPAQUE"]
  F --> L["BuildGroups：DSU"]
  G --> L
  H --> L
  I --> L
  J --> L
  K --> L
  L --> M{"VF 内有 Parallel ?"}
  M -->|"否"| N["不写 annotation"]
  M -->|"是"| O["写 tl.vsimd.program<br/>+ ReadVSimdProgram 自校验"]
  O -->|"校验失败"| P["加 validation_error"]
```

### 3.3 item 分类

| 顶层语句 | kind | analysis_status | p1_liftable |
| --- | --- | --- | --- |
| `For(kind=Parallel)` 且 `nest_index` 有 capture info、`BuildMinimalRegion.lifted` | `REGION` | `LIFTED` | true |
| `For(kind=Parallel)` 但未 lift | `REGION` | `OPAQUE` | false |
| `Evaluate(Call)`，op 名含 `copy` | `COPY` | `TYPED` | true |
| `Evaluate(Call)`，op 名含 `reduce` | `REDUCE` | `TYPED` | true |
| `Bind` | `BIND` | `PRESERVE` | false |
| `AllocBuffer` / `DeclBuffer` | `ALLOC` / `DECL` | `PRESERVE` | false |
| 其它（`if` / 嵌套 `SBlock` / `AttrStmt` / `While` / 未识别 `Evaluate`） | `OPAQUE` | `OPAQUE` | false |

- COPY 的识别是 op 名子串匹配（`IsCopyOp`，`lift_vsimd.cc:628`），因此 P2 必须重新
  校验 `src_region`/`dst_region`/scope（见 §4.4）。
- `REDUCE clear=false` 时 P1 把 dst 同时记入 `reads`，依赖组因此能连上 dst 的 producer
  （`AddCopyReduceItem`，`lift_vsimd.cc:546`）。

### 3.4 group 规则（`BuildGroups`，`lift_vsimd.cc:635`）

- 用并查集把 item 合成 group，边有三类：
  1. **读后写**：item 读某 buffer 且此前有 writer；
  2. **fragment 使用**：所有读写同一 fragment 的 item 强制同组（fragment 是跨语句值）；
  3. 其余无依赖的 item 保持独立 group。
- group 状态：
  - 只含 metadata（BIND/ALLOC/DECL）→ `PRESERVE`；
  - 混入 metadata 或任一 item 不可 lift → `OPAQUE`；
  - 全部 item 可 lift 且无 metadata → `P1_LIFTABLE`。
- item 的 `status`/`effective_status` 由 group 状态覆盖：
  `PRESERVE` / `OPAQUE` / 否则等于 `analysis_status`（`LIFTED` 或 `TYPED`）。

> `P1_LIFTABLE` 只表示“允许 P2 尝试”，不是发射许可。

### 3.5 annotation 字段

| 字段 | 说明 |
| --- | --- |
| `version` | 固定 1 |
| `vf_index` | 按 VF 出现顺序编号 |
| `provenance` | 固定 `LIFTED` |
| `item_count` | 展平后顶层 item 数 |
| `body_fingerprint` | 展平语句序列的结构指纹 |
| `items[]` | 公共字段 + `kind` 专属 payload（`region`、`src/dst/src_region/dst_region/reduce_kind/reduce_dim/clear`、`var/value`、`buffer/storage`） |
| `groups[]` | `{id, status, items}` |
| `captures[]` | `{buffer, name, storage}` |
| `fragments[]` | `{buffer, name, definition_vf, definition_scope, use_vfs, cross_vf, escapes_vf, uses_outside_vf}` |
| `assumptions[]` / `attrs[]` | 锚点（`anchor`/`anchor_end`）与 `T.assume` 条件 |
| `validation_error` | 自校验失败时出现，P2 见到即整 VF 拒绝 |

### 3.6 边界与不变量

1. **无 `T.Parallel` 的 VF 不写 annotation**（`lift_vsimd.cc:834`）：例如只有
   `T.copy` / `T.reduce_*` 的 VF，P1 不产出 program，legacy 直接处理。
2. 只有**嵌套** Parallel（无顶层 item）时会写一个全 `OPAQUE` 的 program 并 `LOG(WARNING)`
   （`lift_vsimd.cc:838`）；P2 不会规划它。
3. P1 不修改 body，因此可以重复运行：重复 lift 只覆盖同一 annotation。
4. P1 的 `ReadVSimdProgram` 自校验会把结构不一致写进 `validation_error`，而不是抛异常。

---

## 4. P2：`AscendPlanVSimd`

### 4.1 入口与 group 事务

- FFI：`tl.transform.AscendPlanVSimd`（`vsimd_group_passes.cc:860`）→ `MakeGroupPass`（830）→ `PlanVSimd`（839）。
- 只读 `tl.vsimd.program`，写 `tl.vsimd.group_plan`；**body 一个字节都不改**。
- 事务粒度 = P1 的 group：
  - `P1_LIFTABLE` → 尝试规划；
  - 任一 item 失败 → 整组 `REJECTED`，`items` 为空数组；
  - `PRESERVE` / `OPAQUE` group 不产生 entry。

**入口与函数调用关系**

```mermaid
flowchart TD
  A0["FFI tl.transform.AscendPlanVSimd<br/>vsimd_group_passes.cc:860"] --> A1["MakeGroupPass<br/>:830"]
  A1 --> A2["PlanVSimd<br/>:839"]
  A2 --> A3["GroupPlanner::VisitStmt_<br/>SBlock「SIMD_VF」:622"]

  A3 -->|"无 program / 解码失败 / validation_error"| Z["原样返回<br/>（P3 只删 P1 annotation）"]
  A3 --> E1["ReadVSimdProgram<br/>vsimd_program.cc:102"]
  A3 --> E2["FlattenVSimdBody + item_count / fingerprint 校验<br/>vsimd_handoff.cc:115"]
  A3 --> E3["CollectProgramLayouts<br/>:95"]
  A3 --> P0["逐 P1_LIFTABLE group：PlanOneGroup<br/>:532"]

  P0 --> R1["GroupUsesOutsideFragment<br/>:499"]
  P0 --> R2["PlanRegionItem :124"]
  P0 --> R3["PlanCopyItem :223"]
  P0 --> R4["PlanReduceItem :429"]

  R2 --> D1["ReadVSimdRegion<br/>vsimd_payload_read.cc:135"]
  R3 --> D2["ReadVSimdCopy :361<br/>BuildCopyNest :181"]
  R4 --> D3["ReadVSimdReduce :377<br/>MapReduceDims :299<br/>BuildReduceNest :351<br/>BuildReduceInitNest :404"]

  R2 --> BP["BuildPlan<br/>plan.cc:2774"]
  R3 --> BP
  R4 --> BP

  P0 --> OK["MakeVSimdGroupPlan（PLANNED）<br/>MakeRejectedGroupPlan（REJECTED）"]
  A3 --> FR["CollectPrograms :597<br/>PlanVfFragmentRegisters<br/>fragment_plan.cc:172"]
  A3 --> OUT["MakeVSimdGroupPlanSet :687<br/>annotations[tl.vsimd.group_plan]"]
```

> 图注：`:NNN` 为 `vsimd_group_passes.cc` 的行号；跨文件符号已标注文件名。

### 4.2 主流程

```mermaid
flowchart TD
  A["Visit SBlock(SIMD_VF)"] --> B{"有 tl.vsimd.program ?"}
  B -->|"否"| Z["原样返回"]
  B -->|"是"| C["ReadVSimdProgram"]
  C -->|"失败 / validation_error"| Z
  C --> D["FlattenVSimdBody + 校验 item_count / fingerprint"]
  D -->|"不一致"| Z
  D --> E["CollectProgramLayouts：<br/>合并本 VF 所有 LIFTED REGION 的 layouts"]
  E --> F["逐 P1_LIFTABLE group：PlanOneGroup"]
  F --> G["收集 PLANNED 的 VectorProgram + span"]
  G --> H["PlanVfFragmentRegisters（V1–V4）"]
  H -->|"enabled"| I["plan_set.fragment_regs = map<br/>mode = register"]
  H -->|"失败且有原因"| J["所有 PLANNED group 改判 REJECTED<br/>mode = rejected"]
  I --> K["写 tl.vsimd.group_plan"]
  J --> K
```

**规划内核调用关系（`BuildPlan`）**

```mermaid
flowchart TD
  BP["BuildPlan :2774"] --> EX["ExpandParallelTileOps<br/>body.cc:547"]
  EX --> EX1["ReduceStagingFuser :443 → TileOpExpander :343"]
  EX1 --> EX2["BuildCopyNestFromRegions :138<br/>BuildReduceNestFromRegions :230"]

  BP --> SB["SegmentBody :2641"]
  SB -->|"tile-op 展开体"| S1["SplitAllSegmentSources :2408"]
  SB -->|"相邻 clean serial 链"| S2["ChainSegmentSources :2335"]
  SB -->|"其余形态"| S3["SingleSegmentSource :329"]
  S1 --> S4["AbsorbParallelChain :188<br/>AbsorbSerialChain :398<br/>CollectAliases"]
  S2 --> S4

  BP --> PS["PlanSources :2675"]
  PS --> PS1["PlanSegment :2225（每段一次）"]
  PS1 --> PS2["PrepareSegment :947<br/>TryFlatten :711<br/>RankVectorAxes :473<br/>AxisDiagnostics :499"]
  PS1 --> PS3["BuildPlanForAxis :1900"]
  PS3 --> PS4["ClassifyStoreLayout :1578<br/>ClassifyFragmentRegisters :1722<br/>ClassifyReduce :1264"]
  PS3 --> PS5["FoldIdentityInit :993<br/>DropDeadPreStores :1032<br/>MarkStoreRoles :1097<br/>MarkLiveChunkRange :1157"]

  BP --> AS["AssembleProgram :2701<br/>MakeVectorProgram(hoisted, segments)"]
```

- `PlanSources` 是唯一的逐段规划入口：单段时把 segmenter 的 notes 作为 base notes，
  失败即最终；多段时收集 `segment <i>: ...` 后重抛。
- `AssembleProgram` 是唯一的装配入口：三段 hoist 规则与三种诊断文案都在这里，
  `BuildPlan` 不再分别处理「单段 / 多段 / tile-op」。

### 4.3 REGION 规划（`PlanRegionItem`，124）

- 前置：`item.status == LIFTED`、span 为单条语句。
- 解码：`ReadVSimdRegion(item.meta, vf_index, &region)` 取
  `layouts / assumptions / bindings / shape_syms / domain / accesses`。
- `PlanOptions`：
  ```text
  layouts        = region.layouts        // NZ 目标由 ClassifyStoreLayout 处理
  assumptions    = region.assumptions    // 进入 Analyzer 约束
  bindings       = region.bindings       // 进入 Analyzer Bind
  shape_syms     = region.shape_syms
  vf_alloc_scope = vf->alloc_buffers     // fragment G0 校验
  ```
- **规划输入是原始 span 语句** `body.stmts[item.span_begin]`，而不是 region 反展开：
  保留 `Bind`、别名、前后语句，`BuildPlan` 的 serial 吸收 / pre-post 能力仍然可用。
- 若该语句是含 `tl.tileop.*` 的 `For(Parallel)`，`BuildPlan` 内部先走 §4.8 的
  展开 + 分段路径（当前 P1 不放行，故生产不可达）。

### 4.4 COPY Phase B（`PlanCopyItem`，223）

**payload 解码**（`ReadVSimdCopy`）：`src/dst`（buffer + storage scope）与
`src_region/dst_region`（`buffer/indices/extents/access_type`）。缺字段即整组拒绝。

**支持范围**：

| 约束 | 说明 |
| --- | --- |
| 两端 scope | 只接受 UB：`shared` / `shared.dyn`（`IsUbScope`，70） |
| rank | src/dst 必须同 rank、非 0 |
| extent | 逐维 `src == dst`，或 `src extent == 1`（广播） |
| layout | 取自同 VF 的 REGION item（`CollectProgramLayouts`） |

**合成规则**（`BuildCopyNest`，181）：

```text
for idx in T.Parallel(dst_extents):
    dst[dst_indices + idx] = src[src_indices + (extent==1 ? 0 : idx)]
```

然后 `BuildPlan(nest, nullopt, options)`。`VSimdCopyPlan.kind` 记录 `ND` 或 `BROADCAST`。
跨 scope（GM / FRAGMENT / L1）不做，整组拒绝并交 legacy。

### 4.5 REDUCE Phase B（`PlanReduceItem`，429）

**payload 解码**（`ReadVSimdReduce`）：在 COPY 的字段上增加
`reduce_kind / reduce_dim / clear`。

**支持范围**：

| 项 | 支持 |
| --- | --- |
| `reduce_kind` | `max` / `sum` / `min`（`ReduceRecipeOf` → `vcmax` / `vcadd` / `vcmin`） |
| `dim` | `[0, src_rank)` |
| dst 形状 | 去掉归约维（rank-1），或保留该维且 extent 为 1（`MapReduceDims`） |
| 其它维 | `src extent == dst extent`，或 src extent 为 1（广播） |
| scope | 两端都必须 UB |

**合成规则**（`BuildReduceNest`，352）——源域 Parallel 归约：

```text
for idx in T.Parallel(src_extents):
    dst[dst_indices + idx'] = op(dst[dst_indices + idx'], src[src_indices + idx])
```

- 该形状正是 `ClassifyReduce` 识别的“lane 变化读 + lane 均匀写”，可直接发射
  `vcadd/vcmax/vcmin`；轴打分也可能选输出轴做向量轴、把归约维留成串行循环，
  两种都是合法发射（由规划器决定）。
- `clear=false`：上式直接累加旧 dst 值。
- `clear=true`：额外生成一个 `init` program（`BuildReduceInitNest`，405）：
  ```text
  for idx in T.Parallel(dst_extents):
      dst[dst_indices + idx] = identity(dtype, recipe)
  ```
  单位元由 `ReduceIdentity` 按 dtype 给出（max→-inf、min→+inf、sum→0），
  避免把“旧 dst 值”当初始值。P3 先发射 `init` 再发射归约。

`VSimdReducePlan` 记录 `reduce_kind / dim / clear / init? / program`。

### 4.6 fragment 策略

**第一层：group 级（`GroupUsesOutsideFragment`，499）**

P1 的 `fragments[]` 中 `definition_scope == OUTSIDE_VF` 且被本 group 任一 item
读写的 fragment，直接让该 group `REJECTED`：

```text
fragment 'x' is allocated outside this VF (cross-VF fragment); stage it through UB first
```

> 约定：fragment 在 VF 外定义、VF 内使用 = 跨 VF fragment，P2 直接拒绝；
> “一个 VF 定义、另一个 VF 使用”的形态不会出现，未做任何设计。

**第二层：VF 级（`PlanVfFragmentRegisters`，`fragment_plan.cc:172`）**

对**已 PLANNED group 的全部 VectorProgram** 做一次 VF 原子判定：

| 检查 | 内容 | 失败后果 |
| --- | --- | --- |
| V1 escape | 用 `FlattenVSimdBody` 扫描**未被规划**的 item，若访问了 register-managed fragment | 整 VF 禁用 register mode |
| V2 预算 | Σ slots ≤ `kManualVRegPool`(24)，且 + auto reserve ≤ `kMaxVRegsPerVf`(32) | 整 VF 拒绝 |
| V3 一致性 | 同一 fragment data Var 的 `slots/lanes/dtype` 必须一致（同名不同 data Var 是两条） | 整 VF 拒绝 |
| V4 生命周期 | 按 program 顺序模拟 `init[data][slot]`：读未初始化 chunk、掩码部分写未初始化 chunk | 整 VF 拒绝 |

- 通过：`VSimdGroupPlanSet.fragment_regs = FragmentRegisterMap`，
  `fragment_mode = "register"`。
- 失败：所有 `PLANNED` group 改判 `REJECTED`（`fragment_mode = "rejected"` +
  `fragment_reason`），P3 不替换任何语句，整 VF 交 legacy。
- 单条 nest 内部的 G0–G8 仍由 `BuildPlan` 的 `ClassifyFragmentRegisters` 决定
  （例如 fragment 必须由当前 VF 分配、identity layout、非 packed store 等）；
  这些失败表现为“该 group 拒绝”，不是 VF 级。

### 4.7 失败语义与诊断

- `group.status` 只有 `PLANNED` / `REJECTED`。
- `REJECTED` 时：`reason` 记录第一个失败原因，并以 `item N:` 标明失败 item；
  `diagnostics` 逐 item 追加；`items` 为空，**绝不发布部分计划**。
- 典型 `reason`（实际输出带 `item N:` 前缀）：
  - `REGION item is not effectively LIFTED`
  - `invalid region payload: ...`
  - `copy source 'x' is not UB (scope FRAGMENT)`
  - `copy dimension 0 has unequal extents ...`
  - `reduce kind 'absmax' is not supported yet (max/sum/min)`
  - `fragment 'x' is allocated outside this VF (cross-VF fragment); ...`
  - `fragment registers rejected: <V1–V4 原因>`
- P2 不删除 P1 annotation：body/annotation 的一致性由“P2 不改 body”保证。

### 4.8 `T.Parallel` 体内 tile-op 展开路径（`ExpandParallelTileOps`，离线可达）

> 触发条件：`BuildPlan` 的输入 nest 里出现 `tl.tileop.copy` / `tl.tileop.ascend_copy` /
> `tl.tileop.reduce`。标准管线目前走不到（前置 `NestedLoopChecker` 规则 2 与 P1
> `NestCollector` 都会拒绝），只有离线 `build_plan` 覆盖；见 §12.3。

**① 前端 staging 折叠（`ReduceStagingFuser`，`body.cc:443`）**

`T.reduce_*`（shared 入参/出参）被宏展开成 fragment 中转：

```text
clear=true : copy(shared_src, frag_src); reduce(frag_src, frag_dst, true); copy(frag_dst, shared_dst)
clear=false: copy(shared_dst, frag_dst); copy(shared_src, frag_src);
             reduce(frag_src, frag_dst, false); copy(frag_dst, shared_dst)
```

fragment 是 VF 级临时量而非用户数据，该序列被整体替换为一次
`BuildReduceNestFromRegions(shared_src, shared_dst, …)`；未匹配的 staging copy
原样保留，由展开器报 “not UB”。

**② 结构重建（`TileOpExpander` / `BuildCopyNestFromRegions` / `BuildReduceNestFromRegions`）**

展开器**只重建循环结构，不做 Δ 判定、不选向量轴**：

| tile-op | 合成规则 |
| --- | --- |
| `copy`（UB↔UB） | `for idx in Parallel(dst_extents): dst[dst_idx+idx] = src[src_idx + (extent==1 ? 0 : idx)]`；rank 必须相同、extent 相等或广播；**extent=1 的维不建循环** |
| `reduce`（max/sum/min） | 源域循环，**归约维是最内层 `kSerial`**，其余维 `kParallel`；`dim` 必须落在源 rank 内；dst 可去掉该维或保留为 extent=1 |
| `clear=true` | 在归约循环**之外、并行维之内**追加 `dst[...] = identity`（max→-inf、min→+inf、sum→0）作为同址 pre-store |

把归约维放最内层是为了命中累加器提升判据：目标 RMW 的地址沿该内层轴不变，
`AccumulatorCandidate` 于是把整条链提升为一个 `local.var` 向量寄存器，每个输出
向量只需 1 次 `vld` + 1 次 `vsts`，而不是每步一次 shared 读改写。

**③ 分段规划（`SplitAllSegmentSources` → `PlanSources`，`plan.cc:2408` / `2675`）**

```mermaid
flowchart TD
  A["BuildPlan"] --> B["ExpandParallelTileOps"]
  B -->|"expanded = false"| C["SegmentBody：单段 / 相邻 serial 链"]
  B -->|"expanded = true"| D["AbsorbParallelChain<br/>吸收外层 Parallel 前缀"]
  D --> E["SplitAllSegmentSources：按 body 语句分段"]
  E --> F["连续非循环语句 → store 段"]
  E --> G["每个循环链 → loop 段<br/>（含 trailing serial 的 pre/post 拆分）"]
  E --> H["相邻同形、buffer 不相交的纯并行链 → 合并成一段"]
  F --> I["PlanSources：逐段 PlanSegment（保持源序）"]
  G --> I
  H --> I
  I --> J["仅无 serial 轴的段允许 flatten"]
  J --> K["AssembleProgram → MakeVectorProgram(hoisted, segments)"]
```

分段规则要点：

- **语句粒度**：每条语句独立尝试规划，一段失败只报
  `tile-op segment planning failed: <原因>`（仍会拒绝整个 program）；
- **源序**：segment 顺序 = body 语句顺序；`T.Parallel` 各迭代互相独立，
  因此“每段覆盖全部 i”与逐 i 执行等价；
- **identity init 合并**：body 恰为 `{uniform 字面量 store} + 唯一循环链`、且该 store
  与链内某个 store 同 buffer 同下标时，合并为一段并把 store 当 pre-store，
  从而被 `FoldIdentityInit`/emit 折进累加器初值；
- **相邻链合并**：两条相邻、形状完全一致（kind/min/extent）、都不含 serial 循环、
  且触碰 buffer 不相交的链合并成一个 nest（右链循环变量替换为左链），
  使它们的 chunk 循环交错（一次迭代两对独立 `vld/vsts`）；
- **flatten 策略**：无 serial 轴的段才允许 `TryFlatten`；带 serial 轴的段必须保留
  轴表（serial 维就是累加器的内层轴）；
- **hoisted 前缀**：只提升“位于所有段 vector_axis 之前”的外层 Parallel 轴；
  被某段当作 lane 轴使用的轴不会被提升（否则该段会被重复执行）。

---

## 5. P3：`AscendEmitVSimd`

### 5.1 入口

- FFI：`tl.transform.AscendEmitVSimd`（`vsimd_group_passes.cc:863`）→ `MakeGroupPass`（830）→ `EmitVSimd`（846）。
- 消费 typed `tl.vsimd.group_plan`；没有 group plan 但有 P1 annotation 时，只删除
  P1 annotation（P2 跳过的情形）。

**入口与函数调用关系**

```mermaid
flowchart TD
  B0["FFI tl.transform.AscendEmitVSimd<br/>vsimd_group_passes.cc:863"] --> B1["MakeGroupPass<br/>:830"]
  B1 --> B2["EmitVSimd<br/>:846"]
  B2 --> B3["GroupEmitter::VisitStmt_<br/>SBlock「SIMD_VF」:755"]

  B3 -->|"无 group_plan"| C0["删 tl.vsimd.program → 返回"]
  B3 --> C1["FragmentRegisterFile::Build<br/>emit.cc:615"]
  B3 --> C2["FlattenVSimdBody（item 定位）<br/>vsimd_handoff.cc:115"]
  B3 --> C3["逐 PLANNED group：逐 item EmitItem<br/>:736"]
  B3 --> C4["ReplaceVSimdSpans<br/>vsimd_handoff.cc:122"]
  B3 --> C5["RewriteFragmentAllocations<br/>:710"]
  B3 --> C6["删 tl.vsimd.program / tl.vsimd.group_plan"]

  C3 --> D1["EmitItem → EmitPlan<br/>emit.h:65 / emit.cc:2710"]
  C3 -->|"任一 item 抛 Unsupported"| D2["整组回滚（保留原语句）"]
  B3 -->|"寄存器模式且有组回滚"| D3["清空全部替换（整 VF 回退）"]
```

> 图注：`:NNN` 为 `vsimd_group_passes.cc` 的行号；跨文件符号已标注文件名。

### 5.2 主流程

```mermaid
flowchart TD
  A["Visit SBlock(SIMD_VF)"] --> B{"有 tl.vsimd.group_plan ?"}
  B -->|"否"| C["删除残留 tl.vsimd.program"] --> Z["返回"]
  B -->|"是"| D["读取 VSimdGroupPlanSet"]
  D --> E["fragment_regs ?<br/>FragmentRegisterFile::Build"]
  E --> F["FlattenVSimdBody（item 定位）"]
  F --> G["逐 PLANNED group：逐 item 发射"]
  G --> H{"EmitItem 全部成功 ?"}
  H -->|"是"| I["记录 span → emitted Stmt"]
  H -->|"否"| J["整组丢弃（保留原语句）"]
  I --> K{"寄存器模式且发生过失败 ?"}
  J --> K
  K -->|"是"| L["清空全部替换（整 VF 回退）"]
  K -->|"否"| M["ReplaceVSimdSpans"]
  L --> M
  M --> N["改写 alloc_buffers / layout_map（仅寄存器模式且有替换）"]
  N --> O["删除两个 annotation"]
```

**发射内核调用关系（`EmitPlan`）**

```mermaid
flowchart TD
  A["EmitItem :736"] --> B["EmitPlan<br/>emit.h:65 / emit.cc:2710"]
  B -->|"VectorProgram"| C["EmitProgram :2665"]
  B -->|"单个 VectorPlan（离线）"| D["EmitSegmentStmts（single）:2146"]

  C --> C1["CollectFragmentRegs :2645<br/>DeclareRegisters :2658"]
  C --> C2["逐段 EmitSegmentStmts :2146"]
  C --> C3["program.axes 包 serial for"]

  C2 -->|"parallel 向量轴 + 内层轴"| E["EmitParallelAxisParts :2440"]
  C2 -->|"elementwise / reduction"| F["Emitter + SplitPreStores :2084<br/>EmitElementwiseChunks :1848 / EmitReduce :2425<br/>SplitLoopInvariant :1962<br/>EmitHoistedPreStores :2106"]

  E --> E1["EmitStore :1669（pre / main / post）"]
  E --> E2["累加器：vld / vdup 初值 + 单次 vsts 写回"]

  F --> F1["StripMine :1756（chunk 展开 / 尾块）"]
  F1 --> F2["Emitter::Build :1261<br/>recipe 族 switch → SimdCall(SimdOpOf(...))"]

  E1 --> G["EmitVectorWrite :1487"]
  G --> G1["EmitRegisterWrite :1451<br/>Vsts / Vscatter / EmitPackedNzStore :1563"]
```

- `EmitPlan` 是 P3 的唯一起点：线上（`EmitItem`）与离线（`emit_plan`）共用；
  `EmitPlan` 收到 `VectorProgram` 时进入 `EmitProgram`（多段 + hoist），收到单个
  `VectorPlan` 时直接走 `EmitSegmentStmts`。
- `EmitSegmentStmts` 按「parallel 向量轴 + 内层轴」分流到三段式
  `EmitParallelAxisParts`，否则走 elementwise / reduction 的扁平路径；
  两条路径最终都落到 `EmitStore` → `Emitter`。

### 5.3 item 发射与整组回滚

`EmitItem`（736）：

| item | 发射 |
| --- | --- |
| `REGION` | `EmitPlan(region_plan, registers)` |
| `COPY` | `EmitPlan(copy_plan->program, registers)` |
| `REDUCE` | `SeqStmt([EmitPlan(init?, registers), EmitPlan(program, registers)])` |

- 每组的 item 先在内存中全部发射成功，才写入替换表；
- 任一 item 抛 `Unsupported` → `LOG(WARNING)` + 整组回滚，其它组不受影响；
- 若 VF 处于寄存器模式且有组回滚 → 清空**所有**替换（一个回滚组可能仍以内存身份
  使用 fragment，保留寄存器文件会破坏它）。

### 5.4 span 替换

`ReplaceVSimdSpans(body, replacements)`（`vsimd_handoff.cc:122`）：

- 按 `span_begin` 定位第 i 个**规范 item**（与 P1 完全相同的展平规则）；
- 保留透明 wrapper（`tl.simdvf_scope` / `tl.assume` / `tl.buffer_version` /
  `tl.manual_multi_buffer`），替换发生在 wrapper 内部的子语句上；
- group 的 item 可以不连续（中间夹着 `PRESERVE` metadata），逐项替换不会误伤其它 group。

### 5.5 寄存器模式下的分配改写

`RewriteFragmentAllocations`（710）：

- 从 `SBlock.alloc_buffers` 里删除被寄存器化的 fragment；
- 追加 `FragmentRegisterFile::Buffers()`（每个 slot 一个 `local.var` 向量 buffer）；
- 同步从 `layout_map` 删除这些 fragment 的条目，避免 `AscendLowerTileOp` 再次重映射。

只有在“寄存器模式 + 确实发生了替换”时才改写，未发射任何语句时保持原分配。

### 5.6 annotation 清理

P3 结束时无条件删除 `tl.vsimd.program` 与 `tl.vsimd.group_plan`；如果 P2 因 body
变化/非法 program 跳过了某个 VF，P3 也会在该 VF 上删除 P1 annotation。
因此最终 IR 中不存在 `tl.vsimd.*`。

### 5.7 段级发射：并行轴三段式、累加器与尾块掩码

离线 `EmitPlan` 与线上 `EmitItem` 共用 `EmitProgram`（`emit.cc:2665`）：
先发 program 级 prologue（共享掩码、索引 ramp、寄存器声明），再逐段
`EmitSegmentStmts`（2146），最后用 `program.axes`（hoisted 前缀）包一层 serial `for`。

**① 并行轴 + 内层轴 → 三段式（`EmitParallelAxisParts`，`emit.cc:2440`）**

当 `vector_axis` 是 parallel 且存在“内层轴”（index > vector_axis）时，一段被拆成
`pre → for inner: main → post`：

| 阶段 | 内容 |
| --- | --- |
| pre | `pre_stores`；与 promoted store **同址且值为 uniform** 的 pre 被 `folded`，不再写内存 |
| 累加器 | 每个 `promote` 命中者分配 `local.var` 向量；初值 = folded pre 值（`vdup`）或旧值 `vld` |
| inner | `SplitLoopInvariant` 提出不变量后，把内层轴发成 serial `for` |
| main | 逐 store 计算；累加器 store 写成 `acc[0] = op(acc[0], …)` |
| write-back | `vsts(addr, acc[0], mask, NORM_B*+)`，同一 chunk 只写一次 |
| post | `post_stores`（可接管 write-back 做寄存器转发） |

**② 外层轴补齐（`emit.cc:2166-2172`）**

三段式只负责内层轴；`vector_axis` 之前、且未被 `program.axes` 提升的轴仍要在这里
包成 serial 循环。多段 program 常常不提升任何轴（某个段把 `i` 当 lane 轴用），
缺了这一步就会发射出未绑定的迭代变量。

**③ 累加器位置**

`AllocBuffer(acc)` 写在 segment body 内（不是 program prologue）：外层 `for i`
因此每轮声明自己的槽位，避免 8 次迭代共享同一寄存器造成假 WAW/WAR 依赖。

**④ 尾块掩码（`ArithMask`，`emit.cc:858`；`TailMask`，863）**

- `mask_<bits>` 的 `<bits>` 是**元素位宽**（`pset(bits, PAT_ALL)` → `asc_create_mask_b32`），
  不是 lane 数；f32 的 `mask_32` 覆盖全部 64 lane；
- 真正按 lane 截断的是 `update_mask(count, bits)`（如 `tail8_32`）；
- 静态尾块谓词按 `(tail, bits)` 走 `SharedMask`，多段共享同一条绑定；
- 尾块的**算术**用尾块谓词（`ArithMask`），inactive lane 不进入数据通路
  （gather 的未定义值不会被 `vadds/vmuls` 传播）。

---

## 6. 跨阶段一致性

### 6.1 三层保障

| 层 | 机制 | 失败后果 |
| --- | --- | --- |
| item 定义 | 唯一的 `FlattenVSimdBody` | 三阶段对 span 的理解必然一致 |
| 条数/内容 | `item_count` + `body_fingerprint` | P2 跳过整 VF、P3 只删 annotation |
| 结构合法性 | `ReadVSimdProgram` + `validation_error` + `VerifyVSimdProgram` | P2 整 VF 拒绝 |

### 6.2 typed reader 与 payload 解码

| 入口 | 位置 | 作用 |
| --- | --- | --- |
| `ReadVSimdProgram`（102） | `vsimd_program.cc` | 解析公共字段、items、groups、captures、fragments、assumptions、attrs |
| `VerifyVSimdProgram`（214） | `vsimd_program.cc` | 校验 index/span/group 引用、状态词表、fragment 事实 |
| `ReadVSimdRegion`（135） | `polyhedral/vsimd_payload_read.cc` | REGION payload → `VSimdRegion`（含 layouts/assumptions/bindings） |
| `ReadVSimdCopy`（361） | `polyhedral/vsimd_payload_read.cc` | COPY payload → `VSimdCopyReducePayload` |
| `ReadVSimdReduce`（377） | `polyhedral/vsimd_payload_read.cc` | REDUCE payload → `VSimdReducePayload` |

### 6.3 回退矩阵

| 失败点 | 处理 | 最终由谁发射 |
| --- | --- | --- |
| P1 无 `T.Parallel` | 无 annotation | legacy |
| P1 item 不 lift（内存非法等） | `OPAQUE`，group 非 `P1_LIFTABLE` | legacy |
| P1 自校验失败 | `validation_error` | legacy（P2/P3 跳过） |
| body 指纹不一致 | P2 不写 group plan | legacy |
| REGION/COPY/REDUCE 任一 item 不支持 | 整组 `REJECTED` | legacy |
| VF fragment V1–V4 失败 | 整 VF 所有 group `REJECTED` | legacy |
| REGION 内 tile-op 展开/分段失败（`tile-op segment planning failed`） | 该 item 抛 `Unsupported` → 整组 `REJECTED`；离线 `build_plan` 直接抛给调用方 | legacy / 调用方 |
| P3 发射失败 | 整组回滚；寄存器模式下整 VF 回滚 | legacy |

---

## 7. 流水线接入与开关

`tilelang/ascend/pipeline.py:120-128`：

```python
mod = ascend_transform.NormalizeBufferVersion()(mod)
if pass_ctx.config.get(tilelang.PassConfigKey.TL_ASCEND_SIMD_LIFT, True):
    mod = ascend_transform.AscendLiftVSimd()(mod)
    mod = ascend_transform.AscendVectorizeVSimd()(mod)
mod = ascend_transform.AscendSimdVFLowerParallel()(mod)
```

| 配置 | 默认 | 作用 |
| --- | --- | --- |
| `tl.ascend_simd_lift` | `True` | P1/P2/P3 总开关；关闭后三个阶段全部不运行，`T.Parallel` 全部交给 legacy |
| `tl.ascend_vsimd_vectorize` | **已删除** | 新路径恒开，不再需要独立开关 |

顺序约束：P1 → P2 → P3 必须连续；legacy 放在 P3 之后，只处理被拒的 statement。

---

## 8. 测试与验收

### 8.1 分层用例

| 层 | 代表用例 | 断言 |
| --- | --- | --- |
| P1 | `test_ascend_vsimd_lift.py` | item/group/状态/fragments/assumptions 字段；无 Parallel 不写 program |
| P1 交接 | `test_staged_passes_exchange_annotations` | P1 不改 body；P2 保留 P1 annotation；P3 清空两者 |
| P2 REGION | `test_simple_group_is_planned_and_emitted` | `PLANNED`、`region_plan` 有 1 段、P3 出 `vld/vsts` |
| P2 事务 | `test_one_rejected_region_rejects_the_whole_group` | 一个 REGION 被拒 → 同组 producer 也不发射 |
| P2 COPY | `test_copy_reduce_group_is_planned_and_emitted`、`test_partial_region_copy_uses_the_region_indices` | 组内 COPY/REDUCE/REGION 全部 `PLANNED`；region 起点被保留 |
| P2 REDUCE | `test_reduce_clear_true_emits_an_identity_init`、`test_reduce_along_dim0_is_planned` | `clear` 语义、`init` program、`dim=0` |
| fragment | `test_in_vf_fragment_uses_the_register_file` | `fragment_mode == "register"`、`local.var`、无 `local.fragment` |
| 跨 VF fragment | `test_fragment_allocated_outside_the_vf_is_rejected` | group `REJECTED` 且理由明确，body 不变 |
| 一致性 | `test_body_change_rejects_the_whole_vf` | 指纹变化 → 无 group plan |
| 流水线 | `test_pipeline_uses_the_new_group_path_by_default`、`test_disabling_the_lift_skips_the_whole_chain` | 默认配置即出 `simd_inst::`，最终源码无 annotation；关闭 `tl.ascend_simd_lift` 时 P2/P3 不再构造 |
| tile-op in Parallel | `test_parallel_body_copy_is_planned`、`test_parallel_body_reduce_clear_false/true_*` | 归约走 `local.var` 累加器、每输出向量 1 次 `vsts`、identity 折成 `vdup`、无 gather |
| tile-op 分段 | `test_parallel_body_mixed_store_copy_reduce_is_planned` | d/c/m/e 四段保序、copy 段 flatten 无 gather、`for i` 存在且无自由变量、尾块算术用尾块掩码 |
| tile-op 合并 | `test_parallel_body_two_tileops_keep_source_order` | 两条相邻独立 copy 合并成 1 段 / 1 个 chunk 循环、2 对 `vld/vsts` |
| tile-op 负向 | `test_parallel_body_unsupported_reduce_kind_rejected`、`copy_global_source_rejected`、`unplannable_store_rejects_the_program` | `absmax` / 非 UB 源 / 不可规划语句分别按预期拒绝 |

### 8.2 回归结果

```text
testing/ascend/transform            : 593 passed
testing/ascend/auto_schedule        : 639 passed, 31 skipped
testing/ascend/target + cache + analysis : 19 + 8 + 22 = 49 passed
```

`auto_schedule/test_schedule_integration.py` 的 4 个 subcore 用例同时覆盖了
fragment/寄存器路径的端到端产物。当前实现已删除 `codegen_ascend.cc` 中的 S4
防御性守卫：它只识别 `tl.access_ptr` 或裸 `BufferLoad`，而流水线在 codegen 前
已执行 `LowerAccessPtr`（地址实参已变成 `tvm_access_ptr`），该分支实际不可达；
“`vld/vsts` 只访问 UB”由 P2 的 group 规划与 V1–V4 判定保证。

---

## 9. 能力矩阵

| 场景 | P1 | P2/P3 |
| --- | --- | --- |
| 1-D 连续逐元素 | `REGION LIFTED` | `BuildPlan` → `vld/vsts` |
| 多维 / 仿射偏移 / 逆序 / 转置 / 步长 / 间接读 | `REGION LIFTED` | 支持；不可构造索引 → 整组拒绝 |
| 值级条件（`T.if_then_else` / `Select`） | `REGION LIFTED` | `vsel` / predicated store |
| Parallel 体内的语句级 `if` | `OPAQUE`（NestCollector: `if inside T.Parallel`） | legacy（P2 的 `NormalizeBodyInto` IfThenElse 分支不可达） |
| Parallel 体内的 `T.serial` | `OPAQUE`（NestCollector: `serial loop inside T.Parallel`） | legacy（pre/main/post、多 segment、serial 吸收不可达） |
| `T.Parallel` 内的 tile-op（`T.copy`/`T.reduce_*`） | 前置 `NestedLoopChecker` 规则 2 直接 `ValueError`；`NestCollector` 也判 `OPAQUE`（`TileOp/copy inside T.Parallel`） | **P2/P3 已实现**（`ExpandParallelTileOps` + `SplitAllSegmentSources`/`PlanSources`，§4.8）：copy 合成并行 copy nest、reduce 合成「归约维最内层 serial」的 nest 并提升为寄存器累加器；标准管线不可达，仅离线 `build_plan` |
| serial 包 parallel（VF 顶层 serial） | `OPAQUE`（非 Parallel 顶层语句） | legacy |
| 多 store | 同一 `REGION` | 多 store 段 |
| NZ 目标 | `REGION LIFTED`（带 layout） | `vsstb` |
| fragment（VF 内定义） | `REGION LIFTED` + `fragments[]` | 寄存器文件（G0–G8 + V1–V4） |
| fragment（VF 外定义、VF 内使用） | `definition_scope=OUTSIDE_VF` | **group 直接拒绝**，legacy |
| 顶层 `T.copy`（UB↔UB） | `COPY TYPED` | Phase B 合成 Parallel copy |
| 顶层 `T.copy`（跨 scope） | `COPY TYPED` | 整组拒绝，legacy |
| 顶层 `T.reduce_max/sum/min` | `REDUCE TYPED` | Phase B 合成归约 nest（`clear` 两种语义） |
| `T.reduce_abs*` / bit 类 | `REDUCE TYPED` | 整组拒绝 |
| `BIND` / `ALLOC` / `DECL` | `PRESERVE` | 不感知、原样保留 |
| 无 `T.Parallel` 的 VF | 不写 program | legacy |

---

## 10. 已知限制与后续工作

| # | 限制 | 影响 | 建议 |
| ---: | --- | --- | --- |
| 1 | 跨 scope copy（UB↔L1、GM）不支持 | 这类 `T.copy` 的 group 整组回退 | Phase C 增加独立 `CopyPlan` |
| 2 | `T.copy` 的 transpose/annotations 没有结构化字段 | 只能按 region 语义处理等 extent/广播 | P1 增加 `copy_transpose` 等 payload |
| 3 | REDUCE 只支持 max/sum/min，无 `batch` | abs/bit/批量归约回退 | P1 结构化 `batch` 后再扩 |
| 4 | fragment 跨 VF 永久拒绝 | 跨 VF 值必须经 UB 中转 | 有真实算例再做 staging |
| 5 | P1 的 `NestCollector` 默认 `allow_control_flow=false`：`T.Parallel` 体内的 `T.serial` / `if` 语句会让整个 nest 变成 `UNSUPPORTED`/`OPAQUE`；VF 顶层 serial 同样是 `OPAQUE` | P2 的 serial 吸收、pre/main/post、多 segment、语句级 `if` 谓词化在标准管线不可达，只有离线 `build_plan` 覆盖 | 若路线图需要：接通 `allow_control_flow` 并加 P1→P2 端到端用例；否则删除对应死路径 |
| 6 | 全 `OUTSIDE_VF` fragment 的 VF 由 legacy 降级：生成的 CCE 会把局部数组 `frag[N]` 强转成 `__ubuf__` 指针 | 设备端是否被 bisheng 接受未验证 | P1 前置拒绝或 legacy 支持 fragment |
| 7 | L4（pre/post 未纳入段级 lane/NZ/pack2 校验）未修；当前 pre/post 在标准管线不可达 | 只影响离线 `build_plan` 或未来 CF 路径 | 段级校验统一扫 `pre ∪ stores ∪ post` |
| 8 | `allow_control_flow_` 是半接线：`VisitStmt` 的 allowlist 受标志控制，但 `VisitStmt_(ForNode)`（serial）与 `VisitStmt_(IfThenElseNode)` 无条件置 `unsupported` | 仅把标志置 true 不会让 serial/if 生效 | 两个 `VisitStmt_` 也要尊重该标志 |
| 9 | P1 的 `unsupported_reason` 停在 VF 级，item 只写 `no capture info for Parallel nest N` | 用户看不到真正的 `serial/if inside T.Parallel` 原因 | 把 VF 级 reason 透传到 item |
| 10 | `NestedLoopChecker` 的连续 Parallel 特例在 `in_parallel_context` 检查之前返回 | 跨 serial/if 的 Parallel 嵌套可绕过检查；与 P1 的 ordinal 口径不一致 | 把特例判断移到上下文检查之后，或补测试 |
| 11 | `TL_DISABLE_PRELOWER_SEMANTIC_CHECK` 未通过 `TVM_REGISTER_PASS_CONFIG_OPTION` 注册 | `PassContext(config={...})` 设置它会抛 `AttributeError`，前置语义检查实际恒开 | 注册该 key，或删除这个无效开关 |
| 12 | `T.Parallel` 体内 tile-op 的展开路径只有离线 `build_plan` 可达（前置检查 + P1 都拒绝） | 生产代码写 `T.Parallel { T.copy/ T.reduce_* }` 仍走 legacy | 若要启用：先放开前置检查规则 2，再让 P1 `NestCollector` 放行 tile-op 并透传 P2 需要的 region 事实 |
| 13 | tile-op 分段仍是**程序粒度**拒绝：任一段规划失败 → 整个 program `Unsupported` | 同 body 内可向量化语句被不可规划语句牵连 | 语句级分区 + 标量回退 + 依赖保序 |
| 14 | sub-lane 连续尾块必须 `vgather2`（`vld` 无谓词形式） | 8 元素级的小尾块无法用一条 `vld` 完成 | 显式“UB 尾部 padding”约定后放开越界 `vld`，或对小尾块走标量 |
| 15 | 归约累加链是单链（`acc = op(acc, x)` 逐 j 串行） | J 较大时受指令延迟限制 | 拆 2/4 个部分累加器，末尾合并 |

---

## 11. 与旧实现的变化清单

| 项 | 旧（`p2_p3_cpp_port_survey.md` 描述） | 现在 |
| --- | --- | --- |
| P1 pass | `LiftParallelToArray`，包 `AttrStmt("tl.vsimd.array_region")` | `AscendLiftVSimd`，写 `SBlock.annotations["tl.vsimd.program"]`，body 不变 |
| P1 输出 | 只有 `status + region` | item/group/状态/fragments/assumptions/fingerprint 全套事实 |
| P2 载体 | `AttrStmt("tl.vsimd.array_plan", VectorProgram)` | typed `VSimdGroupPlanSet`（含 COPY/REDUCE/fragment_regs） |
| P2 输入 | `RegionToPlanInput(region)` 反展开成 For 链 | **原始 span 语句**；region 只提供 layouts/assumptions |
| P2 粒度 | 单 nest 独立替换 | group 事务（任一失败整组拒绝） |
| P2 对 COPY/REDUCE | 无路径，整组拒绝 | Phase B：合成 nest 后规划 |
| fragment 规划 | 独立 pass `PlanFragmentRegisters`（AttrStmt 载体） | 合并进 P2（`PlanVfFragmentRegisters`），P3 消费 `fragment_regs` |
| P3 入口 | `LowerArrayPlan`（遍历 `array_plan` marker） | `AscendEmitVSimd`（遍历 typed group plan，按 span 替换） |
| legacy 兜底 | 被注释，无第二 emitter | 恢复为 P3 之后的固定一步 |
| 组合 pass | `AscendPolyhedralVectorize` | `AscendVectorizeVSimd`（P2 + P3），不再包含 P1 |
| 开关 | `TL_ASCEND_VSIMD_VECTORIZE` 默认 off | 删除；新链路默认开启，`TL_ASCEND_SIMD_LIFT` 为总开关 |
| 一致性 | 依赖 item 顺序隐式对齐 | `item_count` + `body_fingerprint` 显式校验 |
| `T.Parallel` 内 tile-op | 无路径（P1 判 opaque，交 legacy） | `ExpandParallelTileOps` 展开 + `SplitAllSegmentSources`/`PlanSources` 分段 + 寄存器累加器（离线可达） |

---

## 12. 能力边界与标准管线可达性

> 本节区分“P2/P3 代码里有实现”与“标准管线真的能走到”。判断依据：
> `NestedLoopChecker`（pre-lowering）+ P1 `NestCollector`（默认
> `allow_control_flow=false`）+ P2 只把 REGION/COPY/REDUCE 的 span 语句喂给
> `BuildPlan`。

### 12.1 三类能力划分

#### A. P1 挡掉，生产不可达

| 能力 | P2/P3 实现 | P1 挡点 |
| --- | --- | --- |
| serial 包 parallel | `SingleSegmentSource` 的 serial 轴、`EmitOuterStore` | VF 顶层非 Parallel For → `OPAQUE`（`lift_vsimd.cc:755-775`）；只有嵌套 Parallel 时整 VF OPAQUE（`:838-851`） |
| 间接/数据相关 store（per-lane 索引 → `vscatter`） | `ClassifyAccess` 标 `kIndirect`；`EmitStore` 走索引向量 | PR-A 把 data-dependent store map 判 `UNPROVEN`（`vsimd_memory_legality.cc:744-757`），lift 要求 `LEGAL`（`vsimd_region.cc:95-103`） |
| 非单射 store（如 `c[i//2]`） | P2 没有单射检查，索引向量可构造就规划 | PR-A 的 Z3 单射证明失败 → `ILLEGAL`/`UNPROVEN`（`vsimd_memory_legality.cc:759-781`） |
| 内存合法性 `UNPROVEN/UNSUPPORTED` 的 nest | P2 没有“必须证明”的门槛 | `BuildMinimalRegion` 只接受 `LEGAL` |
| 任意 Stmt 作为 `BuildPlan` 输入 | 离线 `build_plan` 入口 | 生产 P2 只传 REGION span（必为 `For(Parallel)`） |
| `T.Parallel` 体内 tile-op（copy/reduce） | `ExpandParallelTileOps` 展开 + `SplitAllSegmentSources`/`PlanSources` 分段 + 寄存器累加器（§4.8） | 前置 `NestedLoopChecker` 规则 2；直接调 P1 时 `NestCollector` 也判 `TileOp/copy inside T.Parallel` |

#### B. P1 只给逻辑事实、P2/P3 实现，生产可达

| 能力 | P1 提供 | P2/P3 额外实现 |
| --- | --- | --- |
| COPY Phase B | op 名子串识别 + src/dst/region | 合成 UB↔UB copy nest，支持 ND/广播/部分区域 |
| REDUCE Phase B | reduce_kind/dim/clear | max/sum/min 归约 nest，`clear=true` 额外发射 init |
| fragment 寄存器文件 | `fragments[]` 事实 | V1–V4 判定、`local.var` 寄存器文件、分配/layout 改写 |
| 值级条件 | 只保证 nest 内存合法 | `T.if_then_else`/`Select` → `vsel` / predicated store |
| 多 store | accesses 事实 | 同一 `SeqStmt` 内多个 `StoreSpec` 组成多 store 段 |
| 多轴 Parallel / flatten | domain | 轴打分、多轴选择、幂次 extent 融合 |
| gather（间接读） | 访问合法性 | 索引向量通道 + `vgather` |
| NZ / pack2 / tail / guard cuts | layouts 透传 | `vsstb`、`PK_B32`、掩码尾块、段间寄存器转发 |

#### C. 反向缺口

`region.outer_vars` / `region.local_views` 会被 `ReadVSimdRegion` 解析
（`vsimd_payload_read.cc:225-259`），但 `PlanOptions` 只有
`layouts/assumptions/bindings/shape_syms/vf_alloc_scope`（`plan.h:36-49`），
这两个字段没有消费者。

### 12.2 前置检查的口径

`NestedLoopChecker` 实际规则：

1. **连续 Parallel 链**允许，任意深度（每层 body 恰好是下一层 Parallel For）；
2. 最内层 body 允许 `T.serial`、`if`、元素级语句；禁止 tile-op
   （`T.copy`/`T.reduce_*`）与 `T.Pipelined`；
3. Parallel 上下文中禁止非连续 Parallel（前面有语句、或夹在 serial/if 里）；
4. 兄弟 Parallel 在 Parallel 外合法，在 Parallel 内非法；
5. 该检查在 pre-lowering 执行（`semantic_check.py:21-31`）。`PassConfigKey.TL_DISABLE_PRELOWER_SEMANTIC_CHECK`
   虽在代码里被读取，但未通过 `TVM_REGISTER_PASS_CONFIG_OPTION` 注册，`PassContext(config=...)`
   设置它会抛 `AttributeError`，因此标准管线里该检查实际**恒开**。

特例分支在 `in_parallel_context` 检查之前返回，存在可绕过形状：

```python
for i in T.Parallel(M):
    for k in T.serial(K):
        for j in T.Parallel(N):    # 本应因 in_parallel_context 被拒绝
            for l in T.Parallel(L):  # child 是 Parallel → 特例直接放行
                ...
```

P1 的 `ParallelOrdinalCollector` 把嵌套 Parallel 视为同一 nest
（`lift_vsimd.cc:161-183`，`depth_ > 0`），因此 checker 与 P1 对该形状口径不一致。
该形状虽然能绕过 checker，但 P1 的 `NestCollector` 仍会因 body 内的 serial 把它判成
`OPAQUE`（最终走 legacy），属于“检查放行、P1 拒绝”的静默回退；建议把特例判断移到
上下文检查之后，避免两个阶段的合法集合不一致。

### 12.3 可达性结论

**标准管线不可达**（只有离线 `build_plan` 覆盖）：

| 能力 | 触发条件 | P1 结果 |
| --- | --- | --- |
| pre/main/post 拆分 | Parallel body 内恰好一个 clean serial | `OPAQUE`（`serial loop inside T.Parallel`） |
| 多 segment `VectorProgram` | Parallel body 内 ≥2 个 clean serial | 同上 |
| `AbsorbSerialChain` / `SerialIsClean` | 任意 body 内 serial 链 | 同上 |
| 语句级 `if` 谓词化 | Parallel body 内含 `IfThenElse` | `OPAQUE`（`if inside T.Parallel`） |
| serial 包 parallel | VF 顶层 `For(kind=Serial)` | `OPAQUE`（非 Parallel 顶层语句） |
| 间接/数据相关 store | store 下标含 `BufferLoad` | `UNPROVEN` |
| 非单射 store | Z3 证明失败 | `ILLEGAL`/`UNPROVEN` |
| `T.Parallel` 体内 tile-op（copy/reduce） | body 内含 `tl.tileop.*` 调用 | 前置 checker 抛 `ValueError`；P1 判 `TileOp/copy inside T.Parallel` |
| tile-op 分段 / identity 折叠 / 相邻链合并 | 由 §4.8 的展开路径产生 | 同上（P1 不放行 tile-op） |

实测（`AscendLiftVSimd` → `AscendPlanVSimd`）：

```text
serial inside parallel        : P1 REGION OPAQUE（no capture info） / P2 无 plan
two serials inside parallel   : P1 REGION OPAQUE                  / P2 无 plan
if statement inside parallel  : P1 REGION OPAQUE                  / P2 无 plan
value-level if_then_else      : P1 REGION LIFTED / P1_LIFTABLE    / P2 PLANNED
```

**仍然可达**：多 store、值级 `T.if_then_else`、多轴 Parallel + flatten、
VF 顶层 COPY/REDUCE Phase B、fragment 寄存器文件、NZ/pack2/tail/gather（间接读）。

> 注意：`test_ascend_polyhedral_vectorize.py` 的 165 个离线用例覆盖了上表各项；
> 它们验证的是 `build_plan`/`emit_plan` 的规划能力，**不代表标准管线能触发**。

### 12.4 建议

- **短期不扩 P1**：把标准管线走不到的能力显式标注为“离线 API”——`AbsorbSerialChain`、
  `SerialIsClean`、`SingleSegmentSource` 的 serial/pre-post 分支、`NormalizeBodyInto` 的
  `IfThenElse` 分支、`VectorPlan::pre_stores/post_stores` 的发射、§4.8 的 tile-op 展开
  与分段；对应离线测试按“build_plan API 能力”表述，不要写成流水线能力。
- **要在 P1 打开 control flow**：不能只翻转 `allow_control_flow_`。`VisitStmt_(ForNode)`
  与 `VisitStmt_(IfThenElseNode)` 目前无条件置 `unsupported`（`vsimd_analysis_common.h:181-196`），
  需要同时让它们尊重该标志，并补一条 P1→P2 的端到端用例。
- **诊断**：P1 目前只把 VF 级 `unsupported_reason` 丢掉，item 写成
  `no capture info for Parallel nest N`；建议把真实原因透传到 item 的 `reason`。
- **`T.Parallel` 内的 tile-op**：P2/P3 已实现（§4.8），只差上游放行。要打通需要两步：
  1. 放开前置 `NestedLoopChecker` 规则 2（tile-op in Parallel）；
  2. 让 P1 的 `NestCollector` 放行 tile-op，并把 `tl.region` 事实透传到 item，
     使 P2 能拿到 buffer/indices/extents（目前 P2 直接读 span 语句，不需要 P1 结构化，
     但 P1 现在会先把该 nest 判成 `OPAQUE`）。
  `NormalizeBodyInto` 已对非 atomic 的 `Evaluate` 显式抛 `Unsupported`，不会再静默丢弃。

### 12.5 group 事务的动机

P2 选择“任一 item 失败整组拒绝”，原因不是 UB 语义，而是**依赖闭包与身份一致性**：

1. **group 是依赖闭包**：P1 用 DSU 连接 RAW 读后写和同 fragment 的 item
   （`lift_vsimd.cc:635-674`），一个 group 是一条 producer→consumer 链；P2/P3
   不重新做跨 item 依赖分析，拆分发射等于把一条链交给两个 emitter。
2. **fragment 寄存器身份**：fragment 写者被规划成寄存器后，P3 会移除其 alloc 并
   改写为 `local.var`；未发射的读者仍按 `local.fragment` 访问，会读到未定义内容。
   group 事务在更早一层消灭“写者已规划、读者回退”，避免 V1 escape 频繁触发。
3. **COPY/REDUCE Phase B 的整组语义**：合成 nest 的 layout/假设上下文来自整个 group，
   只规划同组 REGION 会破坏依赖闭包（见设计文档 §6.1 Phase A）。
4. **失败与回滚粒度**：`PLANNED`/`REJECTED` 二值、`items` 要么全有要么为空，
   P3 整组替换或整组保留，不存在“半发布计划”。

代价是**拒绝放大**：一个 item 失败会带走同组本可向量化的 item（例如
`test_one_rejected_region_rejects_the_whole_group` 中 `c[i] = a[i] + 1.0`）。
对纯 UB、无 fragment 的 group，按 item 发射在语义上通常可行；若要放宽，需要
先证明无 fragment、无 COPY/REDUCE 合成依赖、跨 item layout/alias 假设一致，
并把回滚粒度改成 item。

---

## 附录 A：typed group plan 字段

```text
VSimdGroupPlanSet {          // 一个 VF 的全部计划
  version: int,
  vf_index: int,
  groups: [VSimdGroupPlan],
  fragment_regs: FragmentRegisterMap?,   // VF 级寄存器文件
  fragment_mode: "none" | "register" | "rejected",
  fragment_reason: string
}

VSimdGroupPlan {
  group_id: int,
  status: "PLANNED" | "REJECTED",
  reason: string,
  diagnostics: [string],
  items: [VSimdItemPlan]      // 仅 PLANNED 时非空
}

VSimdItemPlan {
  item_index: int,
  span_begin: int, span_end: int,
  kind: "REGION" | "COPY" | "REDUCE",
  plan_kind: "REGION" | "COPY" | "REDUCE",
  region_plan: VectorProgram?,
  copy_plan: VSimdCopyPlan?,
  reduce_plan: VSimdReducePlan?
}

VSimdCopyPlan {
  kind: "ND" | "BROADCAST",
  src_scope, dst_scope,
  src_extents, dst_extents,
  program: VectorProgram
}

VSimdReducePlan {
  reduce_kind: "max" | "sum" | "min",
  dim: int, clear: bool,
  init: VectorProgram?,       // clear=true 的单位元初始化
  program: VectorProgram
}
```

## 附录 B：文件/函数索引

| 位置 | 符号 |
| --- | --- |
| `lift_vsimd.cc:635` | `BuildGroups` |
| `lift_vsimd.cc:728` | `BuildProgram` |
| `lift_vsimd.cc:973` | `LiftMutator` |
| `lift_vsimd.cc:1033` | FFI `AscendLiftVSimd` |
| `vsimd_handoff.cc:104/115/122` | `FingerprintVSimdBody` / `FlattenVSimdBody` / `ReplaceVSimdSpans` |
| `vsimd_program.cc:102/214` | `ReadVSimdProgram` / `VerifyVSimdProgram` |
| `polyhedral/vsimd_payload_read.cc:135/361/377` | `ReadVSimdRegion` / `ReadVSimdCopy` / `ReadVSimdReduce` |
| `vsimd_group_passes.cc:124` | `PlanRegionItem` |
| `vsimd_group_passes.cc:181/223` | `BuildCopyNest` / `PlanCopyItem` |
| `vsimd_group_passes.cc:299/351/404/429` | `MapReduceDims` / `BuildReduceNest` / `BuildReduceInitNest` / `PlanReduceItem` |
| `vsimd_group_passes.cc:499/532` | `GroupUsesOutsideFragment` / `PlanOneGroup` |
| `vsimd_group_passes.cc:620/753` | `GroupPlanner` / `GroupEmitter` |
| `vsimd_group_passes.cc:710/736` | `RewriteFragmentAllocations` / `EmitItem` |
| `vsimd_group_passes.cc:839/846/853` | `PlanVSimd` / `EmitVSimd` / `VectorizeVSimd` |
| `fragment_plan.cc:172` | `PlanVfFragmentRegisters` |
| `group_plan.cc:282-310` | `IRDocsifier` 内联打印注册 |
| `group_plan.h:38–214` | typed plan 对象 |
| `emit.h:28/65` | `FragmentRegisterFile` / `EmitPlan` |
| `polyhedral/body.cc:138/230/343/443/547` | `BuildCopyNestFromRegions` / `BuildReduceNestFromRegions` / `TileOpExpander` / `ReduceStagingFuser` / `ExpandParallelTileOps` |
| `polyhedral/plan.cc:428/650/2225/2335/2408/2641/2675/2701/2774` | `AbsorbLoopChain` / `FlattenAxes` / `PlanSegment` / `ChainSegmentSources` / `SplitAllSegmentSources` / `SegmentBody` / `PlanSources` / `AssembleProgram` / `BuildPlan` |
| `polyhedral/plan.cc:993/1264` | `FoldIdentityInit` / `ClassifyReduce` |
| `polyhedral/emit.cc:858/863` | `ArithMask` / `TailMask` |
| `polyhedral/emit.cc:2146/2440/2665/2710` | `EmitSegmentStmts` / `EmitParallelAxisParts` / `EmitProgram` / `EmitPlan` |
| `pipeline.py:120-128` | 新链路接入 |

## 附录 C：复现命令

```bash
cd /home/cjl/workspace/tilelang

# 构建（新增/删除源文件后先 reconfigure）
cmake -S . -B build && cmake --build build -j$(nproc)

# P1→P2→P3 契约（11 例）
python -m pytest testing/ascend/transform/test_ascend_vsimd_group_passes.py -q

# 三阶段全场景 + 离线 plan/emit（169 例，含 T.Parallel 体内 tile-op 8 例）
python -m pytest testing/ascend/transform/test_ascend_polyhedral_vectorize.py -q

# 只跑 tile-op in Parallel 的展开/分段/累加器用例
python -m pytest testing/ascend/transform/test_ascend_polyhedral_vectorize.py -q -k "parallel_body"

# 整个 Ascend 变换域（593 例）
python -m pytest testing/ascend/transform -q

# 自动调度 + 变换域（含 4 个 subcore 端到端用例）
python -m pytest testing/ascend/auto_schedule testing/ascend/transform -q
```
