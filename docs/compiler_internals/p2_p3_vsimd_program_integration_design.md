# P2/P3 对接新 P1（`tl.vsimd.program`）设计方案

> **历史文档**：本文记录写作时的实现/设计状态。后续改动已删除或重命名部分
> 文件、测试与行号；当前实现见 `src/ascend/transform/parallel_vectorize/` 与
> `src/ascend/transform/vsimd_*`，以代码为准。


> **⚠️ 已废弃（`p_123_parallel_new_p1` 分支）**：本设计针对的 `tl.vsimd.program`
> 载体与 fragment 寄存器规划已**全部删除**；`src/ascend/transform/vsimd_program.{h,cc}`
> 已不存在。现行方案是 P1 在每个顶层 `T.Parallel` 上挂
> `For.annotations["tl.vsimd.region"]`，触碰到 `local.fragment` 的 Parallel 按
> Parallel 粒度判 OPAQUE 并走 legacy 标量路径。本文仅作为历史设计记录保留。
> 现行说明见 [p1_region_p2_p3_integration_report.md](p1_region_p2_p3_integration_report.md)。

## 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档状态 | 设计稿 v2（Phase 0/1/2/3 与旧路径清理已实现，见附录 D） |
| 编写目的 | 定义 P2/P3 如何消费 `vsimd-frag` 分支的新 P1 输出，并给出能力差距与待完善列表 |
| P1 设计基线 | `vsimd-frag` 分支，尖端 `45bebb8b`；P1 实现提交 `c1133a7f`、`45bebb8b` |
| P1 旧基线 | `stable-parallel` 尖端 `5b254ed7`（PR-E，只读分析库） |
| P2/P3 基线 | `p2_p3_based_stable-parallel`（已合并 `stable-parallel`，P2/P3 C++ 位于 `src/ascend/transform/parallel_vectorize/`） |
| 关联文档 | `docs/compiler_internals/vsimd_p1_analysis.md`、`p2_p3_cpp_port_survey.md`、`p2_p3_design_implementation_report.md` |
| 适用范围 | P2（规划）、P3（发射）、流水线接入、测试与迁移；不包含 P1 算法本身的修改 |

> 本文是设计提案，不代表当前代码已经实现。文中的“现状”均以 P2/P3 基线分支和 P1 `vsimd-frag` 分支的静态代码为准。

---

## 0. 执行摘要

### 0.1 一句话结论

新 P1 已经把“逻辑事实 + 事务边界”做出来了；P2/P3 的正确对接方式不是“把 `array_region` 换成 annotation”，而是：

> **以 group 为事务单位、以原始 span 语句为规划输入、以 body fingerprint 为一致性保障、以 legacy 为兜底，重建 P2/P3 的输入/输出契约。**

### 0.2 五个设计原则

| # | 原则 | 说明 |
| ---: | --- | --- |
| 1 | **P1 annotation 是唯一输入** | 不再使用 `tl.vsimd.array_region` / `VSimdRegionRef` / `LiftParallelToArray`；P2 只读 `SBlock("SIMD_VF").annotations["tl.vsimd.program"]` |
| 2 | **group 是事务单位** | P2 按 `P1_LIFTABLE` group 整体规划；任一 item 失败，整组 `REJECTED`；P3 整组替换或整组保留 |
| 3 | **span 必须可验证** | P1/P2/P3 共用同一套 VF body 展平规则，并增加 `body_fingerprint`；P2/P3 发现 body 变化就整 VF 回退 |
| 4 | **P1 逻辑事实 ≠ P2 物理计划** | P2 仍要重新做轴/lane/layout/fragment 规划；`P1_LIFTABLE` 只是“允许尝试” |
| 5 | **legacy 兜底不能删** | 在 COPY/REDUCE、跨 VF fragment、P2/P3 已知缺陷收口前，被拒 group 必须由 legacy 处理 |

### 0.3 目标流水线

```text
NormalizeBufferVersion
  AscendLiftVSimd                 # P1，默认 on（TL_ASCEND_SIMD_LIFT 可关），metadata-only
  AscendPlanVSimd                 # P2，默认 on，消费 tl.vsimd.program，产出 typed group plan
  AscendEmitVSimd                 # P3，默认 on，按 group 事务替换 item span
  AscendSimdVFLowerParallel       # legacy fallback，始终保留
  AscendLowerTileOp
  ...
```

### 0.4 当前最大缺口

1. **载体不兼容**：旧 P2 只认 `AttrStmt("tl.vsimd.array_region")`，新 P1 只产出 SBlock annotation。
2. **没有 group 事务**：旧 P2 按单个 nest 独立替换，无法表达 P1 的依赖闭包。
3. **COPY/REDUCE 无规划路径**：P1 已把它们记为 `TYPED` item，但旧 P2/P3 没有独立的 copy/reduce plan/emit。
4. **fragment 跨 VF/escape 无策略**：P1 只记录事实，旧 P2 的 register 规划是 VF 原子且 G0 直接拒绝 VF 外分配。
5. **span/body 一致性无保障**：旧 P2 按 nest 顺序配对，新 P1 用展平 item span；中间插入 body-mutating pass 会静默错位。

---

## 1. 背景与现状

### 1.1 三个“P1”必须区分

| 名称 | 位置 | 形态 | 是否是 P2/P3 的输入 |
| --- | --- | --- | --- |
| **旧 P1-A** | `stable-parallel` 的 `vsimd_memory_legality` / `vsimd_capture_scope` / `vsimd_region` | 只读分析库：报告 + 类型化 `VSimdCaptureInfo` / `VSimdRegion`，无 pass、无载体、不进流水线 | 否（只是算法基础） |
| **旧 P1-B** | `p2_p3_based_stable-parallel` 的 `polyhedral/stage_passes.cc` `LiftParallelToArray` | 在 P2/P3 分支内部临时实现的 P1 包装：调用 `BuildP2PlanInputs`，把 LIFTED nest 包成 `AttrStmt("tl.vsimd.array_region")` | 是（旧 P2 的实际输入） |
| **新 P1** | `vsimd-frag` 的 `lift_vsimd.cc` + `vsimd_program.{h,cc}` | 默认运行的 metadata pass：给 `SIMD_VF` SBlock 挂 `tl.vsimd.program` annotation，body 不变；附带 typed reader/verifier | 目标上应是（当前 P2 尚未适配） |

本文后续“旧 P1”指旧 P1-B，“新 P1”指 `AscendLiftVSimd`。

### 1.2 新 P1 的输出回顾

唯一契约：

```text
SBlock("SIMD_VF").annotations["tl.vsimd.program"] = Map<String, Any>
```

关键语义（详见 `vsimd_p1_analysis.md` §3–§5）：

- `items` 按 VF body 顶层展平源序稠密排列，`items[i].span_begin == i`；
- item 类型：`REGION / COPY / REDUCE / BIND / ALLOC / DECL / OPAQUE`；
- status：`analysis_status`（单独分析）与 `status/effective_status`（group 事务后）；
- group：`P1_LIFTABLE / PRESERVE / OPAQUE`，只有 `P1_LIFTABLE` 允许 P2 事务替换；
- `REGION` 带 `region` payload；`COPY/REDUCE` 带 `src/dst/src_region/dst_region` 等逻辑字段；
- `BIND/ALLOC/DECL` 是 `PRESERVE` metadata，必须原样保留；
- `captures`、`fragments`、`assumptions`、`attrs` 是 P1 给出的上下文；
- `validation_error` 出现时，整个 VF 不可用；
- `ReadVSimdProgram` / `VerifyVSimdProgram` 是唯一推荐的读取入口。

### 1.3 旧 P2/P3 的输入路径

```text
AscendParallelVectorize
  = LiftParallelToArray
      BuildP2PlanInputs(func)
        AnalyzeVSimdMemoryStatuses(func)     // per-VF
        AnalyzeVSimdCaptureInfo(func)
        BuildMinimalRegion(info, statuses[vf])
        RegionToPlanInput(region)             // 反展开成 Parallel For 链
        RestoreBindings(...)
      VfRegionMarker / RegionMarkerInserter
        -> AttrStmt(VSimdRegionRef, "tl.vsimd.array_region", 1, nest)
  -> PlanArray
      PlanMarkerReplacer
        RegionToPlanInput(carrier->region)
        BuildPlan(input, nullopt, options)
        -> AttrStmt(VectorProgram, "tl.vsimd.array_plan", 1, nest)
  -> PlanFragmentRegisters
      FragmentRegisterPlanner（VF 原子，G0/V1/V2/V3/V4）
  -> LowerArrayPlan
      PlanMarkerLower -> EmitPlan
```

关键事实：

- `VSimdRegionRefNode` 定义在 `stage_passes.cc` 内部，未在头文件暴露；
- P2 只消费 `carrier->region`，不读 SBlock annotation；
- `BuildP2PlanInputs` 使用 **per-VF** `AnalyzeVSimdMemoryStatuses`；
- `PlanOptions` 中 `layouts` 和 `vf_alloc_scope` 被读取；`assumptions/bindings/shape_syms` 仍是死字段；
- P2/P3 分支的 pipeline **注释掉了 legacy**，被拒 nest 留在 `T.Parallel`，没有第二 emitter；
- 旧 P2/P3 有 160 个测试函数 / 165 collected case，8332 行实现（含头文件），4 项 P0 缺陷（L1–L4）。

### 1.4 为什么不能直接合并

| # | 冲突 | 直接合并的后果 |
| ---: | --- | --- |
| 1 | 载体不同 | `PlanArray` 找不到 `array_region`，P2 空跑；新 annotation 被忽略 |
| 2 | group vs nest | P2 可能 lift consumer、放弃 producer，语义断裂 |
| 3 | per-item vs per-VF 状态 | P2 可能拒绝 P1 已 lift 的独立 Region（损失优化） |
| 4 | COPY/REDUCE item | 旧 P2 无路径，含 copy/reduce 的 group 无法整组规划 |
| 5 | fragment 事实 | `OUTSIDE_VF/cross_vf/escapes_vf` 未消费，可能错误进入 register mode |
| 6 | 流水线顺序 | P1 后接 legacy 会改 body，annotation span 失效 |
| 7 | legacy 被注释 | 被拒 group 残留在 `T.Parallel`，没有 emitter 兜底 |
| 8 | 状态词表 | 旧 P2 依赖 `FALLBACK/INTERNAL_ERROR`，新 P1 没有这两个状态 |

---

## 2. 目标架构

### 2.1 分层与数据流

```text
┌────────────────────────────────────────────────────────────────────┐
│ pipeline                                                           │
│  NormalizeBufferVersion                                            │
│  AscendLiftVSimd                      P1 metadata-only             │
│    ├─ AnalyzeVSimdCaptureInfo                                     │
│    ├─ AnalyzeVSimdMemoryRegions                                   │
│    ├─ FlattenVSimdBody + item 构造 + BuildGroups                  │
│    └─ 写 SBlock.annotations["tl.vsimd.program"]                    │
│  AscendPlanVSimd                      P2 group planner             │
│    ├─ ReadVSimdProgram + validation_error                          │
│    ├─ FlattenVSimdBody + fingerprint                              │
│    ├─ PlanGroup（REGION/COPY/REDUCE）                              │
│    └─ 写 SBlock.annotations["tl.vsimd.group_plan"]                 │
│  AscendEmitVSimd                      P3 group emitter             │
│    ├─ 预发射所有 PLANNED group                                     │
│    ├─ SpanReplacer 逐 item 替换                                    │
│    └─ 删除 P1/P2 annotation                                       │
│  AscendSimdVFLowerParallel            legacy fallback              │
│  AscendLowerTileOp                                                │
│  ...                                                              │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 阶段职责与载体

| 阶段 | 输入 | 输出 | 载体 | 失败语义 |
| --- | --- | --- | --- | --- |
| P1 | `SIMD_VF` body | 逻辑 item / group / region / fragment 事实 | `SBlock.annotations["tl.vsimd.program"]` | body 不变；分析失败写 `validation_error` |
| P2 | `tl.vsimd.program` + 原始 body | 每组的规划结果 | `SBlock.annotations["tl.vsimd.group_plan"]` | 整组 `REJECTED`；不写部分 item plan |
| P3 | `tl.vsimd.group_plan` + 原始 body | 替换后的 `tl.simd.*` TIR | 直接改 body；删除 P1/P2 annotation | 整组回滚；原语句保留 |
| legacy | 未被 P3 替换的 `T.Parallel` / `T.copy` / `T.reduce_*` | 原有 SIMD lowering | 无新载体 | 原有行为 |

**为什么继续用 SBlock annotation 而不是 AttrStmt：**

- AttrStmt 会改变 body，P1 的 span 立即失效；
- 一个 VF 可能有多个 group，`Map<group_id, GroupPlan>` 放在一个 annotation 里最简单；
- 失败时不需要“解载体”，body 本来就是原样。

---

## 3. 接口设计

> 本节中的 `body_fingerprint`、`ReadVSimdRegion/Copy/Reduce`、`VSimdGroupPlanSet` 等均为**本设计新增接口**，当前 P1 分支尚未实现；已有的 `tl.vsimd.program`、`ReadVSimdProgram`、`VerifyVSimdProgram` 继续沿用。

### 3.1 共享 VF body 展平 + 指纹

把 `lift_vsimd.cc` 里的 `FlattenTopLevel` 抽到 `src/ascend/transform/vsimd_handoff.{h,cc}`：

```cpp
struct VSimdFlattenedBody {
  std::vector<tirx::Stmt> stmts;              // 顶层 item，按源序
  std::vector<ffi::Map<ffi::String, ffi::Any>> assumptions;  // anchor/anchor_end
  std::vector<ffi::Map<ffi::String, ffi::Any>> attrs;
  uint64_t fingerprint{0};                    // StructuralHash(stmts)
};

VSimdFlattenedBody FlattenVSimdBody(const tirx::Stmt& body);
```

P1 写 annotation 时增加：

```text
body_fingerprint : uint64
item_count       : int
```

P2/P3 重新展平后检查：

```cpp
if (body.stmts.size() != program.item_count ||
    body.fingerprint != program.body_fingerprint) {
  // 中间有 body-mutating pass，或展平规则不一致：整 VF 交 legacy
}
```

设计要点：

- 展平规则必须与 P1 完全一致：`SeqStmt` 递归；`tl.simdvf_scope` 透明；`tl.assume` / `tl.buffer_version` / `tl.manual_multi_buffer` 透明；`AllocBuffer` / `DeclBuffer` / `Bind` 各占一个 item；其它节点各占一个 item。
- `fingerprint` 只在同一进程内使用，不要求跨序列化稳定。
- P2/P3 不应各自复制一份展平代码；统一使用 `FlattenVSimdBody`。

### 3.2 typed payload reader

当前 `VSimdProgramItem` 只解析公共字段，`region/src/dst` 仍在 `meta` map 里。建议增加：

```cpp
bool ReadVSimdRegion(const ffi::Map<ffi::String, ffi::Any>&,
                     VSimdRegion*, std::string*);
bool ReadVSimdCopy(const VSimdProgramItem&, VSimdCopyPayload*, std::string*);
bool ReadVSimdReduce(const VSimdProgramItem&, VSimdReducePayload*, std::string*);
```

同时补：

- `VSimdProgram::has_validation_error()`（现在只能查 `raw`）；
- `VSimdProgramItem::effective_status`（现在只有 `status`）；
- `kind` 枚举校验；
- `accesses.value` 可选处理；
- `has_layout` 与 `layout` 的一致性校验。

`VSimdRegion` 继续复用 `src/ascend/transform/vsimd_region.h` 的现有结构，不重新定义。

### 3.3 group plan 对象

**已实现**：`polyhedral/group_plan.{h,cc}` 用 typed FFI 对象而不是再拼 Map。P3 直接消费，测试也能断言真实字段：

```text
VSimdGroupPlanSet {
  version: int,
  vf_index: int,
  groups: [VSimdGroupPlan]
}

VSimdGroupPlan {
  group_id: int,
  status: "PLANNED" | "REJECTED",
  reason: string,
  diagnostics: [string],
  items: [VSimdItemPlan]
}

VSimdItemPlan {
  item_index: int,
  span_begin: int,
  span_end: int,
  kind: "REGION" | "COPY" | "REDUCE",
  plan_kind: "REGION" | "COPY" | "REDUCE",
  region_plan: VectorProgram?,     // REGION
  copy_plan: CopyPlan?,           // COPY，Phase B
  reduce_plan: ReducePlan?        // REDUCE，Phase B
}
```

约束：

- 只有整组所有 item 都规划成功，`status=PLANNED`；
- 任一失败，整组 `REJECTED`，**不写部分 item plan**；
- `PRESERVE/OPAQUE` group 不产生 entry；
- `group.items` 的顺序与 P1 一致；`span_begin/end` 直接抄 P1，不能重算。

### 3.4 pass API 与开关

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `AscendLiftVSimd` | P1 pass | 已有；默认 on；`TL_ASCEND_SIMD_LIFT` opt-out |
| `AscendPlanVSimd` | P2 pass | 新；读 annotation，写 group plan |
| `AscendEmitVSimd` | P3 pass | 新；读 group plan，替换 item span |
| `AscendVectorizeVSimd` | 组合 pass | P2 + P3；测试/兼容入口，不在默认 pipeline 中 |
| `TL_ASCEND_VSIMD_VECTORIZE` | config | P2+P3 总开关，默认 `False` |
| `StripVSimdProgram` | cleanup | P2/P3 关闭时删除 P1 annotation，避免进入 codegen |

开关策略：

```python
TL_ASCEND_SIMD_LIFT = True          # P1，默认 on
TL_ASCEND_VSIMD_VECTORIZE = False   # P2+P3，默认 off
```

- P1 默认 on：保持现状；
- P2/P3 默认 off：合并初期不改变默认编译结果；
- 打开后：P1 → P2 → P3 → legacy；
- 关闭后：P1 annotation 由 `StripVSimdProgram` 在流水线尾部删除，legacy 照常工作。

---

## 4. P2：Group Planner

### 4.1 主流程

```cpp
for (SBlock vf : all SIMD_VF) {
  auto ann = vf.annotations["tl.vsimd.program"];
  if (!ann) continue;                          // 无 P1 输出，整 VF 交 legacy

  VSimdProgram program;
  if (!ReadVSimdProgram(ann, &program, &err) ||
      program.has_validation_error()) {
    LOG(WARNING) << "invalid P1 program: " << err;
    continue;                                  // 整 VF 拒绝
  }

  VSimdFlattenedBody body = FlattenVSimdBody(vf->body);
  if (body.stmts.size() != program.item_count ||
      body.fingerprint != program.body_fingerprint) {
    LOG(WARNING) << "body changed between P1 and P2";
    continue;
  }

  VSimdGroupPlanSet plans;
  for (const VSimdGroupPlan& group : program.groups) {
    if (group.status != "P1_LIFTABLE") continue;   // PRESERVE/OPAQUE 原样
    plans.groups.push_back(PlanGroup(program, group, body, vf));
  }
  vf.annotations["tl.vsimd.group_plan"] = plans;
}
```

### 4.2 PlanGroup

```cpp
VSimdGroupPlan PlanGroup(program, group, body, vf) {
  VSimdGroupPlan out;
  for (int item_index : group.items) {
    const auto& item = program.items[item_index];
    switch (item.kind) {
      case REGION:
        if (item.status != "LIFTED") return Reject(...);
        VSimdRegion region;
        if (!ReadVSimdRegion(item, &region, &err)) return Reject(...);
        // 关键：直接规划原始 span 语句，不再 RegionToPlanInput 反展开
        PlanOptions options = MakeOptions(program, item, region, vf);
        try {
          out.items.push_back({item, BuildPlan(body.stmts[item.span_begin],
                                               std::nullopt, options)});
        } catch (const Unsupported& e) {
          return Reject(e.message(), e.diagnostics());
        }
        break;
      case COPY:
        // Phase B：合成 UB<->UB 的 Parallel copy nest，复用 BuildPlan
        return PlanCopyItem(...);
      case REDUCE:
        // Phase B：合成源域 Parallel 归约 nest（clear=true 另加 init）
        return PlanReduceItem(...);
      default:
        return Reject("unexpected item in P1_LIFTABLE group");
    }
  }
  out.status = "PLANNED";
  return out;
}
```

### 4.3 规划原始语句，而不是 region 反展开

设计选择：**P2 直接规划 `body.stmts[item.span_begin]`，region 只作为逻辑上下文。**

理由：

1. 新 P1 同时给了 region payload 和 span，P2 已经能拿到原始 `T.Parallel` 语句；
2. 原始语句保留了 `Bind`、`DeclBuffer`、前后语句、aliases，`BuildPlan` 的 serial 吸收/pre-post 能力仍然可用；
3. `RegionToPlanInput` 的 `ExpandVSimdRegionToLoops` 只重建 store，会丢 Bind/别名/上下文，且需要 `RestoreBindings`；
4. 原始语句路径让 P2 的输入与自定义/手写 region 解耦，未来 `T.vsimd` 可以复用同一 planner。

保留：

- `RegionToPlanInput` 作为离线 API / 测试 helper；
- `BuildP2PlanInputs` 作为旧 P1-B 兼容入口，不进入新流水线。

### 4.4 PlanOptions 接线

```cpp
PlanOptions MakeOptions(program, item, region, vf) {
  PlanOptions o;
  o.layouts      = region.layouts;
  o.assumptions  = program assumptions intersecting item span
                 + region.assumptions;
  o.bindings     = region.bindings;
  o.shape_syms   = region.shape_syms;
  o.vf_alloc_scope = vf->alloc_buffers;
  return o;
}
```

同时修改 `BuildPlan`：

```cpp
arith::Analyzer analyzer;
for (const PrimExpr& a : options.assumptions) analyzer.ctx->EnterConstraint(a);
for (auto& [var, value] : options.bindings) analyzer.Bind(var, value);
// ... 原有规划逻辑
```

这是让 P1 的 `T.assume` / 尾块 Bind 真正参与 P2 证明的必要改动（已实现）。

### 4.5 Fragment 策略

P2 在规划 group 前先做两项检查：

1. **group 级**：group 内任一 item 的读写 buffer 命中 `definition_scope == OUTSIDE_VF`
   的 fragment，整组直接 `REJECTED`，理由
   `"fragment 'x' is allocated outside this VF (cross-VF fragment); stage it through UB first"`。
   跨 VF fragment 不会进入 register mode。
2. **VF 级**（`polyhedral/fragment_plan.cc` 的 `PlanVfFragmentRegisters`）：

| `program.fragments` 事实 | P2 动作 |
| --- | --- |
| `definition_scope == OUTSIDE_VF` | group 级直接拒绝（上面第 1 项）；`BuildPlan` 的 G0 也会兜底 |
| `IN_VF` + 单 VF + 无 escape | 允许，VF 级 V1–V4 通过后写 `fragment_regs` |
| fragment 被未规划 item 使用（V1 escape） | 整个 VF 的 group 全部 `REJECTED` |
| 同一 data Var 的寄存器形状不一致（V3） | 整个 VF 拒绝 |
| 寄存器预算超限（V2） | 整个 VF 拒绝 |
| 生命周期不合法（V4：先读后写 / 掩码部分写） | 整个 VF 拒绝 |
| fragment 有非 identity layout（G6）或 packed store（G5） | `BuildPlan` 抛 `Unsupported` → 整组拒绝 |

**VF 原子性**：`PlanVfFragmentRegisters` 是 VF 级判定，不能“一个 group 用寄存器、
另一个 group 用内存”。任一检查失败 ⇒ 该 VF 所有 `PLANNED` group 改判 `REJECTED`，
P3 不替换任何语句，整 VF 交 legacy。P3 只有在确实发生替换时才改写 VF 的
`alloc_buffers` 与 `layout_map`（把 fragment 分配换成 `local.var` 寄存器）；若某个
group 在 P3 发射失败且 VF 处于 register mode，则整 VF 回滚，避免“部分寄存器、
部分内存”。

### 4.6 失败语义与诊断

- 单个 item 失败 → 整组 `REJECTED`；`reason` 记录第一个失败原因，`diagnostics` 保留逐 item 原因；
- 不允许“部分 item PLANNED”；
- `P1_LIFTABLE` 不是 P2 的发射许可；`BuildPlan` 仍可能因 dtype/lane/NZ/pack2/fragment 拒绝；
- 诊断中带 `group_id`、`item_index`、`span_begin/end`、`kind`，便于定位；
- P2 不修改 body，不删除 annotation。

---

## 5. P3：Group Emitter

### 5.1 主流程

```cpp
for (SBlock vf : all SIMD_VF) {
  auto plan_ann = vf.annotations["tl.vsimd.group_plan"];
  if (!plan_ann) continue;

  VSimdGroupPlanSet plans = ...;
  VSimdFlattenedBody body = FlattenVSimdBody(vf->body);
  if (body.stmts.size() != ...) { continue; }

  // 1) 先发射所有 PLANNED group，失败整组丢弃
  std::map<int, tirx::Stmt> replacements;   // span_begin -> emitted stmt
  for (const VSimdGroupPlan& group : plans.groups) {
    if (group.status != "PLANNED") continue;
    std::vector<tirx::Stmt> emitted;
    bool ok = true;
    for (const VSimdItemPlan& item : group.items) {
      try {
        emitted.push_back(EmitItem(item));
      } catch (const Unsupported& e) {
        LOG(WARNING) << "P3 rejected group " << group.group_id << ": " << e;
        ok = false;
        break;
      }
    }
    if (!ok) continue;                      // 整组回滚：不写任何 replacement
    for (size_t i = 0; i < group.items.size(); ++i)
      replacements[group.items[i].span_begin] = emitted[i];
  }

  // 2) 按 span 替换；保留 wrapper / metadata
  if (!replacements.empty())
    vf.body = SpanReplacer(body, replacements)(vf->body);

  // 3) 清理 annotation
  vf.annotations.erase("tl.vsimd.program");
  vf.annotations.erase("tl.vsimd.group_plan");
}
```

### 5.2 SpanReplacer

必须和 P1 `FlattenVSimdBody` 使用同一套规则：

- `SeqStmt` 递归；
- `tl.simdvf_scope` 透明；
- `tl.assume` / `tl.buffer_version` / `tl.manual_multi_buffer` 透明；
- `AllocBuffer` / `DeclBuffer` / `Bind` 各占一个 item；
- 其它节点各占一个 item。

替换时只替换第 `i` 个顶层 item 的语句，**不重建整个 body**：

- 保留 `DeclBuffer(view)`、`Bind`、`AllocBuffer`；
- 保留 `tl.assume` / `buffer_version` wrapper；
- 保留 `tl.simdvf_scope`；
- 不需要按 anchor 重发 assumptions/attrs；
- group item 不连续也能逐项替换。

如果未来必须重建 body，再走 `program.assumptions/attrs` 的 anchor 重发路径。

### 5.3 整组回滚

- P2 已 `REJECTED` 的 group：P3 不替换；
- P3 发射任一 item 失败：整组不替换，原语句保留；
- 已成功发射的 group 不受其它 group 失败影响；
- 失败 group 交 legacy；P3 只 `LOG(WARNING)`，不 `LOG(FATAL)`。

### 5.4 metadata 与 wrapper

- `BIND/ALLOC/DECL` 永远不在 `P1_LIFTABLE` group 内（P1 保证），所以 P3 不会替换它们；
- `SpanReplacer` 保留原语句，所以 `DeclBuffer(view)` 不会丢；
- `tl.assume` / `buffer_version` / `manual_multi_buffer` 保持原位；
- 如果 P3 选择重建 body，必须按 `program.assumptions/attrs` 的 `[anchor, anchor_end)` 重发；v1 不采用重建路径。

### 5.5 annotation 清理

- P3 成功后删除 `tl.vsimd.program` 与 `tl.vsimd.group_plan`；
- 如果 P3 只替换部分 group，annotation 仍指向旧 body，必须整块删除；
- legacy 不需要 annotation；
- P2/P3 关闭时，由 `StripVSimdProgram` 在流水线尾部删除 `tl.vsimd.program`。

### 5.6 legacy 兜底

- P1 → P2 → P3 连续运行，legacy 放在 P3 之后；
- legacy 处理所有未被替换的 `T.Parallel`、顶层 `T.copy`、`T.reduce_*`；
- P2/P3 分支当前“legacy 注释掉、无第二 emitter”的状态不能直接用于新 P1；
- 如果坚持 legacy off，P2/P3 必须覆盖所有 group；v1 不推荐。

---

## 6. COPY/REDUCE 演进路线

### 6.1 阶段划分

| 阶段 | 做法 | 影响 |
| --- | --- | --- |
| Phase A（先合并） | P2 对含 COPY/REDUCE 的 group 直接 `REJECTED`；P3 不替换；legacy/`AscendLowerTileOp` 处理 | 安全；FA 的 softmax group 会整组回退，但不会错误拆分 |
| Phase B（UB↔UB copy） | 把 `T.copy` 合成为单个 `T.Parallel` store nest，复用 `BuildPlan`/`EmitPlan`；支持 ND/NZ/transpose/broadcast；跨 scope（UB↔L1）继续拒绝 | 覆盖大多数 copy |
| Phase B（reduce） | 合成为“输出域 Parallel + 归约轴 serial”的 nest，复用 `SerialIsClean`/`ClassifyReduce`；或新增 `ReducePlan` 复用 `EmitReduce`；先支持 max/sum/min | 覆盖 FA 的 reduce |
| Phase C | 独立 `CopyPlan`/`ReducePlan`，支持跨 scope、batch、复杂 layout | 完整能力 |

Phase A 的关键：**含 COPY/REDUCE 的 group 必须整组拒绝，不能只规划同组的 REGION**。否则 P1 的依赖闭包会被破坏。

### 6.2 CopyPlan 设计（Phase B/C）

**Phase B 已实现**（`polyhedral/group_plan.h` 的 `VSimdCopyPlan`）：

```text
VSimdCopyPlan {
  kind: "ND" | "BROADCAST",     // 由 src extents 是否有 1 决定
  src_scope, dst_scope,          // 只接受 UB（shared / shared.dyn）
  src_extents, dst_extents,
  program: VectorProgram,        // 合成 nest 的规划结果
}
```

做法：P2 直接合成

```text
for idx in T.Parallel(dst_extents):
    dst[dst_indices + idx] = src[src_indices + stride * idx]   // stride ∈ {0, 1}
```

再复用 `BuildPlan`/`EmitPlan`：

- src/dst 同 rank、逐维 extent 相等或 src extent 为 1（广播）；
- 目标 layout 取自同一 VF 的 REGION item（`CollectProgramLayouts`），NZ 目标仍由 `ClassifyStoreLayout` 处理；
- 非 UB 端（GM / FRAGMENT / L1）整组拒绝，交 legacy；
- 无 transpose/annotations 结构化字段，仍按 P1 payload 的 region 语义处理。

### 6.3 ReducePlan 设计（Phase B/C）

**Phase B 已实现**（`VSimdReducePlan`）：

```text
VSimdReducePlan {
  reduce_kind: "max" | "sum" | "min",
  dim, clear,
  init: VectorProgram?,          // clear=true 时的单位元初始化
  program: VectorProgram,        // 源域 Parallel 归约 nest
}
```

做法：

- `clear=false`：合成 `for idx in T.Parallel(src_extents):
  dst[dst_indices + idx'] = op(dst[...], src[src_indices + idx])`，
  由 `ClassifyReduce` 识别跨 lane 归约并发射 `vcadd/vcmax/vcmin`；
- `clear=true`：额外合成一个 `dst[...] = identity` 的 `init` program
  （`ReduceIdentity` 按 dtype 给出 ±inf / 0），P3 先发射 init 再发射归约，
  避免把“旧 dst 值”当成初值；
- dst 允许去掉归约维（rank-1）或保留该维且 extent 为 1；
- 两端必须都是 UB；`batch`/abs/bit 类归约继续整组拒绝。


---

## 7. 迁移与实施阶段

### Phase 0：接口与一致性基础设施

- 新增 `vsimd_handoff.{h,cc}`：`FlattenVSimdBody`、`SpanReplacer`、fingerprint；
- 新增 typed `ReadVSimdRegion/Copy/Reduce`；
- 新增 `VSimdGroupPlanSet` FFI 对象；
- P1 annotation 增加 `body_fingerprint` / `item_count`；
- 测试：P1/P2/P3 展平一致；`Simplify` 后仍一致；legacy lowering 后检测到失效。

### Phase 1：REGION-only 端到端

- 新增 `AscendPlanVSimd`：只规划 `P1_LIFTABLE` 且全 `REGION` 的 group；
- 新增 `AscendEmitVSimd`：预发射 + SpanReplacer + 整组回滚；
- 流水线改为 P1 → P2 → P3 → legacy；
- P2/P3 默认 off；打开后跑简单 1-D / 多维 / 广播 / 转置 / 多 store / 独立 Region 部分 lift；
- 含 COPY/REDUCE 的 group 整组拒绝，legacy 处理。

### Phase 2：fragment + 算术契约

- 消费 `program.fragments`，实现 v1 policy；
- VF 级 register mode 开关；
- `PlanOptions.assumptions/bindings/shape_syms` 接入 `BuildPlan`；
- `T.view` / `DeclBuffer` / wrapper 保留测试；
- 非连续 group 替换测试。

### Phase 3：COPY/REDUCE

- Phase B copy/reduce；
- FA softmax group 端到端；
- `clear=False` dst read、transpose、layout 目标；
- 仍不支持跨 scope copy，交 legacy。

### Phase 4：清理与收口

- 删除 `LiftParallelToArray` / `array_region` / `VSimdRegionRef`；
- `polyhedral/` 去掉对 P1 头文件的反向依赖；
- 删除死字段/死 API；
- 决定 legacy 是否永久保留；
- 更新测试与文档。

---

## 8. 能力差距分析

### 8.1 P1 放行场景 vs P2/P3 现状

符号：✅ 当前 P2/P3 有对应路径；△ 部分支持/Branch B；✘ 缺失。

| # | P1 放行场景 | P2/P3 现状 | 结论 |
| ---: | --- | --- | --- |
| 1 | 1-D 连续逐元素 | 支持 `vld/vsts` | ✅ |
| 2 | 仿射偏移 | 支持 | ✅ |
| 3 | 负步长/逆序写 | 支持 permute/strided + scatter | ✅（需实际验证） |
| 4 | 多维逐元素 | 支持外层轴 + 单向量轴 | ✅ |
| 5 | 广播读 | 支持 `uniform_load/vdup` | ✅ |
| 6 | 转置 | 支持 gather/scatter | ✅ |
| 7 | 步长读 | 支持 `vload_strided/gather` | ✅ |
| 8 | 间接读 | 支持 gather；索引向量不可构造时拒绝 | △ Branch B |
| 9 | 摊平/计算下标 | 支持 piecewise/strided | ✅ |
| 10 | 多 store | 支持多 store 段 | ✅ |
| 11 | 值级条件 | 支持 `vsel` | ✅ |
| 12 | 符号 extent + `T.assume` | 符号 extent 支持；`PlanOptions.assumptions` 是死字段 | △ 需接线 |
| 13 | 尾块 Bind | 原始语句路径可用；反展开路径会丢 Bind | △ 需改为原始语句规划 |
| 14 | `T.view` | P2 可规划；P3 重建 body 时会丢 `DeclBuffer` | △ 需 span 替换 |
| 15 | NZ layout 直接 store | 支持 `vsstb` | ✅ |
| 16 | 多版本/外层捕获 | 取决于 nest 形状 | △ |
| 17 | Region 内 fragment | G0–G8 部分支持；cross/outside 不支持 | △/✘ |
| 18 | 独立 Region 部分 lift | 当前 P2 无 group 事务 | ✘ 需新增 |
| — | 顶层 `T.copy` | 无独立规划/发射路径 | ✘ |
| — | 顶层 `T.reduce_*` | 无独立规划/发射路径 | ✘ |
| — | `BIND/ALLOC/DECL` | P3 只处理 carrier，不感知 item | △ 需保留 |

### 8.2 P1 放行但 P2/P3 不支持的缺口

| ID | 缺口 | 影响 | 优先级 | 方案 |
| --- | --- | --- | --- | --- |
| G1 | 载体不匹配：`vsimd.program` vs `array_region` | P2 完全看不到新 P1 | **P0** | 新 `PlanVSimd` 读 annotation；删除旧 lift |
| G2 | 无 group 事务 | 可能 producer 回退、consumer lift | **P0** | `VSimdGroupPlan` + 整组 plan/commit |
| G3 | span/body 映射不稳定 | 中间插 pass 后错误替换 | **P0** | 共享 `FlattenVSimdBody` + `body_fingerprint` |
| G4 | COPY 无规划/发射 | 含 copy 的 group 无法规划 | **P0** | Phase A 整组拒绝；Phase B 合成 Parallel nest |
| G5 | REDUCE 无规划/发射 | FA softmax 无法闭环 | **P0** | Phase A 整组拒绝；Phase B 合成归约 nest/`ReducePlan` |
| G6 | fragment `OUTSIDE_VF`/`cross_vf`/`escapes_vf` | P2 可能按 VF 规划，跨 VF 值丢失 | **P0** | v1 拒绝或 VF 级禁用 register mode；Phase C staging |
| G7 | fragment 被 COPY/REDUCE 使用 | P2 V1 escape scanner 整 VF 拒绝 | **P0** | copy/reduce 未规划前，含 fragment 的 group 整组拒绝 |
| G8 | group item 可能不连续 | P3 单区间替换会覆盖别的 group | **P0** | P3 按 item span 逐项替换 |
| G9 | 流水线顺序 | P1 后接 legacy 会改 body，span 失效 | **P0** | P1→P2→P3 连续；legacy 放 P3 后 |
| G10 | 旧 P2/P3 分支无 legacy fallback | 被拒 group 残留在 `T.Parallel` | **P0** | 恢复 legacy，放在 P3 后 |
| G11 | annotation 残留 | P3 后 annotation 指向旧 body | **P1** | P3 成功后删除；P2/P3 off 时 `StripVSimdProgram` |
| G12 | typed reader 不解析 region/copy/reduce | P2 需要自己解析 Map | **P1** | 增加 `ReadVSimdRegion/Copy/Reduce` |
| G13 | `PlanOptions.assumptions/bindings/shape_syms` 死字段 | P1 已证明的 assume 不能帮 P2 | **P1** | `BuildPlan` 接入 `Analyzer` constraint |
| G14 | `T.copy` transpose/annotations 未结构化 | P2 无法做通用 copy | **P1** | P1 payload 增加 `copy_transpose`/annotations |
| G15 | REDUCE `clear=False` 未记 dst read | 依赖组漏边 | **P1** | P1 补 read，或 P2 规划时补 |
| G16 | 程序级 captures 无 capture kind | P2 需要 param/alloc/view 时信息不足 | **P1** | 读 `region.captures`，或 P1 增加 kind |
| G17 | `AnalyzeVSimdRegionReport` per-VF vs pass per-nest | 诊断与 annotation 不一致 | **P1** | report 改用 per-nest，或标注以 annotation 为准 |
| G18 | 状态词表变化（无 FALLBACK/INTERNAL_ERROR） | 旧 P2 状态映射失效 | **P1** | 用 `status/effective_status` + `reason` |
| G19 | `validation_error` 不在 typed 结构 | P2 可能消费非法 program | **P1** | `has_validation_error()` |
| G20 | 零次迭代（extent=0） | P3 越界 | **P0** | P2 拒绝 `extent==0`，或 P3 返回空语句 |
| G21 | pack2 + 内层轴 | build 成功、emit 拒绝 | **P0** | `ValidatePackedNzSegment` 增加最内轴门禁 |
| G22 | pre/post 绕过段级门禁 | 窄化/NZ 可能越界 | **P0** | 段级校验统一扫 `pre ∪ stores ∪ post` |
| G23 | 归约初始化 `CopyOnWrite` 写丢失 | 结果多并入旧值 | **P0** | 修复 `FoldIdentityInit`（L1） |
| G24 | int16/uint16 `vcadd` 类型不一致 | 发射期失败 | **P1** | 拒绝该组合或先 `vcvt` 拓宽 |
| G25 | 非原子 `Evaluate` 静默丢弃 | 直接 API 会丢副作用 | **P1** | `NormalizeBodyInto` 抛 `Unsupported` |

### 8.3 反向缺口：P2 已有能力但 P1 不放行

这些不是 P2/P3 要补的，而是 **P1 语句模型限制**导致 P2 能力不可达。如果希望提升覆盖率，需要扩 P1，而不是改 P2。

| P2 能力 | P1 现状 | 影响 |
| --- | --- | --- |
| serial 吸收（`SerialIsClean`、pre/main/post） | `T.Parallel` 体内串行 `For` → PR-B unsupported | P2 的 serial 能力完全用不到 |
| 语句级 `if` 谓词化 | `IfThenElse` 在 `T.Parallel` 体内 → unsupported | P2 的 mask/predicate 能力用不到 |
| 体内 `Bind` 别名内联 | `Bind` 白名单移除 → unsupported | P2 的 alias 能力用不到 |
| 多段 `VectorProgram`（并列 serial 段） | P1 只 lift VF 顶层 Parallel；serial 包裹变 OPAQUE | P2 的多段能力用不到 |
| Parallel 内 TileOp/copy/reduce | PR-B 拒绝 | P2/P3 只能看到顶层 copy/reduce |
| 非 0 下界 | P1 可表达；P2 `RequireZeroMin` 拒绝 | 需 P1 归一化或 P2 放开 |

**设计建议**：v1 不扩 P1 语句模型；先把“P1 已放行”的场景在 P2/P3 打通。P1 扩 serial/if/Bind 应作为独立阶段，避免和 P2/P3 对接互相阻塞。

### 8.4 Branch B：P1 放行但 P2 物理拒绝

这些多数是**预期行为**，不是必须补的缺口；关键是要整组回退，而不是部分替换。

| P1 放行 | P2 拒绝点 | 处理 |
| --- | --- | --- |
| bool / 不支持 dtype | `MaskBits` / `_ELEM_BITS` | 保持拒绝，文档标注 |
| GM 操作数 | `LoadNode` | 前置 `VFChecker` 应拦截；P1 单独运行会误判 |
| 非 0 下界 | `RequireZeroMin` | 保持拒绝或 P1 归一化 |
| 读侧 true-random | `IndexVectorReason` | 保持拒绝（Branch B） |
| 沿向量轴 stencil | `CrossAddressRead` | 保持拒绝 |
| 私有标量不可提升 | `AccumulatorCandidate` | 保持拒绝 |
| 混合宽度/尾块/谓词 | `SelectSegmentLanes` 等 | 保持拒绝 |
| NZ 约束不满足 | `ClassifyStoreLayout` | 保持拒绝 |
| pack2 约束不满足 | `ValidatePackedNzSegment` | 保持拒绝；补 L3 门禁 |
| fragment G0–G8 不满足 | `ClassifyFragmentRegisters` | 保持拒绝；P1 不放宽 |
| 不支持的 reduce op/dim | `ClassifyReduce` | 保持拒绝 |

### 8.5 现有 P2/P3 缺陷对集成的影响

以下缺陷不是“新 P1 对接”引入的，但会影响新链路的正确性，必须在 P0 阶段一起收口：

| ID | 问题 | 对新链路的影响 |
| --- | --- | --- |
| L1 | 归约初始化 `CopyOnWrite` 写丢失 | 任何规划后的归约结果可能多加一份旧值 |
| L2 | 零次迭代计划在 P3 越界 | P1 可能放行 extent=0 的 Region |
| L3 | pack2 内层轴 build 成功、emit 拒绝 | 违反“P2 成功 ⇒ P3 成功”，组事务会回滚但诊断混乱 |
| L4 | pre/post 绕过段级门禁 | 窄化/NZ 可能越界 |
| L5 | 非原子 `Evaluate` 静默丢弃 | 旧离线 API 有风险；P1 group 通常不会包含 |
| L9 | int16/uint16 `vcadd` 类型不一致 | 可能发射期失败 |
| D3 | 传输层反向依赖 P1 头文件 | 新设计应把 handoff 移到 `vsimd_handoff`，`polyhedral/` 独立 |
| D4 | `VSimdRegionRef` 私有在 `.cc` | 新载体不再使用它，随旧 path 删除 |
| D5 | 双 driver（离线 pre-order / 三段 pass） | 新 P2 入口统一从 annotation 读，离线 API 只做测试 |
| D6 | `PlanOptions` 死字段 | G13 必须修 |

---

## 9. 待完善列表

### P0：合并前必须完成

| # | 任务 | 验收 |
| ---: | --- | --- |
| 1 | `vsimd_handoff`：共享展平 + `body_fingerprint` | P1/P2/P3 展平结果一致；插入 `Simplify` 后仍一致；legacy lowering 后检测到失效 |
| 2 | `PlanVSimd` 读 annotation + typed payload reader | 简单 1-D program 可读；`validation_error` 整 VF 拒绝 |
| 3 | `VSimdGroupPlan` + P2 group planner | 全 REGION group 成功；任一 item 失败整组 `REJECTED` |
| 4 | P3 span replacer + 整组回滚 | 非连续 group 正确替换；P3 失败整组原样；metadata/wrapper 保留 |
| 5 | 流水线重排 + legacy 恢复 + annotation 清理 | P1→P2→P3→legacy；最终 IR 无 `tl.vsimd.*` |
| 6 | COPY/REDUCE Phase A（整组拒绝） | 含 copy/reduce 的 FA group 不被部分替换 |
| 7 | Fragment v1 policy（outside/cross/escape 拒绝） | 跨 VF fragment 不会进入 register mode |
| 8 | 修复 L1–L4 | 归约写丢失、零次迭代、pack2 内层轴、pre/post 门禁 |

### P1：合并后第一轮完善

| # | 任务 | 验收 |
| ---: | --- | --- |
| 9 | `PlanOptions` 接入 assumptions/bindings/shape_syms | 带 `T.assume` 的符号 extent 用例 P2 可通过 |
| 10 | `T.copy` transpose/annotations 结构化 | copy payload 可表达 transpose |
| 11 | REDUCE `clear=False` 补 dst read | 依赖组不再漏边 |
| 12 | P3 保留 `DeclBuffer(view)` / `Bind` / `ALLOC` | `test_p1_view_region` 端到端通过 |
| 13 | 状态/诊断映射 + per-nest report 统一 | report 与 annotation 一致 |
| 14 | `AnalyzeVSimdRegionReport` 改用 per-nest | 多 nest 一非法一合法时诊断正确 |
| 15 | annotation 清理 pass（P2/P3 off 时） | 最终 IR 无 `tl.vsimd.program` |
| 16 | `T.copy`/`T.reduce` 的 storage scope 校验 | 跨 scope copy 明确拒绝或走 legacy |

### P2：能力扩展

| # | 任务 | 验收 |
| ---: | --- | --- |
| 17 | COPY Phase B：UB↔UB 合成 Parallel nest | 连续/广播/转置/NZ copy 可规划 |
| 18 | REDUCE Phase B：合成归约 nest 或 `ReducePlan` | FA softmax group 可规划 |
| 19 | 跨 VF fragment staging | `cross_vf/escapes_vf` 可桥接（或明确永久拒绝） |
| 20 | 跨 scope copy（UB↔L1、ND→NZ） | 独立 `CopyPlan` |
| 21 | P1 扩语句模型（serial/if/Bind/TileOp） | P2 已有能力可达 |
| 22 | `polyhedral/` 去 P1 反向依赖、typed `EmitPlan` | 可独立编译/单测 |
| 23 | 160 个旧测试适配新入口 | 直接 API 回归全绿 |

---

## 10. 验收与测试

### 10.1 新增契约测试

| # | 用例 | 断言 |
| ---: | --- | --- |
| 1 | `ReadVSimdProgram` + `FlattenVSimdBody` | item 数与 span 一致；fingerprint 一致 |
| 2 | 全 REGION `P1_LIFTABLE` group | `PLANNED`；P3 后出现 `tl.simd.*` |
| 3 | group 中一个 REGION 被 P2 拒绝 | 整组 `REJECTED`；group 内语句逐字节不变 |
| 4 | group 含 COPY/REDUCE | Phase A 整组拒绝；不被部分替换 |
| 5 | group item 非连续 | P3 按 item 逐项替换，不覆盖别的 group |
| 6 | P3 发射失败 | 整组回滚；其它 group 不受影响 |
| 7 | `DeclBuffer(view)` / `Bind` / `AllocBuffer` | P3 后仍存在且结构一致 |
| 8 | assumptions/attrs | P3 后 wrapper 仍在原位 |
| 9 | body fingerprint 不匹配 | 整 VF 不被 P2/P3 处理，legacy 兜底 |
| 10 | `validation_error` | 整 VF 被拒绝 |
| 11 | `cross_vf`/`escapes_vf`/`OUTSIDE_VF` fragment | 按 v1 policy 拒绝或禁用 register mode |
| 12 | P3 成功 | `tl.vsimd.program` / `tl.vsimd.group_plan` 均不存在 |
| 13 | P2/P3 关闭 | legacy 正常；最终 IR 无 `tl.vsimd.program` |
| 14 | 流水线顺序 | P1→P2→P3→legacy；中间无 body-mutating pass |

### 10.2 回归

- P1 现有 72 个测试保持通过；
- P2/P3 现有 160 个直接 API 测试保持通过；
- 旧的 staged pass 测试改为新入口：`array_region` → `vsimd.program`，`array_plan` → `group_plan`；
- 新增一个“组合 pass 与离线 API 等价”的护栏测试。

### 10.3 性能

- P1 默认 on 的编译时间：无 Parallel 零成本；有 Parallel 的增量需要 benchmark；
- P2/P3 opt-in 后，group 规划不应重复跑 PR-A/Z3；
- 不建议同时保留新 P1 与 `BuildP2PlanInputs` 的完整分析。

---

## 11. 风险与待确认

| # | 问题 | 建议 |
| ---: | --- | --- |
| 1 | P2 规划原始语句还是 region 反展开？ | **原始语句**；region 只做上下文/校验，`RegionToPlanInput` 降级为离线 API |
| 2 | group plan 用 typed FFI 还是 Map？ | **typed FFI**；P3 直接消费，测试可断言 |
| 3 | COPY/REDUCE 第一版做不做？ | **第一版整组拒绝**，先打通 REGION-only 端到端；Phase B 再做 |
| 4 | legacy 是否保留？ | **保留**，至少到 COPY/REDUCE 和 fragment staging 完成 |
| 5 | P1 是否扩 serial/if/Bind？ | 单独阶段；不要和 P2/P3 对接耦合 |
| 6 | 跨 VF fragment 永久拒绝还是 staging？ | v1 拒绝；有真实算例再 staging |
| 7 | annotation 是否进入最终 codegen？ | 不进入；P3 后删除，P2/P3 off 时 cleanup pass 删除 |
| 8 | group item 不连续是否要在 P1 修正？ | P3 按 item 替换是更通用的方案；P1 可选增加连续性诊断 |
| 9 | `PlanOptions` 死字段是否本轮修？ | 是；否则 P1 的 `T.assume` 对新链路无效 |
| 10 | 旧 160 个测试是否改写？ | 直接 API 保留；staged 测试适配新载体，不降低断言强度 |

---

## 附录 A：group plan 示例

一个 VF 有两个 group：group 0 全 REGION 被规划，group 1 含 COPY 被拒绝。

```text
SBlock("SIMD_VF").annotations["tl.vsimd.group_plan"] = {
  version: 1,
  vf_index: 0,
  groups: [
    {
      group_id: 0,
      status: "PLANNED",
      reason: "",
      diagnostics: [],
      items: [
        {
          item_index: 0,
          span_begin: 0, span_end: 1,
          kind: "REGION",
          plan_kind: "REGION",
          region_plan: VectorProgram { axes: [...], segments: [...] }
        },
        {
          item_index: 2,
          span_begin: 2, span_end: 3,
          kind: "REGION",
          plan_kind: "REGION",
          region_plan: VectorProgram { ... }
        }
      ]
    },
    {
      group_id: 1,
      status: "REJECTED",
      reason: "COPY/REDUCE plan not implemented",
      diagnostics: [
        "item 1: COPY m_old -> m_new",
        "item 3: REDUCE max S_frag -> m_new"
      ],
      items: []
    }
  ]
}
```

P3 只提交 group 0 的 item 0/2；group 1 的 item 1/3 原样保留，交 legacy。

## 附录 B：文件与函数索引

| 文件 | 作用 | 本设计中的变化 |
| --- | --- | --- |
| `src/ascend/transform/lift_vsimd.cc` | 新 P1 pass | 抽出 `FlattenVSimdBody`；增加 `body_fingerprint/item_count` |
| `src/ascend/transform/vsimd_program.{h,cc}` | typed reader/verifier | 增加 `ReadVSimdRegion/Copy/Reduce`、`has_validation_error` |
| `src/ascend/transform/vsimd_handoff.{h,cc}` | 新 handoff 层 | 展平、span 替换、fingerprint |
| `src/ascend/transform/vsimd_group_plan.{h,cc}` | 新 group plan 对象 | FFI 反射 + P2/P3 共用 |
| `src/ascend/transform/parallel_vectorize/stage_passes.cc` | 旧 P2/P3 driver + 载体 | 改为读 annotation / group plan；删除旧 region 载体 |
| `src/ascend/transform/parallel_vectorize/vsimd_p2_input.cc` | 旧 P1→P2 交接 | 降级为离线/兼容 API |
| `src/ascend/transform/parallel_vectorize/plan.cc` | P2 规划 | 接入 assumptions/bindings；补 L1–L4 |
| `src/ascend/transform/parallel_vectorize/emit.cc` | P3 发射 | 支持 group 级发射；补 L2/L3 |
| `tilelang/ascend/pipeline.py` | 流水线 | P1→P2→P3→legacy；P2/P3 开关 |
| `tilelang/ascend/transform/__init__.py` | Python API | `AscendPlanVSimd` / `AscendEmitVSimd` / `AscendVectorizeVSimd` |

## 附录 C：参考文档

- `docs/compiler_internals/vsimd_p1_analysis.md`：新 P1 架构、输出契约、放行场景、annotation 打印示例；
- `docs/compiler_internals/p2_p3_cpp_port_survey.md`：P2/P3 现有实现、字段契约、边界与缺陷；
- `docs/compiler_internals/p2_p3_design_implementation_report.md`：P2/P3 设计—实现汇报、能力与待优化项；
- `debugging/docs/p1_vsimd_interface.md`（vsimd-frag 分支）：P1→P2/P3 handoff 接口文档；
- `debugging/docs/vsimd_p1_next_plan.md`（vsimd-frag 分支）：P1 阶段计划与 Phase 1 实现记录。

## 附录 D：当前实现状态

实现分支：`p_123_parallel`（基于 `p2_p3_based_stable-parallel`，保留其未提交的 P2.5 fragment register 改动）。

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 新 P1 源码合入 | ✅ | `lift_vsimd.cc`、`vsimd_program.{h,cc}`、PR-A/B 增量 |
| `vsimd_handoff` 展平 / 指纹 / span 替换 | ✅ | `FlattenVSimdBody`、`FingerprintVSimdBody`、`ReplaceVSimdSpans` |
| P1 写入 `item_count` / `body_fingerprint` | ✅ | P2/P3 会做一致性校验 |
| typed `ReadVSimdRegion/Copy/Reduce` | ✅ | `vsimd_program.{h,cc}`；P2 不再自己解析 payload map |
| typed `VSimdGroupPlan` FFI | ✅ | `polyhedral/group_plan.{h,cc}`：`VSimdGroupPlanSet/GroupPlan/ItemPlan/CopyPlan/ReducePlan` |
| P2 `PlanVSimd`（REGION/COPY/REDUCE） | ✅ | group 事务；任一 item 失败整组 `REJECTED` |
| COPY Phase B | ✅ | UB↔UB、等 extent / 广播；跨 scope 拒绝 |
| REDUCE Phase B | ✅ | max/sum/min；`clear=false` 累加、`clear=true` 先发射单位元 init |
| P3 `EmitVSimd` | ✅ | 预发射、整组回滚、按 item span 替换、清理 annotation |
| fragment VF register 规划 | ✅ | 从旧 P2.5 抽到 `polyhedral/fragment_plan.{h,cc}`，由 P2 调用、P3 消费 `fragment_regs` |
| 跨 VF fragment（OUTSIDE_VF） | ✅ | P2 直接拒绝该 group，理由含 “allocated outside this VF” |
| `PlanOptions.assumptions/bindings/shape_syms` 接线 | ✅ | `BuildPlan` 进入 `Analyzer` constraint / `Bind` |
| L1/L2/L3 修复 | ✅ | 归约初始化写回、零 extent 拒绝、pack2 最内轴门禁 |
| L4 修复 | ⏳ | pre/post 尚未纳入段级 lane/NZ/pack2 校验 |
| 旧 `array_region` / `LiftParallelToArray` 清理 | ✅ | `stage_passes.{h,cc}` 删除；新链路为默认路径 |
| 默认路径切换 | ✅ | pipeline 固定 P1 → P2 → P3 → `AscendSimdVFLowerParallel` |
| `TL_ASCEND_VSIMD_VECTORIZE` 开关 | ✅ 移除 | 新路径恒开；`TL_ASCEND_SIMD_LIFT=False` 即整体回退 legacy |
| 回归 | ✅ | `testing/ascend/transform` 598 例 + `auto_schedule` 全绿 |

---

> 一句话总结：**新 P1 负责“逻辑事实 + 事务边界 + 可验证的 annotation”；P2 负责“按 group 决定能不能做、怎么做”；P3 负责“按 item 生成 IR 并原子替换”。三者唯一的契约是 `tl.vsimd.program` / `tl.vsimd.group_plan` 与共享的 `VSimdFlattenedBody`。**
