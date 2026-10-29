# P1 输出完备性审计与 P2 能力路线选型

> 依据文档：`/home/cjl/workspace/tilelang-stable/docs/compiler_internals/vsimd_p1_analysis.md`（2026-10-27 更新版，1270 行，新增第 8 章“与多面体向量化 66 场景的对照”）。
> 对照实现：`/home/cjl/workspace/tilelang` 分支 `Parallel_lowering` 的 `tilelang/ascend/analysis/polyhedral/`。
> 前置约束：**暂不扩展 P1 阶段功能**；P2 输入定义为 **P1 输出 + 前置 Pass 输出的 TIR 中间表示**。

---

## 0. 结论摘要

| 问题 | 结论 |
| --- | --- |
| Q1：P1 输出文本是否预留了控制流 / 归约 / 多层 parallel 的标识、扩展字段或接口？ | **没有输出级预留。** 只有三处“非输出级”的口子：① `NestCollector::allow_control_flow_` 代码开关（默认 false，为 CF1–CF3 预留，不进入任何报告字段）；② `VSimdRegion` 头文件注明“未来可执行载体将直接消费 VSimdRegion”，但没有 schema/字段；③ `domain` 是 `vector<LoopInfo>`，数据模型上可放多层，但 PR-C 的 G3 强制 `size()==1`，且 `LoopInfo` 只有 `var/min/extent`，没有 `kind`、轴角色、向量轴等属性。归约连状态位都没有：`Legality` 无 `REDUCTION`，`VSimdRegion` 无归约描述。 |
| Q2：仅凭 P1 输出能否完整完成 P2/P3 全部任务？ | **不能。** 文本输出连 P2 基线都不够（缺 dtype/shape/strides、`bindings`、语句结构与谓词）；类型化 `VSimdRegion` 覆盖 66 场景中的 18 个可 lift（其中 15 个与 P2 的 OK 集交集），另有 5 个“可分析但不 lift”的多层直线场景可由 **PR-B `VSimdCaptureInfo`** 重建（合计 20 个可规划），仍有 43 个被拒（归约/语句级控制流/串行）。P3 的输入本来就是 P2 的 `VectorProgram`，P1 单独无法提供 lane/chunk/尾块/掩码/recipe/指令选择。 |
| Q3/Q4：两条路线怎么选？ | **字面路线 2 只有两个可用触发**：基础逐元素（`LEGAL` + 单层）与多层直线（`LEGAL` + `CLOSED` + `domain.size()>1` + G3 fallback）；归约/语句级控制流/serial/多段/守卫没有正向触发，按“没有触发就跳过”执行时，路线 2 的运行时能力面与“接入 PR-B 桥的路线 1”**完全相同**（20/55）。建议采用 **路线 2 的工程修正版**：完整保留 P2 代码资产，但把触发器改为 P2 侧显式、版本化的 `P1CapabilitySet`（当前只声明 v1 能力，含 `multi_level_straightline`），高级能力位缺省关闭；任何被规划的 nest 仍必须通过 P1 `LEGAL` 门禁。若团队要求“代码最少、一次交付”，则选路线 1，但路线 1 也必须显式接入 `VSimdCaptureInfo` 多层直线桥，否则只剩 15 个场景。 |

**代码级核实修正（基于 `vsimd_region.cc` 的逐行核对，重要）**：

1. `AnalyzeVSimdRegionReport`（[vsimd_region.cc#L298-L340](file:///home/cjl/workspace/tilelang-stable/src/ascend/transform/vsimd_region.cc#L298-L340)）是 P1 的**组合报告入口**，不是 typed 中间表示的“唯一构造器”。typed `VSimdRegion` 由 `BuildMinimal1DRegion`（[L44-L101](file:///home/cjl/workspace/tilelang-stable/src/ascend/transform/vsimd_region.cc#L44-L101)）构造；L298 只是调用它并把结果渲染成字符串。
2. `ExpandVSimdRegionToLoops`（[L153-L179](file:///home/cjl/workspace/tilelang-stable/src/ascend/transform/vsimd_region.cc#L153-L179)，即 L326 调用的函数）是**参考展开器**：它把 region 重建成**标量串行循环**，用途是自检与 `expanded:` 文本；它不是 P2 输入，也不是 P3 发射，更不会改写原始 `SIMD_VF` nest。代码注释原文：“Reference expansion is intentionally a serial scalar loop; it is a semantic oracle, not the P3 physical lowering.”
3. P1 的三个 Pass 全部是**只读**（`return f`），不在默认流水线中；P1 判定 `LIFTED` 只表示“逻辑上可向量化”，不产生任何 IR 改写。
4. 现有 P2/P3 **不能“直接承接”** `VSimdRegion`：P2 的 `build_plan(stmt)` 接收原始 TIR `For`，P3 接收 `VectorPlan/VectorProgram`；P1 与 P2/P3 之间没有适配器，也没有可执行的 P1→P2 载体。
5. 因此“P2/P3 的全部原有能力都可以完整保留”只能理解为**代码资产保留**；**运行时能力**只能激活 P1 v1 覆盖的 20/55 场景。详见 §6。

---

## 1. P1 输出文本：完整结构、字段定义与元数据范围

### 1.1 三个输出面

P1 不是一个单一输出，而是三个 Pass 各自产出报告，另有 C++ 类型化结构：

| 输出面 | 入口 | 形态 | 是否机器可读 | 备注 |
| --- | --- | --- | --- | --- |
| PR-A 内存合法性 | `AnalyzeVSimdMemoryLegality(PrimFunc) → String`；`AnalyzeVSimdMemoryStatuses → vector<string>` | 文本 + 每 VF 状态数组 | 半结构化（状态数组可读，细节只在文本） | 状态是 **每 VF 一个** |
| PR-B 捕获/作用域 | `AnalyzeVSimdCaptureScope → String`；`AnalyzeVSimdCaptureInfo → vector<VSimdCaptureInfo>`；`CaptureInfoToFFI → Array<Map<String,Any>>` | 文本 + 类型化结构 + FFI Map | 是（FFI Map 字段全部字符串化） | 每 nest 一条 |
| PR-C 区域 lift | `AnalyzeVSimdRegionReport → String`；`BuildMinimal1DRegion → VSimdRegionLift` | 文本 + C++ `VSimdRegion` | C++ 可读，**无 Python FFI** | 只有 `LIFTED` 才产出 region |

文档 §3.5 明确三者的统一形态：“Report：纯函数，直接返回字符串；Pass：只读装饰器；静态注册”。文档 §3.4.5 明确 region 的载体是**不可执行**的：

> “载体是一个**不可执行**的 AttrStmt：key 为 `tl.vsimd.region`，node 为完整 dump 文本，body 为 `Evaluate(0)`。它只出现在 `AnalyzeVSimdRegionReport` 的文本输出中，从不注入 IR——注释直言‘the executable P2/P3 carrier will consume VSimdRegion directly’。”

**跨语言可用性提示**：PR-B 的 FFI 输出 `CaptureInfoToFFI` 把字段全部字符串化（文档 §3.3.5），PR-C 的 `VSimdRegion` **没有 Python FFI**。因此：

- C++ 侧的 P2/P3 可以直接消费 `std::vector<VSimdCaptureInfo>` / `VSimdRegion`；
- Python 侧的 P2 只能拿到字符串化 Map 或报告文本，无法直接拿到 `Buffer/PrimExpr`；
- 若 P2 要保持 Python 实现，必须新增 typed FFI（把 `Buffer/PrimExpr/Var` 原样放进 `Any`），或解析字符串——后者不可作为正式契约。

### 1.2 文本输出逐行结构

#### 1.2.1 PR-A 内存报告（文档 §3.2.9）

文档给出的行结构：

```
VF#i status=<LEGAL|UNPROVEN|ILLEGAL|UNSUPPORTED|OPAQUE>
nest k: domain=... accesses=total/unique queries=n cache_hits=n [budget=exceeded]
R/W name[indices] class=<affine|quasi-affine|general|data-dependent> layout=none|changing injective=yes|no (reason)
local_views: ...
message: ...
```

元数据范围：

- `status` 是**每 VF**的聚合态；`nest k` 行有 `domain`、去重计数、Z3 查询数、缓存命中数、预算超限标记；
- 访问行有读写标记、buffer 名、索引表达式、映射分类、`layout_changing` 标记、单射结论与反例/原因；
- `local_views` 与 `message` 是文本；**没有** assumptions、bindings、dtype、shape/strides、storage scope、lane。

#### 1.2.2 PR-B 捕获报告（文档 §3.3.5 + 源码 `RenderVSimdCaptureReport`）

文档给出的字段顺序：

> “`RenderVSimdCaptureReport`：按 `vf_index` 变化输出 `VF#i status=...` 头，再逐 nest 输出 `domain` / `captures` / `outer_vars` / `shape_syms` / `local_views` / `layouts` / `assumptions` / `errors`”。

实际文本（源码 `vsimd_capture_scope.cc:526-589`）：

```
VF#<i> status=<CLOSED|OPEN|OPAQUE>
  nest <k>: domain=<var> in [<min>, <min+extent>) ...
    captures: name[kind] ...
    outer_vars: name[kind] ...
    shape_syms: name ...
    local_views: view -> base ...
    layouts: name ...
    assumptions: <expr> ...
    errors: <error> ...
```

关键观察：

- `domain` 是**逐层打印**的（`for (const LoopInfo &loop : info.domain)`），因此 PR-B 文本可以显示多层并行域；
- `captures` 只打印 `name[kind]`，**不打印** buffer 的 shape/strides/elem_offset/dtype/scope；
- `layouts` 只打印 buffer 名，**不打印布局映射**；
- **`bindings` 不在文本里**：`VSimdCaptureInfo` 有 `bindings` 字段，`CaptureInfoToFFI` 也把它放进 Map（源码 `CaptureToMap` 第 159 行），但 `RenderVSimdCaptureReport` 没有输出它；
- `errors` 是自由文本，没有错误码；
- 空输入返回 `no SIMD_VF blocks\n`。

#### 1.2.3 PR-C 区域报告（文档 §3.4.5 + 源码 `AnalyzeVSimdRegionReport` / `DumpVSimdRegion`）

报告拼装顺序（文档 §3.4.5）：

```
VF#i nest k: status=FALLBACK reason=...          // 未 lift
VF#i nest k: status=INTERNAL_ERROR reason=...    // 展开器自检失败
VF#i nest k: status=LIFTED
region VF#i nest#k
  domain: <var> in [<min>, <min+extent>) ...
  captures: name[kind] ...
  outer: name[kind] ...
  shape_syms: name ...
  local_views: view -> base ...
  layouts: name ...
  assumptions: <expr> ...
  accesses:
    R name[indices]
    W name[indices] = <value>
  expanded:
    <参考展开后的标量串行循环>
  verify: OK
  carrier:
    <AttrStmt(...)>
```

与 PR-B 的差异：

| 项 | PR-B 文本 | PR-C region dump |
| --- | --- | --- |
| `outer_vars` 字段名 | `outer_vars:` | `outer:` |
| `bindings` | 无 | **无**（`DumpVSimdRegion` 没有输出） |
| 访问分类 `class` | PR-A 文本有 | **无** |
| `layout_changing` | PR-A 文本有 | **无** |
| `errors` | 有 | 无（lift 成功才输出） |
| `expanded` / `verify` / `carrier` | 无 | 有 |
| 读写顺序 | 源码顺序 | 源码顺序（`accesses` 原样） |
| store 谓词/mask | 无 | 无 |

> 注意：`expanded` 是“语义 oracle，而非 P3 物理 lowering”（文档 §3.4.3），它只是把 region 重建成标量串行循环；`carrier` 是不可执行 AttrStmt。两者都不能作为 P3 的输入。

### 1.3 字段与元数据范围总表

| 类别 | 字段 | 文本可见 | 类型化可见 | 语义/范围 |
| --- | --- | --- | --- | --- |
| 定位 | `vf_index`, `nest_index` | ✅ | ✅ | VF 序号 / VF 内 nest 序号；`nest_index=-1` 表示整 VF 不可分析 |
| 状态 | `vf_status`, `status`, `message` | ✅ | ✅ | `CLOSED/OPEN/OPAQUE`；PR-A 另有 5 态 |
| 迭代域 | `domain[].var/min/extent` | ✅（`var in [min, min+extent)`） | ✅ | **只有并行层**；无 `kind`、无轴角色 |
| 访存 | `accesses[].buffer/indices/value/is_write` | ✅（buffer 名 + 索引 + 写值） | ✅（完整 Buffer/PrimExpr） | 源码顺序；读写在一条扁平列表 |
| 捕获 | `captures[].buffer/kind` | ✅（`name[kind]`） | ✅（Buffer 对象） | `param/alloc/local/view` |
| 外层变量 | `outer_vars[].var/kind` | ✅ | ✅ | `thread/outer_loop/bind` |
| 形状符号 | `shape_syms[]` | ✅ | ✅ | 自由符号名 |
| 视图 | `local_views[]` | ✅（`view -> base`） | ✅ | 只给名字，不给 offset/shape |
| 布局 | `layouts[]` | ✅（仅 buffer 名） | ✅（`Layout` 对象） | 文本不含映射细节 |
| 假设 | `assumptions[]` | ✅ | ✅ | 纯布尔算术；已过滤 |
| 绑定 | `bindings[]` | ❌ | ✅ | 只给 `Var -> PrimExpr` |
| 错误 | `errors[]` | ✅ | ✅ | 自由文本，无错误码 |
| 合法性 | PR-A `status` + 访问行 `class/layout/injective` | ✅（PR-A 文本） | ❌（region 不含 class/injective） | 每 VF 聚合；region 不携带逐访问分类 |
| 指令/物理 | lane/chunk/tail/mask/recipe/指令 | ❌ | ❌ | 明确声明“不含物理布局、lane 或指令选择” |

### 1.4 预留标识 / 扩展字段 / 预留接口审计

逐一排查文档与代码中所有可能被误认为“为多层 parallel / 控制流 / 归约预留”的内容：

| 候选 | 位置/原文 | 它实际预留了什么 | 是否进入输出文本/可查询 | 判定 |
| --- | --- | --- | --- | --- |
| `NestCollector::allow_control_flow_` | 文档 §3.1.3：`For(kSerial)` 行“仅 `allow_control_flow_` 时递归；**为未来 CF1–CF3 预留开关，默认关闭**”；§3.1.3 末：“`allow_control_flow_` 默认 `false`……属于为 CF 系列预留的守卫” | 一个**代码级构造参数**，允许 visitor 递归 serial/IfThenElse | ❌ 不进入任何报告字段；P2 无法从输出判断它是否开启 | **不是输出级预留**；不能作为 P2 的触发标识 |
| `domain` 的 `vector<LoopInfo>` | 文档 §3.1.1：`ParallelNest::loops` 按“外层→内层”排列；“决定了多级 nest 的字典序语义”；PR-B `domain` 为 `std::vector<LoopInfo>` | 数据模型上可容纳多层并行 | ✅ PR-B 文本逐层打印；PR-C 只在 `size()==1` 时 lift | **仅数据模型可容纳**；PR-C 输出不支持多层 |
| `LoopInfo` 字段 | 文档 §3.1.1：`struct LoopInfo { Var var; PrimExpr min; PrimExpr extent; };` | 只有 var/min/extent | ✅ 文本/类型化 | **没有 `kind`**（无法区分 parallel/serial）、没有轴序号、轴角色、向量轴候选 |
| `VSimdRegion` 的“未来可执行载体” | 文档 §3.4.5 引头文件：“the executable P2/P3 carrier will consume VSimdRegion directly”；§7.6：“可执行载体 待实现；当前只有 debug 载体 `tl.vsimd.region`” | 一个**未来接口承诺** | ❌ 当前没有实现、没有 schema、没有 FFI | **只是注释级预留**，不能消费 |
| `tl.vsimd.region` AttrStmt | 文档 §3.4.5 / §3.4.4 | debug dump 文本 | ✅ 只在报告中出现 | **不可执行**；不是载体 |
| `Legality` 枚举 | 文档 §3.2.1：`kLegal/kUnproven/kIllegal/kUnsupported/kOpaque` | 5 态合法性 | ✅ 状态字符串 | **无 `REDUCTION` 状态**；无控制流/serial 状态 |
| `MapClass` / `Relation` | 文档 §3.2.1：`kAffine/kQuasiAffine/kGeneral/kDataDependent`；`kSafe/kViolation/kUnknown` | 映射分类与配对关系 | 只在 PR-A 文本 | 与归约/控制流无关 |
| `AccessInfo.value` | 文档 §3.1.1：“只在写访问时有意义——让 PR-C 的参考展开器无需重新扫描 IR 就能复现存储表达式” | store 值表达式 | ✅ PR-C 文本 `W ... = value` | **没有 mask/predicate**；不能表达条件写 |
| `bindings` | 文档 §3.3.1：“相关的 `Bind` 定义”；`CaptureInfoToFFI` 含 `bindings` | 尾块/符号边界表达式 | ❌ PR-B/PR-C **文本均不打印**；类型化/FFI 可见 | 文本契约缺失；类型化可用 |
| `assumptions` | 文档 §3.3.1 | 纯布尔算术契约 | ✅ 文本/类型化 | 可用于 P2 证明，但不是能力触发 |
| `layouts` | 文档 §3.3.1：“P2/P3 决定 `vsts` vs `vsstb`” | 布局对象 | 文本仅名字；类型化完整 | 与多层/CF/归约无关 |
| `layout_changing` | 文档 §3.2.8 / §3.2.9 | 写访问是否改布局 | ✅ 仅 PR-A 文本 | 不在 region 中；不是能力触发 |
| `thread_bounds` | 文档 §3.3.1：“**不设**……PR-C/P2/P3 可从函数/模块读取” | 刻意不设 | ❌ | **P1 输出不提供**；P2 若要边界证明必须回读函数/模块，违反“仅 P1 输出” |
| `versions` | 文档 §3.3.1：“**不设**……version 是 buffer 的首维” | 刻意不设 | ❌ | 由 buffer shape 隐式表达 |
| `CaptureInfoToFFI` | 文档 §3.3.5：`Array<Map<String, Any>>`，字段全部字符串化 | per-nest 结构化 Map | ✅ | **没有 schema_version、没有 capability/feature 字段、没有扩展位** |
| `status` / `message` / `errors` | 各报告 | 自由文本 | ✅ | 没有机器可读错误码；不能作为正向能力标识 |

### 1.5 Q1 判定结论

**结论：P1 输出没有为“控制流 + 归约 + 多层 parallel 属性”预留任何输出级的标识、扩展字段或可查询接口。**

分项判定：

1. **多层 parallel 属性**：
   - `domain` 的容器类型是 vector，PR-B 文本会逐层打印——这是**唯一**的多层“痕迹”；
   - 但 `LoopInfo` 没有 `kind`，无法表达 serial；PR-C 的 G3 强制 `domain.size()==1`（原文：“PR-C minimal lift only supports one T.Parallel level”），因此**可消费的 region 永远只有一层**；
   - 没有轴序号、轴角色（外层串行轴 / 向量轴）、lane/chunk/tail 等属性。
   - 判定：**只有“可容纳多层的容器”，没有“多层可执行属性”。**

2. **控制流**：
   - 只有 `allow_control_flow_` 这个**代码开关**，原文明确它是“为未来 CF1–CF3 预留开关，默认关闭”；
   - 它不进入任何状态/文本/FFI 字段；`IfThenElse` 在默认路径直接被拒（`UNSUPPORTED`/`OPAQUE`）；
   - `AccessInfo` 没有 mask/predicate；`VSimdRegion` 没有分支/守卫/边界字段。
   - 判定：**没有输出级控制流标识；只有未启用的代码级开关。**

3. **归约**：
   - `Legality` 没有 `REDUCTION`；`VSimdRegion` 没有 `ReduceInfo`；`AccessInfo` 没有 reduce op/axis/identity/accumulate；
   - 文档 §8.3.2 原文：“这条差异不是‘还没实现’，而是**判定模型本身不包含归约语义**（`Legality` 没有 `REDUCTION` 状态，`VSimdRegion` 也没有归约描述）。”
   - 判定：**完全没有预留，连状态位都没有。**

4. **通用扩展机制**：
   - 没有 `schema_version`、没有 `capabilities`/`features` 位图、没有 `extensions` map、没有 `reserved` 字段、没有错误码；
   - 所有状态/原因都是自由字符串。
   - 判定：**没有版本化/能力化的扩展接口；只有少量辅助元数据（assumptions/bindings/layouts/local_views）。**

**原文依据清单**（均可直接在 `vsimd_p1_analysis.md` 中检索）：

| 结论 | 原文 | 位置 |
| --- | --- | --- |
| serial/if 只有代码开关，默认关闭 | “`For(kSerial)`｜仅 `allow_control_flow_` 时递归｜**为未来 CF1–CF3 预留开关，默认关闭**” | §3.1.3 白名单表 |
| 开关不进生产路径 | “`allow_control_flow_` 默认 `false`……属于为 CF 系列预留的守卫” | §3.1.3 末 |
| region 只支持单层 | “PR-C minimal lift only supports one T.Parallel level” | §3.4.2 G3 |
| region 不是可执行 IR | “它是 P1→P2/P3 的**逻辑交接**，不是可执行 IR” | §3.4.1 |
| 可执行载体未实现 | “the executable P2/P3 carrier will consume VSimdRegion directly”；“可执行载体 待实现；当前只有 debug 载体 `tl.vsimd.region`” | §3.4.5、§7.6 |
| 归约无状态位 | “`Legality` 没有 `REDUCTION` 状态，`VSimdRegion` 也没有归约描述” | §8.3.2 |
| 语句级控制流一律拒绝 | “P1 对控制流只有一条判据——**语句级 `if` 一律拒绝，值级条件表达式一律放行**” | §8.3.4 |
| 串行维度一概不接受 | “P1 的模型假设‘并行域之外无迭代’，串行维度一概不接受” | §8.3.5 |
| 多层只是容器可容纳 | “`ParallelNest::loops` 中‘外层→内层’的顺序决定了多级 nest 的字典序语义”；`LoopInfo` 仅 `var/min/extent` | §3.1.1 |
| `thread_bounds`/`versions` 刻意不设 | “**`thread_bounds` 不设**……”“**`versions` 不设**……” | §3.3.1 |

---

## 2. 仅凭 P1 输出完成 P2/P3 的可行性

### 2.1 P2 / P3 任务需求清单

| ID | 阶段 | 任务 | 所需信息 |
| --- | --- | --- | --- |
| P2-01 | P2 | 收集并行迭代域，支持 N 层 | 每层 var/min/extent + 层数 |
| P2-02 | P2 | 吸收直接内嵌/夹在 SeqStmt 中的干净 serial 轴 | serial 循环结构、循环变量、`_serial_is_clean` 依赖判定 |
| P2-03 | P2 | body 规范化：多语句、Bind 别名、if 谓词化、同址 if/else 惰性合并 | body 语句树（Seq/If/For/Bind）、别名、谓词 |
| P2-04 | P2 | 拆 pre/main/post（§6 第二步） | 语句分组与顺序 |
| P2-05 | P2 | 多段 program（≥2 个干净 serial 段） | 段边界、源序 |
| P2-06 | P2 | 逐轴 Δ 分类：CONTIGUOUS/UNIFORM/STRIDED/PERMUTE/PIECEWISE/INDIRECT/TRUE_RANDOM | buffer shape/strides/elem_offset + indices + 域 |
| P2-07 | P2 | 轴排序与回退（联合可行性 + 读写加权） | 每个 store/load 的逐轴分类 |
| P2-08 | P2 | lane 数推导（`2048/elem_bits`） | store 元素 dtype |
| P2-09 | P2 | chunks/tail（常量/符号 extent） | extent + lanes |
| P2-10 | P2 | 值 DAG：load/uniform/elementwise/select/cast/索引向量 | store value 表达式 + buffer 元数据 |
| P2-11 | P2 | 归约识别与计划：Δ==0 store + 取值变化 + op Add/Max/Min + identity + accumulate | 写地址、取值、旧值、dtype |
| P2-12 | P2 | 短/长归约策略、尾块单位元填充 | chunks/tail + identity |
| P2-13 | P2 | 装载守卫与 `guard_boundaries`（D16） | 语句级谓词 + 地址边界证明 |
| P2-14 | P2 | 累加器提升（`local`/`local.var` RMW） | buffer scope + RMW 结构 |
| P2-15 | P2 | 诊断：成功/失败都收集所有候选轴与原因 | 结构化诊断出口 |
| P2-16 | P2 | `VectorProgram.axes` 只含并行轴前缀 | 轴角色（并行/串行/向量） |
| P3-01 | P3 | recipe → `tl.simd.*` 指令 | 每个 ValueNode 的 recipe/dtype/args |
| P3-02 | P3 | 地址构造：`access_ptr`、offset、NORM/ONEPT/BRC | buffer/indices/elem_offset/lanes |
| P3-03 | P3 | 掩码：全 lane/尾块/谓词/lane0 | lanes/tail/谓词 |
| P3-04 | P3 | chunk strip-mine + 外层 serial 循环 | chunks/tail/axes |
| P3-05 | P3 | 归约发射：短展开/长循环/local.var 累加/跨 lane 合并 | ReduceInfo + chunks |
| P3-06 | P3 | 多段发射：共享 prologue、寄存器直通、源序 | segments + 段间依赖 |
| P3-07 | P3 | 写后读转发/累加器寄存器直通 | 计划中的 promote/carry 信息 |
| P3-08 | P3 | 不重新选轴/不改分块/不重新推断连续性 | 完整计划（P2 输出） |

### 2.2 信息完备度矩阵

记号：✅ 直接提供；◐ 可从 P1 输出推导（需要 P2 自己的分析）；❌ 缺失。

| 需求 | 文本输出 | 类型化 `VSimdRegion` | 说明 |
| --- | --- | --- | --- |
| P2-01 N 层域 | ◐（PR-B 可打印多层） | ❌ region；◐ PR-B | region 只有一层；但 `VSimdCaptureInfo` 的多层 `domain` + 扁平写访问可重建“直线多层” nest（5 个场景），见 §2.2 桥接说明 |
| P2-02 serial 轴 | ❌ | ❌ | `For(kSerial)` 被拒；无 `serial_axis` |
| P2-03 body 结构 | ❌ | ❌（直线场景例外） | `accesses` 是扁平列表，**直线场景**可由有序写 + `value` 重建；`Bind` 由 `bindings` 提供；**if/serial 结构**缺失 |
| P2-04 pre/main/post | ❌ | ❌ | 无语句分组 |
| P2-05 多段 program | ❌ | ❌ | 无段边界 |
| P2-06 逐轴 Δ 分类 | ◐（有索引表达式，但无 dtype/shape/strides） | ◐（Buffer 完整，可推导） | P1 不提供 P2 的 `AccessClass` |
| P2-07 轴排序/回退 | ◐ | ◐ | P2 可自行计算 |
| P2-08 lane 数 | ❌（无 dtype） | ◐（Buffer dtype） | 文本缺 dtype |
| P2-09 chunks/tail | ◐（extent 有） | ✅（extent 有） | 需要 lanes |
| P2-10 值 DAG | ◐（store value 文本） | ✅（PrimExpr） | P2 可重建；P1 不提供拓扑 DAG |
| P2-11 归约语义 | ❌ | ❌ | `Legality` 无 REDUCTION；region 无 ReduceInfo |
| P2-12 短/长归约 | ❌ | ❌ | 同上 |
| P2-13 守卫/边界 | ❌ | ❌ | 无谓词、无地址边界模型 |
| P2-14 累加器提升 | ❌ | ❌ | `local.var` 被拒；无 RMW 结构 |
| P2-15 诊断 | ◐（自由文本） | ◐ | 无机器可读错误码 |
| P2-16 轴角色 | ❌ | ❌ | 无并行/串行/向量角色 |
| P3-01 recipe | ❌ | ❌ | 是 P2 输出，不是 P1 |
| P3-02 地址 | ◐（文本无 dtype/shape） | ◐（Buffer 完整） | P3 需要 P2 计划 |
| P3-03 掩码 | ❌ | ❌ | 无 lane/tail/谓词 |
| P3-04 chunk 循环 | ❌ | ❌ | 无 chunk/lane |
| P3-05 归约发射 | ❌ | ❌ | 无 ReduceInfo |
| P3-06 多段发射 | ❌ | ❌ | 无 segments |
| P3-07 转发/直通 | ❌ | ❌ | 是 P2 计划字段 |
| P3-08 计划完整性 | ❌ | ❌ | P3 输入必须是 P2 `VectorProgram` |

#### 2.2.1 关键补充：`VSimdCaptureInfo` 的“多层直线桥”

`VSimdRegion` 只在 `domain.size()==1` 时产出（G3），但 `VSimdCaptureInfo`（PR-B）对多层 nest 仍然给出：

- 多层 `domain`（var/min/extent，外层→内层）；
- 源码顺序的 `accesses`（写访问带 `value`）；
- `captures/outer_vars/shape_syms/local_views/layouts/assumptions/bindings`；
- `status=CLOSED`；
- PR-A 的 per-VF `LEGAL` 状态。

对于**直线多层**（无语句级 if / serial / 归约）的 5 个场景（A3/A4/A7/C2/C4），这些信息足以重建等价 `StoreSpec[]` 与迭代域，因而 **P2 可以规划**——不需要 `VSimdRegion`，也不需要扩展 P1。它不是“P1 无法支撑”，而是“PR-C 的 G3 lift 限制没有把已分析好的信息交出来”。

安全前提（必须同时满足）：

1. PR-A 状态 `LEGAL`；
2. PR-B `status == CLOSED`；
3. `domain.size() >= 2`；
4. `accesses` 中没有 `local.var`、没有语句级 `IfThenElse`/`For(kSerial)`/归约形态（这些在 PR-B 阶段会分别导致 `OPAQUE`/`OPEN` 或 PR-A `ILLEGAL`）；
5. 由 `accesses` 重建的写列表覆盖全部写，读由 `value` 内嵌。

这 5 个场景加上 `VSimdRegion` 的 15 个共同场景，构成 **20 个可规划场景**（20/55 ≈ 36% 的 P2 OK 集）。

桥接的已知限制：它继承 PR-A 的 **per-VF** 粒度——同一 VF 内若还有另一个非法 nest，整 VF 状态会拖累这个合法 nest（M09）。这不是桥接引入的新问题，而是 P1 v1 的既有粒度；在单 nest/VF 的 66 场景中不触发。

### 2.3 缺失的必要信息清单

| 编号 | 缺失项 | 影响的 P2/P3 任务 | 严重度 | 依据 |
| --- | --- | --- | --- | --- |
| M01 | 多层并行域（`VSimdRegion` 只有 1 层；PR-B 桥可覆盖 5 个直线多层场景） | P2-01/07/16、P3-04 | **阻断 region 路径**；桥接可解 5 个 | G3：“PR-C minimal lift only supports one T.Parallel level”；PR-B 的 `domain` 仍是多层 |
| M02 | 语句级控制流结构（Seq/If/For/Bind 树） | P2-03/04/05/13 | **阻断** | `IfThenElse`/`For(kSerial)` 默认拒绝；`accesses` 扁平 |
| M03 | store 谓词 / mask | P2-03/13、P3-03 | **阻断** | `AccessInfo` 无 mask；region 无谓词字段 |
| M04 | serial 轴与 clean 判定 | P2-02/04/05 | **阻断** | 文档 §8.3.5：“P1 的模型假设‘并行域之外无迭代’，串行维度一概不接受” |
| M05 | 归约语义（op/axis/identity/accumulate/varying） | P2-11/12、P3-05 | **阻断** | 文档 §8.3.2：“判定模型本身不包含归约语义” |
| M06 | 轴角色（并行/串行/向量轴）与轴序号 | P2-16、P3-04 | **阻断** | `LoopInfo` 只有 var/min/extent；无 kind/role |
| M07 | lane/chunk/tail | P2-09、P3-03/04 | **阻断** | P1 明确“不含物理布局、lane 或指令选择” |
| M08 | 地址边界/守卫证明 | P2-13、P3-03 | 高 | P1 无 lane/边界模型；D16 类场景全拒 |
| M09 | 逐 nest 内存合法性（PR-C lift 结果逐 nest，但 `memory_status` 按 VF 聚合） | P2 门禁、P3 决策 | 高 | PR-A 状态按 VF 聚合；PR-C 用 `statuses[info.vf_index]` 给同 VF 所有 nest 取同一状态，同 VF 多 nest 一损俱损 |
| M10 | 元素 dtype / shape / strides / elem_offset（文本） | P2-06/08/10 | 高（文本契约） | 文本只打印 buffer 名；类型化 Buffer 有 |
| M11 | `bindings`（文本） | P2-09/13 | 中（文本契约） | `RenderVSimdCaptureReport`/`DumpVSimdRegion` 都不打印 |
| M12 | 布局映射细节（文本） | P3-02（`vsts` vs `vsstb`） | 中（文本契约） | 文本只打印 buffer 名 |
| M13 | storage scope（GM/UB） | P2-06 的 `in_global` | 中（文本契约） | P1 不检查；类型化 Buffer 有 scope |
| M14 | 线程/外层变量边界（`thread_bounds`） | P2-13 的边界证明 | 中 | 文档明确“不设”；P2 若要需回读函数/模块 |
| M15 | 逐访问 `class`/`injective`/`layout_changing`（region） | P2 诊断/可信度 | 中 | 只在 PR-A 文本；region 不携带 |
| M16 | 机器可读错误码 / 诊断结构 | P2-15 | 中 | 所有状态/原因都是自由文本 |

### 2.4 Q2 可行性判定

**判定：仅凭 P1 输出，无法完整完成 P2/P3 的全部既定任务。**

分三层说明：

1. **文本输出层：连基线都不够。**
   - 文本没有 dtype/shape/strides/elem_offset，P2 无法可靠推导 lane 数、Δ 分类、地址；
   - 文本没有 `bindings`，尾块/符号 extent 的表达式无法完整复现；
   - 文本没有语句树/谓词/serial/归约结构；
   - 因此“只读文本”的 P2 最多能做人工诊断，不能做完整规划。

2. **类型化输出层：region 覆盖一个子集，PR-B 桥再补 5 个。**
   - 文档 §8.4 的量化结论：66 场景中 **18 个可 lift**（15 个 polyhedral OK + 3 个 polyhedral REJ），**5 个只分析不 lift**（A3/A4/A7/C2/C4），**43 个拒绝**；
   - 若 P2 只消费 `VSimdRegion`：与 P2 的 55 个 OK 集求交，**15 个场景**可规划（A1/A2/A5a/b/c/A6/A8/A9/A10/A11/A12/A13/C1/C3/D2）；
   - 若 P2 同时消费 PR-B `VSimdCaptureInfo`（多层直线桥）：再加 5 个（A3/A4/A7/C2/C4），**合计 20 个可规划场景**；
   - 20/55 ≈ 36% 的 P2 OK 场景；按 66 场景计，18/66 ≈ 27% 可 lift，23/66 ≈ 35% 可分析；
   - 归约、语句级控制流、serial、多段、守卫边界仍然全部不可达；20 个场景全部是“直线 + 写单射 + 值级 select”。

3. **P3 层：P1 不是 P3 的输入。**
   - P3 的输入是 P2 的 `VectorProgram`（axes/lanes/chunks/tail/stores/recipes/reduce/segments）；
   - P1 没有 lane/chunk/尾块/掩码/recipe/指令选择；即使 P2 完成规划，P3 也不能“仅凭 P1”发射。

**结论一句话**：P1 输出是“逻辑合法性 + 捕获上下文”的门禁与语义 oracle，不是 P2 的完整规划输入，更不是 P3 的发射输入。它能支撑的是“单层/多层直线 + 写单射 + 值级 select”的 **20 个场景子集**（15 个 region + 5 个 PR-B 多层桥）；归约、语句级控制流、串行、多段、守卫边界仍需 P1 模型扩展或 P2 自己的合法性分析。

> 文档 §8.9 的“衔接观察”也给出同样的定位：“两套实现的强项恰好互补……若后续把 P1 作为 P2/P3 的前置门禁、把 polyhedral 的 plan/emit 作为 P3 的发射后端，则 N1–N5 中只有 N3（归约语义）需要两侧共同扩模型，N1/N2/N4 属于 P1 侧补齐已有设计、N5 属于分工边界。”

---

## 3. 两条技术路线对比（暂不扩展 P1）

### 3.1 前提澄清

- P2 输入 = **P1 输出 + 前置 Pass 输出的 TIR 中间表示**；
- P1 不扩展；
- P1 的契约以**类型化结构**为准（`VSimdCaptureInfo` / `VSimdRegion`），文本只作诊断；
- 任何被 P2 规划、被 P3 发射的 nest，仍必须以 P1 的合法性结论为前提（否则“P1 门禁”失效）。

**“P1 输出 + TIR”的分工**：TIR 补**结构**（body/循环/分支），P1 补**合法性**（单射、跨迭代依赖、闭包）。
这两者在 20 个直线场景上可以拼成完整前提；但在归约/语句级 if/serial 上，TIR 虽然给出结构，P1 给出的却是
`ILLEGAL/UNSUPPORTED/OPAQUE`（不是“合法”），两者并集仍不构成安全的规划前提。除非 P2 自带一套合法性分析
（路线 2′），否则这些能力不能启用。

### 3.2 路线定义

**路线 1：裁剪 P2。**
删除/停用所有 P1 v1 无法支撑的能力（归约、语句级控制流、serial 吸收、多段、多层、守卫边界、`local.var` 提升等），只保留单层、直线、写单射、值级 select、多 store 的分析与发射。

**路线 2：保留 P2 全部能力 + 触发判断。**
P2 代码不删；在入口处判断“P1 输出是否包含对应能力的触发标识”，有则执行对应分析，无则跳过。

### 3.3 路线 2 的触发标识审计（关键）

逐项检查 P1 v1 输出能否作为“能力触发”：

| 能力 | 期望的触发标识 | P1 v1 实际输出 | 能否触发 |
| --- | --- | --- | --- |
| 多层直线（A3/A4/A7/C2/C4） | `PR-A LEGAL` + `PR-B CLOSED` + `domain.size()>1` + PR-C reason=`one T.Parallel level` | ✅ 四项都可从 P1 输出取得 | ✅ **可触发**（派生信号，不是单一字段） |
| 多层 + 控制流/串行 | `domain.size()>1` 且 body 允许 | PR-B 会因 if/serial 变 `OPAQUE`/`OPEN`；PR-A 可能 `UNSUPPORTED` | ❌ |
| 语句级控制流 | `has_if` / `control_flow=true` / 谓词字段 | 无；`IfThenElse` 直接 `UNSUPPORTED`/`OPAQUE`；`allow_control_flow_` 不进入输出 | ❌ |
| 归约 | `reduce_op` / `reduce_axis` / `identity` / `REDUCTION` 状态 | 无；归约写被判 `ILLEGAL` | ❌ |
| serial 轴 | `serial_axis` / `loop_kind` / `pre/main/post` 标记 | 无；`For(kSerial)` 直接拒绝；`LoopInfo` 无 kind | ❌ |
| 多段 program | `segments` / 段边界 | 无 | ❌ |
| 守卫/边界 | `mask` / `guard_boundaries` / 边界证明 | 无 | ❌ |
| 间接写/散列写 | `scatter` / 每 lane 索引向量 | 数据相关写 `UNPROVEN`；无 scatter 标识 | ❌（读侧间接有，写侧没有） |
| `local.var` 提升 | `local_var` capture / RMW 结构 | `ManualDetector`/`NestCollector` 判 OPAQUE/UNSUPPORTED | ❌ |
| 基础逐元素 | `legality=LEGAL` + `CLOSED` + `domain.size()==1` | ✅ | ✅ 正向触发 |

**结论：P1 v1 输出中不存在“归约 / 语句级控制流 / serial / 多段 / 守卫”的正向触发标识；可用的正向触发只有两个：基础逐元素（`LEGAL` + 单层）和多层直线（`LEGAL` + `CLOSED` + `domain.size()>1` + G3 fallback）。**

因此：

- **字面路线 2 部分可实现**：按“有触发才执行”的规则，基础逐元素与多层直线会执行，其余高级能力全部跳过；路线 2 的运行时能力面 = **20 个场景**（15 region + 5 多层直线桥），与“正确接入 PR-B 桥的路线 1”相同。
- 若把触发器改为“从 TIR 结构检测”来激活归约/控制流/serial，则 P1 对这些构造的结论是 `ILLEGAL/UNSUPPORTED/OPAQUE`；要么忽略 P1（P1 门禁失效，出现第二合法性权威），要么仍然跳过（回到 20 个场景）。这不是“路线 2 原义”，而是 **路线 2′：TIR 检测 + P1 仅对子集生效**，风险见 §3.4。

### 3.4 多维度对比

评分：5 = 最好，1 = 最差。括号内为理由。

| 维度 | 路线 1：裁剪 | 路线 2：保留 + 触发（字面） | 路线 2′：TIR 检测 + P1 子集门禁 |
| --- | --- | --- | --- |
| 架构合理性 | **5**：单一合法性权威、单一输入契约、职责清晰 | **3**：多层直线有派生触发，但归约/控制流/serial 仍无触发，触发矩阵不完整 | **2**：P1 与 TIR 双源，P1 只对子集生效，合法性边界模糊 |
| 代码可维护性 | **4**：代码少、无死路径；但删除 4.6k 行是破坏性变更 | **3**：代码保留，但归约/控制流/serial 路径在生产中永远走不到 | **2**：双源判断 + 子集门禁，分支组合多 |
| 后续迭代扩展性 | **2**：P1 v2 到来后需重新引入已删代码 | **5**：能力位一开即可用，升级成本最低 | **4**：可扩展，但需要维护两套合法性 |
| 运行时性能 | **4**：无门禁开销；但高级算子不向量化，端到端可能变慢 | **4**：门禁开销可忽略；运行时与接入桥的路线 1 相同 | **3**：高级路径可运行，但 P1 不覆盖，正确性风险高 |
| 出错风险 | **5**：所有发射都经过 P1 LEGAL；不会绕过 | **4**：门禁关闭时安全；但“死代码”可能被误开 | **1**：最危险——归约/散列写可能在 P1 判 ILLEGAL 的情况下被发射 |
| P1 升级向下兼容成本 | **2**：需重新移植/接通被删代码 | **5**：能力位切换即可 | **4**：需同时升级 P1 子集门禁与 TIR 检测 |
| 运行时能力面（P1 v1） | 20/55（接入 PR-B 桥；只接 region 则 15/55） | **20/55（基础 + 多层直线；高级仍无触发）** | 可达 55/55，但其中 35 个无 P1 合法性覆盖 |
| 实现工作量 | 中（删除/重排 + 桥接 + 回归） | 中（触发矩阵 + 跳过语义 + 桥接） | 大（双源 + 合法性融合） |

### 3.5 量化场景覆盖（P1 v1）

| 分类 | 数量 | 说明 |
| --- | --- | --- |
| P1 可 lift 且 P2 OK（region 路径） | **15** | A1/A2/A5a/A5b/A5c/A6/A8/A9/A10/A11/A12/A13/C1/C3/D2 |
| P1 可分析不 lift（◐）且 P2 OK（PR-B 桥） | **5** | A3/A4/A7/C2/C4（多层直线） |
| **合计可规划** | **20** | 15 region + 5 PR-B 桥；20/55 ≈ 36% |
| P1 可 lift 但 P2 拒绝 | 3 | `c_gm`（GM 操作数）、`c4_true_random`（读侧非 2 幂）、`d15_bool_store`（bool store） |
| P1 拒绝（❌） | 43 | 归约 11 + 语句级控制流 8 + 串行 24 |
| 合计 | 66 | 与文档 §8.4 一致 |

**关键结论**：在“暂不扩展 P1”的前提下，路线 2（字面，含多层直线派生触发）与“正确接入 PR-B 桥的路线 1”的**运行时能力面完全相同（20/55）**；两者的差异只在“代码是否保留”和“未来升级成本”。如果路线 1 只接 `VSimdRegion` 而不接 PR-B 桥，则它会少 5 个场景（15/55）——这 5 个场景不需要扩展 P1，放弃它们纯属实现选择。任何声称“路线 2 能保留归约/控制流/serial 全部能力”的说法，实际上都隐含了路线 2′（绕过 P1 合法性），风险不可接受。

### 3.6 选型结论

**推荐：路线 2 的工程修正版——“保留代码资产 + 显式能力门控（默认关闭）”。**

理由：

1. **运行时等价于“接入 PR-B 桥的路线 1”，安全性不打折**：`P1CapabilitySet` 当前声明 v1 能力（基础逐元素 + 多层直线桥 + 读侧任意形态 + 多 store + 值级 select + 符号 extent + typed layout/view），归约/语句级控制流/serial/多段/守卫等高级能力位全部 false；门控在 P2 入口强制跳过，因此不会绕过 P1。运行时能力面 20/55。
2. **保留 4.6k 行、135 个用例的已验证资产**：不删除归约/控制流/serial/多段/守卫等模块；它们仍在测试中运行（不是死代码），只是不进入生产路径。
3. **P1 v2 到来时升级成本最低**：新增能力位 + 适配器即可打开，不需要重新移植。
4. **避免了字面路线 2 的概念矛盾**：触发器不是“P1 输出字段”（P1 没有），而是 P2 侧显式、版本化的能力表；这符合“暂不扩展 P1”的约束。

**何时改选路线 1**：如果团队要求“最小代码、一次交付、不保留实验模块”，或 P1 v2 在可预见周期内不会落地，则路线 1 更简单；但必须接受“未来重新引入高级能力”的移植成本。注意：路线 1 也应接入 §2.2.1 的 PR-B 多层直线桥，否则会无谓地少 5 个场景（15/55 而非 20/55）。

**不推荐路线 2′**：它让 P2 在 P1 判 `ILLEGAL/UNSUPPORTED` 的构造上继续发射（归约、散列写、serial），等于取消 P1 的合法性权威；在编译器里这是最高等级的正确性风险。

---

## 4. 落地实施（推荐路线：保留代码 + 显式能力门控）

### 4.1 架构改造

在 P2 入口增加三个薄层，不改 P2 内部算法：

```
P1 输出（VSimdRegion + VSimdCaptureInfo + PR-A status） + 前置 TIR
        │
        ▼
[1] RegionClassifier        识别构造种类：elementwise / multi_level_straightline / reduction / statement_if / serial / ...
        │
        ▼
[2] P1CapabilitySet         版本化能力位；v1 声明 basic_elementwise + multi_level_straightline 等
        │
        ├─ 能力缺失 → PlanOutcome::SkippedByCapability(reason)
        ├─ P1 非 LEGAL → PlanOutcome::RejectedByP1(status, reason)
        └─ 能力具备且 P1 LEGAL → 进入 P2 Plan
        │
        ▼
[3] PlanOutcome             统一结果：Planned / SkippedByCapability / RejectedByP1 / RejectedByP2 / InternalError
```

**硬不变量（必须断言）**：

```
Planned ⇒ (P1.legality == LEGAL)
          ∧ (capability bit for construct == true)
          ∧ (domain.size() == 1  ∨  (multi_level_straightline ∧ PR-B CLOSED ∧ 无 if/serial/归约))
```

- `SkippedByCapability` 不是错误，不产生新诊断噪音；但必须有稳定 reason code，便于统计；
- `RejectedByP1` 保留 P1 原始原因；
- `RejectedByP2` 保留 P2 的 `Unsupported` 诊断；
- `InternalError` 单独上报，不能静默回退。

### 4.2 代码组织

推荐把 P2 分成三层（C++ 或 Python 原型都适用）：

| 层 | 内容 | 是否进入生产 Pass |
| --- | --- | --- |
| `polyhedral/core` | 单层逐元素、多层直线桥、多 store、值级 select、Δ 分类、轴选择、chunk/尾块、基础发射 | ✅ |
| `polyhedral/ext` | 归约、语句级控制流、serial 吸收、多段 program、守卫边界、`local.var` 提升 | ❌（由能力位关闭；测试中保持运行） |
| `polyhedral/capabilities` | `P1CapabilitySet`、`RegionClassifier`、`PlanOutcome`、门禁、PR-B 多层直线桥 | ✅ |

这样既满足“生产路径只保留 P1 v1 能力”，又满足“代码资产不丢失”。

### 4.3 分步实施

| 步骤 | 内容 | 交付物 | 验收标准 |
| --- | --- | --- | --- |
| S1 | 冻结 P1 v1 契约：字段清单、状态、文本格式、typed struct 版本号 | `P1ContractV1` 文档/头文件 | 字段与 §1.3 一致；无遗漏 |
| S2 | 定义 `P1CapabilitySet v1`：列出能力位与默认值（`basic_elementwise`、`multi_level_straightline`、读侧形态、多 store、值级 select、符号 extent 为 true；归约/语句级控制流/serial/多段/守卫为 false） | capability 头文件 | 每个高级能力位有明确 false 及理由 |
| S3 | 实现 `RegionClassifier`：从 P1 region/capture info + TIR 识别构造种类（含“多层直线”与“多层含 if/serial”） | classifier | 66 场景分类与 §3.5 表一致 |
| S3b | 实现 PR-B 多层直线桥：从 `VSimdCaptureInfo.domain + accesses + bindings` 重建 `StoreSpec[]` 与合成 nest；仅当 `LEGAL + CLOSED + 无 if/serial/归约` 时启用 | bridge | 5 个 ◐ 场景可规划；含 if/serial 的多层场景仍拒绝 |
| S4 | 实现 `PlanOutcome` 与门禁：能力缺失/ P1 拒绝/ P2 拒绝分离 | gate | 每个场景有唯一 outcome；无静默回退 |
| S5 | 生产路径只调用 core；ext 模块保留但由门禁隔离 | 目录/命名空间调整 | 生产 Pass 不引用 ext；ext 测试仍运行 |
| S6 | P3 只消费 `Planned` 的 `VectorProgram` | P3 入口 | 无 plan 不发射；不重新分析 |
| S7 | 增加统计与日志：按 outcome/capability 计数 | 日志/指标 | 可看到“跳过原因分布” |
| S8 | 灰度与回滚：配置开关 | 配置项 | 关闭时行为与当前分支一致 |

### 4.4 测试验证标准

| 测试 | 断言 |
| --- | --- |
| P1 契约测试 | `VSimdRegion`/`VSimdCaptureInfo` 字段与 §1.3 一致；文本格式稳定；`bindings` 在 typed/FFI 可见、文本不可见的行为被显式记录 |
| 能力门禁单测 | 每个能力位 on/off 时，P2 分别进入/跳过对应路径；跳过 reason 稳定 |
| 66 场景分类测试 | 20 Planned（15 region + 5 PR-B 桥）/ 3 RejectedByP2 / 43 RejectedByP1（或按实现细分 SkippedByCapability） |
| 基线 parity | 20 个 Planned 场景的 plan 与 Python `build_plan` 逐项一致；发射 IR `structural_equal` |
| 硬不变量测试 | 任何 `Planned` 都满足 `P1 LEGAL ∧ capability ∧ (单层 ∨ 多层直线桥条件)`；用断言/注入测试验证 |
| 无 P1 绕过测试 | 归约/语句级 if/含 serial 的多层场景绝不进入 emit；用覆盖率或 instrumentation 证明 |
| 回归 | P1 49 用例 + P2 135 oracle 全绿 |
| E2E | 至少 3 个真实 kernel：无残留目标 `T.Parallel`；预期 `tl.simd.*` 序列；非目标 IR `structural_equal` |
| 性能 | 门禁开销可忽略；编译时间/IR 体积与基线对比无显著恶化 |

### 4.5 后续迭代的兼容保障方案

1. **契约版本化**：`P1ContractVersion = 1`；每次 P1 新增字段/状态，版本号递增；P2 声明它需要的字段集合。
2. **能力位版本化**：`P1CapabilitySet v1` 只含 v1 能力；P1 v2 到来时新增 `v2` 能力集，不修改 v1 默认值。
3. **适配器分层**：`RegionAdapterV1` / `RegionAdapterV2` 并存；P2 core 只依赖抽象 `PlanInput`，不直接依赖具体版本字段。
4. **一致性测试**：`CapabilityConformanceTest` 断言 P2 没有读取当前版本未声明的字段；新增字段必须同步更新能力表。
5. **黄金语料**：每个版本维护 66 场景的 outcome 快照；版本升级时 diff 必须可解释（新增 Planned 或明确的能力位变化）。
6. **双读/弃用窗口**：v1/v2 适配器并存至少一个发布周期；v1 标记 deprecated；移除前跑全量回归。
7. **回滚**：能力位与 Pass 开关均可独立关闭；`ext` 模块不进入生产路径，回滚不需要删代码。
8. **诊断稳定**：`SkippedByCapability`/`RejectedByP1`/`RejectedByP2` 的 reason code 固定；自由文本只作补充。
9. **CI 矩阵**：P1 v1 构建 + P1 v2（可用时）构建；两条流水线都跑 66 场景分类与基线 parity。

---

## 5. 风险与应对

| 风险 | 触发条件 | 应对 |
| --- | --- | --- |
| 把“跳过”误当“失败” | 能力门禁返回 `SkippedByCapability` | 明确 outcome 分类；日志/指标分开统计；测试断言不产生 error 级日志 |
| 能力位被误开 | 手工改配置或未来升级 | 硬不变量断言；CI 跑“关闭态”全量；代码评审要求能力位变更有测试 |
| ext 模块 bit-rot | 长期不进入生产路径 | ext 模块保留独立测试；每次 P1/P2 变更都跑 ext 单测 |
| 双源合法性（若走路线 2′） | TIR 检测 + P1 子集门禁 | **不采用**；如必须，增加“P1-unverified”显式标记并禁止默认开启 |
| 文本契约被误用 | 下游直接解析报告文本 | 文档明确文本仅诊断；P2/P3 只消费 typed struct/FFI Map |
| P1 v2 字段变更 | 升级 P1 | 版本化契约 + 适配器 + 一致性测试 + 黄金语料 diff |
| 性能回退 | 高级能力被关闭后算子退回 legacy | 记录 legacy 回退率；按算子优先级规划 P1 v2 能力顺序 |

**一键回滚**：关闭 `TL_ENABLE_P1_CAPABILITY_GATE`（或等价开关）→ P2 不进入生产路径，全部交给 legacy `AscendSimdVFLowerParallel`；ext 模块与测试不受影响。

---

## 6. 基于 `vsimd_region.cc` 的代码级核实

> 本节逐行核对 `/home/cjl/workspace/tilelang-stable/src/ascend/transform/vsimd_region.cc`，回答“P1 三个 Pass 如何执行、L298 产出什么、L326 改写成什么、P2/P3 能否直接承接”四个问题。所有结论以代码为准，不以文档表述为准。

### 6.1 三个 P1 Pass 的完整执行链路

#### 6.1.1 Pass 注册与调用关系

| Pass / FFI | 文件与入口 | 核心调用 | 产出 | 是否修改 IR |
| --- | --- | --- | --- | --- |
| `AscendAnalyzeSimdVFMemory` | `analyze_simdvf_memory.cc:21-27` | `AnalyzeVSimdMemoryLegality(f)` | PR-A 文本报告 | 否，`return f` |
| `AscendAnalyzeSimdVFCapture` | `analyze_simdvf_capture.cc:26-32` | `AnalyzeVSimdCaptureScope(f)`（内部 `AnalyzeVSimdCaptureInfo` + 渲染） | PR-B 文本报告；另有 `AscendAnalyzeSimdVFCaptureInfo` FFI 返回字符串化 Map | 否，`return f` |
| `AscendAnalyzeVSimdRegion` | `analyze_vsimd_region.cc:21-27` | `AnalyzeVSimdRegionReport(f)` | PR-A + PR-B + PR-C 联合文本报告 | 否，`return f` |
| `AscendAnalyzeVSimdRegionReport` | `analyze_vsimd_region.cc:17-19` | 直接调用 `AnalyzeVSimdRegionReport(f)` | `ffi::String`（联合报告） | 只读函数，无 Pass 包装 |

三个 Pass 的共同形态：

```cpp
auto pass_func = [](PrimFunc f, const IRModule &mod, PassContext ctx) {
  LOG(INFO) << "\n" << AnalyzeXxx(f);
  return f;                    // 原样返回，绝不改 IR
};
return CreatePrimFuncPass(pass_func, 0, "tl.AscendAnalyzeXxx", {});
```

关键事实：

1. **三个 Pass 相互独立**，不是一条串行流水线；每个 Pass 只调用自己的分析函数。
2. 只有 `AscendAnalyzeVSimdRegion` / `AscendAnalyzeVSimdRegionReport` 把 PR-A、PR-B、PR-C **组合**起来；它直接调用 `AnalyzeVSimdMemoryStatuses` 与 `AnalyzeVSimdCaptureInfo`，而不是调用另外两个 Pass。
3. 三个 Pass 均不在默认编译流水线中（文档 §7.5 已核查 `pipeline.py` 无调用点），属于只读诊断工具。
4. `AscendAnalyzeVSimdRegionReport` 是 **FFI 查询函数**，不是 Pass；它返回字符串，不返回 IR。

#### 6.1.2 联合入口的实际执行顺序

`AnalyzeVSimdRegionReport`（`vsimd_region.cc:298-340`）的执行顺序：

```
AnalyzeVSimdRegionReport(func)                    // L298
 ├─ statuses = AnalyzeVSimdMemoryStatuses(func)    // L299, PR-A：每 VF 一个状态
 ├─ infos    = AnalyzeVSimdCaptureInfo(func)       // L300, PR-B：每 nest 一条
 └─ for info in infos:                             // L303
      if info.nest_index < 0:                      // L304
         输出 "VF#i status=OPAQUE reason=..."       // L305-308
         continue
      memory_status = statuses[info.vf_index]      // L311-315（越界为 MISSING）
      lift = BuildMinimal1DRegion(info, memory_status)  // L316, PR-C
      if !lift.lifted:
          INTERNAL_ERROR / FALLBACK + reason       // L318-324
          continue
      expanded_stmt = ExpandVSimdRegionToLoops(lift.region)  // L326
      输出 status=LIFTED                            // L327
      输出 DumpVSimdRegion(lift.region)             // L328
      输出 expanded_stmt                            // L329-331
      输出 verify: OK                               // L332
      输出 BuildVSimdRegionCarrier(lift.region)     // L333-335
```

要点：

- `statuses` 是 **每 VF** 粒度；`infos` 是 **每 nest** 粒度；PR-C 用 `statuses[info.vf_index]` 给同 VF 的所有 nest 取同一个内存状态（文档 §4.2 的“一损俱损”）。
- `VSimdRegionLift lift` 是 **循环内的局部变量**；函数只把它的文本形式写进 `std::ostringstream out`，**不返回 typed region**。
- `expanded_stmt` 只用于 `expanded:` 文本；`BuildVSimdRegionCarrier` 只用于 `carrier:` 文本；两者都**不注入 IR**。

### 6.2 `AscendAnalyzeVSimdRegionReport` 的作用与产出定义

#### 6.2.1 它是“组合 + 渲染”，不是“变换”

| 维度 | 事实 | 代码依据 |
| --- | --- | --- |
| 输入 | `PrimFunc` | `vsimd_region.cc:298` |
| 输出 | `ffi::String`（联合报告文本） | `vsimd_region.cc:298-340` |
| 是否修改函数体 | 否；外层 Pass `return f` | `analyze_vsimd_region.cc:22-25` |
| 是否返回 typed IR | 否；`VSimdRegionLift` 是局部变量 | `vsimd_region.cc:316` |
| 是否生成可执行载体 | 否；`BuildVSimdRegionCarrier` 是 debug AttrStmt | `vsimd_region.cc:181-186` |
| 是否发射指令 | 否；没有任何 `tl.simd.*` | 全文无 `tl.simd` |

#### 6.2.2 产出文本的完整格式

对每个 `VSimdCaptureInfo`：

1. `nest_index < 0`：`VF#<i> status=OPAQUE[ reason=<message>]`
2. 未 lift：`VF#<i> nest <k>: status=FALLBACK reason=<reason>` 或 `status=INTERNAL_ERROR reason=<reason>`
3. lift 成功：
```
VF#<i> nest <k>: status=LIFTED
region VF#<i> nest#<k>
  domain: <var> in [<min>, <min+extent>)
  captures: <name>[<kind>] ...
  [outer: <name>[<kind>] ...]
  [shape_syms: <name> ...]
  [local_views: <view> -> <base> ...]
  [layouts: <buffer> ...]
  [assumptions: <expr> ...]
  accesses:
    R <name>[<indices>]
    W <name>[<indices>] = <value>
  expanded:
    <reference scalar serial loop>
  verify: OK
  carrier:
    <AttrStmt(...)>
```
4. 全空：`no SIMD_VF blocks\n`

注意：`bindings` **不在文本里**；`dtype/shape/strides/elem_offset` 也不在文本里；`accesses` 是扁平读写列表，不是语句树。

#### 6.2.3 结论：L298 与“P1→P2 中间表示”的关系

- 如果“中间表示”指 **报告文本**：是的，完全由 `AnalyzeVSimdRegionReport`（L298-340）生成。
- 如果“中间表示”指 **P2 可消费的 typed 结构**：**不是**。typed `VSimdRegion` 由 `BuildMinimal1DRegion`（L44-101）构造，L298 只是在 L316 调用它，并把结果渲染成文本；函数不导出该结构。
- 当前代码中 **不存在** “P1 产出 typed IR → 直接交给 P2” 的接口；`VSimdRegion` 是 C++ 内部结构，没有 Python FFI；`CaptureInfoToFFI` 只序列化 PR-B 且字段字符串化。

### 6.3 P1→P2 中间表示规范（以 typed `VSimdRegion` 为准）

#### 6.3.1 数据结构

```cpp
// vsimd_region.h
struct VSimdRegion {
  int vf_index;                                  // 源 SIMD_VF 序号
  int nest_index;                                // VF 内 nest 序号
  std::vector<vsimd::LoopInfo> domain;           // 恰好一层 T.Parallel
  std::vector<vsimd::AccessInfo> accesses;       // 源码顺序的读写；写带 value
  std::vector<std::pair<Buffer, std::string>> captures;
  std::vector<std::pair<Var, std::string>> outer_vars;
  std::vector<Var> shape_syms;
  std::vector<std::pair<Buffer, Buffer>> local_views;
  std::vector<std::pair<Buffer, Layout>> layouts;
  std::vector<PrimExpr> assumptions;
  std::vector<std::pair<Var, PrimExpr>> bindings;
};

struct VSimdRegionLift {
  bool lifted;                                   // 是否成功 lift
  bool internal_error;                           // 展开器自检失败（编译器缺陷）
  std::string reason;                            // 失败原因
  VSimdRegion region;                            // lifted=true 时有效
};
```

字段语义与 P2 用途：

| 字段 | 语义 | P2 用途 | 备注 |
| --- | --- | --- | --- |
| `vf_index/nest_index` | 定位 | 报告键、与 PR-A 状态对齐 | 不保证 P3 可定位 |
| `domain` | 一层并行域 | 迭代域、轴选择 | 只允许 1 层（G3） |
| `accesses` | 读写列表；写带 `value` | 重建 `StoreSpec[]`、值表达式 | 扁平；读与写混排 |
| `captures` | buffer + 种类 | 区分 param/alloc/local/view | 文本只给名字 |
| `outer_vars` | 外层变量 + 种类 | 符号上下文 | `thread/outer_loop/bind` |
| `shape_syms` | 形状/步长自由符号 | 符号 extent | 文本只给名字 |
| `local_views` | view→base | 别名/地址同一性 | 需要 P2 的 `_same_buffer` |
| `layouts` | `Layout` 对象 | `vsts` vs `vsstb`（P3 扩展） | 当前 P3 未消费 |
| `assumptions` | 纯布尔算术 | analyzer 约束 | 文本可见 |
| `bindings` | `Var -> PrimExpr` | 内联/边界表达式 | **文本不可见** |

#### 6.3.2 生成规则（`BuildMinimal1DRegion`，L44-101）

| 顺序 | 规则 | 代码 | 失败结果 |
| --- | --- | --- | --- |
| G1 | `info.status == "CLOSED"` | L47-50 | `reason="capture/scope is not CLOSED"` |
| G2 | `memory_status == "LEGAL"` | L51-54 | `reason="memory legality is <STATUS>"` |
| G3 | `info.domain.size() == 1` | L55-58 | `reason="PR-C minimal lift only supports one T.Parallel level"` |
| G4 | 任一访问不是 `local.var` | L61-65 | `reason="local.var access is not liftable"` |
| G5 | 所有访问 rank == buffer rank | L66-69 | `reason="access rank does not match buffer rank"` |
| G6 | 至少一个 `BufferStore` | L70-75 | `reason="nest has no BufferStore"` |
| — | 字段拷贝 | L77-88 | — |
| — | 自检：`ExpandVSimdRegionToLoops` + `VerifyExpandedRegion` | L90-99 | `lifted=false`, `internal_error=true` |

**生成关系链**：

```
AnalyzeVSimdRegionReport (L298)       // 组合入口
  └─ BuildMinimal1DRegion (L44)       // typed VSimdRegion 构造器 + G1-G6 + 自检
       ├─ 字段拷贝 (L78-88)
       └─ ExpandVSimdRegionToLoops (L153)  // 参考展开，仅用于自检
```

因此，严格表述应是：**L298 是 P1 联合分析/报告的入口；typed `VSimdRegion` 的构造规则在 `BuildMinimal1DRegion`（L44-101）。**

#### 6.3.3 语义约束

- **只读**：P1 不修改 IR；`LIFTED` 不等于“已改写”。
- **单层**：`domain` 恰好一层 `T.Parallel`；多层场景 `FALLBACK`（G3）。
- **无控制流**：`IfThenElse`/`For(kSerial)` 在 PR-B 阶段就被拒，不会进入 region。
- **无归约**：写地址沿轴均匀 = 非单射 = PR-A `ILLEGAL`，G2 拒绝。
- **无 `local.var`**：G4 拒绝。
- **无 lane/chunk/mask/指令**：region 明确“不含物理布局、lane 或指令选择”。
- **文本不完整**：`bindings`、dtype/shape/strides 不在文本中；P2 若只拿文本，信息不足。

### 6.4 可向量化后的结构改写规则（L326 的真实含义）

#### 6.4.1 先纠正一个关键前提

**P1 不会把原始 `SIMD_VF` 嵌套结构改写成任何形式。** 三个 Pass 都是只读的，`return f`；`AnalyzeVSimdRegionReport` 只生成字符串。L326 调用的 `ExpandVSimdRegionToLoops` 是**参考展开器**，不是生产改写器。

代码注释原文（L167-168）：

> “Reference expansion is intentionally a serial scalar loop; it is a semantic oracle, not the P3 physical lowering.”

#### 6.4.2 `ExpandVSimdRegionToLoops` 的实际规则（L153-179）

```cpp
std::vector<Stmt> stores;
for (const AccessInfo &acc : region.accesses)
  if (acc.is_write)
    stores.push_back(BufferStore(acc.buffer, acc.value, acc.indices));
if (stores.empty()) return Evaluate(0);
Stmt body = stores.size() == 1 ? stores[0] : SeqStmt(stores);
Stmt result = region.domain.empty()
    ? body
    : For(region.domain[0].var, region.domain[0].min,
          region.domain[0].extent, ForKind::kSerial, body);
for (auto it = region.local_views.rbegin(); it != region.local_views.rend(); ++it)
  result = SeqStmt({DeclBuffer(it->first), result});
return result;
```

规则总结：

1. 只重建 **写**（`is_write`），按 `accesses` 源序；读已经在 `value` 里，不单独重建。
2. 循环固定为 `ForKind::kSerial`；只有 `domain[0]`（单层）。
3. `local_views` 用 `DeclBuffer` 前置声明（不是 `AllocBuffer`）。
4. 无 lane/chunk/mask/`tl.simd.*`；这是标量语义参考。

#### 6.4.3 变换前后结构对比

| 阶段 | 结构 | 说明 |
| --- | --- | --- |
| 原始 IR | `SBlock("SIMD_VF") { ... For(i, 0, E, kParallel, SeqStmt(stores...)) ... }` | 用户写的 `T.Parallel` 嵌套；可能有多层、if、serial、local.var |
| P1 判定后 | **IR 不变**；只产出 `VSimdRegion` / 报告文本 | `LIFTED` 只表示逻辑上可向量化 |
| L326 参考展开（仅文本/自检） | `SeqStmt(DeclBuffer(view)..., For(i, min, extent, kSerial, SeqStmt(stores...)))` | 标量串行循环；不是向量化；不注入 IR |
| 真实向量化（P3/legacy） | `For(serial outer) { For(chunk) { vld/.../vsts } }` | P1 不生成；由 P3 或 `AscendSimdVFLowerParallel` 生成 |

**结构差异（原始 → L326 参考展开）**：

- `kParallel` → `kSerial`；
- 多层并行域 → 只保留第一层（G3 已保证只有一层）；
- 读写混排的 `accesses` → 只保留写，读嵌入 `value`；
- `T.view` 的声明从原作用域 → `DeclBuffer` 前置；
- 向量语义/lane/mask/chunk/指令 → **全部没有**。

#### 6.4.4 约束条件

L326 的展开仅在以下条件成立（由 `BuildMinimal1DRegion` 与 `VerifyExpandedRegion` 保证）：

1. G1–G6 全部通过；
2. `domain.size() == 1`；
3. 无 `local.var`；
4. 所有访问 rank 匹配；
5. 至少一个写；
6. `VerifyExpandedRegion` 的 V1–V4：
   - 所有 `local_views` 都被 `DeclBuffer` 声明；
   - 所有 load 的 buffer 属于 captures ∪ local_views ∪ 展开内声明；
   - 展开后的循环层数/var/min/extent 与 `domain` 一致；
   - store 数量、buffer/indices/value 与 region 写访问逐一结构相等。

#### 6.4.5 结论

- L326 的对应关系是：**“可向量化判定后的语义重建”**，不是“向量化改写”。
- 如果你在报告里看到 `expanded:` 的 `for i in range(...)`，它是 **scalar serial loop**，用于证明 region 忠实保留了原 nest 的语义；它不能作为 P2 输入，也不能作为 P3 输出。
- 真正的向量化改写发生在 P3（未来）或既有 `AscendSimdVFLowerParallel`（当前兜底）。

### 6.5 P1-P2-P3 全流程路由规则

#### 6.5.1 四个分支

| 分支 | P1 判定 | P2/P3 能力 | 路由 | 数量/示例 |
| --- | --- | --- | --- | --- |
| **A：正常全流程** | `CLOSED` + `LEGAL` + 单层（`VSimdRegion`）或多层直线（PR-B 桥） | 覆盖 | P2 规划 → P3 发射 → legacy 处理剩余 | 20：15 region + 5 桥 |
| **B：P1 可向量化、P2/P3 能力不足** | `LIFTED` 或桥接条件满足 | 不覆盖（`Unsupported`） | 跳过 P3；记录诊断；迭代 P2/P3 | 3：`c_gm`、`c4_true_random`、`d15_bool_store` |
| **C：P1 不可向量化** | `ILLEGAL`/`UNPROVEN`/`UNSUPPORTED`/`OPAQUE`/`FALLBACK`（非桥接） | 可能覆盖（35 个 P2 OK） | 不进入 P2/P3；legacy/scalar；记为 P1 gap | 43（35 P2 OK + 8 P2 REJ） |
| **D：可分析不 lift（过渡）** | `CLOSED` + `LEGAL` + `domain.size()>1` + 直线 | 覆盖 | PR-B 多层直线桥 → 归入 A | 5：A3/A4/A7/C2/C4（**已含在 A 的 20 内，不额外计数**） |

> 说明：用户要求的三个分支是 A/B/C；D 是 A 的过渡子类（P1 没有 `VSimdRegion`，但 PR-B 信息足够），单独列出是为了不把 5 个可实现场景误判为 C。最终计数：A=20（15 直接 + 5 经 D）、B=3、C=43，合计 66。

#### 6.5.2 完整流转图

```
原始 IR（SIMD_VF 内 T.Parallel）
  │
  ├─ ManualDetector / NestCollector 拒绝 ──────────────► C
  │     （tl.simd.* / local.var / if / serial / TileOp / Evaluate）
  │
  ├─ PR-A: ILLEGAL / UNPROVEN ─────────────────────────► C
  │
  ├─ PR-B: OPEN / OPAQUE ──────────────────────────────► C
  │
  └─ PR-B: CLOSED + PR-A: LEGAL
        │
        ├─ domain.size()==1 ──► BuildMinimal1DRegion ──► VSimdRegion ──┐
        │                                                             │
        ├─ domain.size()>1 且 body 为直线 ──► PR-B 多层直线桥 ────────┤
        │                                                             │
        └─ domain.size()>1 且 body 含 if/serial/归约 ──► C（P1 gap）  │
                                                                      │
                                                                      ▼
                                                                    P2 规划
                                                                      ├─ 成功 ──► P3 发射 ──► legacy 处理剩余 nest
                                                                      └─ Unsupported ──► B：跳过 + 诊断 + 迭代 P2/P3
```

#### 6.5.3 分支判定顺序（必须严格按此顺序）

1. **P1 可分析性**：`ManualDetector`/`NestCollector` 是否拒绝？是 → C。
2. **PR-A 合法性**：`memory_status != LEGAL`？是 → C。
3. **PR-B 闭包**：`status != CLOSED`？是 → C。
4. **PR-C lift**：`domain.size()==1` 且 G1–G6 通过？是 → A/B。
5. **多层直线桥**：`domain.size()>1`、无 if/serial/归约、`LEGAL + CLOSED`？是 → A/B。
6. **否则** → C（P1 gap）。
7. 进入 P2 后：`build_plan` 成功 → A；`Unsupported` → B。

这个顺序保证：**只有 P1 判定可向量化的代码段才会进入 P2/P3；P1 不可向量化的代码段不会被 P2/P3 擅自接管。**

### 6.6 P2/P3 能力保留与迭代触发条件/优先级

#### 6.6.1 能力保留的真实含义

**验证结论（先给答案）**：

| 问题 | 结论 |
| --- | --- |
| 现有 P2/P3 能否“直接承接”P1 输出？ | **否。** P2 `build_plan(stmt)` 接收原始 TIR `For`；P3 接收 `VectorPlan/VectorProgram`；P1 与它们之间没有 adapter，也没有可执行载体。 |
| P2/P3 的全部**代码/测试**能力能否保留？ | **是。** 不删除任何模块；135 个 Python oracle 用例继续运行。 |
| P2/P3 的全部**运行时**能力能否保留？ | **否。** P1 v1 只激活 20/55；35 个 P2 OK 场景被 P1 拒绝，无法进入 P2/P3。 |
| P1 可向量化且 P2/P3 覆盖的场景能否跑通？ | **能。** 20 个（15 region + 5 PR-B 桥），需要先实现 P1→P2 adapter。 |
| P1 可向量化但 P2/P3 不足的场景？ | **3 个**（GM / 读侧 true_random / bool store）；跳过并迭代 P2/P3。 |
| P1 不可向量化的场景？ | **43 个**；不进入 P2/P3，交给 legacy/scalar；其中 35 个是 P1 模型缺口，不是 P2/P3 迭代能解决的。 |

| 保留类型 | 结论 | 说明 |
| --- | --- | --- |
| 代码资产 | ✅ 全部保留 | 归约/控制流/serial/多段/守卫模块不删除；ext 层保留独立测试 |
| 测试资产 | ✅ 全部保留 | 135 个 Python oracle 用例继续运行 |
| 运行时能力 | ❌ 不能全部保留 | P1 v1 只激活 20/55；35 个 P2 OK 场景被 P1 拒绝 |
| 接口兼容 | ✅ 可保留 | P2 `build_plan` / P3 `emit_plan` 签名不变；新增 P1 adapter 与能力门禁 |

#### 6.6.2 迭代触发条件与优先级

| 能力 | P1 v1 状态 | 运行时 | 触发迭代的条件 | 迭代对象 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| 基础逐元素 / 多 store / 值级 select | 覆盖（15+5） | ✅ 激活 | 无 | — | — |
| 多层直线 | PR-B 桥（5） | ✅ 激活 | 无 | — | — |
| 归约（Add/Max/Min，短/长/尾块/符号） | 拒绝（11） | ❌ | P1 增加 REDUCTION 模型，或 P2 提供独立合法性 | **P1 路线图**（不是 P2/P3） | **最高**（11 场景） |
| serial 吸收 / 多段 program | 拒绝（24） | ❌ | P1 增加 serial 轴模型，或 P2 独立合法性 | **P1 路线图** | **最高**（24 场景） |
| 语句级控制流 / 守卫 | 拒绝（8） | ❌ | P1 打开 `allow_control_flow_` + 谓词字段 | **P1 路线图** | 高（8 场景） |
| 地址边界 / 越界守卫 | 未建模 | ❌ | P1 lane/bounds 模型，或 P2 `_guard_free_bounds` 独立证明 | P1 或 P2 | 中高 |
| GM / 读侧 true_random / bool store | P1 可 lift，P2 拒绝（3） | ❌ | P2 增加对应 recipe / 校验 | **P2/P3** | 中（3 场景） |
| P2/P3 发射质量 | 已激活场景内 | ✅ | 指令数/寄存器压力/编译时间回归 | P2/P3 | 按性能数据 |

**优先级原则**：

1. **先按场景数量与算子热度**：serial 24、归约 11、控制流 8 是最大缺口，但它们是 P1 模型缺口，不是 P2/P3 迭代能解决的（在“暂不扩展 P1”约束下）。
2. **P2/P3 的迭代只对分支 B（3 个场景）和已激活场景的发射质量负责**；不要为了分支 C 去改 P2/P3，否则会绕过 P1 合法性。
3. **分支 B 的优先级按实际算子频率调整**：如果某个高频算子命中 GM/true_random/bool，可提前；否则低于 P1 路线图。
4. **P1 v2 到来后**：按能力位逐个打开，先开覆盖场景最多的能力（serial、归约），再开控制流/守卫；每开一个能力跑全量 66 场景回归。

#### 6.6.3 运营指标

| 指标 | 用途 |
| --- | --- |
| `Planned` 数 / 场景 | 衡量 P1+P2+P3 实际激活面 |
| `SkippedByCapability` 数 / 能力位 | 衡量 P2 侧能力门禁的缺口 |
| `RejectedByP1` 数 / 原因 | 衡量 P1 模型缺口（归约/CF/serial） |
| `RejectedByP2` 数 / 诊断 | 分支 B 的迭代输入 |
| legacy 回退率 | 衡量未向量化比例 |
| 编译时间 / IR 体积 / 指令数 | 衡量已激活场景的发射质量 |

### 6.7 本节结论

1. **执行链路**：三个 P1 Pass 独立、只读；`AscendAnalyzeVSimdRegionReport` 是唯一的 A+B+C 组合入口，返回字符串；它不返回 typed IR、不改写 IR。
2. **中间表示**：typed `VSimdRegion` 由 `BuildMinimal1DRegion`（L44-101）构造；L298 是组合/渲染入口；字段与生成规则见 §6.3；当前没有 P1→P2 的可执行载体或 FFI。
3. **“改写”**：L326 的 `ExpandVSimdRegionToLoops` 是**标量参考展开**，用于自检与 `expanded:` 文本；P1 不改写原始 nest；真实向量化在 P3/legacy。
4. **P2/P3 承接**：不能“直接承接”；需要 P1→P2 adapter。运行时能力只能保留 20/55；代码/测试资产可全部保留。
5. **路由**：A（20 个正常全流程）、B（3 个 P1 可向量化但 P2/P3 不足）、C（43 个 P1 不可向量化）、D（5 个多层直线桥）。只有 B 触发 P2/P3 迭代；C 触发 P1 路线图。

---

## 7. 最终回答（对应四个问题）

### 7.1 对本次 `vsimd_region.cc` 核实的直接回答

1. **P1 三个 Pass 的执行链路**：三者独立、只读；`AscendAnalyzeVSimdRegionReport` 是唯一的 PR-A+B+C 组合入口，返回字符串；它不返回 typed IR、不改写 IR。详见 §6.1/§6.2。
2. **P1→P2 中间表示**：typed `VSimdRegion` 由 `BuildMinimal1DRegion`（L44-101）构造；L298 在 L316 调用它并渲染文本；当前没有 P1→P2 的可执行载体或 Python FFI。详见 §6.3。
3. **L326 的“改写”**：`ExpandVSimdRegionToLoops` 是**标量参考展开**（`ForKind::kSerial`），用于自检与 `expanded:` 文本；P1 不改写原始 `SIMD_VF` nest；真实向量化在 P3/legacy。详见 §6.4。
4. **P2/P3 直接承接与能力保留**：不能“直接承接”，需要 P1→P2 adapter；运行时能力只能保留 20/55，代码/测试资产可全部保留。详见 §6.5/§6.6。

### 7.2 此前四个问题的回答（保留）

1. **P1 输出是否为控制流/归约/多层 parallel 预留了标识、字段、接口？**
   **没有输出级预留。** 只有 `allow_control_flow_` 代码开关（为 CF1–CF3 预留、默认关闭、不进入报告）、`VSimdRegion` 的“未来可执行载体”注释、以及 `domain` 的 vector 容器（可放多层但 PR-C G3 强制单层）。`LoopInfo` 无 kind/轴角色；`Legality` 无 REDUCTION；`AccessInfo` 无 mask；`VSimdRegion` 无 ReduceInfo/语句树。文本还缺 `bindings`、dtype/shape/strides、逐访问 class。

2. **仅凭 P1 输出能否完整完成 P2/P3？**
   **不能。** 文本层连基线都不够；typed region 只覆盖 66 场景中的 18 个可 lift（与 P2 OK 交集 15 个）；加上 PR-B `VSimdCaptureInfo` 的多层直线桥可到 **20 个可规划场景**（约 36% 的 P2 OK 集）；归约 11、语句级控制流 8、串行 24 全部缺失；P3 的输入是 P2 的 `VectorProgram`，P1 单独无法提供 lane/chunk/掩码/recipe/指令。

3. **路线 1 vs 路线 2 怎么选？**
   字面路线 2 **只有两个可用触发**：基础逐元素（`LEGAL` + 单层）与多层直线（`LEGAL` + `CLOSED` + `domain.size()>1` + G3 fallback）；归约/语句级控制流/serial/多段/守卫没有正向触发，按“无触发即跳过”执行时，路线 2 与“接入 PR-B 桥的路线 1”运行时能力面相同（20/55）。**推荐路线 2 的工程修正版：保留全部 P2 代码，但用 P2 侧显式、版本化的 `P1CapabilitySet` 做门禁（v1 含 basic + multi_level_straightline，高级能力位 false）；任何 Planned nest 仍必须通过 P1 LEGAL。** 若团队要求最小代码、一次交付，则选路线 1（同样要接 PR-B 桥，否则只有 15/55）；不推荐路线 2′（TIR 检测 + 绕过 P1 合法性）。

4. **落地与兼容保障**
   见 §4：`RegionClassifier + P1CapabilitySet + PlanOutcome` 三层门禁、core/ext/capabilities 代码分层、PR-B 多层直线桥、S1–S8 实施步骤、§4.4 测试标准、§4.5 契约/能力位版本化与 v1/v2 双适配器。核心不变量：`Planned ⇒ P1 LEGAL ∧ capability ∧ (单层 ∨ 多层直线桥条件)`。
