# Ascend `T.Parallel` 三阶段（P1→P2→P3）跨模块对接与 C++ 改造方案

> 目标数据流（严格按 `Parallel_ascend.md` 顶层设计）：
>
> ```
> 原始 IR
>   │  P1  AscendLiftParallelToArray（只读函数体，产出结构化区域载体）
>   ▼
> LogicalArrayRegion（硬件无关：迭代域 + body 子树 + captures/assumptions/bindings + legality）
>   │  P2  AscendPlanArray
>   ▼
> VectorProgram（指令选择无关：轴/lane/chunk/尾块 + 逐值/逐访问 recipe + diagnostics）
>   │  P3  AscendLowerArrayPlan
>   ▼
> TIR + tl.simd.*（真实 Ascend 向量指令）
> ```
>
> 三个 Pass 连续执行；P3 之后未处理的 `T.Parallel` 仍由既有
> `AscendSimdVFLowerParallel` 兜底，保证灰度期可回退。

本文档覆盖现状差异、对接设计、C++ 迁移、测试验证与分阶段执行计划。

---

## 0. 结论摘要

1. **P1 当前不在目标分支**：P1（PR-A/PR-B/PR-C，6 个提交）位于 `stable-parallel` 分支，
   目标工作树 `/home/cjl/workspace/tilelang` 的 `Parallel_lowering` 分支不包含这些文件。
   两者共同祖先是 `c55eeb1c`，`stable-parallel` 领先 28 个提交、`Parallel_lowering` 领先 9 个提交。
   第一步必须做分支收敛，否则“P1→P2→P3”无法在同一编译器里链接。
2. **P1 现状是“最小 1-D 只读分析原型”，不是顶层设计要求的完整 `LogicalArrayRegion`**：
   - `BuildMinimal1DRegion` 只接受**单层** `T.Parallel`（G3）；
   - `NestCollector` 默认拒绝 `IfThenElse`、`T.serial`、`local.var`、`Evaluate`；
   - 产出的是扁平 `accesses` 列表，没有 body 语句树、没有 store 谓词、没有 pre/post 分段。
3. **P2/P3 现状是离线 Python 原型，输入是原始 `tirx.Stmt` + 自建 `arith.Analyzer`**：
   `build_plan` 直接遍历原始 `T.Parallel` For，完全不消费 P1 的 `VSimdRegion`；
   `emit_plan` 输出 `tl.simd.*` TIR。能力面覆盖 66 个场景 / 135 个用例，远大于 P1 当前放行面。
4. **不能只做“字段搬运式适配器”**：P2 需要的 body 结构（多语句、if/else、serial、pre/post、mask）
   在 P1 输出里不存在。必须把 P1 扩展成**结构化区域载体**（detached TIR 子树 + 上下文 + 逐 nest 合法性），
   否则要么 P2 退化到 P1 的单层直线子集，要么 P2 偷偷回读原始 IR（违反“唯一基准输入”）。
5. **推荐架构**：P1 v2 输出 `LogicalArrayRegion{nest_root, domain, context, legality, locator}`；
   适配层只做校验/剥离声明/注入分析上下文，不做语义翻译；P2/P3 只依赖该契约，不包含任何
   `PrimFunc`/`IRModule`/`SIMD_VF` 依赖；P2→P3 用注册的 `VSimdArrayPlan` Object 经 `IRModule` 属性传递。
6. **验收基线分两层**：P1 当前六道闸门放行的场景作为**必过基线**；
   `polyhedral_parallel_scenarios.md` 的 66 个场景作为**扩展验收**。任何 P1 判 `ILLEGAL` 的场景
   不得为了“能力对齐”而放行——合法性判断始终以 P1 为权威。

---

## 1. 现状梳理

### 1.1 P1 输出契约（C++ 现状）

来源：`/home/cjl/workspace/tilelang-stable/src/ascend/transform/`（分支 `stable-parallel`）。
说明文档：`/home/cjl/workspace/tilelang-stable/docs/compiler_internals/vsimd_p1_analysis.md`。

#### 1.1.1 三个阶段的职责与产物

| 阶段 | 入口函数 | 产物 | 粒度 |
| --- | --- | --- | --- |
| PR-A | `AnalyzeVSimdMemoryStatuses(PrimFunc)` | `std::vector<std::string>`：`LEGAL/UNPROVEN/ILLEGAL/UNSUPPORTED/OPAQUE` | **每 VF 一个** |
| PR-B | `AnalyzeVSimdCaptureInfo(PrimFunc)` | `std::vector<VSimdCaptureInfo>` | **每 nest 一条**（`nest_index=-1` 表示整 VF 不可分析） |
| PR-C | `BuildMinimal1DRegion(info, memory_status)` | `VSimdRegionLift{lifted, internal_error, reason, region}` | 每 nest 一个 |

对外 Python API 共 7 个（3 个 Pass + 4 个查询），均为只读分析；三个 Pass 不在默认流水线中。

#### 1.1.2 核心数据结构

`vsimd::LoopInfo` / `vsimd::AccessInfo`（`vsimd_analysis_common.h`）：

| 结构 | 字段 | 语义 |
| --- | --- | --- |
| `LoopInfo` | `var, min, extent` | 一层循环；`ParallelNest::loops` 按外层→内层排列 |
| `AccessInfo` | `buffer, indices, value, is_write` | 一次访存；`value` 只在写访问时有意义，供参考展开器复现 store 值 |

`VSimdCaptureInfo`（PR-B 主产物，`vsimd_capture_scope.h`）：

| 字段 | 语义 |
| --- | --- |
| `vf_index / nest_index` | 定位（VF 序号 / VF 内 nest 序号） |
| `vf_status / status / message` | VF 聚合态（`CLOSED/OPEN/OPAQUE`）/ 本 nest 态 / 原因 |
| `domain` | 并行域（外层→内层），仅 `T.Parallel` 层 |
| `accesses` | 全部读写，源码顺序；写访问携带 `value` |
| `captures` | 去重 buffer + 种类（`param/alloc/local/view`） |
| `outer_vars` | 非域变量 + 种类（`thread/outer_loop/bind`） |
| `shape_syms` | shape/stride/extent 中的自由符号 |
| `local_views` | `view buffer -> base buffer` |
| `layouts` | 捕获 buffer 的 `Layout`（供 P2/P3 选 `vsts` vs `vsstb`） |
| `assumptions` | 去重后的相关 `tl.assume`（纯布尔算术） |
| `bindings` | 相关的 `Bind` 定义（如 `tail = min(...)`） |
| `errors` | 闭包错误；非空即 `OPEN` |

`VSimdRegion`（PR-C 逻辑载体，`vsimd_region.h`）：

| 字段 | 语义 |
| --- | --- |
| `vf_index / nest_index` | 源定位 |
| `domain` | **恰好一层** `T.Parallel` |
| `accesses` | 源码顺序的读写；写访问带 `value` |
| `captures / outer_vars / shape_syms / local_views / layouts / assumptions / bindings` | 与 `VSimdCaptureInfo` 同义，去掉状态/错误字段 |

`VSimdRegion` 头文件明确声明：**不含物理布局、lane 或指令选择**；它是 P1→P2/P3 的逻辑交接，
不是可执行 IR。当前“可执行载体”尚未实现，只有 debug 用的 `AttrStmt("tl.vsimd.region", dump_text)`。

#### 1.1.3 语义约束与准入闸门

`BuildMinimal1DRegion` 的六道闸门：

| 闸门 | 条件 | 失败文案 |
| --- | --- | --- |
| G1 | `info.status == "CLOSED"` | `capture/scope is not CLOSED` |
| G2 | `memory_status == "LEGAL"` | `memory legality is <STATUS>` |
| G3 | `info.domain.size() == 1` | `PR-C minimal lift only supports one T.Parallel level` |
| G4 | 无 `local.var` 访问 | `local.var access is not liftable` |
| G5 | 所有访问 index rank == buffer rank | `access rank does not match buffer rank` |
| G6 | 至少一个 `BufferStore` | `nest has no BufferStore` |

隐式语义：

- `NestCollector` 的语句白名单只接受 `BufferStore / SeqStmt / DeclBuffer / Bind / Evaluate /
  For(kParallel)`；`Evaluate` 随后被显式拒绝；`IfThenElse`、`For(kSerial)`、`While` 默认拒绝
  （`allow_control_flow_` 默认 `false`）。
- `BufferStore` 的 `accesses` 顺序是“先写记录，再递归 value 内的读”，读访问与写访问混在一条扁平列表里。
- `local.var` 在 `ManualDetector`、`NestCollector` 两处都判不可分析；前者记 `OPAQUE`，后者记
  `UNSUPPORTED`（PR-A）/`OPAQUE`（PR-B）。
- PR-A 的合法性是**每 VF 取最差**；PR-C 用 `statuses[info.vf_index]` 给同一 VF 的所有 nest 取同一状态，
  因此同 VF 多 nest 时“一损俱损”。
- PR-A 与 PR-B 各自独立遍历 `SIMD_VF`，顺序一致性只靠约定，无运行时断言。
- 写映射要求单射（R1）、跨迭代无 RAW/WAR/WAW（R2）；数据相关/`select`/`min`/`max` 索引判
  `UNPROVEN`，即**不放行**；`i//C`、`i%C` 走 F3 快速通道或 Z3。

#### 1.1.4 与顶层设计契约的差距

`Parallel_ascend.md` §4 要求 `LogicalArrayRegion` 至少包含：

- `domain`：每轴循环变量、下界、范围、步长；
- `captures`：域外标量/缓冲；
- `nodes`：拓扑排序的值 DAG（`Index / Uniform / Read / Elementwise`）；
- `stores`：有序写（buffer / 目标坐标 / 值）；
- `assumptions`：分析依赖的约束及来源。

现状 `VSimdRegion` 只有 `domain + accesses`，没有 `nodes` 和独立的 `stores`；写访问的
`value` 是原始表达式，P2 需要自己重建值 DAG。**这是 P1 结构性缺口，不是字段命名差异。**

---

### 1.2 P2/P3 现状（Python 原型）

来源：`/home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/`（分支 `Parallel_lowering`）。
设计文档：`tilelang/ascend/analysis/polyhedral_design.md`；场景目录：
`docs/compiler_internals/polyhedral_parallel_scenarios.md`。

#### 1.2.1 模块与职责

| 模块 | 行数 | 输入 | 核心逻辑 | 输出 |
| --- | --- | --- | --- | --- |
| `primitives.py` | 255 | — | 算子事实表、`Unsupported`、小工具 | 事实与诊断类型 |
| `access.py` | 301 | buffer + indices + `Axis[]` | 线性化、有限差分 Δ、`classify_access` | `AccessInfo`（逐轴 `AccessClass`） |
| `body.py` | 151 | body `Stmt` | `normalize_body`：多语句 / if 谓词化 / 同址 if-else 合并 | `StoreSpec[]` |
| `value.py` | 697 | store value 表达式 | 值 DAG、`is_uniform`、装载守卫、索引向量、归约单位元 | `ValueNode` |
| `model.py` | 186 | — | `StorePlan / ReduceInfo / VectorPlan / VectorProgram` | 计划数据结构 |
| `plan.py` | 1352 | 一个 `T.Parallel` For | `_split_nest`、轴排序/回退、归约判定、多段 program、诊断 | `VectorProgram` |
| `emit.py` | 1491 | `VectorPlan / VectorProgram` | chunk strip-mine、掩码、归约短/长策略、多段发射 | `tl.simd.*` TIR |
| `render.py` | 55 | 计划 | 可读渲染 | 文本 |
| `driver.py` | 35 | 整棵语句树 | 自顶向下替换 `T.Parallel` 嵌套 | 变换后的 `Stmt` |

#### 1.2.2 现有输入依赖（关键）

- `build_plan(stmt, *, lanes=None)` 的 `stmt` 是**原始 `tirx.Stmt`**（某个 `T.Parallel` For）；
  内部 `arith.Analyzer()` 在函数体内新建（`plan.py:828`），不消费任何外部假设；
- `_split_nest` 从该 For 出发，连续吸收 `T.Parallel` 轴；再吸收“直接内嵌且干净”的 `T.serial`
  轴；当平行链结束后是 `SeqStmt` 且**恰好一个**干净 serial 循环时，切出 `pre / main / post`；
- `normalize_body` 只认 `SeqStmt / BufferStore / BufferLoad / Evaluate / IfThenElse / For / Bind`；
  其它语句（含 `AllocBuffer`、`DeclBuffer`）一律 `Unsupported`；
- `_collect_aliases` 在整段 body 上收集 `Bind` 值并内联（跳过 Buffer / Call 值）；
- `classify_access` 从 buffer 的 `shape/strides/elem_offset` 和 indices 线性化，再对每根轴求 Δ；
- `emit_plan` 直接构造 `tl.simd.*` 调用，地址用 `BufferLoad(buffer, indices)` 交给
  `T.simd` 自动包 `access_ptr`（Python 侧行为，见 `simd.py:324-411`）。

#### 1.2.3 输出规范

`VectorProgram` 是 `build_plan` 的恒定返回类型：

- `program.axes`：提到最外层只包一次的**并行轴前缀**（`_outer_axes`），不含被吸收的 serial 轴；
- `program.segments`：1..N 个 `VectorPlan`；每段含 `axes / vector_axis / lanes / chunks / tail /
  stores / pre_stores / post_stores / reduce / guard_boundaries / diagnostics`；
- `StorePlan` 含 `spec（StoreSpec）/ access（AccessInfo）/ value（ValueNode）/ mask / kind /
  index / reduce / promote / mask_free_until / mask_free_from`；
- `ReduceInfo` 含 `axis / op / recipe / varying / accumulate`；
- 计划中**没有** `T.simd.*` 指令名；指令选择只在 `emit.py`。

`emit_plan` 输出 `TIR`（`SeqStmt / For / Bind / BufferStore / Evaluate` + `tl.simd.*` Call），
覆盖 `vld / vgather2 / vdup / vci / vadd... / vadds... / vcmp(s) / pand/por/pnot / pset /
update_mask / vsel / vcvt / vcadd/vcmax/vcmin / vsts / vscatter`。

#### 1.2.4 能力面（摘要）

| 结构 | 支持 | 关键机制 |
| --- | --- | --- |
| A 单语句逐元素 | A1–A13（A8 间接、A5/A6/A7 散列/置换/分片） | Δ 分类 + 索引向量通道 |
| B 单语句归约 | Add/Max/Min；短/长/尾块/符号 extent | `ReduceInfo` + `vcadd/vcmax/vcmin` |
| C 多语句 | 多 store、标量中间值、strided 写、复杂标量 | `normalize_body` + `ValueNode` |
| D 控制流 | 单臂谓词化、同址 if/else 惰性合并、三元、臂内越界访存守卫 | `mask` + `if_then_else` + `guard_boundaries` |
| E 单个 serial 段 | 直接内嵌 / 夹在语句序列中的干净 serial | serial 轴吸收 + `pre/main/post` |
| F 并列多个 serial | ≥2 段 | `VectorProgram` 多段、源序发射、共享外层轴 |
| G 拒绝 | 发散分支、atomic、嵌套 serial、非 0 下界、跨迭代 WAW、非 2 幂常量除、GM 操作数 | `Unsupported` + 诊断 |

规模基线：135 个 pytest 用例；`test_drive_plan.py` 66 个场景，55 OK / 11 REJ。

---

### 1.3 P1 输出与 P2/P3 输入的完整差异清单

记号：`p1` 指 `stable-parallel` 的 PR-A/B/C 现状；`p2` 指 Python 原型。

#### 1.3.1 数据结构 / 字段级差异

| 编号 | p1 现状 | p2/p3 需求 | 差异性质 | 影响 | 适配/改造动作 |
| --- | --- | --- | --- | --- | --- |
| D01 | 交付单元：`VSimdCaptureInfo`（每 nest）+ `VSimdRegion`（每 nest，仅单层） | `build_plan(stmt)` 的输入是一个**最外层 `T.Parallel` For** | 单元/形态不一致 | P2 无法从 region 直接得到可规划语句 | P1 v2 增加 `nest_root`（detached 完整 nest 子树） |
| D02 | `domain` 只允许 1 层（G3） | `_split_nest` 支持 N 层并行链 | 能力缺口 | 2-D/3-D 场景在 P1 被拒 | P1 v2 去掉 G3；保留 N 层 `domain` 供校验 |
| D03 | 只收集 `For(kParallel)`；`kSerial` 默认拒绝 | 吸收直接内嵌/夹在 SeqStmt 中的干净 serial 轴 | 能力缺口 | E/F 类场景全部不可达 | P1 v2 的结构化 lift 保留 serial 子树；P2 复用 `_serial_is_clean` |
| D04 | `IfThenElse` 默认拒绝 | 语句级 if → 谓词化 store；同址 if/else → 惰性 `if_then_else`；臂内访存挂守卫 | 能力缺口 | D 类场景全部不可达 | P1 v2 保留 `IfThenElse` 子树；P2 `normalize_body` 原样消费 |
| D05 | `accesses` 是**扁平列表**（写+读混排，源码顺序） | `StoreSpec[]` 有序写列表 + `pre/main/post` 分段 + `SeqStmt` 结构 | 结构缺失 | 无法恢复语句分组/分段/多段 program | P1 v2 携带 `nest_root`；适配层只保留写访问，读由 value 表达式内嵌 |
| D06 | `AccessInfo` 无 `mask` 字段 | `StoreSpec.mask`（语句级谓词） | 字段缺失 | 单臂 if、guard 边界不可表达 | 由 `nest_root` 的 `IfThenElse` 承载；P2 规范化时生成 mask |
| D07 | 写访问 `value` 是原始 `PrimExpr`；读访问单独成条 | `ValueNode` DAG（`vload/vmuls/vadd/vsel/...`） | 表示层差异 | 读访问若被当作独立语句会重复发射 | 适配层忽略“纯读访问条目”；P2 从 store value 重建 DAG |
| D08 | `bindings` 是 `Var -> PrimExpr`（只含相关的外层 Bind） | `_collect_aliases` 只扫描 nest 体内的 Bind；外层 Bind 不在扫描范围 | 作用域差异 | `tail = min(...)` 等外层绑定不参与 P2 分析 | 适配层把 `bindings` 作为 `PlanOptions` 的分析上下文（可选）；不强制内联，保证 P3 原位发射时变量仍在作用域内 |
| D09 | `assumptions` 是纯布尔 `PrimExpr` 列表 | `build_plan` 内部新建 `arith.Analyzer()`，完全不消费假设 | 语义缺口 | P1 已证明的约束在 P2 规划期丢失 | P2 增加 `PlanOptions{assumptions, bindings, analyzer_mode}`；迁移期用 `legacy_parity` 模式保证与 Python 逐场景一致 |
| D10 | 合法性状态**每 VF 一个** | 计划/诊断**每 nest 一份** | 粒度不一致 | 同 VF 多 nest 一损俱损 | P1 v2 提供逐 nest `legality`；PR-A 新增 per-nest API |
| D11 | 合法性语义：写单射 + 跨迭代无 RAW/WAR/WAW；数据相关/`select` 索引 `UNPROVEN` | 支持 STRIDED / PERMUTE / PIECEWISE / INDIRECT 写（`vscatter`） | 权威边界冲突 | A8 等间接写会被 P1 拒绝；反之 P2 对重复地址 scatter 无单射证明 | **P1 始终是合法性权威**；P1 放行面即基线。间接/无结构写作为扩展，需 P1 增加显式 scatter 契约后才放行 |
| D12 | `local.var` 一律 `OPAQUE`/`UNSUPPORTED`；`ManualDetector` 检查 **SBlock 自身** 的 `alloc_buffers`，因此 VF 内（nest 外）的 `T.alloc_var` 也会把整个 VF 判 OPAQUE | 私有标量提升（S2）：`local`/`local.var` RMW 提升为向量寄存器 | 能力冲突 | S2 场景在 P1 被拒；仅放宽 nest 内 `local.var` 不够，SBlock 级 detector 仍会短路 | P1 v2 结构化模式增加 `local_var_as_capture` 开关：SBlock/nest 内的 `local.var` 登记为 capture 而非 OPAQUE；`tl.simd.*`/`tl.ascend_nd2nz*` 仍判 OPAQUE；legacy 报告行为不变 |
| D13 | `captures` 种类 `param/alloc/local/view` | P2 需要 buffer scope（尤其 `global` 判拒） | 字段可映射 | 无阻断 | 适配层按 `Buffer::scope()` 补齐/校验 `in_global` |
| D14 | `layouts: Buffer -> Layout`；`layout_changing` 只在 PR-A 报告里 | P2/P3 当前未消费；设计上用于 `vsts` vs `vsstb` | 预留字段未接线 | 暂无功能影响 | 契约保留；P3 增加 layout-aware store 作为扩展 |
| D15 | `accesses` 要求 index rank == buffer rank（G5） | `linear_element_offset` 对 store 同样要求 rank 匹配 | 一致 | 无 | 适配层保留 G5 诊断 |
| D16 | `LoopInfo.min` 任意 | `_require_zero_min` 要求可证为 0 | 语义一致 | 非 0 下界明确拒绝 | 适配层把 `min` 传给 P2；P2 保留 R5 诊断 |
| D17 | `shape_syms` + `bindings` 支撑符号 extent | D9 符号 extent 走运行时 chunk 循环；非归约/归约都支持 | 表示层差异 | 符号 extent 可支持 | 域变量 + shape_syms 原样进入 `PlanOptions`；`legacy_parity` 模式下 analyzer 只绑定轴范围 |
| D18 | 无 dtype/lane 信息 | `lanes = 2048 / elem_bits`；`_require_supported_element` | 可派生 | 无阻断 | P2 从 `Buffer::dtype` 派生；P1 不新增 field |
| D19 | 无 `AllocBuffer`/`DeclBuffer` 语句的显式契约 | `normalize_body` 不认识 `AllocBuffer`/`DeclBuffer`，会 `Unsupported` | 语句白名单不一致 | nest 内分配/视图声明会失败 | 适配层剥离 `DeclBuffer`（登记到 `local_views`）；nest 内 `AllocBuffer` 明确拒绝并给诊断 |
| D20 | `VSimdRegion` 无 nest 定位指纹 | P3 需要把 plan 映射回原 nest | 定位缺失 | 无法安全替换 | P1 v2 增加 `(vf_index, nest_index)` + `structural fingerprint`；P3 替换前做结构校验 |
| D21 | PR-A 与 PR-B 两次独立遍历 | 需要同序对齐 | 隐式约定 | 未来改动可能错位 | P1 v2 改为单次遍历同时产出 legality/capture/region；legacy 报告保留 |
| D22 | P1 只读，不写 IR | P3 必须替换 nest | 阶段职责差异 | 无 | 替换只发生在 P3；P1/P2 对函数体 `structural_equal` |

#### 1.3.2 阻断性差异（必须先解决）

| 编号 | 阻断点 | 不解决的后果 |
| --- | --- | --- |
| B1 | P1 代码不在 `Parallel_lowering` 分支 | 无法链接、无法注册 Pass |
| B2 | P1 只支持单层、无 if/serial、无 body 树 | P2 的 D/E/F 类场景全部不可达；只能做 1-D 直线子集（A1/A2/A5/A6/A12/A13/C1/C2） |
| B3 | P1 合法性按 VF 聚合 | 一个 VF 内多 nest 时误拒 |
| B4 | P2 直接消费原始 `tirx.Stmt` | 违反“P1 输出唯一基准输入”，无法证明数据流分层 |
| B5 | P2/P3 无 C++ 计划载体 | 两个 Pass 之间无法传计划，只能融合成一个 Pass |
| B6 | P1 的 `local.var`/间接写策略与 P2 能力面冲突 | 要么功能退化，要么绕过 P1 合法性 |

---

## 2. 对接方案设计

### 2.1 原则：P1 是唯一输入与唯一合法性权威

- **唯一输入**：P2 的公共 API 只接受 `LogicalArrayRegion`；P2/P3 的**分析与发射库**禁止出现
  `PrimFunc / IRModule / SBlock / SIMD_VF` 等原始 IR 概念（用编译期 include 白名单 + CI grep 断言）。
  只有 Pass 入口层可以接触 `IRModule` 属性与函数体。
- **唯一合法性权威**：P1 的 `legality` 是门禁；P2 只做“轴选择/分块/recipe 绑定”，
  不重新证明并行域的单射性与跨迭代独立性。P2 的 `_serial_is_clean` 属于**向量轴吸收的
  规划前置条件**，抽到共享 `vsimd_dependence` 模块，由 P2 调用；P1 不重复实现。
- **责任边界**：
  - P1：lift 语义（迭代域 + body 子树 + 上下文 + 合法性），不碰 lane/指令/target；
  - P2：分块与 recipe（轴、chunk、尾块、mask、诊断），不发 `tl.simd.*`；
  - P3：按 recipe 发真实指令，不重新选轴、不改分块、不重新推断连续性。

### 2.2 P1 v2 契约：`LogicalArrayRegion`

在既有 `VSimdRegion` 基础上增加“可规划语句根”和“逐 nest 合法性”，不引入新 TIR Stmt 子类：

```cpp
struct ArrayAxis {                 // 并行轴（外层→内层）
  tirx::Var var;
  tirx::PrimExpr min;              // 必须可证为 0 才进入 P2
  tirx::PrimExpr extent;
  int index;                       // 轴序号，0 为最外层
};

struct LogicalArrayRegion {
  int vf_index{-1};
  int nest_index{-1};

  // P1 合法性（逐 nest）；字符串与 P1 报告/测试一致
  std::string legality;            // "LEGAL" 才进入 P2
  std::string legality_reason;     // ILLEGAL/UNPROVEN/UNSUPPORTED 的原因

  std::vector<ArrayAxis> domain;   // 并行轴；不含 serial 轴
  tirx::Stmt nest_root;            // detached 完整 nest 子树（含并行 For + body）

  // 以下与 VSimdRegion 同义，原样携带
  std::vector<std::pair<tirx::Buffer, std::string>> captures;
  std::vector<std::pair<tirx::Var, std::string>> outer_vars;
  std::vector<tirx::Var> shape_syms;
  std::vector<std::pair<tirx::Buffer, tirx::Buffer>> local_views;
  std::vector<std::pair<tirx::Buffer, Layout>> layouts;
  std::vector<tirx::PrimExpr> assumptions;
  std::vector<std::pair<tirx::Var, tirx::PrimExpr>> bindings;

  // P3 定位：替换前做结构校验，失败则回退
  std::string nest_fingerprint;    // StructuralHash(nest_root) 的稳定字符串
};
```

设计要点：

1. **`nest_root` 是 detached TIR 子树**，不是新的节点类型。它包含 `For(kParallel)`、
   `For(kSerial)`、`IfThenElse`、`SeqStmt`、`Bind`、`BufferStore`，与原始 IR 共享
   `Buffer/Var` 对象引用，但不包含父节点/模块引用。
2. 这样 P2 可以**逐行复用现有 Python 逻辑**：`_split_nest` 接收 `nest_root`，
   `normalize_body` 接收 body，`classify_access` 接收 indices——只是入口从
   `tirx::Stmt`（原始 IR）换成 `region.nest_root`（P1 输出）。语义上仍是“唯一基准输入”。
3. `domain` 是并行轴的权威副本；P2 启动时校验 `nest_root` 的并行链与 `domain` 的
   `(var, min, extent)` 一一对应，不一致即拒绝（防御 P1 自身缺陷）。
4. `legality` 逐 nest 化：PR-A 增加 `AnalyzeVSimdMemoryStatusesPerNest(func)`
   （返回 `vector<vector<string>>` 或 `vector<pair<vf,nest,status>>`）；
   legacy 报告仍用 per-VF 聚合，保证 49 个 P1 测试不变。
5. 单次遍历：新的 `LiftStructuredRegions(func)` 在一次 `StmtExprVisitor` 中同时完成
   manual 检测、nest 枚举、capture/scope、legality 关联，避免 PR-A/PR-B 两次遍历错位。

#### 与顶层设计 `nodes` 字段的对应关系

顶层设计把 `nodes`（`Index / Uniform / Read / Elementwise` 的拓扑 DAG）列在 P1 输出中。
本方案用 `nest_root` 作为 `nodes` 的**唯一语义来源**：

- `nest_root` 完整保留了产生 `nodes`/`stores` 所需的全部信息（循环变量、索引表达式、
  值表达式、写顺序）；从它出发可无损物化顶层设计里的 `stores` 与 `nodes`；
- P2 的 `ValueNode` DAG 就是 `nodes` 层的物化实现；把它放在 P2 规划期构造，
  避免 P1 与 P2 各维护一份 DAG 导致语义漂移；
- 若评审需要独立检查 `nodes`，提供只读 `DumpLogicalNodes(region)` 调试渲染，
  但它**不是**生产载体，不参与 P2/P3 数据流。

这样既满足“P1 输出可行性中间表示、P2 只消费 P1 输出”的分层约束，又避免重复实现值 DAG。

### 2.3 适配层：`RegionToPlanInput`

适配层是纯函数，位于 `vsimd_array_region.{h,cc}`，只做**校验 + 浅层语句过滤**，不做语义翻译：

```cpp
struct PlanOptions {
  int lanes{0};                    // 0 = 按元素位宽推导
  enum class AnalyzerMode { kLegacyParity, kRegionContract };
  AnalyzerMode analyzer_mode{AnalyzerMode::kLegacyParity};  // 迁移期默认与 Python 对齐
  std::vector<tirx::PrimExpr> assumptions;      // 来自 region
  std::vector<std::pair<tirx::Var, tirx::PrimExpr>> bindings;
  std::vector<std::pair<tirx::Var, std::string>> outer_vars;
};

struct PlanInput {
  std::vector<ArrayAxis> axes;
  tirx::Stmt nest_root;            // P2 的唯一语句输入
  PlanOptions options;
};

// 校验失败返回错误；不修改任何 IR（只构造过滤后的浅拷贝）
std::optional<PlanInput> BuildPlanInput(const LogicalArrayRegion &region,
                                        std::string *error);
```

适配规则：

| 步骤 | 规则 |
| --- | --- |
| V1 | `legality == "LEGAL"`，否则不进入 P2（保留 P1 诊断） |
| V2 | `domain` 非空且至少一根并行轴；`nest_root` 是 `For(kParallel)` 且并行链与 `domain` 一致 |
| V3 | 对 `nest_root` 做一次浅层语句过滤：剥离 `DeclBuffer`（登记到 `local_views`）；遇到 nest 内 `AllocBuffer` 立即返回错误（P2 不支持 nest 内分配，不静默丢弃） |
| V4 | `nest_root` 的自由变量必须 ⊆ `domain ∪ outer_vars ∪ shape_syms ∪ bindings.keys()`，否则拒绝 |
| V5 | `nest_root` 内访问的 buffer 必须 ⊆ `captures ∪ local_views`，否则拒绝 |
| V6 | `PlanOptions.assumptions/bindings` 原样注入；`kLegacyParity` 模式下 P2 忽略它们（仅绑定轴范围），用于与 Python 逐场景对齐 |
| V7 | 不内联外层 `Bind`：P3 原位替换时变量仍在作用域内；P2 只需把 `bindings` 作为可选分析上下文 |

**关键决策：不把外层 Bind 表达式内联进 `nest_root`。** 原因：
- 内联会改变表达式结构，破坏与 Python 原型的逐场景 parity；
- P3 在原 nest 位置替换，外层 `Bind` 仍在作用域内，直接引用 `Var` 是合法的；
- P2 需要证明时，通过 `PlanOptions.bindings` 注入 analyzer（`kRegionContract` 模式）。

### 2.4 P2→P3 计划载体

P2/P3 是两个独立 Pass，计划不能只存在于栈上。推荐：

```
IRModule attrs:
  "tl.vsimd.array_regions" : Array<VSimdArrayRegion>   // P1 产出
  "tl.vsimd.array_plans"   : Array<VSimdArrayPlan>     // P2 产出
```

- `VSimdArrayRegion` / `VSimdArrayPlan` 是注册的 TVM Object 节点（`TVM_FFI_DECLARE_OBJECT_INFO_FINAL`），
  内部持有 `std::shared_ptr<LogicalArrayRegion>` / `std::shared_ptr<VectorProgram>`；
- 同时提供 `ArrayRegionToFFI` / `VectorPlanToFFI`，把结构序列化成 `Map<String, Any>`，
  供 Python 测试直接断言与“传入 P1 输出”；
- P3 的输入只有计划与一份只读 `EmitContext`（不含任何分析器）：

```cpp
struct EmitContext {
  std::vector<std::pair<tirx::Buffer, Layout>> layouts;        // layout-aware store（扩展）
  std::vector<std::pair<tirx::Var, tirx::PrimExpr>> bindings;  // 地址表达式复用（只读）
  int lanes_override{0};
};

tirx::Stmt EmitPlan(const VectorProgram &plan, const EmitContext &ctx);
```

- 三个 Pass **必须连续执行**，中间不得插入普通优化（与顶层设计 §9 的 annotation carrier 约束一致）；
- P3 读取 plan 后，替换前用 `nest_fingerprint` 做结构校验；校验失败 → `LOG(WARNING)` + 跳过，
  由 legacy `AscendSimdVFLowerParallel` 兜底。

> 备选方案：若自定义 Object 节点成本过高，可先把 P2/P3 实现为同一个
> `AscendPolyhedralVectorize` Pass 内的两个阶段（仍保留独立 FFI 查询入口）。
> 该方案不改变 P2/P3 的职责边界，只放弃“两个独立 Pass 对象”。建议优先做正式的双 Pass 载体。

### 2.5 语义映射表

| P1 状态/信息 | P2 行为 |
| --- | --- |
| `legality == LEGAL` | 进入规划 |
| `legality == ILLEGAL/UNPROVEN` | 不规划；保留 P1 原因（诊断前缀 `P1: ...`） |
| `legality == UNSUPPORTED/OPAQUE` | 不规划；静默回退到 legacy（与 P1 报告一致） |
| `bindings/assumptions` | `kRegionContract` 模式下注入 analyzer；`kLegacyParity` 模式忽略 |
| `layouts` | 进入 plan 的 `EmitContext`；当前 P3 仍发 NORM，layout-aware store 作为扩展 |
| `local_views` | 供 P2 的 `_same_buffer` / 地址同一性判断；不参与发射 |
| `outer_vars/shape_syms` | analyzer 的符号；不参与发射 |
| `nest_fingerprint` | P3 替换前结构校验 |

### 2.6 关键决策（建议）

| 决策 | 建议 | 理由 |
| --- | --- | --- |
| P1 是否扩展 | **扩展为 P1 v2**（结构化区域） | 否则 P2 只能覆盖 P1 单层直线子集，D/E/F 类全丢；扩展不是改 P1 语义，而是补齐顶层设计要求的 `LogicalArrayRegion` |
| P2 输入 | **`LogicalArrayRegion`**，不是 `PrimFunc`/`Stmt` | 满足“唯一基准输入”；P2/P3 可独立测试 |
| 区域表示 | **detached TIR 子树** | 最大化复用 P2 现有 `_split_nest/normalize_body` 逻辑，降低 C++ 迁移风险 |
| P2/P3 Pass 形态 | **两个独立 Pass + 注册 Object 载体** | 符合顶层三阶段；便于观察和单测；失败可回退 |
| 间接/散列写 | **基线期保持 P1 拒绝**，扩展期增加显式 scatter 契约 | P1 的单射证明是正确性底线；不能为了能力对齐绕过 |
| Analyzer 模式 | 迁移期 `kLegacyParity`，验收后默认 `kRegionContract` | 先保证与 Python 逐场景一致，再启用 P1 假设增强 |

---

## 3. C++ 迁移与 Pass 接入

### 3.1 目标文件布局

所有文件放在 `src/ascend/transform/`，由现有 CMake `file(GLOB src/ascend/transform/*.cc)` 自动收集，
无需改构建脚本：

| 文件 | 职责 |
| --- | --- |
| `vsimd_array_region.h` | P1→P2 契约：`ArrayAxis / LogicalArrayRegion / PlanOptions / PlanInput` |
| `vsimd_array_region_lift.cc` | P1 v2 结构化 lift（单次遍历、多级、serial/if、local.var、per-nest legality） |
| `vsimd_dependence.h/.cc` | 共享依赖判定：`serial_is_clean`、跨地址读、RMW 判据（P2 使用，P1 可选） |
| `vsimd_plan_primitives.h/.cc` | 算子/recipe 事实表、dtype/mask 宽度、诊断类型 |
| `vsimd_plan_access.h/.cc` | 线性化、有限差分、`AccessClass`、`classify_access` |
| `vsimd_plan_body.h/.cc` | `StoreSpec`、`normalize_body`、别名收集 |
| `vsimd_plan_value.h/.cc` | `ValueNode` DAG、`is_uniform`、装载守卫、索引向量 |
| `vsimd_plan_model.h` | `StorePlan / ReduceInfo / VectorPlan / VectorProgram` |
| `vsimd_plan_build.h/.cc` | 轴排序/回退、归约判定、多段 program、`BuildPlan(PlanInput)` |
| `vsimd_plan_render.h/.cc` | `RenderPlan`（与 Python `render_plan` 同构）与诊断渲染 |
| `vsimd_emit.h/.cc` | `EmitPlan(VectorProgram, EmitContext)`：全部 `tl.simd.*` 构造 |
| `analyze_vsimd_region.cc` | 扩展现有 P1 报告入口：新增 v2 region dump / FFI |
| `plan_parallel_nests.cc` | P2 Pass + FFI 入口（三件套模式） |
| `lower_array_plan.cc` | P3 Pass + FFI 入口（三件套模式） |
| `vsimd_polyhedral_pipeline.cc` | 定位器、结构校验、载体读写、orchestration 辅助 |

### 3.2 Python → C++ 模块映射

| Python 模块 | 行数 | C++ 目标 | 迁移要点 |
| --- | --- | --- | --- |
| `primitives.py` | 255 | `vsimd_plan_primitives` | 事实表用 `static const` 表；`Unsupported` 改成 `PlanError{message, diagnostics}` |
| `access.py` | 301 | `vsimd_plan_access` | `tirx::Substitute` + `arith::Analyzer::Simplify`；常量判定用 `const_int_bound` |
| `body.py` | 151 | `vsimd_plan_body` | `Stmt` 模式匹配；`IfThenElse` 谓词化；别名用 `ExprDeepEqual` |
| `value.py` | 697 | `vsimd_plan_value` | `ValueNode` 用 `shared_ptr` DAG；守卫证明用 `arith::Analyzer` |
| `model.py` | 186 | `vsimd_plan_model` | 纯 C++ struct；`StorePlan.promote` 保持计划期单一决策 |
| `plan.py` | 1352 | `vsimd_plan_build` | 最大模块，按 `axis / reduce / program` 拆三个 cc，避免单文件过大 |
| `emit.py` | 1491 | `vsimd_emit` | 地址/掩码/斜坡/前向/寄存器直通；`tl.simd.*` 用 `src/ascend/op/builtin.h` 的 `simd_*()` 访问器 |
| `render.py` | 55 | `vsimd_plan_render` | 与 Python 文本格式对齐，供 golden diff |
| `driver.py` | 35 | `vsimd_polyhedral_pipeline` | 自顶向下定位 + 结构校验 + 替换 |
| `__init__.py` | 114 | `tilelang/ascend/transform/__init__.py` + FFI | 新增 Python 薄封装，不改既有导出顺序 |

### 3.3 Pass 注册与流水线接入

三个 Pass 采用与 P1 相同的“三件套”模式（Report / Pass / 静态注册）：

| Pass | 注册名 | FFI 名 | 行为 |
| --- | --- | --- | --- |
| P1 | `tl.AscendLiftParallelToArray` | `tl.transform.AscendLiftParallelToArray` | 读函数体，写 `tl.vsimd.array_regions`；函数体不变 |
| P2 | `tl.AscendPlanArray` | `tl.transform.AscendPlanArray` | 读 regions，写 `tl.vsimd.array_plans`；函数体不变 |
| P3 | `tl.AscendLowerArrayPlan` | `tl.transform.AscendLowerArrayPlan` | 读 plans，替换 nest 为 `tl.simd.*` TIR |

流水线插入点（`tilelang/ascend/pipeline.py`）：

```python
mod = ascend_transform.NormalizeBufferVersion()(mod)
mod = ascend_transform.AscendLiftParallelToArray()(mod)   # 新增 P1
mod = ascend_transform.AscendPlanArray()(mod)             # 新增 P2
mod = ascend_transform.AscendLowerArrayPlan()(mod)        # 新增 P3
mod = ascend_transform.AscendSimdVFLowerParallel()(mod)   # 既有兜底
mod = ascend_transform.AscendLowerTileOp()(mod)
```

约束：

- 三个 Pass 必须相邻；P2/P3 不依赖任何其它 Pass 的输出形态；
- 增加配置开关（建议 `TL_ENABLE_POLYHEDRAL_VECTORIZE`），灰度期默认关闭，验收后默认开启；
- P3 只替换“有 plan 且指纹校验通过”的 nest；其余留给 legacy emitter；
- P1/P2 对函数体 `structural_equal`；只有 P3 修改 IR；
- P3 不得调用 `arith::Analyzer`、`classify_access`、`build_value` 等分析函数（CI 架构断言）。

### 3.4 与 legacy 路径的共存

- `AscendSimdVFLowerParallel` 仍是唯一的“最后兜底”；P3 不做整 VF 替换，只替换单 nest；
- 若一个 VF 内部分 nest 被 P3 替换、部分回退，legacy emitter 仍能处理剩余的 `T.Parallel`；
- 需要新增回归用例：同一 VF 内“一个可规划 + 一个不可规划 nest”，验证混合 IR 不产生重复发射/漏发射；
- 当 `TL_ENABLE_POLYHEDRAL_VECTORIZE=0` 时，P1/P2/P3 全部 no-op，行为与当前 `Parallel_lowering` 完全一致。

### 3.5 编码与仓库规范

- 所有新文件加 `TVM_DLL` 导出声明与 include guard，风格对齐 `vsimd_region.h`；
- 诊断信息保持与 Python 原型**逐字一致**（便于 parity diff 与用户迁移）；
- `tl.simd.*` 调用必须使用 `src/ascend/op/builtin.h` 的 `simd_vld()/simd_vsts()/...`，
  禁止字符串 `Op::Get("tl.simd.xxx")` 散落；
- 地址包装必须复刻 `tilelang/ascend/language/simd.py` 的 `access_ptr` 规则
  （`vld` extent=lanes；`vsts` 的 `NORM_B{bits}` / `ONEPT_B{bits}`；`vgather2/vscatter` 的 base/idx）；
- 计划结构用 `std::shared_ptr<const ...>` 持有，避免拷贝大 DAG；
- 不做跨阶段缓存/全局状态；所有状态通过参数或 `IRModule` attrs 传递。

---

## 4. 测试验证方案

### 4.1 Python 侧调用入口（保留并扩展）

在 `tilelang/ascend/transform/__init__.py` 新增薄封装（与 P1 的 7 个 API 风格一致）：

| Python API | 返回 | 用途 |
| --- | --- | --- |
| `AscendLiftParallelToArray()` | `Pass` | P1 Pass |
| `AscendLiftParallelToArrayInfo(func)` | `list[dict]` | 直接取 P1 输出（`nest_root`、`domain`、context、legality） |
| `AscendPlanArray()` | `Pass` | P2 Pass |
| `AscendPlanArrayInfo(region_or_func)` | `list[dict]` | **直接传入 P1 输出**，返回计划 |
| `AscendPlanArrayReport(region_or_func)` | `str` | 计划文本 |
| `AscendLowerArrayPlan()` | `Pass` | P3 Pass |
| `AscendLowerArrayPlanInfo(plan_or_region)` | `Stmt / str` | 直接传入计划，返回发射 IR |
| `AscendVectorizeParallelNests()` | `Pass` | P1→P2→P3 一体化（测试/便捷入口） |

`AscendPlanArrayInfo` 必须同时支持：
1. `func`：内部跑 P1 再跑 P2（便捷）；
2. `region list[dict]`：**只消费 P1 输出**（验证唯一输入原则）。

### 4.2 分层测试矩阵

| 层级 | 测试对象 | 断言 |
| --- | --- | --- |
| L1 P1 契约 | `AscendLiftParallelToArrayInfo` | 每个 nest 的 `domain` 与 `nest_root` 并行链一致；context 闭包；legality 逐 nest 正确；函数体 `structural_equal` |
| L2 P1→P2 适配 | `BuildPlanInput` | 合法 region 通过；非法 region 给出稳定错误；`nest_root` 自由变量/buffer 闭包 |
| L3 P2 计划 | `AscendPlanArrayInfo` | 与 Python `build_plan` 的 plan 结构化字段逐项相等（axes / vector_axis / lanes / chunks / tail / store kind / reduce / guard_boundaries / diagnostics） |
| L4 P3 发射 | `AscendLowerArrayPlanInfo` | 与 Python `emit_plan` 的 IR 在 `map_free_vars=true` 下 `structural_equal`；`tl.simd.*` op 序列完全一致 |
| L5 Pass 流水线 | 编译管线 | P1/P2 不改函数体；P3 后目标 nest 无 `PARALLEL`；非目标 IR 不变；混合 VF 不重复发射 |
| L6 端到端 | 真实 kernel | IR 含预期 `vld/vsts/vgather2/...`；无残留 `T.Parallel`；编译不报错 |

### 4.3 基线场景与扩展场景

**基线（必过，P1 当前放行面）**：

- 单层 `T.Parallel`；
- 无 `IfThenElse` / `T.serial` / `local.var` / `Evaluate` / TileOp；
- 写映射单射（F1–F5 / Z3 `LEGAL`）；
- 覆盖：A1 连续拷贝、A2 广播、A5 正步长读（写仍单射）、A6 逆序读/写（仿射单射）、
  A12 尾块、A13 符号 extent、C1 多 store、C2 局部标量中间值（均取 **1-D 形式**）。
  A3/A4/A7 是 2-D 及以上形态，受 P1 v1 的 G3 限制，归入扩展批次，不列入基线。
- 用例来源：P1 的 49 个 pytest + 从 66 场景中筛出的“P1 可 lift”子集；
  对每个用例断言 `AscendLiftParallelToArrayInfo` 输出 `LEGAL` 且 `LIFTED`。

**扩展（逐批启用）**：

| 批次 | 场景 | 前置能力 |
| --- | --- | --- |
| X1 | A7 分片写 + A5/A6 的多维形态（步长/逆序/分片写） | P1 v2 多级/结构化；P2 `vscatter` 复用 |
| X2 | B1–B7（归约，含尾块/int/f16/符号 extent） | P2 归约迁移；P1 无新增 |
| X3 | C3–C6（多语句/标量/部分归约） | P1 body 树 + P2 `normalize_body` |
| X4 | D1–D5（if/else/三元/守卫/臂内越界） | P1 保留 `IfThenElse`；P2 guard 证明 |
| X5 | E/F（serial 吸收、多段 program） | P1 保留 serial 子树；P2 `_serial_is_clean` |
| X6 | A8（间接 gather） | P1 显式 scatter 契约（单独设计，不在基线） |

每批的验收：该批场景的 plan/emit parity 全绿，且前序批次不回归。

### 4.4 验收命令（建议固化到 CI）

```bash
# P1 legacy（不得回归）
pytest testing/ascend/transform/test_ascend_vsimd_memory_legality.py \
       testing/ascend/transform/test_ascend_vsimd_capture_scope.py \
       testing/ascend/transform/test_ascend_vsimd_region.py -q

# P2 原型 oracle（不得回归）
python tmp/run_poly_test.py

# 新增 P1/P2/P3 C++ 测试
pytest testing/ascend/transform/test_ascend_polyhedral_p1.py \
       testing/ascend/transform/test_ascend_polyhedral_plan.py \
       testing/ascend/transform/test_ascend_polyhedral_emit.py -q

# parity（逐场景）
python testing/ascend/transform/run_polyhedral_parity.py --all
```

---

## 5. 分阶段执行计划

> 每一阶段都以“可构建 + 可测试 + 可回退”为出口；未通过验收不得进入下一阶段。
> 阶段编号不代表时间承诺，只表示依赖顺序。

### Phase 0：分支收敛与基线固化

| 任务 | 内容 | 交付物 | 验收标准 |
| --- | --- | --- | --- |
| T0.1 | 从 `Parallel_lowering` 切 `feature/p1p2p3-integration`；cherry-pick P1 六个提交 `5bd8264e..af3f41ac`（或 merge `stable-parallel`，二选一） | 集成分支 | P1 文件出现在目标分支；无未解决冲突 |
| T0.2 | 构建（`USE_ASCEND=ON`、`USE_Z3=ON`）；跑 P1 49 用例 + P2 135 用例 + drive 66 场景 | 基线报告 | P1 49 pass；P2 135 pass；drive 55 OK / 11 REJ |
| T0.3 | 固化 CI 与基线快照（plan dump、op 序列、IR 文本） | CI 配置 + 快照 | CI 可重复；快照与当前 Python 逐字节一致 |

**出口**：同一分支上 P1/P2 均可构建、可测试；后续所有改动有基线可比。

### Phase 1：P1 v2 契约与结构化 lift

| 任务 | 内容 | 交付物 | 验收标准 |
| --- | --- | --- | --- |
| T1.1 | 定义 `LogicalArrayRegion / ArrayAxis / PlanOptions / PlanInput` 与校验函数 | `vsimd_array_region.h` | 头文件自包含；字段与 §2.2 一致 |
| T1.2 | 实现单次遍历的结构化 lift：多级并行、serial/if/seq 保留、`local.var` capture、per-nest legality、剥离 `DeclBuffer` | `vsimd_array_region_lift.cc` | 66 场景中所有 P1 可分析 nest 都产出 region；`nest_root` 与原 nest 结构相等 |
| T1.3 | 增加 per-nest 合法性 API；legacy per-VF 报告保持不变 | `vsimd_memory_legality.{h,cc}` 增量 | P1 49 用例全部不变；新增 per-nest 单测通过 |
| T1.4 | FFI + Python `AscendLiftParallelToArrayInfo` | Python 入口 + FFI 注册 | Python 可直接拿到 P1 输出并回传给 P2 |
| T1.5 | 适配层 `BuildPlanInput`（V1–V7 规则） | `vsimd_array_region.cc` | 合法/非法 region 的稳定诊断；自由变量/buffer 闭包单测 |

**出口**：P1 输出具备 P2 所需的全部结构信息；P1 旧行为零变化。

### Phase 2：P2 C++ 迁移（按场景族分批）

| 批次 | 任务 | 交付物 | 验收标准 |
| --- | --- | --- | --- |
| 2A | `primitives / access / body` 迁移 | `vsimd_plan_primitives/access/body` | 与 Python 对应函数的单测逐项一致（Δ 分类、`StoreSpec` 列表、诊断文本） |
| 2B | `value` 迁移（DAG、uniform、guard、索引向量） | `vsimd_plan_value` | value DAG 单测 parity（recipe 序列 / `is_uniform` / 守卫区间）；无参数叶子的 uniform 判定不回归 |
| 2C | `model / plan` 迁移（轴排序、回退、归约、多段 program） | `vsimd_plan_model/build` | A1–A13、B1–B7、C1–C6、E/F 的 plan parity；`VectorProgram.axes` 只含并行轴 |
| 2D | 诊断 + 渲染 + FFI | `vsimd_plan_render`、`plan_parallel_nests.cc` | P1 放行场景的 diagnostics 逐条一致；Python 可直接传 P1 输出得到 plan |

**出口**：`AscendPlanArrayInfo` 对 P1 v2 放行的场景（基线 + X1–X5）的 plan 与 Python `build_plan` 逐项一致；
P2 代码不包含任何 `PrimFunc/IRModule/SIMD_VF` 依赖（CI grep）。

### Phase 3：P3 C++ 迁移（按发射形态分批）

| 批次 | 任务 | 交付物 | 验收标准 |
| --- | --- | --- | --- |
| 3A | 发射核心：地址/掩码/斜坡/prologue/前向/寄存器直通 | `vsimd_emit` 核心 | A1/A2 发射 IR `structural_equal`；op 序列一致 |
| 3B | 逐元素 + pre/post + guard 分段 | `vsimd_emit` 元素路径 | P1 放行的 A3–A13 子集、C1–C4、D1–D5 发射 parity |
| 3C | 归约：短展开/长循环/尾块/单位元/lane0 收窄 | `vsimd_emit` 归约路径 | B1–B7 发射 parity；`vcadd/vcmax/vcmin` 序列一致 |
| 3D | 多段 program：共享 prologue、寄存器直通、源序发射 | `vsimd_emit` 多段路径 | E/F 场景发射 parity；段间 carry 守卫条件全部覆盖 |

**出口**：`AscendLowerArrayPlanInfo` 对 P1 v2 放行的场景的发射 IR 与 Python `emit_plan`
在 `map_free_vars=true` 下 `structural_equal`；P3 不含任何分析调用（CI 断言）。

### Phase 4：Pass 注册与流水线接线

| 任务 | 内容 | 交付物 | 验收标准 |
| --- | --- | --- | --- |
| T4.1 | 注册 P1/P2/P3 三个 Pass（三件套 + 静态注册） | 三个入口 `.cc` | `tvm.transform` 可发现；Python 薄封装可调用 |
| T4.2 | 实现 `VSimdArrayRegion/VSimdArrayPlan` 载体与 `IRModule` attrs | 载体头/实现 | P1→P2→P3 连续执行可拿到同一计划；载体丢失时可检测 |
| T4.3 | 流水线插入 + 配置开关 + legacy 回退 | `pipeline.py` 改动 | 不显式调用也自动执行；开关关闭行为与基线一致 |
| T4.4 | 定位器 + 指纹校验 + 混合 VF 处理 | `vsimd_polyhedral_pipeline.cc` | 指纹不匹配时跳过并 WARNING；混合 VF 不重复/漏发射 |

**出口**：真实编译管线自动执行三阶段；非目标 IR `structural_equal`；legacy 兜底可用。

### Phase 5：测试验证与灰度

| 任务 | 内容 | 交付物 | 验收标准 |
| --- | --- | --- | --- |
| T5.1 | Python 端到端脚本：`func → P1 info → P2 info → P3 stmt` | `testing/.../test_ascend_polyhedral_e2e.py` | 不经过 Pass 直接调 FFI 即可跑通；与 Pass 路径结果一致 |
| T5.2 | 基线场景全量 + 扩展批次 X1–X5 | 测试矩阵 | 基线全绿；扩展逐批全绿；X6 按需 |
| T5.3 | 真实 kernel 编译（至少 3 个：elementwise / reduce / 多语句） | 编译日志 + IR 断言 | 无残留 `T.Parallel`；预期 `tl.simd.*` 序列；无编译错误 |
| T5.4 | 回归 + 编译时间/IR 体积对比 | 报告 | P1 49 + P2 135 + parity 全绿；编译时间与 IR 体积无显著恶化 |

**出口**：达到“p1 基线无退化 + 扩展场景可控”的验收标准；可以进入默认开启灰度。

### Phase 6：收尾与默认开启

| 任务 | 内容 | 交付物 | 验收标准 |
| --- | --- | --- | --- |
| T6.1 | 文档更新（P1 v2 契约、P2/P3 数据流、Pass 顺序、回退开关） | 设计文档 | 与实现一致；含故障排查 |
| T6.2 | 清理 Python 原型的生产引用（保留为测试 oracle） | 目录/入口调整 | 生产管线不再 import `polyhedral`；oracle 测试可独立运行 |
| T6.3 | 默认开启 + 观察期 | 配置默认值 | 连续一轮 CI/回归通过；保留一键关闭 |
| T6.4 | PR 拆分与 lint/format | PR 列表 | 每个 PR 可独立构建/测试；ruff/format 通过 |

---

## 6. 风险与回滚

| 风险 | 触发条件 | 缓解 |
| --- | --- | --- |
| P1 契约扩展破坏 legacy | 修改 `NestCollector` 默认行为 | 新增结构化模式，legacy 模式默认不变；49 用例守护 |
| C++/Python parity 漂移 | 移植时改写了判定顺序/诊断文本 | 保留 Python oracle；逐场景结构化 diff；诊断逐字对齐 |
| 载体丢失/被优化 pass 清理 | P1/P2/P3 之间插入其它 pass | 三 Pass 相邻；指纹校验；失败回退 legacy |
| P1 合法性过严导致能力回退 | 间接写/散列写/`local.var` | 基线/扩展分层；显式 scatter 契约单独评审；不得绕过 P1 |
| `USE_Z3=OFF` 构建判 `ILLEGAL` | 未开 Z3 | CI 固定 `USE_Z3=ON`；运行期自检并给出 warning |
| C++ 单文件过大/编译慢 | `vsimd_plan_build`/`vsimd_emit` 体量大 | 按 axis/reduce/program 与 emit 子路径拆 `.cc` |
| 混合 VF 重复发射 | P3 替换 + legacy 同 VF 兜底 | 只替换单 nest；新增混合 VF 用例；P3 后统计残留 Parallel |
| 性能回退 | 计划/发射与旧 emitter 不一致 | 记录 IR 体积/编译时间/指令数；先灰度后默认开启 |

**一键回滚**：`TL_ENABLE_POLYHEDRAL_VECTORIZE=0` 时 P1/P2/P3 全部 no-op，
`AscendSimdVFLowerParallel` 行为与当前分支完全一致。

---

## 7. 交付物清单（按阶段）

| 阶段 | 代码 | 测试 | 文档 |
| --- | --- | --- | --- |
| Phase 0 | 集成分支 | 基线报告/CI | — |
| Phase 1 | `vsimd_array_region.{h,cc}`、P1 lift 扩展 | P1 契约测试、per-nest legality 测试 | P1 v2 契约说明 |
| Phase 2 | `vsimd_plan_*` | plan parity、诊断一致性 | P2 迁移对照表 |
| Phase 3 | `vsimd_emit.{h,cc}` | emit parity、op 序列 | P3 ABI/地址规则 |
| Phase 4 | 三个 Pass 入口、载体、pipeline 接线 | Pass 流水线/混合 VF/回退测试 | Pass 顺序与回退开关 |
| Phase 5 | E2E 脚本 | 66 场景 + 135 oracle + 真实 kernel | 验收报告 |
| Phase 6 | 清理/默认开启 | 灰度结果 | 最终设计文档 |
