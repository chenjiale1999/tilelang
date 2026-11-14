# `T.Parallel` 向量化：本分支 P1/P2/P3 与 PTO PR-1+PR-2 的对比

> **历史文档**：本文记录写作时的实现/设计状态。后续改动已删除或重命名部分
> 文件、测试与行号；当前实现见 `src/ascend/transform/parallel_vectorize/` 与
> `src/ascend/transform/vsimd_*`，以代码为准。


> **对齐对象**
>
> | | 本方（本文简称 **P123**） | 对方（本文简称 **PTO**） |
> | --- | --- | --- |
> | 来源 | 本工作树：`src/ascend/transform/{lift_vsimd,vsimd_region,vsimd_ir}.*` + `polyhedral/{plan,emit,value,body,access,model,primitives}.*` | `docs/compiler_internals/woker.md`（PR-1 #3512 element-wise 框架 + PR-2 控制流） |
> | 指令层 | `tl.simd.*` → `codegen_ascend.cc` 的 `simd_inst::*`（CCE/MicroAPI 模板调用） | `tl.vmi.*` → `pto.vmi.*` PTODSL 文本 → PTOAS |
> | 分支 | `p_123_parallel_new_p1` | tile-ai mainline（PR-1 → PR-2 里程碑） |
> | 目标机 | Ascend（`T.SimdVF()` 默认路径） | Ascend 950（PTO 目标） |
>
> 只比较 `T.Parallel` 向量化链路本身；前端宏展开、tileop 通用降级、DMA 等共享部分不在此列。
>
> **更新（2026-11-10）**：§8 是本轮落地后的增量对比；§2/§3 的机制描述仍然成立，但缺口与风险清单以 §8 为准。

---

## 0. 结论速览

1. **语义目标相同，后端契约不同**：两边都是“`T.Parallel` 标量风格 → chunk 循环 + 宽向量指令 + 掩码”，但 P123 在编译器内做完指令选择/掩码/寄存器宽度决策，PTO 把掩码物化、寄存器分配、`mem_bar` 交给 PTOAS。
2. **控制流模型相反**：P123 用**谓词化 store**（`vsts(..., mask=cond)`），`if` 一直保留到 P2 规划期；PTO 在 Legalize 里把 `if` 折成 **`Select` + 读回**，向量 IR 中不存在谓词化 store（只剩 region/tail mask）。
3. **P123 强在“面”**：任意 N-D 折叠、strided/gather/pair 访存、NZ/layout、`T.reduce_*` 体内归约、多段程序、per-loop 事务回退、region 结构 oracle。
4. **PTO 强在“工程闭环”**：VMI 描述符覆盖性证明、分配元数据跨 lowering 的身份传递、codegen 值来源分类、masked store 对齐 legality；设备级数值回归（11 例）已被 P123 的 20 例追平。
5. **P123 的实质缺口只有 1 项**：`lanes` 不可配（固定 2048/元素位宽）。`fp8→f32` / 整数符号扩展转换、运行期与不可吸收的 serial shell、serial 循环下的 Parallel 均已补齐（§8.5–§8.7）；转换只剩 `i8/u8→i16/f16`、`i32→i64`、`i64→*` 回退。

---

## 1. 定位、流水线

### 1.1 目标与职责边界

| 维度 | P123 | PTO |
| --- | --- | --- |
| 用户写法 | `T.SimdVF()` + `T.Parallel(...)`（标量风格） | `T.SimdVF(lanes=L)` + `T.Parallel(...)`（`lanes ∈ {64,128,256}`） |
| 向量宽度 | 由元素位宽推出：`2048 / elem_bits`（f32 → 64 lane），调用方无需声明 | 由用户 `lanes` 指定，布局推导按它取模 |
| 掩码粒度 | `MaskBits = max(elem_bits, 8)`，谓词统一 `boolx256` | VMI mask，由 PTOAS 物化 |
| 编译器职责 | 轴选择、分段、layout/NZ、pack2、取值 DAG、指令配方、掩码表达式、回退 | 契约校验、布局推导、分支折叠、值→VMI 转换、描述符证明、分配元数据、对齐 legality |
| 后端职责 | `codegen_ascend` 只是逐指令翻译成 `simd_inst::*` 模板调用 | PTOAS：物理寄存器切分、mask 物化、Fragment/向量寄存器分配、mem_bar、优化 |
| 未向量化时的去向 | P3 保留原 `T.Parallel`，由 codegen 出标量循环（`AscendSimdVFLowerParallel` 当前被注释） | 非 PTO 目标走原 `AscendSimdVFLowerParallel`；PTO 目标在 Verify 处直接报错中断 |

### 1.2 流水线形态

P123（三阶段 + 类型化载体，全部在 `AscendLiftVSimd` / `AscendVectorizeVSimd` 内）：

```text
TIR
 └─ AscendLiftVSimd (P1)      per-Parallel: 语法白名单 + 内存合法性 + capture 闭包
      → For.annotations["tl.vsimd.region"] = tl.vsimd.Region（LIFTED / OPAQUE + reason）
 └─ AscendPlanVSimd (P2)      per-loop: BuildPlan → VectorProgram（segments / 轴 / 掩码 / 布局）
      → For.annotations["tl.vsimd.plan"]  = tl.parallel_vectorize.VSimdRegionPlan
 └─ AscendEmitVSimd (P3)      per-loop: EmitPlan → tl.simd.* 语句；失败只回滚该 loop
      → 无条件清理两处注解
```

PTO（PR-1 + PR-2，全部 in-place 改 TIR，靠 annotation 传递布局）：

```text
NormalizePTOAllocMarkers → LegalizePTOVMIMemoryAccess → (早期 unroll)
 → AscendLayoutInference(pto)  → AutoSchedule + InsertSync（读 descriptor）
 → VerifyParallelToPTO（只读，首个错误即中断）
 → LegalizeParallelToPTO（PR-1 空壳；PR-2：if → Select、min/max → Select）
 → VectorizeParallelToPTO（整 region 统一转 tl.vmi.*）
 → LowerOpaqueBlock → ImportPTOAllocMarkers → VerifyPTOCodegenLegality（只读）
 → codegen_pto（pto.vmi.*）
```

### 1.3 阶段职责对照

| 关注点 | P123 负责阶段 | PTO 负责阶段 |
| --- | --- | --- |
| 输入合法性 | P1（per-nest，`OPAQUE` 可局部失效） | Verify（全局，首个错误中断） |
| 内存建模 / 依赖 | 无独立 pass；`access_ptr` 由发射层生成 | `LegalizePTOVMIMemoryAccess` 提前构造并证明 descriptor |
| 向量轴 / 布局 | P2 `RankVectorAxes` + `ClassifyAccess`（不写 IR annotation） | `AscendLayoutInference` + `pto_parallel_layout`（写 IR annotation） |
| 分支归一 | P2 `NormalizeBodyInto`（规划期，产物是 StoreSpec+mask） | Legalize（向量化前，产物是 `Select` store） |
| 值 → 向量值 | P2 `BuildValue`（recipe DAG）+ P3 `Emitter` | Vectorize `ConvertExpr`（直接产 `tl.vmi.*`） |
| 掩码 | P3 按 chunk 现算（`vcmps`/`pand`/`update_mask`/`pnot`/`pxor`） | Vectorize 产 `create_mask`，PTOAS 物化 |
| 最终合法性 | 无（依赖 codegen 的 ICHECK） | `VerifyPTOCodegenLegality`（store 对齐） |

---

## 2. 能力矩阵

图例：**✔** 支持；**△** 支持但有条件/不同语义；**✘** 不支持（有诊断）；**—** 不适用。

### 2.1 形状与迭代空间

| 能力 | P123 | PTO | 说明 |
| --- | --- | --- | --- |
| 1D `T.Parallel(E)` | ✔ | ✔ | 双方核心场景 |
| 2D `T.Parallel(M,N)` | ✔ | ✔ | P123 折叠为一个迭代空间后选轴；PTO 只允许 1D/2D，且只有被选维向量化 |
| ≥3D 嵌套 Parallel | ✔ | ✘ | P123 `AbsorbParallelChain` 支持任意层；PTO 只认两层紧嵌套 |
| `min != 0` | ✘（P1 闸门，见 §8；P2 `RequireZeroMin` 兜底） | ✘（Verify） | 同结论、不同层 |
| `step != 1` | ✘（P1 闸门，见 §8） | ✘（Verify） | 手写 TIR 不再静默错编译 |
| 非常量 extent | ✔（符号 extent + 运行期 tail mask） | ✘（要求编译期常量） | P123 的 `test_symbolic_tail_context_reaches_the_planner` 覆盖 |
| extent 超 int32 | ✘（无溢出保护） | ✔（`CheckedPtoPadding`） | PTO 显式拒绝 |
| `lanes` 可配置 | ✘（固定 `2048/elem_bits`） | ✔（64/128/256） | P123 的寄存器宽度即向量宽度 |
| 非整除 extent（tail） | ✔（`update_mask(remaining)`） | ✔（`create_mask(remaining, L)`） | 机制同构 |
| 越界访问面 | tail 只保证不写越界；读按整 lane 读，要求 UB 后备可访问 | 同（PR-1 §4 明确说明） | 一致 |

### 2.2 访存形态

| 能力 | P123 | PTO | 说明 |
| --- | --- | --- | --- |
| 连续 load/store | ✔ | ✔ | — |
| lane-uniform（同址） | ✔（`ONEPT_B` store / 标量广播） | ✔（scalar load + `vbrc`） | 一致 |
| 固定步长（strided） | ✔（`vgather2` + ramp） | ✘（Vectorize 只做连续/同址；group/stride 仅手写 VMI） | P123 的优势项 |
| 间接索引（gather） | ✔（逐 lane 索引向量） | ✘（PR-5 才做） | P123 的优势项 |
| 双流打包（`vld2`/`vintlv`） | ✔（pack2 配对发射） | ✘ | P123 的优势项 |
| NZ / 标注 layout 目标 | ✔（`vsstb`、`vsts` PK 形态、P1 捕获 layout） | ✘（Fragment 只表达 lane 映射，UB 容量规划为 PR-4） | P123 的优势项 |
| GM / 非 UB | ✘（前端 `vf_checker.py` 拒绝） | ✘（Verify 拒绝） | 同结论、不同层 |
| volatile 分配 | 无此概念 | ✘（Verify 拒绝） | — |
| view over `Bind` 指针 | 未检查 | ✘（PR-2 新增规则） | PTO 更严 |

### 2.3 运算与类型

| 能力 | P123 | PTO | 说明 |
| --- | --- | --- | --- |
| 二元 `+ - * /`、`max/min` | ✔（`vadd/vsub/vmul/vdiv/vmax/vmin` + scalar 变体） | ✔ | P123 直接发 `vmax/vmin`；PTO 在 Legalize 改写成 `Select` 以保 NaN/±0 |
| 一元 `abs/exp/log/sqrt` | ✔ | ✔ | — |
| 类型转换 | ✔ 窄化 / 同宽 / 加宽主路径均支持，完整清单见 §8.6 | ✔ 覆盖 f32↔fp8、f16/bf16↔f32、整数同宽/扩展（含 `vinterpret_cast`） | 仅 `i8/u8→i16/f16`、`i32→i64`、`i64→*` 仍回退 |
| fp8 → f32 | ✔（`UNPK4_B8` 装载 + `vcvt PART_P0`） | ✔ | 已对齐 |
| 同宽有符号/无符号互转 | ✔ `reinterpret` recipe（零指令；PTO `vinterpret_cast` 的对应物） | ✔ | 已对齐，机制不同 |
| 非支持 dtype | `Unsupported`（`VRegLanes` 白名单） | Verify/`PlanValueCast` 拒绝并给原因 | 都显式拒绝 |
| FP4 | ✘ | △（手写 VMI descriptor 支持，自动向量化在 PR-5） | — |

### 2.4 控制流（PTO PR-2 的主战场）

| 能力 | P123 | PTO | 实现差异 |
| --- | --- | --- | --- |
| 两臂同址 `if/else` | ✔ `vsel`（`if_then_else` 惰性合并） | ✔ `Select` | 一致 |
| 单臂 `if` | ✔ **谓词化 store**（`vsts` 带 mask，不读回目标） | ✔ `Select(c, x, read-back dest)` | 语义一致；P123 少一次 `vload`，PTO 少一次 mask 组合 |
| 两臂异址 | ✔ **每臂各自谓词化 store**（互补掩码） | ✔ 每目标一个 value choice + 读回 | 语义一致、机制不同 |
| 嵌套 `if` | ✔ 逐层 mask 叠加（`pand`） | ✔ 自底向上折叠、每目标一个 value choice | 语义一致、机制不同 |
| 分支体含非 store 语句 | P1 建模、P2 拒绝该 loop（回退） | Legalize 直接诊断拒绝 | 结果相同 |
| lane-varying 比较 | ✔ `vcmp`/`vcmps` | ✔ 同 | 一致 |
| lane-uniform 比较 / 裸 scalar bool | ✔ `update_mask` | ✔ `create_mask` | 一致 |
| `and/or/not` | ✔ `pand/por/pnot` | ✔ `vand/vor/vnot` | 指令族不同（我方是谓词寄存器族） |
| mask `==` / `!=` | ✔ `pnot(pxor)` / `pxor`（排序比较拒绝） | ✔ 真值表（`vand/vor/vnot`） | 一致；PTO 无可用的 xor 谓词指令 |
| 条件中带 RAW（条件别名被写） | △ 条件在 P2 期按 SSA 求值；未见显式“条件快照”规则 | ✔ 提交 4 专做 condition snapshot | PTO 更严谨 |
| 分支内 `T.alloc_var` 目标 | ✘（P1 拒绝 `local.var`） | ✘（PR-4 才开） | 一致 |

### 2.5 循环与 shell

| 能力 | P123 | PTO | 说明 |
| --- | --- | --- | --- |
| `T.serial` 在 Parallel 内、**常量边界** | ✔ 归约型吸收成轴；否则 ≤8 次逐迭代展开；再否则保留为标量壳（§8.7） | ✔ 原样复制为标量 shell | P123 会向量化壳内工作；PTO 保留标量壳 |
| `T.serial`、**运行期边界** | ✔ 标量壳载体（bound 须 lane 不变且不含 buffer 读） | ✔ | 已对齐（§8.7） |
| shell 边界引用 lane var / unit-local var | ✘（P1 判 OPAQUE；P2 亦有防线） | ✘（明确诊断） | 同结论 |
| shell 内写共享标量（`cnt[0] += 1`） | ✘（`body.nested_loop` 回退） | △（复制成标量壳，语义天然正确） | PTO 更宽松 |
| `T.Parallel` 位于 **serial 循环**下（VF 内） | ✔ 照常 lift，serial 循环留作外层标量壳（§8.7） | ✔ | 已对齐 |
| `T.Parallel` 位于 branch / while 下 | 不 lift（仅 WARNING） | ✘（Verify 拒绝） | 都不向量化；PTO 诊断更硬 |
| `while` / `T.Pipelined` | ✘ | ✘ | 一致 |
| `break` / `continue` | ✘ | ✘ | 一致 |
| `T.Parallel` 被 thread-bound / vectorized 包裹 | 不 lift（WARNING） | ✘（Verify 拒绝） | 同上 |

### 2.6 归约 / reducer

| 能力 | P123 | PTO | 说明 |
| --- | --- | --- | --- |
| Parallel 体内 `T.reduce_{sum,max,min}` | ✔（P1 `reduce` op → P2 展开 → 归约发射） | ✘（PR-3） | P123 领先一个里程碑 |
| 整行归约（`reduce` 跨 chunk） | ✔ | ✘ | — |
| 累加器提升（`acc +=` 跨 chunk 保持在寄存器） | ✔（`AccumulatorCandidate`） | ✘ | — |
| `T.alloc_reducer` / `T.finalize_reducer` | ✘（依赖 fragment，本分支 fragment 链路已移除） | ✘（PR-6） | 都未支持 |

### 2.7 布局与分配

| 能力 | P123 | PTO | 说明 |
| --- | --- | --- | --- |
| 布局推导 | P1 读既有 layout 注解（`FindLayout`）后参与 store 分类，不新写布局 | 独立 `AscendLayoutInference` + Fragment 表达 lane 映射，写 `pto_parallel_loop_layout` | PTO 有“布局”这一等公民；P123 只消费既有 layout |
| 寄存器/Fragment 分配 | 不涉及（fragment 全链路已移除，PlanOptions 删除 `vf_alloc_scope`） | PTOAS 负责；PTO 侧 PR-4 规划 UB 容量 | 分工不同 |
| `T.vmi.alloc_var` / `alloc_local` 元数据 | 无（`local.var` 在 Parallel 内被拒） | ✔（AttrStmt carrier + 身份匹配 + early unroll 重映射） | PTO 的独有能力 |
| 多段程序（同一 nest 拆多段发射） | ✔（`VectorProgram.segments`，每段独立 recipe/布局） | ✘（一个 unit 一个 chunk 循环） | P123 的优势项 |

### 2.8 诊断、降级与可观测性

| 能力 | P123 | PTO |
| --- | --- | --- |
| 失败粒度 | **per-loop 事务**：P2 不写 plan / P3 回滚该 loop，同 VF 其他 loop 照常 | **全局**：Verify 首个错误即中断编译 |
| 未向量化时的行为 | 保留原 `T.Parallel`，后续出标量循环 | 非 PTO 目标走旧 pass；PTO 目标直接报错 |
| 诊断载体 | `tl.vsimd.Region`（`status`/`reason_code`/`reason`）+ `Unsupported{code, diagnostics}` + WARNING 日志 | 带 `[VerifyParallelToPTO]` 前缀的分类诊断（category + location） |
| 只读分析入口 | `AscendAnalyzeSimdVFMemoryReport` / `...CaptureReport` / `...VSimdRegionReport` / `AscendDumpVSimd` | PR 描述中未列出独立的只读分析/dump pass（依赖 IR 打印） |
| 结构自校验 | ✔ `VerifyRegion` + 参考展开 + `CompareRegionWithOriginal`（结构 oracle） | ✘（靠 Verify 的前置契约） |
| 打印可读性 | P1/P2 注解经 `IRDocsifier` 内联打印（region/plan 内容可见） | annotation 为普通 Map/Fragment |

### 2.9 与 PTO roadmap 的对齐

| 里程碑 | PTO 状态 | P123 状态 |
| --- | --- | --- |
| M1 element-wise（含非整除 extent） | PR-1 | 已有（且覆盖 strided/gather/pair/NZ） |
| M2 控制流 + lane-uniform serial | PR-2 | 已有（含运行期 / 不可吸收 shell 与 serial 外置形态，§8.7） |
| M3 `T.reduce` max/min/sum | PR-3 计划 | **已有** |
| M4 `alloc_var`/`alloc_local`、寄存器分配 | PR-4 计划 | 明确不支持（fragment 链路已移除） |
| M5 gather/scatter、FP4 | PR-5 计划 | gather/scatter **已有**；FP4 无 |
| M6 reducer | PR-6 计划 | 不支持（依赖 fragment） |

---

## 3. 方案（机制）对比

### 3.1 向量化模型：一致的骨架，不同的“决策物”

- **相同**：都是“把一个矩形并行域按 `lanes` 切成 chunk，chunk 内用宽向量指令，最后一条 store 携带掩码”。
- **不同**：PTO 的决策物是 **layout annotation**（`lane = idx % VL, q = idx // VL`），Vectorize 只做“按布局反转 + 逐语句翻译”。P123 的决策物是 **plan**（轴打分排序、store 布局分类、每 store 的 recipe、掩码表达式、分段），发射阶段才决定指令。
- **后果**：PTO 的 Vectorize 是“翻译器”，非常薄；P123 的 P3 是“代码生成器”，plan 里已含全部决策，因此同一个 loop 可以有不同的分段/布局/打包策略（例如 pack2、NZ、多段）。

### 3.2 分支模型（最核心的差异）

| | P123：谓词化 store | PTO：value choice + 读回 |
| --- | --- | --- |
| 归一发生点 | P2 规划期（`NormalizeBodyInto`），产物 `StoreSpec{value, mask}` | Legalize（向量化前），产物 `BufferStore(Select(...))` |
| 单臂代价 | 1 条 `vsts` + 谓词表达式 | 1 条 `vload`（读回目标）+ 1 条无谓词 `vsts` |
| 异址双臂代价 | 2 条谓词化 `vsts`（互补掩码） | 每目标 1 条读回 + 1 条 `vsts` |
| 嵌套分支 | mask 逐层 `pand` | 逐目标 Select 折叠 |
| 向量 IR 形态 | 带谓词的 store | **不存在谓词化 store**（只剩 region/tail mask） |
| 语义风险点 | 谓词必须与 tail mask 正确合取；目标同址多写要有序 | 读回会把“目的地上一次值”引入依赖链，遇到 UB 污染需快照（PTO 提交 4） |
| 目标地址约束 | 无（异址/同址都行） | 必须 lane-private，否则读回语义不成立 |

一句话：**P123 让谓词进硬件，PTO 让值选择进硬件**。两者都能覆盖 PR-2 表里的全部四种分支形态，代价模型不同：读回多一次内存流量，谓词多一次谓词寄存器压力。

### 3.3 tail / 非整除 extent

| | P123 | PTO |
| --- | --- | --- |
| 可分 | 全 lane 掩码（`PSET PAT_ALL`），无 per-chunk 计算 | 共享常量全掩码 |
| 不可分 | 每 chunk `update_mask(min(remaining, lanes), bits)`；`remaining` 由 `extent - chunk*lanes` 现算 | 标量 `remaining` 递减 + `create_mask(remaining, L)` |
| 符号 extent | 支持（运行期算） | 不支持（Verify 要求常量） |
| 掩码位宽 | `max(elem_bits, 8)`，谓词统一 `boolx256` | 由 `lanes` + dtype 决定 |

### 3.4 条件求值

| | P123 | PTO |
| --- | --- | --- |
| 比较 | `vcmp`/`vcmps`，scalar 侧自动翻转换向 | 同 |
| 标量 bool / lane-uniform | `update_mask(Select(cond, lanes, 0))` | `create_mask` |
| 布尔代数 | `pand`/`por`/`pnot`/`pxor`（谓词寄存器族） | `vand`/`vor`/`vnot`（无 xor，EQ 用真值表） |
| 数值类型歧义 | 无：值节点带 recipe id，类型即语义 | 有：1-lane VMI 寄存器与标量 int 同 dtype，靠 codegen 值来源表区分 |

### 3.5 非连续访存

- P123：`ClassifyAccess` 沿每个轴做有限差分 → `contiguous / strided / uniform / indirect / pair`，再选 `vld`、`vlds_*`、`vgather2`、`vld2`；store 侧还有 `vsts(PK_B*)`、`vsstb`（NZ）。
- PTO：Vectorize 只处理 **连续** 与 **lane-uniform**；strided/group/block_stride 只在手写 `tl.vmi.*` 里通过 descriptor 支持（PR-5 才把 gather/scatter 带进自动向量化）。
- 取舍：P123 的“面”更宽，但代价是依赖分析的声明面（见 3.7）不如对方严格。

### 3.6 `min` / `max` 语义

- PTO：Legalize 把 `T.min/T.max` 改写成 `Select(a<b? a:b)`，理由是保住 IEEE `NaN` 与 `±0` 顺序；并为此写了设备级数值回归。
- P123：直接发 `vmax/vmin/vmaxs/vmins`，**契约以 CANN 硬件为准**（950PR 实测 `asc_max` / `asc_min`）：
  - NaN 操作数**传播**：任一操作数为 NaN 时结果为 NaN（与 `a>b?a:b` 不同，后者会取另一个操作数）；
  - `±0`：`max` 仅在两个操作数都是 `-0` 时得 `-0`（有 `+0` 则得 `+0`）；`min` 仅在两个都是 `+0` 时得 `+0`（有 `-0` 则得 `-0`）。
  设备用例 `test_parallel_max_min_nan_and_signed_zero` 按该契约校验 NaN 左右/双侧与 `±0` 顺序（见 §8）。
- 评估：语义护栏已补，仍是“直接发射 + 用例钉死”，比 PTO 的 `Select` 重写少指令；代价是 NaN 行为与 IEEE `maximum/minimum` 不同，需要在使用方按硬件契约理解。

### 3.7 描述符 / 依赖声明

| | P123 | PTO |
| --- | --- | --- |
| 声明形式 | 发射时生成 `T.access_ptr(base, extent, rw)` | 手写调用：`LegalizePTOVMIMemoryAccess` 提前构造；Vectorize 产物自带 descriptor |
| 覆盖范围 | 连续访问：一个寄存器；常量 stride：`(lanes-1)*stride+1` 真实包络；索引访问（gather/scatter）与 NZ 散射：**整个 allocation owner** | 普通访问：物理包围 `[low, high)`；索引访问：**整个 allocation owner** |
| 未覆盖时 | 不可覆盖的形态（负 stride、加宽 + 谓词/stride/gather 装载）抛 `Unsupported`，该 loop 回退标量 | 编译期诊断，绝不发不完整声明 |
| 影响面 | `AscendThreadSync` 按 `[offset, offset+extent)` 做冲突判定，声明面已 ≥ 实际触及面 | 调度只看 descriptor，因此强制证明 |

“声明面必须 ≥ 实际触及面”这条已经落到 P123（见 §8.6 第 2 项）：区别只在 PTO 有覆盖性证明，P123 给的是保守上界（不可证明就回退）。

### 3.8 分配元数据与 codegen 值分类

- PTO 有 `T.vmi.alloc_var/alloc_local`：AttrStmt carrier（按 data Var 身份）→ early unroll 重映射 → `LowerOpaqueBlock` 保留 → `ImportPTOAllocMarkers` 落到 `AllocBuffer.annotations` → codegen 按 slot 类别而非 dtype 决定初始化/赋值/读取。目的是解决“1-lane vreg 与 scalar 同 dtype”的歧义。
- P123 没有这个问题：`ValueNode.recipe` 是闭合枚举，`vsts`/`vld` 的操作数类型由 plan 决定，`local.var` 在 Parallel 内被 P1 直接拒绝，不需要 slot 类别。

### 3.9 校验闸门

| 闸门 | P123 | PTO |
| --- | --- | --- |
| 输入契约 | P1 白名单（语句/表达式）+ PR-A 内存合法性 + PR-B capture 闭包 | `VerifyParallelToPTO`（分类诊断，首个错误中断） |
| 结构自校验 | `VerifyRegion` + 参考展开 + 与原 body 结构比对 | 无 |
| 发射前 legality | 无 | `VerifyPTOCodegenLegality`（masked store 32B 对齐） |
| 指令合法性 | 依赖 codegen `ICHECK`（内部错误而非用户诊断） | 尽量前移到编译期诊断 |

P123 缺一条“最终 IR legality”闸门：`vsts` 的某些分布（如 `NORM_B32`）与 NZ 散写在硬件上有对齐/跨度要求，目前没有前置检查，错了会在 codegen 或设备上暴露。

### 3.10 本地 DCE / 快照

- PTO 在 Vectorize 后跑 `DeadBindElim` 清掉只服务地址推导的绑定，并在 Legalize 里做 condition snapshot；Condition 别名写时有专门提交。
- P123 的等价物是发射层的 `forward_`/`promoted_`/掩码缓存；本轮已修正“谓词化/tail store 不得做值转发”，但没有条件快照的显式规则（条件在 P2 期求值，SSA 上天然是旧值，风险较低）。

---

## 4. 设计对比

### 4.1 阶段划分与载体

| 维度 | P123 | PTO |
| --- | --- | --- |
| 阶段数 | 3（lift / plan / emit） | 2 主动阶段（Verify+Legalize / Vectorize）+ 2 legality/元数据阶段 + codegen |
| 阶段间载体 | **类型化 FFI 对象**：`tl.vsimd.Region` → `tl.parallel_vectorize.VSimdRegionPlan` | **IR annotation**（`pto_parallel_loop_layout`）+ 就地改写 + AttrStmt carrier |
| 载体是否改 body | 否（P1/P2 只挂注解；P3 才替换） | 是（Legalize/Vectorize 直接改写语句） |
| 定位方式 | `For` 节点身份（无 span / fingerprint） | 就地改写 + annotation 查表 |
| 阶段契约 | region 是“忠实描述”；plan 是“发射唯一输入” | Verify 是“输入契约”；Legalize 保证 Vectorize 输入形态 |

**取舍**：P123 的三段式让“规划”和“发射”可独立测试（离线 `build_plan`/`emit_plan`），也让失败可以只丢一段；PTO 的两段式更贴近常规编译器（分析→变换），但没有可离线检查的中间计划。

### 4.2 事务粒度与失败语义

| | P123 | PTO |
| --- | --- | --- |
| 粒度 | 每个 `T.Parallel` 一个事务 | 整个编译单元 |
| 一个 loop 不被支持 | 该 loop 不向量化，其余照常 | 编译失败 |
| 适用场景 | 大 kernel 里混用可向量化/不可向量化代码 | 严格 PTO 目标，契约外的程序直接挡住 |

**评价**：P123 的“局部回退”在真实 kernel 上更宽容（FA 那种混合体可继续编译），代价是性能不确定且难以察觉；PTO 的“硬拒绝”保证“编译通过即向量化”，但迁移成本高。

### 4.3 自校验（结构 oracle）vs 前后置 Verify

- P123 的 oracle 是**编译期自证**：把 region 重新展开成标量循环，与原始 body 逐语句比对（循环范围、store 目标/下标/值），不一致直接判“编译器 bug”，再叠加 `VerifyRegion` 的 schema 校验。它保护的是“P1 没撒谎”。
- PTO 的 Verify 保护的是“用户程序在契约内”，`VerifyPTOCodegenLegality` 保护的是“最终 IR 合法”。两者互补但都不自证 region 的忠实性。
- 结论：P123 在“P1 建模是否忠实”上更稳，PTO 在“最终产物是否合法”上更稳；理想状态是两者都有。

### 4.4 编译器做多少 vs 后端做多少

| 决策 | P123 | PTO |
| --- | --- | --- |
| 掩码物化 | 编译器（`vcmps`/`pand`/`update_mask`） | PTOAS |
| 寄存器分配 | 无需（固定 2048-bit 宽 + recipe 决定 lane 数） | PTOAS |
| 指令选择 | 编译器（plan 里的 recipe） | Vectorize 选 `tl.vmi.*`，PTOAS 再优化 |
| 同步 / mem_bar | 无（沿用既有 pass） | PTOAS |
| 布局（lane 映射） | 不产生 | 编译器（`pto_parallel_loop_layout`） |

### 4.5 可扩展性

| 扩展动作 | P123 | PTO |
| --- | --- | --- |
| 新增一元/二元算子 | 在 `value.cc` recipe 表加一行 + `emit.cc` 的 `SimdOpTable` 加一行（family 分派） | `ConvertBinary`/`ConvertExpr` 表 + VMI 注册 + codegen 分支（三处） |
| 新增 dtype/转换 | `Vcvt` 的 (src,dst) 表 + `ElemBitsTable` | `PlanValueCast` + VMI 指令 + codegen 值分类 |
| 新增访存形态 | `ClassifyAccess` + `value.cc` recipe + emit | 多为 PR-5 的 descriptor/指令扩展，需 PTOAS 配合 |
| 新增控制流形态 | P1 白名单 + P2 `NormalizeBodyInto` | Verify 契约 + Legalize 折叠 + Vectorize |

### 4.6 调试与可观测性

| | P123 | PTO |
| --- | --- | --- |
| 注解内容 | `print(mod)` 直接看到 region/plan 全文（含 axes/ops/status/reason/segments） | annotation 为普通对象，需自行 dump |
| 只读分析 pass | 三份报告（memory / capture / region）+ 合并 dump | 无 |
| 失败原因 | `reason_code`（ILLEGAL/UNPROVEN/UNSUPPORTED/INTERNAL）+ 具体原因 + P2/P3 WARNING | 分类诊断（category + location + 补救建议） |
| 用户可见性 | 回退是静默的（除 WARNING）；性能退化不易察觉 | 编译期即失败，用户被迫改写 |

### 4.7 测试策略

| | P123 | PTO |
| --- | --- | --- |
| 结构测试 | 641（transform）+ 22（analysis）：离线 plan/emit 字段断言、三阶段契约、oracle、报告 | 414 PTO 结构用例（本 PR 新增 96 / 68 个 node 在两个控制流文件） |
| 设备测试 | 20 例设备数值回归（原 `test_tilelang_ascend_vsimd_numeric.py`，已并入 `test_tilelang_ascend_simdvf_parallel*.py`，见 §8；无 NPU 时 skip） | 11 例数值回归（含 UB 污染、NaN/±0、runtime trip count） |
| 覆盖倾向 | “机制是否按设计工作”（plan 字段、指令序列） | “语义是否正确”（NumPy golden 对比） |

设备级数值回归已补齐（20 例，见 §8）：覆盖分支谓词化 × tail、serial shell 展开/归约、strided/间接访存、窄化/加宽/同宽符号 cast。

---

## 5. 同一程序的两条路径

源程序：

```python
for i in T.Parallel(128):
    if a[i] > 0.0:
        y[i] = a[i] + 1.0
```

P123（谓词化，1 chunk × 2）：

```text
mask = pset(32, PAT_ALL)
v0   = vld(a_ub[i], NORM)
p0   = vcmps(v0, 0.0, mask, "gt")
v1   = vadds(v0, 1.0, mask, MODE_ZEROING)
vsts(y_ub[i], v1, p0, "NORM_B32")
```

PTO PR-2（value choice + 读回，经 Legalize 后的形态）：

```text
y[i] = Select(a[i] > 0.0, a[i] + 1.0, y[i])     # Legalize 产物
→ vload y_old; vcmps cond; vadds; vsel(mask, new, y_old); vstore(...)
```

两臂异址时：

```python
if a[i] > 0: y[i] = 1.0
else:        z[i] = 2.0
```

- P123：`vsts(y, 1.0, cond)`, `vsts(z, 2.0, !cond)`（1 次 `vcmps` + 1 次 `pnot`）
- PTO：`y = Select(cond, 1.0, readback(y))`、`z = Select(cond, readback(z), 2.0)`（2 次读回）

---

## 6. 优势 / 缺口 / 风险

> 下面的 §6.2 缺口表与 §6.3 风险表是文档初版快照；本轮落地后的最新状态见 **§8.2 / §8.3**。

### 6.1 P123 相对优势（按价值排序）

1. **访存面**：strided / gather / pair(`vld2`) / NZ(`vsstb`) / pack2 全部在自动向量化里，PTO 的自动路径只有连续与同址。
2. **形状面**：任意 N-D 折叠、符号 extent + 运行期 tail。
3. **归约**：体内 `T.reduce_*`、整行归约、累加器提升（PTO 到 PR-3）。
4. **局部回退**：per-loop 事务，混合 kernel 可继续编译。
5. **自校验**：region 结构 oracle，能抓“P1 建模撒谎”这类编译器 bug。
6. **可观测性**：注解内联打印 + 三份只读报告 + reason_code。
7. **无谓词 store 依赖读回**：单臂/异址分支少一次目标读回。

### 6.2 P123 的实质缺口

| # | 缺口 | 影响 | 建议 |
| --- | --- | --- | --- |
| 1 | 无 `lanes` 配置（固定 `2048/elem_bits`） | 无法按 dtype/硬件选 64/128/256 lane；f16 下自动变成 128 lane，与用户预期不同 | `BuildPlan(stmt, lanes, ...)` 已有 `lanes` 形参，只需前端 `T.SimdVF(lanes=...)` 写注解、P2 driver 读出并传入 |
| 2 | fp8→f32、同宽符号互转缺失（无 `vinterpret_cast`） | 量化/反量化 kernel 需要手写路径 | `Vcvt` 表补反向项；同宽符号互转用现有 `reinterpret` 语义包一层 recipe |
| 3 | 运行期边界的 serial shell | 动态 trip count 的标量壳无法走本链路 | 需要 plan/emit 支持“标量壳包 chunk 循环”（当前 ≤8 次常量展开是权宜） |

### 6.3 P123 需要关注的风险

| # | 风险 | 说明 |
| --- | --- | --- |
| R1 | `access_ptr` 只声明 `lanes` 个元素 | strided/gather 真实触及面更大；若调度/依赖分析以它为准会欠覆盖（PTO 用“整 owner + 覆盖性证明”解决） |
| R2 | `vmax/vmin` 直接发射 | 未固定 NaN/±0 语义（PTO 用 `Select` 重写 + 数值回归解决） |
| R3 | 无发射前 legality 闸门 | `vsts` 分布/NZ 跨度的对齐要求没有前置检查，错了会在 codegen/设备暴露 |
| R4 | `step != 1`、`min != 0` 的检查不完整 | `min` 在 P2 有 `RequireZeroMin`；`step` 完全未看（DSL 不产生，但手写 TIR 会静默错编译） |
| R5 | 无设备级数值回归 | 谓词化 + tail + shell 展开的组合语义只靠结构断言 |

### 6.4 PTO 相对缺口（供参考）

1. 非常量 extent、`min != 0`、`step != 1` 全部拒绝（P123 支持前两项）。
2. 自动向量化只有连续/同址访存（PR-5）。
3. 无归约（PR-3）、无 `local.var`（PR-4）、无 reducer（PR-6）。
4. 契约外程序直接编译失败，迁移成本高。
5. 无中间计划载体，规划不可离线测试/复现。

---

## 7. 可借鉴清单（按性价比排序）

> 状态（2026-11-10）：#1（声明面）、#3（`min/max` 语义护栏）、#4（设备数值回归）、#6（同宽符号 / 加宽转换）均已落地；仅 #2（发射前 legality 闸门）与 #5（`lanes` 可配）未做。详见 §8.6。

| # | 借鉴点 | 落到 P123 的位置 |
| --- | --- | --- |
| 1 | **声明面 ≥ 实际触及面**：为 strided/gather 访问声明真实包围（或整 owner） | `EmitPlan` 的 `AccessPtr` 生成处（`emit.cc`） |
| 2 | **最终 IR legality 闸门**：store 对齐/分布要求前置诊断 | 新增只读 pass，放在 P3 之后、`LowerAccessPtr` 之前 |
| 3 | **`min/max` 语义护栏**：NaN/±0 用 `vsel` 或加用例固定硬件语义 | `value.cc` recipe 选择 + 设备回归 |
| 4 | **设备级数值回归**：分支×tail×shell 组合 | `testing/ascend/device`（新增） |
| 5 | `lanes` 可配 | P2 driver 读 `T.SimdVF(lanes=...)` |
| 6 | `vinterpret_cast` 同宽符号互转 | `Vcvt`/新 recipe + emit |

---

## 8. 最新对比（2026-11-10，两轮落地后的增量）

> 本节是相对 §0–§7 的增量快照：§2/§3 的机制描述仍然成立，§6.2/§6.3 的缺口与风险清单已被本节替代。
> §8.1–8.5 是第一轮（形状闸门 / max-min 语义 / 设备回归 / 窄化转换）；**§8.6 是第二轮**（同宽符号、访存声明面、加宽转换）。

### 8.1 第一轮落地四项（P123）

| # | 落地项 | 代码落点 | 验证 |
| --- | --- | --- | --- |
| 1 | `min == 0` / `step == 1` 从“P2 兜底”前移成 **P1 闸门**；常量 `T.serial` shell 同样要求 `step == 1` | `vsimd_analysis_common.h::NestCollector`（`IsZeroLowerBound` / `IsUnitStep`）；`plan.cc::RequireZeroMin` 保留作纵深防御 | `test_ascend_vsimd_lift.py` 新增 3 例；探针：`min=3` / `step=2` / shell `step=2` 均 `OPAQUE + UNSUPPORTED`，合法形态仍 `LIFTED` |
| 2 | `vmax/vmin` 的 NaN / ±0 语义**以 CANN 为基准**并配设备用例 | 契约（950PR 实测 `asc_max`/`asc_min`）：NaN 传播（任一操作数为 NaN 则结果 NaN）；`max` 仅双 `-0` 得 `-0`，`min` 仅双 `+0` 得 `+0`；发射是直接 `vmax/vmin` | `test_parallel_max_min_nan_and_signed_zero`（`max` / `min` 两个参数）：NaN 在左/右/双侧、`±0` 四种顺序、`±inf`，并对零结果单独校验 `signbit` |
| 3 | 设备级数值回归文件 | 原 `testing/ascend/language/test_tilelang_ascend_vsimd_numeric.py`（20 例，第二轮扩到同宽符号/加宽，见 §8.6）；该文件已去重删除，用例并入 `test_tilelang_ascend_simdvf_parallel*.py` | 每例先断言 CCE 源里出现对应 `simd_inst::*`（防静默回落），再与 torch golden 逐位比对；本机无 `torch_npu` → 全部 skip |
| 4 | `vcvt` 转换表补齐（新增 12 个转换对） | `emit.cc::Vcvt` | 逐对离线探针：`build_plan`/`emit_plan` 的实参顺序与 `simd.py` 逐字一致，且经全链路产出 `vcvt + PK` 装载 |

设备用例覆盖（20 例）：max/min 的 NaN/±0、tail chunk（`update_mask`）、单臂谓词化 store、异址双臂互补掩码、mask `==`/`!=`（`pxor`/`pnot`）、常量 shell 末写生效（4 条 store）、shell 归约（`vcadd`）、stride-2 装载（`vgather2`）、间接 gather、`f32→bf16`、`i32→u8`、`i16→u8`、`f16→fp8`、`i32→u32` reinterpret、`f16/i16→f32/i32`（`UNPK_B16`）、`i8→i32`/`fp8→f32`（`UNPK4_B8`）。

`vcvt` 现状（第二轮后：窄化 / 同宽 / 加宽主路径均已覆盖；同宽符号走 `reinterpret` 不占 `vcvt`）：

| 方向 | P123 现支持 |
| --- | --- |
| float → int | f32→{i64,i32,i16}、f16→{i32,i16,i8,u8}、bf16→i32 |
| float → float | f32→{f16,bf16,fp8}、f16→{fp8,bf16,f32}、bf16→{fp8,f16,f32}、fp8→f32 |
| int → float | i16→{f16,f32}、i32→f32 |
| int → int | i32→{i16,u16,u8}、u32→{i16,u16,u8}、i16→{i32,u32,u8}、u16→{u32,u8}、i8→i32、u8→u32 |
| 同宽符号 | `reinterpret`（`i32↔u32`、`i16↔u16`、`i8↔u8`），零指令 |
| 明确不支持（按 nest 回退） | `i8/u8→i16/f16`（板级是半寄存器 chunk）、`i32→i64`、`i64→f32`、`i64→i32`、fp4 系列；加宽 + 谓词/stride/gather 装载 |

### 8.2 缺口复评（替代 §6.2）

| # | 原缺口 | 现状 | 剩余工作 |
| --- | --- | --- | --- |
| 1 | `lanes` 不可配（固定 `2048/elem_bits`） | **仍存在** | 前端 `T.SimdVF(lanes=...)` 写注解 + P2 driver 传入（`BuildPlan` 已有 `lanes` 形参） |
| 2 | fp8→f32、同宽符号互转缺失（无 `vinterpret_cast`） | **已关闭（主路径）**：同宽符号改走 `reinterpret`（零指令）；加宽补 `f16/bf16→f32`、`i16→i32/u32`、`u16→u32`、`i8/u8→i32/u32`、`fp8→f32`（配 `UNPK_*` 装载） | `i8/u8→i16/f16`（板级配方是半寄存器 chunk，与全寄存器 store 不符）与 `i32→i64`/`i64→*` 仍按 nest 回退 |
| 3 | 运行期边界的 serial shell | **已关闭（§8.7）**：plan 级 `shells` 载体，标量壳包 chunk 程序；不可吸收的常量壳（末写覆盖、多地址写）走同一载体；多壳体内置循环分布 | 仅剩：壳内写共享标量、壳内归约需要壳载体的组合仍回退 |

### 8.3 风险复评（替代 §6.3）

| # | 原风险 | 现状 |
| --- | --- | --- |
| R1 | `access_ptr` 只声明 `lanes` 个元素 | **已关闭**：常量 stride 声明 `(lanes-1)*stride+1` 包络，gather/scatter/NZ 散射声明整个 allocation owner（may-access）；负 stride 装载本就走绝对索引通道（base=owner 原点），加宽 load 的不覆盖形态显式回退 |
| R2 | `vmax/vmin` 未固定 NaN/±0 语义 | **已关闭**：契约以 CANN 硬件为基准（NaN 传播、`max` 双 `-0` 才 `-0`、`min` 双 `+0` 才 `+0`）+ 设备用例；保留直接发射（比 Select 重写少指令） |
| R3 | 无发射前 legality 闸门 | **仍存在**：`vsts` 分布/NZ 跨度的对齐要求仍无前置检查 |
| R4 | `step != 1` 未检查、`min != 0` 检查在 P2 | **已关闭**：P1 per-nest 闸门；P2 `RequireZeroMin` 降为兜底 |
| R5 | 无设备级数值回归 | **已关闭（用例就绪）**：20 例覆盖谓词化 × tail × shell × cast × 同宽符号 × 加宽；设备侧执行待 NPU 环境 |

### 8.4 与 PTO 的差异面（最新）

只列**仍有差异**的项；未列出的项视为两边等价。

| 能力 | P123 | PTO | 差异性质 |
| --- | --- | --- | --- |
| `min/step` 形状校验 | ✔ P1 per-nest | ✔ Verify 全局 | 结论一致（P123 粒度更细，可局部回退） |
| `max/min` NaN/±0 | ✔ 契约（以 CANN 为基准：NaN 传播/`±0` 规则）+ 设备用例（直接 `vmax/vmin`） | ✔ `Select` 重写 + 设备用例（IEEE 语义） | **语义不同**：P123 跟随硬件 NaN 传播，PTO 保 IEEE；使用方需按各自契约理解 |
| 设备数值回归 | ✔ 20 例 | ✔ 11 例 | 已对齐 |
| `vcvt` 窄化（fp8 / 16 位 / 8 位目标） | ✔ | ✔ | 已对齐 |
| `vcvt` 加宽（16/8 位源） | ✔：`UNPK_B16`/`UNPK4_B8` 装载 + `PART_EVEN`/`PART_P0` | ✔ | 已对齐（`i8/u8→i16/f16`、`i32→i64`、`i64→*` 仍回退） |
| 同宽符号互转 | ✔：`reinterpret`（零指令） | ✔：`vinterpret_cast` | 已对齐，机制不同 |
| `lanes` 可配 | ✘ | ✔ | P123 缺口 |
| 运行期 serial shell | ✔：plan 级 `shells` 载体（标量壳包 chunk 程序） | ✔ | 已对齐（§8.7） |
| `T.Parallel` 位于 serial 循环下 | ✔：照常 lift，外壳留在原位 | ✔ | 已对齐（§8.7） |
| 非常量 extent | ✔ | ✘ | P123 优势 |
| strided / gather / pair / NZ | ✔ | ✘（PR-5） | P123 优势 |
| Parallel 体内归约 | ✔ | ✘（PR-3） | P123 优势 |
| 失败粒度 | per-loop 回退 | 全局中断 | 取舍 |
| 访存声明面（descriptor 覆盖） | ✔：strided 包络 / 索引访问整 owner | ✔：物理包围 / 索引访问整 owner + 证明 | 已对齐（P123 无“覆盖性证明”，只给保守上界） |
| 发射前 legality | 无 | ✔ | P123 风险（R3） |

### 8.5 收敛度小结

- 实质缺口：仅剩 `lanes` 可配；运行期 / 不可吸收 serial shell 与 serial 循环下的 Parallel 已在第三轮补齐（§8.7）。转换只剩 `i8/u8→i16/f16`、`i32→i64`、`i64→*` 按 nest 回退。
- 风险从 **5 条**收敛为 **1 条**（R3 发射前 legality）；R1 / R2 / R4 / R5 已关闭。
- 唯一尚未闭环的验证是**设备执行**：22 个设备用例与 NaN/±0 契约在无 NPU 主机上只能 skip。建议在 950 环境跑一次
  `pytest testing/ascend/language/test_tilelang_ascend_simdvf_parallel*.py -v`，把实测结果回填本节（尤其 `vmax/vmin` 的 ±0 顺序、
  `UNPK_*` 加宽取值、同宽符号位型、壳循环的末写覆盖）。
- 测试规模：`testing/ascend/transform` 647 + `testing/ascend/analysis` 22 = **669 passed**（三轮共新增 22 个 P1/P2/P3 结构用例）；
  设备用例 22 个（无 NPU 时 skip）。§4.7 的旧计数以本节为准。

### 8.6 第二轮落地（同宽符号 / 访存声明面 / 加宽）

| # | 落地项 | 代码落点 | 验证 |
| --- | --- | --- | --- |
| 1 | 同宽整数换符号：`reinterpret` recipe（零指令） | `value.h::kReinterpret`、`value.cc` Cast 分类、`emit.cc::ReinterpretAs` | 离线：recipe=`reinterpret`、无 `tl.simd.vcvt`；设备：`i32→u32` 的位型全保留（含 INT32_MIN） |
| 2 | 访存声明面：strided 用真实包络、gather/scatter/NZ 用整个 owner | `emit.cc::AccessPtr` / `StridedExtent` / `OwnerElements` / `Vgather2` / `Vscatter` / `Vsstb` | 离线：`a[2*i]` 声明 127、`a[b[i]]` 声明 128；设备：stride-2 与间接 gather 数值不变 |
| 3 | 加宽转换：`UNPK_B16` / `UNPK4_B8` 装载 + 新增 6 个转换对（`u16→u32`、`i8→i32`、`u8→u32`、`fp8→f32` × 3 拼写） | `emit.cc::UnpkDist`、`Emitter::Build`（`kVLoad` 判宽）、`Vcvt` | 离线：8 组 (src,dst) 断言 dist/part；设备：`f16→f32`、`i16→i32`、`i8→i32`、`fp8→f32` 逐位比对 |
| 4 | 不覆盖形态显式回退 | 加宽 + 谓词/stride/gather 装载、非正 stride、8→16 位加宽 | 全链路探针：`i8→i16` 发出 `Unsupported vcvt conversion int8->int16` 并保留原 `T.Parallel` |

### 8.7 第三轮落地（两种 serial 形态）

| # | 形态 | 落点 | 行为 |
| --- | --- | --- | --- |
| 1 | **serial 在 Parallel 外**（`VF → T.serial → T.Parallel`） | `lift_vsimd.cc::RegionBuilder`（下沉进 serial 循环）；`NestCollector` 记录 `outer_shells`；`vsimd_capture_scope` 把外壳计数器判为已绑定 | Parallel 照常 lift/plan/emit，只替换 Parallel 体，外壳连计数器留在原位；`AscendAnalyzeVSimdRegionReport` 打印 `outer shells:` |
| 2 | **shell 在 Parallel 内且可归约** | 既有 `split_trailing_serial` / `SerialIsClean` | 吸收成规约轴（`vcadd` / `vmax` 累加器），与之前一致 |
| 3 | **shell 在 Parallel 内且不可吸收**（末写覆盖、多地址写、运行期边界） | `body.cc::NormalizeBodyInto` 产出 `ShellLoop`；`model.h::VectorPlan.shells`；`emit.cc` 用壳包住 chunk 程序 | 标量壳原样保留、壳内向量化：`for j { for chunk { vld/vadd/vsts } }`；`j` 作为 lane 均匀值进入地址 |
| 4 | 同一 Parallel 体内**多个 shell** | `lift_vsimd.cc::NestFission`（按 shell 边界做循环分布） | 拆成每壳一个 nest，各自独立规划（FA 的「scale 趟 + rowmax 趟」不再整段回退） |
| 5 | 跨 shell 访问对（W 在壳 1、R 在壳 2） | `vsimd_memory_legality.cc`：按 `shell_scope` 跳过 | 两个壳是不同的顺序语句，不再误报跨迭代 RAW/WAR/WAW |
| 6 | shell bound 依赖 lane var / 含 buffer 读 | P1 `IsLaneInvariantBound` + P2 防线 | P1 判 OPAQUE；P2 对直接调用者抛 `Unsupported` |
| 7 | **累积写（归约）不再按"非单射"判非法** | `vsimd_memory_legality.cc::IsCumulativeStore_`（store 级 + 同址 R/W 对） | `l_cur[i] += P_ub[i,j]` 由 OPAQUE 变 LIFTED，是否按归约向量化交给 P2 决定（能规划就发 `vcadd`/累加器，不能就保留原 nest 走 legacy 标量） |

**边界（仍回退）**：壳内写共享标量（`cnt[0] += 1`）、壳内归约与其它 element-wise store 混在同一段且需要壳载体（归约只能走吸收路径）、`while`/`T.Pipelined`、shell bound 读 buffer。

**FA 实测**（`parallel_fa_v2.py` L111-135）：第一段（scale + rowmax）经分布后两段都向量化（`vmuls`+`vscatter`、`vmax` 累加器），第三段（alpha/l_old）为 `vsub/vexp/vmul/vadd`；`P_ub` 那段的内联归约 `l_cur[i] += P_ub[i,j]` 现在 **P1 放行**（累积写按归约处理），但 P2 仍整体拒绝：一个 nest 只能有一个向量轴，NZ 直存要求向量轴连续且必须是段内最后一条 store，而行归约 `l_cur` 又要求整段只有归约/均匀写 —— 轴 `j`（`P_ub`/NZ 连续）与轴 `i`（`l_cur` 连续）各自都撞上另一族约束。源级拆成「elementwise + NZ」和「行和壳」两段即可全向量化（`vsstb` + `vgather2`/`vadd` 累加器）；整 kernel `tilelang.lower` 成功。

---

## 附录：术语对照

| P123 | PTO | 说明 |
| --- | --- | --- |
| `tl.vsimd.Region` | `pto_parallel_loop_layout` annotation | 前者是完整 body 描述，后者只是 lane 映射 |
| `VectorProgram` / `VectorPlan` | （无对应物） | P123 的中间计划 |
| P1 / P2 / P3 | Verify / Legalize+Vectorize | 阶段语义不完全对齐 |
| `update_mask` | `create_mask` | 同一个“生成前缀掩码” |
| `pand/pnot/pxor` | `vand/vor/vnot` + 真值表 | 谓词寄存器族 vs 数据寄存器族 |
| 谓词化 store | value choice + read-back | 两种分支实现 |
| `simd_inst::*` | `pto.vmi.*` | 两个指令层 |
| region `serial` op / shell 展开 | 标量 shell 原样复制 | 两种 shell 处理 |
| `reinterpret`（recipe，零指令） | `vinterpret_cast` | 同宽整数换符号 |
| `UNPK_B16` / `UNPK4_B8` 装载 + `vcvt` part | descriptor/layout 决定的取值布局 | 加宽转换的源寄存器布局 |
| `T.access_ptr` 的 `extent` | VMI descriptor 的声明范围 | 前者只是区间声明，后者带覆盖性证明 |
