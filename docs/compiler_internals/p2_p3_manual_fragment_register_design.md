# Ascend P2/P3 手动 `alloc_fragment`（V 核寄存器）支持设计

> **⚠️ 已废弃（`p_123_parallel_new_p1` 分支）**：本设计描述的 fragment 寄存器文件
> （`PlanOptions.vf_alloc_scope`、G0–G8 门禁、V1–V4 校验、`FragmentRegisterFile`、
> `kRegLoad` recipe、`vf_alloc_scope` 等）已从 P1/P2/P3 与离线 planner 中**全部删除**。
> 当前行为：P1 按 Parallel 粒度把触碰到 `local.fragment` 的 Parallel 标记为 OPAQUE，
> 该 nest 走 legacy 标量路径；P2/P3 无任何 fragment 代码，离线 planner 只保留
> "直接对 fragment nest 调 `build_plan` 抛 `Unsupported`"的守卫。
> 本文仅作为历史设计记录保留。现行说明见
> [vsimd_p1_analysis.md](vsimd_p1_analysis.md) 与
> [p1_region_p2_p3_integration_report.md](p1_region_p2_p3_integration_report.md)。

> **文档定位**：在现有 P1→P2→P3 `T.Parallel` 降级链路上，把
> `frag = T.alloc_fragment((N,), "float32")` 在 `T.SimdVF` 内被 `T.Parallel`
> 直接读写的场景，从「按 UB 内存发射」升级为「按 V 核寄存器发射」。
>
> **现状事实来源**（行号对应当前工作树）：
>
> - 测试：`testing/ascend/transform/test_ascend_polyhedral_vectorize.py:487-534`（§0 Case III）
> - 规划：`src/ascend/transform/polyhedral/plan.cc`、`value.cc`、`model.cc`
> - 发射：`src/ascend/transform/polyhedral/emit.cc`、`stage_passes.cc`
> - 后端：`src/ascend/transform/lower_tile_op.cc`、`src/ascend/codegen/codegen_ascend.cc`
> - 管线：`tilelang/ascend/pipeline.py:95`（VFChecker）、`:124`（polyhedral）、`:126`（LowerTileOp）

---

## 0. 结论速览

1. P1/P2 目前**已经能放行并规划**该场景：P1 只把 `local.var`/`Bind`/语句级
   `if` 判为 OPAQUE，fragment 会正常 `LIFTED`（`vsimd_region.cc:109-113`、
   `vsimd_analysis_common.h:120-144`）；P2 把 fragment 当沿向量轴连续的 ND
   buffer 规划（`plan.cc:1374-1382` 对 `Fragment` layout 直接返回 ND）。
2. **阻断点在 P3 的发射形态**：P3 对 fragment 发 `vld`/`vsts`，codegen 把地址
   强转 `(__ubuf__ float*)`（`codegen_ascend.cc:2369-2372`、`:2424-2426`），而
   `AscendLowerTileOp` 已把 `local.fragment` 重映射成 `local` 标量数组
   （`lower_tile_op.cc:100-105`）。厂商编译器直接拒绝：
   `casting 'float *' to type '__ubuf__ float *' is not allowed in aicore function`。
3. **GM 不在 P2/P3 的建模范围内**：`T.SimdVF` 内的 GM 访存由前端
   `VFChecker` 直接禁止（`vf_checker.py:161-170`、`:152-158`、`:109-116`），
   在进入 P1/P2 之前就已失败。因此 P2 现有的 `in_global` 字段与
   `global-memory operand` 拒绝分支（`access.cc:213`、`value.cc:780-784`）
   属于**不可达的冗余代码**，相关 GM 测试也不再需要；删除清单见 §2.9。
4. **本设计的核心决策**：fragment 在 SimdVF 内就是「每 chunk 一个向量寄存器」，
   P3 不再对它发 `vld`/`vsts`；改为在 VF 作用域声明 `local.var` 向量寄存器，
   寄存器读写走 SSA/`local.var` 通道。寄存器模式按 **VF 原子**生效：
   任一 fragment 不满足约束时，整个 VF 的相关 nest 回退为原来的标量循环
   （不再产生不可编译的 `vld/vsts`）。
5. **现状基线（专项分析结论，见
   `p2_p3_register_usage_flow_analysis.md`）**：P2 不分配寄存器；P3 的寄存器
   全部是系统新建（SSA/`local.var`）；用户 fragment 只被当作内存 buffer。
   同一 fragment 跨多个 nest 时值经内存往返，**没有寄存器复用**。另外，
   两个同名 `alloc_fragment`（不同 data var）会得到相同的 `BufferKey`
   （name/shape 结构键），同一 nest 内的 `forward_` 会串台并已复现误编译。
6. **P2 必须新增 VF 归属守卫**：`T.SimdVF` 外分配、VF 内或多 VF 使用的
   fragment 不会被 eager builder / VFChecker 拦截（只会以 `float*` 参数捕获），
   最终仍会走到 bisheng 的 `__ubuf__` 强转错误。本设计在 `build_plan` 入口
   以 VF 分配集合（`PlanOptions.vf_alloc_scope`）做硬校验，直接拒绝这类场景，
   使 nest 保持标量，避免把错误推迟到厂商编译器。

---

## 1. 场景需求梳理

### 1.1 目标形态与行为定义

```python
with T.SimdVF():
    frag = T.alloc_fragment((N,), "float32")   # 用户手动分配寄存器 tile
    for i in T.Parallel(N):                    # 填寄存器（数据来自 UB）
        frag[i] = T.if_then_else(i < n, a_ub[i], T.float32(-3.4e38))
    for i in T.Parallel(N):                    # 消费寄存器
        c_ub[i] = frag[i] * 2.0
```

行为定义：

| 项 | 定义 |
| --- | --- |
| 存储语义 | 每个 `ceil(size / lanes)` 元素块是一个 **向量寄存器**；`frag[off]` 映射到寄存器 `off // lanes` 的第 `off % lanes` 条 lane |
| 读语义 | 读一个已定义的寄存器 chunk，得到该 chunk 的完整向量值；不产生任何内存访问 |
| 写语义 | 整 chunk 写覆盖寄存器；带谓词的写是 `vsel(旧寄存器, 新值, mask)`（读改写），旧值必须已定义 |
| 持久性 | 寄存器值在**同一个 VF 的一次执行内**跨 nest 有效；进入/退出 VF、跨越 core、跨 VF 调用都不保证 |
| 别名 | 同一 `buffer->data` 的所有 view 共享寄存器文件（按 data var 索引） |
| 未初始化 | 未写过就读 / 对未初始化 chunk 做带谓词写 = 编译期错误，不做隐式清零 |

### 1.2 parallel 作用域访问约束

设 `off(idx)` 为 fragment 的线性元素偏移，`v` 为向量轴，`lanes` 为 chunk 宽度，
`slot = off // lanes`，`lane = off % lanes`。寄存器模式成立当且仅当：

| # | 约束 | 反例（拒绝） |
| --- | --- | --- |
| R1 | `Δ_v off == 1`（沿向量轴连续） | `frag[2*i]`、`frag[i*i]` |
| R2 | `off(v=0) % lanes == 0`（chunk 对齐） | fragment 从非对齐偏移开始 |
| R3 | 所有非向量轴 `a` 满足 `Δ_a slot == 0`（slot 不随循环轴变化） | 2D fragment `frag[r, i]` 中 `r` 被提层为 serial |
| R4 | `slot` 在每个展开 chunk 上是编译期常量 | 符号 slot、动态寄存器下标 |
| R5 | `store_lane_mode == "full"`（无 pack2 / 窄化打包） | bf16 结果写 f32 布局的窄化配对 |
| R6 | fragment layout 在展平索引上是恒等映射（无 padding/replicate） | 带 padding 的 2D fragment layout |
| R7 | 该 fragment 在 VF 内的**所有**使用都由寄存器模式覆盖 | 同一 fragment 又被 `T.copy`/`T.reduce_*`/`T.simd.*` 使用 |
| R8 | fragment 的分配在 VF 内（或 VF block 的 stolen alloc） | 在 VF 外分配、跨多个 VF 共享 |

R7 是「逃逸规则」：任何非 polyhedral 消费者（tile op、手写 `tl.simd.*`、标量
语句）都要求 fragment 仍是可寻址存储，寄存器模式无法同时成立，必须在 VF 级
整体关闭。

#### R1 的适用范围（读、写都必须满足）

R1 **不是**「只要求 store 的 Δ=1」，而是对**该 fragment 的每一次访问**
（load 和 store）都成立。原因：寄存器槽的 lane 布局由 fragment 的线性偏移
唯一确定（`e = off` → `reg = e // L`、`lane = e % L`）。写用 Δ=1 只是把元素
按 lane 0..L-1 装满；如果读用 Δ=k≠1，读到的 lane 顺序就不再是寄存器里的
顺序，需要跨寄存器/跨 lane 置换，而寄存器文件本身无法表达。

| 代码形态 | fragment 访问 | 其他操作数 | 是否支持 |
| --- | --- | --- | --- |
| `frag[i] = a_ub[i]` | store Δ=1 | UB load Δ=1 | ✅ |
| `frag[i] = a_ub[2*i]` | store Δ=1 | UB load stride 2 | ✅（UB 侧 `vload_strided`/`vgather2`） |
| `frag[i] = a_ub[b[i]]` | store Δ=1 | UB gather | ✅（UB 侧索引向量） |
| `c_ub[i] = frag[i]` | load Δ=1 | UB store Δ=1 | ✅ |
| `c_ub[2*i] = frag[i]` | load Δ=1 | UB store stride 2 | ✅（UB 侧 `vscatter`） |
| `frag[i] = frag[i] * 2` | load/store 同址 Δ=1 | — | ✅（原地读改写） |
| `frag[i] = frag2[i]` | load/store Δ=1 | 另一个 fragment Δ=1 | ✅（两组槽） |
| `frag[2*i] = a_ub[i]` | store Δ=2 | — | ❌ G1 |
| `frag[i] = frag2[2*i]` | load Δ=2（frag2 也是 fragment） | — | ❌ G1 |
| `frag[0] = ...` | store Δ=0（lane-uniform） | — | ❌ G1（部分寄存器写） |
| `frag[N-1-i]` | Δ=-1 | — | ❌ G1（lane 反转） |
| `frag[b[i]] = ...` | 间接下标 | — | ❌ G1（跨寄存器 gather） |
| `frag[i] = frag[i+1]` | 读写不同址（stencil） | — | ❌ 现有 `CrossAddressRead` 规则 |

结论：

- **stride 只在非 fragment 操作数上允许**：UB 侧的 load/store 可以是任意
  常量 stride 或 gather，只要写/读 fragment 的那一侧是 Δ=1。
- **对 fragment 本身**，v1 只接受 Δ=1 + chunk 对齐（R1+R2）。其他整数
  stride（0、-1、|k|>1）和间接下标一律拒绝，因为都要求寄存器 lane 的
  置换/跨寄存器 gather。
- 若未来要支持 fragment 的 stride k，需要新增「寄存器内/跨寄存器 lane
  置换」recipe，或者把 fragment 的 layout 定义成 stride-k 并把所有访问
  统一到同一 layout；也可以先 `T.copy` 到 UB 再按 stride 访问（放弃寄存器
  收益）。这些都不属于 v1。

### 1.3 生命周期

```
enter SIMD_VF block
  ├─ register file 建立：fragment -> R 个 vector<elem x lanes> 槽
  ├─ nest #0（写）: slot 定义，init[slot]=true
  ├─ nest #1（读/写）: 读已定义槽；未定义 -> 诊断
  ├─ ...             （只允许直线序列；跨 if/for 的 join 需保守判定）
exit SIMD_VF block
  └─ 寄存器文件销毁（无跨 VF 持久化）
```

- **捕获**：P1 的 `captures`/`layouts` 已经记录 fragment 及其 layout
  （`vsimd_region.cc:129-152`、`vsimd_p2_input.cc`），无需新增捕获字段。
- **作用域传递**：`T.SimdVF()` 退出时会把 fragment 分配 steal 进 `SIMD_VF`
  SBlock 的 `alloc_buffers`（`src/ascend/ir.cc:271-299`），因此寄存器声明
  可以稳定地挂在 VF block 上，跨 nest 可见。
- **并发安全**：SimdVF 没有线程维度，lane 之间是 SIMD 并行、无数据竞争；
  真正的共享风险只有 (a) view 别名、(b) fragment 逃逸出 VF、(c) 跨 VF 复用
  同一 data var。三者分别由「按 data var 建寄存器文件」「R7」「R8」处理。

### 1.4 资源模型

- 每个 fragment 需要 `R = ceil(prod(OutputShape) / lanes)` 个向量寄存器。
- 手动寄存器是 **pinned**：先于自动临时寄存器分配，且不被复用/驱逐。
- 自动临时寄存器（P3 的 SSA `Bind`、`vdup`、`vci`、掩码等）由厂商编译器做
  物理分配；我们的职责是保证「手动 + 自动」的峰值不超过 VF 寄存器预算，
  超限时在规划期拒绝，而不是让 ccec 报错。
- 不做寄存器 spill 到 UB（v1）。

### 1.5 现有架构中相关模块边界

| 阶段 | 模块/入口 | 与 fragment 的关系 | 现状 |
| --- | --- | --- | --- |
| 前端 | `T.alloc_fragment` → `local.fragment`（`tilelang/language/allocate.py:81`） | 分配 | 与 SimtVF 共用 |
| 前端检查 | `tilelang/ascend/analysis/vf_checker.py` | 禁止 GM、禁止分配逃逸 | 已覆盖，无需改 |
| P1 | `BuildP2PlanInputs` / `BuildMinimalRegion`（`vsimd_region.cc`） | 捕获 fragment/layout，判 OPAQUE | 放行 fragment；`local.var`/`if` 拒绝 |
| P2 | `BuildPlan`（`plan.cc`）/ `BuildValue`（`value.cc`） | 分类/规划 fragment 读写 | 当 ND buffer 规划；`IsPrivateScope` 把 `local.*` 当私有标量 |
| P2 驱动 | `PlanMarkerReplacer`（`stage_passes.cc:182-217`） | 逐 nest 规划，失败回退 | 无 VF 级视角 |
| P3 | `EmitPlan`（`emit.cc:2323+`） | 发 `vld/vsts`、SSA、carry | 无寄存器文件 |
| P3 驱动 | `PlanMarkerLower`（`stage_passes.cc:219-240`） | 逐 marker 发射 | 无 VF 级状态 |
| 后处理 | `AscendLowerTileOp`（`lower_tile_op.cc:100`） | `local.fragment`→`local` | 会破坏寄存器语义 |
| 后端 | `codegen_ascend.cc` | `vector_T` / `local.var` / `__ubuf__` 强转 | 已有寄存器类型通道 |

### 1.6 多 parallel 循环与跨 VF 的前置事实（专项分析实测）

| 事实 | 证据 | 对设计的影响 |
| --- | --- | --- |
| 一个 fragment 被多个 nest 使用：两个 nest 都 LIFTED，但 P3 分别发射；写 nest 发 `vsts` 到 fragment，读 nest 发 `vld` | M1 完整 lowering 源码 | 必须引入 per-VF 寄存器文件才能跨 nest 复用 |
| 不同循环重复 `alloc_fragment`（循环体内也合法）→ 多个独立 Buffer，全部被 steal 到同一 `SIMD_VF` block 的 `alloc_buffers` | M2 IR | 寄存器槽按 data var 分组；不同 alloc 永不合并 |
| 同一 Python 变量名分配两次 → 两个不同 data var，但 `buffer.name`/`data.name` 相同；codegen 重命名为 `frag`/`frag_1` | M3 IR + 源码 | 寄存器文件/转发状态必须按 data var 索引，不能按名字 |
| 两个同名 fragment 在同一 nest 内一个被写、另一个被读 → P3 `forward_` 串台，`c_ub` 被写成错误值 | M6 plan 正确、emit 错误 | 先修 `BufferKey`（加唯一 id），否则复用机制会继承该缺陷 |
| VF 内分配、VF 外/VF2 使用 → eager builder 报 `Immutable variable ... used outside its defining region!`，打印 defining frame 与 current frames | C1/C3 实测 | 该方向可排除出 P2/P3；无需新增机制 |
| VF 外分配、VF 内/多 VF 使用 → 不报错，fragment 以 `float*` 参数传入 `__simd_vf__` | C2/C4 实测 | **必须在 P2 `build_plan` 拦截**（见 2.1.2） |
| GM 在 `vf_checker` 中被显式拒绝（load/store/copy 三处），早于 P2/P3 | `vf_checker.py:109-116/152-158/161-170` | P2/P3 **不建模 GM**；删除 P2 的 `in_global`/GM 分支与相关测试（见 §2.9） |

---

## 2. 系统性设计方案

### 2.1 存储分类与接口适配

#### 2.1.1 存储分类（P2）

在 `access.h` 增加存储类别，替代目前只有 `in_global` 的二分：

```cpp
enum class StorageClass { kShared, kFragment, kPrivate };
```

`AccessInfo` 新增反射字段 `storage`。判定：

- `buffer.scope() == "local.fragment"` → `kFragment`
- `buffer.scope() == "local"` / `"local.var"` → `kPrivate`（保持现有 S2 语义）
- 其余 → `kShared`

**不包含 `kGlobal`**：GM 已被前端 `VFChecker` 拒绝，永远到不了 P2/P3
（见 §2.9）；因此也不保留 `in_global` 字段。

`IsPrivateScope`（`model.cc:70-73`）**不再匹配 `local.fragment`**：fragment 是
寄存器数组，不是「每迭代私有标量」。这会同时修正：

- `ClassifyReduce` 不再无条件跳过 fragment store（`plan.cc:1073-1076`）；
- `plan.cc:1618-1628` 的 private-scalar 拒绝不再误伤 fragment。

注意：`frag[0] = max(frag[0], x[i])` 这类 lane-uniform 写 fragment 仍然非法，
但现在由 R1/R2（写必须覆盖整 chunk）拒绝，诊断从 `private scalar` 改为
`fragment register: non-contiguous fragment access`，不会误入归约路径。

#### 2.1.2 参数/合法性校验（P2 `build_plan` 入口 + P2.5 VF 级）

`T.alloc_fragment` 本身不改签名（SimtVF 继续使用）。校验分两层：**P2
`build_plan` 负责逐访问、逐 nest 的硬校验**（包括用户要求的「VF 归属」拦截），
**P2.5 负责需要 VF 全局视角的校验**（逃逸、预算、跨 nest 状态）。

##### A. `build_plan` 的输入契约

P2 只拿到 nest 语句，无法从语句本身判断 fragment 是否在**当前 VF**内分配。
因此 `PlanOptions` 增加一个只读字段：

```cpp
struct PlanOptions {
  // ... 现有 layouts / assumptions / bindings / shape_syms
  // 当前 SIMD_VF block 的 alloc_buffers 中的 buffer（按 data var 去重）。
  // 为空表示「离线调用，无 VF 上下文」。
  std::vector<tirx::Buffer> vf_alloc_scope;
};
```

- **流水线路径**：P1 的 `captures` 已经是 `(buffer, tag)` 列表
  （`vsimd_region.h:70`；`vsimd_capture_scope.cc:185-186` 对 VF block 的
  `alloc_buffers` 打 `"alloc"` 标签），P2 驱动只需过滤 `tag == "alloc"` 的
  条目即可得到 `vf_alloc_scope`，**无需新增 region 字段**。`PlanMarkerReplacer`
  （`stage_passes.cc:195-206`）把它填进 `PlanOptions`。
- **离线路径**：Python `build_plan(stmt, layouts=None, vf_alloc_scope=None)`
  增加同名可选参数；不传时视为「无 VF 上下文」，任何 fragment 访问都按
  G0 拒绝（避免离线工具悄悄生成不可编译的 plan）。
- 该字段只用于**校验**，不参与规划结果；`render_plan` 不输出它。

##### B. 逐访问/逐 nest 校验（在 `BuildPlan` 内，先于任何 store 规划）

§1.2 的 R1–R8 是语义规则名；本节 G1–G8 / V1–V3 是落地校验码（G≈R，
外加 G0/G0b 的 VF 归属与上下文校验）。新增 `ClassifyFragmentAccess(...)`
（`plan.cc`），对每个 `StorageClass::kFragment` 的 access 执行：

| 码 | 校验 | 拒绝文案（`Unsupported`） |
| --- | --- | --- |
| **G0** | fragment 的 `buffer->data` 必须属于 `options.vf_alloc_scope`；否则它是 VF 外分配（会被 bisheng 以 `float*` 参数捕获） | `fragment register: '<name>' is allocated outside this VF; bisheng would capture it as a pointer. Move T.alloc_fragment inside T.SimdVF` |
| **G0b** | `vf_alloc_scope` 为空（无上下文）时，任何 fragment 访问都拒绝 | `fragment register: no VF allocation scope provided; fragment register mode requires the P1 region hand-off` |
| G1 | `Δ_v off == 1`（沿向量轴连续） | `fragment register: '<name>' is not contiguous along vector axis '<axis>' (Δ=<...>)` |
| G2 | `off(v=0) % lanes == 0`（chunk 对齐） | `fragment register: '<name>' chunk base <off> is not aligned to <lanes> lanes` |
| G3 | 非向量轴 `a`：`Δ_a slot == 0` | `fragment register: '<name>' slot changes along axis '<a>'; a register has no dynamic index` |
| G4 | `slot` 在每个展开 chunk 上是常量 | `fragment register: '<name>' slot is not a compile-time constant` |
| G5 | `store_lane_mode == "full"` | `fragment register: '<name>' needs a packed/narrowing store (<mode>)` |
| G6 | fragment layout 恒等（无 padding/replicate） | `fragment register: '<name>' has a non-identity layout (<layout>)` |
| G7 | shape 静态、dtype 在支持表内、`lanes` 匹配 | `fragment register: unsupported dtype/shape/lane width ...` |
| G8 | 单次访问覆盖整 chunk；带谓词写只在 `init=true` 时允许（init 由 P2.5 提供的逐 nest 状态决定；`build_plan` 单 nest 内以「同 nest 先整写」为保守判据） | `fragment register: '<name>' partially writes chunk <k> before it is initialized` |

说明：

- G0/G0b 是**用户要求的拦截点**：它专门覆盖「VF 外分配、VF 内/多 VF 使用」
  这条 eager builder 与 VFChecker 都不拦的路径（见 1.6 C2/C4）。
- G0 的拒绝发生在 `BuildPlan` 内部，抛出的是既有 `Unsupported`；P2 驱动
  `PlanMarkerReplacer` 捕获后**不挂 `tl.vsimd.array_plan` 载体**
  （`stage_passes.cc:207-215`），nest 原样保留，后续按标量循环降级，
  不再进入 P3 的 `vld/vsts` 发射，也就不会把 `__ubuf__` 强转带到 bisheng。
- G1–G8 只做**拒绝**，不做「寄存器/内存」二选一；fragment 一旦不能寄存器化，
  整个 nest 不规划，避免混用两套语义。

##### C. VF 级校验（P2.5，`build_plan` 之后）

以下三项无法在单个 nest 的 `build_plan` 内完成，放在 P2 驱动的 VF 级阶段
（新增 `PlanFragmentRegisters`）：

| 码 | 校验 | 说明 |
| --- | --- | --- |
| V1 | 逃逸：该 fragment 在 VF 内的所有访问都必须被寄存器化 plan 覆盖 | 扫描 VF body 中所有 `BufferLoad/Store`；被 `T.copy`、`T.reduce_*`、`T.simd.*`、标量语句使用时拒绝（R7） |
| V2 | 预算：`ΣR ≤ kManualVRegPool`；`R_total + peak_temps ≤ kMaxVRegsPerVf` | 见 2.3 |
| V3 | 一致性：同一 data var 在所有 plan 中的 `R/lanes/dtype` 一致；同名不同 data var 不得合并 | 见 2.8 |
| V4 | 生命周期：按 nest 程序序维护 `init[data][slot]`；读未初始化槽、未初始化带谓词写、跨 `if`/`for` join 的状态不一致 → 拒绝 | 见 2.2/2.7.3 |

VF 级拒绝同样是**原子**的：丢弃该 VF 下全部 `tl.vsimd.array_plan` 载体，
所有 nest 一起回退标量。

##### D. 参数表（逐项）

| 校验项 | 规则 | 层 |
| --- | --- | --- |
| VF 归属 | `data` ∈ `vf_alloc_scope` | `build_plan`（G0） |
| shape | 静态常量，`prod(shape) > 0` | `build_plan`（G7） |
| dtype | int8/16/32、uint8/16/32、f16/bf16/f32；bool/fp8 暂不支持 | `build_plan`（G7） |
| lanes | `VRegLanes(elem)` 且与 `plan.lanes` 一致 | `build_plan`（G7） |
| layout | 恒等 `Fragment`（`forward_index=(_i,)`、replicate=1） | `build_plan`（G6） |
| 访问 | G1–G5、G8 | `build_plan` |
| 逃逸 | V1 | P2.5 |
| 预算 | V2 | P2.5 |
| 跨 nest 一致性 | V3 | P2.5 |

#### 2.1.3 内存类型映射

| 源 | 现状发射 | 目标发射 |
| --- | --- | --- |
| GM（SimdVF 内） | 前端 `VFChecker` 拒绝（P2/P3 不可达） | **不建模**：删除 P2 的 `in_global`/GM 分支与 GM 测试（见 §2.9） |
| `shared`/`shared.dyn` | `vld`/`vsts` | 不改变 |
| `local.fragment`（寄存器模式） | `vld`/`vsts` + `__ubuf__` cast（不可编译） | `local.var` 向量寄存器读写，**零内存指令** |
| `local.fragment`（未启用/不满足） | 同上（不可编译） | **不发射**，nest 保持标量循环 |
| `local.var` / `local` 标量 | private scalar / 累加器提升 | 不改变 |

#### 2.1.4 寄存器预留机制

- P2.5 为每个寄存器模式 fragment 预留 `R` 个槽，写入 VF 级
  `FragmentRegisterMap`（新载体 `tl.vsimd.fragment_regs`）：
  `{buffer_data_var, chunks, lanes, dtype}`。
- P3 进入 `SIMD_VF` block 时一次性建立寄存器文件：
  - 每个槽 = 一个 `local.var` 标量 buffer，元素类型 `elem x lanes`
    （与 `emit.cc:1967-1974` 的 reduce 累加器同一形态，codegen 发 `vector_f32`）；
  - 命名 `frag_<buffer>_r<k>`，与自动临时 SSA 前缀（`s{i}_vs...`）分开。
- 原 fragment 的 `sblock_alloc_buffer` 与祖先 block 的 `alloc_buffers` 条目
  在 P3 中移除；对应 `layout_map` 条目一并删除，避免 `AscendLowerTileOp`
  再做 layout 重映射。

### 2.2 生命周期管理

1. **捕获**：沿用 P1 的 region（`captures`/`layouts`/`local_views`）。
   `local_views` 用于把 view 归并到同一 data var。
2. **作用域**：寄存器文件挂在 `SIMD_VF` SBlock 上；声明只能出现在 VF block 内，
   保证 codegen 生成的 `vector_f32` 是 `__simd_vf__` 函数的局部变量。
3. **定义-使用**（P2.5 做 VF 级数据流）：
   - 按 VF body 的程序序维护 `init[fragment][slot]`；
   - 整 chunk 无谓词写 → `init=true`；
   - 带谓词写 → 若 `init=false` 拒绝；否则发射 `vsel(旧值, 新值, mask)`；
   - 读 `init=false` 的槽 → 拒绝；
   - 遇到 `if`/`for` 等控制流且 fragment 跨该边界活跃 → 保守拒绝
     （v1 只支持同一 block 层级的直线 nest 序列）。
4. **别名**：寄存器文件 key = `buffer->data`；`T.view` 只允许恒等/纯 reshape
   视图，非恒等 view 直接拒绝。
5. **并发安全**：
   - lane 间无竞争（SIMD 语义）；
   - 跨 VF：G0 拒绝 VF 外分配；VF 内分配跨 VF 使用由 eager builder 拒绝；
   - 跨 core：`T.Kernel` 内每个 block 独立执行同一 VF，寄存器天然私有；
   - 手动 `T.simd.*`/tile op 与 fragment 混用：V1 拒绝（避免两套存储语义）。

### 2.3 寄存器资源冲突检测与调度

**冲突来源**：手动 fragment 槽 vs P3 自动 SSA 临时值 vs 掩码/索引寄存器。

策略（按优先级）：

1. **pinned-first**：手动槽先占 `R_total`；自动临时只能用剩余预算。
2. **静态预算**：`kMaxVRegsPerVf`（ISA 事实，默认暂定 32，手动池上限
   `kManualVRegPool = 24`，留 8 个给自动临时；两个常量放在
   `primitives.h`，验证阶段用硬件/编译器标定后固化）。
3. **压力估算**：
   - P2.5 用 `R_total` 做硬校验；
   - P3 对每个寄存器段做一次 **dry-run 计数**（复用 `EmitSegmentStmts`
     构造 AST 后只统计 `lets_` 峰值，不落盘），若 `R_total + peak_temps >
     kMaxVRegsPerVf` 则整个 VF 回退；
   - 该检查必须在写任何语句前完成，保证 VF 原子性。
4. **调度规则**：
   - 寄存器槽不参与 SSA 的 memo/CDM 消重，也不被 LICM 提升出 VF；
   - 不同 fragment 不复用槽（v1 保守；v2 可做基于 last-use 的槽复用）；
   - chunk 循环对寄存器段**完全展开**（动态寄存器下标不可用），
     `chunks` 必须是常量且 `≤ kFragmentUnrollLimit`（默认 16）。
5. **超限行为**：VF 级回退为标量，并在 `Unsupported.diagnostics` 里给出
   `required/ budget` 明细；不允许 ccec 阶段才失败。

### 2.4 兼容性适配

| 场景 | 处理 | 保证 |
| --- | --- | --- |
| 现有 UB 路径（`shared`/`shared.dyn`） | 代码路径不变 | 166 个既有用例逐字节行为不变 |
| SimtVF 中的 `alloc_fragment` | polyhedral pass 不处理 | 不进入本设计 |
| SimdVF 中 fragment 满足 R1–R8 | 自动启用寄存器模式 | 业务代码零修改，直接得到正确代码 |
| 不满足 R1–R8 | 整个 VF 的相关 nest 保持标量（`tl.vsimd.array_plan` 被丢弃） | 不再生成不可编译的 `vld/vsts`；行为可预测 |
| 语句级 `if` 填充（topk_gate 原形） | P1 仍判 OPAQUE（`vsimd_analysis_common.h:120-144`） | 文档明确建议用值级 `T.if_then_else`；P1 CF 扩展另立项 |
| 依赖旧「fragment 发 vld/vsts」的测试（`test_manual_fragment_register_in_parallel`） | 更新断言为「fragment 不出现 vld/vsts」 | 测试随实现一起改 |

「回退」只丢弃 P2 的 plan 载体，P1 的 region 载体也一并解开，最终 nest 与输入
逐字节一致（复用 `stage_passes.cc:207-215` 的回退路径）。

### 2.5 异常处理与报错提示

所有用户可见错误都在 P2/P2.5 以 `Unsupported` 抛出，带 buffer 名、轴名、
slot、原因与修复建议；进入 `program.diagnostics`，`render_plan` 与
`TL_POLYHEDRAL_DUMP_IR=1` 可定位。

| 诊断 | 触发 | 文案要点 |
| --- | --- | --- |
| `fragment register: non-identity layout` | G6 | 打印 layout 的 forward/输出形状；建议改 1D fragment 或 stage 到 UB |
| `fragment register: non-contiguous fragment access` | G1/G2 | 打印 `Δ_v` 与该 access 的 offset |
| `fragment register: dynamic register slot` | G3/G4 | 打印随哪个轴变化、slot 表达式；建议把该轴放到 VF 外或改 1D |
| `fragment register: packed/narrowing store` | G5 | 打印 `store_lane_mode`；建议显式 cast 到 UB |
| `fragment register: 'x' is also used by '<op>'` | V1 | 指出逃逸语句位置；建议 `T.copy(frag, ub)` 后再用 |
| `fragment register: read of uninitialized chunk` | 生命周期 | 打印 nest 序号与 slot；建议先整块写 |
| `fragment register: partial write to uninitialized chunk` | G8 | 建议用 `T.if_then_else` 写满整个 chunk |
| `fragment register: N registers required, budget is M` | V2 | 列出每个 fragment 的 R；建议缩小 fragment/拆分 VF |
| `fragment register: '<name>' is allocated outside this VF` | G0 | 打印 data Var id；建议把 `T.alloc_fragment` 移进 `T.SimdVF` |
| `fragment register: no VF allocation scope provided` | G0b | 提示离线调用需要 `vf_alloc_scope` / P1 region 上下文 |

P3 内部不一致（plan 标记 register 但 VF 表缺失）用 `TVM_FFI_ICHECK`，
视为编译器 bug，不向用户暴露。

### 2.6 单 SimdVF 多 parallel 循环：全流程处理逻辑

本节按阶段逐一说明「一个 VF 内多个 `T.Parallel` 循环、在不同循环中重复调用
`alloc_fragment`」时每个环节做什么、依据什么身份、产生什么状态。核心原则：

1. **寄存器归属的唯一身份是 `buffer->data`（data Var），不是 buffer 名字、
   不是 Python 变量名、不是分配顺序。**
2. **一次 `alloc_fragment` = 一组寄存器槽**；同一个 buffer 在多个 nest 中出现
   = 同一组槽；不同 buffer（即使同名）= 不同组槽，永不合并。
3. **P2 逐 nest、P3 逐 marker**；跨 nest 的寄存器状态只存在于 P3 的 per-VF
   `RegisterFile` 与 P2.5 的 VF 级 `FragmentRegisterMap` 中。

#### 2.6.1 阶段一：eager builder 与 IR 形态

- 每次 `T.alloc_fragment` 都产生一个独立 Buffer；`T.SimdVF()` 退出时，
  `SimdVFNode::ExitWithScope` 把 VF 体内所有分配 steal 到 `SIMD_VF` SBlock 的
  `alloc_buffers`（`src/ascend/ir.cc:271-299`）。因此：
  - 分配写在某个 `T.Parallel` 循环体内也合法，最终仍是 VF 级分配；
  - 一个 VF 内的 N 次分配 = N 个 `alloc_buffers` 条目。
- 同名分配：eager builder 为第二个 data Var 生成 `frag_1` 之类的名字，
  但 `buffer.name`/`data.name` 可能都仍是 `frag`（实测 M3/M6）。
  **两者的 `same_as` 为 false，必须按 data Var 区分。**
- 绑定生命周期：在已关闭的 VF frame 内绑定的 buffer 变量，不能在 VF 外使用
  （eager `Immutable variable ... used outside its defining region`，见 1.6）。
  但「VF 外分配、VF 内使用」不报错（指针捕获，见 1.6）。

#### 2.6.2 阶段二：P1 逐 nest 交付

- `BuildP2PlanInputs` 按 VF、按 nest 顺序产出 `VSimdP2Input`：
  `vf_index` / `nest_index` / `status` / `region` / `stmt`。
- **同一个 VF 的多个 nest 共享同一个 `SIMD_VF` SBlock 的 alloc_buffers**，
  但每个 nest 的 region 是独立的；P1 不合并 nest，也不做寄存器决策。
- P1 的 `captures` 已把 VF block 的 `alloc_buffers` 标记为 `"alloc"`
  （`vsimd_capture_scope.cc:185-186`），P2 驱动过滤后即可得到
  `vf_alloc_scope`；P1 无需新增字段。

#### 2.6.3 阶段三：P2 `build_plan` 逐 nest 规划

- 每个 nest 独立调用 `BuildPlan`。G0 先校验该 nest 访问的每个 fragment 的
  data Var 是否在 `vf_alloc_scope` 内：
  - 是 → 继续 G1–G8；
  - 否 → `Unsupported`，该 nest 不产生 plan（Branch B）。
- G1–G8 通过后，plan 里每个 fragment access 带 `storage=fragment`，其中可
  寄存器化的 store 带 `StorePlan.storage=register`；**slot 编号不写死在 plan
  里**，由 P3 按 chunk 基址 + 展开序号计算（保证同一 buffer 在不同 nest 中
  落到同一组槽）。
- 各 nest 的 plan 相互独立；P2 不维护「fragment 已被哪个 nest 写过」的状态。
- 只要一个 nest 的 G0–G8 失败，P2.5 会把**整个 VF** 的 plan 载体一起丢弃
  （原子回退，见 2.1.2 C）。

#### 2.6.4 阶段四：P2.5 VF 级寄存器规划

对每个 `SIMD_VF` block：

1. 收集该 VF 下所有 `tl.vsimd.array_plan` 的 plan；若 P2 已记录该 VF 有任一
   nest 被拒（G0–G8 失败），直接整体回退，不再进入 V1–V4。
2. **V1 逃逸扫描**：遍历 VF body 中所有 `BufferLoad/BufferStore`，把每个
   fragment 的访问与 plan 覆盖的访问比对；有未被 plan 覆盖的访问 → 拒绝。
3. **V3 一致性**：按 data Var 汇总 `{R, lanes, dtype}`；同一 data Var 的 R
   必须一致；同名不同 data Var 各自独立（用 id 区分），不合并。
4. **V4 生命周期**：按 nest 程序序推进 `init[data][slot]`；读未初始化槽、
   未初始化带谓词写、跨 `if`/`for` join 状态不一致 → 拒绝。
5. **V2 预算**：`ΣR` 与 P3 dry-run 的临时寄存器峰值之和 ≤ `kMaxVRegsPerVf`。
6. 全部通过 → 生成 `FragmentRegisterMap`（`tl.vsimd.fragment_regs`）：
   `[{data_var, buffer_name, id, chunks, lanes, dtype}]`，并给 plan 打上
   `register` 标记；任一失败 → 丢弃该 VF 下所有 plan 载体，记录诊断。

#### 2.6.5 阶段五：P3 逐 marker 发射（共享 per-VF RegisterFile）

- 进入 `SIMD_VF` block 时，P3 按 `FragmentRegisterMap` 为每个 data Var 创建
  `R` 个 `local.var` 向量寄存器（`vector_<T>`），命名
  `frag_<sanitized_name>_<id>_r<k>`；`id` 来自 data Var 唯一编号，
  避免同名冲突。
- 按 **程序序** 依次发射 nest：
  - 写 fragment：`reg[slot] = value`（整 chunk）；带谓词 → `vsel(旧, 新, mask)`；
  - 读 fragment：`value = reg[slot]`（不产生 `vld`）；
  - 同一 nest 内的写后读：`forward_` 按 data Var 身份缓存，直接复用；
  - 跨 nest：直接读同一个 `local.var`，值天然保持。
- 发射完成后，VF block 的 alloc_buffers 去掉 fragment，改为寄存器声明；
  祖先 block 的 fragment alloc 与 `layout_map` 条目一并清除。
- VF 退出后寄存器状态销毁，不写回内存（寄存器语义）。

#### 2.6.6 场景逐项对照

| 场景 | 分配身份 | 寄存器槽 | 跨循环读写 | 说明 |
| --- | --- | --- | --- | --- |
| 一次 alloc，nest1 写、nest2 读 | 同一 data Var | 同一组 R 槽 | 直接读寄存器 | 本设计的核心收益 |
| 一次 alloc，nest1 写、nest2 改写、nest3 读 | 同一 data Var | 同一组 R 槽 | 覆盖后读到新值 | 程序序；init 状态保持 |
| 不同循环各 alloc（不同名） | 两个 data Var | 两组独立槽 | 互不影响 | 不合并、不复用 |
| 不同循环各 alloc（同名） | 两个 data Var | 两组独立槽 | 互不影响 | 身份键 + 诊断 id；禁止 `BufferKey` 合并 |
| alloc 写在循环体内 | 被 steal 到 VF block | 同 VF 内正常分配 | 取决于 Python 可见性 | 只要 buffer 引用可见即可用 |
| 同一 fragment 出现在两个 VF | 不可能（VF 内分配） | — | — | eager 报错（方向 A） |
| VF 外分配、VF 内使用 | 不在 `vf_alloc_scope` | — | — | P2 G0 拒绝（方向 B） |
| fragment 同时被 `T.copy` 使用 | — | — | — | V1 逃逸拒绝，整个 VF 回退 |

### 2.7 跨 parallel 循环的安全复用机制

**定义**：复用 = 同一个手工定义的 fragment（同一 `data` Var）在同一个 VF 的
一次执行内，所有寄存器模式 nest 共享同一组 `R` 个寄存器槽；值以寄存器形式
在 nest 之间传递，不经过内存。

#### 2.7.1 约束条件

| 编号 | 约束 | 违反时 |
| --- | --- | --- |
| C1 | 身份：按 `buffer->data` 识别；view 必须恒等 | 非恒等 view 拒绝 |
| C2 | 作用域：分配在**当前 VF** 内（`vf_alloc_scope`） | G0 拒绝 |
| C3 | 覆盖：VF 内所有访问都被寄存器 plan 覆盖 | V1 拒绝，VF 回退 |
| C4 | 静态：slot 编译期常量、chunk 数常量且可展开 | G3/G4 拒绝 |
| C5 | 整块：每次写覆盖整 chunk（或带谓词时旧值已定义） | G8 拒绝 |
| C6 | 顺序：同一 fragment 的定义/使用按 nest 程序序，不允许跨分支 join | 生命周期拒绝 |
| C7 | 预算：`ΣR + peak_temps ≤ kMaxVRegsPerVf` | V2 拒绝，VF 回退 |
| C8 | 隔离：不同 data Var 不共享槽；手动槽不与自动 SSA 混用 | 命名/分配隔离 |

#### 2.7.2 资源分配与冲突规避

1. **pinned-first**：P2.5 先为每个 fragment 预留 R 个槽，剩余预算才给自动
   临时寄存器；手动槽不参与 SSA 的 memo/LICM/复用。
2. **身份键**：`RegisterFile` 用 `data` Var 指针（或唯一 id）作 key；**禁止**
   使用现有 `BufferKey`（name/shape 结构键）作为寄存器身份——同名 fragment
   会冲突（已复现误编译）。S1 阶段建议同时给 `BufferKey` 加唯一 id。
3. **命名隔离**：手动槽 `frag_<name>_<id>_r<k>`；自动临时 `s{i}_v<m>`；
   即使同名也不会生成相同 C 变量。
4. **不跨 buffer 复用槽**：v1 保守，避免 last-use/liveness 误判；v2 可在
   证明不重叠时复用。
5. **不 spill**：超预算直接拒绝，不回退到 UB 内存（避免回到不可编译路径）。
6. **原子性**：V2 失败时整个 VF 回退，不能部分 nest 用寄存器、部分用内存。

#### 2.7.3 状态同步逻辑

每个 fragment 的每个槽维护一个状态：

```
UNDEFINED --整 chunk 写--> DEFINED
DEFINED   --整 chunk 写--> DEFINED（值覆盖）
DEFINED   --带谓词写----> DEFINED（vsel(旧, 新, mask)）
UNDEFINED --带谓词写----> 拒绝（G8）
UNDEFINED --读-----------> 拒绝（read of uninitialized chunk）
DEFINED   --读-----------> 输出寄存器值
```

- 状态按 **nest 程序序** 在 VF 内推进；跨 nest 保持，直到 VF 退出。
- 同一 nest 内「先写后读」由 `forward_`（按 data Var + slot 索引）直接转发，
  不重复读寄存器。
- 不允许把寄存器值隐式写回 fragment 内存，也不允许从内存隐式重载；
  fragment 在寄存器模式下没有内存实体。
- **分支/循环边界**：v1 只支持同一 block 层级的直线 nest 序列；若 nest 之间
  存在 `if`/`for`，且同一 fragment 跨该边界活跃，则拒绝（无法保证 join 后
  所有路径的槽状态一致）。VF 外层的循环不影响：每次 VF 调用重新建立寄存器
  状态。
- **并发隔离**：SimdVF 无线程，lane 间是 SIMD 并行，不存在 data race；
  每个 VF 调用的寄存器文件相互独立；不同 core 执行同一 VF 也各自私有。

#### 2.7.4 异常边界

| 边界 | 处理 |
| --- | --- |
| 读未初始化槽 | 拒绝，打印 nest 序号/slot；建议先整块写 |
| 未初始化就带谓词写 | 拒绝；建议 `T.if_then_else` 写满整 chunk |
| 动态 slot（2D + 提层轴） | 拒绝；建议改 1D 或把该轴移出 VF |
| fragment 逃逸到 `T.copy`/`T.reduce_*`/`T.simd.*` | 拒绝整个 VF；建议先 `T.copy(frag, ub)` |
| 跨 VF 共享（VF 外分配） | G0 拒绝；建议把分配移进 VF |
| 同名不同 data Var | 允许，但必须两个独立槽；诊断带唯一 id |
| 预算超限 | 拒绝整个 VF；建议缩小 fragment/拆分 VF |
| 分支 join 状态不一致 | 拒绝（v1 不支持） |

#### 2.7.5 与用户预期的一致性

- **同一 buffer 跨循环**：同一组寄存器，值保持，无 `vld/vsts`，无额外内存
  流量——符合「手动寄存器复用」的预期。
- **不同 alloc**：不同寄存器组，生命周期独立——符合「每次分配是独立资源」。
- **同名 alloc**：仍按 data Var 独立；不会因为名字相同而合并——符合语义，
  且诊断可区分。
- **parallel 循环之间**：在同一 VF 内按程序序执行（它们不是并发线程），
  前一个循环的寄存器写入对后一个循环可见——与顺序语义一致。
- **VF 之间**：不共享寄存器（VF 内分配）或直接拒绝（VF 外分配）——不会出现
  跨 VF 的隐性状态。

### 2.8 改动清单与实施阶段

| 阶段 | 文件 | 改动 |
| --- | --- | --- |
| S0 前置修复 | `polyhedral/emit.cc` | `BufferKey` 加入唯一 buffer id（或在寄存器/转发状态中改用 data Var 身份），修复同名 fragment 串台（M6） |
| S1 分类/模型 | `polyhedral/access.h/.cc` | `StorageClass`、`AccessInfo.storage`；**删除 `in_global` 及其反射项**（§2.9） |
| | `polyhedral/model.h/.cc` | `StorePlan.storage`、`IsPrivateScope` 排除 fragment、`FragmentReg` 反射 |
| | `polyhedral/plan.h` | `PlanOptions.vf_alloc_scope` |
| | `polyhedral/plan.cc` | **G0/G0b VF 归属拦截** + G1–G8 `ClassifyFragmentAccess`、slot 常量性、reduce 分类修正 |
| | `polyhedral/value.cc` | fragment load 走寄存器 recipe（`reg_load`），不套 guard |
| | `polyhedral/render.cc` | 渲染 `storage=register` 与诊断 |
| S1b P1 交付 | `polyhedral/stage_passes.cc` | P2 驱动从 `region.captures` 过滤 `tag=="alloc"` 得到 `vf_alloc_scope` 并填入 `PlanOptions`；离线调用不传时 G0b 拒绝 |
| S2 VF 规划 | `polyhedral/stage_passes.cc` | 新增 P2.5 `PlanFragmentRegisters`：V1 逃逸/V2 预算/V3 一致性/V4 生命周期、原子回退；新 attr `tl.vsimd.fragment_regs` |
| S3 发射 | `polyhedral/emit.cc` | `RegisterFile`（per-VF、按 data Var key）、寄存器 store/load、masked `vsel` 合并、chunk 展开、dry-run 压力计数、VF alloc 重写 |
| | `polyhedral/stage_passes.cc` | P3 driver 维护 per-VF 寄存器文件；移除 fragment alloc/layout 条目 |
| S4 后端 | `transform/lower_tile_op.cc` | 对 register-managed data var 跳过 layout 重映射 |
| | `codegen/codegen_ascend.cc` | 防御性检查：SimdVF 内 `local.fragment` 不得再走 `vld/vsts`；`local.var` 向量声明沿用 |
| S5 测试/基准 | `testing/ascend/transform/test_ascend_polyhedral_vectorize.py`、`testing/ascend/language/simdvf/`、`examples/ascend/` | 见第 3 节 |

实施顺序建议：S0（修 key）→ S1/S1b（默认仍按内存，但 G0 立即拦截 VF 外分配）
→ S2（VF 级校验/原子回退）→ S3（寄存器发射）→ S4（清理后端）→ S5。

### 2.9 GM 场景裁剪与删除清单

**决策**：`T.SimdVF` 内的 GM 访存由前端 `VFChecker` 在进入 P1/P2 之前直接
拒绝（load / store / copy 三处，`vf_checker.py:161-170`、`:152-158`、
`:109-116`；调用点 `pipeline.py:95`）。因此：

- P2/P3 **不建模 GM**，不为 GM 保留字段、分支或测试；
- `build_plan` 的输入契约收紧为「已经过 P1 + VFChecker 的 nest」；
- 离线直接调用 `build_plan` 构造 GM 操作数属于**契约外**，不保证诊断，
  不应作为测试用例。

#### 2.9.1 代码删除清单

| 文件 | 删除内容 |
| --- | --- |
| `src/ascend/transform/polyhedral/access.h` | `AccessInfoNode::in_global` 字段（约 L111）及其反射项（约 L128） |
| `src/ascend/transform/polyhedral/access.cc` | `node->in_global = buffer.scope() == "global";`（约 L213） |
| `src/ascend/transform/polyhedral/value.cc` | `LoadNode` 开头的 `if (info->in_global) throw Unsupported("global-memory operand ...")`（约 L780-784） |
| `src/ascend/transform/polyhedral/plan.h/.cc` | 本设计新增的 `StorageClass` 不包含 `kGlobal`（见 2.1.1） |
| `tilelang/ascend/analysis/vf_checker.py` | **保留**（它正是 GM 在 P1/P2 之前被拒绝的原因） |

删除后，P2/P3 内不再出现 `global`/`in_global`/`global-memory operand` 字样；
GM 的拒绝行为完全由前端负责。

#### 2.9.2 测试删除/改造清单

文件：`testing/ascend/transform/test_ascend_polyhedral_vectorize.py`

| 用例/位置 | 处理 |
| --- | --- |
| 模块 docstring `§0 Case II a global-memory operand is rejected`（约 L23） | 删除该条目录项 |
| `test_global_operand_rejected`（约 L473-483） | 删除 |
| `test_expression_subscript_rejected_on_global_operand`（约 L1245-1259） | 删除 |
| `_p1_kernel_gm_operand_in_vf`（约 L4070-4090） | 删除 |
| `test_p1_gm_operand_branch_b`（约 L4400-4414） | 删除 |
| `test_rejected_nests_fall_through_every_stage` 的 part (b)（约 L4497-4510） | 删除该分支；part (a)（P1 内存非法）保留 |
| `test_multi_segment_failure_reason_is_preserved`（约 L2061-2085） | **不删除**：该用例验证「段级失败原因不被吞」，当前用 GM 作为失败触发器；改为其他段级失败（如 bool store、不支持的 select），保留诊断合并覆盖 |
| `test_manual_fragment_register_in_parallel` docstring 中「a GM operand inside T.SimdVF is a front-end semantic error」 | 保留这句话作为背景说明即可（不构成 GM 测试） |

删除后，测试套件的负例只覆盖「P1/P2/P3 真正可见」的输入；GM 由
`testing/ascend/` 中 `vf_checker` 相关用例负责（不在本文件的 P2/P3 范围）。

#### 2.9.3 验收

1. P2/P3 源码中不再有 GM 分支（grep `in_global|global-memory operand` 为空）；
2. 删除上述 GM 用例后，polyhedral 套件全绿；
3. 构造 `T.SimdVF` 内 GM 访存的 kernel，报错仍来自 `VFChecker`
   （`SIMD_VF blocks cannot access global memory`），而非 P2；
4. `test_multi_segment_failure_reason_is_preserved` 改用非 GM 触发器后仍能
   证明段级诊断不被吞掉。

---

## 3. 验证方案

### 3.1 单元测试（离线、无硬件）

文件：`testing/ascend/transform/test_ascend_polyhedral_vectorize.py`（沿用
`build_plan`/`emit_plan`/P1/P2/P3 三段离线断言）。

| 用例 | 断言 |
| --- | --- |
| `test_fragment_register_fill_has_no_memory_ops` | fill 段 plan `storage=register`；emit 中**没有**对 frag 的 `vld/vsts`；有寄存器定义 |
| `test_fragment_register_consume_reads_register` | consume 段直接读寄存器；只有对 `c_ub` 的 `vsts` |
| `test_fragment_register_multi_chunk_unrolled` | N=256 → 4 个槽、chunk 循环完全展开、无动态下标 |
| `test_fragment_register_guarded_tail_uses_select` | `T.if_then_else` 尾块填充：1 条 `vsel` + 寄存器写 |
| `test_fragment_register_partial_write_after_init_merges` | 先整写再带谓词写 → `vsel(旧寄存器, 新值, mask)` |
| `test_fragment_register_read_before_write_rejected` | 诊断码/文案 |
| `test_fragment_register_partial_write_uninitialized_rejected` | 同上 |
| `test_fragment_register_dynamic_slot_rejected` | 2D `frag[r,i]` + 提层 r → G3 诊断 |
| `test_fragment_register_non_identity_layout_rejected` | 手工传入 padded layout → G6 |
| `test_fragment_register_mixed_width_rejected` | 窄化 store → G5 |
| `test_fragment_register_escape_rejected` | fragment 同时被 `T.reduce_max` 使用 → V1 |
| `test_fragment_register_budget_rejected` | 构造超预算 → V2，文案含 required/budget |
| `test_fragment_register_view_alias_shares_state` | 恒等 view 读写同一槽 |
| `test_fragment_register_vf_atomic_fallback` | 一个 nest 非法 → 同 VF 所有 plan 载体都被丢弃，IR 逐字节不变 |
| **`test_fragment_alloc_outside_vf_rejected_by_build_plan`** | **G0**：VF 外分配的 fragment 在 `build_plan` 直接 `Unsupported`，nest 不被规划，后续 lowering 无 `__ubuf__` cast |
| **`test_fragment_without_vf_scope_rejected`** | **G0b**：离线 `build_plan` 不传 `vf_alloc_scope` 时拒绝并提示需要 P1 region 上下文 |
| `test_shared_paths_unchanged` | 现有 166 用例全绿，`git diff` 仅新增 |

> **不包含 GM 用例**：GM 由前端 `VFChecker` 在 P1/P2 之前拒绝，相关 P2/P3
> 测试按 §2.9.2 删除；本文件只保留 P1/P2/P3 真正可见的输入。

### 3.2 集成测试（多场景）

1. **端到端 lowering**（无需设备，需 ccec 可用时）：
   - 断言源码含 `vector_f32 frag_<name>_r<k>`；
   - 断言源码中不存在对 `frag` 的 `(__ubuf__ ...)` 强转；
   - 断言 `simd_inst::vld/vsts` 只作用于 UB buffer。
2. **功能矩阵**（`testing/ascend/language/simdvf/`）：
   - N ∈ {32, 64, 100, 256, 1000}；dtype ∈ {f32, f16, bf16, i32}；
   - 单 fragment / 多 fragment（scores + indices）/ fragment→fragment；
   - 2~4 个 nest 的填-读-再写序列；外层 serial 循环内重复使用；
   - 多个 SimdVF block；VF 在 `T.Pipelined`/`T.serial` 内；
   - 与 UB copy、NZ store、reduce（UB 版）共存；
   - 负例全部覆盖 3.1 的诊断。
3. **硬件正确性**（113/950PR）：与 torch golden 对比，`rtol/atol` 按 dtype；
   重复运行 3 次验证确定性；覆盖尾块、N=1、N<lanes、N 非 2 的幂。

### 3.3 性能基准

- 基线：
  - **B0 标量回退**（当前 P1/P2 拒绝后的串行循环）；
  - **B1 UB 版本**（把 fragment 替换成 `T.alloc_shared` + 显式 copy）。
- 指标：
  - `do_bench(..., backend="msprof")`，30 warmup / 50 repeat，3 次取中位数；
  - 源码指令计数：fragment 相关 `vld/vsts` = 0；`vector_f32 frag_*` 声明数 = R；
  - UB/ GM 字节数（应 ≤ B1）；
  - 峰值寄存器声明数 ≤ `kMaxVRegsPerVf`。
- 通过标准：
  - 正确性：与 B1/ golden 完全一致；
  - 性能：`T_reg ≤ 1.0 × T_ub`（相比内存实现不退步），
    且对 N ≥ 64 的纯搬运/逐元素场景 `T_reg ≤ 0.85 × T_scalar`
    （即至少 15% 提升）；
  - 资源：无额外 UB 分配；寄存器预算不超限；
  - 稳定性：连续 3 次测量波动 ≤ 5%。

### 3.4 回归门槛

- `pytest testing/ascend/transform/test_ascend_polyhedral_vectorize.py` 全绿；
- `pytest testing/ascend/` 全绿；
- 现有 166 个 polyhedral 用例的 plan/emit 输出对 UB 场景逐字段不变；
- `TL_POLYHEDRAL_DUMP_IR=1` 下所有诊断可读、无 P3 内部 WARNING。

### 3.5 多 parallel 循环全链路验证矩阵

对应需求「寄存器分配、生命周期、数据读写一致性、并行执行隔离性」四个维度。
每一行都必须同时在 **plan / IR / 源码 / 运行时** 四层取证，缺一不可。

| 维度 | 验证点 | plan/IR 断言 | 源码断言 | 运行时断言 |
| --- | --- | --- | --- | --- |
| 寄存器分配 | 同一 buffer 跨 nest 只有一组槽 | P2.5 `FragmentRegisterMap` 只有一个 `{data, R}`；所有 plan 的 fragment store 引用同一 data | `vector_f32 frag_<id>_r0..R-1` 只声明一次 | 多次运行结果稳定 |
| | 不同 alloc 不共享槽 | 两个 data 各一条 map；R 分别正确 | 两组声明、命名带 id | 两次分配互不影响 |
| | 同名 alloc 不合并 | 两个 data 各一条 map；诊断 id 不同 | `frag`/`frag_1` 两组声明 | 交叉读写结果正确（M6 回归） |
| | 预算不超 | `ΣR ≤ kManualVRegPool` | 声明数 = ΣR | 无编译失败 |
| 生命周期 | VF 入口未定义 | 首个访问若是读 → 拒绝 | — | — |
| | 跨 nest 保持 | 第二个 nest 的 load 由寄存器提供 | 无 `vld` 从 frag | nest2 读到 nest1 写入的值 |
| | VF 退出销毁 | 不生成写回 | 无 `vsts` 到 frag | 第二个 VF 不依赖第一个 VF 的 frag（或按方向 A 报错） |
| | 外层循环重置 | 每个 VF 调用重新声明/初始化状态 | 声明在 `__simd_vf__` 内 | 循环多轮结果一致 |
| 读写一致性 | 同 nest 写后读 | `forward_` 命中，只有一份值 | 无重复 `vld` | 结果等于内存语义 |
| | 跨 nest 覆盖写 | 后写覆盖先写 | 后写更新槽 | nest3 读到新值 |
| | 带谓词写（已初始化） | `vsel(旧, 新, mask)` | 1 条 `vsel` | 未命中 lane 保持旧值 |
| | 部分写未初始化 | G8 拒绝 | — | — |
| | 别名 view | 同 data 同槽 | 同一组声明 | 两个 view 语义一致 |
| 并行隔离 | lane 间 | 无跨 lane 依赖（Δ=1 逐 lane） | 单条向量指令 | 各 lane 结果独立 |
| | nest 间 | 程序序状态推进 | 发射顺序 = 源码顺序 | 后一个 nest 可见前一个 nest 的值 |
| | VF 间 | 方向 A 报错 / 方向 B G0 拒绝 | — | — |
| | 手动 SIMD/tile op 混用 | V1 拒绝整个 VF | — | 回退标量后仍正确 |
| | core 间 | 寄存器为 VF 局部 | 声明在 `__simd_vf__` 内 | 多 block 并行结果一致 |

**判定标准**：以上 20 项全部通过，且 §3.1/§3.6 用例全绿、§3.3 性能达标，
才可判定「处理逻辑完全匹配用户预期」。

### 3.6 多 parallel 循环专项测试用例集

文件建议：`testing/ascend/transform/test_ascend_polyhedral_vectorize.py`
（离线结构）+ `testing/ascend/language/simdvf/test_fragment_register_reuse.py`
（端到端/运行时）。

#### 正常场景（N1–N10）

| 编号 | 用例 | 构造 | 断言 |
| --- | --- | --- | --- |
| N1 | `test_reuse_one_alloc_two_nests` | 一次 alloc，nest1 写、nest2 读 | 同一 map 条目；nest2 无 frag `vld`；源码只有一组 `vector_f32` |
| N2 | `test_reuse_write_read_write_read` | 4 个 nest：写/读/写/读 | 槽状态按序推进；第二个写覆盖；无 `vld/vsts` 到 frag |
| N3 | `test_reuse_fill_then_transform_then_store` | UB→frag→frag 变换→UB | 两个 nest 间无内存往返；只有 UB 侧 `vld/vsts` |
| N4 | `test_reuse_two_independent_fragments` | 两个不同名 fragment，各自两 nest | 两组槽；互不干扰；命名带 id |
| N5 | `test_reuse_same_name_two_allocs` | 同名两次 alloc，各自 nest 使用（不同 nest） | 两组槽；诊断/命名可区分；无串台 |
| N6 | `test_reuse_alloc_inside_loop_body` | `alloc_fragment` 写在 `T.Parallel` 体内 | 被 steal 到 VF block；正常寄存器化 |
| N7 | `test_reuse_multi_chunk_four_registers` | N=256，两个 nest | 4 个槽；chunk 展开；无动态下标 |
| N8 | `test_reuse_guarded_tail_then_consume` | `T.if_then_else` 尾块填满，再消费 | 1 条 `vsel`；无部分写；消费读到完整值 |
| N9 | `test_reuse_outer_serial_loop` | VF 外层 `T.serial` 循环，fragment 状态每轮重置 | 声明在 VF 内；多轮结果一致 |
| N10 | `test_reuse_view_alias` | 同一 fragment 的恒等 view 在两个 nest 使用 | 同一 data/同一槽；语义一致 |
| N11 | `test_reuse_strided_non_fragment_operand` | `frag[i] = a_ub[2*i]`，`c_ub[2*i] = frag[i]` | fragment 访问 Δ=1；UB 侧 `vload_strided`/`vscatter`；无 frag `vld/vsts` |

#### 异常场景（E1–E10）

| 编号 | 用例 | 构造 | 断言 |
| --- | --- | --- | --- |
| E1 | `test_reject_alloc_outside_vf` | VF 外分配、VF 内使用 | `build_plan` G0 `Unsupported`；后续 lowering 为标量，无 `__ubuf__` cast |
| E2 | `test_reject_no_vf_scope` | 离线 `build_plan` 不传 `vf_alloc_scope` | G0b 拒绝 |
| E3 | `test_reject_cross_vf_use` | VF1 内分配、VF2 使用 | eager `Immutable variable ... defining region`（P2/P3 不参与） |
| E4 | `test_reject_read_before_write_across_nests` | nest1 读、nest2 写 | 生命周期拒绝，打印 nest 序号/slot |
| E5 | `test_reject_partial_write_before_init` | 未初始化就带谓词写 | G8 拒绝 |
| E6 | `test_reject_dynamic_slot` | 2D fragment + 提层轴 | G3 拒绝 |
| E7 | `test_reject_mixed_width` | 窄化/打包 store | G5 拒绝 |
| E8 | `test_reject_escape_to_reduce_or_copy` | fragment 另被 `T.reduce_max`/`T.copy` 使用 | V1 拒绝，整个 VF 回退 |
| E9 | `test_reject_budget` | 多 fragment 超过 `kManualVRegPool` | V2 拒绝，文案含 required/budget |
| E10 | `test_reject_branch_join_state` | 两个 nest 之间有 `if`，fragment 跨边界活跃 | 生命周期拒绝（v1） |
| E11 | `test_same_name_same_nest_no_alias` | M6 复现：同名两 alloc，同一 nest 一写一读 | 修复后结果 = `a_ub+7`；两个 data 不共享 forward key |
| E12 | `test_reject_strided_fragment_access` | `frag[2*i] = a_ub[i]` / `frag[i] = frag2[2*i]` | G1：fragment 的读和写都必须是 Δ=1 |
| E13 | `test_reject_fragment_stencil_same_nest` | `frag[i] = frag[i+1]` | 现有 `CrossAddressRead` 拒绝（同 nest 读写不同址） |

#### 验证「拦截逻辑有效性」的端到端断言

对 E1/E2/E8/E9 额外执行整条 pipeline（`tilelang.lower`），断言：

1. 源码中不存在 `(__ubuf__ <T>*)` 对 fragment `local` 数组的强转；
2. 对应 nest 保持标量（无 `simd_inst::` 或只有合法的 UB 侧指令）；
3. 不出现 bisheng 的 `casting ... to type '__ubuf__ ...'` 错误。

---

## 4. 风险与未决问题

1. **寄存器预算常量**：`kMaxVRegsPerVf` 需用目标硬件/编译器文档或实测标定；
   在标定前用保守值，并在 S2 阶段只做校验不开启发射。
2. **`local.var` 向量在一个 VF 内的数量上限**：若 ccec 对 `vector_f32` 变量
   有额外约束（例如必须在同一作用域声明），需要 S3 调整声明位置。
3. **P1 控制流白名单**：语句级 `if` 仍是 OPAQUE；topk_gate 风格尾块填充建议
   改写为值级 `T.if_then_else`。是否扩展 P1 CF 能力应独立评估。
4. **多 VF / 循环内 VF**：寄存器每次 VF 调用重新初始化；若业务代码依赖
   「跨 VF 保留」，当前设计明确不支持，需要在文档中显式说明。
5. **与 NZ/pack2/混合宽度的交互**：v1 一律拒绝（G5/G6）；后续如需支持
   （如 bf16 fragment 的打包对），需要把 `store_lane_mode` 的 pair 语义
   扩展到寄存器槽。
6. **`captures` 的 tag 语义**：G0 依赖 P1 `captures` 的 `"alloc"` 标签；
   `vf_alloc_scope` 必须按 **data Var** 去重后传入。若未来 P1 增加新的
   capture 标签/嵌套 block，需要同步更新过滤逻辑（缺标签时按 G0 拒绝，
   不允许静默放行）。
7. **离线 `build_plan` 的行为变化**：新增 `vf_alloc_scope` 参数后，未传该
   参数的调用会拒绝 fragment 访问（G0b）。现有离线用例
   `test_manual_fragment_register_in_parallel` 必须改为传入 VF 分配集合
   或改走 P1→P2 三段路径；这是有意的破坏性变更，需在提交说明中标注。
8. **`BufferKey` 修复的回归面**：把结构键改为 identity-based 会同时影响
   `forward_`/`store_gen`/`written`/LICM 的判定；需用现有 166 用例 + M6
   回归确认「仅修复同名串台，不改变其他语义」。
9. **`vf_alloc_scope` 的粒度**：当前按整个 `SIMD_VF` block 的 alloc_buffers
   取集合；如果未来支持嵌套 VF 或多个 VF 共享外层分配，需要把「分配所属
   VF」显式建模，而不是仅靠集合成员关系。
