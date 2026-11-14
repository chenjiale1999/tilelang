> **历史文档**：本文记录写作时的实现/设计状态。后续改动已删除或重命名部分
> 文件、测试与行号；当前实现见 `src/ascend/transform/parallel_vectorize/` 与
> `src/ascend/transform/vsimd_*`，以代码为准。

读完了报告和全部 23 个源文件（共 11594 行）。结论先给：**分层骨架是正确的，但实现层已经出现了三类系统性的结构问题**——同一语义有 2–3 个事实来源、策略在 emit 层被重复推导、类型靠字符串传递。下面按严重程度列出，每条都给出代码位置。

---

## 一、结构性缺陷（按严重程度排序）

### D1 ⚠️ P1→P2 载体是 untyped Map，P2 解码后丢掉 90% ★最严重

- P1 把 item 事实写成 `rec.meta.Set("kind", ...)` 这种手写键值（[lift_vsimd.cc](file:///home/cjl/workspace/tilelang/src/ascend/transform/lift_vsimd.cc#L520-L535)），P2 再用 405 行手工解码（[vsimd_payload_read.cc](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/vsimd_payload_read.cc#L135-L260)，`GetBuffer/GetVar/ReadOptionalInt/Fail`）。
- 但 [PlanRegionItem](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/vsimd_group_passes.cc#L141-L157) 只用了 `layouts / assumptions / bindings / shape_syms`；`domain`、`accesses`、`outer_vars`、`local_views` 解出来后全部丢弃（报告 §12 C 也承认了）。
- 真正的规划输入是 `body.stmts[span_begin]` **重新跑一遍**规范化与分类（[plan.cc:2593-2598](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/plan.cc#L2593-L2598)）。

**后果**：同一个"访问事实"存在两套推导（P1 PR-A/PR-C 一套，`ClassifyAccess` 一套），两套已经出现差异——报告 §12.1 的 A 类表（P1 拒绝、P2 不拒绝）就是这个差异的证据；另外还有 ~600 行编码 + 405 行解码，只为了传 4 个字段，而且两边都没有 schema 校验。

**改法**：P1 直接写 typed 对象（`VSimdRegion` 已经是 struct），或让 P2 自己从 stmt 推导 layout，然后删掉整个 payload codec。这一条做完可删掉约 800 行。

---

### D2 ⚠️ Plan 已决定的事，emit 又推导一遍，而且写了两遍

Plan 阶段已经算出 `promote`（[plan.cc:1981-1988](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/plan.cc#L1981-L1988)），emit 阶段却重新计算了 5 类策略：

| 策略 | 位置 |
|---|---|
| folded pre（单位元折进累加器） | [emit.cc:2380-2389](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L2380-L2389) **和** [2432-2456](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L2432-L2456)（两遍） |
| invariant pre 是否可外提 | [2092](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L2092) **和** [2393](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L2393) |
| superseded post（寄存器转发） | [2513-2523](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L2513-L2523) |
| carry 资格 | [2548-2557](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L2548-L2557) |
| 外层轴补齐 | [2063-2068](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L2063-L2068) |

而 `EmitSegmentStmts`（2089-2174）与 `EmitParallelAxisParts`（2390-2420）是**同一段 pre/post/prologue 逻辑的两份拷贝**。任何一条规则变化（例如"folded pre 不能被 hoist"）必须同步改 3 处——你记忆里记录的"COW 静默丢写"就是这类多写点造成的。

**改法**：在 `StorePlan` 上加一个 `role` 枚举（`MAIN / PRE_INVARIANT / PRE_FOLDED / POST_SUPERSEDED / ACC_INIT`），由 plan 一次性判定，emit 只 `switch`。两份 pre 处理合并成一个 `EmitPrePost()`。

---

### D3 ⚠️ 三条并存的 segmenter，语义各不相同

| 路径 | 入口 | 拆段方式 | flatten | hoist 计算 |
|---|---|---|---|---|
| 单段 | [SplitNest](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/plan.cc#L313-L375) | 单 serial + pre/post | 允许 | [2630-2635](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/plan.cc#L2630-L2635) |
| 多 serial 段 | [TryBuildProgram](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/plan.cc#L2462-L2570) | `AbsorbSerialChain` | 允许 | [2547-2560](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/plan.cc#L2547-L2560) |
| tile-op 段 | [TryBuildSplitAllProgram](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/plan.cc#L2194-L2459) | `build_loop_source` + `split_trailing_serial` | **禁止**带 serial 轴段 | [2440-2452](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/plan.cc#L2440-L2452) |

三处还各自实现了一遍「相邻链合并」（`merge_chain_bodies/zip_loop_vars/same_loop_shape`，2359-2379），而 `AbsorbSerialChain` 完全不复用。`BuildPlan` 末尾的 [诊断合并 hack](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/plan.cc#L2614-L2628)（用 `std::equal` 判断 notes 是否已经合并）是控制流过于复杂、难以梳理的直接证据。

**改法**：收敛成一个 `Segmenter(body) -> vector<SegmentIR>`，`SplitNest` 退化为"只有一段"的特例；`TryBuildProgram` 的 `seg_positions < 2 → return nullopt` 这类探测逻辑全部消失。

---

### D4 `BuildPlan` 有 4 条入口路径，能力集合互不一致

`BuildPlan → TryBuildSplitAllProgram(可能返回 null) → TryBuildProgram(可能返回 null) → SplitNest`。这四条路径的「允许什么」不同（flatten、serial 吸收、hoist），但对外只暴露一个 `BuildPlan`。新增一种 body 形状时，你无法预知它会走哪条路径；报告 §10 第 13 条「程序粒度拒绝」以及 §12.3 那张"不可达"表，本质上都是这个问题的外在表现。

---

### D5 128 处 `throw Unsupported`，能力边界不可枚举

`plan.cc` 74 处、`body.cc` 26 处、`emit.cc` 15 处。仅 [BuildPlanForAxis](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/plan.cc#L1765-L2069) 一个函数 300 行里就有 15 个 throw，与规划逻辑完全交错。

**后果**："这条 nest 为什么没被向量化"无法结构化回答；报告 §9/§10 的能力表只能人工维护；新增支持必须读懂 300 行才能找到该删哪个 throw。

**改法**：`Capability` 检查列表（每项 `id + 谓词 + 消息模板 + 阶段`），顺序执行并收集 diagnostics；`throw` 只留真正不可恢复处。副产品：能力表可由代码生成，P1 与 P2 的判定标准也就能对齐（报告 §12.2 提到的 `NestedLoopChecker` 与 `ParallelOrdinalCollector` 判定标准不一致问题也一并解决）。

---

### D6 类型全部字符串化

- `recipe` 字符串 if-else 链：[Emitter::Build](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L1252-L1406) 150 行、50 处 `recipe == "..."`；
- `kind_value` 存 `AccessClass` 名字再 `AccessClassFromName` 转回：`emit.cc` 7 处、全库 31 处；
- `status`（`PLANNED/REJECTED/LIFTED/P1_LIFTABLE`）、`storage`（`shared/register`）、`store_lane_mode`（`full/pk_half/pk_quarter`）、`layout_kind`（`nd/nz`）全是裸字符串（30 处）；
- item kind `"REGION"/"COPY"/"REDUCE"` 散落 6 个文件。

**后果**：编译期零校验，拼错只在运行期变成 `no emitter for value recipe 'xxx'`；`Build` 平均要走 10+ 次 `std::string` 比较。

**改法**：内部 `enum class` + `switch`/函数表，只在 FFI 反射边界做一次 to/from。emit 的 150 行 if 链变成 dispatch table，既更快也更好扩展。

---

### D7 `Emitter` 是 550 行 god class

[emit.cc:708-1250](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L708-L1250)：30 个成员、11 参构造、同时负责 SSA 绑定/掩码缓存/谓词/地址/值 memo/pair 域/carry 转发/寄存器文件/prologue。

尤其 [InlineFreshBind](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L771-L785)：通过"回滚最后一条 let + 清理 memo"实现 in-place 写，隐含了对发射顺序的强依赖。这类改写正确与否取决于上一行是否恰好是那条 bind，很难测试。

**改法**：拆 `ValueBuilder`（值 DAG → SSA）、`AddressBuilder`、`MaskCache`、`StoreEmitter`；或者改成"先建完整表达式 DAG，再统一 SSA 化 + 打印"两段式，彻底消除回滚式改写。

---

### D8 值同一性靠"打印字符串"，且 Var 只用名字

[ValueKey](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L535-L550) 递归打印整个子树；[ExprKeyInto](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L327-L331) 对 `VarNode` **只取 `name_hint`**。Buffer 已经用 data 指针修正（495-512），Var 没有。

**风险**：
1. 同名不同 Var（两个 nest 的 `i`、或两次 codegen 重名的 `v0`）会命中同一 cache 键 → 串值；
2. key 无节点级 memo，每个新节点都重打印整个子树，DAG 共享时是重复的平方级工作；
3. `Call` 分支（465-475）在 `op` 是 `GlobalVar` 时只打 `"C:("`，丢掉函数身份 → 不同调用同参数会碰撞；
4. `ExprKeyInto` 里 [396-397](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L396-L397) 是明显的残留死代码（`static_cast<...>(nullptr)` 再 `(void)`）。

**改法**：构造期 hash-consing 分配节点 id，key 用 id 组合（O(1) 且正确）；或给 `Var` 带上唯一 id。

---

### D9 全局互斥锁在 dtype 最热路径上

[primitives.cc:332-383](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/primitives.cc#L332-L383)：`DataTypeOf` / `ElemDataType` / `ElemDtype` 每次调用都 `lock_guard` + `std::map` 查找。而 emit 每条指令都要构造 dtype。TVM 并行编译多个 PrimFunc 时，这条全局锁会把所有编译线程串行化。

**改法**：`thread_local` 缓存，或固定小表（code/bits/lanes 组合有限）用数组/switch。

---

### D10 测试约定绑在英文诊断文案上

`pytest.raises(..., match=...)` 匹配 `Unsupported::message`，报告称其 "byte-identical"（D3）。这是 Python port 时代用 reference 实现当 oracle 的约束——而 `polyhedral_reference` 已经删除（`tilelang/ascend/analysis/parallel_vectorize/` 只剩 `__init__.py` + 60 行 `_ffi.py`），C++ 里仍有 30 处 "C++ counterpart of polyhedral/xxx.py / Python 语义" 注释，是纯粹的历史遗留负担。

**改法**：诊断改成 `reason_code + 参数` 结构化，文案自由；测试断言 code。同时清掉 parity 注释。

---

## 二、编译期性能（按收益排序）

| # | 问题 | 位置 | 建议 |
|---|---|---|---|
| P1 | `written_keys` 每段用未缓存的 `BufferKey()` 重建（会打印 shape/strides/elem_offset） | [emit.cc:2075-2077](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L2075-L2077), [2370-2372](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L2370-L2372) | 用已有的 `CachedBufferKey` |
| P2 | `ProveGuard` **每次调用新建 Analyzer** | [value.cc:60-64](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/value.cc#L60-L64) | 传入共享 analyzer（顺带修掉"用不上 T.assume 导致误判"） |
| P3 | `BuildCopyNest` 自建 analyzer，丢掉 region assumptions | [vsimd_group_passes.cc:194](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/vsimd_group_passes.cc#L194) | 复用 PlanOptions 的 analyzer |
| P4 | 每个 chunk × 每个 store 做 `Substitute+Simplify` | [emit.cc:938-948](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/emit.cc#L938-L948) | 静态 chunk 展开成常量表；或把 lane_base 符号化后复用同一表达式 |
| P5 | 同一段语句被 `PostOrderVisit` 遍历 5 次以上（SerialIsClean / touched_buffers / has_serial_loop / identity_init / ExprReadsBuffers） | plan.cc 多处 | 一次遍历建 reads/writes/loops 索引 |
| P6 | `plan.CopyOnWrite()` 32 次 + `Array.Set` + StorePlan 逐字段 COW（1866-1907） | plan.cc | 构造时一次填满，用 aggregate/builder |
| P7 | 每段新建 `Emitter`，`dup_/forward_/memo_` 缓存随对象销毁，跨段只共享了 mask/ramp | emit.cc | 把 value 级缓存提升到 Program 级 |

---

## 三、优雅与可扩展性的重构目标结构

```
body/normalize ─┐
                ├→ Segmenter ──→ vector<SegmentIR> ──→ Planner ──→ VectorProgram
capabilities ───┘            (唯一拆段逻辑)          (唯一策略判定)
                                                          │
                                              ┌───────────┴──────────┐
                                        StoreRole 枚举          诊断 code
                                              │
                                        Emitter (纯执行, 无策略)
```

四个关键收敛动作：

1. **事实单一来源**（D1）：P1 只写 typed 事实，或者 P2 完全不依赖 P1 payload；删 codec。
2. **策略前移**（D2）：所有 pre/post/累加器/转发判定进 plan，emit 零推导。
3. **类型化**（D6）：`enum class` + dispatch table 取代 recipe/kind/status 字符串。
4. **拆段归一**（D3/D4）：一条 segmenter，消灭 4 条入口。

拆分建议（与 2702/2650 行的大文件对应）：
- `emit.cc` → `emit_value.cc` / `emit_store.cc` / `emit_reduce.cc` / `emit_program.cc` + `emit_internal.h`
- `plan.cc` → `plan_nest.cc`（segmenter）/ `plan_axis.cc`（打分与分类）/ `plan_store.cc`（布局/pack/NZ）/ `plan_reduce.cc` / `plan_api.cc`

---

## 四、建议的执行顺序（按 ROI，且每步可独立验收）

**Step 1 — 零风险清理（1 天内，不动语义）**
- P1/P2/P4/P5 缓存与遍历收敛
- D9 thread_local dtype 缓存
- D8 死代码 + `ExprKey` 的 Var 改成指针/唯一 id
- 收益：编译期可测量变快，并消除一个潜在串值隐患

**Step 1 — 落地记录（2026-11-06）**

| 项 | 改动 | 位置 |
| --- | --- | --- |
| D9 | dtype 三个缓存改 `thread_local`，删除全局 `mutex` | `primitives.cc` |
| P1 | `BufferKey` 增加按 `BufferNode*` 的 thread-local memo，`EmitPlan` 入口清空；覆盖 `SplitLoopInvariant` / `StoreIsLoopInvariant` 里未走 `Emitter` 缓存的调用 | `emit.cc` |
| D8 | 删 `ExprKeyInto` 残留死代码；`Var` 键改对象身份；`Call` 把 callee 纳入键 | `emit.cc` |
| P2 | `GuardFreeBounds` 复用单个 probe analyzer（`allow_override`）；新增"整轴一次证明"快路径；含 `BufferLoad` 的谓词直接短路 | `value.cc` |
| P5 | `touched_buffers` + `has_serial_loop` 合并为单次 `collect_facts`，并缓存上一条链的 facts | `plan.cc` |
| P4 | 测量后不做：单 chunk 发射约 0.5 µs，地址索引不是瓶颈；符号化改写会改变 IR 形态，收益不抵风险 | — |

实测（100 次均值，`copy_1d(4096)` / `masked_1d(4096)` / `reduce_2d(64x128)`）：

| case | before | after |
| --- | ---: | ---: |
| masked_1d plan | 2801 µs | 32.5 µs |
| masked_1d plan + emit | 2928 µs | 92 µs |
| copy_1d plan + emit | 31.5 µs | 31.1 µs |
| reduce_2d plan + emit | 126.6 µs | 108 µs |

> `masked_1d` 的 86× 来自 `ProveGuard` 短路：原来每个 chunk 一次 `CanProve`，其 Z3 回退对含 `BufferLoad` 的谓词必定失败，单次约 0.7 ms。
> 语义影响：含 load 的谓词不再尝试证明（原本也证不出来，保守保留 mask），其余路径不变。

回归：`testing/ascend/{transform,auto_schedule,target,cache}` → **1256 passed / 31 skipped**。

**Step 1 顺带发现（未修，留待 Step 2/3）**

1. ~~`GuardFreeBounds` 尾部探测的 extent 有问题~~ —— **误判，原实现正确**。本 fork 的
   `Range` 构造函数是 `Range(begin, end)`（不是 TVM 传统的 `(min, extent)`），
   `Range(chunk*lanes, extent)` 正是尾块 `[chunk*lanes, extent)`。反面验证：改成
   `Range(chunk*lanes, extent - chunk*lanes)` 会得到空区间 `[128, 2)`，可证谓词立刻漏掉。
2. probe analyzer 从未注入 `PlanOptions.assumptions`，`T.assume` 在 guard 证明里仍不生效（原 P2 描述中的"顺带修掉"未做，属语义变更）。

**Step 2 — emit 去重（中等风险，测试可直接验证）**
- 合并两份 pre/post 处理为一个函数
- 把 foldable/invariant/superseded 判定提到 plan（`StoreRole`）
- 收益：emit 减 ~250 行，策略单一来源

**Step 2 — 落地记录（2026-11-06）**

1. **`GuardFreeBounds` 尾部探测：复核后确认无需修改**。`Range` 在本 fork 是
   `(begin, end)` 语义，原式 `Range(chunk*lanes, extent)` 正确；真正的隐患是把它误读成
   `(min, extent)` 后"修"成 `extent - chunk*lanes`（空区间）。已把 `ProveGuard` 的形参
   改名为 `begin/end` 并注明语义，并新增 `test_tail_chunk_guard_is_proved_free`
   （`N=130` + `if i >= 128` → `mask_free_from == 2`、尾块不再发 `vcmps`）锁住该行为；
   用错误实现跑过该用例，确认会失败。
2. **emit 去重**：新增 `WrittenStoreKeys` / `SplitPreStores` / `EmitHoistedPreStores` /
   `RoleOf`，两条发射路径共用。`EmitSegmentStmts` 与 `EmitParallelAxisParts` 的
   pre 分类、written-buffer 集合、hoisted-pre 发射由各 2 份收敛为各 1 份；
   `EmitParallelAxisParts` 里"先分 invariant/loop、再重算 remaining"的两段式也合并成一次。
3. **`StoreRole`**：`StorePlan.role`（`main` / `pre_folded` / `post_superseded`），
   由 `MarkStoreRoles` 在 plan 阶段判定，emit 只按 role 分组：
   - `pre_folded` 的判据（uniform、无 mask、与累加器同址、无兄弟 pre 读该 buffer）全部进 plan；
   - `post_superseded` 取代 emit 里的 `SameAddressLoad` 扫描；
   - `PRE_INVARIANT` 仍留在 emit：它依赖 `single`（program 段数）与 VF 寄存器文件，
     属发射上下文而非规划事实；但已收敛到 `SplitPreStores` 一处。

| 指标 | before | after |
| --- | ---: | ---: |
| folded 策略实现处数 | 2 | 1（plan） |
| superseded 策略实现处数 | 1 | 1（plan） |
| pre 分类 / written_keys / hoisted-pre 代码处数 | 2 / 2 / 2 | 1 / 1 / 1 |
| `emit.cc` 行数 | 2702 | 2731 |
| `plan.cc` 行数 | 2650 | 2727 |

> 行数没有下降（新增了 role 字段、枚举与共享 helper 的注释），收益是**决策点从 3 处收敛到 1 处**；
> 原估的 "−250 行" 不成立。
> 新增契约断言：`test_serial_sandwiched_statements_absorbed` 直接断言
> `pre_stores[0].role == "pre_folded"`、`stores[0].role == "post_superseded"`；
> `test_tail_chunk_guard_is_proved_free` 断言尾块谓词的 `mask_free_from` 与 `vcmps` 数量。

回归：`testing/ascend/{transform,auto_schedule,target,cache}` → **1256 passed / 31 skipped**。
基准无回退（copy 31 µs、masked 68 µs、reduce 96 µs、tile-op 178 µs）。

**Step 2 补充 —— 死 chunk 消除（2026-11-06）**

`GuardFreeBounds` 是单向证明（谓词恒真 → 免掩码）。对 `not(mask)` 复用同一入口即可得到
互补事实（谓词恒假 → 整块不写）：

```
chunk_lo = max over stores(dead_prefix)      // [0, chunk_lo) 整块恒假
chunk_hi = min over stores(dead_from ?: chunks)
```

`StripMine` 只在 `[chunk_lo, chunk_hi)` 内发块；尾块仅在窗口仍覆盖最后一个 chunk 时发。
触发条件是保守的：静态 chunks、全部 store 都带 mask、无 pre/post、非 reduce 段。

例（`N=130` + `if i >= 128`）：原来 `for chunk in range(2)` 里 2×(vci + vcmps + vld + vsts)
共 8 条死指令，现在整段只剩尾块的 `update_mask + vgather2 + vsts`。

用例：`test_dead_guard_chunks_are_elided`（窗口 `(2,3)`、无 chunk 循环、无 vld/vcmps）、
`test_unmasked_store_keeps_every_chunk_live`（无谓词 store 不误删）。

**Step 3 — typed 载体（收益最大）**
- P1 写 typed、删 `vsimd_payload_read.cc` 的死字段（先删 domain/accesses/outer_vars/local_views，再整体替换 codec）
- 收益：-800 行，并强制 P1/P2 的判定标准同步（报告 §12.2 的不一致自然消失）

**Step 4 — enum/dispatch + 诊断 code（测试要改，量大但机械）**
- recipe/kind/status 枚举化；`Unsupported` 带 code
- 收益：编译期校验 + 性能 + 扩展点局部化（新增一个 tile-op 从改 8 处降到改 2 处）

**Step 5 — segmenter 合并（最大重构，最后做）**
- 三条路径归一；`SplitNest` 变特例

---

## 五、顺手记下的小问题

1. **打分器的魔法权重与注释矛盾**：[AxisWeight](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/plan.cc#L101-L119) 是 100/10/8/4/3/2/1 的拍脑袋权重，[AxisScore](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/plan.cc#L445-L450) 末尾 `+ axis->index` 与 [468 行注释](file:///home/cjl/workspace/tilelang/src/ascend/transform/parallel_vectorize/plan.cc#L468) "equal scores keep nest order" 自相矛盾——实际上内层轴总是被优先。要么写明这是"偏好内层轴"，要么去掉 `+index`。
2. **P1 的 copy 识别用子串**（`IsCopyOp` 查 `"copy"`），body.cc 用精确名，P2 用 payload 解码——三种"什么是 copy"的定义。任何名字里带 copy 的非 tile-op 会被 P1 标成 COPY，然后 P2 解码失败导致整组拒绝。
3. **`VSimdItemPlan` 用 3 个 optional 表达 tagged union**，`kind` 与 `plan_kind` 冗余，不变量（恰好一个非空）无任何强制。
4. **`VRegLanes`/`kMaxVRegsPerVf`(32)/`kManualVRegPool`(24)** 是硬编码的 ISA 假设，V2 只统计手动 fragment，未把单 nest 的自动临时量计入预算 → 多段 VF 理论上可超 32。
5. **P1 的 `validation_error` 与 `unsupported_reason` 都停在 VF 级**，item 只写 `no capture info`，用户看不到真正原因（报告 §10 第 9 条）。

---

**一句话总结**：这套代码的问题不在"功能不对"，而在"同一件事被定义了 2–3 遍"——事实（P1 payload vs 重分析）、策略（plan vs emit×2）、类型（string vs enum）、拆段（3 条路径）。1.16 万行里我估计 **15–20% 是可以直接删掉的重复与历史遗留负担**。按上面 Step 1→3 走，能先删掉 ~1000 行且不改变任何行为，再谈 Step 4/5 才安全。

需要我把这份分析整理成 `docs/compiler_internals/p2_p3_design_review.md`，或者直接开始做 Step 1 的零风险清理吗？

---

## 六、现状盘点（2026-11-06，未完成项）

> 依据：逐项对照当前工作树复核。已完成部分见 §四的 Step 1 / Step 2 落地记录。
> 当前规模：polyhedral 库 **11810 行**（`emit.cc` 2749 / `plan.cc` 2760 /
> `vsimd_payload_read.cc` 405 / `body.cc` 661 / `vsimd_group_passes.cc` 868）；
> 回归 `testing/ascend/{transform,auto_schedule,target,cache}` → **1259 passed / 31 skipped**。

### 6.1 未实现的 Step

| Step | 状态 | 证据 |
| --- | --- | --- |
| **Step 3 — typed 载体（D1）** | 未开始 | `vsimd_payload_read.cc` 仍 405 行；`PlanRegionItem` 只消费 `layouts / assumptions / bindings / shape_syms`，`domain / accesses / outer_vars / local_views` 解码即丢；规划输入仍是 `body.stmts[span_begin]` 重跑分析 |
| **Step 4 — enum/dispatch + 诊断 code** | **已完成（4.1–4.5）** | `recipe` 比较 **0 处**（内部全走 `ValueRecipe` + 族级 dispatch）；`kind_value` 读取 **0 处**（`StorePlanNode::kind`）；`storage / store_lane_mode / layout_kind` 三个小 enum；P2 产出的 group plan 状态走 `VSimdPlanStatus`；`Unsupported` 带 reason code 并贯通到 Python |
| **Step 5 — segmenter 归一** | **已完成** | 只剩 `SegmentBody → vector<SegmentSource> → PlanSources → AssembleProgram` 一条主链；`SplitNest` / `TryBuildSplitAllProgram` / `TryBuildProgram` 三个旧入口已删除；`BuildPlan` 只为「链式切分失败」保留一次单段回落 |
| Step 2 遗留 | 部分 | `PRE_INVARIANT` 仍在 `SplitPreStores`（依赖 `single` 与 VF 寄存器文件）；**carry 资格**与**外层轴补齐**仍是 emit 侧策略 |

### 6.2 未修复的结构性缺陷

| ID | 状态 | 关键证据 |
| --- | --- | --- |
| **D1** P1→P2 untyped Map，解码后 90% 丢弃 | 未修（最严重） | 编码 `meta.Set(...)` + 405 行解码；事实双源导致报告 §12.1 的 P1/P2 口径漂移 |
| **D2** 策略在 emit 重复推导 | 部分（2/5 前移） | folded / superseded 已进 plan；carry 资格、外层轴补齐仍在 emit |
| **D3** 三条 segmenter | **已修（Step 5）** | 切段事实收敛到 `SegmentSource`：`parallel_axes / serial_axis / pre / main / post / aliases / allow_flatten` 一次算清；`SingleSegmentSource` 与 `SplitAllSegmentSources` 产出同一种 source，`PlanSources` 不再关心来源 |
| **D4** 4 条入口路径 | **已修（Step 5）** | `BuildPlan → SegmentBody → PlanSources → AssembleProgram`；旧的 `SplitNest` / `TryBuildSplitAllProgram` / `TryBuildProgram` 已删除；仅保留「链式切分失败 → 单段重试」这一条语义回退 |
| **D5** `throw Unsupported` 不可枚举 | 未修 | **128 处**（plan 74 / body 26 / emit 15 / value 9 / primitives 2 / group_passes 1）；无 Capability 注册表 |
| **D6** 类型字符串化 | 已修（4.1/4.2/4.3/4.4） | recipe → `ValueRecipe` 枚举 + 族级 `switch` + 指令表；`AccessClass` → 影子 `kind`；storage / lane mode / layout 三个影子 enum；item/group status → typed enum。剩纯展示用字符串（`kind` / `plan_kind` / `fragment_mode`） |
| **D7** `Emitter` god class | 未修 | 544 行、30 成员、11 参构造、8 处实例化 |
| **D10** 测试绑英文文案 | 部分（基础设施 + 首批 11 类 code） | `Unsupported.code` 已贯通 C++ → FFI → Python；14 条断言改为 `_rejects(code)`；其余 `throw` 仍用默认 code `unsupported`，仍靠 `match=` |
| ~~D8~~ / ~~D9~~ | 已修 | Var 键改对象身份 / dtype 缓存 thread_local |

小问题 1–5 全部未修（魔法权重与 `+axis->index` 注释矛盾；P1 子串 `IsCopyOp`；
`VSimdItemPlan` 三 optional；`kMaxVRegsPerVf` 硬编码；P1 reason 不落 item）。

### 6.3 未实现的编译期性能优化

| # | 状态 | 现状 / 证据 |
| --- | --- | --- |
| ~~P1~~ / ~~P2~~ / ~~P5~~ | 已完成 | BufferKey memo / probe analyzer 复用 + 快路径 / 单次遍历 |
| ~~P3~~ | **已完成** | `PlanOneGroup` 持有一个 analyzer，下传 `PlanCopyItem` / `PlanReduceItem` / `BuildCopyNest` / `MapReduceDims`；不再每个 item 新建 |
| **P4** 地址索引 Substitute+Simplify | 判定不做 | 单 chunk 约 0.5 µs，非瓶颈；符号化改写会变 IR |
| ~~P6~~ | **已完成（结构）** | `BuildPlanForAxis` 改为一个 `VectorPlanNode* writer`，7 处 `CopyOnWrite()` 收敛为 1 处；实测收益 ~0（见 6.5） |
| ~~P7~~ | **已完成** | `Emitter` 改持 `arith::Analyzer*`，由 `EmitProgram` / `EmitPlan` 传入；不再每 chunk 一个 Analyzer |

> P7 的量化修正：原先"Analyzer 占 67%"的估计是错的（`arith.Analyzer()` 的 5.85 µs 含 Python
> FFI 开销）。用 A/B 探针实测（寄存器模式 16 chunk，emit-only，3 次取最小）：
> 每 chunk 一个 Analyzer 175.8 µs / 共享 145.7–166.3 µs，**per-chunk 从 10.9 µs 降到 7.8–8.5 µs**，
> 约 5–17%（受机器噪声限制）。结构性收益（所有权正确、少一次分配）比数字更确定。

### 6.4 建议顺序

1. **P7（Emitter 共享 Analyzer）** —— 纯性能、不动语义，收益最大（−40~60% 编译期）。
2. **P3 + P6** —— 小改动，继续收敛分配。
3. **Step 3（D1）** —— 删 ~800 行，顺带消解小问题 2、对齐 P1/P2 口径。
4. **Step 4（D6 + 诊断 code）** —— 机械但面广，测试随之改（D10 一并解决）。
5. **Step 5（D3/D4）** —— 最后做；做完才谈 D2 的 carry/invariant 完全前移。

> 进度：第 1、2 项已完成（见 6.5）。第 3 项（Step 3）按用户要求暂缓；Step 4 已全部完成
> （4.2 → 4.3 → 4.1 → 4.4 → 4.5，见 6.6）；Step 5（segmenter 归一，D3/D4）已完成（见 6.7）。
> 计划内剩余：D5 capability registry、小问题 1–5、Step 3（暂缓）。

---

### 6.5 P7 / P3 / P6 落地记录（2026-11-06）

| 项 | 改动 | 结果 |
| --- | --- | --- |
| **P7** | `Emitter` 构造参数新增 `arith::Analyzer*`；成员改为指针；`EmitProgram` / `EmitPlan` 各持一个 analyzer，沿 `EmitSegmentStmts → EmitElementwiseChunks / EmitParallelAxisParts / EmitHoistedPreStores` 下传 | 16-chunk 寄存器段 emit 175.8 → 145.7–166.3 µs；per-chunk 10.9 → 7.8–8.5 µs |
| **P3** | `PlanOneGroup` 持有一个 analyzer，`BuildCopyNest` / `MapReduceDims` 接受指针；删除两处局部构造 | 每 group 一次构造（原每 item 一次） |
| **P6** | `BuildPlanForAxis` 顶部取一次 `VectorPlanNode* writer`，后续字段写入全部走它（7 → 1 次 COW） | 实测无可见变化：节点唯一所有时 COW 只是 refcount 检查；保留是为可读性 |

> P6 的结论：原判断"COW 泛滥导致编译期开销"**不成立**。真正会复制节点的是
> `MarkStoreRoles` / `ClassifyReduce` / `FoldIdentityInit` 里对数组元素的按值取用 + COW，
> 但只发生在被标记的 store 上（每段 0–2 个），量级可忽略。后续不必再投入。

回归：`testing/ascend/{transform,auto_schedule,target,cache}` → **1259 passed / 31 skipped**。
基准（同机、含噪声）：copy 32.5 µs / masked 85.8 µs / reduce 109.7 µs / tile-op 185.4 µs。

---

### 6.6 Step 4 落地记录（4.2 / 4.3 / 4.1 / 4.4 / 4.5，2026-11-06）

| 小步 | 改动 | 结果 |
| --- | --- | --- |
| **4.2 AccessClass** | `StorePlanNode` 新增非反射字段 `AccessClass kind`；`MakeStorePlan` 同时写 `kind_value`（FFI 字符串，Python 测试继续可读）与 `kind`；`plan.cc` / `emit.cc` / `model.cc` 的全部读取改 `store->kind`，诊断文案用 `AccessClassName(...)` 现场生成 | `kind_value` 读取点 **24 → 0**；无 `AccessClassFromName` 往返 |
| **4.3 storage / lane / layout** | 新增影子 enum `StoreStorage{kShared,kRegister}`、`StoreLaneMode{kFull,kPkHalf,kPkQuarter}`、`StoreLayoutKind{kNd,kNz}`；反射字符串保留，节点内写 id；emit 的字符串比较全部改 id（`PairDomainAllowed` 等） | 30 处裸字符串比较 → enum；FFI 契约不变 |
| **4.1 ValueRecipe** | `value.h` 新增 `enum class ValueRecipe`（38 个 recipe）+ `ValueRecipeFamily`（19 个**按 lowering 归类**的族）+ 单一配方表（id / 名字 / 族）；`ValueNodeNode` 新增非反射 `id`，`MakeValueNode` 在构造边界做一次 `string → id`；`Emitter::Build` 按**族** `switch`（leaf 各自一族，二元/一元/标量-向量各一族），族内统一走 emit.cc 的 `SimdOpTable()`（配方 id → `simd_*`，O(1) 数组索引）；`IsElementwiseBinary` 等谓词改为读族，不再各自维护 `switch`；reduction 的 element op 也在入口转 id；删除随迁后无用的 6 个字符串辅助函数 | 新增一个 elementwise 算子 = 枚举 1 行 + `value.cc` 配方表 1 行 + `emit.cc` 指令表 1 行，**Build 与族谓词零改动**；`recipe == "..."` 对节点的比较 **0 处**；`ValueRecipeFromName` 用一次性 `unordered_map` |
| **4.4 group plan status** | `group_plan.h` 新增 `VSimdPlanStatus`，`MakeVSimdGroupPlan` 改收枚举并同时写 `status` 字符串 + `status_id`；P3 的 3 处 `group->status != "PLANNED"` 改读 `status_id` | FFI 字段不变，P2/P3 之间不再比较状态字面量 |
| **4.5 reason code** | `Unsupported` 新增 `code`（默认 `unsupported`）；`MakeError` 增加 `"code"` 键；`_ffi.Unsupported` 新增 `.code` 并由 `_raise` 透传；首批 11 类 code：`body.nested_loop` / `body.unsupported_statement` / `body.divergent_branch` / `body.masked_branch_merge` / `body.statement_predicate` / `value.unsupported_intrinsic` / `reduce.conditional` / `plan.no_vector_axis` / `plan.partial_chunks` / `fragment.non_identity_layout` / `fragment.not_contiguous`；`BuildPlanForAxis` 的 per-axis 回退保留**第一个已分类**的 code（未分类时回落 `plan.no_vector_axis`），`BuildPlan` 的合并重抛沿用原 code；测试侧新增 `_rejects(code)` 上下文助手 | 14 条 `pytest.raises(..., match="英文文案")` 改为断言 code，诊断文案此后可自由改写；其余未分类 `throw` 仍为 `unsupported`（D5 的 Capability 表可在此基础上生成） |

> 关键约束：`recipe` / `kind_value` / `storage` / `store_lane_mode` / `layout_kind` 仍是
> FFI 反射字段，Python 测试（约 38 处 `value.recipe`、`store.kind_value`、`plan.stores[i].storage`）
> 全部保持不变；enum 只是编译期内部事实来源。

验证：
- `testing/ascend/transform` → **593 passed**（另有一个会话期间新增的未跟踪用例
  `test_ascend_SimdVF_alloc_fragment.py` 引用了未定义的 `_programs` 助手，非本次改动，未计入）。
- `testing/ascend/{auto_schedule,target,cache,analysis,layout}` → **700 passed / 41 skipped / 36 failed**，
  36 个失败全部为环境依赖（`npu` device 不存在、`torch_npu` 缺失、bisheng 对系统 glibc
  `__float128` 报错），与本次改动无关。
- 基准（`/tmp/bench_poly.py`，3 次）：copy emit 27.6–39.3 µs、masked emit 60.2–73.1 µs、
  reduce 112.5–127.9 µs、tile-op 159.3–169.2 µs，与 6.5 基线（32.5 / 85.8 / 109.7 / 185.4）
  在噪声范围内，无回退。
顺带发现一处疑似缺陷： WithGuard 重建`ValueContext` 时漏拷`register_fragments` 。在谓词/惰性条件内部读 fragment 时会误报 “fragment register ... without a register-mode plan”。因为涉及“寄存器读是否应当受谓词约束”的语义判断，先没改。

---

### 6.7 Step 5 落地记录（segmenter 归一，D3/D4，2026-11-06）

目标：把「三条并存 segmenter + 四条入口」收敛成一条主链，`SplitNest` 退化为「只有一段」的特例。

| 旧结构 | 新结构 | 说明 |
| --- | --- | --- |
| `SplitNest(stmt, ...)` 直接返回 `vector<Axis>` + 单段规划 | `SingleSegmentSource(...) → SegmentSource` | 吸收 parallel 链 + 后续 parallel/clean serial 循环的逻辑原样搬入；`has_parallel` 检查仍抛 `no T.Parallel axis found`；`CollectAliases(pre+main+post)` |
| `TryBuildSplitAllProgram` 内联构建 source（identity-init 合并、pending store 段、相邻同形链合并） | `SplitAllSegmentSources(...) → Segmentation` | 保留 `same_loop_shape / merge_chain_bodies / zip_loop_vars` 三项合并；`allow_flatten = !serial_axis`、`aliases = CollectAliases(current)` 直接写进 source |
| `TryBuildProgram` 先探测 `seg_positions < 2 → nullopt`，再逐段 `BuildSegment` | `ChainSegmentSources(...)` + `SegmentBody(...)` | 探测逻辑变成「返回空 vector 即不适用」；`SegmentBody` 在空结果时回落 `SingleSegmentSource` |
| 两条路径各自 `PlanSegment(...)`、各自 hoist 规则、各自诊断文案 | `PlanSources(...)` + `AssembleProgram(...)` | 单段：传 base notes、失败即最终；多段：收集 `segment <i>: ...` 后重抛。hoist 的 3 条规则与 3 种诊断文案收敛到一处 |
| `BuildPlan` 的 try A → B → C | `BuildPlan → SegmentBody → PlanSources → AssembleProgram` | 只剩一条语义回退：非 tile-op 的链式切分失败时，把整段 body 当单段重试一次，并用 `MergeNotes` 把各段原因并入最终异常（defect B） |

其它：
- 删除多段路径专用的 `BuildSegment` 包装（归一后无调用者）。
- `Segmentation { parallel_axes, sources }`、`PlannedSegments { segments, flattened_any }` 是两个纯数据中间层，
  规划阶段不再需要知道「自己处在哪条入口路径」。

验证：
- `testing/ascend/transform` → **593 passed**（另有会话期间新增的未跟踪用例
  `test_ascend_SimdVF_alloc_fragment.py` 引用未定义助手，非本次改动，`--ignore`）。
- `testing/ascend/{auto_schedule,target,cache,analysis}` → **688 passed / 31 skipped**。
- 回归期间发现并修复 1 例：多写别名用例丢失 "may alias" 诊断，原因是单段失败路径没有再合并
  segmenter notes；新增 `MergeNotes` 后恢复。
- 基准（`/tmp/bench_poly.py`，4 次）：copy emit 28.9–31.4 µs、masked emit 59.8–96.9 µs（首轮热身偏高），
  与 6.5/6.6 基线同噪声带，无回退。
