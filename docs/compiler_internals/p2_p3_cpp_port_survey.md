# P2/P3 C++ 化调研与契约（Gate 1 交付物）

> 配套计划：[p2_p3_cpp_port_plan.md](p2_p3_cpp_port_plan.md)。
> 本文固化「现状清点 / 输入输出契约 / 依赖接口 / 调度关系 / 测试耦合面 / 性能基线 / 分支清单 / 验收口径」，
> 是后续阶段三~六的对照基准。任何与本文冲突的实现都属于语义漂移，必须先改本文再改代码。

***

## 1. 现状清点

### 1.1 模块与规模（实测 `wc -l`）

| 模块              | 行数   | def/class | 职责                                                 | C++ 目标                     |
| --------------- | ---- | --------- | -------------------------------------------------- | -------------------------- |
| `__init__.py`   | 114  | -         | 包导出（16 个符号）                                        | 保留为 FFI 薄封装                 |
| `primitives.py` | 255  | 13 / 2    | recipe 表、归约表、硬件常量、`Unsupported`、小工具               | `polyhedral/primitives.h`  |
| `access.py`     | 301  | 11 / 3    | `Axis`/`AccessClass`/`AccessInfo`、线性化、有限差分、分类        | `polyhedral/access.{h,cc}` |
| `body.py`       | 151  | 5 / 1     | `normalize_body`、`StoreSpec`、if→select、别名折叠          | `polyhedral/body.{h,cc}`   |
| `value.py`      | 697  | 29 / 2    | `ValueNode` DAG、`build_value`、装载守卫、索引向量              | `polyhedral/value.{h,cc}`  |
| `model.py`      | 186  | 2 / 4     | `StorePlan/ReduceInfo/VectorPlan/VectorProgram`、累加器判据 | `polyhedral/model.{h,cc}`  |
| `plan.py`       | 1352 | 39 / 1    | `build_plan` 全流程、多段、展平、归约、轴回退                       | `polyhedral/plan.{h,cc}`   |
| `emit.py`       | 1491 | 23 / 2    | strip-mine、掩码、`T.simd.*`、归约两条策略                    | `polyhedral/emit.{h,cc}`   |
| `render.py`     | 55   | 3 / 0     | 计划文本渲染（对拍 oracle）                                   | `polyhedral/render.{h,cc}` |
| `driver.py`     | 35   | 1 / 0     | pre-order 就地替换                                     | 并入新 driver pass            |
| **合计**          | **4637** |       |                                                    |                            |

### 1.2 现状定位

| 组件     | 现在是什么                                                      | 入口                                       |
| ------ | ---------------------------------------------------------- | ---------------------------------------- |
| P2     | 纯函数库（无 IR 副作用）                                             | `build_plan(stmt) -> VectorProgram`      |
| P3     | 纯函数库（plan → TIR）                                           | `emit_plan(plan) -> Stmt`                |
| driver | Python 函数，pre-order 就地替换                                   | `vectorize_parallel_nests(stmt)`         |
| 默认流水线  | **未接入**；`T.Parallel` 仍由 C++ `AscendSimdVFLowerParallel` 降级 | `tilelang/ascend/pipeline.py:121`        |
| P1     | 已是 C++（只读分析 + 适配层）                                         | `src/ascend/transform/vsimd_*.{h,cc}`    |

结论：C++ 化的目标不是「新建能力」，而是**同一套语义换实现**，因此对拍（plan 文本 / IR / message）就是唯一验收口径。

***

## 2. 输入输出契约

### 2.1 P2 输入契约

- 类型：单个 `tirx::Stmt`。
- 形状：形如 `For(kParallel)…` 的 1..N 层链；允许在并行链之后是一个 `SeqStmt`
  （多 store、或**恰好一个**干净的 `For(kSerial)` + 兄弟语句）。
- 前置条件：
  - 所有循环 `min` 必须可证为 0（否则 R5 拒绝）；
  - 允许 `Bind` 别名（仅非 `Call` 的值绑定会被内联）；
  - 不允许 `local.var` 写、任意控制流、`tl.simd.*`、原子调用、GM 操作数。
- 调用方：现为 `AscendBuildP2PlanInputs`（P1 适配层），未来为 driver pass。

### 2.2 P2 输出契约（字段即 FFI Object 字段，D2）

| 对象              | 字段（顺序即 `def_ro` 注册顺序）                                                                                                  |
| --------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `VectorProgram` | `axes: Axis[]`、`segments: VectorPlan[]`、`diagnostics: string[]`                                                         |
| `VectorPlan`    | `axes`、`vector_axis`、`lanes: int`、`chunks: int?`、`tail: int?`、`stores: StorePlan[]`、`pre_stores`、`post_stores`、`reduce: ReduceInfo?`、`guard_boundaries: int[]`、`diagnostics: string[]` |
| `StorePlan`     | `spec: StoreSpec`、`access: AccessInfo`、`value: ValueNode`、`mask: ValueNode?`、`kind: AccessClass`、`index: ValueNode?`、`reduce: ReduceInfo?`、`promote: ValueNode?`、`mask_free_until: int`、`mask_free_from: int?` |
| `StoreSpec`     | `buffer: Buffer`、`indices: Expr[]`、`value: Expr`、`mask: Expr?`                                                           |
| `ReduceInfo`    | `axis: Axis`、`op: string`、`recipe: string`（默认 `"vcadd"`）、`varying: ValueNode?`、`accumulate: bool`                     |
| `AccessInfo`    | `buffer`、`indices`、`offset`、`is_store: bool`、`per_axis: Map<int, (class,stride)>`、`indirect: bool`、`in_global: bool`、`reason: string`；方法 `axis_class(axis) -> (AccessClass, int?)` |
| `ValueNode`     | `recipe: string`、`dtype: string`、`args: ValueNode[]`、`access: AccessInfo?`、`scalar: Expr?`、`detail: string`、`predicate: ValueNode?`、`predicate_free_until: int`、`predicate_free_from: int?`、`is_uniform: bool` |
| `Axis`          | `var: Var`、`extent: Expr`、`index: int`、`kind: ForKind`、`is_parallel: bool`、`name: string`                              |

约束（测试直接断言，必须逐字满足）：

- Python 侧 `program.segments[0].stores[0].value.recipe` 等 **40+ 处**字段访问零改写 → 字段名、类型、可读性完全一致；
- `Axis.name` / `is_parallel` 是**派生只读**字段（Python 是 `@property`），C++ 侧同样只能读；
- `AccessInfo.axis_class(axis)` 是方法，FFI 侧注册为同名可调用成员；
- `chunks` / `tail` 可为 `None`（符号 extent），FFI 用 `std::optional<int>`（`Optional<IntImm>`）。

### 2.3 P3 输入/输出契约

- 输入：`VectorPlan` 或 `VectorProgram`（**只读**，`const&`；Python 版也不修改）；
- 输出：单个 `tirx::Stmt`；
- 不变量：
  - P2 成功 ⇒ P3 必成功（bool store、条件归约、非均匀 pre/post 等已在 build 期拦截）；
  - `emit` 不引入 plan 之外的指令，只做指令选择；
  - SSA 命名必须一致：`v{prefix}{counter}`、`chunk` / `chunk_s{i}`、`mask{prefix}{bits}`、`lane0{prefix}{bits}`、`ramp_{stride}`、`acc{prefix}{counter}`、`{prefix}ck{counter}`。

### 2.4 `render_plan` 契约（D7）

- `render_plan(VectorPlan)` 与「1 段 program」逐字同形：不打印 `program:` 头；
- 多段 program 打印 `program: axes=[...] segments=N` + 逐段 `-- segment i --`；
- 值树用 `- {recipe}: {dtype} ({detail})` 两空格缩进递归。

***

## 3. 依赖接口清单

| 依赖                                                             | Python 现状                                                                      | C++ 可用性                                              | 风险                                   |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------ |
| `arith.Analyzer`（`simplify` / `const_int_bound` / `can_prove`） | P2 每处新建/复用（`build_plan` 一个、`_access_provably_in_bounds` 每次新建）                    | `tvm::arith::Analyzer` 同名方法                          | 化简结果可能不同 → 符号用例必须对拍                  |
| Z3                                                             | Python analyzer 默认不带 z3 配置                                                     | `analyzer.z3_prover`（P1 已用）                          | P2 **不启用** z3，保持与 Python 一致          |
| `tirx` 节点                                                      | `For/BufferStore/BufferLoad/Call/Bind/DeclBuffer/AllocBuffer/AttrStmt/SeqStmt/IfThenElse` | 全可用                                                  | —                                    |
| `T.simd.*` 算子                                                  | `tirx.call_intrin(Op)`                                                         | `Op::Get("tl.simd.vld")` 已验证可用                       | 参数顺序/默认值需逐条对齐 `language/simd.py`     |
| dtype 工具                                                       | `_vreg_lanes/_vec_dtype/_mask_bits`                                            | 需移植（`primitives.h`）                                   | 表必须 D4 对拍                           |
| `expr_deep_equal`                                              | `tirx.analysis.expr_deep_equal`                                                | `tvm::tirx::ExprDeepEqual`（待确认符号名）                    | `_same_indices` 语义                    |
| `stmt_functor.substitute` / `post_order_visit` / `ir_transform`  | Python 版                                                                       | `tirx::Substitute` / `PostOrderVisit` / `IRTransform` | 遍历顺序必须一致（诊断顺序即契约）                    |
| 别名/相等                                                          | `same_as` + `_same_buffer/_same_indices`                                        | `same_as` + 结构比较                                       | 逐条对齐                                 |

`language/simd.py` 关键事实（C++ 必须复刻）：

- `_vreg_lanes(elem) = 2048 / elem_bits`，`bool` 特例 256；
- `_vec_dtype(elem) = f"{elem}x{lanes}"`；
- `_mask_bits(elem)`：由元素位宽派生，`bool` 抛 `ValueError`（build 期必须转 `Unsupported`）；
- 所有 `T.simd.*` 的 `mode` 参数默认 `"MODE_ZEROING"`，`dist` 默认 `"NORM"` / `"NORM_B32"`；
- `access_ptr` 包装由 `simd.py` 在 Python 侧完成，C++ 侧必须直接构造 `BufferLoad`（`simd.py`
  内部对 `BufferLoad` 的处理是包 `access_ptr`，见各算子实现）。

***

## 4. 调度关系

```
现状：
  AscendSimdVFLowerParallel (pipeline.py:121)   ← T.Parallel 的唯一下游消费者
  ↓ AscendLowerTileOp → DecoupleTypeCast → LegalizeVectorizedLoop → LowerAccessPtr → FlattenBuffer …

目标（阶段五之后）：
  mod = NormalizeBufferVersion()
  mod = AscendPolyhedralVectorize()   ← 新增；失败即原样保留
  mod = AscendSimdVFLowerParallel()   ← 兜底
  ↓ 其余不变
```

约束：

1. 新 pass 位于 `NormalizeBufferVersion` 之后、`AscendSimdVFLowerParallel` 之前；
2. **放在调度之后**（与 legacy 同位置），本轮不改 `AutoSchedule`；
3. 产物必须能通过 `DecoupleTypeCast / LegalizeVectorizedLoop / LowerAccessPtr / FlattenBuffer`（阶段五验证）。

***

## 5. 测试覆盖与耦合面

| 测试集       | 数量    | 位置                                                            | 对 C++ 化的约束                                                                                       |
| --------- | ----- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| P2 单测     | 113   | `testing/ascend/transform/test_ascend_polyhedral_vectorize.py`        | 34 处 `.recipe`、13 处 `segments[...]`、8 处 `.spec./.access.`、85 处 `emit_plan`、4 处 `render_plan`、20 处 `pytest.raises(Unsupported, match=…)` |
| P1→P2 全链路 | 33    | `testing/ascend/transform/test_ascend_vsimd_p2_input.py`      | `render_plan` parity + `structural_equal(emit_plan)`                                             |
| P1 回归     | 55    | `test_ascend_vsimd_{memory_legality,capture_scope,region}.py` | 报告文本零回归（P1 不改）                                                                                 |
| Ascend 全量 | 107 文件 | `testing/ascend/**`                                           | 阶段六回归范围                                                                                        |

**测试零改写**（CI 可用文件 hash 校验）。因此 D2/D3/D7 是硬约束。

***

## 6. 性能基线（同机复测，2026-10-27）

命令：`python tmp/bench_p2_p3.py`（`bench` 默认 n=200，P1+P2+P3 组合 n=50）

| 场景                                      | 本机实测         | 计划文档记录       |
| --------------------------------------- | ------------ | ------------ |
| P1 交接（`AscendBuildP2PlanInputs`）\[8,64] | 51.1 µs      | 47.4 µs      |
| P1 交接 \[64,4096]                        | 48.8 µs      | 52.1 µs      |
| P2 `build_plan` \[8,64]                 | 168.0 µs     | 170.1 µs     |
| P2 `build_plan` \[64,4096]              | 170.2 µs     | 181.7 µs     |
| P3 `emit_plan` \[8,64]                  | 368.9 µs     | 362.0 µs     |
| P3 `emit_plan` \[64,4096]               | 415.6 µs     | 414.4 µs     |
| **P1+P2+P3 单 nest \[64,4096]**           | **750.2 µs** | **767.3 µs** |

门槛（A8）：单 nest `plan+emit` ≤ 150 µs（即 ≤ 基线 1/5）。

***

## 7. 功能分支清单（C++ 侧必须逐条实现）

1. 轴打分 / 排序（`stable_sort` + `axis.index` tie-break）/ 逐个回退；
2. 多段（多个并列干净 serial）→ `_try_build_program`；
3. `pre`/`post` 两种发射模式（沿 serial 轴在循环外 / 沿并行轴在 chunk 内）；
4. 展平通道（`_flatten_axes` / `_try_flatten`，2 的幂 extent 链）；
5. 归约：短展开（`chunks <= 4`，多链树形合并）+ 长循环（`local.var` 累加器）；
6. 累加器提升（`accumulator_candidate`，plan 期单一决策）；
7. guard：`predicate_free_until/from`、`mask_free_until/from`、`guard_boundaries` 切分；
8. 索引向量通道：STRIDED（含负步长）/ PERMUTE / PIECEWISE / INDIRECT → `vgather2`/`vscatter`；
9. 广播（UNIFORM → `vdup` / `BRC`）；
10. 标量-向量特化（`vadds/vmuls/vmaxs/vmins/vcmps`）与反向退回；
11. 类型转换（`vcvt`，f16↔f32，掩码取 min bits）；
12. 多 store、`T.view` 别名、符号 extent（运行期尾块掩码）；
13. `FloorDiv/FloorMod` 值算子 strength-reduction（`vshrs` / `vand`）；
14. LICM（`_split_loop_invariant`）、写后读转发（`_forward`）、段间直通（`_carry`）、累加器 `_promoted`。

口径要求：

- diagnostics 的**顺序与措辞**逐字一致（测试按前缀断言，`diagnostics` 按列表顺序比较）；
- 轴排序稳定：`std::stable_sort`，比较键相同则保持输入顺序；
- 字典遍历一律用有序容器（`std::map` / `std::vector`），禁止 `unordered_map` 影响输出顺序。

***

## 8. 验收标准（A1–A10）

| #   | 验收项                            | 判定方式                                        |
| --- | ------------------------------ | ------------------------------------------- |
| A1  | 113 个 P2 用例 100% 通过            | 测试文件零修改                                     |
| A2  | 33 个全链路用例 100% 通过              | 同上（`impl=cpp`）                              |
| A3  | 55 个 P1 用例 100% 通过             | 同上                                          |
| A4  | 40 个拒绝点 message 逐字一致           | `pytest.raises(Unsupported, match=…)` + `p2_unsupported_catalog.md` 逐条核对 |
| A5  | `render_plan` 文本逐字一致           | 113 用例的 Python 输出冻结为 golden 快照             |
| A6  | `emit_plan` 逐节点一致              | `structural_equal(..., map_free_vars=True)` |
| A7  | 开关关闭时默认流水线零变化                  | 全量 IR `structural_equal` / 文本 diff          |
| A8  | 性能：单 nest `plan+emit` ≤ 150 µs | 同机同脚本，≤ 基线 1/5                             |
| A9  | 无内存安全报告                        | C++ 单测在 ASAN + UBSAN 下运行                     |
| A10 | 无新增编译告警                        | `-Wall -Wextra` 零新增                         |

***

## 9. 已识别的对拍风险与处置

| 风险                       | 处置                                                                        |
| ------------------------ | ------------------------------------------------------------------------- |
| `arith::Analyzer` 化简差异   | 阶段 3.2/3.6 对含符号 extent / `T.min` / `FloorDiv` 的用例逐一对比 `render_plan`；不一致以 Python 输出为准调整 C++ |
| Z3 超时差异                 | P2/P3 不启用 z3（与 Python 一致）                                                 |
| Object 注册样板量             | 阶段二先做 `Axis` + `VectorProgram` 最小对象，验证 Python 字段可读后再批量                       |
| `Unsupported` 文本漂移      | 阶段一冻结 `p2_unsupported_catalog.md`（40 条），测试即验收                               |
| 顺序不稳定                    | `std::stable_sort` + 显式 tie-break；有序容器                                       |
| `_value_key` 用 `id()` 做缓存 | C++ 侧按结构键实现（字符串化 scalar / access），与 Python 结构键等价；`id()` 只用于 Python 内部的 `_promoted`/`_foldable` 语义 → C++ 用节点指针身份 |
| 测试被「改写式通过」               | Gate 1 明确「测试零修改」，CI 校验 hash                                             |
