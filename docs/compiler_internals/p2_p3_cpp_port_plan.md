# P2/P3 C++ 化改造分步执行计划

> 目标：把 `tilelang/ascend/analysis/polyhedral/`（P2 规划 + P3 发射，共 4637 行 Python）逐模块移植为 C++，并以 PrimFuncPass 的形式无缝接入 Ascend 编译流水线，做到**行为零差异、测试零改写、性能显著优于现状**。
>
> 依据代码：
>
> - P2：`tilelang/ascend/analysis/polyhedral/{access,body,value,model,plan}.py`
> - P3：`tilelang/ascend/analysis/polyhedral/{emit,render}.py`
> - P1（已是 C++，直接复用）：`src/ascend/transform/{vsimd_memory_legality,vsimd_capture_scope,vsimd_region}`
> - P1→P2 适配：`src/ascend/transform/polyhedral/vsimd_p2_input.{h,cc}`
> - 流水线：`tilelang/ascend/pipeline.py:121`（`AscendSimdVFLowerParallel`）
> - 测试：`testing/ascend/transform/test_ascend_polyhedral_vectorize.py`（113）、`testing/ascend/transform/test_ascend_vsimd_p2_input.py`（33）、P1 三文件（55）

***

## 0. 关键结论（先对齐，再动手）

### 0.1 现状澄清

P2/P3 目前**不是 pass**，而是：

| 组件     | 现在是什么                                                      | 入口                                  |
| ------ | ---------------------------------------------------------- | ----------------------------------- |
| P2     | 纯函数库（无 IR 副作用）                                             | `build_plan(stmt) -> VectorProgram` |
| P3     | 纯函数库（plan → TIR）                                           | `emit_plan(plan) -> Stmt`           |
| driver | Python 函数，逐 nest 就地替换                                      | `vectorize_parallel_nests(stmt)`    |
| 默认流水线  | **未接入**，`T.Parallel` 仍由 C++ `AscendSimdVFLowerParallel` 降级 | `pipeline.py:121`                   |

### 0.2 架构决策

| #      | 决策                                                                                                                                                     | 理由                                                                                                                |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **D1** | P2/P3 落成 **C++ 库 + FFI 调试入口**，另加**一个** driver pass `AscendPolyhedralVectorize`                                                                         | plan 不是 IR；若拆成两个 pass 必须把 plan 塞进 IR（AttrStmt/annotation），引入序列化与失效风险。单 pass 与现有 `vectorize_parallel_nests` 语义一一对应 |
| **D2** | plan 对象（`VectorProgram/VectorPlan/StorePlan/ValueNode/AccessInfo/StoreSpec/Axis`）直接注册为 **TVM FFI Object**（`def_ro` 字段），不做 Map 序列化                      | P3 直接消费同一份 C++ 对象；Python 侧 `program.segments[0].stores[0].value.recipe` 等 40+ 处断言零改写；无往返拷贝                        |
| **D3** | 拒绝统一为 C++ `polyhedral::Unsupported`（带 `message` + `diagnostics`），FFI 层转成 `Map{ok:false,message,diagnostics}`，Python wrapper 再 `raise Unsupported(...)` | 现有测试用 `pytest.raises(Unsupported, match=...)` 与 `err.diagnostics`，message 必须逐字兼容                                  |
| **D4** | 硬件常量表（`lanes/mask bits/MODE/recipe 表/归约表`）集中到 `primitives.h`，并提供 FFI 导出做表对拍                                                                            | 40 个拒绝点 + 大量特化都依赖这些表，表错=行为错                                                                                       |
| **D5** | 与 legacy 并存，`PassContext` 开关 `tl.ascend.polyhedral.impl = legacy \| python \| cpp`，默认 `legacy`                                                         | 灰度与回滚需要三态；对拍需要同一 IR 跑三份实现                                                                                         |
| **D6** | 数学三件套（`linear_element_offset / finite_difference / classify_delta`）也 C++ 化并FFI 导出                                                                      | 它们是 P2 的数学核心，Python 保留实现会形成双份真理                                                                                   |
| **D7** | `render_plan` 一并 C++ 化（FFI 返回字符串）                                                                                                                      | 现有测试用 `render_plan(original) == render_plan(p1)` 做 parity（4 处），且它是 Phase 3/4 的对拍 oracle                           |

### 0.3 非目标

- 不扩展 P2/P3 能力（不新增形态、不新指令、不放开归约/控制流）；
- 不改 P1（只调用）；
- 不改 `T.simd.*` 语义与 codegen；
- 不在本轮删除 Python 实现（灰度完成后再删，见阶段八）。

***

## 1. 阶段一：现状梳理与需求对齐

### 1.1 任务 1.1 — 模块与规模清点（已完成初稿，需固化成文档）

| 模块              | 行数       | def/class | 职责                                                    | C++ 目标                     |
| --------------- | -------- | --------- | ----------------------------------------------------- | -------------------------- |
| `primitives.py` | 255      | 13 / 2    | recipe 表、归约表、硬件常量、`Unsupported`、小工具                   | `polyhedral/primitives.h`  |
| `access.py`     | 301      | 11 / 3    | `Axis`/`AccessClass`/`AccessInfo`、线性化、有限差分、分类         | `polyhedral/access.{h,cc}` |
| `body.py`       | 151      | 5 / 1     | `normalize_body`、`StoreSpec`、if→select、别名折叠           | `polyhedral/body.{h,cc}`   |
| `value.py`      | 697      | 29 / 2    | `ValueNode` DAG、`build_value`、装载守卫、索引向量               | `polyhedral/value.{h,cc}`  |
| `model.py`      | 186      | 2 / 4     | `StorePlan/ReduceInfo/VectorPlan/VectorProgram`、累加器判据 | `polyhedral/model.{h,cc}`  |
| `plan.py`       | 1352     | 39 / 1    | `build_plan` 全流程、多段、展平、归约、轴回退                         | `polyhedral/plan.{h,cc}`   |
| `emit.py`       | 1491     | 23 / 2    | strip-mine、掩码、`T.simd.*`、归约两条策略                       | `polyhedral/emit.{h,cc}`   |
| `render.py`     | 55       | 3 / 0     | 计划文本渲染（对拍 oracle）                                     | `polyhedral/render.{h,cc}` |
| `driver.py`     | 35       | 1 / 0     | pre-order 就地替换                                        | 并入新 pass                   |
| **合计**          | **4637** | <br />    | <br />                                                | <br />                     |

### 1.2 任务 1.2 — 输入输出契约固化

必须写成 `p2_p3_cpp_port_survey.md` 的 §契约，逐字段签字确认：

**P2 输入契约**

- 类型：单个 `tirx::Stmt`，形如 `For(kParallel)…` 的 1..N 层链；
- 前置条件：`min` 可证为 0（否则 R5 拒绝）；body 之后允许 `SeqStmt`（多 store / 唯一干净 serial）；允许 `Bind` 别名；不允许 `local.var`、控制流、`tl.simd.*`；
- 调用方：现为 `AscendBuildP2PlanInputs`（P1 适配层），未来为 driver pass。

**P2 输出契约**（字段即 FFI Object 字段）

| 对象              | 字段                                                                                                     |
| --------------- | ------------------------------------------------------------------------------------------------------ |
| `VectorProgram` | `axes: Axis[]`、`segments: VectorPlan[]`、`diagnostics: string[]`                                        |
| `VectorPlan`    | `axes/vector_axis/lanes/chunks/tail/stores/pre_stores/post_stores/reduce/guard_boundaries/diagnostics` |
| `StorePlan`     | `spec/access/value/mask/kind/index/reduce/promote/mask_free_until/mask_free_from`                      |
| `StoreSpec`     | `buffer/indices/value/mask`                                                                            |
| `AccessInfo`    | `offset/axis_class(...)/map_class/injective/...`                                                       |
| `ValueNode`     | `recipe/dtype/args/access/scalar/detail/predicate/predicate_free_until/predicate_free_from/is_uniform` |
| `Axis`          | `var/extent/index/is_parallel/name`                                                                    |

**P3 输入/输出契约**

- 输入：`VectorPlan` 或 `VectorProgram`（**只读**，禁止修改）；输出：单个 `tirx::Stmt`；
- 不变量：P2 成功 ⇒ P3 必成功（bool store 等已在 build 期拦截）；`emit` 不引入 plan 之外的指令。

### 1.3 任务 1.3 — 依赖上下文接口清单

| 依赖                                                             | Python 现状                                                                      | C++ 可用性                                              | 风险                      |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------- | ----------------------- |
| `arith.Analyzer`（`simplify` / `const_int_bound` / `can_prove`） | P2 每处新建/复用                                                                     | `tvm::arith::Analyzer` 同名方法                          | 化简结果可能不同 → 符号用例必须对拍     |
| Z3                                                             | Python analyzer 默认不带 z3 配置                                                     | `analyzer.z3_prover`（P1 已用，`SetRLimit/SetTimeoutMs`） | 超时策略需与 Python 对齐，否则判定漂移 |
| `tirx` 节点                                                      | `For/BufferStore/BufferLoad/Call/Bind/DeclBuffer/AllocBuffer/AttrStmt/SeqStmt` | 全可用                                                  | —                       |
| `T.simd.*` 算子                                                  | `tirx.call_intrin(Op)`                                                         | `Op::Get("tl.simd.vld")` 已验证可用                       | dtype/掩码宽度需自建 helper    |
| dtype 工具                                                       | `_vec_dtype/_mask_bits/_elem_dtype`                                            | 需移植（`DataType` + 字符串后缀）                              | 表必须 D4 对拍               |
| FFI                                                            | `ffi::Map/Array/String/Any`                                                    | 同                                                    | Object 注册样板量            |
| 别名/相等                                                          | `same_as` + `_same_buffer/_same_indices`                                       | `same_as` + `ffi::StructuralEqual`                   | 语义逐条对齐                  |

### 1.4 任务 1.4 — 上下游调度关系图（现状 → 目标）

```
现状：
  AscendSimdVFLowerParallel (pipeline.py:121)   ← T.Parallel 的唯一下游消费者
  ↓ AscendLowerTileOp → DecoupleTypeCast → LegalizeVectorizedLoop → LowerAccessPtr → FlattenBuffer …

目标（阶段五之后）：
  P1 诊断（只读）/ 可选
  AscendPolyhedralVectorize   ← 新增 C++ pass（plan + emit + 就地替换）
       ├─ 成功：把 Parallel 链换成 T.simd.*（与 legacy 产物同级）
       └─ 失败：原样保留，后续仍由 AscendSimdVFLowerParallel 兜底
  ↓ 其余 pass 不变
```

必须确认的调度约束：

1. 新 pass 必须位于 `NormalizeBufferVersion` 之后、`AscendSimdVFLowerParallel` 之前（buffer version 已物理化、`T.simd.*` 尚未被消费）；
2. 与 `AutoSchedule` 的关系：P3 产物含 `Bind`/`AllocBuffer(local.var)`，若要在调度前插入需验证 `MaterializeScheduleUnits` 能否穿过；**本轮默认放在调度之后**（与 legacy 同位置），避免改调度器；
3. 产物必须能通过 `DecoupleTypeCast / LegalizeVectorizedLoop / LowerAccessPtr / FlattenBuffer`（阶段五用 IR diff 验证）。

### 1.5 任务 1.5 — 测试覆盖与耦合面盘点

| 测试集       | 数量     | 位置                                                            | 对 C++ 化的约束                                                                                                                                               |
| --------- | ------ | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P2 单测     | 113    | `testing/ascend/transform/test_ascend_polyhedral_vectorize.py`        | 34 处 `.recipe`、13 处 `segments[...]`、8 处 `.spec./.access.`、85 处 `emit_plan`、4 处 `render_plan`、20 处 `pytest.raises(Unsupported, match=…)`、5 个数学核心单测（8 处调用） |
| P1→P2 全链路 | 33     | `testing/ascend/transform/test_ascend_vsimd_p2_input.py`      | `render_plan` parity + `structural_equal(emit_plan)`                                                                                                     |
| P1 回归     | 55     | `test_ascend_vsimd_{memory_legality,capture_scope,region}.py` | 报告文本零回归（P1 不改）                                                                                                                                           |
| Ascend 全量 | 107 文件 | `testing/ascend/**`                                           | 阶段六回归范围                                                                                                                                                  |

> 结论：**测试不能改写**。因此 D2（FFI Object 字段同名）+ D3（`Unsupported` 文本兼容）+ D7（`render_plan` C++ 化）是硬约束，不是可选优化。

### 1.6 任务 1.6 — 性能基线（已实测，`tmp/bench_p2_p3.py`）

| 场景                                      | 现状耗时         |
| --------------------------------------- | ------------ |
| P1 交接（`AscendBuildP2PlanInputs`）\[8,64] | 47.4 µs      |
| P1 交接 \[64,4096]                        | 52.1 µs      |
| P2 `build_plan` \[8,64]                 | 170.1 µs     |
| P2 `build_plan` \[64,4096]              | 181.7 µs     |
| P3 `emit_plan` \[8,64]                  | 362.0 µs     |
| P3 `emit_plan` \[64,4096]               | 414.4 µs     |
| **P1+P2+P3 单 nest**                     | **767.3 µs** |

要求：基线脚本纳入仓库，阶段六用同一脚本、同一机器复测。

### 1.7 任务 1.7 — 功能分支与异常分支清单

- **拒绝分支 40 个**（`raise Unsupported`）：`plan.py` 21、`value.py` 9、`body.py` 5、`emit.py` 4、`access.py` 1 → 用脚本抽取成 `p2_unsupported_catalog.md`（文件:行号 + message + 触发条件），C++ 实现逐条打勾，阶段六逐条注入用例。
- **功能分支清单**（必须逐条列出并在 C++ 侧标注对应函数）：轴打分/排序/回退、多段（多个并列干净 serial）、`pre/post` 两种发射模式、展平通道、归约（短展开 / 长循环 + `local.var` 累加器）、累加器提升、guard（`mask_free_until/from`、`guard_boundaries`）、索引向量通道（STRIDED/PERMUTE/PIECEWISE/INDIRECT）、广播（UNIFORM）、标量-向量特化、类型转换（f16↔f32）、多 store、`T.view` 别名、符号 extent。
- **口径要求**：diagnostics 的**顺序与措辞**（测试按前缀断言）、轴排序的**稳定性**（Python `sort` 稳定，C++ 必须 `stable_sort` + 同序 tie-break）。

### 1.8 任务 1.8 — 验收标准（Gate 1 输出，后续阶段共同遵守）

| #   | 验收项                            | 判定方式                                           |
| --- | ------------------------------ | ---------------------------------------------- |
| A1  | 113 个 P2 用例 100% 通过            | 测试文件零修改                                        |
| A2  | 33 个全链路用例 100% 通过              | 同上（`impl=cpp`）                                 |
| A3  | 55 个 P1 用例 100% 通过             | 同上                                             |
| A4  | 40 个拒绝点 message 逐字一致           | `pytest.raises(Unsupported, match=…)` + 目录逐条核对 |
| A5  | `render_plan` 文本逐字一致           | 阶段一先冻结 113 用例的 Python 输出为 golden 快照            |
| A6  | `emit_plan` 逐节点一致              | `structural_equal(..., map_free_vars=True)`    |
| A7  | 开关关闭时默认流水线零变化                  | 全量 IR `structural_equal` / 文本 diff             |
| A8  | 性能：单 nest `plan+emit` ≤ 150 µs | 同机同脚本，≤ 基线 1/5                                 |
| A9  | 无内存安全报告                        | C++ 单测在 ASAN + UBSAN 下运行                       |
| A10 | 无新增编译告警                        | `-Wall -Wextra` 零新增（以现有文件告警集为基线）               |

**阶段一交付物**：`docs/compiler_internals/p2_p3_cpp_port_survey.md`（现状 + 契约 + 依赖 + 调度 + 覆盖 + 基线 + 验收）、`p2_unsupported_catalog.md`、`tmp/bench_p2_p3.py`。

**退出条件**：上述两份文档评审通过；A1–A10 逐条确认口径无歧义。

***

## 2. 阶段二：工程环境与基础适配

### 2.1 目录与构建

```
src/ascend/transform/polyhedral/
├── primitives.h              # recipe/归约/硬件常量表 + Unsupported
├── access.h / access.cc      # Axis / AccessClass / 线性化 / 有限差分
├── body.h   / body.cc        # StoreSpec / normalize_body
├── value.h  / value.cc       # ValueNode DAG / build_value
├── model.h  / model.cc       # StorePlan/ReduceInfo/VectorPlan/VectorProgram + FFI 注册
├── plan.h   / plan.cc        # build_plan 全流程
├── emit.h   / emit.cc        # emit_plan
├── render.h / render.cc      # render_plan
└── ffi_polyhedral.cc         # FFI 入口 + 错误桥 + driver pass
```

**必须修改的一处构建脚本**：`src/ascend/CMakeLists.txt:101-109` 的 glob 不递归，需要新增
`src/ascend/transform/polyhedral/*.cc`（P1 那次新增文件之所以免改，是因为在同一层目录）。

### 2.2 规范对齐

| 项       | 要求                                                                                                                                            |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 格式      | 仓库 `.clang-format`（pre-commit `clang-format v23.1.2`），提交前必须过                                                                                  |
| 命名      | TVM 风格：类 `PascalCase`、函数 `PascalCase`（对外）/ `snake_case`（本地 helper 可随 Python 名）、成员 `field_`                                                    |
| 导出      | 对外 API `TVM_DLL`；内部实现放匿名 namespace                                                                                                            |
| 错误      | 拒绝用 `polyhedral::Unsupported`（异常），**不用** `LOG(FATAL)`；编译器内部不变量用 `TVM_FFI_ICHECK`                                                              |
| pass 注册 | `CreatePrimFuncPass(pass_func, 0, "tl.AscendPolyhedralVectorize", {})` + `refl::GlobalDef().def("tl.transform.AscendPolyhedralVectorize", …)` |
| 内存      | 全部 `ffi::ObjectRef` / 值语义，禁止裸 `new`；容器传 `const &`                                                                                             |

### 2.3 基础适配件（先于业务逻辑落地）

1. **FFI 错误桥**：`Unsupported` → `Map{"ok": false, "message": ..., "diagnostics": [...]}`；
2. **Object 注册样板**：先注册 `Axis` + `VectorProgram` 两个最小对象，Python 侧验证 `@register_object` 字段可读；
3. **dtype helper**：`_vec_dtype/_mask_bits/_elem_dtype` 移植 + FFI 导出做表对拍；
4. **常量表 FFI 导出**：`AscendPolyhedralPrimitives()` 返回所有表，Python 与 `primitives.py` 全等比对。

### 2.4 退出条件（Gate 2）

- `./build.sh` 通过、零新增告警；
- `python -c "from tilelang.ascend import transform as at; at.AscendPolyhedralVectorize()"` 可用（空 pass 原样返回 IR）；
- 常量表/ dtype helper 对拍 100% 相等；
- 空 pass 接入 `pipeline.py` 后（默认关闭）全量测试与改动前一致。

***

## 3. 阶段三：P2 的 C++ 实现

**顺序严格自底向上**（依赖方向 `primitives → access → body → value → model → plan`），每完成一层立刻编译 + 对拍，不攒到最后。

| 步骤  | 移植内容                                                                                                                                                                                                                                 | 对拍方式                                            | 退出条件       |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- | ---------- |
| 3.1 | `primitives`：recipe 表、`_REDUCE_PRIMITIVES`、`_MODE`、`_MASK_DTYPE`、`Unsupported`、`_same_buffer/_same_indices/_seq`                                                                                                                     | 表全等 + 小工具单测                                     | 表一致        |
| 3.2 | `access`：`Axis`、`AccessClass`、`linear_element_offset`、`finite_difference`、`classify_delta`、`classify_access`、`axis_class`                                                                                                            | 5 个数学单测 + 逐用例 `offset` 表达式 `structural_equal`   | 数学核心一致     |
| 3.3 | `body`：`StoreSpec`、`normalize_body`（多语句/if→select/单臂谓词）、`_collect_aliases`、`_apply_aliases`                                                                                                                                          | 113 用例中所有含 if / 多语句 / `Bind` 的用例                | 规范化结果一致    |
| 3.4 | `value`：`ValueNode` DAG、`build_value`、`load_node`、`index_value`、`_with_guard`、`_guard_free_bounds`、`is_uniform`                                                                                                                      | 值 DAG 的 `_recipes()` 树逐一一致                      | 树形一致       |
| 3.5 | `model`：`StorePlan/ReduceInfo/VectorPlan/VectorProgram`（FFI Object 注册）、`accumulator_candidate`、`is_private_scope`                                                                                                                    | Python 侧直接读字段（不重建 dataclass）                    | 类型可导入、字段同名 |
| 3.6 | `plan`：`_split_nest`、`_prepare_segment`、`_plan_segment`、`_try_flatten`、`_flatten_axes`、`rank_vector_axes`、`_axis_score`、`_build_plan_for_axis`、`_classify_reduce`、`_guard_cuts`、`_outer_axes`、`build_plan`、多段路径 `_try_build_program` | 113 用例：`render_plan` 全等 + 20 个 `Unsupported` 正则 | 全绿         |

### 3.1 代码审查重点（每层结束都做）

- 每个 Python `raise Unsupported` 在 C++ 有且仅有一条对应分支（对照 `p2_unsupported_catalog.md` 打勾）；
- 所有 `if` 级功能分支（1.7 清单）在 C++ 有对应实现，无“先简化后补”；
- 不得引入 Python 侧没有的新启发式（例如“轴数量超过 N 就拒绝”）。

### 3.2 单模块编译校验

- `cmake --build --target tilelang_objs` 只编译本次新增文件（可用 `-j1` 观察告警）；
- 每个 `.cc` 提交前单独跑 `clang-format --dry-run` + `-Wall -Wextra -Werror`（仅对新增文件开启 `-Werror`）；
- ASAN/UBSAN 单测：每个模块一个最小 gtest（或 pytest 走 FFI），覆盖空输入、单层、多层、拒绝型。

### 3.3 退出条件（Gate 3）

113 用例在 `impl=cpp` 下全绿（P3 仍用 Python 时如何跑：`build_plan` 走 C++，`emit_plan` 走 Python，验证 P2 单侧兼容）。

***

## 4. 阶段四：P3 的 C++ 实现

### 4.1 顺序

1. `render`（先做，给所有对拍提供可读 oracle；文本逐字兼容）；
2. `emit`：`_Emitter`（SSA 命名、`let` 收集、写后读转发）、`_strip_mine`、`_emit_elementwise_chunks`、`_emit_parallel_axis_parts`、`_emit_reduce`（短展开 + 长循环两条策略）、`_emit_outer_store`、`_emit_segment_stmts`、`_emit_program`、`emit_plan`。

### 4.2 接口兼容是硬约束

| 项          | 要求                                                                                             |
| ---------- | ---------------------------------------------------------------------------------------------- |
| 输入类型       | C++ `VectorProgram`/`VectorPlan` 对象（与 P2 同一份，**不经 FFI 序列化**）                                   |
| 只读性        | emit 不修改 plan（Python 版也不修改；C++ 用 `const&` 强制）                                                  |
| 输出         | 单 `Stmt`；节点类型/顺序与 Python 版逐节点一致                                                                |
| SSA 命名     | `vs0_0 / vs0_tail_1 / mask_32 …` 必须一致（`render_plan` 之外的 IR 文本对比会用到）                            |
| `T.simd.*` | `tirx::Call(Op::Get("tl.simd.vld"), args, dtype)`；`access_ptr` 参数构造必须与 `language/simd.py` 完全一致 |
| 累加器        | `AllocBuffer(local.var)` + 初值 + `Bind` 结构一致                                                    |
| 常量         | `_REDUCE_UNROLL_LIMIT`、`MODE_ZEROING`、`PAT_ALL/PAT_VL1`、mask bits 全部来自 `primitives.h`          |

### 4.3 独立编译校验

- 只编译 P3 单元（不依赖 `plan.cc`）时，用一个手写 plan 构造器（`test-only`）验证发射；
- 对 113 用例做 `structural_equal(emit_cpp(plan), emit_python(plan))`；
- 逐条核对 `emit.py` 的 4 个 `Unsupported`：`bool` 等已在 P2 拦截，剩 4 条要确认是“build 期已排除的死代码”还是“emit 期真会触发”，后者必须在 C++ 保留。

### 4.4 退出条件（Gate 4）

- `emit_plan` 全量 `structural_equal` 一致；
- `render_plan` 文本逐字一致（A5）；
- 无 `LOG(FATAL)` 路径残留。

***

## 5. 阶段五：全链路集成联调

### 5.1 新 pass 设计

```
AscendPolyhedralVectorize(PrimFunc) -> PrimFunc
  for each SIMD_VF nest (P1 已产出 VSimdP2Input):
      if status != LIFTED:               continue           # legacy 接管
      try:  program = BuildPlan(stmt)                        # P2 (C++)
            new     = EmitPlan(program)                      # P3 (C++)
      except Unsupported:                continue           # Branch B → legacy
      replace(nest, new)                                     # pre-order，整段替换
```

- 遍历顺序：**pre-order**（与 `driver.py` 一致；自底向上会让外层 Parallel 失配）；
- 失败语义：**只跳过该 nest，不改 IR**，由 `AscendSimdVFLowerParallel` 兜底；
- 开关：`PassContext` 配置 `tl.ascend.polyhedral.impl`（`legacy`/`python`/`cpp`），默认 `legacy`。

### 5.2 集成验证清单

| 项    | 验证方式                                                                                              |
| ---- | ------------------------------------------------------------------------------------------------- |
| 调度位置 | 新 pass 插在 `NormalizeBufferVersion` 与 `AscendSimdVFLowerParallel` 之间；断言 `T.simd.*` 未被提前消费          |
| 数据传递 | nest 被替换后，`outer_vars`（thread/outer\_loop var）、`Bind`、`DeclBuffer`、`layout` 注解仍可见；用自由变量集对比        |
| 作用域  | 替换后 buffer 的 `data` var 仍被正确声明/捕获；`T.view` 别名不悬空                                                  |
| 下游兼容 | 产物单独跑 `DecoupleTypeCast → LegalizeVectorizedLoop → LowerAccessPtr → Simplify → FlattenBuffer` 不报错 |
| 等价性  | 同一 kernel 三态（`legacy` / `python` / `cpp`）产物 `structural_equal`，或记录**允许的差异**（如 SSA 名）              |
| 混合场景 | 一个 PrimFunc 内多个 VF，部分 LIFTED 部分 OPAQUE：只有 LIFTED 的被替换                                             |

### 5.3 退出条件（Gate 5）

- 三态产物在 33 个全链路用例 + 若干真实 kernel（`examples/ascend/**`）上等价；
- 新 pass 默认关闭时，全量 107 个 Ascend 测试文件行为不变。

***

## 6. 阶段六：全量测试验证

### 6.1 分层测试矩阵

| 层  | 内容                                     | 命令                                                                                      |
| -- | -------------------------------------- | --------------------------------------------------------------------------------------- |
| L0 | C++ 单测（ASAN/UBSAN）：数学核心、规范化、值 DAG、发射骨架 | `ctest -R polyhedral` 或等价                                                               |
| L1 | P2 单测 113（`impl=cpp`）                  | `python tmp/run_poly_test.py testing/ascend/transform/test_ascend_polyhedral_vectorize.py -q`   |
| L2 | 全链路 33（`impl=cpp`）                     | `python tmp/run_poly_test.py testing/ascend/transform/test_ascend_vsimd_p2_input.py -q` |
| L3 | P1 回归 55                               | 同上三文件                                                                                   |
| L4 | Ascend 全量 107 文件                       | `python tmp/run_poly_test.py testing/ascend -q`                                         |
| L5 | 端到端：kernel → `.o`（可选，113/950 板）        | 复用 `build_DSA.sh` 流程                                                                    |

### 6.2 一致性判定（与 A1–A6 对齐）

1. **计划层**：`render_plan` 文本逐字（阶段一冻结 golden）；
2. **IR 层**：`structural_equal(..., map_free_vars=True)`；
3. **拒绝层**：`Unsupported.message` 正则匹配 + `diagnostics` 前缀；
4. **诊断顺序**：`diagnostics` 列表逐项比较（不是集合）。

### 6.3 边界与异常注入

- `extent = 1 / 63 / 64 / 65 / 100 / 4096 / 符号 / 0`（0 需明确拒绝路径）；
- dtype：fp32 / fp16 / int32 / bool（拒绝）/ 混位宽（拒绝）；
- 访存形态：连续 / 广播 / 步长 / 逆序 / 分片 / 间接 / true\_random（拒绝）/ stencil（拒绝）；
- 结构：单 store / 多 store / 多段 serial / `pre+post` 三明治 / `T.view` / 多 nest / 外层 `Bind` / `T.assume`；
- 异常注入：40 个拒绝点各至少 1 个用例（来自 `p2_unsupported_catalog.md`），断言**不崩溃、message 稳定**。

### 6.4 性能验证

- 同机同脚本复测 `tmp/bench_p2_p3.py`，记录 P1/P2/P3 三段拆分耗时；
- 门槛：单 nest `plan+emit` ≤ 150 µs（A8）；IR 节点数不增；`Simplify` 调用轮次不增；
- 另加一个批量场景：一个 PrimFunc 内 100 个 nest 的总耗时（验证无 O(n²)）。

### 6.5 退出条件（Gate 6）

L0–L4 全绿、A1–A10 全部满足、性能报告归档。

***

## 7. 阶段七：代码优化与评审

### 7.1 性能优化（有数据才动）

| 热点（预期）                                               | 手段                                                         | 验证                    |
| ---------------------------------------------------- | ---------------------------------------------------------- | --------------------- |
| `arith::Analyzer::simplify` 调用次数（P3 每 lane 表达式都可能化简） | 缓存 `(expr → simplified)`；同一 `Analyzer` 复用；避免重复 `can_prove` | 计数器对比 Python 版调用次数    |
| Expr 复制 / 传值                                         | 全链路 `const ffi::ObjectRef&`、`std::move`、避免 `Array` 反复构造    | ASAN + 计时             |
| diagnostics 字符串（即使成功也拼接）                             | 惰性拼接：只有失败或需要时生成                                            | 成功路径耗时                |
| `ffi::StructuralEqual` 频繁构造                          | 用指针/哈希快速判等（`same_as` 快路径优先）                                | 微基准                   |
| 值 DAG 重复节点                                           | 同构子表达式去重（保持 recipe 树语义不变）                                  | `structural_equal` 回归 |

### 7.2 埋点

`PolyhedralStats`（每次 pass 运行汇总，`LOG(INFO)` 或 PassContext instrument 输出）：

- `nests_total / lifted / planned / emitted / fallback_legacy`
- `unsupported_histogram`（按 message 首段归类）
- `axis_attempts`、`flatten_hits`、`multi_segment_hits`、`reduce_hits`
- `build_us / emit_us`（分段计时）

### 7.3 评审清单

- [ ] 40 个拒绝分支逐条对照，无遗漏/无新增；
- [ ] 诊断文本与顺序一致（诊断也是契约的一部分）；
- [ ] 无 Python 语义漂移（稳定排序、字典顺序、集合去重顺序）；
- [ ] 无 `LOG(FATAL)` 替代 `Unsupported`；
- [ ] 所有 `TVM_FFI_ICHECK` 只用于“不可能发生”的内部不变量，并有注释说明依据；
- [ ] 注释解释**为什么**（对齐 Python 的哪条规则），不是复述代码；
- [ ] clang-format / ruff（Python 包装层）/ 无死代码；
- [ ] 对外 API 命名与 P1 风格一致（`BuildP2PlanInputs` / `BuildPlan` / `EmitPlan` / `RenderPlan`）。

### 7.4 退出条件（Gate 7）

评审意见全部关闭；优化前后性能数据归档；无新增告警。

***

## 8. 阶段八：交付与灰度上线

### 8.1 文档

1. **设计文档** `docs/compiler_internals/p2_p3_cpp_design.md`：模块图、数据流、与 P1/legacy 的边界、关键决策（D1–D7）、已知差异；
2. **接口文档**：C++ API（`BuildPlan/EmitPlan/RenderPlan/AsParallelNest/RestoreBindings`）、FFI 表（`tl.transform.AscendPolyhedral*`）、pass 选项（`tl.ascend.polyhedral.impl`）、错误码/diagnostics 约定；
3. **使用指南**：如何开开关、如何看 diagnostics、常见拒绝原因与改法（GM 需 staging、bool 需转 f32、stencil 需改写法…）；
4. **迁移说明**：Python 实现进入 deprecated，删除时间点与替代入口。

### 8.2 灰度策略

| 步骤 | 配置                        | 观察项                  | 回滚          |
| -- | ------------------------- | -------------------- | ----------- |
| G0 | `legacy`（默认）              | 基线稳定性                | —           |
| G1 | `cpp` 仅内部对拍任务，IR diff     | 差异清单、Unsupported 直方图 | 关开关         |
| G2 | `cpp` 灰度集群（按 kernel 白名单）  | 编译成功率、pass 耗时、产物节点数  | 白名单移除       |
| G3 | `cpp` 默认，`legacy` 保留为回滚开关 | 长期稳定性（多轮回归）          | 改回 `legacy` |
| G4 | 删除 Python 实现与双实现开关        | —                    | 依赖 G3 稳定期   |

每步的准入条件：上一步至少一轮完整回归 + 性能报告无回退 + 无新增 Unsupported（相对 legacy 的等价性差异为零）。

### 8.3 退出条件（Gate 8）

设计/接口/使用文档评审通过；G1–G3 各阶段记录归档；G4 有明确决策记录（可延后）。

***

## 9. 风险与应对

| 风险                 | 触发条件                                                                  | 应对                                                                         |
| ------------------ | --------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 化简/证明结果漂移          | C++ `arith::Analyzer` 与 Python 版对同一表达式给出不同 `simplify/const_int_bound` | 阶段 3.2/3.6 对拍所有含符号 extent / `T.min` / `FloorDiv` 的用例；必要时在测试中固定 analyzer 配置 |
| Z3 超时差异            | Python 侧无 z3 配置、C++ 侧有 `SetRLimit/SetTimeoutMs`                       | P2 默认**不启用** z3 证明（与 Python 一致），只在 P1 用；如必须用，加开关并单独对拍                      |
| 对象模型样板量            | 7 个类型 × 字段注册 + Python `@register_object`                              | 先做 `Axis`+`VectorProgram` 样板，确认可行再批量                                       |
| `Unsupported` 文本漂移 | C++ 拼接与 Python f-string 细节不同                                          | 阶段一冻结 message 目录，逐条 alias；测试即验收                                            |
| 顺序不稳定              | `std::sort` 与 Python 稳定排序、`unordered_map` 遍历                          | 统一 `std::stable_sort` + 显式 tie-break；遍历用有序容器                               |
| 与 legacy 产物不一致     | 两套发射对同一 nest 产出不同 IR                                                  | 阶段五三态对拍；差异必须能解释并记录，不允许“静默不同”                                               |
| 调度器不接受             | P3 产物含 `Bind`/`local.var` 被 AutoSchedule 误处理                          | 本轮放在调度之后（与 legacy 同位置）；若前置需单独设计                                            |
| 测试被“改写式通过”         | 为省事修改断言/放宽 match                                                      | Gate 1 明确“测试零修改”，CI 校验测试文件 hash                                            |

***

## 10. 阶段依赖与交付节奏（不含时间估计）

```
P1(调研/契约/基线) ──► P2(工程基座) ──► P3(P2 实现) ──► P4(P3 实现)
                                              │              │
                                              └──────► P5(集成) ◄──┘
                                                          │
                                              P6(全量测试) ──► P7(优化评审) ──► P8(交付灰度)
```

- P2 可并行做“常量表/dtype helper 对拍”；
- P3 每层结束即可对 113 用例做部分对拍，不必等全部完成；
- P4 依赖 P3.5（model）完成，`render` 可提前；
- P5 依赖 P3+P4 全绿；
- P6 依赖 P5 的开关可用；
- P7 依赖 P6 的性能数据；P8 依赖 P7 评审关闭。

**总交付物清单**

1. `src/ascend/transform/polyhedral/**`（P2/P3/driver/FFI）；
2. `src/ascend/CMakeLists.txt` 一行 glob 增补；
3. `tilelang/ascend/analysis/polyhedral/__init__.py` 改为 FFI 薄封装（实现保留至灰度结束）；
4. `tilelang/ascend/transform/__init__.py` 新 pass + FFI 封装；
5. `tilelang/ascend/pipeline.py` 开关化接入；
6. 文档四件套（调研 / 设计 / 接口 / 使用）+ `p2_unsupported_catalog.md`；
7. 测试：C++ 单测、113/33/55 回归报告、107 文件全量报告、性能对比报告。


***

## 11. 实现记录（2026-10-27）

### 11.1 阶段一~四（已完成）

| 阶段 | 交付物                                                                                                        | 验收                                       |
| -- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| 一  | `p2_p3_cpp_port_survey.md`、`p2_unsupported_catalog.md`（40 条）、`tmp/bench_p2_p3.py`                        | 基线复测 750.2 µs                            |
| 二  | `src/ascend/transform/polyhedral/{primitives,access,body,value,model}.{h,cc}` + CMake glob + FFI 错误桥 + 常量表导出 | `-Wall -Wextra` 零新增告警（仅第三方头告警）            |
| 三  | `plan.{h,cc}`、`model.cc` 的累加器判据                                                                          | **113/113**（`impl=cpp`），`dual` 模式 `render_plan` 逐字一致 |
| 四  | `render.{h,cc}`、`emit.{h,cc}`                                                                              | **113/113**（`dual`：Python/C++ 发射 `structural_equal` 一致） |
| 五  | `ffi_polyhedral.cc`（driver pass + 8 个 FFI 入口）、`transform/__init__.py` 的 C++ pass、`pipeline.py` `legacy|cpp` 开关 | 同 kernel 下 `cpp` 与 Python 参考实现产物 `structural_equal` 一致；默认 `legacy` 不插入（灰度期临时 Python 孪生 pass 已在筛选阶段删除） |

### 11.2 测试结果

| 层 | 命令 | 结果 |
| -- | --- | --- |
| L1（113） | `TL_ASCEND_POLYHEDRAL_IMPL=cpp python tmp/run_poly_test.py testing/ascend/transform/test_ascend_polyhedral_vectorize.py -q` | 113 passed |
| L1 对拍 | 同上，`impl=dual`（Python vs C++ plan 文本 + 发射 IR） | 113 passed |
| L2（33） | `TL_ASCEND_POLYHEDRAL_IMPL=cpp ... test_ascend_vsimd_p2_input.py` | 33 passed |
| L3（55） | P1 三文件 | 23 + 15 + 17 = 55 passed |
| L4 | 全量 `testing/ascend`（默认 `legacy`） | 1436 passed / 210 failed / 170 skipped；210 条全部为环境依赖（133 条 `Ascend device compilation failed`、71 条缺 `torch_npu`/npu 设备、4 条无可用 target、2 条 device_assert），无一条涉及 polyhedral/vsimd |

### 11.3 性能（同机、同脚本）

| 场景 | Python 基线 | C++ | 加速 |
| --- | --- | --- | --- |
| P1 交接 \[64,4096] | 48.8 µs | 52.3 µs | — |
| P2 `build_plan` \[64,4096] | 170.2 µs | 30.9 µs | 5.5× |
| P3 `emit_plan` \[64,4096] | 415.6 µs | 74.1 µs | 5.6× |
| **P2+P3 单 nest** | **585.8 µs** | **105.0 µs** | **5.6×** |
| P1+P2+P3 | 750.2 µs | 185.4 µs | 4.0× |

A8（单 nest `plan+emit` ≤ 150 µs）**满足**。关键优化：`ffi::ReprPrint` 单次约 7 µs，
改为自建表达式序列化 + 值键按「写代数」缓存后，发射耗时从 160 µs 降到 74 µs。

### 11.4 待补齐项（未在本轮完成）

| 项 | 状态 | 说明 |
| -- | --- | --- |
| A9（ASAN/UBSAN） | 未执行 | 需要单独的 sanitizer 构建目录（全量 TVM 重编译），本机未跑；C++ 层无裸指针 / 无手工内存管理，风险集中在 `ffi::Array` 越界访问 |
| L5（板上 `.o`） | 未执行 | 需要 113/950 板与 CANN 编译链 |
| L4 全量 | 已跑 | 210 条失败均为设备/工具链环境依赖（见上表），另有一个文件因缺 `torch_npu` 在 collection 阶段报错，已单列 `--ignore` |
| 三态等价（pipeline 级） | 已验证 1 个 kernel | `[8,64]` 逐元素 kernel 走完整 `tilelang.lower(target="ascend")`：`python` 与 `cpp` 的 device IR 文本**逐字相同**，`legacy` 不同（预期，且默认不启用）；`examples/ascend/**` 未逐一跑 |
| 开关可用性 | 已修 | `tl.ascend.polyhedral.impl` 必须用 `TVM_REGISTER_PASS_CONFIG_OPTION` 注册，否则 `PassContext(config=...)` 直接抛 `AttributeError`（已注册为 `ffi::String`） |
