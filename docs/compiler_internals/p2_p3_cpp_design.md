# P2/P3 C++ 化设计、接口与使用说明

> 配套文档：[p2_p3_cpp_port_plan.md](p2_p3_cpp_port_plan.md)（分步计划）、
> [p2_p3_cpp_port_survey.md](p2_p3_cpp_port_survey.md)（现状与契约）、
> [p2_unsupported_catalog.md](p2_unsupported_catalog.md)（40 条拒绝点）。

***

## 1. 模块图

```
src/ascend/transform/polyhedral/
├── primitives.h / .cc   recipe 表、归约表、硬件常量、Unsupported、dtype 工具
├── access.h     / .cc   Axis / AccessClass / AccessInfo、线性化、有限差分、分类
├── body.h       / .cc   StoreSpec、normalize_body、别名折叠
├── value.h      / .cc   ValueNode DAG、build_value、装载守卫、索引向量
├── model.h      / .cc   StorePlan / ReduceInfo / VectorPlan / VectorProgram + FFI 注册
├── plan.h       / .cc   build_plan（拆 nest / 轴打分 / 展平 / 归约 / 多段）
├── emit.h       / .cc   emit_plan（strip-mine / 掩码 / T.simd.* / 归约两策略）
├── render.h     / .cc   render_plan（文本对拍 oracle）
├── stage_passes.h / .cc P1/P2/P3 三个 pass + 载体 + 组合 pass
└── ffi_polyhedral.cc    FFI 入口 + 错误桥 + 离线 driver
```

依赖方向严格向下：`primitives → access → {body, value} → model → plan → {emit, render}`，
`ffi_polyhedral` 只做入口与异常桥。

***

## 2. 数据流与 pass 位置

三个阶段是**三个独立 pass**，同一条 PrimFunc 上依次执行，阶段间用两个
AttrStmt 载体传递（`stage_passes.{h,cc}`）：

```
P1  AscendLiftParallelToArray   只读 P1 分析（PR-A/PR-B/PR-C-D + 展开器自检）
      ↓  每个 LIFTED nest 包一层 AttrStmt(region, "tl.vsimd.array_region")；nest 本身不改
P2  AscendPlanArray             消费 "tl.vsimd.array_region"（只读 region 字段）
      ↓  RegionToPlanInput(region) → BuildPlan → AttrStmt(VectorProgram, "tl.vsimd.array_plan")
      ↓  规划失败 → 去掉载体原样交回（Branch B）
P3  AscendLowerArrayPlan        消费 "tl.vsimd.array_plan"
      ↓  EmitPlan(program) → T.simd.* TIR（残留 region 载体一律剥除）
legacy  AscendSimdVFLowerParallel  兜底（未被 P1 lift 或 P2 拒绝的 nest）
```

`AscendPolyhedralVectorize` 是组合 pass：在同一个 PrimFunc 上顺序调用
`LiftParallelToArray → PlanArray → LowerArrayPlan`，等价于把三个 pass 串起来。

三个 pass 的输入输出契约：

| pass | 输入 | 输出 | IR 副作用 |
| --- | --- | --- | --- |
| P1 | `PrimFunc`（未改动的 SIMD_VF 体） | 同一 body + `tl.vsimd.array_region` 载体 | 只加载体 |
| P2 | `tl.vsimd.array_region` 载体 | `tl.vsimd.array_plan` 载体（`VectorProgram`） | 载体替换 |
| P3 | `tl.vsimd.array_plan` 载体 | `T.simd.*` 语句 | 载体被发射结果替换 |

`pipeline.py` 里的位置（`NormalizeBufferVersion` 之后、`AscendSimdVFLowerParallel`
之前），由 `tl.ascend.polyhedral.impl` 控制，插入的是组合 pass：

| 取值        | 行为                                  |
| --------- | ----------------------------------- |
| `legacy`（默认） | 不插入新 pass，流水线与改动前完全一致（A7）          |
| `cpp`     | 插入 `AscendPolyhedralVectorize`（P1→P2→P3） |

流水线只接受 C++ 实现；此前的 Python 孪生 pass 仅用于灰度对拍，已删除。

离线 API（`build_plan` / `emit_plan` / `render_plan` / `vectorize_parallel_nests`
/ `finite_difference` / `classify_delta`）**只有 C++ 一个实现**：
`tilelang/ascend/analysis/polyhedral/__init__.py` 是纯 C++ 门面（`_cpp` 的转发，
不 import 任何 Python 实现模块），共享值类型 `AccessClass` / `Unsupported`
定义在 `polyhedral/_contract.py`。旧的 `TL_ASCEND_POLYHEDRAL_IMPL` 环境变量与
`dual` 对拍模式已删除。

Python 参考实现整体移到独立包 `tilelang/ascend/analysis/polyhedral_reference/`
（oracle，需要时显式 import），生产门面与测试的依赖树里都不出现它。

***

## 3. 接口

### 3.1 C++ API

| 函数                                                          | 说明                            |
| ----------------------------------------------------------- | ----------------------------- |
| `VectorProgram BuildPlan(const tirx::Stmt&, std::optional<int64_t> lanes)` | `T.Parallel` 嵌套 → 计划（1..N 段） |
| `tirx::Stmt EmitPlan(const ffi::ObjectRef& plan)`            | 计划 → `T.simd.*` TIR            |
| `std::string RenderPlan(const ffi::ObjectRef& plan)`         | 计划 → 可读文本（对拍 oracle）           |
| `PrimFunc LiftParallelToArray(PrimFunc)`                     | P1 阶段：加 `tl.vsimd.array_region` 载体 |
| `PrimFunc PlanArray(PrimFunc)`                               | P2 阶段：region 载体 → `tl.vsimd.array_plan` |
| `PrimFunc LowerArrayPlan(PrimFunc)`                          | P3 阶段：plan 载体 → `T.simd.*`   |
| `PrimFunc VectorizeParallelNests(PrimFunc)`                  | 组合：P1 → P2 → P3               |
| `Pass AscendPolyhedralVectorize()`                           | 组合 pass（流水线入口）                |

### 3.2 FFI 表（Python 侧 `tilelang.transform._ffi_api`）

| 名称                                             | 返回                                                     |
| ---------------------------------------------- | ------------------------------------------------------ |
| `AscendPolyhedralBuildPlan(stmt, lanes)`       | `{ok, program}` 或 `{ok: False, message, diagnostics}`   |
| `AscendPolyhedralEmitPlan(plan)`               | `{ok, stmt}` / 错误同形                                    |
| `AscendPolyhedralRenderPlan(plan)`             | `{ok, text}` / 错误同形                                    |
| `AscendPolyhedralVectorizeNests(stmt, lanes)`  | `{ok, stmt}`（driver）                                   |
| `AscendPolyhedralAxisClass(info, axis)`        | `[kind, stride?]`                                      |
| `AscendPolyhedralFiniteDifference(expr, var, use_analyzer)` | 差分表达式                                     |
| `AscendPolyhedralClassifyDelta(delta, use_analyzer)`        | `[kind, stride?]`                              |
| `AscendLiftParallelToArray()`                  | P1 PrimFunc pass                                      |
| `AscendPlanArray()`                            | P2 PrimFunc pass                                      |
| `AscendLowerArrayPlan()`                       | P3 PrimFunc pass                                      |
| `AscendPolyhedralVectorize()`                  | 组合 PrimFunc pass（P1→P2→P3）                            |

### 3.3 计划对象（FFI Object，Python 字段同名）

`Axis` / `AccessInfo` / `StoreSpec` / `ValueNode` / `StorePlan` / `ReduceInfo` /
`VectorPlan` / `VectorProgram`，字段与 Python dataclass 一一对应（见 survey §2.2）。
两处差异是刻意的：

- `StorePlan` 的 C++ 字段名是 `kind_value`（字符串），Python 类用属性 `kind`
  返回 `AccessClass`。原因：`def_ro` 会以同名属性覆盖类里定义的 property；
- `AccessInfo.axis_class(axis)` 由 C++ 转发（`AscendPolyhedralAxisClass`），
  避免把内部 `per_axis` 映射暴露成 FFI 字段。

### 3.4 错误约定

- C++ 侧只抛 `polyhedral::Unsupported`（`message` + `diagnostics`），
  **不用** `LOG(FATAL)`；内部不变量才用 `TVM_FFI_ICHECK`。
- FFI 层统一转成 `Map{ok:false, message, diagnostics}`；Python 包装层重新抛
  `Unsupported(...)`，`str(err)` 与 `err.diagnostics` 与 Python 实现逐字一致。

***

## 4. 关键实现决策与已知差异

| #   | 决策                                                                        | 说明                                                                 |
| --- | ------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 1   | 计划对象直接用 FFI Object，不做 Map 序列化                                             | 测试里 40+ 处 `plan.segments[0].stores[0].value.recipe` 零改写；P3 直接消费同一份对象 |
| 2   | `ValueNode.is_uniform` 在构造时算好并作为字段                                            | Python 是递归 property；节点不可变，二者等价                                     |
| 3   | 缓存键用「快速结构化序列化」而非 `ffi::ReprPrint`                                           | `ReprPrint` 单次约 7 µs，会占满整个发射耗时；自建序列化保持"结构等 ⇔ 键等"（A8 的关键）         |
| 4   | 值键缓存按「写代数」失效                                                              | `ValueKey` 含 buffer 写代数，`NoteWrite` 时清空缓存                            |
| 5   | `finite_difference` / `classify_delta` 的 `analyzer` 参数在 `impl=cpp` 下创建**新的** C++ analyzer | Python `arith.Analyzer` 不是 FFI 对象，无法跨界传递；测试只传未绑定变量的 analyzer，行为一致      |
| 6   | `emit.py` 的 store 列表在入口处快照（`list(plan.pre_stores)` 等）                        | Python 用 `id(store)` 做累加器簿记；FFI Array 每次取元素都是新包装，快照保证身份稳定          |
| 7   | `emit.py:emit_plan` 对 program 的判断改为鸭子类型                                      | 让 Python 发射器能消费 C++ 的 `VectorProgram`（Gate-3 混合模式）                  |
| 8   | plan 节点声明 `_type_mutable = true`，规划期就地改字段                                     | 与 Python dataclass 就地赋值同义（`promote` / `reduce` / `diagnostics`）     |

已知差异（均不影响测试与产物）：

1. 诊断文本里出现的表达式经由 `ffi::ReprPrint` 打印，与 Python `str()` 同源；
   但快速键序列化只用于内部缓存，不进入任何对外文本。
2. `AccessInfo.per_axis` 不作为 FFI 字段暴露（测试不读），只能通过
   `axis_class(axis)` 访问。
3. `impl=cpp` 下 `finite_difference`/`classify_delta` 忽略调用方 analyzer 的
   绑定状态（见决策 5）。

***

## 5. 使用指南

```python
import tilelang.ascend.language as T
from tilelang.ascend.analysis.polyhedral import build_plan, emit_plan, render_plan

program = build_plan(parallel_nest)     # VectorProgram（C++ FFI 对象）
print(render_plan(program))             # 可读计划（含诊断）
stmt = emit_plan(program)               # T.simd.* TIR
```

需要 Python 参考实现（oracle）时显式导入独立包：

```python
from tilelang.ascend.analysis.polyhedral_reference import build_plan, emit_plan
```

流水线灰度：

```python
import tvm
with tvm.transform.PassContext(config={"tl.ascend.polyhedral.impl": "cpp"}):
    mod = tilelang.lower(func, target="ascend")
```

### 5.1 常见拒绝原因与改法

| 诊断前缀                                                | 原因                         | 改法                                    |
| --------------------------------------------------- | -------------------------- | ------------------------------------- |
| `global-memory operand inside T.Parallel`           | GM 直接参与并行体                  | 先搬到 UB（显式 copy）再做逐元素                 |
| `store element dtype 'bool' ... is not supported`   | bool 推不出谓词宽度                | 用 f32/int8 承载，或改用显式转换                  |
| `non-affine access ... index vector is not constructible` | 数据相关下标 / 非 2 的幂除法          | 改为 2 的幂拆分，或把下标预先算成数组                  |
| `read of 'x' at a different address ... is a stencil` | 并行轴上的跨址读（`s[i-1]`）          | 沿 serial 轴向量化，或先复制一份                  |
| `nested serial loop inside T.Parallel ...`          | 不干净的 serial（跨迭代依赖 / 多并列）    | 去掉循环携带依赖，或拆成多个 nest                   |
| `partial reduction mixed with element-wise stores`  | 同一段里"lane 均匀写 + 逐 lane 值"混排 | 拆成独立段，或用完整归约形式                         |
| `conditional reduction into 'x' is not supported`   | 归约带谓词                      | 先把掩码外元素填单位元，再无条件归约                     |

诊断按顺序输出（`plan.diagnostics` / `Unsupported.diagnostics`），
包含每根候选轴的 store/load 分类、打分与回退原因。

***

## 6. 迁移说明

- 默认关闭（`legacy`）；流水线内**只有 C++ 一个 pass**（Python 孪生 pass 已删除）；
- 灰度顺序：G0 `legacy` → G1 `cpp` 对拍（与 `polyhedral_reference` 的 IR/文本 diff）
  → G2 白名单集群 → G3 `cpp` 默认；
- Python 参考实现整体位于独立包 `tilelang/ascend/analysis/polyhedral_reference/`，
  不再是任何 pass、也不被生产门面或测试导入；只有在 A1–A10 全部通过、与参考实现
  在真实 kernel 集合上零差异、性能数据归档后，才考虑连同该包一起删除。

***

## 7. 注册面清点（先纠正一个前提）

`ffi_polyhedral.cc` 注册 **7 个 FFI 全局函数 + 4 个 PrimFunc pass**。
Python 孪生 pass 已在筛选阶段删除，见 §7.1：

| 名称                                     | 类型                | 用途                             | Python 调用方                                    | 必需性     |
| -------------------------------------- | ----------------- | ------------------------------ | --------------------------------------------- | ------- |
| `AscendPolyhedralBuildPlan`            | FFI 函数            | P2 入口                          | `_cpp.build_plan`                             | 必需      |
| `AscendPolyhedralEmitPlan`             | FFI 函数            | P3 入口                          | `_cpp.emit_plan`                              | 必需      |
| `AscendPolyhedralRenderPlan`           | FFI 函数            | 文本 oracle                      | `_cpp.render_plan`                            | 必需（A5）  |
| `AscendPolyhedralAxisClass`            | FFI 方法转发          | 读 `AccessInfo.per_axis`         | `_cpp.AccessInfo.axis_class`                  | 必需      |
| `AscendPolyhedralVectorizeNests`       | FFI 函数            | driver（离线替换）                    | `_cpp.vectorize_parallel_nests`               | 必需      |
| `AscendPolyhedralFiniteDifference`     | FFI 函数            | 数学核心对拍                         | `_cpp.finite_difference`                      | 对拍/调试   |
| `AscendPolyhedralClassifyDelta`        | FFI 函数            | 数学核心对拍                         | `_cpp.classify_delta`                         | 对拍/调试   |
| `AscendLiftParallelToArray`            | **PrimFunc pass** | P1 阶段（加 region 载体）              | `transform.AscendLiftParallelToArray()`       | 必需      |
| `AscendPlanArray`                      | **PrimFunc pass** | P2 阶段（region → plan 载体）          | `transform.AscendPlanArray()`                 | 必需      |
| `AscendLowerArrayPlan`                 | **PrimFunc pass** | P3 阶段（plan 载体 → TIR）             | `transform.AscendLowerArrayPlan()`            | 必需      |
| `AscendPolyhedralVectorize`            | **PrimFunc pass** | 组合 pass（P1→P2→P3，流水线入口）          | `transform.AscendPolyhedralVectorize()`       | 必需      |

对象类型 8 个（`Axis` / `AccessInfo` / `StoreSpec` / `ValueNode` / `ReduceInfo` /
`StorePlan` / `VectorPlan` / `VectorProgram`），全部被测试的字段路径直接引用。
另有 1 个内部载体类型 `tl.vsimd.Region`（P1→P2 的 `VSimdRegion` 句柄，不暴露字段）。

C++ 公开 API 只有 3 个 `TVM_DLL`：`BuildPlan` / `EmitPlan` / `RenderPlan`
（阶段函数 `LiftParallelToArray` / `PlanArray` / `LowerArrayPlan` /
`VectorizeParallelNests` 在 `stage_passes.h` 中同样 `TVM_DLL`）。

### 7.1 全量 pass 盘点与筛选结论

全项目 pass 注册面（不含 `3rdparty/tvm` 自带 pass）：

| 归属                 | 数量    | 语言  | 分类                                                     |
| ------------------ | ----- | --- | ------------------------------------------------------ |
| `src/ascend/transform` | 35    | C++ | P1=3（`AscendAnalyzeSimdVF{Capture,Memory}`、`AscendAnalyzeVSimdRegion`，只读）；P2/P3=4（`AscendLiftParallelToArray` / `AscendPlanArray` / `AscendLowerArrayPlan` / 组合 `AscendPolyhedralVectorize`）；其余 31 为既有 Ascend 流水线 pass |
| `src/transform`      | 40    | C++ | 后端无关流水线 pass（`Simplify` / `FlattenBuffer` / `SplitHostDevice` …） |
| `src/cuda/transform` | 16    | C++ | CUDA 后端                                            |
| `src/cpu/transform`  | 4     | C++ | CPU 后端                                             |
| `tilelang/**/*.py`   | 12    | Python | 既有诊断/后端 pass（`LayoutVisual` / `VFChecker` / `DecoupleTypeCast` / `MetalFragmentToSimdgroup` …），均非本次新增 |
| `PassPipeline`       | 7     | Python | `ascend` / `cuda` / `c` / `llvm` / `metal` / `hip` / `webgpu` |

筛选规则：**本地 git 更改内，pass 只保留 C++ 的 P2/P3 实现**。执行结果：

- 删除 `AscendPolyhedralVectorizePython`（Python 孪生 pass）及其 `__all__` 条目；
- `pipeline.py` 的开关从三态收敛为 `legacy|cpp`（删除 `impl=python` 分支）；
- 保留 `AscendPolyhedralVectorize`（C++ pass）与 9 个 FFI 入口、8 个对象类型，逻辑与接口零改动；
- 保留 `tilelang/ascend/analysis/polyhedral_reference/*.py`：它们**没有 pass 注册**，
  是 P2/P3 的参考实现与 113 个用例的 oracle；生产门面
  `tilelang/ascend/analysis/polyhedral/` 只剩 `__init__.py`（纯 C++ 转发）、`_cpp.py`
  与 `_contract.py`，与参考实现之间没有任何导入边；
- 其余 94 个 C++ pass、12 个既有 Python pass 不在本次更改范围内，未触碰。

***

## 8. 可合并性评估

### 8.1 文件层（17 个文件 / 6855 行）

| 候选合并                                       | 合并后规模      | 收益           | 代价                                  | 结论       |
| ------------------------------------------ | ---------- | ------------ | ----------------------------------- | -------- |
| `access` + `body`                          | ~710 行     | 少一对手册；同为"输入侧建模" | 两个概念（迭代域 / 语句规范化）混在一处               | **可选，低收益** |
| `value` + `model`                          | ~1420 行    | 少一层头文件依赖     | model 依赖 value 的 reads 判据，合并后仍是"值 DAG + 计划"两件事；review 变难 | **不建议**  |
| `render` → `ffi_polyhedral.cc`             | 213 → 342 行 | 少一对文件        | 削弱"render 是独立 oracle"的语义；单测想直接用 render 时只能走 FFI | **可选**   |
| `plan` + `emit`                            | ~3600 行    | 无            | 增量编译变慢（改 emit 会重编 plan）；review 面积翻倍  | **明确反对** |
| `primitives` 拆表 / 合表                      | —          | 无            | 无                                   | **不动**   |

**反向建议：`emit.cc` 应拆分而不是合并。** 它已 2011 行，是编译耗时与 review 的
热点。按职责可拆为：

```
emit_value.cc   _Emitter / ValueKey / 快序列化 / T.simd.* 调用构造   (~700 行)
emit_store.cc   store / outer-store / strip-mine / 掩码            (~600 行)
emit_reduce.cc  归约两条策略 + 段/程序发射                          (~700 行)
```
只共享 `emit_internal.h`（`Emitter` 声明）。收益：增量编译并行、单文件可读；
代价：`Emitter` 的私有状态需要暴露给同目录的实现（用 `friend` 或 protected 成员）。

### 8.2 FFI 入口层

- 6 个"在线"入口（build/emit/render/axis_class/vectorize_nests/pass）**不建议合并**：
  合并成一个 `Invoke(op, ...)` 会丢掉参数类型、丢掉文档，Python 侧还需手写分支。
- 2 个"对拍/调试"入口（`FiniteDifference` / `ClassifyDelta`）保留：它们是数学核心
  的唯一单测通道。常量表 FFI（`AscendPolyhedralPrimitives`）零调用方且每次重建
  Map 约 10 µs，**已删除**（含 `_cpp.primitives()` 与 `PrimitiveTables()`）。
- pass 入口只保留 C++ 一个（`AscendPolyhedralVectorize`）。Python 孪生
  `AscendPolyhedralVectorizePython` 已随筛选删除；Python 侧参考实现整体位于
  独立包 `polyhedral_reference/`（无 pass 注册、不被门面或测试导入），
  需要 oracle 对拍时显式 import。

### 8.3 对象类型层

**不能合并。** 8 个类型对应测试里被直接断言的字段路径
（`program.segments[i].stores[j].value.recipe`、`store.spec.buffer`、
`store.access.offset`、`plan.reduce.op`、`plan.axes[0].name` …）。
把它们内联成普通字段会同时破坏 D2（字段同名）与 A1（测试零改写）。

### 8.4 结论

| 结论                     | 说明                                                              |
| ---------------------- | --------------------------------------------------------------- |
| 不存在"注册了太多 pass"的问题      | 7 个 FFI 函数 + 4 个 pass；入口数量由 API 语义决定，已是最小集              |
| 文件**不建议合并**            | 每个文件对应一个 Python 模块，是双实现逐模块对拍的天然锚点；合并会牺牲编译粒度与 review 效率         |
| 唯一推荐的结构调整              | `emit.cc` **拆分**（见 8.1），可选把 `render` 并进 `ffi`                     |
| API 收敛                 | 已执行：删除零调用的 `AscendPolyhedralPrimitives`                          |

***

## 9. 优化点清单

### 9.1 已实施（有数据）

| # | 位置 | 优化 | 效果 |
| -- | --- | --- | --- |
| 1 | `emit.cc` `ValueKey` / `IndexKey` / `AccessKey` | 自建表达式序列化替代 `ffi::ReprPrint`（单次约 7 µs） | 发射 160 → 76.5 µs |
| 2 | `emit.cc` `BufferKey` | 同上（每个 segment 要算 3 个 buffer key，原来各含 3~5 次打印） | 发射 76.5 → **19.3 µs** |
| 3 | `Emitter` | 值键按"写代数"缓存，`NoteWrite` 时失效 | 与 #1/#2 合计约 **8.6×** |
| 4 | `primitives.cc` | `DataTypeOf` / `ElemDataType` / `ElemDtype` 加互斥保护的小表缓存 | 约 1–2 µs（噪声内） |
| 5 | `plan.cc` | `finish` / `diagnostics` 用 `std::vector<std::string>` 拼接，减少 FFI 装箱 | 微小 |
| 6 | `emit.cc` | 删除死代码 `ForwardKey` | — |

当前（同机、`tmp/bench_p2_p3.py`）：

| 场景 | Python 基线 | C++ | 加速 |
| --- | --- | --- | --- |
| P2 `build_plan` \[64,4096] | 170.2 µs | 31.1 µs | 5.5× |
| P3 `emit_plan` \[64,4096] | 415.6 µs | 19.3 µs | **21.5×** |
| **P2+P3 单 nest** | **585.8 µs** | **50.4 µs** | **11.6×** |
| P1+P2+P3 | 750.2 µs | 151.9 µs | 4.9× |

### 9.2 待做（按收益/风险排序）

| 优先级 | 位置 | 优化 | 预期收益 | 风险 |
| --- | --- | --- | --- | --- |
| P1 | `plan.cc` `ClassifyAccess` | 每轴一次 `Substitute(v→v+1)` + `Simplify`；可把同一 offset 的 `v+1 代入` 只做一次，多轴共享 | P2 的 31 µs 中约一半（轴数 × 访问数） | 中：化简结果必须与参考实现逐位一致，需全量回归 |
| P1 | `plan.cc` 展平诊断 | `PrintExpr(extent)` 改 `IntImm→to_string` 快路径（文本不变） | 展平路径 −7 µs/nest | 低 |
| P2 | `emit.cc` `AccessClassFromName` | store 上存枚举或静态 `std::map` 查表（当前每次线性 7 项字符串比较） | −1~2 µs | 低 |
| P2 | `emit.cc` `EmitSegmentStmts` | 8 个参数收敛为 `EmitOptions` 结构体；`BuildPlanForAxis` 11 参数改为接收 `SegmentSpecs` | 可维护性 | 低 |
| P3 | `emit.cc` | 拆分（见 8.1） | 增量编译 | 低 |

（原表中 `dual` 对拍、`polyhedral_impl()` 环境变量、`AscendPolyhedralPrimitives`
缓存三条已随相应用户/文档项落地而失效。）

### 9.3 明确不做

1. 不合并 `plan`/`emit`，不为了"少几个文件"牺牲编译粒度和 review 效率；
2. 不把 8 个 FFI 对象合并或内联（破坏测试契约）；
3. 不引入 Python→C++ 的回调式统一 pass（跨语言回调更难调试）；
4. 不为性能改动 `arith::Analyzer` 配置或跳过 `Simplify`（会改变判定结果）；
5. 不在本轮删除 Python 实现（灰度 G4 之前保留）。

***

## 10. 推荐重构方案（若要落地）

### 方案 A（保守，推荐）

1. ~~删除 `AscendPolyhedralPrimitives` 入口与 `_cpp.primitives()`~~（已完成）；
2. `emit.cc` 按 8.1 拆成 3 个 `.cc` + 1 个 `emit_internal.h`；
3. Python 参考实现保留在独立包 `polyhedral_reference/`，不并入生产门面（已完成）；
4. `EmitSegmentStmts` / `BuildPlanForAxis` 参数收敛；
5. 文件数 17 → 20（拆多 3 个），但最大单文件从 2011 → ~700 行。

### 方案 B（激进，不推荐）

把 `access+body`、`value+model`、`render+ffi` 各合并一次：17 → 14 文件。
收益仅是"文件数少"，代价是分层与 Python 模块不再一一对应，对拍定位变难。

### 验证方式（任何结构调整都必须过）

```bash
python tmp/run_poly_test.py testing/ascend/transform/test_ascend_polyhedral_vectorize.py -q
python tmp/run_poly_test.py testing/ascend/transform/test_ascend_vsimd_p2_input.py -q
python tmp/bench_p2_p3.py          # 调整结构不应使性能回退
```

`test_ascend_vsimd_p2_input.py` 合并了 P1→P2 交接契约（33）与 P2/P3 pass 级
（`AscendPolyhedralVectorize` + 三个分段 pass + 流水线开关，10）两套用例。

`test_ascend_polyhedral_vectorize.py` 自带 `test_cpp_implementation_is_the_only_one_loaded`：
断言测试期间没有加载 `polyhedral_reference` 的任何模块、每个入口都由 `_cpp` 拥有、
计划对象来自 C++ FFI。
