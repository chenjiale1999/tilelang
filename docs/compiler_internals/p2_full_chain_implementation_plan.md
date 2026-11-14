# P2 全链路实现方案（方案 A：复用 P1 展开 + Parallel 适配）

> 目标：在**不修改** P1 分析/展开逻辑、**不修改** `build_plan` 算法的前提下，新增一个独立 C++ 函数，复用 P1 三个 Pass 的内部函数，构造 `Stmt` 并交给现有 `build_plan`，打通 `P1 → P2 → P3` 全链路。
>
> 依据代码：
>
> - P1：`/home/cjl/workspace/tilelang-stable/src/ascend/transform/vsimd_region.cc`
> - P2：`/home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py`（`build_plan`）
> - P3：`/home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py`（`emit_plan`）
> - 样例：`/home/cjl/workspace/p.log`、`testing/ascend/transform/test_ascend_vsimd_region.py`

***

## 0. 摘要

| 项    | 结论                                                                                                                                                             |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 方案   | **方案 A**：复用 `ExpandVSimdRegionToLoops` 生成参考展开，再用薄适配器把**整条 For 链**的 `ForKind::kSerial` 全部改回 `kParallel`，交给现有 `build_plan` |
| 新增逻辑 | “提取 For + 改 ForKind”的适配器，以及用 P1 `bindings` 做作用域恢复（`RestoreBindings`）；G1–G6、字段拷贝、store 重建、展开逻辑全部复用 P1                                                           |
| 不修改  | P1 的分析/展开/报告；`build_plan` 的签名与算法；P3 的 `emit_plan`                                                                                                              |
| 新增接口 | `BuildP2PlanInputs` / `BuildP2PlanInput`（C++）+ FFI + Python 薄封装                                                                                                |
| 全链路 | `AnalyzeVSimdMemoryStatuses` → `AnalyzeVSimdCaptureInfo` → `BuildMinimalRegion` → `ExpandVSimdRegionToLoops` → `AsParallelNest` → `build_plan` → `emit_plan` |
| 验收 | `p.log` golden（1-D + 2-D）、`test_simple_region_lifted` / `test_2d_lifted`、plan/emit parity、边界/负向矩阵、P1 55 + P2 135 + 新增用例 100% pass |

***

## 1. 背景与目标

### 1.1 现有基础

- P1 三个 Pass（只读诊断）：
  - `AscendAnalyzeSimdVFMemory` → `AnalyzeVSimdMemoryLegality` / `AnalyzeVSimdMemoryStatuses`
  - `AscendAnalyzeSimdVFCapture` → `AnalyzeVSimdCaptureScope` / `AnalyzeVSimdCaptureInfo`
  - `AscendAnalyzeVSimdRegion` → `AnalyzeVSimdRegionReport`（PR-A+B+C+D 联合文本）
- `AnalyzeVSimdRegionReport`（`vsimd_region.cc:363-405`）的组合顺序：
  1. `AnalyzeVSimdMemoryStatuses`（每 VF 状态）；
  2. `AnalyzeVSimdCaptureInfo`（每 nest 信息）；
  3. 对每个 info 调 `BuildMinimalRegion`（PR-D，支持 N 维矩形域）；
  4. lift 成功时调 `ExpandVSimdRegionToLoops`（当前 L219-249，逐层递归包循环），并把结果写入 `expanded:` 文本。
- P1 当前能力（PR-D）：`domain` 可为任意矩形 rank；`accesses` 为 `VSimdAccess`（含 `dep_axes/value_axes/value_shape`）；参考展开为 N 层 `For(kSerial)` 嵌套。
- P2 `build_plan(stmt, *, lanes=None)` 接收一个 **`T.Parallel` For（支持 N 层并行链）**，返回 `VectorProgram`。
- P3 `emit_plan(plan)` 接收 `VectorPlan/VectorProgram`，输出 `tl.simd.*` TIR。

### 1.2 目标

1. 新增一个独立函数，直接复用 P1 内部函数，产出 `Stmt`；
2. 把该 `Stmt` 传给现有 `build_plan`，完成全链路；
3. 保持 P1 报告文本、P2/P3 现有行为不变；
4. 用 `p.log`（2-D 样例）与 `test_simple_region_lifted` / `test_2d_lifted` 做逐维度验证；
5. 覆盖 P1 当前（PR-D）的全部支持/拒绝边界，测试 100% 通过。

### 1.3 非目标

- 不扩展 P1（归约/控制流/serial/体内 `Bind`）；
- 不实现归约、语句级控制流、serial 吸收、体内 `Bind` 等 P1 当前（PR-D）未覆盖的能力；
- 不把 P2/P3 全量迁移到 C++；
- 不修改 `build_plan` 的算法与签名；
- 不实现新的指令发射。

> 本轮可规划范围（按更新后的 PR-D 计数）：
> - **18 个**“P1 lift + P2 OK”场景：A 类 15（含 A3/A4/A7 N 维）+ C1 + C3 + D2；
> - **3 个**“P1 lift + P2 拒绝”场景（Branch B）：`c_gm`、`c4_true_random`、`d15_bool_store`；
> - **45 个** P1 拒绝场景（其中 37 个 P2 OK、8 个 P2 REJ），包含归约 11、语句级控制流 8、串行 24、体内 `Bind` 2（C2/C4）。
>
> 更新要点：PR-D 让 A3/A4/A7 从“仅分析”变为可 lift；`2e9beb8e` 让 C2/C4（体内 `Bind`）变为拒绝；◐ 归零。

***

## 2. 关键接口矛盾与方案 A 的解决方式

### 2.1 矛盾

`ExpandVSimdRegionToLoops`（`vsimd_region.cc:219-249`）生成的是 **N 层嵌套的 `For(kSerial)` 链**：

```cpp
// 2-D 示例：
For(i, min_i, ext_i, ForKind::kSerial,
    For(j, min_j, ext_j, ForKind::kSerial, body))
```

而 `build_plan` 的 `_split_nest` 只从 `T.Parallel` 轴开始收集迭代域；传入 `kSerial` For 会报：

```
no T.Parallel axis found
```

因此 **不能把 `expanded_stmt` 原样传给 `build_plan`**。

### 2.2 方案 A

在 P1 展开之后、`build_plan` 之前，加一个**薄适配器** `AsParallelNest`，**递归地把整条 `For` 链的 `kSerial` 全部改为 `kParallel`**：

```cpp
Stmt expanded = ExpandVSimdRegionToLoops(lift.region);  // 复用 L219
Stmt parallel = AsParallelNest(expanded);                // 逐层改 ForKind
// 2-D 结果：
// For(i, min_i, ext_i, kParallel, For(j, min_j, ext_j, kParallel, body))
```

安全前提（必须断言）：

1. `lift.lifted == true`；
2. `!lift.region.domain.empty()`（PR-D 允许任意矩形 rank）；
3. 原始域是 `T.Parallel`（PR-B 的 `ParallelNest` 只收集 `ForKind::kParallel`）；
4. PR-A `memory_status == LEGAL`（无跨迭代 RAW/WAR/WAW）。

在这四个前提下，把参考展开的 `kSerial` 改回 `kParallel` 不改变语义：P1 的串行循环只是“语义 oracle”，原始域本来就是并行且已证明无跨迭代依赖。

### 2.3 为什么不是其他方案

| 方案                                             | 问题                                                   |
| ---------------------------------------------- | ---------------------------------------------------- |
| B. 修改 `build_plan` 接受 serial                   | 需要改 `_split_nest` 语义，影响所有 P2 路径；违反“使用现有 build\_plan” |
| C. 修改 `ExpandVSimdRegionToLoops` 直接生成 Parallel | 改变 P1 参考展开的“语义 oracle”定位，影响 P1 自检与报告                 |
| D. 从 region 重新手工构造 For                         | 重复实现 store 重建/DeclBuffer/顺序逻辑，违反“禁止重复实现”             |

***

## 3. 总体设计

### 3.1 数据流

```
PrimFunc
  │
  ├─ AnalyzeVSimdMemoryStatuses(func)       // PR-A：每 VF 状态
  ├─ AnalyzeVSimdCaptureInfo(func)          // PR-B：每 nest 信息
  │
  └─ for each nest:
        BuildMinimalRegion(info, status)    // PR-D：G1-G6 + N-D 字段拷贝 + 自检
          ├─ ExpandVSimdRegionToLoops       // 复用 L219：N 层标量参考展开
          └─ AsParallelNest                 // 新增：递归提取 For 链 + 全部改 kParallel
                │
                ▼
          VSimdP2Input{status, reason, stmt}
                │
                ▼ FFI
          Python: build_plan(stmt) → VectorProgram
                │
                ▼
          Python: emit_plan(program) → TIR + tl.simd.*
```

### 3.2 分层

| 层        | 职责                                                             | 是否新增  |
| -------- | -------------------------------------------------------------- | ----- |
| P1 分析层   | PR-A/PR-B/PR-C/PR-D，全部复用                                        | 否     |
| P2 输入适配层 | `BuildP2PlanInputs` + `AsParallelNest` + `RestoreBindings`       | **是** |
| P2 规划层   | `build_plan`                           | 否     |
| P3 发射层   | `emit_plan`                            | 否     |

***

## 4. 详细实现

### 4.1 数据结构

```cpp
// 建议放在 vsimd_p2_input.h
struct VSimdP2Input {
  int vf_index{-1};
  int nest_index{-1};

  // "LIFTED" / "FALLBACK" / "INTERNAL_ERROR" / "OPAQUE" / "NOT_FOUND"
  std::string status;
  std::string reason;

  // LIFTED 时有效
  VSimdRegion region;   // PR-D typed region；accesses 为 VSimdAccess（含 dep_axes/value_axes/value_shape）
  tirx::Stmt stmt;      // N 层 For(kParallel) 链，可直接传给 build_plan
};
```

字段说明：

| 字段                    | 语义                                                            |
| --------------------- | ------------------------------------------------------------- |
| `vf_index/nest_index` | 与 P1 报告/P2 定位一致                                               |
| `status`              | P1 判定；`LIFTED` 仅表示 P1 可向量化，不保证 P2 一定成功                        |
| `reason`              | `FALLBACK`/`INTERNAL_ERROR`/`OPAQUE`/`NOT_FOUND` 的原因           |
| `region`              | PR-D typed region；`domain` 可为任意矩形 rank，`accesses` 为 `VSimdAccess` |
| `stmt`                | 适配后的 `For(kParallel)` 链（1..N 层）；`LIFTED` 且适配成功时有效         |

`VSimdAccess`（PR-D）字段：

| 字段 | 语义 |
| --- | --- |
| `access` | 访存侧：`buffer + indices(+value) + is_write` |
| `dep_axes` | 仅由 `indices` 推出的“用到的域轴”；缺轴 = 广播 |
| `value_axes` | v1 等于域轴（占位）；未来表示值的轴 |
| `value_shape` | v1 等于域 extent（占位）；未来表示值的形状 |

> `build_plan` 仍从 `stmt` 重新分析，因此 `dep_axes/value_*` 不参与本轮 P2 输入；它们保留在 `region` 中供调试与后续 `PlanOptions` 使用。

### 4.2 新增 C++ API

```cpp
// 返回函数内所有 nest 的 P2 输入（按 P1 报告顺序）
TVM_DLL std::vector<VSimdP2Input>
BuildP2PlanInputs(const tirx::PrimFunc &func);

// 按 vf_index/nest_index 取单个；找不到返回 status="NOT_FOUND"
TVM_DLL VSimdP2Input
BuildP2PlanInput(const tirx::PrimFunc &func, int vf_index, int nest_index);
```

实现位置二选一（推荐新文件，保持 `vsimd_region.cc` 只负责 P1 报告）：

- `src/ascend/transform/polyhedral/vsimd_p2_input.h`
- `src/ascend/transform/polyhedral/vsimd_p2_input.cc`

CMake 已用 `file(GLOB src/ascend/transform/*.cc)`，新文件无需修改构建脚本。

### 4.3 实现流程

```cpp
std::vector<VSimdP2Input> BuildP2PlanInputs(const PrimFunc &func) {
  std::vector<std::string> statuses = AnalyzeVSimdMemoryStatuses(func);
  std::vector<VSimdCaptureInfo> infos = AnalyzeVSimdCaptureInfo(func);

  std::vector<VSimdP2Input> out;
  for (const VSimdCaptureInfo &info : infos) {
    VSimdP2Input item;
    item.vf_index = info.vf_index;
    item.nest_index = info.nest_index;

    if (info.nest_index < 0) {
      item.status = "OPAQUE";
      item.reason = info.message;
      out.push_back(std::move(item));
      continue;
    }

    std::string memory_status =
        (info.vf_index >= 0 && static_cast<size_t>(info.vf_index) < statuses.size())
            ? statuses[info.vf_index]
            : "MISSING";

    VSimdRegionLift lift = BuildMinimalRegion(info, memory_status);  // PR-D
    if (!lift.lifted) {
      item.status = lift.internal_error ? "INTERNAL_ERROR" : "FALLBACK";
      item.reason = lift.reason;
      out.push_back(std::move(item));
      continue;
    }

    item.region = lift.region;
    Stmt expanded = ExpandVSimdRegionToLoops(lift.region);  // 复用 L219
    Stmt parallel;
    std::string error;
    if (!AsParallelNest(expanded, &parallel, &error)) {
      item.status = "INTERNAL_ERROR";
      item.reason = error;
      out.push_back(std::move(item));
      continue;
    }
    // 参考展开只重建 store，不包含外层 Bind；把 region.bindings 中
    // 在 stmt 里自由出现的 Var 内联回去（与 P2 _collect_aliases 同语义）
    parallel = RestoreBindings(parallel, lift.region.bindings);

    item.status = "LIFTED";
    item.stmt = parallel;
    out.push_back(std::move(item));
  }
  return out;
}
```

### 4.4 适配器 `AsParallelNest`（递归，支持 N 维）

```cpp
bool AsParallelNest(const Stmt &expanded, Stmt *out, std::string *error);
```

规则：

| `expanded` 形态 | 处理 |
| --- | --- |
| `For(kSerial, var, min, extent, body)` | 递归处理 `body`；返回 `For(var, min, extent, kParallel, new_body)`；**所有嵌套的 `kSerial` 层都改为 `kParallel`** |
| `SeqStmt`，其中恰好一个外层 `For`，其余为 `DeclBuffer` | 提取该外层 `For`，递归改 `kParallel`；`DeclBuffer` 丢弃（Buffer 对象仍在 store/load 中，P3 在原位置发射时 view 仍在作用域） |
| `SeqStmt` 含多个**并列** `For` | 失败：`"expanded has multiple sibling loops"`（P1 单 nest 不应出现） |
| 无 `For`（如 `Evaluate(0)`） | 失败：`"expanded has no loop"` |
| `For` 不是 `kSerial` | 失败：`"unexpected expanded loop kind"` |

实现要点：

1. 递归遍历 `SeqStmt`/`For`；嵌套 `For` 链是合法的（N 维），**不是**多个并列循环；
2. 只允许 `DeclBuffer` 作为非 `For` 语句；
3. 用同一个 `Var`、`min`、`extent` 重建每一层 `For`；
4. 保留最内层 body 原样（`SeqStmt` 或多条 `BufferStore`）；
5. 不复制任何地址分析、值分析、G1–G6 逻辑。

2-D 示例：

```
expanded: For(i, kSerial, For(j, kSerial, body))
   ↓
parallel: For(i, kParallel, For(j, kParallel, body))
```

### 4.4.1 Bind 作用域恢复（`RestoreBindings`）

**先澄清 P1 的当前边界**：更新后的 P1 对**体内 `Bind`**（例如 `acc = a[i] * 2; c[i] = acc + 1`）已收紧为 `OPAQUE`（C2/C4，N4）；这类场景不会进入本轮全链路。

`RestoreBindings` 只处理**外层 `Bind`**（例如 `tail = T.min(n - bx * bn, bn)`，定义在 `T.SimdVF` 之外、用于 `T.Parallel(tail)` 的 extent）。`ExpandVSimdRegionToLoops` 只重建 store，不重建外层 `Bind`；如果 extent/index 引用 `tail`，直接传给 `build_plan` 可能出现未定义 Var。

```cpp
Stmt RestoreBindings(const Stmt &stmt,
                     const std::vector<std::pair<Var, PrimExpr>> &bindings);
```

规则：

1. 只对 `bindings` 中**在 stmt 里自由出现**的 `Var` 做替换；替换范围包括各层 `For` 的 `min`/`extent` 与最内层 body；
2. 跳过值为 `Buffer` / `Call` 的绑定（与 P2 `_collect_aliases` 一致）；
3. 允许链式替换，但最多迭代 8 次（与 P2 一致），避免环；
4. 替换后如果仍有未定义 Var，不崩溃；交给 `build_plan` 以 `Unsupported` 报出，归入 Branch B；
5. `assumptions` 当前不注入 `build_plan`（`build_plan` 内部新建 analyzer）；涉及 `T.assume` 的用例记录为“已知边界”，后续通过 `PlanOptions` 扩展。

这一步是**作用域恢复**，不是重新分析：它只把 P1 已经捕获的外层 `bindings` 放回表达式，不改变任何地址/值的语义。

### 4.5 FFI

```cpp
ffi::Array<ffi::Map<ffi::String, ffi::Any>>
AscendBuildP2PlanInputs(const tirx::PrimFunc &func);

ffi::Map<ffi::String, ffi::Any>
AscendBuildP2PlanInput(const tirx::PrimFunc &func, int vf_index, int nest_index);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendBuildP2PlanInputs",
                        AscendBuildP2PlanInputs);
  refl::GlobalDef().def("tl.transform.AscendBuildP2PlanInput",
                        AscendBuildP2PlanInput);
}
```

Map 字段：

| key          | 类型          | 说明                                         |
| ------------ | ----------- | ------------------------------------------ |
| `vf_index`   | int         | 定位                                         |
| `nest_index` | int         | 定位                                         |
| `status`     | String      | `LIFTED/FALLBACK/INTERNAL_ERROR/OPAQUE/NOT_FOUND` |
| `reason`     | String      | 失败原因                                       |
| `stmt`       | `tirx.Stmt` | 仅 `LIFTED` 时存在；可直接被 Python `build_plan` 使用 |

> `VSimdP2Input.region` 保留在 C++ 侧（typed `VSimdRegion`），不放进 FFI Map：避免 `VSimdAccess` 的表达式/形状被字符串化；Python 侧本轮只需要 `status` 与 `stmt`。

### 4.6 Python 入口

`tilelang/ascend/transform/__init__.py` 追加（沿用现有 `_ffi_api` 动态绑定）：

```python
def AscendBuildP2PlanInputs(func):
    """Return per-nest P1→P2 inputs (status + optional parallel Stmt)."""
    return _ffi_api.AscendBuildP2PlanInputs(func)


def AscendBuildP2PlanInput(func, vf_index, nest_index):
    """Return one P1→P2 input by vf/nest index."""
    return _ffi_api.AscendBuildP2PlanInput(func, vf_index, nest_index)
```

### 4.7 全链路调用

```python
from tilelang.ascend import transform as at
from tilelang.ascend.analysis.polyhedral import build_plan, emit_plan

inputs = at.AscendBuildP2PlanInputs(main)
for item in inputs:
    if item["status"] != "LIFTED":
        continue
    program = build_plan(item["stmt"])     # 现有函数，不改
    out = emit_plan(program)               # 现有 P3 原型
```

P2 失败归类：

```python
try:
    program = build_plan(item["stmt"])
except Unsupported as err:
    # Branch B：P1 可向量化，但 P2/P3 能力不足
    record_branch_b(item, err)
```

***

## 5. 文件变更清单

| 文件                                                       | 变更     | 说明                         |
| -------------------------------------------------------- | ------ | -------------------------- |
| `src/ascend/transform/polyhedral/vsimd_p2_input.h`                  | 新增     | `VSimdP2Input` + 两个 API 声明 |
| `src/ascend/transform/polyhedral/vsimd_p2_input.cc`                 | 新增     | 实现、`AsParallelNest`、`RestoreBindings`、FFI 注册 |
| `tilelang/ascend/transform/__init__.py`                  | 追加     | 两个 Python 薄封装 + `__all__`  |
| `testing/ascend/transform/test_ascend_vsimd_p2_input.py` | 新增     | 全链路/边界/负向测试                |
| 现有 P1/P2 文件                                              | **不改** | 保证复用与零回归                   |

***

## 6. 验证方案

### 6.1 `p.log` / 测试用例 golden（1-D + 2-D）

输入（`test_simple_region_lifted`，`n=64`）：

```python
with T.Kernel(1):
    a_ub = T.alloc_shared((n,), "float32")
    c_ub = T.alloc_shared((n,), "float32")
    T.copy(A, a_ub)
    with T.SimdVF():
        for i in T.Parallel(n):
            c_ub[i] = a_ub[i] + 1.0
    T.copy(c_ub, C)
```

报告断言（对应 `p.log`）：

| 维度       | 断言                                                    |
| -------- | ----------------------------------------------------- |
| 状态       | `VF#0 nest 0: status=LIFTED`                          |
| domain   | `domain: i in [0, 64)`                                |
| captures | 包含 `c_ub[alloc]`、`a_ub[alloc]`                        |
| accesses | 包含 `W c_ub[i] = a_ub[i] + T.float32(1.0) shape=[64] axes=[0] dep={0}`、`R a_ub[i] shape=[64] axes=[0] dep={0}` |
| expanded | 包含 `for i in range(64)`、`c_ub[i] = a_ub[i]`           |
| verify   | `verify: OK`                                          |
| carrier  | 包含 `tl.vsimd.region`                                  |

新接口断言：

| 维度     | 断言                                                                                     |
| ------ | -------------------------------------------------------------------------------------- |
| status | `"LIFTED"`                                                                             |
| stmt   | `tirx.For`，`kind == ForKind.PARALLEL`，`loop_var.name == "i"`，`min == 0`，`extent == 64` |
| body   | 1 条 `BufferStore`：`c_ub[i] = a_ub[i] + 1.0`                                            |

`build_plan` 断言：

| 字段                         | 期望                                                                                                 |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| 返回类型                       | `VectorProgram`                                                                                    |
| `segments`                 | 1 段                                                                                                |
| `program.axes`             | `[]`（唯一轴是向量轴）                                                                                      |
| `segment.vector_axis.name` | `i`                                                                                                |
| `lanes`                    | 64（fp32）                                                                                           |
| `chunks`                   | 1                                                                                                  |
| `tail`                     | 0                                                                                                  |
| `stores`                   | 1 条                                                                                                |
| `stores[0].kind`           | `CONTIGUOUS`                                                                                       |
| `stores[0].value.recipe`   | 以 `build_plan(original_for)` 为 oracle；`a_ub[i] + 1.0` 可能是 `vadd` 或 `vadds`（取决于 P2 的标量-向量特化），测试不硬编码 |
| `reduce`                   | `None`                                                                                             |

**2-D golden（当前 `p.log:22-39` / `test_2d_lifted`，`m=8, n=64`）**：

| 维度 | 断言 |
| --- | --- |
| domain | `domain: i in [0, 8) j in [0, 64)` |
| accesses | `W c_ub[i, j] = a_ub[i, j] + T.float32(1.0) shape=[8,64] axes=[0,1] dep={0,1}`；`R a_ub[i, j] shape=[8,64] axes=[0,1] dep={0,1}` |
| expanded | `for i, j in T.grid(8, 64)` |
| verify | `verify: OK` |
| 新接口 stmt | `For(i, kParallel, For(j, kParallel, body))`（两层都是 Parallel） |
| `build_plan` / `emit_plan` | 以 `build_plan(original_for)` 为唯一 oracle；不硬编码 `vector_axis`（可能是 `i` 或 `j`），只断言 parity |

`emit_plan` 断言：

- **以** **`emit_plan(original_plan)`** **的** **`tl.simd.*`** **op 序列为唯一 oracle**，断言两者完全一致；
- 1-D 情况下地址为 `c_ub[i*64]` / `a_ub[i*64]`；2-D 以 parity 为准；
- 满 chunk 无用户谓词；
- 不硬编码固定 op 名单（`vdup` 是否出现取决于 P2 对 `+1.0` 的特化）。

> `p.log` 的 `expanded:` 由 TVM printer 输出，含 buffer 声明；测试以 IR 结构断言为准，不做逐字节文本比较。

### 6.2 强 parity 校验（主断言）

对每个 **P1 可 lift 且 P2 可规划**的场景（18 个：A 类 15 + C1 + C3 + D2）：

```python
original_for = _first_parallel(main)          # 原始 T.Parallel For
original_plan = build_plan(original_for)
p1_stmt = at.AscendBuildP2PlanInputs(main)[0]["stmt"]
p1_plan = build_plan(p1_stmt)
assert render_plan(original_plan) == render_plan(p1_plan)
assert structural_equal(emit_plan(original_plan), emit_plan(p1_plan), map_free_vars=True)
```

比较维度：

- `program.axes`、`segments[*].axes/vector_axis/lanes/chunks/tail`；
- 每个 `StorePlan` 的 `spec.buffer/indices/value`、`kind`、`mask`、`reduce`；
- `guard_boundaries`、`diagnostics`；
- `emit_plan` 的 `tl.simd.*` op 序列。

3 个“P1 lift + P2 拒绝”的场景（GM / 读侧 true\_random / bool store）不参与 parity；按 §6.4 的 Branch B 断言 `build_plan` 抛出 `Unsupported`。

### 6.3 边界矩阵

| 维度                   | 用例                            | 期望                                       |
| -------------------- | ----------------------------- | ---------------------------------------- |
| 1-D extent=64        | 整 chunk                       | `chunks=1, tail=0`                       |
| 1-D extent=100       | 尾块                            | `chunks=2, tail=36`；尾块带掩码                |
| 1-D extent=1         | 最小                            | `chunks=1, tail=1`；掩码/单元素                |
| 2-D `[8,64]`         | `test_2d_lifted` / `p.log:22-39` | `domain: i in [0,8) j in [0,64)`；`dep={0,1}`；两层 Parallel |
| 2-D 广播              | `c[i,j]=a[i,j]+bias[j]`        | `bias` 的 `dep={1}`；plan/emit parity        |
| 2-D 转置              | `c[i,j]=a[j,i]`               | `dep={0,1}`；plan/emit parity               |
| 3-D/4-D              | 矩形 N 维域                      | 递归 Parallel 链；plan/emit parity            |
| 符号 extent            | `T.dynamic` / `T.Parallel(n)` | P1 `LIFTED`；P2 D9 运行时 chunk 循环           |
| dtype=float16        | lanes=128                     | plan/emit 使用 f16 掩码宽度                    |
| dtype=int32          | lanes=64                      | 同上                                       |
| 多 store              | `c[i]=...; d[i]=...`          | plan 多 store；emit 按源序                    |
| 值级 select            | `T.if_then_else`              | plan `vsel`；emit `vcmp/vsel`             |
| `T.view`             | 局部视图                          | 适配器提取外层 For；parity 成功                    |
| 外层 `Bind` / `T.min`  | 尾块约束                          | `RestoreBindings` 内联；记录与原始 plan 的差异          |
| 多 nest 单 VF          | 两个 Parallel                   | 两个 `VSimdP2Input`；按 vf/nest 定位           |

### 6.4 负向矩阵

| 场景                    | P1 结果                  | 新接口                        | P2 行为                                     |
| --------------------- | ---------------------- | -------------------------- | ----------------------------------------- |
| 体内 `Bind`（C2/C4）     | `OPAQUE`（N4）           | `OPAQUE`/`FALLBACK`，无 stmt | 不调用                                       |
| 归约 `acc[i] += a[i,j]` | `ILLEGAL`（PR-A）        | `FALLBACK`，无 stmt          | 不调用                                       |
| 语句级 `if`              | `UNSUPPORTED`/`OPAQUE` | `OPAQUE`/`FALLBACK`，无 stmt | 不调用                                       |
| 内层 `T.serial`         | `UNSUPPORTED`/`OPAQUE` | 同上                         | 不调用                                       |
| `local.var`           | `OPAQUE`/`UNSUPPORTED` | 同上                         | 不调用                                       |
| rank mismatch         | `FALLBACK`（G5）         | `FALLBACK`，无 stmt          | 不调用                                       |
| 无 store               | `FALLBACK`（G6）         | `FALLBACK`，无 stmt          | 不调用                                       |
| GM 操作数                | `LIFTED`               | `LIFTED` + stmt            | `build_plan` 抛 `Unsupported`（Branch B）    |
| bool store            | `LIFTED`               | `LIFTED` + stmt            | `build_plan` 抛 `Unsupported`（Branch B）    |
| 读侧 true\_random       | `LIFTED`               | `LIFTED` + stmt            | `build_plan` 可能抛 `Unsupported`（Branch B）  |
| 非 0 下界                | `LIFTED`               | `LIFTED` + stmt            | `build_plan` 抛 `Unsupported`（R5，Branch B） |
| 同 VF 多 nest 且一个非法     | 整 VF 状态拖累              | 两个都 `FALLBACK`             | 不调用；记录 P1 粒度限制                            |

> 更新后的 P1 总账：21 个可 lift（18 个 P2 OK + 3 个 P2 REJ），0 个 ◐，45 个拒绝（37 个 P2 OK + 8 个 P2 REJ）。C2/C4 因体内 `Bind` 从“可分析”变为拒绝，不再属于本轮范围。

负向用例的验收：**不崩溃、无未捕获异常、诊断稳定**；Branch B 的 `Unsupported` 被测试显式捕获并断言。

### 6.5 报告格式回归

- `AnalyzeVSimdRegionReport` 在改动前后**逐字不变**；
- 对新接口的每个 `LIFTED`，报告中必须有对应的 `status=LIFTED` + `region` + `expanded` + `verify: OK` + `carrier`；
- 对每个 `FALLBACK/INTERNAL_ERROR/OPAQUE`，报告中必须有对应状态与 reason；
- 报告中 `domain/captures/accesses` 与 `VSimdP2Input.region` 的字段一致；
- 访问行必须包含 PR-D 的值元数据后缀：`shape=[...] axes=[...] dep={...}`（如 `R bias_ub[j] shape=[8,64] axes=[0,1] dep={1}`）；
- 2-D/N-D 的 `expanded` 必须是 `T.grid(...)` 的递归串行展开，且 `verify: OK`。

### 6.6 100% 通过标准

```bash
# 1) P1 55
pytest testing/ascend/transform/test_ascend_vsimd_memory_legality.py \
       testing/ascend/transform/test_ascend_vsimd_capture_scope.py \
       testing/ascend/transform/test_ascend_vsimd_region.py -q

# 2) P2 135 oracle
python tmp/run_poly_test.py

# 3) 新增全链路测试
pytest testing/ascend/transform/test_ascend_vsimd_p2_input.py -q
```

要求：

- 三组全部通过；
- 新增用例覆盖 §6.3/§6.4 全部维度；
- 无 crash、无未捕获异常、无静默跳过；
- 报告文本零回归；
- `p.log` golden 与 `test_simple_region_lifted` 全部断言通过。

***

## 7. 分阶段实施计划

| 阶段          | 任务                                                                                                                                    | 交付物                     | 验收标准                                           |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ---------------------------------------------- |
| **Phase A** | 分支收敛与构建基线：从 `Parallel_lowering` 切集成分支；cherry-pick P1 六个提交（或 merge `stable-parallel`）；解决 `__init__.py` 冲突；`USE_ASCEND=ON/USE_Z3=ON` 构建 | 集成分支、构建产物 | P1 55 + P2 135 基线全绿；Python 能 import P1 FFI |
| **Phase B** | 新增 `VSimdP2Input`、`BuildP2PlanInputs`、`AsParallelNest`；复用四个 P1 内部函数                                                                   | `vsimd_p2_input.{h,cc}` | 简单用例返回 `LIFTED` + `For(kParallel)`；代码审查确认无重复实现 |
| **Phase C** | FFI 注册 + Python 薄封装；跑通 `build_plan(stmt)`                                                                                             | FFI + `__init__.py`     | Python 拿到 Stmt 并成功生成 `VectorProgram`           |
| **Phase D** | `p.log` golden（1-D + 2-D）、`test_simple_region_lifted` / `test_2d_lifted`、plan/emit parity | 测试文件 | 逐维度断言通过；parity 无差异 |
| **Phase E** | 边界/负向矩阵、报告回归、全量测试 | 测试报告 | P1 55 + P2 135 + 新增用例 100% pass；无 crash |

***

## 8. 风险与应对

| 风险                                        | 触发条件                        | 应对                                                           |
| ----------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| `ForKind` 适配改变语义                          | 原域并非并行/有跨迭代依赖               | 断言 P1 `LEGAL` + `domain.size()==1` + 原域为 `kParallel`；不满足则不适配 |
| `SeqStmt + DeclBuffer` 导致 `build_plan` 失败 | `T.view` 场景                 | 适配器提取 `For`；专项 `T.view` parity 用例                            |
| P1 可 lift 但 P2 拒绝                         | GM/bool/true\_random/非 0 下界 | 归入 Branch B；测试显式断言 `Unsupported`，不视为 bug                     |
| 外层 Bind/assume 丢失                         | `T.min` 尾块、`T.assume`       | 先用无 Bind 基线；记录差异；后续给 `build_plan` 增加 `PlanOptions`           |
| 跨仓库/构建不一致                                 | P1 与 Python 指向不同扩展库         | Phase A 先验证 FFI import 与同一 build                             |
| 同 VF 多 nest 粒度                            | PR-A 状态按 VF 聚合              | 文档化限制；测试覆盖；不把合法 nest 误判为 P2 bug                              |
| 报告文本回归                                    | 改动误碰 P1 渲染                  | 报告 golden 逐字回归；新接口不改 P1 文件                                   |

***

## 9. 交付物与验收

### 9.1 交付物

1. `vsimd_p2_input.h/.cc`：`VSimdP2Input`、`BuildP2PlanInputs`、`BuildP2PlanInput`、`AsParallelNest`、`RestoreBindings`、FFI。
2. `tilelang/ascend/transform/__init__.py`：两个 Python 薄封装 + `__all__`。
3. `test_ascend_vsimd_p2_input.py`：golden、parity、边界、负向、报告回归。
4. 开发说明：复用关系、适配规则、Branch A/B/C 归类、已知边界。
5. 全量测试报告。

### 9.2 验收清单

- [x] 新接口对 `test_simple_region_lifted` 返回 `LIFTED` + `For(kParallel, i, 0, 64)`。
- [x] 新接口对 `test_2d_lifted` 返回 `LIFTED` + `For(kParallel, i, For(kParallel, j, ...))`，报告含 `T.grid(8, 64)` 与 `dep={0,1}`。
- [x] `build_plan` 对 1-D 返回 1 段 `VectorProgram`，字段符合 §6.1；对 2-D/N-D 以 parity 为准。
- [x] `emit_plan` 的 op 序列与 Python 原型一致。
- [x] `AnalyzeVSimdRegionReport` 输出与 `p.log` 维度一致且零回归（含 `shape/axes/dep` 后缀）。
- [x] “P1 lift + P2 OK”场景 plan/emit parity 通过；Branch B 场景按 §10.2 断言 `Unsupported`。
- [x] 负向场景返回预期状态/异常，无 crash（含 C2/C4 体内 `Bind`）。
- [x] P1 + P2 + 新增用例 100% pass（见 §12）。
- [x] 代码审查确认未复制 P1 的 G1–G6/展开逻辑。
- [x] 明确记录 C2/C4（体内 `Bind`）与 P1 拒绝场景不在本轮范围。

***

## 10. 附录：状态映射与路由

### 10.1 P1 状态 → 新接口 → P2

| P1 情况                                          | 新接口 status             | stmt             | P2 动作                                         |
| ---------------------------------------------- | ---------------------- | ---------------- | --------------------------------------------- |
| `CLOSED + LEGAL + !domain.empty()` 且 G1–G6 通过 | `LIFTED`               | `For(kParallel)` 链（1..N 层） | `build_plan`；成功 → P3；`Unsupported` → Branch B |
| `domain.empty()`（G3）                          | `FALLBACK`             | 无                | 不调用；legacy                                    |
| 体内 `Bind`（N4）                                 | `OPAQUE`/`FALLBACK`    | 无                | 不调用；legacy                                    |
| PR-A `ILLEGAL/UNPROVEN`                        | `FALLBACK`             | 无                | 不调用；legacy                                    |
| PR-B `OPEN/OPAQUE`                             | `OPAQUE`/`FALLBACK`    | 无                | 不调用；legacy                                    |
| `local.var`/if/serial/TileOp/Evaluate          | `OPAQUE`/`UNSUPPORTED` | 无                | 不调用；legacy                                    |
| 展开器自检失败                                        | `INTERNAL_ERROR`       | 无                | 不调用；LOG(WARNING)                              |

### 10.2 全链路分支

| 分支 | 条件 | 路由 | 本轮计数 |
| --- | --- | --- | --- |
| **A 正常** | P1 `LIFTED` + P2 `build_plan` 成功 | P3 `emit_plan` → legacy 处理剩余 | 18（A 15 + C1 + C3 + D2） |
| **B P2/P3 能力不足** | P1 `LIFTED` + P2 `Unsupported` | 跳过；记录诊断；迭代 P2/P3 | 3（GM / true_random / bool） |
| **C P1 不可向量化** | P1 `FALLBACK/ILLEGAL/OPAQUE` | 不进入 P2；legacy/scalar；记为 P1 gap | 45（37 P2 OK + 8 P2 REJ） |

> 本方案只实现 P1 当前（PR-D）覆盖的 A/B 路径；C 路径保持现有 legacy 行为，不在本轮扩展。C2/C4 因体内 `Bind` 属于 C。

***

## 11. 参考

- P1 报告入口：`vsimd_region.cc:363-405`（调用 `BuildMinimalRegion`）
- P1 region 构造（PR-D）：`vsimd_region.cc:91-157`
- P1 1-D 兼容壳：`vsimd_region.cc:159-166`（`BuildMinimal1DRegion`）
- P1 参考展开（N-D 递归）：`vsimd_region.cc:219-249`
- P1 `VSimdAccess`：`vsimd_region.h:37-42`
- P1 `VSimdRegion`：`vsimd_region.h:53-75`
- P2 `build_plan`：`tilelang/ascend/analysis/polyhedral/plan.py`
- P3 `emit_plan`：`tilelang/ascend/analysis/polyhedral/emit.py`
- 测试样例：`testing/ascend/transform/test_ascend_vsimd_region.py`（1-D + 2-D）
- 运行样例：`/home/cjl/workspace/p.log`（当前 2-D `[8,64]`）

***

## 12. 实现记录（方案 A 已落地）

### 12.1 文件

| 文件 | 内容 |
| --- | --- |
| `src/ascend/transform/polyhedral/vsimd_p2_input.h` | `VSimdP2Input`；`BuildP2PlanInputs` / `BuildP2PlanInput` / `AsParallelNest` / `RestoreBindings` 声明 |
| `src/ascend/transform/polyhedral/vsimd_p2_input.cc` | 实现 + `tl.transform.AscendBuildP2PlanInputs` / `AscendBuildP2PlanInput` FFI |
| `tilelang/ascend/transform/__init__.py` | 两个 Python 薄封装 + `__all__` |
| `testing/ascend/transform/test_ascend_vsimd_p2_input.py` | 33 个全链路/边界/负向/报告回归用例 |

P1 / P2 / P3 既有文件零改动；CMake 的 `src/ascend/transform/*.cc` glob 自动收录新源文件（需重新 configure）。

### 12.2 复用关系

`BuildP2PlanInputs` 只串联 P1 的四个内部函数，并做两件纯适配：

1. `AsParallelNest`：递归把参考展开的 `For(kSerial)` 链改回 `For(kParallel)`；`SeqStmt` 中唯一的 `For` 被提取，`DeclBuffer` 兄弟节点丢弃（store 仍在原位置引用 Buffer，P3 发射时 view 依旧在作用域）；并列多循环 / 无循环 / 非 serial 循环都归入 `INTERNAL_ERROR` + `LOG(WARNING)`。
2. `RestoreBindings`：仅对 `stmt` 中**自由出现**的 Bind 变量做内联（跳过值为 `Call` 的绑定，链式替换最多 8 轮），把 `ExpandVSimdRegionToLoops` 不重建的外层 `Bind` 放回 extent/index。

未复制任何 G1–G6、字段拷贝、store 重建或地址/值分析逻辑。

### 12.3 分支归类

- **Branch A**（P1 `LIFTED` + `build_plan` 成功）：1-D / 2-D / 3-D / 4-D、广播、转置、高维摊平、尾块、`extent=1`、符号 extent、f16 / int32、多 store、值级 select、`T.view`、多 nest 单 VF、外层 `Bind` 尾块。
- **Branch B**（P1 `LIFTED` + `build_plan` 抛 `Unsupported`）：GM 操作数、bool store、非 0 下界（R5）、stencil 尾块（`a_ub[i] = a_ub[i + bn]`）。
- **Branch C**（P1 拒绝）：体内 `Bind`（C2/C4）、归约（PR-A `ILLEGAL`）、`local.var`、opaque `Evaluate`/`T.view` 空体、同 VF 一个 nest 非法导致整个 VF `FALLBACK`。

### 12.4 已知边界

1. 外层 `Bind`/`T.min` 被内联：plan 结构一致，但 `emit_plan` 文本中尾块表达式是展开后的 `T.min(...)`，与原计划的绑定变量不同（测试记录为已知差异，不做 `structural_equal`）。
2. `assumptions` 仍未注入 `build_plan`（`PlanOptions` 留待后续）。
3. 无 store（G6）在当前前端不可达：`T.Parallel` 体内没有 store 时必然出现 `Evaluate`，会先被 PR-B 正向白名单判为 `OPAQUE`。
4. PR-A 按 VF 聚合，同一 VF 内一个 nest 非法会拖累兄弟 nest（`test_mixed_legality_marks_whole_vf`）。

### 12.5 测试结果

```bash
python tmp/run_poly_test.py \
  testing/ascend/transform/test_ascend_vsimd_memory_legality.py \
  testing/ascend/transform/test_ascend_vsimd_capture_scope.py \
  testing/ascend/transform/test_ascend_vsimd_region.py \
  testing/ascend/transform/test_ascend_vsimd_p2_input.py -q
# 88 passed（P1 55 + 新增 33）

python tmp/run_poly_test.py testing/ascend/transform/test_ascend_polyhedral_vectorize.py -q
# 113 passed
```

> `tmp/run_poly_test.py` 先摘掉 site-packages 的 `ScikitBuildRedirectingFinder` 再跑 pytest；直接 `pytest` 会把 `tilelang` 解析到另一份 checkout。
> 说明：本仓 P2 用例当前共 113 个测试函数（方案 §6.6 记为 135，应在其它分支/计数口径下），全部通过且 P2 代码零改动。

### 12.6 端到端调用

```python
from tilelang.ascend import transform as at
from tilelang.ascend.analysis.polyhedral import build_plan, emit_plan, Unsupported

for item in at.AscendBuildP2PlanInputs(main):
    if item["status"] != "LIFTED":
        continue
    try:
        out = emit_plan(build_plan(item["stmt"]))
    except Unsupported as err:  # Branch B
        record_branch_b(item, err)
```

