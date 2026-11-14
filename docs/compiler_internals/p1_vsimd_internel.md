
# P1 → P2/P3：VSimdRegion 接口规则

状态：PR-D staged（2026-10-27）。默认 pipeline 仍为 legacy；本文描述的是 P1 交给
后续阶段的 typed interface，不是已接入的 codegen 路径。

相关实现：

- `src/ascend/transform/vsimd_region.{h,cc}`
- `src/ascend/transform/analyze_vsimd_region.cc`
- `vsimd_intermediate_layer_design.md`（总体设计）

## 1. 入口与生命周期

P1 产出 `VSimdRegionLift`：

```text
BuildMinimalRegion(info, memory_status)
  -> VSimdRegionLift { lifted, internal_error, reason, region }
```

- 只有 `lifted=true` 的 region 才可交给 P2/P3；
- `internal_error=true` 表示 P1 的 region→expander 自校验失败，是 compiler bug，
  不是用户程序的 fallback；
- 当前 `BuildVSimdRegionCarrier` 是 debug-only AttrStmt，不是可执行 carrier；
- P2/P3 的 executable carrier 后续直接消费 `VSimdRegion`，不经过 TIR Call。

P1 的 gating：

- PR-A `LEGAL`；
- PR-B `CLOSED`；
- 至少一个 store；
- 无 `local.var`；
- `T.Parallel` 内不得有 opaque `Evaluate`/`call_extern`；
- 当前不允许 Parallel 内 `Bind`（见 §6）。

## 2. VSimdRegion 字段

| 字段 | 语义 | P2/P3 使用方式 |
| --- | --- | --- |
| `domain` | 有序 Parallel axes，含 var/min/extent | 定义 index space；不要当物理 lane |
| `accesses` | 源序 access 列表，元素为 `VSimdAccess` | 地址/内存规划、reference 校验 |
| `captures` | 去重 buffer + capture kind | region buffer 集合 |
| `outer_vars` | 非 domain 变量 + kind | scalar/thread capture |
| `shape_syms` | 自由 shape/stride 符号 | 符号 shape 上下文 |
| `local_views` | view buffer -> base buffer | alias/addressing |
| `layouts` | buffer -> Layout | layout-aware store / vsstb 决策 |
| `assumptions` | 去重后的 `tl.assume` | 算术契约 |
| `bindings` | 相关 Bind 定义（当前为外层 Bind） | 表达式上下文；不要假设 inner Bind |

`VSimdRegion` 不包含：

- 物理 layout / lane / mask / 指令 / 寄存器；
- 物理 footprint 或地址集合；
- 将 access map 物化后的 transpose。

## 3. Access 与 Value：必须区分

`VSimdAccess`：

```cpp
struct VSimdAccess {
  vsimd::AccessInfo access;          // buffer、indices、value、is_write
  std::vector<int> dep_axes;         // access.indices 实际用到的 domain axes
  std::vector<int> value_axes;       // v1: domain axes
  std::vector<PrimExpr> value_shape; // v1: domain extents
};
```

规则：

1. **`access` 是内存侧**：buffer 身份、逻辑 index map、store value。
2. **`value_*` 是值侧**：这个 access 消费/产生的逻辑值的 shape/axes。
3. **`dep_axes` 只由 access map 决定**：不要用 `value_shape`/`value_axes`
   推断 broadcast。
4. **v1 的 `value_shape`/`value_axes` 是占位常量**：
   - 对 domain-bound load/store，它们恒等于 domain 的 shape/axes；
   - 它们不是 value identity，也不是 access map；
   - 后续引入 value op / fusion 后，一个值会有自己的 shape/axes（例如
     reduce 后 value axes 变少），这些字段才会体现差异；
   - 在那之前，不要把“value shape = domain shape”当成普遍不变量写进 P2/P3。

### 例子

2D identity：

```text
domain      = [i: M, j: N]
store c[i,j]  value_shape=[M,N] value_axes=[0,1] dep={0,1}
load  a[i,j]  value_shape=[M,N] value_axes=[0,1] dep={0,1}
```

2D broadcast：

```text
load bias[j]  value_shape=[M,N] value_axes=[0,1] dep={1}
```

- broadcast 由 `dep_axes` 缺轴表达；
- `value_shape` 仍是 domain shape，不要把它当成 bias 的物理长度。

2D transpose：

```text
load a[j,i]  value_shape=[M,N] value_axes=[0,1] dep={0,1}
```

- transpose 由 access map 表达；
- 不要通过重排 `value_axes` 来表达 transpose。

## 4. Shape / Axes / Dep Axes 的边界

- `value_shape` / `value_axes`：逻辑值；
- `dep_axes`：access map 对 domain 的依赖；
- `access.indices`：逻辑 index map，rank 必须等于 buffer rank；
- 物理 shape/footprint/stride：不在 P1 region 中，由 P2 layout/物理计划决定。

因此：

- 不要用 `dep_axes` 作为向量化 lane 选择；
- 不要用 `value_shape` 作为 buffer extent；
- 不要把 `layouts` 合进 `value_*`；layout 是 buffer/context 属性。

## 5. 当前支持与 fallback

支持：

- 1..N 矩形 Parallel rank；
- 连续/stride/transpose access map；
- broadcast（`dep_axes` 缺轴）；
- 直接 computed index（如 `a[i*n+j]`）；
- 多 store（源序）；
- `T.view` local view（P1 expander 会补 `DeclBuffer`）；
- `tl.assume` / outer Bind 上下文；
- layout context 保留。

fallback / opaque：

- 2D 以上没问题，但 guard/control flow 尚未支持（PR-G）；
- Parallel 内 `Bind` 当前 OPAQUE；
- `local.var` / `T.alloc_var` OPAQUE；
- opaque `Evaluate` / `call_extern` OPAQUE；
- PR-A `ILLEGAL` / `UNPROVEN`、PR-B `OPEN` 不 lift；
- P1 自校验失败是 `INTERNAL_ERROR`，不是 fallback。

## 6. Inner Bind 的边界

当前决定：

- `t = i*n+j; c[t] = a[t]` **不支持**，结果为 OPAQUE；
- 原因：expander 尚未重建 Parallel 内的 Bind scope；如果放行会引用未声明变量；
- 直接写 `c[i*n+j] = a[i*n+j]` 支持；
- 外层 Bind（如 `tail = min(...)`）不影响，因为它在 Parallel 外。

未来若支持 inner Bind，需要：

1. PR-B 记录 inner Bind 及其 lexical scope；
2. expander 把 Bind 放回循环体；
3. P2/P3 看到的是完整、可重建的 region。

## 7. P2/P3 接入检查清单

- [ ] 只消费 `lifted=true` 的 region；
- [ ] 不把 `value_shape`/`value_axes` 当 access map 或物理 footprint；
- [ ] broadcast 用 `dep_axes` 判断；
- [ ] transpose 用 `access.indices` 判断；
- [ ] `layouts` 作为 buffer/context 读取，不写入 value；
- [ ] 保持逻辑 access 与 store 源序；
- [ ] 不把 P1 的 debug AttrStmt carrier 当可执行 IR；
- [ ] 后续 value op/fusion 接入后，重新检查 `value_shape`/`value_axes` 是否已从
      “domain 占位”变成 per-value 语义。

## 8. 后续演进

- PR-E：layout-aware store / FA-like direct store 结构；
- PR-F：P2/P3 `vsts`/`vsstb`/`vld`，自动 FA 端到端；
- PR-G：CF1–CF3 guarded store/merge/guarded load；
- value op / fusion 引入后：
  - 每个值有独立 shape/axes/dep_axes；
  - `VSimdAccess` 的 `value_*` 成为该 access site 的值描述；
  - 仍然禁止把 value 和 access 混用。
