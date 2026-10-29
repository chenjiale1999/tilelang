# P2/P3 拒绝点目录（40 条，逐条对拍契约）

> 来源：`tilelang/ascend/analysis/polyhedral/` 全部 `raise Unsupported`（实测 40 条）。
> 分布：`plan.py` 21 / `value.py` 9 / `body.py` 5 / `emit.py` 4 / `access.py` 1。
> 口径：**message 逐字一致**（测试用 `pytest.raises(Unsupported, match=...)` 做前缀/正则匹配），
> `diagnostics` **按列表顺序**比较。C++ 侧每条对应**有且仅有一条**分支，用 `polyhedral::Unsupported` 抛出。

图例：`[触发期]` = build（P2）期 / emit（P3）期；`[死代码?]` = 需在阶段四确认是否可达。

***

## A. `access.py`（1 条）

### A1 `access.py:117` — 访存秩不匹配

```
access rank {len(indices)} does not match buffer rank {len(offsets)}
```

触发：`BufferLoad/BufferStore` 的下标个数与 buffer 的 rank 不一致（含 `strides` 非空但长度不同的畸形 IR）。
C++ 目标：`LinearElementOffset`。

***

## B. `body.py`（5 条）

### B1 `body.py:104` — 原子调用

```
atomic op (T.atomic_add) is not supported yet: no vector+scalar atomic recipe
```

触发：`T.Parallel` body 里出现 `Evaluate(Call(...))` 且 `_call_tail` 含 `atomic`。
C++ 目标：`NormalizeBody`。

### B2 `body.py:132` — 同址 if/else 但单臂本身带谓词

```
masked branch merging is not supported: one arm of this same-address if/else is itself predicated
(e.g. a nested else-if), so there is no single select condition; rewrite it as two consecutive masked stores
```

触发：两臂各写 1 条且同 buffer 同下标，但其中一臂的 `StoreSpec.mask != None`（嵌套 if）。
C++ 目标：`NormalizeBody`。

### B3 `body.py:140` — 两臂写不同地址

```
divergent branch arms: an if/else whose arms write different locations cannot be vectorized (方案 §8.1)
```

触发：`IfThenElse` 两臂非空、写地址不同（或写数量 != 1:1）。
C++ 目标：`NormalizeBody`。

### B4 `body.py:143` — body 内嵌循环

```
nested {kind} loop inside T.Parallel is not affine-vectorizable (方案 §6): only directly-nested,
dependence-free serial loops are absorbed as axes
```

`{kind}` = `serial` / `parallel` / `unrolled` / `vectorized` 等（`_for_kind_name`）。
触发：`normalize_body` 撞到未吸收的 `For`（典型：两个并列 serial、serial 不干净）。
C++ 目标：`NormalizeBody`。

### B5 `body.py:151` — 其它语句

```
unsupported statement in T.Parallel body: {type(stmt).__name__}
```

触发：`normalize_body` 遇到非 `SeqStmt/BufferStore/BufferLoad/Evaluate/IfThenElse/For/Bind` 的节点
（如 `DeclBuffer`、`AllocBuffer`、`AttrStmt`、`AssertStmt`、`While`）。
C++ 目标：`NormalizeBody`（用 `GetTypeKey()` 尾段生成与 Python 类名一致的文本）。

***

## C. `value.py`（9 条）

### C1 `value.py:253` — 惰性条件里的 lane 均匀标量装载无法谓词化（`_uniform_guarded`）

```
lane-uniform load of '{buffer}' under a lane-varying lazy conditional cannot be predicated:
a scalar load has no masked form (方案 §1 Case II / P0-3)
```

触发：`_uniform_guarded` 在 `ctx.guard != None` 时遍历标量表达式，发现某个 `BufferLoad` 的访问
无法证明在所有 lane 内（`_access_provably_in_bounds` 为假）。
C++ 目标：`UniformGuarded`。

### C2 `value.py:399` — 不支持的值表达式

```
unsupported value expression: {node_name} ({expr})
```

触发：`build_value` 递归到未覆盖的节点类型（如 `tirx.Not` 之外的非 `Call` 布尔运算、
`Shuffle`、`Ramp`、`Broadcast`、`StringImm`、`tirx.Let` 等）。`{expr}` 是 `str(expr)`（含换行的完整文本）。
C++ 目标：`BuildValue`。

### C3 `value.py:517` — FloorDiv/FloorMod 除数不是编译期正奇数常量

```
{node_name} as a value operator needs a positive compile-time constant divisor (got {expr.b})
```

触发：值上下文里的 `FloorDiv/FloorMod`，且 `_const_int(expr.b)` 为 `None`、`<= 0` 或右操作数非 lane 均匀。
C++ 目标：`FloorDivMod`。

### C4 `value.py:519` — 非 2 的幂除数

```
{node_name} by non-power-of-two constant {divisor} is not strength-reduced yet:
the SIMD layer has no integer vdiv/vmod (方案 §5.4(a))
```

触发：同上，且 `divisor & (divisor-1) != 0`。
C++ 目标：`FloorDivMod`。

### C5 `value.py:572` — GM 操作数

```
global-memory operand inside T.Parallel (方案 §0 Case II): needs an explicit UB staging copy before the parallel nest
```

触发：`load_node` 里 `info.in_global == True`（`buffer.scope() == "global"`）。
C++ 目标：`LoadNode`。

### C6 `value.py:604` — 惰性条件里的 lane 均匀装载（`load_node` 分支）

```
lane-uniform load of '{buffer}' under a lane-varying lazy conditional cannot be predicated:
a scalar load has no masked form (方案 §1 Case II / P0-3)
```

触发：`kind == UNIFORM` 且 `guard != None`（`_access_provably_in_bounds` 证不出安全）。
C++ 目标：`LoadNode`。

### C7 `value.py:643` — 非常数 Δ 且索引向量不可构造

```
non-affine access with respect to axis '{axis}': the per-lane index vector is not constructible ({reason})
```

`{reason}` = `info.reason or 'unknown reason'`。
触发：`kind == TRUE_RANDOM` 的装载。
C++ 目标：`LoadNode`。

### C8 `value.py:656` — 值里的原子调用

```
atomic op inside T.Parallel is not supported (方案 §5)
```

触发：`_build_call` 的 `tail` 含 `atomic`。
C++ 目标：`BuildCall`。

### C9 `value.py:697` — 不支持的 intrinsic 调用

```
unsupported intrinsic call in T.Parallel body: {tail}
```

触发：`Call` 的 tail 不在 `if_then_else/select`、比较、布尔、二元、一元白名单内。
C++ 目标：`BuildCall`。

***

## D. `plan.py`（21 条）

### D1 `plan.py:61` — 循环下界非 0

```
{kind} loop '{var}' must start at 0 to be vectorized (min={min})
```

`{kind}` = `_for_kind_name(loop.kind)`（`parallel` / `serial` …）；`{min}` 是 `str(loop.min)`。
触发：`_require_zero_min` 无法证明 `min == 0`（含符号下界）。
C++ 目标：`RequireZeroMin`。

### D2 `plan.py:136` — 没有并行轴

```
no T.Parallel axis found
```

触发：`_split_nest` 收集完迭代域后没有任何 `is_parallel` 轴（如只吸收了 serial 链）。
C++ 目标：`SplitNest`。

### D3 `plan.py:238` — 主体无写

```
T.Parallel body has no store to vectorize
```

触发：`_prepare_segment` 里 `main_specs` 为空（body 只有裸读 / `Bind` / 空 `Evaluate`）。
C++ 目标：`PrepareSegment`。

### D4 `plan.py:312` — 所有轴都 true-random

```
no vectorizable axis: every store is true-random (its index cannot be built in the index-vector channel) on all axes
```

触发：`rank_vector_axes` 返回空（每根轴都至少有一个 store 的 `axis_class == TRUE_RANDOM`）。带 `diagnostics`。
C++ 目标：`PlanSegment`。

### D5 `plan.py:337` — 所有候选轴逐个回退都失败

```
no vectorizable axis: {err1}; {err2}; ...
```

`errors` 是逐轴的 `axis '{name}': {err}` 用 `"; "` 连接。带 `diagnostics`。
触发：`ranked` 非空但每个轴 `plan_for` 都抛 `Unsupported`。
C++ 目标：`PlanSegment`。

### D6 `plan.py:675` — 不支持的归约算子

```
unsupported reduction operator(s) {names} (方案 §2 Case II): only Add/Max/Min reductions are supported
```

`{names}` = 按字典序排序、`, `  连接的去重 `type(value).__name__`（空则 `<empty>`）。
触发：`_reduce_op` 在 store 取值里找不到 `Add/Max/Min` 之一（如 `Mul` 乘积归约）。
C++ 目标：`ReduceOp`。

### D7 `plan.py:854` — 单段回落时合并多段失败原因

```
{err} [{'; '.join(notes)}]
```

触发：多段路径记录了 `notes`，但单段路径在 `_split_nest/_plan_segment` 阶段就拒绝（典型「nested serial loop」）。
带 `diagnostics = [*notes, *(err.diagnostics or [])]`。
C++ 目标：`BuildPlan`（注意：这是**包装**既有错误，必须保持 `err.diagnostics` 前缀判断 `notes` 的去重逻辑）。

### D8 `plan.py:902` — pre/post 的向量轴不合法

```
a T.Parallel body with sibling statements around a serial loop can only be vectorized
along that serial axis or a single outer parallel axis (方案 §6 第二步)
```

触发：有 `pre/post` 且（向量轴不是那根 serial 轴）且（不是「并行轴 + serial 链恰好一层」）。
C++ 目标：`BuildPlanForAxis`。

### D9 `plan.py:977` — 沿 serial 轴向量化时 pre/post 非均匀

```
a statement {before|after} the serial loop must be uniform along vector axis '{axis}' (got {kind})
```

触发：`along_serial` 且某条 pre/post 的 `kind != UNIFORM`。
C++ 目标：`BuildPlanForAxis`。

### D10 `plan.py:988` — stencil 跨址读（并行轴）

```
read of '{buffer}' at a different address than any write in the same nest is a stencil along
vector axis '{axis}': vectorizing would read a different element than the scalar order
```

触发：`_cross_address_read` 在并行向量轴上找到「本段写过、地址不同、沿轴变化」的读。
C++ 目标：`CrossAddressRead`。

### D11 `plan.py:1004` — 私有标量沿并行轴变化且不可提升

```
store to private scalar '{buffer}' varies along vector axis '{axis}': vectorizing the outer axis
needs a promotable accumulator (a per-iteration private scalar cannot be shared by all lanes) (方案 §2 Case II)
```

触发：`store.kind == UNIFORM`、`!value.is_uniform`、`is_private_scope(buffer)`、`promote == None`。
C++ 目标：`BuildPlanForAxis`。

### D12 `plan.py:1017` — 归约段的 pre/post 非均匀

```
a statement {before|after} the serial loop must be uniform along the reduction vector axis '{axis}' (got {kind})
```

触发：`plan.reduce != None` 且某条 pre/post `kind != UNIFORM`（沿并行轴向量化时补的门禁，R7）。
C++ 目标：`BuildPlanForAxis`。

### D13 `plan.py:1033` — 并行轴上 lane 均匀写但取值变化（无向量形式）

```
store to '{buffer}' is lane-uniform along vector axis '{axis}' but its value varies with it:
a lane-varying value written to a lane-uniform address has no vector form (方案 §2 Case II)
```

触发：非归约段、`kind == UNIFORM`、`!value.is_uniform`、并行向量轴、`promote == None`。
C++ 目标：`BuildPlanForAxis`。

### D14 `plan.py:1038` — 被吸收 serial 轴上的部分归约

```
store to '{buffer}' is lane-uniform along the absorbed serial axis '{axis}' but its value varies with it:
a partial reduction mixed with element-wise stores has no vector form (方案 §2 Case II)
```

触发：同 D13，但向量轴是**被吸收的 serial 轴**（缺陷 A0）。
C++ 目标：`BuildPlanForAxis`。

### D15 `plan.py:1208` — 条件归约

```
conditional reduction into '{buffer}' is not supported yet: the cross-lane reduction mask only
carries the tail predicate, so the guard would be silently dropped (方案 §2 Case II / R3)
```

触发：归约 store 的 `mask != None`。
C++ 目标：`PlanReduce`。

### D16 `plan.py:1213` — lane 均匀写但取值不读目标 buffer

```
store to '{buffer}' is lane-uniform along axis '{axis}' but its value does not read '{buffer}':
only the accumulate form 'dst[...] = dst[...] <op> x' (op ∈ Add/Max/Min) is a reduction
(方案 §2 Case II); a lane-varying value written to a lane-uniform address is a
last-write-wins overwrite and has no vector form
```

触发：`_plan_reduce` 里 `!reads_buffer(value, buffer)`。
C++ 目标：`PlanReduce`。

### D17 `plan.py:1222` — 旧值不是归约算子的直接操作数

```
the old value of '{buffer}' must be a direct operand of the '{op}' reduction
(方案 §2 Case II); it cannot be scaled or nested inside another operation
```

触发：`_strip_reduce` 返回 `accumulated == 0` 或 `varying == None`。
C++ 目标：`StripReduce` / `PlanReduce`。

### D18 `plan.py:1273` — 旧值出现多次

```
the old value of '{buffer}' appears more than once inside the reduction expression
(方案 §2 Case II): only a single same-address old-value operand can be folded into the
reduction's initial value
```

触发：`_strip_reduce` 中 `accumulated > 1`（R4）。
C++ 目标：`StripReduce`。

### D19 `plan.py:1279` — 归约体在累加位置之外读目标 buffer

```
the reduction body reads the destination buffer '{buffer}' outside the accumulate position
(方案 §2 Case II): the old value must be a direct operand of the reduction operator
```

触发：剥离后 `varying` 仍读到目标 buffer。
C++ 目标：`StripReduce`。

### D20 `plan.py:1334` — store 元素类型不支持（谓词宽度推不出）

```
store element dtype '{elem}' of buffer '{buffer}' is not supported: {err}
```

触发：`_mask_bits(elem)` 抛 `ValueError`（典型 `bool`）。`{err}` 是 `ValueError` 的文本，C++ 侧必须复刻同一句。
C++ 目标：`RequireSupportedElement`。

### D21 `plan.py:1349` — 同段混位宽

```
mixed element widths in one vector segment ({detail}): every store of a segment must have the
same element bit width because the chunk stride is shared; split the nest or stage an explicit cast
```

`{detail}` = 按 bits 升序、`, `  连接的 `{buffer}:{elem} ({bits}bit)`。
触发：`_require_uniform_element_width` 里 `_mask_bits` 去重后 > 1。
C++ 目标：`RequireUniformElementWidth`。

***

## E. `emit.py`（4 条，需在阶段四确认可达性）

### E1 `emit.py:467` — 不能标量化

```
cannot scalarize vector value '{recipe}'
```

触发：`_Emitter.scalar()` 收到 `imm/uniform_expr/uniform_load/vbrc` 之外的 recipe。
**可达性**：`scalar()` 只在 pre/post 标量写、`vadds` 类、`vcmps`、`dup` 中调用，调用点都先检查
`is_uniform`；理论上不可达。C++ 保留为 `Unsupported`（不改成 `ICHECK`，保持行为一致）。

### E2 `emit.py:684` — 无发射器

```
no emitter for value recipe '{recipe}'
```

触发：`_build` 遇到未覆盖的 recipe。**可达性**：recipe 全部由 `value.py` 生成，闭合；理论不可达。
C++ 保留。

### E3 `emit.py:743` — 缺每 lane 索引向量

```
no per-lane index vector for a {kind} store
```

触发：`kind` 为 PERMUTE/PIECEWISE/INDIRECT/负步长 STRIDED，但 `store.index == None`。
**可达性**：`plan._plan_store` 对同一集合都填了 `index`；理论不可达。C++ 保留为防御。

### E4 `emit.py:757` — 循环外语句非均匀

```
a statement outside the serial loop must be uniform along the vector axis (got {kind})
```

触发：`_emit_outer_store` 收到 `kind != UNIFORM`。
**可达性**：build 期 D9/D12 已拦截；`_emit_reduce` 的 `store.reduce == None` 分支走同一函数，
此时 store 是「取值沿轴均匀」的归约段 store，`kind == UNIFORM`。理论不可达。C++ 保留。

***

## F. 统计与对拍方式

| 模块          | 条数     | 对拍方式                                                    |
| ----------- | ------ | ------------------------------------------------------- |
| `access.py` | 1      | 单测注入秩不匹配                                                |
| `body.py`   | 5      | 注入原子调用 / 嵌套 if / 分叉臂 / 并列 serial / `DeclBuffer`         |
| `value.py`  | 9      | 注入 GM 操作数 / 非 2 的幂除 / 数据相关下标 / 未支持 intrinsic / 惰性条件均匀装载 |
| `plan.py`   | 21     | 20 个既有 `pytest.raises` 用例 + 逐条注入                        |
| `emit.py`   | 4      | 确认死代码后保留（不新增用例，不改写为 `LOG(FATAL)`）                       |
| **合计**      | **40** | `pytest.raises(Unsupported, match=...)` 的 message 逐字一致  |

阶段六「异常注入」要求：每个 build 期拒绝点至少 1 个用例（`emit.py` 的 4 条按死代码处理，
在阶段四评审记录中标注依据）。
