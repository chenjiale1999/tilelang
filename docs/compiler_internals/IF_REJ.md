
## 一、`if` 目前有四条处理通道

| 形态 | 转换 | 代码位置 |
|---|---|---|
| 语句级 `if`（无 else / else 为空） | 写成 `spec.mask &= cond` 的**谓词化 store** | [body.py#L107-L121](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/body.py#L107-L121) |
| 语句级 `if/else`（同址、各 1 条写、臂内无谓词） | `tirx.Select(cond, then_v, else_v)` → **`vsel`（急切）** | [body.py#L123-L135](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/body.py#L123-L135) |
| 表达式 `T.if_then_else` | `vsel` + **装载守卫（惰性）** | [value.py#L545-L558](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/value.py#L545-L558) |
| 表达式 `tirx.Select` | `vsel`（两臂都急切求值，不额外加守卫） | [value.py#L312-L318](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/value.py#L312-L318) |

### 1.1 单臂 if → 谓词化 store

```
spec.mask = cond ∧ 外层 mask ∧ 臂内 mask
```

发射时 [\_store_mask](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py#L667-L682) 调 [predicate()](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py#L367-L397)：

- 逐 lane 条件（`a[i] > 0`）→ `vcmp/vcmps`，直接当 `vsts`/`vscatter` 掩码；
- **沿向量轴均匀**的条件（如 `if a[0] > 0`）→ `update_mask(Select(cond, lanes, 0), bits)`，整 chunk 同开同关（r2 用例）；
- `and/or/not` → `pand/por/pnot`，宽度按**数据元素**位宽（f16=16），不是固定 b32。

嵌套 `if` 在臂内是 `mask` 的 `And` 组合，天然支持；**臂内 `if/else`** 会走下面的合并通道，失败则拒绝。

### 1.2 同址 if/else → Select → vsel

合并需要同时满足：**同一 buffer + 下标结构相等 + 两臂各恰好 1 条 store + 两臂都没有自己的 mask**。满足时构造：

```python
StoreSpec(buffer, indices, tirx.Select(cond, then_value, else_value), mask)
```

`tirx.Select` 是**急切**语义，所以 [value.py](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/value.py#L312-L318) 对两臂都不加守卫，最后由 [emit.py#L646-L652](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py#L646-L652) 发 `vsel(t, f, predicate(cond))`。

### 1.3 表达式 `T.if_then_else` → vsel + 装载守卫

惰性内建会把条件传给分支上下文：

```python
then_ctx = guard = outer ∧ cond
else_ctx = guard = outer ∧ ¬cond
```

分支里的访存在 [value.py#L500-L508](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/value.py#L500-L508) 处判断：

- `_access_provably_in_bounds` 能证安全 → 丢弃守卫，照常发 `vld`（常见的 `select(cond, a[i], b[i])` 不退化成 gather）；
- 证不了 → `node.predicate = guard`，发射层退化成**带谓词的 `vgather2`**，未选中 lane 不访问内存；
- 分支里是**标量（lane 均匀）load** → 标量读没有掩码形式，只能拒绝（见 §三）。

---

## 二、`if` 相关的 REJ 判据

### 2.1 语句层（body.normalize_body）

| 判据 | 诊断 |
|---|---|
| 两臂写入**不同地址**（或某臂 store 数 ≠ 1） | `divergent branch arms`（[L137](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/body.py#L137)） |
| 合并时发现**某臂自身带谓词**（嵌套 else-if） | `masked branch merging is not supported`（[L129-L133](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/body.py#L129-L133)） |
| 分支内 `T.atomic_add` | `atomic op is not supported yet`（[L103-L104](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/body.py#L103-L104)） |
| 分支内还有未被吸收的 `For` | `nested serial loop …`（[L139-L143](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/body.py#L139-L143)） |
| `else` 只有 `pass`/`0` | 视为无 else（`_is_truthy_else`） |

> 注意：`if cond: pass else: c[i]=1` 会走“两臂不等”的分支 → 拒绝（其实可等价改写为 `¬cond` 的谓词化写，属保守缺口）。

### 2.2 值层（value.py）

| 判据 | 诊断 |
|---|---|
| 惰性分支里 **lane 均匀 load**，且地址不能证明在界内 | `lane-uniform load … cannot be predicated: a scalar load has no masked form`（[L171-L194](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/value.py#L171-L194)） |
| 分支内非 2 幂除法/取模、未知内建、atomic | 各自 `Unsupported` |

### 2.3 计划层（plan.py，与 mask 直接相关）

| 判据 | 诊断 |
|---|---|
| **条件归约** `if cond: s[0] += a[i]` | `conditional reduction … guard would be silently dropped`（R3，[L1109-L1117](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L1109-L1117)）——跨 lane 归约的掩码只承载尾块谓词 |
| 逐 lane 变化的值写到 **lane 均匀地址**，且不是归约/可提升累加器 | `has no vector form`（[L978-L994](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L978-L994)） |
| 私有标量沿并行轴变化且不可提升 | `needs a promotable accumulator`（[L949-L960](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L949-L960)） |
| 带谓词的 store 不参与 `accumulator_candidate` | [model.py#L142-L143](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/model.py#L142-L143)（返 None → 可能触发上一条拒绝） |

### 2.4 实测（我构造的 10 个探针）

| 形态 | 结果 |
|---|---|
| 同址 if/else | OK：`vld x, vld y, vcmp, vsel, vsts` |
| 异址 if/else | REJ `divergent branch arms` |
| else-if 嵌套 | REJ `masked branch merging` |
| 单臂两 store | OK：两条 `vsts` 共用同一个谓词（只算一次 `vcmps`） |
| 惰性 + 均匀 load 不可证界 | REJ |
| 惰性 + 逐 lane load 不可证界 | OK：`vgather2` 带谓词 + `vsel` |
| 条件归约 | REJ R3 |
| 并行轴 WAW `c[i]=a[i]; c[i+1]=b[i]` | **OK**（见 §四） |
| 语句级 `if i>=32: c[i]=a[i-32]` | **OK 但发出 `vld a[-32]`（越界读）** |
| 惰性同形态 | OK：`vgather2(a[-32], mask=i>=32)`，只访问 `a[0..31]` |

---

## 三、数据依赖分析是怎么做的

判据全部建立在**仿射访问函数 + 有限差分**上，不做完整依赖多面体（没有距离向量、GCD 测试），是**保守充分条件**：

### 第 1 层：能否吸收一根轴（`_serial_is_clean`，[plan.py#L442-L495](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L442-L495)）

对 serial 循环收集全部 store/load，计算 `linear_element_offset`，对循环变量做 `finite_difference`：

1. **不变量写**（Δ==0）：必须是 `dst = dst op x` 的累积写**且** `x` 沿该轴确实变化，否则（`cnt[0]=a[i,j]`、`cnt[0]+=1`）→ 拒；
2. **WAW**：同一 buffer 两条 store 的 offset 表达式不 deep-equal → 跨迭代可能别名（`c[i,2j]` vs `c[i,j]`）→ 拒（R6）；
3. **RAW 跨迭代**：写过的 buffer 被读，且读地址不**结构等于**任一写地址（stencil `t[i,j-1]`）→ 拒；同址读（`s[i] += a[i,j]`）是 RMW，放行。

### 第 2 层：向量轴上的依赖（`_cross_address_read`，[plan.py#L1211-L1226](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L1211-L1226)）

选中的**并行**向量轴上，凡“本 nest 写过的 buffer 在别的地址被读” → 拒。因为向量化是“所有 lane 同时读写前向量”，与标量顺序不同。

### 第 3 层：归约 / 累加器的数据流（`_strip_reduce` + `accumulator_candidate`）

只有同时满足以下条件才算合法归约（[plan.py#L1135-L1189](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L1135-L1189)、[model.py#L127-L165](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/model.py#L127-L165)）：

- 目标地址沿轴 Δ==0（lane 均匀）；
- 取值在**算子脊**上直接读同址旧值，且**恰好出现 1 次**（R4）；
- 算子 ∈ {Add, Max, Min}；
- store **无谓词**（R3）；
- 累加器还要求：地址沿所有 inner 轴不变、该 buffer 在本段只有这一条主写、同层其它写不读它的其它地址、同址 pre 写不带谓词、私有标量必须存在逐 lane 消费者。

### 判据的构造原则

1. **语义义务 → 可判定谓词**：每个义务（lane 独立、跨 lane 只允许显式归约、RMW 必须同址）都落成对 offset 差分或值 DAG 的结构判定；
2. **build 期拒绝**：`normalize_body` → `_prepare_segment` → `_build_plan_for_axis` 全链路只在 build 期抛 `Unsupported`，保证 “build 成功 ⇒ emit 成功”；
3. **逐轴回退**：`rank_vector_axes` 排序后逐轴 try，失败记录 `axis 'x': reason -> fallback` 再试下一个；全部失败才抛；
4. **宁拒勿错**：结构相等用 `expr_deep_equal`/字符串，不做别名推断，因此 `c[i*1]` 与 `c[i]` 会被判不同址而拒绝。

---

## 四、这套判据目前有三个缺口（实测确认）

### 4.1 语句级 if/else 合并到 **急切 Select** 后，臂访问失去越界守卫（最严重）

```
if i >= 32:  c[i] = a[i - 32]
else:        c[i] = 0.0
```

源程序只在 lane 32..63 读 `a[0..31]`；合并成 `tirx.Select` 后两臂都被急切求值，发出：

```python
vs0_0 = vld(access_ptr(a[-32], 64, 1))     # 读 a[-32..31]，越界 32 个元素
vs0_4 = vsel(vs0_0, vdup(0), vcmp(vci, 32, "ge"))
```

而表达式级 `T.if_then_else` 同形态会发**带谓词的 gather**，只访问有效 lane。原因是 [value.py#L312-L318](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/value.py#L312-L318) 对 `tirx.Select` 不注入 `guard`，只有 `if_then_else` 注入。

**建议**：语句级合并时也把 `cond` 作为两个臂的 guard 传下去（`_with_guard`），或至少对“臂访问不可证在界内”的合并直接 REJ。这条应补进 §2.1 的判据表。

### 4.2 并行轴的 WAW 没有判据

`c[i]=a[i]; c[i+1]=b[i]` 被接受，按源序发两次整向量 `vsts`，结果恰好等于 lane 顺序执行（`c[j]=b[j-1]`）。这与 serial 轴上的 R6 拒绝形成**不对称**：serial 轴向量化会重排迭代，last-write-wins 不保；并行轴每个 store 是一条整向量、按源序发射，等价于 lane 序。这个不对称是对的，但**没有显式判据/注释**，建议在 `_build_plan_for_axis` 里写明前提（“同 buffer 多写仅允许在并行轴上，且按源序发射”）。

### 4.3 `if cond: pass else: ...` 被误判为 divergent

`else` 有内容、`then` 为空时直接走 `divergent branch arms`。可以等价改写成 `¬cond` 的谓词化写，属保守拒绝，可补。

---

需要我按 §4.1 → §4.2/4.3 的顺序补判据和回归用例吗？§4.1 是唯一会产生越界访问的实质问题。




















​          
​          
## 与 commit `a796388f`（"add：支持向量轴合并"）的差异总览

工作区相对 HEAD：**13 个文件，+4226 / −706 行**。核心包 `polyhedral/` 中 emit.py 改动最大（1172 行），plan.py 634，value.py 348；测试 `test_ascend_polyhedral_vectorize.py` 从 **89 → 108 个用例**，`test_drive_plan.py` 从 **56 → 64 个场景**（实测 53 OK / 11 REJ / 0 ERR）。

---

## 一、能力提升（HEAD 做不到 → 现在支持）

| 能力 | HEAD | 现在 | 关键实现 |
|---|---|---|---|
| **跨 lane 归约收尾** | 全 lane 合并 + `vld(BRC)` 取旧值 | 收窄到 lane 0（`PAT_VL1`）+ 标量广播 `vadds/vmaxs/vmins` | [lane0_mask](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py#L310-L328)、[_emit_reduce_tail](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py#L1293-L1313) |
| **长归约 / 符号 extent 归约** | 只支持编译期展开；符号 extent 在 plan 期直接拒绝 | 两条策略：≤4 chunk 展开+树合并；否则真实循环 + `local.var` 累加器；符号 extent 用 `update_mask(min(...))` | [_REDUCE_UNROLL_LIMIT](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py#L95-L101)、[_emit_reduce_unrolled](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py#L1335-L1380)、[_emit_reduce_loop](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py#L1383-L1437) |
| **外层并行轴 + 兄弟语句** | 只允许沿 serial 轴向量化（否则直接拒） | `for chunk: pre; for serial: main; post`，relax 到“serial 链恰好一层” | [_emit_parallel_axis_parts](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py#L1015-L1202)、[plan.py#L879-L886](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L879-L886) |
| **私有标量累加器（S2）沿并行轴** | 无法提升，只能拒 | 提升成 `local.var` 向量寄存器，循环后写回 | [accumulator_candidate](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/model.py#L127-L177) |
| **惰性条件分支的装载守卫** | 无守卫：`if_then_else` 分支里的访存无条件读 | `guard = cond ∧ outer`，不可证在界内时退化为带谓词 `vgather2`；均匀标量读不可谓词化则拒绝 | [value.py#L545-L558](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/value.py#L545-L558)、[_access_provably_in_bounds](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/value.py#L228-L261) |
| **逐操作数元素类型** | 整段一个 `vector_dtype`，bool/int 混用时类型与掩码错位 | 每个值节点按**被访问 buffer** 的元素类型派生 `_vec_of`；同段不同**位宽**在 build 期拒绝 | [\_vec_of](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/value.py#L87-L95)、[_require_uniform_element_width](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L1243-L1258) |
| **按 lane 利用率选轴** | 归约轴**无条件**排第一（`return [axis]`），`(64,8)` 形态永远选 8/64 lane | 打分乘 `active/lanes`，小 extent 的 serial 轴不再压制满 lane 的并行轴 | [_active_lanes](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L568-L578)、[_axis_score](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L580-L585) |

---

## 二、新增的正确性判据（HEAD 全部缺失）

| 编号 | 问题 | 位置 |
|---|---|---|
| **R1** | `pand/por/pnot` 掩码宽度固定 b32 → f16/int8 尾块只覆盖一半 lane | `_PREDICATE_MASK_BITS` 32→8；`predicate_combine(lhs,rhs,bits)` 按数据宽度 |
| **R2** | 循环不变 bool 条件被当数据 `vdup` → emit `ValueError` | store 条件统一走 [predicate()](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py#L367-L397) |
| **R3** | 条件归约静默对全 lane 求和 | [_plan_reduce](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L1106-L1117) |
| **R4** | 旧值出现两次被折叠成一次，丢一份 | `_strip_reduce` 返回剥离次数，≠1 拒绝 |
| **R5** | 非 0 下界被静默改成从 0 开始 | [_require_zero_min](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L42-L51) |
| **R6** | serial 轴上跨迭代写写别名（`c[i,2j]` vs `c[i,j]`） | [_serial_is_clean](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L481-L495) 新增 WAW 检查 |
| **R7** | 并行轴归约 + 非均匀 pre：build OK、emit 崩 | [plan.py#L962-L978](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L962-L978) |
| **R8** | scatter/标量写/回写后，写后读转发仍命中陈旧值 | `_store_gen` 代数 + `note_write` 同时作废 `_forward`/`_carry` |
| **D13 配套** | 并行轴上的跨位读（stencil）未检查 | [_cross_address_read](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L1211-L1226) |

---

## 三、性能优化（从 log 可直接对比的 IR 变化）

1. **累加器寄存器提升**：`s3_serial_reduce` 从 HEAD 的「每轮 `vld s` + `vadd` + `vsts s`」变为「1 次 `vld` + 8 次寄存器 `vadd` + 1 次 `vsts`」。
2. **归约收尾**：去掉 BRC 广播装载和 64 lane 加法，只算 lane 0。
3. **短归约树合并**：4 条 `vcadd` 链两两合并，跨 lane 串行延迟 4→2。
4. **长归约不展开**：指令段从 `O(chunks)` 降为 `O(1)`。
5. **多段 prologue 共享**：`p_serial_reduce_then_use` 的 `pset`/`ramp` 从 2+2 降为 1+1。
6. **段间寄存器直通**：段 1 的 `vld s` 消失，直接用段 0 的累加器寄存器（严格门禁：单 chunk、无尾块/谓词、同一向量轴）。
7. **LICM**：`vdup`、地址分解、只读 buffer 加载提到 chunk/inner 循环外（[_split_loop_invariant](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py#L877-L905)、[_scalar_hoistable](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py#L546-L562)）。
8. **chunk 骨架简化**：`chunks==1` 展开成直线；常量尾块由 `for range(1)` 改为常量 lane 基址的直发块；符号 extent 由「每 chunk 都带运行期掩码」改为「`E//lanes` 个满块 `vld` + `if E%lanes>0` 的条件尾块」。
9. **死写消除**：`s[i]=0` 折叠为累加器初值（[_fold_identity_init](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L1040-L1067)）；`c[i,0]=1.0` 被主体覆盖时删除（[_drop_dead_pre_stores](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L1070-L1103)）。
10. **`vector - scalar` → `vadds(-scalar)`**，省一条 `vdup`；新增 `_SCALAR_TO_VECTOR_BINARY` 处理“标量操作数其实已提升成向量”的情形。
11. **浮点归约单位元改为 `±inf`**（打印干净、语义精确），整型仍用 `min/max_value`。

---

## 四、结构重构（重复代码 / 特判收敛）

| 重构 | 之前 | 现在 |
|---|---|---|
| chunk 骨架 | 逐元素 / 并行轴三段 / 归约循环各维护一份，尾块修正要改三处 | 统一 [_strip_mine](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py#L754-L806) |
| 规划主流程 | 单段 `build_plan` 与多段 `_build_segment` 各写一遍「展平→排序→逐轴回退」 | 统一 [_plan_segment](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py#L252-L326) |
| 并行轴吸收 | `_split_nest` / `_try_build_program` 各一份 | `_absorb_parallel_chain` |
| 累加器判据 | emit 自带 `arith.Analyzer` 重算 | 单一来源 `accumulator_candidate` + `StorePlan.promote` |
| 谓词体系 | `load_mask` + 单独的 predicate 组合 | `predicate` / `predicate_combine` / `ramp_mask` / `load_guard` / `lane0_mask` 职责分离 |
| 死参数 | `build_value(is_global_ok)`、`_ValueContext.lanes/vector_dtype`、`rank_vector_axes(stores)`、`_strip_reduce(vector_axis)`、`AccessClass.is_index_vector` | 全部删除 |
| 多段提层 | `axes=parallel_axes` 全提，向量轴被吃成 lane 维后又包一层 → 64 次空转 | 只提「第一根被某段用作向量轴的并行轴」之前的轴；展平段不留前缀 |
| 诊断 | 顶层一句泛化 “nested serial loop” | `_record_serial_reason` 带具体循环/原因；逐轴失败原因成功时也保留（`-> fallback`）；打分带 `active=n/lanes` |

---

## 五、测试与文档

- `test_ascend_polyhedral_vectorize.py`：89 → 108（新增 R1–R8 回归、长归约/长归约尾块、段间共享与直通门禁、多段拆分等）。
- `test_drive_plan.py`：56 → 64（新增 `r_long_reduce`、`r_long_reduce_tail`、`r1`–`r7`；删除过期的 `d10_non_reduction_tail_overread`）。
- 三份文档重写/扩充：`polyhedral_design.md`、`polyhedral_design_external.md`、`docs/compiler_internals/polyhedral_parallel_scenarios.md`（64 场景索引、R/C 行、门禁实测）。

---

## 六、遗留问题（非本次引入，但当前实现仍存在）

1. **语句级 if/else → 急切 `Select` 不做越界守卫**：`if i>=32: c[i]=a[i-32]` 会发 `vld a[-32]`，读越界 32 个元素；表达式级 `T.if_then_else` 同形态是安全的（带谓词 gather）。
2. **并行轴 WAW 无判据**：`c[i]=a[i]; c[i+1]=b[i]` 被接受并按源序串行化；serial 轴同形态被 R6 拒绝，两者不对称（有语义理由，但未写入判据）。
3. **`ana.md` 仍是中间快照**：写的是 `96 passed / 50 OK / 7 REJ`，与当前 `108 / 53 / 11` 不符。
4. 文档 §3 B7 仍把 `d1_product_reduce`、`d2_plain_overwrite` 标为 REJ，实际已通过轴回退成功。































