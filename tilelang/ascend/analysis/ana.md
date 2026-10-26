        
          
## 0. 基线（本轮实测）

- 实现：[polyhedral/](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral) 包，10 个模块
- 模块：`primitives` / `access` / `body` / `value` / `model` / `plan` / `emit` / `render` / `driver` / `__init__`
- 测试：`96 passed`（0 xfail）
- 复现：`python tilelang/ascend/analysis/test_drive_plan.py --tir &> log_poly.log` → **50 OK / 7 REJ / 0 ERR**，log 2598 行
- ruff：`check` / `format --check` 全绿

> 本轮已修：P0-1（`scalar - vector` 符号翻转）、P0-2（同段混合元素位宽 + load 节点 dtype）、
> P0-3（`if_then_else` 惰性分支的 predicated load）、P0-4（累加器提升推广到 inner parallel 轴）、
> 以及三份 strip-mine / 两份 flatten / 两份累加器判据的收敛（顺带修掉长归约 tail==0
> 只跑 `chunks-1` 块的漏算）。

---

## 1. 设计质量问题清单

### A. 正确性 / 契约残余（按严重度）

| 编号 | 问题 | 证据 | 影响 | 建议 |
|---|---|---|---|---|
| **A1** | **`bool` 作"数据"操作数仍是 build OK → emit `ValueError`**（唯一会抛非 `Unsupported` 的路径） | `d[i] = T.Cast("float32", c[i])`（`c: bool`）：build OK，emit 抛 `ValueError: Cannot infer default mask width for dtype 'boolx256'`。根因：`build_value` 的 `Cast` → `vcvt`，`_build` 里 `_mask_bits(src.dtype)` 遇到 bool 向量；而 [`_require_supported_element`](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py) 只校验 **store** 侧 | 契约裂缝（D11 同族），调用方无法优雅兜底 | 在 `build_value` 对「bool 作为数据操作数」（Cast / 算术 / FloorDiv 等）显式 `Unsupported`；bool 只允许出现在**谓词**位置 |
| **A2** | **`ValueNode.dtype` 统一取"store 的元素类型"**，而非被访问 buffer 的元素类型 | [`load_node`](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/value.py) 一律用 `ctx.vector_dtype`；`bool` 载入节点因此声称 `float32x64`，实际 TIR 是 `boolx256` | 混合元素类型时节点的类型/掩码宽度与实际寄存器不一致 | ✅ **已修**（P0-2）：`load_node` / `_elementwise` / `vsel` 都按**自身元素类型**派生向量 dtype；同段不同**位宽**在 build 期显式拒绝 |
| **A3** | A0 门禁只覆盖**被吸收的 serial 轴**；并行向量轴上的"lane 均匀地址 + 变化取值"仍是 `ONEPT` 取 lane 0 | [`_build_plan_for_axis`](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/plan.py) | 该形态在 `T.Parallel` 下写同一地址本身就跨迭代竞争（语义未定义），因此视为边界而非缺陷 | 文档写明；若将来要求"确定性 last-write-wins"，需额外拒绝 |
| **A4** | 多段路径失败时，顶层消息是泛化的 "nested serial loop"，真实原因（A0 / 段级拒绝）只在 `diagnostics` | `log_poly.log` 中 `d14_partial_reduce` 的 `[REJ]` 首行是 nested serial loop，下一行 diag 才是 partial reduction | 诊断可读性 | 顶层 `Unsupported` 消息优先使用第一条段级原因 |
| **A5** | `_require_supported_element` 只查 store；**load/谓词** dtype 未校验 | 同 A1 | 见 A1 | 与 A1 一并处理 |

### B. 性能

| 编号 | 问题 | 证据 | 影响 | 建议 |
|---|---|---|---|---|
| **B1** | **符号 extent 时所有 chunk 的连续装载都退化成带谓词的 `vgather2`**，且每 chunk 执行一次 `min(...) + update_mask` | `log_poly.log` `d9_symbolic_extent`：`for chunk in range((n+64-1)//64)` 内 `update_mask(T.min(n-chunk*64,64))` + `vgather2` | 可向量化但比 `vld` 慢；符号形状场景的性能损失 | 若能证 `extent >= lanes`，前 `floor(E/L)` 个 chunk 用 `vld` + 全掩码，仅最后一块走掩码 gather（等价于常量路径的 peel，但边界需运行时判断） |
| **B2** | 尾块拆块后**每个块独立建 `_Emitter`**，可能的等价 `pset` 重复绑定（`mask32` + `masktail_32`） | [`_emit_elementwise_chunks`](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py) | 代码体积/指令数轻微增加 | 可让 tail 块复用整块的 prologue（传入同一个 `all_mask` 变量） |
| **B3** | 归约的 chunk 循环**编译期展开**，代码体积 ∝ `extent/lanes` | [`_emit_one_reduce`](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/emit.py) | 大 extent 归约的编译产物膨胀 | 已知设计约束（纯 SSA 无 PHI）；长期可引入显式 PHI/`scf.for` 承载累加器 |

### C. 可维护性

| 编号 | 问题 | 证据 | 建议 |
|---|---|---|---|
| **C1** | **索引向量通道分散在三处**：`load_node`（选 `gather`）/ `_plan_store`（建 `index`）/ `_emit_store`（选 `vscatter` vs 斜坡），改一处易漏 | D8 正是这种分散的产物：修负步长要同时改 3 个函数 + `classify_access` | 抽一个 `_index_channel(info)` 判定（"是否需要绝对索引向量"），三处共用一个判据 |
| **C2** | `Unsupported` 无机器可读 error code | 无消费者；改 20+ 处 `raise` | 暂缓（已记录在文档） |
| **C3** | 单文件 2457 行，分析（SCoP/Δ）、规划、发射三类职责混在一起 | `_build` 长 if-chain、`_emit_*` 多 emitter | ✅ **已拆分**：`polyhedral/` 包按依赖分层为 `primitives / access / body / value / model / plan / emit / render / driver`；依赖单向向下，公共 API 由 `__init__` 统一再导出。表按语义归属放置：ISA/算子事实与硬件常量集中在 `primitives.py`，`AccessClass` 留在 `access.py`，`_AXIS_WEIGHT` 留在 `plan.py`，计划数据结构单独放在 `model.py` |
| **C4** | 类型标注小瑕疵：`_render_single_plan(extra_diagnostics: tuple[str, ...])` 实际收到 `list` | [render.py](file:///home/cjl/workspace/tilelang/tilelang/ascend/analysis/polyhedral/render.py) | ✅ **已修**：改为 `Sequence[str]` |
| **C5** | 归约/索引通道的"判据重复"风险：`_axis_class` 的 stride 语义在 `classify_delta`/`load_node`/`_emit_store` 各解释一次 | 同上 C1 | 同 C1 |

### D. 文档

| 编号 | 问题 | 证据 | 建议 |
|---|---|---|---|
| **D1** | 绝对行号随拆分全部失效 | 拆分前的 `polyhedral.py` 行号（`_reduce_op` / `_is_reduce_axis` / `_plan_reduce` / `is_uniform` / `_reduce_identity` / `normalize_body` 等） | ✅ **已修**：`ana.md` / `polyhedral_design*.md` 改为指向具体模块文件 + 函数名，不再写绝对行号 |
| **D2** | **§4.1 能力矩阵没有收录本轮新增的 5 类能力**（负步长 / 符号 extent / 非归约尾块 / lane 不变间接地址 / 部分归约拒绝+回退），只在 §4.2 缺陷表出现 | 场景表最后一行仍是 "诊断（改进 7）" | 在能力矩阵补 5 行，否则读者会以为这些仍不支持 |
| **D3** | 文档写"约 2454 行"，实际 2457 | — | ✅ **已修**：拆分后改为「10 个模块共 2678 行」并给出最大模块行数 |

---

## 2. 支持场景清单

> "支持"= `build_plan` + `emit_plan` 均成功；括号内为 drive/pytest 用例。

### 2.1 逐元素 / 分支 / 多语句

| 场景 | 发射形态 | 用例 |
|---|---|---|
| 连续拷贝 `c[i]=a[i]` | `vld + vsts(NORM)` | `c0_copy` / `test_copy` |
| 标量广播 `c[i]=a[0]` | `vdup + vsts` | `c0_broadcast` / `test_scalar_broadcast` |
| 单臂 `if` | 谓词化 store（`vcmp*` 作 mask） | `c1_single_arm` |
| 双臂 `if` 同址 | `vsel` | `c1_ifelse` |
| 三元（循环不变条件） | 全 lane 谓词 `update_mask(cond?lanes:0)` | `c1_ternary` |
| 三元（逐 lane 条件） | `vcmp` + `vsel` | `c1_ternary` |
| 维度不一致 `g[i]-g[j]` | Δ=0 侧 `vdup` 广播 | `c2_rank` |
| 多语句 / 标量别名 | 逐 store 有序发射，`Bind` 内联 | `c3_multi` / `c3_scalars` |
| `//`、`%` 2 的幂常量除数 | `vshrs` / `vand` | `c3_value_pow2` |
| 非 2 的幂除数 | 该轴被拒 → **轴回退**到次优轴 | `c3_im2col` |

### 2.2 归约（§2 Case II）

| 场景 | 发射形态 | 用例 |
|---|---|---|
| `+=` / `max` / `min` | `vcadd/vcmax/vcmin` + `ONEPT` 回写 | `c2_reduce` |
| 累加写 `dst op= x` | 末尾 `BRC` 读旧值并入 | `c2_reduce` |
| 尾块（`extent % lanes != 0`） | 尾块带谓词装载 + `vsel` 单位元 | `c2_reduce_tail` |
| 一个 nest 多条归约 / 混合算子 | 逐 store 独立累加链 | `d6_two_reduces` / `d6b_sum_max_reduce` |
| 归约体含 lane 均匀加数 | 加数保留在 `varying` | `d4_uniform_addend` |
| `int32`/`f16` 的 max/min | 单位元按 dtype 派生 | `d3_int_max_reduce` / `d3_f16_max_reduce` |
| 归约段内均匀广播写 | 一次 `ONEPT`，不参与归约 | `test_reduce_segment_with_uniform_broadcast_sibling` |

### 2.3 访存形态（§4）

| 场景 | 发射形态 | 用例 |
|---|---|---|
| 高维摊平 `h[i//16,i%16]` | 化简后 Δ=1 → `vld` | `c4_flat` |
| 步长 `a[2*i]` | `vgather2` + 步长斜坡 | `c4_strided` |
| **负步长 `a[126-2*i]`（本轮新增）** | **索引向量 `vgather2`**（读+写两侧） | `d8_negative_stride` / `d8b_negative_stride_store` |
| 逆序 `a[N-1-i]` | `PERMUTE` → `vgather2` | `c4_reverse` |
| 分块置换 `cs[i//4,j//4,i%4,j%4]` | `PIECEWISE` → `vgather2` | `c4_tiled` |
| 间接 `a[b[i]]` | `INDIRECT` → `vgather2`（索引来自寄存器） | `c_gather` |
| **沿轴不变的间接地址 `a[b[0]]` / `c[b[0]]=1.0`（本轮新增）** | **`UNIFORM` → `vdup` / 一次 `ONEPT`**（支持部分向量化） | `d11_indirect_pre` / `d11b_lane_invariant_indirect_load` |
| 表达式下标（UB） | 线性化+化简后连续/均匀 | `test_expression_subscript` |

### 2.4 T.Parallel 内的循环结构（§6）

| 场景 | 发射形态 | 用例 |
|---|---|---|
| 直接内嵌干净 serial（S1 拷贝/谓词、S3 归约） | 吸收为候选轴并选为向量轴 | `s1_serial_copy` / `s1_serial_predicated` / `s3_serial_reduce` |
| 一个 serial + 前置语句（M2） | `pre` 在 chunk 循环外 | `m_serial_after_init` |
| 一个 serial + 后置语句（M3） | `post` 在 chunk 循环外 | `m_serial_before_post` |
| 前置 + serial + 后置（夹心） | 顺序 `pre → 主体 → post` | `m_serial_middle` |
| 独立 store 组 + serial（M1） | 同上 | `m_serial_indep_group` |
| 私有标量累加器（S2） | 前置初始化 + 后置写回 | `s2_serial_scalar` |
| ≥2 个并列干净 serial | `VectorProgram` 多段、源序发射、共享外层并行轴 | `p_two_serial_independent` 等 4 例 |
| **部分归约 + 并行轴可行（本轮新增）** | **serial 轴被拒 → 自动回退到并行轴** | `d14b_partial_reduce_fallback` |

### 2.5 其它

| 场景 | 说明 | 用例 |
|---|---|---|
| **符号 extent（非归约，本轮新增）** | 运行时 `ceildiv(extent, lanes)` chunk 循环 + 运行时夹紧尾块掩码 | `d9_symbolic_extent` |
| **非归约尾块（本轮新增）** | 拆整块 + 尾块；尾块装载带谓词，不越界 | `d10_non_reduction_tail_overread` / `d10b_one_dimensional_tail` |
| 轴打分 + 回退 | 写侧×100、读侧×10、同分取内层；逐轴尝试 | `c2_axis_pick` / `c3_im2col` |
| 整段语句树变换 | `vectorize_parallel_nests` 自顶向下替换 | `test_vectorize_parallel_nests_*` |

---

## 3. 暂不支持场景清单

> 默认行为：**该轴**被拒后尝试次优轴；所有轴都失败才整段拒绝（`Unsupported` + 全轴 diagnostics）。

| 场景 | 当前行为 / 消息 | 原因 | 性质 |
|---|---|---|---|
| GM 操作数在 T.Parallel 内 | 拒绝：`needs an explicit UB staging copy` | 未建模 GM↔UB 搬运 | 设计边界（需 staging） |
| `T.atomic_add` 等原子 | 拒绝：`no vector+scalar atomic recipe` | 缺原子配方 | 设计边界 |
| 真无结构下标 `a[i//3]` | 拒绝：`FloorDiv by 3 is not a power of two` | 索引向量通道只支持 2 的幂除数 | 设计边界（可扩展 magic-number） |
| 共享标量 / 跨迭代 stencil / 非直接内嵌 serial | 拒绝：`nested serial loop ... is not affine-vectorizable` | `_serial_is_clean` 保守判据 | 设计边界（宁拒绝勿错发） |
| 并列 serial 中有一段不干净 | 同上（回落到单段路径后拒绝，段级原因在 diagnostics） | 同上 | 设计边界 |
| 吸收 serial 轴上的**部分归约** | 单段：回退并行轴；多段：拒绝（diag 含 `partial reduction`） | 逐元素路径只能 `ONEPT` lane 0 | 设计边界（A0） |
| **归约 + 符号 extent** | 支持：发运行时 chunk 循环 + `local.var` 累加器 | 长归约 / 符号 extent 不再展开（D9 归约分支已解除） | 已支持 |
| 不支持的 store 元素类型（`bool`） | 拒绝：`store element dtype 'bool' ... is not supported` | `update_mask` 无 b1 宽度 | 设计边界 |
| **`bool` 作数据操作数（`Cast(float32, c[i])`）** | **build OK → emit `ValueError`** | 见 A1 | **缺陷（应转为拒绝）** |
| 非 Add/Max/Min 归约（乘积） | 拒绝：`unsupported reduction operator(s) Mul` | 无对应跨 lane 原语 | 设计边界 |
| 覆写 `s[i]=a[i,j]`（lane 均匀地址 + 变化取值） | 拒绝：`does not read 's' ... accumulate form` | last-write-wins 无向量形式 | 设计边界 |
| 归约旧值被缩放/不在算子上 | 拒绝：`must be a direct operand ... accumulate position` | 无法只在末尾并入 | 设计边界 |
| 发散分支（两臂写不同地址） | 拒绝：`divergent branch arms ...` | 无 scatter+select 组合 | 设计边界 |
| 双臂 `if` 的臂内再嵌 `if`（D12） | 拒绝：`masked branch merging is not supported` | 未实现掩码合并 | 未声明路径（已文档化） |
| pre/post 非沿向量轴均匀 | 拒绝：`a statement before/after the serial loop must be uniform` | 防御性门禁（当前不可达） | 设计边界 |
| **沿并行轴向量化 + 兄弟语句** | 拒绝：`can only be vectorized along that serial axis` | 发射器把剩余轴统一包在 pre/post 外，结构不支持 | 设计边界（上轮已论证：仅"serial 轴不可向量化 + 外轴可用 + 有兄弟语句"时才有价值） |
| `FloorDiv/FloorMod` 作值算子、非 2 的幂 | 拒绝：`not strength-reduced yet` | 需 `vmull` magic-number | 长期 |
| `T.Parallel` 体内无写语句 | 拒绝：`no store to vectorize` | — | 正常边界 |

---

## 4. 建议的下一步（按性价比）

1. **A1/A2（唯一会崩的契约裂缝）**：bool 数据操作数在 build 期拒绝 + `load_node` 用 buffer 元素类型派生 dtype，约 10 行，配 1 个用例。
2. **A4**：多段失败顶层消息改用第一条段级原因（纯诊断改进）。
3. **D1/D2**：清理 9 处漂移行号 + 能力矩阵补 5 行（纯文档）。
4. **B1**：符号 extent 的全 chunk gather 优化（有实际性能收益，但需要运行时分支/peel，风险中等）。
5. **C1/C5**：把"是否需要绝对索引向量"收敛成单一判据，消除 D8 类分散修改的隐患