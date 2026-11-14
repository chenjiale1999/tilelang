# Ascend P2 Plan / P3 发射寄存器使用逻辑专项分析

> **历史文档**：本文记录写作时的实现/设计状态。后续改动已删除或重命名部分
> 文件、测试与行号；当前实现见 `src/ascend/transform/parallel_vectorize/` 与
> `src/ascend/transform/vsimd_*`，以代码为准。


> **分析对象**：`T.alloc_fragment`（V 核寄存器 tile）在 `T.SimdVF` 内经
> `T.Parallel` 读写的场景，重点回答「P2/P3 是否复用用户手动申请的寄存器」。
>
> **方法**：代码溯源（`src/ascend/transform/parallel_vectorize/*`、`stage_passes.cc`、
> `emit.cc`、`model.cc`、`vf_checker.py`、`eager/builder.py`）+ 离线实测
> （构造 kernel → 看 eager IR → 看 P1 交付 → 看 plan/emit → 看完整 lowering 源码）。
>
> **实测环境**：`tilelang.lower(..., target="ascend")`，NPU 未参与；行号对应
> 当前工作树（与设计文档 `p2_p3_manual_fragment_register_design.md` 同一基线）。

---

## 0. 结论速览

| # | 问题 | 结论 |
| --- | --- | --- |
| 1 | P2/P3 是否复用用户 `alloc_fragment` 的寄存器 | **不复用**。P2 不分配任何寄存器；P3 的寄存器全部是系统新建（SSA `Bind`、mask、`vdup`、`local.var` 累加器），用户 fragment 始终按**内存 buffer** 处理（`vld`/`vsts`）。 |
| 2 | 单 SimdVF 多 parallel 循环、重复 `alloc_fragment` | **未实现跨循环寄存器复用**：每个 nest 独立 plan/emit，同一 fragment 的跨循环值通过**内存往返**保持；重复 alloc 会产生**多个独立 buffer**（不合并）。另发现一个**已复现的 P3 误编译**：同名 fragment 在同一 nest 内被 store/load 时，`BufferKey` 冲突导致 forward 串台。 |
| 3 | `alloc_fragment` 跨 SimdVF | **方向 A（VF 内分配、VF 外使用）**：eager IR builder 报 `Immutable variable ... used outside its defining region!`，并打印 defining frame 与 current frames，诊断准确，可排除出 P2/P3 分析。**方向 B（VF 外分配、VF 内或多 VF 使用）**：**不报错**，fragment 以 `float*` 参数传入 VF helper，必须保留在分析范围内（设计中的 R8 需补约束）。 |
| 4 | `vf_checker` 支持范围是否只有 UB+寄存器、GM 不在范围 | **前提不成立**。`vf_checker` 对 GM 有**三处显式拒绝**（load/store/copy），并对 `local.var`、VF 内分配逃逸、VF 外 `T.Parallel` 做检查；它没有「UB + 寄存器」的正向白名单。正确表述是：**GM 在 VFChecker 内被显式拒绝，拦截点早于 P2/P3，因此 P2/P3 无需新增 GM 机制**。 |
| 5 | 寄存器使用规则与边界 | 见第 6 节矩阵；当前实现下「用户 fragment = 可寻址内存」，「系统寄存器 = SSA/`local.var`」，两者没有建立映射。 |

---

## 1. P2 plan 与 P3 发射的完整流程与寄存器来源

### 1.1 执行链路

```
T.Parallel nest（SimdVF 内）
  → P1 AscendLiftParallelToArray
       BuildP2PlanInputs: 捕获/合法性/Region
       AttrStmt(tl.vsimd.array_region, region, body=原 nest)
  → P2 AscendPlanArray
       PlanMarkerReplacer 逐 marker：
         RegionToPlanInput(region) → BuildPlan(input, layouts)
         AttrStmt(tl.vsimd.array_plan, VectorProgram, body=原 nest)
  → P3 AscendLowerArrayPlan
       PlanMarkerLower 逐 marker：EmitPlan(program) → 向量化语句
  → AscendLowerTileOp（local.fragment → local + layout 重映射）
  → codegen
```

关键点：

- P2/P3 的驱动都是**逐 marker（逐 nest）**的
  （`stage_passes.cc:182-217`、`:219-240`），**没有 VF 级状态**。
- 一个 VF 内的多个 nest 是多个独立的 `array_plan` marker；P3 对每个 marker
  调一次 `EmitPlan`（`stage_passes.cc:227`），寄存器状态不跨 marker 共享。

### 1.2 P2 是否分配寄存器：否

在 `polyhedral/{plan,value,model,body,access,render}.{cc,h}` 中搜索
`decl_buffer` / `AllocBuffer` / `local.var`，只命中 `model.h:232` 的一行注释
（累加器判据说明），**没有任何寄存器分配代码**。P2 产出的是纯数据：

| 字段 | 含义 | 是否寄存器 |
| --- | --- | --- |
| `VectorPlan.lanes` | 段 chunk 宽度（向量寄存器 lane 数） | 只是宽度，不是分配 |
| `StorePlan.store_lanes` / `store_lane_mode` | 单个 store 的自然寄存器宽度/打包形态 | 只是宽度描述 |
| `StorePlan.promote` | 「该 store 应提升为 `local.var` 累加器」的判据结果 | 只是标记，P3 才建寄存器 |
| `ReduceInfo.recipe` | 归约指令配方 | 只是指令选择 |
| `AccessInfo.buffer/indices/offset` | 访存模型 | 仍是 buffer 语义 |

`StorePlan.spec.buffer` 直接就是用户 fragment 的 Buffer 对象；P2 没有把它
映射成任何寄存器实体。

### 1.3 P3 的寄存器来源：全部是系统新建

| 来源 | 代码位置 | 产物 | 命名 |
| --- | --- | --- | --- |
| SSA 值绑定 | `emit.cc:623-628` `Bind` | `tirx::Var` + `Bind` | `v<prefix><n>` |
| prologue 提升 | `emit.cc:630-634` `PrologueBind` | 提升到 prologue 的新 Var | `v<prefix><n>` |
| 掩码 | `emit.cc:654-690` `AllMask`/`Lane0Mask`，`:637-652` `SharedMask` | `vector_bool` Var | `mask_*` / `lane0_*` / `mask_N` |
| 标量广播缓存 | `emit.cc:902-914` `Dup` | `vdup` 结果 Var（可提升） | `v<prefix><n>` |
| 归约累加器 | `emit.cc:1967-1974` `EmitReduceLoop` | `local.var` 向量 buffer `acc...` | `acc<prefix><n>` |
| 提升累加器 | `emit.cc:2150-2169` `EmitParallelAxisParts` | `local.var` 向量 buffer `acc...` | `acc<prefix><n>` |
| 段间寄存器直通 | `emit.cc:867-882` `RegisterCarry`/`CarryEntryLookup` | 仅复用**同一 program 内**已发射的 SSA 值 | 不新建 |

这些实体全部由 P3 自己命名/分配；**没有一个来自用户的 fragment Buffer**。
`BufferKey` 只用于值缓存/转发（见 3.4），不是寄存器分配。

### 1.4 用户 fragment 在 P3 中的真实角色

- load：`value.cc:780-807`（`LoadNode`）按 buffer 访问分类，连续访问在
  `:814-816` 变成 `vload`；emit 时 `vld` 绑到新的 SSA 寄存器。
- store：`emit.cc:1220-1257`（`EmitVectorWrite`）统一发 `Vsts`，目标是
  `store->spec->buffer`；不区分 scope，也没有「寄存器目标」分支。
- 因此 fragment 的读写就是「从寄存器发 `vsts` 到 fragment 地址」和「从
  fragment 地址 `vld` 到寄存器」。

实测（M1：一个 fragment、两个 nest）生成的源码：

```c
__simd_vf__ inline void main_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  float frag[64];                                     // fragment 被当成 local 数组
  auto vs0_0 = simd_inst::vlds_norm<float>(...);      // 系统新建 SSA 寄存器
  auto vs0_1 = simd_inst::vadds(vs0_0, 1.0f, ...);    // 系统新建 SSA 寄存器
  simd_inst::vsts_norm(vs0_1, (__ubuf__ float*)(&(frag[0])), 0, mask_32);  // 写回 fragment 内存
  auto vs0_0_1 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(frag[0])), 0); // 从 fragment 内存再读
  auto vs0_1_1 = simd_inst::vmuls(vs0_0_1, 2.0f, ...);
  simd_inst::vsts_norm(vs0_1_1, ..., ...);
}
```

`float frag[64]` 来自 `AscendLowerTileOp` 把 `local.fragment` 重映射成 `local`
（`lower_tile_op.cc:100-105`）；`__ubuf__` 强转来自 codegen 对
`tl.simd.vld/vsts` 的硬编码（`codegen_ascend.cc:2369-2372`、`:2424-2426`）。
这正是此前确认的不可编译点。

### 1.5 结论

P2/P3 **没有复用用户 fragment 作为寄存器**；P3 内部重新定义了一套全新寄存器
（SSA + `local.var`），用户 fragment 只是这些寄存器的**内存落点**。因此
「用户手动申请寄存器」在现有实现中名不符实：它申请到的是一块被当成 UB
地址访问的 local 数组。

---

## 2. 单 SimdVF 内多个 `T.Parallel` 循环

### 2.1 同一 fragment 跨循环（M1）

- 同一个 Buffer 被 nest0 写、nest1 读。
- P1：两个 nest 都 `LIFTED`，`stmt` 里引用的 fragment 是同一个 Buffer。
- P2：两个独立 plan；nest0 的 store 目标是 fragment，nest1 的 value 是一次
  fragment `vload`。
- P3：两个独立 `EmitPlan`；nest0 发 `vsts` 到 fragment，nest1 发 `vld` 从
  fragment 读回。
- **没有寄存器复用**：值跨循环通过内存往返保持；SSA 寄存器 `vs0_*` 在每个
  nest 重新创建，不复用上一个 nest 的结果。

结论：功能语义上「同一个 fragment 跨循环可用」，但实现方式是**内存**
（且该内存形态在 ccec 上因 `__ubuf__` 强转不可编译），**不是用户预期的
寄存器复用**。

### 2.2 不同循环重复 `alloc_fragment`（M2/M3）

- **M2**：两个循环各自 `T.alloc_fragment`（即使写在 `T.Parallel` 循环体内），
  eager builder 合法；两个分配都被 steal 到同一个 `SIMD_VF` block 的
  `alloc_buffers`（`ir.cc:271-299`），成为两个独立 buffer `f1`/`f2`。
  各自 nest 只用自己的 buffer，无共享/复用。
- **M3**：用**同一个 Python 变量名**分配两次，TIR 里是两个**不同的 Buffer**
  （eager builder 自动把第二个的数据 Var 重命名为 `frag_1`），但两个 Buffer 的
  `buffer.name` 与 `data.name` 都仍打印为 `frag`。

| 场景 | Buffer 数量 | 寄存器复用 | 值持久化 |
| --- | --- | --- | --- |
| 一次 alloc，多个循环使用 | 1 | 无（每 nest 重新 SSA） | 通过 fragment 内存 |
| 多次 alloc（不同名） | N | 无 | 各自内存 |
| 多次 alloc（同名） | N（data Var 不同） | 无 | 各自内存，但存在 key 冲突风险（见 2.3） |

### 2.3 已复现缺陷：同名 fragment 的 `BufferKey` 冲突导致 P3 误编译

**触发条件**：同一 VF 内两次 `alloc_fragment` 使用相同变量名（第二次分配前把
第一次的引用保存在列表/闭包里），且**同一个 nest** 同时写其中一个、读另一个。

复现代码（精简）：

```python
with T.SimdVF():
    f = T.alloc_fragment((N,), "float32")
    keep = [f]                       # 保住第一次分配
    f = T.alloc_fragment((N,), "float32")   # 第二次分配，同名
    for i in T.Parallel(N):
        keep[0][i] = a_ub[i] + 7.0   # 写第一个
    for i in T.Parallel(N):
        f[i] = a_ub[i]               # 写第二个
        c_ub[i] = keep[0][i]         # 读第一个
```

实测结果：

- P2 plan **正确**：store0 的 buffer 与 store1 value 里 `vload` 的 buffer
  是**两个不同的 data Var**（实测 id 不同）。
- P3 发射**错误**：`c_ub` 的 store 使用了写第二个 fragment 的值，等价于
  `c_ub = a_ub`；正确结果应是 `c_ub = a_ub + 7.0`。
- 根因：`emit.cc:485-500` 的 `BufferKey` 只由
  `data->name_hint + dtype + scope + shape + strides + elem_offset` 组成，
  **不含 data Var 身份**；两个同名同形 fragment 得到相同 key。
  `NoteStore`（`emit.cc:857-861`）写入 `forward_[key^indices]`，
  `LoadNode`（`emit.cc:1068-1080`）用同一 key 查 forward，于是把「写第二个
  fragment 的值」错误转发给了「读第一个 fragment」。
- 影响面：`forward_`/`store_gen`/`written` 等所有按 `BufferKey` 建索引的
  状态都可能串台；P2 的 `SameBuffer` 用的是 data Var（`primitives.cc`），
  所以 plan 层不受影响，**缺陷在 P3**。

**结论**：现有实现对「重复 alloc_fragment」并不完全符合用户预期：不仅没有
寄存器复用，同名场景还存在可复现的正确性缺陷。未来的 per-VF 寄存器文件必须
按 **data Var 身份**索引（`buffer->data` / `same_as`），并建议顺手把
`BufferKey` 改为 identity-based，或在 key 中加入唯一 buffer id。

### 2.4 与用户预期对照

| 用户预期 | 现状 | 差距 |
| --- | --- | --- |
| 同一 fragment 跨多个 parallel 循环保持寄存器值 | 通过 fragment 内存往返 | 不是寄存器复用；且生成代码不可编译 |
| 多次 alloc 得到多个独立寄存器 tile | 得到多个独立 Buffer，但都按内存处理 | 没有寄存器实体 |
| 同名重复 alloc 互不干扰 | P2 独立，P3 有 `BufferKey` 串台风险（已复现） | 正确性缺陷 |
| 循环之间不产生额外内存流量 | 每 nest 都有 `vsts`/`vld` | 额外内存流量 |

---

## 3. `alloc_fragment` 跨 SimdVF 的约束

### 3.1 三个方向

| 方向 | 形态 | 实测行为 |
| --- | --- | --- |
| A | VF0 内分配，VF1（或 VF 外）使用 | eager builder 报错 |
| B | VF 外分配，VF 内使用（或多个 VF 使用） | **不报错**，指针捕获 |
| C | VF0 内分配，VF 外使用 | eager builder 报错（同 A） |

### 3.2 方向 A/C：eager IR builder 明确报错

实测错误（C1/C3 相同）：

```
RuntimeError: Immutable variable `frag` is used outside its defining region!
variable `frag` is defined in frame: tl.SimdVFFrame(stmts=(for i in T.parallel(64): ...))
current frames: [..., tl.SimdVFFrame(stmts=()), ForFrame(vars=(i,), ...)]
```

- 机制：eager builder 在绑定时记录 `name_inside_frame`
  （`builder.py:695-713`），读取时 `binding_expired` 判断定义 frame 是否仍在
  当前 frame 栈（`builder.py:716-719`），不在则抛出上述错误
  （`builder.py:908-913`）。
- 诊断质量：消息同时打印 defining frame 与完整 current frames，能准确看出
  「定义在已关闭的 SimdVF frame、当前在另一个 frame」。
- 结论：**方向 A/C 可以排除出 P2/P3 分析范围**（根本到不了 P1/P2）。

### 3.3 方向 B：不报错，必须保留在分析范围内

实测（C2：VF 外分配、单个 VF 内使用；C4：VF 外分配、两个 VF 使用）：

- eager builder 不报错；
- VFChecker 也不报错（它只检查「VF 内分配 → VF 外使用」，
  `vf_checker.py:57-69`；不检查「VF 外分配 → VF 内写」）；
- lowering 成功，fragment 变成 kernel 作用域 `float frag[64]`，以
  `float* frag` 参数传入 `__simd_vf__` helper：

```c
__simd_vf__ inline void main_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, float* frag)
...
main_kernel_simd_vf_0(buf_dyn_shmem, frag);
```

- 该形态仍然落进「fragment 被 `__ubuf__` 强转」的不可编译路径。

**结论**：需求 3 的前提只对「VF 内分配、跨 VF 使用」成立；「VF 外分配、多 VF
共享」不会被前端拦截，**必须纳入分析**。设计文档中的 R8（分配必须在 VF 内）
不能依赖前端检查，需要在 P2.5 自行实现。

---

## 4. `vf_checker` 模块支持范围核查

### 4.1 检查项清单（全量）

| 检查 | 代码位置 | 作用域 |
| --- | --- | --- |
| VF 内分配 → VF 外访问 | `vf_checker.py:57-69` | 任意 buffer（含 `local.fragment`） |
| VF 外 `local.var` → VF 内写 | `vf_checker.py:71-78` | `local.var` |
| `T.Parallel` 必须在 VF 内 | `vf_checker.py:80-88` | 循环结构 |
| `T.copy` 的分配逃逸 | `vf_checker.py:90-98` | 任意 buffer |
| VF 外 `T.copy` dtype 不一致（隐式 cast） | `vf_checker.py:100-107` | UB/L0C |
| **SimdVF 内 `T.copy` 涉及 GM** | `vf_checker.py:109-116` | **GM 显式拒绝** |
| VF 外 BufferStore 隐式 cast | `vf_checker.py:141-150` | 任意 buffer |
| **SimdVF 内 BufferStore 到 GM** | `vf_checker.py:152-158` | **GM 显式拒绝** |
| **SimdVF 内 BufferLoad 从 GM** | `vf_checker.py:161-170` | **GM 显式拒绝** |

调用点：`pipeline.py:95`，在 `AscendParallelVectorize`（`:124`）之前。

### 4.2 对「GM 不在支持范围」的修正

- GM **不是被忽略**，而是被三处规则**显式识别并拒绝**，错误信息明确
  （`SIMD_VF blocks cannot access global memory`）。
- VFChecker 没有「只支持 UB 和寄存器两类操作数」的正向白名单；它按
  「分配逃逸 / 隐式 cast / GM / local.var / T.Parallel 位置」逐条检查，
  覆盖 UB、`local.var`、`local.fragment`、GM。
- 因此正确的结论是：**GM 由 VFChecker 在 P2/P3 之前拦截，P2/P3 无需新增 GM
  机制；但 GM 场景属于 VFChecker 的显式拒绝范围，不能表述为「不在其支持范围」**。
- **后续决策（设计文档 §2.9）**：GM 不再纳入 P2/P3 的建模与测试范围；
  P2 的 `in_global`/`global-memory operand` 分支与 GM 相关测试列入删除清单，
  GM 的拒绝行为由 `vf_checker` 独占负责。本节保留为前端行为的记录。

### 4.3 覆盖盲区

- `local.fragment` 在 VF 内分配、VF 外使用：由 `_check_vf_allocation_scope`
  覆盖（报错）。
- `local.fragment` 在 VF 外分配、VF 内使用/多 VF 使用：**无检查**（见 3.3），
  这是 VFChecker 的盲区，也是设计 R8 需要补的地方。
- `local.var` 之外的寄存器形态（`local.fragment`）没有「写入来源」检查；
  目前由 P1/P2/P3 自行处理。

---

## 5. 场景—规则—系统行为矩阵

| 场景 | eager builder | VFChecker | P1 | P2 | P3 | 端到端 |
| --- | --- | --- | --- | --- | --- | --- |
| 单 fragment、单 nest 读写（UB 数据） | 通过 | 通过 | LIFTED | ND plan | `vld/vsts`，系统 SSA | 不可编译（`__ubuf__` cast） |
| 单 fragment、多 nest 复用 | 通过 | 通过 | 两个 LIFTED | 两个独立 plan | 内存往返，无寄存器复用 | 不可编译 |
| 多次 alloc（不同名） | 通过 | 通过 | 各自 LIFTED | 各自 plan | 各自内存 | 不可编译 |
| 多次 alloc（同名，单 nest 同时访问） | 通过 | 通过 | 各自 LIFTED | plan 正确 | **BufferKey 串台，误编译** | 错误 |
| VF 内分配、VF 外/VF2 使用 | **报错** | — | — | — | — | 前端拦截 |
| VF 外分配、VF 内/多 VF 使用 | 通过 | 通过（盲区） | LIFTED | ND plan | 指针捕获 + `vsts/vld` | 不可编译 |
| GM 直读/直写 | 通过 | **显式拒绝** | — | — | — | 前端拦截 |
| 语句级 `if` 填充 | 通过 | 通过 | OPAQUE | — | — | 保持标量 |

---

## 6. 对既有设计文档的修订建议

1. **明确「现状不复用」**：在 `p2_3_manual_fragment_register_design.md` 的
   §0/§1 增加一句：P2 无寄存器分配，P3 的寄存器全部是系统新建，用户 fragment
   只是内存落点；本设计的目标是建立「fragment ↔ 寄存器」的映射。
2. **R8 必须包含前端盲区**：不仅是「VF 内分配跨 VF 使用」（eager 已拦），
   还包括「VF 外分配、VF 内/多 VF 使用」（前端不拦），后者必须由 P2.5 拒绝。
3. **寄存器文件按 data Var 索引**：现有 `BufferKey`
   （`emit.cc:485-500`）是 name/shape 结构键，同名 fragment 会冲突；
   per-VF `RegisterFile` 必须用 `buffer->data` 指针/`same_as` 或唯一 buffer id。
   建议同时修复 `BufferKey`（加入唯一 id），否则 `forward_`/`store_gen`/
   `written` 都有串台风险（已复现 2.3 的误编译）。
4. **诊断携带唯一标识**：同名 fragment（`frag`/`frag_1`）在 P1 stmt、
   `VerifyBufferInit` 与 P3 日志里都可能显示为同一个名字；寄存器诊断应同时
   打印 data Var 的唯一标识或分配序号。
5. **多 nest 复用规则要写清楚**：同一 buffer 跨 nest → 同一组寄存器槽；
   不同 alloc → 不同槽，不做合并；同名不同 data Var → 视为不同 buffer，
   并显式报错/重命名，禁止 key 合并。

---

## 7. 配套验证建议（针对本次分析结论）

1. `test_fragment_register_single_alloc_across_nests`：断言两个 nest 的 plan
   引用同一 buffer，P3 寄存器模式下只有一组槽、无 fragment `vld/vsts`。
2. `test_two_allocs_same_name_same_nest_do_not_alias`：复现 2.3 的误编译，
   断言两个 buffer 的 key/转发互不影响（修复后应得到 `a_ub+7`）。
3. `test_fragment_alloc_outside_vf_is_rejected_by_plan`：方向 B，断言 P2.5
   给出「not allocated inside this VF」诊断（前端不拦）。
4. `test_fragment_alloc_in_vf_used_outside_raises_eager`：方向 A/C，断言
   eager `RuntimeError` 文案含 `defined in frame` 与当前 frame 栈。
5. `test_vf_checker_gm_load_store_copy_rejected`：三条 GM 规则各一例，
   断言错误信息分别指向 load/store/copy。
6. `test_same_name_fragment_diagnostics_carry_identity`：断言日志/诊断能区分
   `frag` 与 `frag_1`。

---

## 8. 附：本次实测的最小证据索引

- M1 单 fragment / 两 nest：完整 lowering 源码（`float frag[64]`、
  `vsts_norm(..., &frag[0])`、`vlds_norm(&frag[0])`）。
- M2 两 fragment / 两 nest：两个独立 `alloc_buffers`（`f1`/`f2`）。
- M3 同名两次分配：TIR 中 `frag`/`frag_1`；P1 stmt 两处都打印 `frag`。
- M6 同名 + 单 nest 交叉读写：P3 输出 `c_ub = a_ub`（错误），plan 中两个
  buffer 的 data Var 不同（证明缺陷在 P3 的 `BufferKey`）。
- C1/C3：eager `Immutable variable ... used outside its defining region`。
- C2/C4：VF 外分配 → `float* frag` 参数捕获，无报错。
- GM：`VFChecker` 在 P1 之前报 `BufferLoad from global buffer 'A' inside a
  SIMD_VF block is not allowed`。
