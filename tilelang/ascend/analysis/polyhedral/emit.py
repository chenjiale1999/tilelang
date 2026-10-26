"""发射层：``VectorPlan`` / ``VectorProgram`` → ``T.simd.*`` TIR。

发射目标全部是 ``T.simd.*``（``tilelang/ascend/language/simd.py``）——CCE 向量
原语的前端封装，与 C++ pass ``AscendSimdVFLowerParallel`` 的发射风格一致：

=========================================== ================== =========================
``T.simd.*``                                 返回类型            语义
=========================================== ================== =========================
``vld(addr, "NORM")``                        ``<elem> x lanes``  连续装载
``vgather2(base, idx, mask)``                ``<elem> x lanes``  按每 lane 索引装载
``vdup(scalar, elem, mask)``                 ``<elem> x lanes``  lane 内均匀广播
``vci(base, "int32")``                       ``int32 x lanes``   lane 序号
``vadd/vsub/vmul/...``                       ``<elem> x lanes``  逐元素
``vadds/vmuls/vmaxs/vmins``                  ``<elem> x lanes``  标量-向量
``vcmp/vcmps``                               ``boolx256``        比较谓词
``pand/por/pnot``                            ``boolx256``        谓词运算
``pset/update_mask``                         ``boolx256``        掩码
``vsel(t, f, mask)``                         ``<elem> x lanes``  mask ? t : f
``vcvt(src, target_elem, mask)``             ``<target> x lanes``类型转换
``vcadd/vcmax/vcmin``                        ``<elem> x lanes``  跨 lane 归约(lane0)
``vsts(..., "NORM_B{bits}")``                ``void``            连续存储
``vsts(..., "ONEPT_B{bits}")``               ``void``            单元素存储(归约回写)
``vscatter(v, base, idx, mask)``             ``void``            按每 lane 索引存储
=========================================== ================== =========================

地址统一用 ``BufferLoad(buffer, indices)`` 交给 ``T.simd`` 自动包成 ``access_ptr``；
连续装载/存储把向量轴变量替换成该 chunk 的 lane 基址（普通路径 = ``chunk * lanes``；
lane 内均匀访问 = ``0``）。**每 lane 索引向量**通道（STRIDED 步长 / PERMUTE 逆序 /
PIECEWISE 分片 / INDIRECT 间接）改用 ``base = BufferLoad(buffer, 全 0)``（缓冲区起点）
+ 绝对元素偏移索引向量（``value.index_value`` 由 ``build_value`` 在整数上下文构造）。

归约（§2 Case II）按 chunk 数**分两种策略**（阈值 :data:`_REDUCE_UNROLL_LIMIT`）：

* **短归约（``chunks <= 4``）——编译期展开 + 多路累加器**：直线代码没有循环控制开销，
  后端可把下一块的 ``vld`` 提前；每个 chunk 的部分归约各成一条独立链，最后按二叉树
  合并，把跨 lane 归约的串行延迟从 ``chunks`` 级压到 ``ceil(log2 chunks)`` 级::

      red0 = vcadd(vld(src(0)))              # 每个 chunk 一条独立链，不需要单位元
      red1 = vcadd(vld(src(1)))
      red2 = vcadd(vld(src(2)))
      red3 = vcadd(vld(src(3)))
      acc0 = vadd(red0, red1)                # 树形合并（lane 0）
      acc1 = vadd(red2, red3)
      acc  = vadd(acc0, acc1)

  （``chunks == 1`` 时退化成一次 ``vcadd`` + 回写，没有中间 ``vadd``。）

* **长归约 / 符号 extent——真实 chunk 循环 + 可变寄存器累加器**：展开会让指令段随
  ``chunks`` 线性膨胀，而收益（去掉循环控制）在 chunks 大时占比很小；此时把累加器落成
  ``local.var`` 作用域的向量寄存器（``alloc_var`` 语义），chunk 循环保留在 IR 里::

      acc[0] = vdup(identity)               # AllocBuffer(acc) + 初值
      for ck in range(ceildiv(E, lanes)):
          acc[0] = vadd(acc[0], vcadd(vld(src(ck))), lane0_mask)
      final   = acc[0]

  符号 extent 只能在循环里用运行期尾块谓词 ``update_mask(min(E - ck*lanes, lanes))``
  + ``vsel`` 填单位元，因此它只走这条路径（不再拒绝）。

**跨 lane 归约之后的算术一律收窄到 lane 0**：``vcadd``/``vcmax``/``vcmin`` 只有 lane 0
有效（硬件模板本身就用 ``PAT_VL1`` 合并），所以链合并、累加器累加与回写都用
``pset(bits, "PAT_VL1")`` 谓词，不再对 63 个无效车道做运算。累加写（``dst op= x``）的旧值
按**标量**读一次直接进 ``vadds``/``vmaxs``/``vmins``，省掉 BRC 广播装载；最后用
``vsts(..., "ONEPT_B{bits}")`` 写回单元素。
"""

from __future__ import annotations

from tvm import arith, tirx

from tilelang.ascend.language import simd
from tilelang.ascend.language.simd import _mask_bits, _vec_dtype

from .access import AccessClass, AccessInfo, Axis, _free_vars
from .model import StorePlan, VectorPlan, VectorProgram
from .primitives import (
    _ELEMENTWISE_BINARY,
    _ELEMENTWISE_UNARY,
    _MASK_DTYPE,
    _MODE,
    _PREDICATE_MASK_BITS,
    _REDUCE_ELEMENT,
    _SCALAR_ONLY_RECIPES,
    _SCALAR_TO_VECTOR_BINARY,
    _VEC_SCALAR_BINARY,
    Unsupported,
    _elem_dtype,
    _int,
    _reduce_identity,
    _same_buffer,
    _same_indices,
    _seq,
)
from .value import ValueNode, _reads_buffer, same_address_load

# 归约发射策略（见模块 docstring 的「短归约 / 长归约」两段）：
#   * ``chunks <= _REDUCE_UNROLL_LIMIT`` → 编译期展开（去掉循环控制，后端可跨块调度）；
#   * 否则（含符号 extent）→ 真实 chunk 循环 + ``alloc_var`` 可写向量寄存器累加器。
# 阈值取 4：展开后约 ``3*4+3 = 15`` 条指令，仍在单个指令段的舒适范围内；再大则
# 指令体积线性膨胀，而「去掉循环控制」的收益占比迅速下降。
_REDUCE_UNROLL_LIMIT = 4


def _unsigned_index(reg):
    """把有符号整数索引向量按位重解释为无符号（``vgather2`` / ``vscatter`` 需要）。"""
    dtype = str(reg.dtype)
    if dtype.startswith("int"):
        return tirx.reinterpret(dtype.replace("int", "uint", 1), reg)
    return reg


def _buffer_key(buffer):
    """Buffer 的结构身份（不依赖 Python 对象身份）。

    同一个 ``T.Buffer`` 的不同 ``BufferLoad`` 可能携带**不同的 Buffer 对象**
    （``id(buffer)`` 不同，但 ``buffer.data.same_as`` 为真），因此 SSA 复用键
    必须按结构字段比较，否则 ``c[i] = a[i]`` / ``d[i] = a[i]*2`` 会各发一条 ``vld``。
    """
    return (
        buffer.data.name,
        str(buffer.dtype),
        buffer.scope(),
        tuple(str(size) for size in buffer.shape),
        tuple(str(stride) for stride in buffer.strides),
        str(buffer.elem_offset),
    )


def _value_key(node: ValueNode, store_gen: dict | None = None):
    """SSA 复用键：结构相同的值节点只发一次（含装载与纯算术）。

    ``build_value`` 对每个 ``BufferLoad`` / 表达式出现都新建 :class:`ValueNode`，
    所以 ``if a[i, j] != 0: c[i, j] = a[i, j]`` 里的两次读、以及 ``access_h`` 这类
    被内联多次的地址子式都是**不同节点**；按 ``id(node)`` 缓存会让它们各自重发
    ``vld``/``vcmp``/``vshrs``。这里改用整棵 DAG 的结构键。

    ``store_gen`` 是「buffer → 已发射的写次数」：装载键里带上代数，写同一 buffer
    之后不会复用写之前的装载（正确性优先；同一 chunk 内未写过的 buffer 仍可复用）。
    """
    access = None
    if node.access is not None:
        buffer_key = _buffer_key(node.access.buffer)
        access = (
            buffer_key,
            tuple(str(index) for index in node.access.indices),
            str(node.access.offset),
            (store_gen or {}).get(buffer_key, 0) if not node.access.is_store else 0,
        )
    return (
        node.recipe,
        node.dtype,
        node.detail,
        str(node.scalar) if node.scalar is not None else None,
        access,
        tuple(_value_key(arg, store_gen) for arg in node.args) if node.args else None,
        _value_key(node.predicate, store_gen) if node.predicate is not None else None,
    )


class _ProgramPrologue:
    """多段 program 的**共享** chunk 无关事实（掩码 / 步长斜坡）。

    每个段原先各绑一份 ``pset(bits, PAT_ALL)`` 与 ``vmuls(vci(0), stride, mask)``，
    多段时数值完全相同的绑定会重复出现。这些值只依赖常量与彼此，放在 program 最前
    只算一次即可（段内仍按 ``_mask_vars`` / ``_ramps`` 命中）。

    这也是「多段发射的最小共享单元」：只共享**纯值**，不共享任何控制流或写语义，
    因此不会改变段的源序执行模型。
    """

    def __init__(self):
        self.statements: list = []
        self.masks: dict = {}  # ("all"|"lane0", bits) -> var
        self.ramps: dict = {}  # (stride, id(mask)) -> var


class _Emitter:
    """把 :class:`VectorPlan` 的 chunk 体发射成 ``T.simd.*`` SSA 序列。

    SSA 用 ``SeqStmt`` 里的 ``tirx.Bind`` 表达（``Bind`` 是独立语句而非
    表达式，因此不能像 ``tirx.Let`` 那样嵌套）。``lane_base`` 是代入向量轴
    变量的 lane 基址：普通路径是 ``chunk * lanes``，归约路径在展开时逐 chunk
    重设（并清空 ``memo``，因为地址随 chunk 变化）。

    ``shared`` / ``carry`` 只在多段 program 里非空：

    * ``shared``：段间共享的掩码 / 斜坡绑定（:class:`_ProgramPrologue`）；
    * ``carry``：**段间寄存器直通表**。上一段把某个连续写的结果登记为
      ``(buffer 键, 下标) -> (向量轴变量, 寄存器值)``；下一段在同址连续读时直接
      复用该寄存器（省掉一次 ``vld``）。键与 ``_forward`` 一致，但生命周期跨段。
    """

    def __init__(
        self,
        plan: VectorPlan,
        chunk: tirx.Var,
        prefix: str = "",
        *,
        masked_tail: bool = False,
        chunk_min: int = 0,
        chunk_max: int | None = None,
        shared: _ProgramPrologue | None = None,
        carry: dict | None = None,
        allow_carry: bool = False,
    ):
        self.plan = plan
        self.chunk = chunk
        self.prefix = prefix  # 多段路径下按段命名（``s0_``/``s1_``...），避免跨段重名
        self.lane_base: tirx.PrimExpr = chunk * _int(plan.lanes)
        self.masked_tail = masked_tail  # 本发射块是否需要屏蔽尾部车道（尾块，见 tail_mask）
        # 本块的 chunk 区间 ``[chunk_min, chunk_max)``：守卫恒真的首/尾段里，满足
        # ``predicate_free_until >= chunk_max`` 或 ``predicate_free_from <= chunk_min``
        # 的守卫可以去掉掩码退回普通 ``vld``（见 ``_strip_mine`` 的切分）。
        self.chunk_min = chunk_min
        self.chunk_max = chunk_max
        self.memo: dict = {}
        self.lets: list = []  # chunk 体内的 Bind / Evaluate
        self.prologue: list = []  # chunk 无关的 Bind（掩码 / 步长斜坡）
        self.counter = 0
        self._mask_vars: dict = {}
        self._predicates: dict = {}
        self._ramps: dict = {}
        self._prologue_vars: set = set()
        self._shared_mask_ids: set = set()  # 由 _ProgramPrologue 提供、可跨段复用的掩码
        # 每个 buffer 已发射的写次数：装载 SSA 键带上代数，写后不复用写前装载。
        self._store_gen: dict = {}
        # 本段写过的 buffer：其上的标量读不能提到 prologue（可能读到写前/写后的不同值）。
        self._written_buffers = [store.spec.buffer for store in (*plan.pre_stores, *plan.stores, *plan.post_stores)]
        self._dups: dict = {}  # 循环不变 vdup 的复用表（prologue 内只绑一次）
        self._vci: dict = {}  # lane_index 的 vci 复用表（键含 lane_base，逐 chunk 自然失效）
        # 同层「写后读」转发：连续写 (buffer, 同下标) → 已算好的向量值，后面对同址的
        # 连续读直接复用，省一条 vld（d7 / d14b 的 `u = s[2*i]`）。
        self._forward: dict = {}
        # 段间寄存器直通：只读（allow_carry 才消费），但任何写都要作废对应项。
        self._carry = carry if carry is not None else {}
        self._allow_carry = allow_carry
        self._shared = shared
        # 累加器寄存器回写后的转发表：``post`` 对同一标量的读直接复用向量寄存器
        # （每 lane 就是该 lane 的累加结果），省一条 ``ONEPT`` 回读 + ``vdup``。
        self._promoted: dict = {}
        # 地址化简：合成轴（P1 展平）的分解式在 chunk 基址代入后（``(chunk*64)//8`` →
        # ``chunk*8``）必须折掉，否则会在生成的 C 代码里留下运行期整数除法。
        self.analyzer = arith.Analyzer()

    # -- SSA 绑定 -----------------------------------------------------------
    def bind(self, expr, target: list):
        var = tirx.Var(f"v{self.prefix}{self.counter}", expr.dtype)
        self.counter += 1
        target.append(tirx.Bind(var, expr))
        return var

    def prologue_bind(self, expr):
        """绑一个**与 chunk 无关**的值（提到 chunk 循环外只算一次）。"""
        var = self.bind(expr, self.prologue)
        self._prologue_vars.add(id(var))
        return var

    def ramp(self, stride: int, mask):
        """步长索引斜坡 ``vci(0) * stride``（``vgather2``/``vscatter`` 的每 lane 偏移）。

        斜坡与 chunk 无关：掩码也是 chunk 无关（整 chunk 的全 lane 谓词）时提到循环外
        只算一次；尾块/符号 extent 的谓词绑在 chunk 体内，此时就地发射，但在同一块内
        仍按 ``(stride, mask)`` 复用（多条访存共用一个斜坡）。多段 program 里若掩码
        本身是共享的，斜坡也登记到 :class:`_ProgramPrologue`，跨段只算一次。
        """
        key = (stride, id(mask))
        cached = self._ramps.get(key)
        if cached is not None:
            return cached
        expr = simd.vmuls(simd.vci(_int(0), "uint32"), _int(stride), mask)
        if id(mask) in self._prologue_vars:
            # 只有**共享掩码**上的斜坡才能提到 program 顶部：段内 prologue 的掩码（如
            # 常量尾块 update_mask）定义在各段 prologue 里，若把依赖它的斜坡放到
            # program 最前会造成「先用后定义」。
            if self._shared is not None and id(mask) in self._shared_mask_ids:
                cached = self._shared.ramps.get(key)
                if cached is None:
                    cached = tirx.Var(f"ramp_{stride}", expr.dtype)
                    self._shared.statements.append(tirx.Bind(cached, expr))
                    self._shared.ramps[key] = cached
                self._prologue_vars.add(id(cached))
            else:
                cached = self.prologue_bind(expr)
        else:
            cached = self.bind(expr, self.lets)
        self._ramps[key] = cached
        return cached

    # -- 掩码 ---------------------------------------------------------------
    def _shared_mask(self, key, name: str, expr):
        """多段 program 的共享掩码：同一个 (kind, bits) 全 program 只绑一次。"""
        if self._shared is None:
            return None
        var = self._shared.masks.get(key)
        if var is None:
            var = tirx.Var(name, _MASK_DTYPE)
            self._shared.statements.append(tirx.Bind(var, expr))
            self._shared.masks[key] = var
        self._prologue_vars.add(id(var))
        self._shared_mask_ids.add(id(var))
        return var

    def all_mask(self, bits: int):
        """全 lane 生效的谓词（``pset(bits, PAT_ALL)``），按宽度缓存、只绑一次。"""
        cached = self._mask_vars.get(bits)
        if cached is not None:
            return cached
        expr = simd.pset(bits, "PAT_ALL")
        var = self._shared_mask(("all", bits), f"mask_{bits}", expr)
        if var is None:
            var = tirx.Var(f"mask{self.prefix}{bits}", _MASK_DTYPE)
            self.prologue.append(tirx.Bind(var, expr))
            self._prologue_vars.add(id(var))
        self._mask_vars[bits] = var
        return var

    def lane0_mask(self, bits: int):
        """只保留 lane 0 的谓词（``pset(bits, "PAT_VL1")``）。

        跨 lane 归约（``vcadd``/``vcmax``/``vcmin``）的结果**只有 lane 0 有效**（硬件模板
        自己也是用 ``PAT_VL1`` 合并的），因此之后的链合并、旧值并入与回写都只需要一个
        车道。用 ``PAT_VL1`` 把后续算术收窄到 lane 0，不再对 63 个无效车道做无意义运算。
        """
        key = ("lane0", bits)
        cached = self._mask_vars.get(key)
        if cached is not None:
            return cached
        expr = simd.pset(bits, "PAT_VL1")
        var = self._shared_mask(key, f"lane0_{bits}", expr)
        if var is None:
            var = tirx.Var(f"lane0{self.prefix}{bits}", _MASK_DTYPE)
            self.prologue.append(tirx.Bind(var, expr))
            self._prologue_vars.add(id(var))
        self._mask_vars[key] = var
        return var

    def data_mask(self, dtype_like):
        """数据向量的执行掩码：元素类型 → 谓词宽度 → 全 lane 谓词（按宽度缓存）。"""
        return self.all_mask(_mask_bits(dtype_like))

    def tail_mask(self, bits: int):
        """尾块谓词：只有在本发射块需要屏蔽尾部车道时非空（整 chunk 块返回 ``None``）。

        常量 extent（``plan.tail`` 已知）用常量车道数，**提到 prologue**（与 chunk/外层
        迭代无关）；符号 extent 则在每个 chunk 运行时算 ``min(extent - chunk*lanes, lanes)``
        ——``update_mask`` 只保证 ``[0, lanes]`` 内有意义，超宽车道数不做假设（缺陷 A5）。

        掩码按宽度缓存：同一个块内只绑定一次，避免每条 ``vsts`` 前重复绑定。
        """
        if not self.masked_tail:
            return None
        key = ("tail", bits)
        cached = self._mask_vars.get(key)
        if cached is not None:
            return cached
        if self.plan.tail is not None:
            var = self.prologue_bind(simd.update_mask(_int(self.plan.tail), bits))
        else:
            remaining = self.plan.vector_axis.extent - self.chunk * _int(self.plan.lanes)
            count = tirx.min(remaining, _int(self.plan.lanes))
            var = self.bind(simd.update_mask(count, bits), self.lets)
        self._mask_vars[key] = var
        return var

    def predicate_combine(self, lhs, rhs, bits: int):
        """两个谓词的合取：``pand(lhs, rhs, ALL)``。

        ``bits`` 必须是**数据元素**的谓词粒度：pand 的执行掩码决定它覆盖多少个 lane
        （``256 / bits`` 个）。用固定 b32 会把 f16/int8 的 lane 64 之后全部清零，
        因此这里按调用方的数据元素宽度取掩码（R1）。
        """
        return self.bind(simd.pand(lhs, rhs, self.all_mask(bits)), self.lets)

    def predicate(self, node: ValueNode, bits: int):
        """条件节点 → 谓词寄存器（``bits`` 是数据元素的掩码宽度）。

        逐 lane 条件（``vcmp``/``vcmps``...）本身就是谓词寄存器，直接返回；循环不变
        条件（沿向量轴均匀，§1 Case II 的三元条件）在 lane 内恒真/恒假，广播成全 lane
        谓词 ``update_mask(cond ? lanes : 0)``——整 chunk 同开或同关。``pand``/``por``/
        ``pnot`` 按同一规则递归：含均匀子式时子式先变成整 chunk 掩码再做谓词运算。

        结果按（结构键, 掩码宽度）缓存：同一个条件既做装载守卫又做 ``vsel`` 的 mask 时
        只算一次（P0-3 的 im2col 形态否则会把整个谓词链重算一遍）。
        """
        key = (_value_key(node, self._store_gen), bits)
        cached = self._predicates.get(key)
        if cached is not None:
            return cached
        if node.is_uniform:
            lane_count = tirx.Select(self.scalar(node), _int(self.plan.lanes), _int(0))
            mask = self.bind(simd.update_mask(lane_count, bits), self.lets)
        elif node.recipe == "pnot":
            mask = self.bind(simd.pnot(self.predicate(node.args[0], bits), self.all_mask(bits)), self.lets)
        elif node.recipe == "pand":
            mask = self.predicate_combine(self.predicate(node.args[0], bits), self.predicate(node.args[1], bits), bits)
        elif node.recipe == "por":
            mask = self.bind(
                simd.por(self.predicate(node.args[0], bits), self.predicate(node.args[1], bits), self.all_mask(bits)),
                self.lets,
            )
        else:
            mask = self.value(node)
        self._predicates[key] = mask
        return mask

    def ramp_mask(self, bits: int):
        """步长斜坡的掩码：尾块谓词 / 全 lane（**不含**用户条件守卫）。

        斜坡只给 ``vgather2``/``vscatter`` 提供每 lane 偏移，未选中 lane 的偏移值
        无关紧要（访问本身被掩码屏蔽）；因此斜坡掩码必须与 chunk / 用户条件无关，
        否则会把 ``vci * stride`` 从循环外拖回循环内。
        """
        tail = self.tail_mask(bits)
        return tail if tail is not None else self.all_mask(bits)

    def guard_free(self, free_until: int, free_from: int | None) -> bool:
        """该守卫在本块是否恒真（本块 chunk 区间完全落在恒真首段或尾段内）。"""
        if free_from is not None and free_from <= self.chunk_min:
            return True
        return self.chunk_max is not None and free_until >= self.chunk_max

    def load_guard(self, node: ValueNode, bits: int):
        """装载掩码 = 尾块谓词 ∧ 惰性条件守卫（都不需要时返回 ``None``）。

        P0-3：``if_then_else`` 分支里的访存只在条件成立时合法，守卫必须合进装载谓词，
        未选中的 lane 不访问内存。

        守卫按 chunk 边界可省：``predicate_free_until/from`` 给出守卫恒真的首/尾 chunk
        区间（plan 期已在对应迭代域上证明）。本块完全落在其中一段时退回整 lane
        ``vld``，不再付 ``vgather2`` 的代价。
        """
        mask = self.tail_mask(bits)
        guard_node = node.predicate
        if guard_node is not None and self.guard_free(node.predicate_free_until, node.predicate_free_from):
            guard_node = None
        if guard_node is not None:
            guard = self.predicate(guard_node, bits)
            mask = guard if mask is None else self.predicate_combine(mask, guard, bits)
        return mask

    # -- 地址 ---------------------------------------------------------------
    def addr_indices(self, access: AccessInfo, uniform: bool = False):
        """把访存下标里的向量轴变量代入当前 chunk 的 lane 基址（或 0 = lane 内均匀）。"""
        base = _int(0) if uniform else self.lane_base
        return [self.analyzer.simplify(tirx.stmt_functor.substitute(index, {self.plan.vector_axis.var: base})) for index in access.indices]

    def addr(self, access: AccessInfo, uniform: bool = False):
        """一次访问的 ``BufferLoad``：向量轴变量代入当前 chunk 的 lane 基址。"""
        return tirx.BufferLoad(access.buffer, self.addr_indices(access, uniform))

    def index_addr(self, access: AccessInfo):
        """索引向量通道的基址：``BufferLoad(buffer, 全 0)`` = 缓冲区起点。

        索引向量（``value.index_value``）给出的是**绝对**元素偏移，因此基址必须落在
        缓冲区起点，而不是某个 chunk 的 lane 基址。
        """
        return tirx.BufferLoad(access.buffer, [_int(0) for _ in access.indices])

    # -- 标量操作数 ---------------------------------------------------------
    def scalar(self, node: ValueNode):
        if node.recipe in ("imm", "uniform_expr"):
            return node.scalar
        if node.recipe == "uniform_load":
            return self.addr(node.access, uniform=True)
        if node.recipe == "vbrc":
            return self.scalar(node.args[0])
        raise Unsupported(f"cannot scalarize vector value '{node.recipe}'")

    # -- 值 ----------------------------------------------------------------
    def note_write(self, buffer) -> None:
        """记录一次写：同一 buffer 的后续装载不再复用写之前的 SSA。

        同时作废该 buffer 上的**写后读转发**（:attr:`_forward`）：这是一次新的写，之前
        转发的向量值可能已经不是内存里的值。连续写会在随后 :meth:`note_store` 里重新登记
        自己的值，因此正常转发链路不受影响；而 scatter / 标量写 / 回写这类不登记转发的
        写，从此不会留下"陈旧但仍可命中"的转发项（R8）。
        """
        key = _buffer_key(buffer)
        self._store_gen[key] = self._store_gen.get(key, 0) + 1
        if self._forward:
            for fwd_key in [k for k in self._forward if k[0] == key]:
                del self._forward[fwd_key]
        # 段间直通表同理：写入同一 buffer 后，之前登记的寄存器值不再等于内存。
        if self._carry:
            for carry_key in [k for k in self._carry if k[0] == key]:
                del self._carry[carry_key]

    def note_store(self, store: StorePlan, value) -> None:
        """记录一条**连续**写：同址的后续 ``vload`` 可直接复用该向量值（写后读转发）。"""
        if store.kind == AccessClass.CONTIGUOUS:
            self._forward[(_buffer_key(store.spec.buffer), tuple(str(index) for index in store.spec.indices))] = value

    def clear_forward(self) -> None:
        """跨越循环边界时清空转发表（循环体内定义的 SSA 不能提到循环外使用）。"""
        self._forward.clear()

    def promote(self, node: ValueNode, vector) -> None:
        """登记「该地址的读已提升成向量寄存器」（累加器回写之后）。"""
        self._promoted[_value_key(node, self._store_gen)] = vector

    # -- 段间寄存器直通 -----------------------------------------------------
    def register_carry(self, store: StorePlan, value, *, axis: tirx.Var) -> None:
        """把一条**整向量、无谓词**连续写的结果登记为下一段可直接复用的寄存器值。

        由 :func:`_emit_parallel_axis_parts` 在累加器回写之后调用（回写值就是
        ``local.var`` 寄存器本身）。它等价于「内存里的这段区间 == 该寄存器」，
        因此下一段同址的连续读可以省掉 ``vld``。

        适用前提由调用方保证：该段**单 chunk、无尾块**（lane 基址 0、所有 lane 有效）、
        写无用户谓词、地址沿 inner 轴不变、且回写没有被 post 接管。键与 ``_forward``
        一致（buffer 结构键 + 原始下标字符串），值多带一份**向量轴变量**：消费段必须
        是同一条轴的 lane 划分，否则 lane k 对应的迭代不同。
        """
        key = (_buffer_key(store.spec.buffer), tuple(str(index) for index in store.spec.indices))
        self._carry[key] = (axis, value)

    def _carry_entry(self, fwd_key, dtype):
        """查段间直通表；只有本段允许消费且向量轴变量一致时才返回寄存器值。"""
        if not self._allow_carry or not self._carry:
            return None
        entry = self._carry.get(fwd_key)
        if entry is None:
            return None
        axis, value = entry
        if not axis.same_as(self.plan.vector_axis.var):
            return None
        if str(value.dtype) != str(dtype):
            return None
        return value

    def value(self, node: ValueNode):
        key = _value_key(node, self._store_gen)
        if key in self.memo:
            return self.memo[key]
        expr = self._build(node)
        self.memo[key] = expr
        return expr

    def vector_operand(self, node: ValueNode):
        """把一个 lane 内均匀的标量提升成向量寄存器（``vdup``），否则走 ``value``。

        与 chunk / 外层轴 / 本段写无关的标量（常量、只读 buffer 的标量读）提到 prologue
        只绑一次；否则仍在 ``lets`` 里逐 chunk 重算（例如 post 里读主循环刚写过的 ``s[i]``）。
        """
        promoted = self._promoted.get(_value_key(node, self._store_gen))
        if promoted is not None:
            return promoted  # 已提升成向量寄存器的同址读（累加器回写后的 post）
        if node.recipe in _SCALAR_ONLY_RECIPES:
            return self.dup(node)
        return self.value(node)

    def dup(self, node: ValueNode):
        """标量 → 向量广播（``vdup``），循环不变时提到 prologue 并按值复用。"""
        elem = _elem_dtype(node.dtype)
        scalar = self.scalar(node)
        if self._scalar_hoistable(node):
            key = (str(scalar), elem)
            cached = self._dups.get(key)
            if cached is not None:
                return cached
            var = self.prologue_bind(simd.vdup(scalar, elem, self.data_mask(elem), _MODE))
            self._dups[key] = var
            return var
        return self.bind(simd.vdup(scalar, elem, self.data_mask(elem), _MODE), self.lets)

    def _scalar_hoistable(self, node: ValueNode) -> bool:
        """标量操作数是否与 chunk / 外层轴 / 本段的写无关（可提到 prologue 只算一次）。"""
        expr = self.scalar(node)
        if not isinstance(expr, tirx.PrimExpr):
            return True  # 纯 Python 常量
        axis_vars = [axis.var for axis in self.plan.axes]
        written = {_buffer_key(buffer) for buffer in self._written_buffers}
        blocked: list[bool] = []

        def visit(sub):
            if isinstance(sub, tirx.BufferLoad) and _buffer_key(sub.buffer) in written:
                blocked.append(True)
            if isinstance(sub, tirx.Var) and any(sub.same_as(var) for var in axis_vars):
                blocked.append(True)

        tirx.stmt_functor.post_order_visit(expr, visit)
        return not blocked

    def _build(self, node: ValueNode):
        recipe = node.recipe
        args = node.args

        if recipe == "imm":
            return node.scalar
        if recipe == "lane_index":
            # 向量轴的绝对取值 = chunk 基址 + lane id（用于谓词比较，如 ``i < j``）。
            # 同一 chunk 内只需一条 ``vci``：按 lane_base 缓存（展开归约逐 chunk 变基址，
            # 键随基址变化自然失效）。
            elem = _elem_dtype(node.dtype)
            key = (str(self.lane_base), elem)
            cached = self._vci.get(key)
            if cached is not None:
                return cached
            var = self.bind(simd.vci(self.lane_base, elem), self.lets)
            self._vci[key] = var
            return var
        if recipe == "vload":
            # 同层写后读转发：刚写过同址的连续 store，直接复用它的向量值。
            fwd_key = (_buffer_key(node.access.buffer), tuple(str(index) for index in node.access.indices))
            fwd = self._forward.get(fwd_key)
            if fwd is not None and str(fwd.dtype) == node.dtype:
                return fwd
            # 段间寄存器直通：上一段把同址的整向量写结果登记在 ``_carry`` 里。只在
            # 本段「单 chunk（lane 基址 0）、无尾块/守卫、向量轴变量一致」时才成立
            # ——这些条件由 ``_emit_program`` 的 ``allow_carry`` 与登记项里的轴变量
            # 共同保证，见 :meth:`_Emitter.register_carry`。
            if not self.masked_tail and node.predicate is None:
                carried = self._carry_entry(fwd_key, node.dtype)
                if carried is not None:
                    return carried
            bits = _mask_bits(node.dtype)
            mask = self.load_guard(node, bits)
            if mask is None:
                return self.bind(simd.vld(self.addr(node.access), "NORM"), self.lets)
            # ``vld`` 没有谓词形式：尾块/惰性条件守卫退化成带谓词的 ``vgather2``
            # （相对 lane 偏移），否则整 lane 读会越过 buffer 尾部或访问未选中 lane
            # 的地址（缺陷 D10 / P0-3）。
            return self.bind(simd.vgather2(self.addr(node.access), simd.vci(_int(0), "uint32"), mask), self.lets)
        if recipe == "vload_strided":
            # simd 没有带步长装载：用 ``vgather2`` + 步长索引斜坡表达。
            _, stride = node.access.axis_class(self.plan.vector_axis)
            bits = _mask_bits(node.dtype)
            guard = self.load_guard(node, bits)
            if guard is None:
                guard = self.all_mask(bits)
            index = self.ramp(stride, self.ramp_mask(bits))
            return self.bind(simd.vgather2(self.addr(node.access), index, guard), self.lets)
        if recipe == "gather":
            # 每 lane 索引向量通道（改进 1）：``args[0]`` 是绝对元素偏移向量。
            bits = _mask_bits(node.dtype)
            mask = self.load_guard(node, bits)
            if mask is None:
                mask = self.all_mask(bits)
            index = _unsigned_index(self.value(args[0]))
            return self.bind(simd.vgather2(self.index_addr(node.access), index, mask), self.lets)
        if recipe in ("uniform_load", "uniform_expr", "vbrc"):
            operand = node if recipe != "vbrc" else args[0]
            return self.dup(operand)
        if recipe in _VEC_SCALAR_BINARY.values():
            vector = self.value(args[0])
            promoted = self.memo.get(_value_key(args[1], self._store_gen))
            if promoted is not None:
                # 「标量」操作数其实已被提升成向量寄存器（``local.var`` 累加器的旧值，
                # 每 lane 不同）→ 标量-向量变体不适用，退回向量-向量指令。
                return self.bind(
                    getattr(simd, _SCALAR_TO_VECTOR_BINARY[recipe])(vector, promoted, self.data_mask(node.dtype), _MODE),
                    self.lets,
                )
            return self.bind(getattr(simd, recipe)(vector, self.scalar(args[1]), self.data_mask(node.dtype), _MODE), self.lets)
        if recipe in ("vshrs", "vshls"):
            # 常量移位量走标量-向量移位（`vshrs` = 算术右移，即整型 FloorDiv by 2^k）。
            return self.bind(getattr(simd, recipe)(self.value(args[0]), node.scalar, self.data_mask(node.dtype), _MODE), self.lets)
        if recipe in _ELEMENTWISE_UNARY.values():
            return self.bind(getattr(simd, recipe)(self.value(args[0]), self.data_mask(node.dtype), _MODE), self.lets)
        if recipe in _ELEMENTWISE_BINARY.values():
            return self.bind(getattr(simd, recipe)(self.value(args[0]), self.value(args[1]), self.data_mask(node.dtype), _MODE), self.lets)
        if recipe == "vcmp":
            return self.bind(simd.vcmp(self.value(args[0]), self.value(args[1]), self.data_mask(args[0].dtype), node.detail), self.lets)
        if recipe == "vcmps":
            return self.bind(simd.vcmps(self.value(args[0]), self.scalar(args[1]), self.data_mask(args[0].dtype), node.detail), self.lets)
        if recipe == "vsel":
            # ``args = [cond, true, false]``；``vsel`` 的顺序是 ``(t, f, mask)``。
            # 循环不变条件（沿向量轴均匀）→ 广播成全 lane 谓词（整个 chunk 同真同假）；
            # 逐 lane 条件（vcmp/vcmps/pand...）本身就是谓词寄存器，直接用。
            true_value = self.vector_operand(args[1])
            false_value = self.vector_operand(args[2])
            return self.bind(simd.vsel(true_value, false_value, self.predicate(args[0], _mask_bits(_elem_dtype(node.dtype)))), self.lets)
        if recipe in ("pand", "por"):
            mask = self.all_mask(_PREDICATE_MASK_BITS)
            return self.bind(getattr(simd, recipe)(self.value(args[0]), self.value(args[1]), mask), self.lets)
        if recipe == "pnot":
            mask = self.all_mask(_PREDICATE_MASK_BITS)
            return self.bind(simd.pnot(self.value(args[0]), mask), self.lets)
        if recipe == "vcvt":
            src = self.value(args[0])
            target = node.detail
            mask = self.all_mask(min(_mask_bits(src.dtype), _mask_bits(target)))
            return self.bind(simd.vcvt(src, target, mask), self.lets)
        raise Unsupported(f"no emitter for value recipe '{recipe}'")


def _store_mask(store: StorePlan, emitter: _Emitter, bits: int):
    """chunk 内 store 的最终谓词 = 用户谓词 ∧ 尾块掩码（都为空时用全 1）。

    用户谓词必须走 :meth:`_Emitter.predicate`（而不是 ``value``）：沿向量轴均匀的
    条件（如 ``if a[0] > 0: c[i] = a[i]``）是 bool 标量，``value`` 会把它当数据向量
    去 ``vdup``，而 bool 没有谓词宽度（emit 期 ``ValueError``，R2）。
    """
    mask = None
    if store.mask is not None and not emitter.guard_free(store.mask_free_until, store.mask_free_from):
        # 谓词在本块恒真（plan 期已证）→ 直接用全 lane 掩码，省掉 `vcmps` 链。
        mask = emitter.predicate(store.mask, bits)
    tail = emitter.tail_mask(bits)
    if tail is not None:
        mask = tail if mask is None else emitter.predicate_combine(mask, tail, bits)
    if mask is None:
        mask = emitter.all_mask(bits)
    return mask


def _emit_vector_write(store: StorePlan, emitter: _Emitter, bits: int, mask) -> None:
    """发射一条 ``vsts``：lane 内均匀（``UNIFORM``）用 ``ONEPT``，否则连续 ``NORM``。

    ``mask`` 由调用方给：chunk 内写用 :func:`_store_mask`（含尾块谓词）；与 serial 循环
    同级的 pre/post 写只用用户谓词（**不能**含尾块谓词，见 :func:`_emit_outer_store`）。
    """
    value = emitter.vector_operand(store.value)
    uniform = store.kind == AccessClass.UNIFORM
    dist = f"ONEPT_B{bits}" if uniform else f"NORM_B{bits}"
    emitter.lets.append(tirx.Evaluate(simd.vsts(emitter.addr(store.access, uniform=uniform), value, mask, dist)))
    emitter.note_write(store.spec.buffer)
    emitter.note_store(store, value)


def _emit_store(plan: VectorPlan, store: StorePlan, emitter: _Emitter) -> None:
    """发射一次写入：连续/均匀写走 ``vsts``，其余走每 lane 索引 ``vscatter``。"""
    elem = _elem_dtype(store.spec.buffer.dtype)
    bits = _mask_bits(elem)
    mask = _store_mask(store, emitter, bits)

    if store.kind in (AccessClass.CONTIGUOUS, AccessClass.UNIFORM):
        _emit_vector_write(store, emitter, bits, mask)
        return

    value = emitter.vector_operand(store.value)
    if store.kind == AccessClass.STRIDED:
        # 常量**正**步长写：chunk lane 基址 + 步长斜坡（与 ``vload_strided`` 对称）。
        # 负步长落到下面的索引向量通道：无符号斜坡乘负步长表达不了负方向（缺陷 D8）。
        _, stride = store.access.axis_class(plan.vector_axis)
        if stride is not None and stride > 0:
            index = emitter.ramp(stride, emitter.ramp_mask(bits))
            emitter.lets.append(tirx.Evaluate(simd.vscatter(value, emitter.addr(store.access), index, mask)))
            emitter.note_write(store.spec.buffer)
            return

    # PERMUTE / PIECEWISE / INDIRECT / 负步长 STRIDED：绝对元素偏移索引向量 + 缓冲区起点基址。
    if store.index is None:  # 防御：这条通道必须带每 lane 索引向量
        raise Unsupported(f"no per-lane index vector for a {store.kind.value} store")
    index = _unsigned_index(emitter.value(store.index))
    emitter.lets.append(tirx.Evaluate(simd.vscatter(value, emitter.index_addr(store.access), index, mask)))
    emitter.note_write(store.spec.buffer)


def _emit_outer_store(store: StorePlan, emitter: _Emitter) -> None:
    """发射一条与 serial 循环**同级**的语句（pre/post，方案 §6 第二步）。

    这类语句不含 serial 向量轴变量，沿该轴 ``Δ==0``，因此每个外层迭代只执行一次。
    取值 lane 内均匀且无谓词时直接发**标量 store**（不再 ``vdup`` + 64 lane ``ONEPT``）；
    其余情况仍用 ``ONEPT`` 单点写。
    """
    if store.kind != AccessClass.UNIFORM:
        raise Unsupported(f"a statement outside the serial loop must be uniform along the vector axis (got {store.kind.value})")
    elem = _elem_dtype(store.spec.buffer.dtype)
    bits = _mask_bits(elem)
    if store.mask is None and store.value.is_uniform:
        try:
            scalar = emitter.scalar(store.value)
        except Unsupported:
            scalar = None
        if scalar is not None:
            # 下标同样代入 lane 均匀基址（与向量通道一致；UNIFORM 时向量轴变量本不出现，
            # 这里只是防御性地保持两条路径的地址完全同源）。
            emitter.lets.append(tirx.BufferStore(store.spec.buffer, scalar, emitter.addr_indices(store.access, uniform=True)))
            emitter.note_write(store.spec.buffer)
            return
    mask = emitter.predicate(store.mask, bits) if store.mask is not None else emitter.all_mask(bits)
    _emit_vector_write(store, emitter, bits, mask)


def _strip_mine(plan: VectorPlan, chunk: tirx.Var, make_block) -> tuple[list, list]:
    """按向量轴 extent 生成 chunk 调度骨架：符号 extent / 无尾块 / 有尾块。

    ``make_block(chunk_index, masked_tail, label, chunk_min, chunk_max)`` 返回
    ``(prologue, lets)``：``chunk_index`` 是 chunk 循环变量，或在单块展开时直接是常量
    基址；``label`` 用于让同段内的 SSA 名不撞车（主体 ``""`` / 尾块 ``"tail_"`` /
    守卫恒真段 ``"gpre_"`` / ``"gfree_"``）；``[chunk_min, chunk_max)`` 是该块的 chunk
    区间（守卫可省判定用）。

    三个发射器（逐元素、并行轴三段、归约循环）共用这一份骨架——此前各自维护一遍，
    任何尾块/展开修正都要改三处。

    * **符号 extent**：主循环跑 ``E // lanes`` 个整块（满 lane），尾块只在
      ``E % lanes != 0`` 时执行一次（运行期夹紧掩码 + 谓词装载，缺陷 D9/A5）；
    * **常量无尾块**：``chunks == 1`` 展开成直线代码，否则整块区间按守卫边界切分；
    * **常量有尾块**：前 ``chunks-1`` 个整块 + 单次尾块；
    * **守卫边界**：按 ``plan.guard_boundaries`` 把整块区间切成若干段；每段携带
      ``[chunk_min, chunk_max)``，守卫在段内恒真时装载回退普通 ``vld``。
    """
    prologue: list = []

    def block(index, masked_tail: bool, label: str, chunk_min: int = 0, chunk_max: int | None = None):
        block_prologue, lets = make_block(index, masked_tail, label, chunk_min, chunk_max)
        prologue.extend(block_prologue)
        return lets

    def range_blocks(lo: int, hi: int, label: str):
        """整块区间 ``[lo, hi)``：单轮展开成直线代码，否则一个 chunk 循环。"""
        if hi <= lo:
            return []
        if hi - lo == 1:
            return [_seq(block(_int(lo), False, label, lo, hi))]
        return [tirx.For(chunk, _int(lo), _int(hi), tirx.ForKind.SERIAL, _seq(block(chunk, False, label, lo, hi)))]

    if plan.chunks is None:
        extent = plan.vector_axis.extent
        blocks = [
            tirx.For(
                chunk,
                0,
                tirx.FloorDiv(extent, _int(plan.lanes)),
                tirx.ForKind.SERIAL,
                _seq(block(chunk, False, "")),
            )
        ]
        tail_chunk = tirx.FloorDiv(extent, _int(plan.lanes))
        blocks.append(tirx.IfThenElse(tirx.FloorMod(extent, _int(plan.lanes)) > 0, _seq(block(tail_chunk, True, "tail_")), None))
        return prologue, blocks

    cuts = plan.guard_boundaries

    def split_range(lo: int, hi: int) -> list:
        """把 ``[lo, hi)`` 按守卫切点分段，每段独立发射（标签区分 SSA 名）。"""
        points = [lo, *(cut for cut in cuts if lo < cut < hi), hi]
        blocks = []
        for index in range(len(points) - 1):
            label = "" if len(points) == 2 else f"gr{index}_"
            blocks.extend(range_blocks(points[index], points[index + 1], label))
        return blocks

    if not plan.tail:
        if plan.chunks == 1:
            # 单次循环展开成直线代码：``for chunk in range(1)`` 只是给后端添乱。
            return prologue, block(_int(0), False, "", 0, 1)
        return prologue, split_range(0, plan.chunks)

    # 尾块**恒为单次**：直接展开成直线代码（lane 基址 = full * lanes 常量）。
    full = plan.chunks - 1
    blocks = split_range(0, full)
    blocks.append(_seq(block(_int(full), True, "tail_", full, full + 1)))
    return prologue, blocks


def _emit_elementwise_chunks(
    plan: VectorPlan,
    chunk: tirx.Var,
    prefix: str,
    *,
    shared: _ProgramPrologue | None = None,
    carry: dict | None = None,
    allow_carry: bool = False,
) -> tuple[list, list]:
    """逐元素主路径的 chunk 发射（尾块单独成块），返回 ``(prologue, blocks)``。

    调度骨架见 :func:`_strip_mine`。每块用独立的 :class:`_Emitter`（SSA 名不撞车），
    因此各自的 ``prologue``（``pset`` 全 lane 掩码）也必须一并返回，由调用方提到最外层。
    """

    def make_block(chunk_index, masked_tail: bool, label: str, chunk_min: int = 0, chunk_max: int | None = None):
        body = _Emitter(
            plan,
            chunk_index,
            f"{prefix}{label}",
            masked_tail=masked_tail,
            chunk_min=chunk_min,
            chunk_max=chunk_max,
            shared=shared,
            carry=carry,
            allow_carry=allow_carry,
        )
        for store in plan.stores:
            _emit_store(plan, store, body)
        return body.prologue, body.lets

    return _strip_mine(plan, chunk, make_block)


def emit_plan(plan: VectorPlan | VectorProgram) -> tirx.Stmt:
    """把 :class:`VectorPlan` / :class:`VectorProgram` 发射成 ``T.simd.*`` TIR。

    外层非向量轴退化成 serial 循环，向量轴被 strip-mine 成 ``chunk``
    （``for chunk in serial(ceil(E/L))``）；chunk 体内的 SSA 用 ``Bind`` 表达。
    归约（§2 Case II）按 chunk 数分两种策略：短归约编译期展开 + 多路累加链，
    长归约 / 符号 extent 保留真实 chunk 循环 + ``local.var`` 累加器（见模块 docstring）。

    §6 第二步的 ``pre``/``post``（与 serial 循环同级的兄弟语句）沿向量轴均匀，
    发射在 chunk 循环**之外**，顺序为 ``pre → 主体 → post``。

    :class:`VectorProgram`（1..N 段）逐段发射并保持源码顺序：``program.axes`` 里的轴
    在**最外层统一包一次**（``chunk`` 变量按段命名区分），各段只包自己剩下的轴。
    单独发射一个 :class:`VectorPlan`（``hoisted`` 为空）等价于「该段自己包全部轴」，
    与 ``build_plan`` 恒返回 program 之前的发射结果**逐节点一致**。
    """
    if isinstance(plan, VectorProgram):
        return _emit_program(plan)
    prologue, body = _emit_segment_stmts(plan, tirx.Var("chunk", "int32"), hoisted=[], single=True)
    if prologue:
        return tirx.SeqStmt([*prologue, body])
    return body


def _load_buffer_keys(expr) -> list:
    """表达式里读到的 buffer 结构键（含 ``vld``/``vgather2`` 与直接 ``BufferLoad``）。"""
    found: list = []

    def visit(node):
        if isinstance(node, tirx.BufferLoad):
            found.append(_buffer_key(node.buffer))

    tirx.stmt_functor.post_order_visit(expr, visit)
    return found


def _split_loop_invariant(lets: list, inner: list[Axis], written_keys: set, pinned: list | None = None) -> tuple[list, list]:
    """把 chunk 体内的纯 SSA 绑定拆成「可提到 inner 循环外」与「必须留在循环内」两段。

    只提 ``Bind``（纯值），``Evaluate``（``vsts``/``vscatter`` 等写）一律不动。
    可提条件：取值不引用 inner 轴变量、不引用循环内定义的 SSA、也不读**本段会写**
    的 buffer（否则每次迭代读到的内存值不同，例如 ``s[i] += a[i,j]`` 的旧值装载）。

    这是纯静态的 LICM：像 ``c3_im2col`` 的 ``vci/vand/vshrs``、``c4_tiled`` 的
    地址分解、``p_serial_reduce_then_use`` 段 1 的 ``vld s + vmuls`` 都会从
    8–16 轮的 inner 循环里提到循环外，只算一次。
    """
    inner_vars = [axis.var for axis in inner]
    loop_defined: list = list(pinned or [])  # 已在外层定义、且不能跨到循环外的 SSA（如 pre lets）
    hoisted: list = []
    kept: list = []
    for stmt in lets:
        if isinstance(stmt, tirx.Bind):
            free = _free_vars(stmt.value)
            blocked = any(any(var.same_as(inner_var) for inner_var in inner_vars) for var in free)
            blocked = blocked or any(any(var.same_as(loop_var) for loop_var in loop_defined) for var in free)
            blocked = blocked or any(key in written_keys for key in _load_buffer_keys(stmt.value))
            if not blocked:
                hoisted.append(stmt)
                continue
            loop_defined.append(stmt.var)
        kept.append(stmt)
    return hoisted, kept


def _store_is_loop_invariant(store: StorePlan, plan: VectorPlan, written_keys: set) -> bool:
    """pre 写是否完全不依赖任何轴变量（且不读本段写过的 buffer）→ 可整段提到最外层。"""
    axis_vars = [axis.var for axis in plan.axes]
    exprs = [*store.spec.indices, store.spec.value]
    if store.spec.mask is not None:
        exprs.append(store.spec.mask)
    blocked: list[bool] = []

    def visit(node):
        if isinstance(node, tirx.Var) and any(node.same_as(var) for var in axis_vars):
            blocked.append(True)
        if isinstance(node, tirx.BufferLoad) and _buffer_key(node.buffer) in written_keys:
            blocked.append(True)

    for expr in exprs:
        tirx.stmt_functor.post_order_visit(expr, visit)
    return not blocked


def _emit_segment_stmts(
    plan: VectorPlan,
    chunk: tirx.Var,
    *,
    hoisted: list[Axis],
    prefix: str = "",
    single: bool = False,
    shared: _ProgramPrologue | None = None,
    carry: dict | None = None,
    allow_carry: bool = False,
) -> tuple[list, tirx.Stmt]:
    """发射**一段** :class:`VectorPlan` 的语句体，返回 ``(prologue, body)``。

    ``prologue`` 是 chunk 无关的掩码绑定（调用方负责提到最前）；``body`` 已按非向量轴
    包裹成嵌套 serial 循环。``hoisted`` 是**已由调用方提到最外层**的轴（多段 program 的
    共享并行轴前缀，见 ``plan._outer_axes``）：按**变量身份**跳过它们，而不是笼统跳过所有
    并行轴——否则当向量轴就是某根外层并行轴时，夹在其后的并行轴会被漏包。

    当向量轴是**外层并行轴**且存在 ``pre``/``post`` 时走
    :func:`_emit_parallel_axis_parts`：chunk 循环包住 ``pre → (serial 轴) main → post``，
    兄弟语句变成与 serial 循环同级的一次向量写，而不是每个外层迭代一次的 ``ONEPT``。
    """
    # 并行向量轴 + 内层轴（serial **或** parallel）：统一走手工三段结构
    # （pre → for inner: main → post）。内层轴一律退化成 serial 循环（并行语义允许任意
    # 顺序），LICM 与累加器提升才对两者一视同仁；只对内层 serial 生效会让
    # ``d2/d3_*`` 这种 ``T.Parallel(M, K)`` 形态每轮白做一次 vld/vsts。
    inner_axes = [axis for axis in plan.axes if axis.index > plan.vector_axis.index and not any(axis.var.same_as(h.var) for h in hoisted)]
    if plan.vector_axis.is_parallel and inner_axes and plan.reduce is None:
        return _emit_parallel_axis_parts(plan, chunk, prefix, hoisted, single, shared=shared, carry=carry, allow_carry=allow_carry)

    emitter = _Emitter(plan, chunk, prefix, shared=shared, carry=carry, allow_carry=allow_carry)
    written_keys = {_buffer_key(store.spec.buffer) for store in (*plan.pre_stores, *plan.stores, *plan.post_stores)}
    invariant_pre = [store for store in plan.pre_stores if single and _store_is_loop_invariant(store, plan, written_keys)]
    invariant_ids = {id(store) for store in invariant_pre}
    loop_pre = [store for store in plan.pre_stores if id(store) not in invariant_ids]

    for store in loop_pre:  # 前置语句：chunk 循环之前，每个外层迭代一次
        _emit_outer_store(store, emitter)
    pre_lets = emitter.lets
    emitter.lets = []
    # pre 与主体被 chunk 循环隔开：pre 的写后读转发不能跨到主体/后置。
    emitter.clear_forward()

    if plan.reduce is not None:
        _emit_reduce(plan, emitter)  # 填 emitter.lets（编译期展开 chunk）
        main_lets = emitter.lets
        emitter.lets = []
        main_prologue: list = []
    else:
        main_prologue, main_lets = _emit_elementwise_chunks(plan, chunk, prefix, shared=shared, carry=carry, allow_carry=allow_carry)

    for store in plan.post_stores:  # 后置语句：chunk 循环之后，每个外层迭代一次
        _emit_outer_store(store, emitter)
    post_lets = emitter.lets

    # 单段且 chunk 体已展开成直线代码时，把只依赖向量轴 / 常量、且不读本段写过的
    # buffer 的 SSA 提到外层轴循环之前（如 ``c4_tiled`` 的 vci/vshrs/vand、
    # ``c3_im2col`` 的掩码/地址分解）。多段路径的 prologue 是跨段共享的，不做。
    wrap_axes = [
        axis for axis in plan.axes if not axis.var.same_as(plan.vector_axis.var) and not any(axis.var.same_as(h.var) for h in hoisted)
    ]
    hoisted_body: list = []
    flat_main: list = []
    for stmt in main_lets:
        if isinstance(stmt, tirx.SeqStmt):
            flat_main.extend(stmt.seq)
        else:
            flat_main.append(stmt)
    if single and (wrap_axes or hoisted) and flat_main and all(isinstance(stmt, (tirx.Bind, tirx.Evaluate)) for stmt in flat_main):
        pre_vars = [stmt.var for stmt in pre_lets if isinstance(stmt, tirx.Bind)]
        hoisted_body, flat_main = _split_loop_invariant(flat_main, [*wrap_axes, *hoisted], written_keys, pre_vars)
        main_lets = flat_main

    body = _seq([*pre_lets, *main_lets, *post_lets])
    for axis in reversed(plan.axes):
        if axis.var.same_as(plan.vector_axis.var):
            continue
        if any(axis.var.same_as(h.var) for h in hoisted):
            continue
        body = tirx.For(axis.var, 0, axis.extent, tirx.ForKind.SERIAL, body)
    hoisted_lets: list = []
    if invariant_pre:  # 与所有轴无关的 pre 写：整段提到最外层，只执行一次
        pre_emitter = _Emitter(plan, _int(0), f"{prefix}hopre_", shared=shared, carry=carry)
        for store in invariant_pre:
            _emit_outer_store(store, pre_emitter)
        emitter.prologue.extend(pre_emitter.prologue)
        hoisted_lets = pre_emitter.lets
    return [*emitter.prologue, *main_prologue, *hoisted_body, *hoisted_lets], body


def _emit_parallel_axis_parts(
    plan: VectorPlan,
    chunk: tirx.Var,
    prefix: str,
    hoisted: list[Axis],
    single: bool = False,
    *,
    shared: _ProgramPrologue | None = None,
    carry: dict | None = None,
    allow_carry: bool = False,
) -> tuple[list, tirx.Stmt]:
    """并行向量轴 + 内层轴：``for chunk: pre; (inner 轴) main; post``（方案 §6 第二步）。

    ``pre``/``post`` 在源程序里是 serial 循环的兄弟语句，沿并行轴向量化后它们与 inner
    循环同级——每个 chunk 执行一次，**不能**被 inner 轴再包一层（否则 `s[i]=0` 这类
    初值会被每个 inner 迭代重置）。因此这里手工组织三段，而不是走
    :func:`_emit_segment_stmts` 的「所有非向量轴包住整段」结构。

    每个 block 用独立 :class:`_Emitter`（SSA 名不撞车），``prologue``（``pset`` 全 lane
    掩码）一并返回给调用方提到最外层。
    """
    assert plan.reduce is None, "a parallel vector axis with pre/post must go through the elementwise path"
    inner = [axis for axis in plan.axes if axis.index > plan.vector_axis.index and not any(axis.var.same_as(h.var) for h in hoisted)]
    prologue: list = []
    written_keys = {_buffer_key(store.spec.buffer) for store in (*plan.pre_stores, *plan.stores, *plan.post_stores)}
    # 累加器提升是**计划期**的单一决策（``StorePlan.promote``，由
    # :func:`model.accumulator_candidate` 在 build 期填好）；发射层只消费，不再重算。
    candidates = [(store, store.promote) for store in plan.stores if store.promote is not None]
    # 会被折叠成累加器寄存器初值的 pre 不能提到最外层：折叠意味着**不再写内存**，
    # 而提层后的累加器 init 依赖 pre 在内存里留下的初值。
    foldable = {
        id(pre)
        for store, _ in candidates
        for pre in plan.pre_stores
        if pre.mask is None
        and pre.value.is_uniform
        and _same_buffer(pre.spec.buffer, store.spec.buffer)
        and _same_indices(tuple(pre.spec.indices), tuple(store.spec.indices))
    }
    invariant_pre = [
        store for store in plan.pre_stores if single and id(store) not in foldable and _store_is_loop_invariant(store, plan, written_keys)
    ]
    invariant_ids = {id(store) for store in invariant_pre}
    loop_pre = [store for store in plan.pre_stores if id(store) not in invariant_ids]

    def finish(body):
        # 与所有轴无关的 pre 写：整段提到最外层，只执行一次（典型 d11 的 c[b[0]]=1.0）。
        if invariant_pre:
            pre_emitter = _Emitter(plan, _int(0), f"{prefix}hopre_", shared=shared, carry=carry)
            for store in invariant_pre:
                _emit_outer_store(store, pre_emitter)
            prologue.extend(pre_emitter.prologue)
            return [*prologue, *pre_emitter.lets], body
        return prologue, body

    def emit_body(chunk_var, masked_tail: bool, body_prefix: str, chunk_min: int = 0, chunk_max: int | None = None):
        body_emitter = _Emitter(
            plan,
            chunk_var,
            body_prefix,
            masked_tail=masked_tail,
            chunk_min=chunk_min,
            chunk_max=chunk_max,
            shared=shared,
            carry=carry,
            allow_carry=allow_carry,
        )
        # 同址 pre 写若是均匀初值，折叠成寄存器初值（不再单独发一次内存写）。
        folded: dict = {}
        consumed_pre: set = set()
        for store, _ in candidates:
            for pre in loop_pre:
                if id(pre) in consumed_pre or pre.mask is not None or not pre.value.is_uniform:
                    continue
                if _same_buffer(pre.spec.buffer, store.spec.buffer) and _same_indices(tuple(pre.spec.indices), tuple(store.spec.indices)):
                    # 折叠 = 这条 pre 不再写内存；同层另一条 pre 若读该 buffer，会读到
                    # 未被更新的旧值，此时保持内存写、累加器从内存装载初值。
                    if any(other is not pre and _reads_buffer(other.value, pre.spec.buffer) for other in loop_pre):
                        continue
                    folded[id(store)] = pre
                    consumed_pre.add(id(pre))
                    break
        for store in loop_pre:
            if id(store) not in consumed_pre:
                _emit_store(plan, store, body_emitter)
        pre_lets = body_emitter.lets
        body_emitter.lets = []
        # pre 在外层、主体被 inner 轴包住：转发不能跨循环边界（否则 j 的每轮都会
        # 复用 pre 写下的那个值）。
        body_emitter.clear_forward()

        # 累加器：地址沿 inner 轴不变 → 在循环外装进 local.var 向量寄存器，循环内
        # 只更新寄存器，循环结束回写一次（替代每轮 vld + vsts 的内存往返）。
        acc_map: dict = {}
        for store, old in candidates:
            acc = tirx.decl_buffer(
                (1,), _vec_dtype(_elem_dtype(store.spec.buffer.dtype)), scope="local.var", name=f"acc{body_prefix}{body_emitter.counter}"
            )
            body_emitter.counter += 1
            body_emitter.prologue.append(tirx.AllocBuffer(acc))
            pre = folded.get(id(store))
            if pre is not None:
                init = body_emitter.dup(pre.value)
            elif store.kind == AccessClass.CONTIGUOUS:
                init = body_emitter.value(old)
            else:  # 私有标量：读一次标量并广播
                init = body_emitter.dup(old)
            body_emitter.lets.append(tirx.BufferStore(acc, init, [0]))
            # 循环内/回写阶段对旧值的读一律指向寄存器。
            body_emitter.memo[_value_key(old, body_emitter._store_gen)] = tirx.BufferLoad(acc, [0])
            acc_map[id(store)] = (store, old, acc)
        init_lets = body_emitter.lets
        body_emitter.lets = []

        for store in plan.stores:
            entry = acc_map.get(id(store))
            if entry is None:
                _emit_store(plan, store, body_emitter)
                continue
            acc = entry[2]
            value = body_emitter.vector_operand(store.value)
            body_emitter.lets.append(tirx.BufferStore(acc, value, [0]))
        main_lets = body_emitter.lets
        body_emitter.lets = []
        # 主体被 inner 轴包住后，循环体内定义的 SSA 不能在 post 里复用。
        body_emitter.clear_forward()

        # 回写：post 若覆写同址并读旧值，则把回写交给 post（寄存器直通，省一次 vsts）。
        superseded = {
            id(store)
            for store, _ in candidates
            if any(
                post.mask is None
                and _same_buffer(post.spec.buffer, store.spec.buffer)
                and _same_indices(tuple(post.spec.indices), tuple(store.spec.indices))
                and same_address_load(post.value, post.spec.buffer, post.spec.indices) is not None
                for post in plan.post_stores
            )
        }
        for store, old, acc in acc_map.values():
            value = tirx.BufferLoad(acc, [0])
            if id(store) not in superseded:
                bits = _mask_bits(_elem_dtype(store.spec.buffer.dtype))
                if store.kind == AccessClass.CONTIGUOUS:
                    mask = _store_mask(store, body_emitter, bits)
                    body_emitter.lets.append(tirx.Evaluate(simd.vsts(body_emitter.addr(store.access), value, mask, f"NORM_B{bits}")))
                else:
                    # 私有标量累加器：地址沿向量轴不变，单点回写 lane 0 即可。
                    body_emitter.lets.append(
                        tirx.Evaluate(
                            simd.vsts(
                                body_emitter.addr(store.access, uniform=True),
                                value,
                                body_emitter.all_mask(bits),
                                f"ONEPT_B{bits}",
                            )
                        )
                    )
                body_emitter.note_write(store.spec.buffer)
            # 回写后 post 对同址的读直接吃寄存器（每 lane 即该 lane 的结果）。
            body_emitter.promote(old, value)
            # 段间寄存器直通：单 chunk、无尾块、无谓词、地址沿 inner 轴不变的连续回写，
            # 其寄存器值就是内存里的整段结果，可交给下一段同址连续读复用（省一次 vld）。
            # 写前已由候选判据保证地址沿 inner 轴不变；这里再收紧到「回写没有被 post
            # 接管」与「向量轴之前没有未提层的外层轴」。
            if (
                carry is not None
                and id(store) not in superseded
                and store.mask is None
                and store.kind == AccessClass.CONTIGUOUS
                and plan.chunks == 1
                and not plan.tail
                and all(any(axis.var.same_as(h.var) for h in hoisted) for axis in plan.axes if axis.index < plan.vector_axis.index)
            ):
                body_emitter.register_carry(store, value, axis=plan.vector_axis.var)
        final_lets = body_emitter.lets
        body_emitter.lets = []

        for store in plan.post_stores:
            _emit_store(plan, store, body_emitter)
        post_lets = body_emitter.lets
        # LICM：与 inner 轴无关的纯 SSA（地址分解 / 只读 buffer 的装载）提到循环外。
        # 累加器 buffer 视为「本段会写」，读它的 SSA 不能外提（每轮值都变）。
        licm_written = written_keys | {_buffer_key(acc) for _, _, acc in acc_map.values()}
        hoisted_main, loop_main = _split_loop_invariant(main_lets, inner, licm_written)
        if not loop_main:
            return body_emitter.prologue, [*pre_lets, *init_lets, *hoisted_main, *final_lets, *post_lets]
        main = _seq(loop_main)
        for axis in reversed(inner):
            main = tirx.For(axis.var, 0, axis.extent, tirx.ForKind.SERIAL, main)
        return body_emitter.prologue, [*pre_lets, *init_lets, *hoisted_main, main, *final_lets, *post_lets]

    def make_block(chunk_index, masked_tail: bool, label: str, chunk_min: int = 0, chunk_max: int | None = None):
        return emit_body(chunk_index, masked_tail, f"{prefix}{label}", chunk_min, chunk_max)

    block_prologue, blocks = _strip_mine(plan, chunk, make_block)
    prologue.extend(block_prologue)
    return finish(_seq(blocks))


def _segment_consumes_carry(segment: VectorPlan, hoisted: list[Axis]) -> bool:
    """该段能否消费上一段的寄存器直通值（lane 划分必须与本段一致）。

    条件：**单 chunk**（lane 基址恒 0）、且向量轴之前没有未提层的外层轴（否则读发生在
    外层循环里，而直通值只对应上一段最后一次外层迭代）。尾块/守卫由 ``vload`` 分支
    自行排除（带谓词的装载不会走直通）。
    """
    if segment.chunks != 1:
        return False
    return all(any(axis.var.same_as(h.var) for h in hoisted) for axis in segment.axes if axis.index < segment.vector_axis.index)


def _emit_program(program: VectorProgram) -> tirx.Stmt:
    """发射 :class:`VectorProgram`：逐段（源序）发射，``program.axes`` 统一包裹一次。

    ``program.axes`` 必须只是**并行轴的提层前缀**（``plan._outer_axes`` / 多段路径的
    共享并行轴）：它们既被这里包一次，又被 :func:`_emit_segment_stmts` 按身份跳过；
    若其中混入了非并行轴或某段的向量轴，就会出现重复包裹 / 漏包，属非法 IR。

    多段之间有两类**纯增量**共享（都不改变段的源序执行模型）：

    * :class:`_ProgramPrologue`：掩码 / 步长斜坡等纯值只绑一次；
    * ``carry``：上一段「整向量、无谓词、单 chunk」的连续写把寄存器值登记下来，下一段
      同址连续读直接复用（省一次 ``vld``）。任何写都会作废对应项（``note_write``）。
    """
    assert all(axis.is_parallel for axis in program.axes), "program.axes may only hold hoisted T.Parallel axes"
    shared = _ProgramPrologue()
    carry: dict = {}
    prologue: list = []
    bodies: list = []
    for index, segment in enumerate(program.segments):
        seg_chunk = tirx.Var(f"chunk_s{index}", "int32")
        seg_prologue, body = _emit_segment_stmts(
            segment,
            seg_chunk,
            hoisted=program.axes,
            prefix=f"s{index}_",
            single=len(program.segments) == 1,
            shared=shared,
            carry=carry,
            allow_carry=_segment_consumes_carry(segment, program.axes),
        )
        prologue.extend(seg_prologue)
        bodies.append(body)
    body = _seq(bodies)
    for axis in reversed(program.axes):
        body = tirx.For(axis.var, 0, axis.extent, tirx.ForKind.SERIAL, body)
    # 共享绑定只依赖常量与彼此，放在 program 最前即可（段内按需引用）。
    prologue = [*shared.statements, *prologue]
    if prologue:
        return tirx.SeqStmt([*prologue, body])
    return body


def _emit_reduce(plan: VectorPlan, emitter: _Emitter) -> None:
    """归约段（§2 Case II）：按源码顺序逐条发射段内每个 store。

    段内可能有**多条**归约（缺陷 D6 原先用 ``next(...)`` 只发第一条）以及取值沿向量轴
    均匀的广播写（一次 ``ONEPT`` 写即可）。每条归约各自展开 chunk、各起一条 SSA 累加链、
    各回写一次。
    """
    for store in plan.stores:
        if store.reduce is None:
            _emit_outer_store(store, emitter)  # 取值沿轴均匀 → 与归约同级，发一次
            continue
        _emit_one_reduce(plan, store, emitter)


def _emit_one_reduce(plan: VectorPlan, store: StorePlan, emitter: _Emitter) -> None:
    """发射单条归约 store：按 chunk 数选「展开」或「真实循环」策略（见模块 docstring）。

    ``chunks <= _REDUCE_UNROLL_LIMIT`` 时在编译期展开并拆多条 SSA 累加链（树形合并）；
    否则（含符号 extent，``chunks is None``）把累加器落成 ``local.var`` 向量寄存器，
    chunk 循环保留在 IR 里。两条路径的收尾一致：并入目标旧值 + ``ONEPT`` 回写。

    逐 lane 取值与旧值已由 ``plan._plan_reduce`` 在 build 期拆好。
    """
    reduce = store.reduce
    chunks = plan.chunks
    if chunks == 0:
        # 零次迭代：归约不执行，目标保持原值（也避免空累加链）。
        return
    if chunks is not None and chunks <= _REDUCE_UNROLL_LIMIT:
        _emit_reduce_unrolled(plan, store, emitter, reduce, chunks)
    else:
        _emit_reduce_loop(plan, store, emitter, reduce)


def _emit_reduce_tail(store: StorePlan, emitter: _Emitter, reduce, elem: str, bits: int, acc):
    """两条归约路径共用的收尾：并入目标旧值（累加写）后 ``ONEPT`` 单元素回写。

    ``acc`` 是**跨 lane 归约结果**，只有 lane 0 有效，因此这里的并入与回写都用
    :meth:`_Emitter.lane0_mask`（``PAT_VL1``）。旧值按**标量**读一次直接进 ``vadds``
    （``vmaxs``/``vmins``），省掉一条 ``vld(..., BRC_B*)`` 广播和一次 64 lane 加法；
    只有归约结果被硬件拓宽（``int16`` 求和 → ``int32``）导致标量类型对不上时，
    才退回广播读 + 全 lane 合并。
    """
    lane0 = emitter.lane0_mask(bits)
    if reduce.accumulate:
        element_op = _REDUCE_ELEMENT[reduce.recipe]
        old = emitter.addr(store.access, uniform=True)
        if _elem_dtype(str(acc.dtype)) == elem:
            acc = emitter.bind(getattr(simd, _VEC_SCALAR_BINARY[element_op])(acc, old, lane0, _MODE), emitter.lets)
        else:
            old_vec = emitter.bind(simd.vld(old, f"BRC_B{bits}"), emitter.lets)
            acc = emitter.bind(getattr(simd, element_op)(acc, old_vec, lane0, _MODE), emitter.lets)
    dst = emitter.addr(store.access, uniform=True)
    emitter.lets.append(tirx.Evaluate(simd.vsts(dst, acc, lane0, f"ONEPT_B{bits}")))
    emitter.note_write(store.spec.buffer)


def _emit_reduce_chunk(plan: VectorPlan, store: StorePlan, body: _Emitter, reduce, bits: int, mask, fill):
    """在 ``body`` 里发射**一个 chunk** 的部分归约（``vcadd``/``vcmax``/``vcmin``）。

    需要尾块谓词时（``body.masked_tail``）：

    * **Add**（``vcadd``）的单位元就是 0，而 ``MODE_ZEROING`` 下 inactive lane 天然为 0
      ——直接把尾块掩码交给 ``vcadd`` 即可，省掉 ``vdup`` 单位元 + ``vsel`` 两条；
    * **Max/Min** 的单位元不是 0（dtype 最小值/最大值），必须先用 ``vsel`` 填成单位元，
      再按全 lane 掩码归约（改进 6 / 缺陷 D10）。
    """
    loaded = body.value(reduce.varying)
    tail = body.tail_mask(bits) if body.masked_tail else None
    if tail is not None:
        if reduce.recipe == "vcadd":
            return body.bind(simd.vcadd(loaded, tail, _MODE), body.lets)
        loaded = body.bind(simd.vsel(loaded, fill, tail), body.lets)
    return body.bind(getattr(simd, reduce.recipe)(loaded, mask, _MODE), body.lets)


def _emit_reduce_unrolled(plan: VectorPlan, store: StorePlan, emitter: _Emitter, reduce, chunks: int) -> None:
    """短归约：编译期展开 chunk 循环，每个 chunk 一条独立链，最后树形合并。

    链首**直接取该 chunk 的部分归约结果**，因此不需要 ``vdup`` 单位元；合并按二叉树
    两两进行，把跨 lane 归约的串行延迟从 ``chunks`` 级压到 ``ceil(log2 chunks)`` 级。
    ``chunks=1`` 时退化成「一次 vcadd + 回写」，不再有中间的 ``vadd``。

    （旧实现的 ``_REDUCE_CHAINS`` 与展开阈值同为 4，``min(4, chunks)`` 个槽位在
    ``chunks <= 4`` 时每槽恰好只用一次，「链内累加」分支是**不可达代码**；这里直接
    「每 chunk 一条链」，行为完全不变。）
    """
    elem = _elem_dtype(store.spec.buffer.dtype)
    bits = _mask_bits(elem)
    mask = emitter.all_mask(bits)
    element_op = _REDUCE_ELEMENT[reduce.recipe]

    # 尾块（``extent % lanes != 0``）：最后一个 chunk 只有 ``tail`` 个有效车道，其余车道
    # 读到越界/脏数据。Add 直接靠 ``vcadd`` 的零化掩码即可；Max/Min 需要先填单位元。
    fill = (
        emitter.bind(simd.vdup(_reduce_identity(reduce.recipe, elem), elem, mask, _MODE), emitter.lets)
        if plan.tail and reduce.recipe != "vcadd"
        else None
    )

    last_chunk = chunks - 1
    chains: list = []
    for ck in range(chunks):
        emitter.lane_base = _int(ck * plan.lanes)
        emitter.memo = {}  # 地址随 chunk 变化，缓存必须按 chunk 失效
        emitter._predicates = {}  # 谓词里可能含 lane_index，同样按 chunk 失效
        # 尾块谓词与 fill 是否存在无关：Add 也要把尾块掩码交给 ``vcadd``（跳过填充）。
        emitter.masked_tail = bool(plan.tail) and ck == last_chunk
        chains.append(_emit_reduce_chunk(plan, store, emitter, reduce, bits, mask, fill))
    emitter.masked_tail = False

    while len(chains) > 1:  # 树形合并：相邻两条链两两相加（结果仍只保留 lane 0）
        lane0 = emitter.lane0_mask(bits)
        merged = [
            emitter.bind(getattr(simd, element_op)(chains[k], chains[k + 1], lane0, _MODE), emitter.lets)
            for k in range(0, len(chains) - 1, 2)
        ]
        if len(chains) % 2:
            merged.append(chains[-1])
        chains = merged

    _emit_reduce_tail(store, emitter, reduce, elem, bits, chains[0])


def _emit_reduce_loop(plan: VectorPlan, store: StorePlan, emitter: _Emitter, reduce) -> None:
    """长归约 / 符号 extent：真实 chunk 循环 + ``local.var`` 向量寄存器累加器。

    展开会让指令段随 ``chunks`` 线性膨胀；这里把累加器落成可写寄存器（``AllocBuffer`` +
    ``BufferStore``/``BufferLoad``，codegen 对 ``local.var`` 的向量 dtype 直接发
    ``vector_f32``），循环体只保留 ``vld → 归约 → 累加`` 三条。

    * 常量 extent：主循环跑 ``chunks-1`` 个整块（全 lane 掩码），尾块单独一个**单次**
      循环块，带尾块谓词 + 单位元填充；
    * 符号 extent：单条循环，每轮用运行期 ``update_mask(min(E - ck*lanes, lanes))`` 夹紧
      并填充单位元（``update_mask`` 只保证 ``[0, lanes]`` 内有意义，缺陷 A5）。
    """
    elem = _elem_dtype(store.spec.buffer.dtype)
    bits = _mask_bits(elem)
    mask = emitter.all_mask(bits)
    element_op = _REDUCE_ELEMENT[reduce.recipe]
    identity = _reduce_identity(reduce.recipe, elem)

    acc = tirx.decl_buffer((1,), _vec_dtype(elem), scope="local.var", name=f"acc{emitter.prefix}{emitter.counter}")
    emitter.counter += 1
    # 累加器声明提到 chunk 循环（以及外层轴）之外：它只在段内存在一份，初值在每个外层
    # 迭代重新写入，因此复用同一个寄存器。
    emitter.prologue.append(tirx.AllocBuffer(acc))
    emitter.lets.append(tirx.BufferStore(acc, simd.vdup(identity, elem, mask, _MODE), [0]))
    # 只有「可能有尾块且单位元不是 0」时才需要填充向量；Add 直接复用尾块掩码。
    fill = (
        emitter.bind(simd.vdup(identity, elem, mask, _MODE), emitter.lets)
        if (plan.chunks is None or plan.tail) and reduce.recipe != "vcadd"
        else None
    )

    def build_chunk(chunk_index, masked_tail: bool, label: str, chunk_min: int = 0, chunk_max: int | None = None):
        body = _Emitter(
            plan,
            chunk_index,
            f"{emitter.prefix}{label}",
            masked_tail=masked_tail,
            chunk_min=chunk_min,
            chunk_max=chunk_max,
            shared=emitter._shared,
            carry=emitter._carry,
        )
        reduced = _emit_reduce_chunk(plan, store, body, reduce, bits, mask, fill)
        # 累加器只保留 lane 0（归约结果的有效车道），循环内的累加也就只算一个车道。
        lane0 = emitter.lane0_mask(bits)
        total = body.bind(getattr(simd, element_op)(tirx.BufferLoad(acc, [0]), reduced, lane0, _MODE), body.lets)
        body.lets.append(tirx.BufferStore(acc, total, [0]))
        return body.prologue, body.lets

    # 每条归约各起一条 chunk 循环，用独立的循环变量（同段多条归约不能共用 For 变量）。
    loop_var = tirx.Var(f"{emitter.prefix}ck{emitter.counter}", "int32")
    emitter.counter += 1
    block_prologue, blocks = _strip_mine(plan, loop_var, build_chunk)
    emitter.prologue.extend(block_prologue)
    emitter.lets.extend(blocks)

    final = emitter.bind(tirx.BufferLoad(acc, [0]), emitter.lets)
    _emit_reduce_tail(store, emitter, reduce, elem, bits, final)
