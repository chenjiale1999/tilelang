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
  后端可把下一块的 ``vld`` 提前；累加器拆成 ``min(4, chunks)`` 条独立链再树形合并，
  把跨 lane 归约的串行延迟从 ``chunks`` 级压到 ``chunks/4`` 级::

      red0 = vcadd(vld(src(0)))              # 每条链的第一块直接作为链首，不需要单位元
      red1 = vcadd(vld(src(1)))
      acc0 = vadd(red0, vcadd(vld(src(2))))  # 链内累加
      acc1 = vadd(red1, vcadd(vld(src(3))))
      acc  = vadd(acc0, acc1)                # 树形合并

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

from .access import AccessClass, AccessInfo, Axis
from .model import StorePlan, VectorPlan, VectorProgram
from .primitives import (
    _ELEMENTWISE_BINARY,
    _ELEMENTWISE_UNARY,
    _MASK_DTYPE,
    _MODE,
    _PREDICATE_MASK_BITS,
    _REDUCE_ELEMENT,
    _SCALAR_ONLY_RECIPES,
    _VEC_SCALAR_BINARY,
    Unsupported,
    _elem_dtype,
    _int,
    _reduce_identity,
    _seq,
)
from .value import ValueNode

# 归约发射策略（见模块 docstring 的「短归约 / 长归约」两段）：
#   * ``chunks <= _REDUCE_UNROLL_LIMIT`` → 编译期展开（去掉循环控制，后端可跨块调度）；
#   * 否则（含符号 extent）→ 真实 chunk 循环 + ``alloc_var`` 可写向量寄存器累加器。
# 阈值取 4：展开后约 ``3*4+3 = 15`` 条指令，仍在单个指令段的舒适范围内；再大则
# 指令体积线性膨胀，而「去掉循环控制」的收益占比迅速下降。
_REDUCE_UNROLL_LIMIT = 4
# 展开路径的累加链数：把跨 lane 归约（vcadd/vcmax/vcmin）的串行依赖拆成 K 条，
# 树形合并后关键路径从 ``chunks`` 级降到 ``ceil(log2 K)`` 级。
_REDUCE_CHAINS = 4


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
    )


class _Emitter:
    """把 :class:`VectorPlan` 的 chunk 体发射成 ``T.simd.*`` SSA 序列。

    SSA 用 ``SeqStmt`` 里的 ``tirx.Bind`` 表达（``Bind`` 是独立语句而非
    表达式，因此不能像 ``tirx.Let`` 那样嵌套）。``lane_base`` 是代入向量轴
    变量的 lane 基址：普通路径是 ``chunk * lanes``，归约路径在展开时逐 chunk
    重设（并清空 ``memo``，因为地址随 chunk 变化）。
    """

    def __init__(self, plan: VectorPlan, chunk: tirx.Var, prefix: str = "", *, masked_tail: bool = False):
        self.plan = plan
        self.chunk = chunk
        self.prefix = prefix  # 多段路径下按段命名（``s0_``/``s1_``...），避免跨段重名
        self.lane_base: tirx.PrimExpr = chunk * _int(plan.lanes)
        self.masked_tail = masked_tail  # 本发射块是否需要屏蔽尾部车道（尾块，见 tail_mask）
        self.memo: dict = {}
        self.lets: list = []  # chunk 体内的 Bind / Evaluate
        self.prologue: list = []  # chunk 无关的 Bind（掩码 / 步长斜坡）
        self.counter = 0
        self._mask_vars: dict = {}
        self._ramps: dict = {}
        self._prologue_vars: set = set()
        # 每个 buffer 已发射的写次数：装载 SSA 键带上代数，写后不复用写前装载。
        self._store_gen: dict = {}
        # 本段写过的 buffer：其上的标量读不能提到 prologue（可能读到写前/写后的不同值）。
        self._written_buffers = [store.spec.buffer for store in (*plan.pre_stores, *plan.stores, *plan.post_stores)]
        self._dups: dict = {}  # 循环不变 vdup 的复用表（prologue 内只绑一次）
        self._vci: dict = {}  # lane_index 的 vci 复用表（键含 lane_base，逐 chunk 自然失效）
        # 同层「写后读」转发：连续写 (buffer, 同下标) → 已算好的向量值，后面对同址的
        # 连续读直接复用，省一条 vld（d7 / d14b 的 `u = s[2*i]`）。
        self._forward: dict = {}
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
        仍按 ``(stride, mask)`` 复用（多条访存共用一个斜坡）。
        """
        key = (stride, id(mask))
        cached = self._ramps.get(key)
        if cached is not None:
            return cached
        if id(mask) in self._prologue_vars:
            cached = self.prologue_bind(simd.vmuls(simd.vci(_int(0), "uint32"), _int(stride), mask))
        else:
            cached = self.bind(simd.vmuls(simd.vci(_int(0), "uint32"), _int(stride), mask), self.lets)
        self._ramps[key] = cached
        return cached

    # -- 掩码 ---------------------------------------------------------------
    def all_mask(self, bits: int):
        """全 lane 生效的谓词（``pset(bits, PAT_ALL)``），按宽度缓存、只绑一次。"""
        cached = self._mask_vars.get(bits)
        if cached is not None:
            return cached
        var = tirx.Var(f"mask{self.prefix}{bits}", _MASK_DTYPE)
        self.prologue.append(tirx.Bind(var, simd.pset(bits, "PAT_ALL")))
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
        var = tirx.Var(f"lane0{self.prefix}{bits}", _MASK_DTYPE)
        self.prologue.append(tirx.Bind(var, simd.pset(bits, "PAT_VL1")))
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

    def load_mask(self, bits: int):
        """装载谓词：尾块用尾块谓词，整 chunk 用全 lane 谓词（缺陷 D10）。

        旧实现只把尾块谓词加在 store 侧，load 一律全 lane，于是尾块整 lane 读越过
        buffer 尾部（``vld`` / 步长 ``vgather2`` / 间接 ``vgather2`` 都有这个问题）。
        """
        tail = self.tail_mask(bits)
        return tail if tail is not None else self.all_mask(bits)

    def predicate_combine(self, lhs, rhs):
        """两个谓词的合取：``pand(lhs, rhs, ALL)``。"""
        return self.bind(simd.pand(lhs, rhs, self.all_mask(_PREDICATE_MASK_BITS)), self.lets)

    def predicate(self, node: ValueNode, data_dtype):
        """条件节点 → 谓词寄存器。

        逐 lane 条件（``vcmp``/``vcmps``/``pand``...）本身就是谓词寄存器，直接返回；
        循环不变条件（沿向量轴均匀，§1 Case II 的三元条件）在 lane 内恒真/恒假，
        广播成全 lane 谓词 ``update_mask(cond ? lanes : 0)``——整 chunk 同开或同关。
        """
        if not node.is_uniform:
            return self.value(node)
        bits = _mask_bits(_elem_dtype(data_dtype))
        lane_count = tirx.Select(self.scalar(node), _int(self.plan.lanes), _int(0))
        return self.bind(simd.update_mask(lane_count, bits), self.lets)

    # -- 地址 ---------------------------------------------------------------
    def addr(self, access: AccessInfo, uniform: bool = False):
        """一次访问的 ``BufferLoad``：向量轴变量代入当前 chunk 的 lane 基址。"""
        base = _int(0) if uniform else self.lane_base
        indices = [
            self.analyzer.simplify(tirx.stmt_functor.substitute(index, {self.plan.vector_axis.var: base})) for index in access.indices
        ]
        return tirx.BufferLoad(access.buffer, indices)

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
        """记录一次写：同一 buffer 的后续装载不再复用写之前的 SSA。"""
        key = _buffer_key(buffer)
        self._store_gen[key] = self._store_gen.get(key, 0) + 1

    def note_store(self, store: StorePlan, value) -> None:
        """记录一条**连续**写：同址的后续 ``vload`` 可直接复用该向量值（写后读转发）。"""
        if store.kind == AccessClass.CONTIGUOUS:
            self._forward[(_buffer_key(store.spec.buffer), tuple(str(index) for index in store.spec.indices))] = value

    def clear_forward(self) -> None:
        """跨越循环边界时清空转发表（循环体内定义的 SSA 不能提到循环外使用）。"""
        self._forward.clear()

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
            tail = self.tail_mask(_mask_bits(node.dtype))
            if tail is None:
                return self.bind(simd.vld(self.addr(node.access), "NORM"), self.lets)
            # ``vld`` 没有谓词形式：尾块退化成带谓词的 ``vgather2``（相对 lane 偏移），
            # 否则整 lane 读会越过 buffer 尾部（缺陷 D10）。
            return self.bind(simd.vgather2(self.addr(node.access), simd.vci(_int(0), "uint32"), tail), self.lets)
        if recipe == "vload_strided":
            # simd 没有带步长装载：用 ``vgather2`` + 步长索引斜坡表达。
            _, stride = node.access.axis_class(self.plan.vector_axis)
            bits = _mask_bits(node.dtype)
            mask = self.load_mask(bits)
            index = self.ramp(stride, mask)
            return self.bind(simd.vgather2(self.addr(node.access), index, mask), self.lets)
        if recipe == "gather":
            # 每 lane 索引向量通道（改进 1）：``args[0]`` 是绝对元素偏移向量。
            bits = _mask_bits(node.dtype)
            mask = self.load_mask(bits)
            index = _unsigned_index(self.value(args[0]))
            return self.bind(simd.vgather2(self.index_addr(node.access), index, mask), self.lets)
        if recipe in ("uniform_load", "uniform_expr", "vbrc"):
            operand = node if recipe != "vbrc" else args[0]
            return self.dup(operand)
        if recipe in _VEC_SCALAR_BINARY.values():
            return self.bind(getattr(simd, recipe)(self.value(args[0]), self.scalar(args[1]), self.data_mask(node.dtype), _MODE), self.lets)
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
            return self.bind(simd.vsel(true_value, false_value, self.predicate(args[0], node.dtype)), self.lets)
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
    """chunk 内 store 的最终谓词 = 尾块掩码 ∧ 用户谓词（都为空时用全 1）。"""
    mask = None
    if store.mask is not None:
        mask = emitter.value(store.mask)
    tail = emitter.tail_mask(bits)
    if tail is not None:
        mask = tail if mask is None else emitter.predicate_combine(mask, tail)
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
            index = emitter.ramp(stride, mask)
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
            emitter.lets.append(tirx.BufferStore(store.spec.buffer, scalar, [*store.spec.indices]))
            emitter.note_write(store.spec.buffer)
            return
    mask = emitter.value(store.mask) if store.mask is not None else emitter.all_mask(bits)
    _emit_vector_write(store, emitter, bits, mask)


def _emit_elementwise_chunks(plan: VectorPlan, chunk: tirx.Var, prefix: str) -> tuple[list, list]:
    """逐元素主路径的 chunk 发射（尾块单独成块），返回 ``(prologue, blocks)``。

    常量 extent 且 ``tail != 0`` 时**拆出尾块**：前 ``chunks-1`` 个整 chunk 用全 lane 掩码，
    尾块单独一块用尾块掩码——整 chunk 仍走 ``vld``，只有尾块退化成带谓词的 ``vgather2``
    （``vld`` 没有谓词形式，整 lane 读会越过 buffer 尾部，缺陷 D10）。

    符号 extent（``chunks is None``）无法预知尾块位置：只发一个运行时 chunk 循环，
    每个 chunk 都带夹紧后的尾块掩码（缺陷 D9 / A5）。

    每块用独立的 :class:`_Emitter`（SSA 名不撞车），因此各自的 ``prologue``（``pset`` 全 lane
    掩码）也必须一并返回，由调用方提到最外层。
    """

    prologue: list = []

    def emit_body(chunk_var, masked_tail: bool, body_prefix: str) -> list:
        body_emitter = _Emitter(plan, chunk_var, body_prefix, masked_tail=masked_tail)
        for store in plan.stores:
            _emit_store(plan, store, body_emitter)
        prologue.extend(body_emitter.prologue)
        return body_emitter.lets

    if plan.chunks is None:
        # 符号 extent：主循环跑 `E // lanes` 个**整块**（满 lane、无运行期掩码），
        # 尾块只在 `E % lanes != 0` 时执行一次（运行期掩码 + 谓词装载）。
        extent = plan.vector_axis.extent
        full_count = tirx.FloorDiv(extent, _int(plan.lanes))
        blocks = [tirx.For(chunk, 0, full_count, tirx.ForKind.SERIAL, _seq(emit_body(chunk, False, "")))]
        tail_chunk = tirx.FloorDiv(extent, _int(plan.lanes))
        tail_body = tirx.IfThenElse(
            tirx.FloorMod(extent, _int(plan.lanes)) > 0,
            _seq(emit_body(tail_chunk, True, f"{prefix}tail_")),
            None,
        )
        blocks.append(tail_body)
        return prologue, blocks

    if not plan.tail:
        if plan.chunks == 1:
            # 单次循环展开成直线代码：`for chunk in range(1)` 只是给后端添乱。
            return prologue, emit_body(_int(0), False, prefix)
        blocks = [tirx.For(chunk, 0, _int(plan.chunks), tirx.ForKind.SERIAL, _seq(emit_body(chunk, False, "")))]
        return prologue, blocks

    blocks: list = []
    full = plan.chunks - 1
    if full > 1:
        blocks.append(tirx.For(chunk, 0, _int(full), tirx.ForKind.SERIAL, _seq(emit_body(chunk, False, ""))))
    elif full == 1:
        # 主循环只有一轮：直接展开，lane 基址是常量 0。
        blocks.append(_seq(emit_body(_int(0), False, "")))
    # 尾块**恒为单次**：直接展开成直线代码（lane 基址 = full * lanes 常量），
    # 只有这一块带尾块谓词，整 chunk 保留全 lane。
    blocks.append(_seq(emit_body(_int(full), True, f"{prefix}tail_")))
    return prologue, blocks


def emit_plan(plan: VectorPlan | VectorProgram) -> tirx.Stmt:
    """把 :class:`VectorPlan` / :class:`VectorProgram` 发射成 ``T.simd.*`` TIR。

    外层非向量轴退化成 serial 循环，向量轴被 strip-mine 成 ``chunk``
    （``for chunk in serial(ceil(E/L))``）；chunk 体内的 SSA 用 ``Bind`` 表达。
    归约（§2 Case II）则把 chunk 循环在编译期展开（见模块 docstring）。

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
    plan: VectorPlan, chunk: tirx.Var, *, hoisted: list[Axis], prefix: str = "", single: bool = False
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
    if plan.vector_axis.is_parallel and (plan.pre_stores or plan.post_stores):
        return _emit_parallel_axis_parts(plan, chunk, prefix, hoisted, single)

    emitter = _Emitter(plan, chunk, prefix)
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
        main_prologue, main_lets = _emit_elementwise_chunks(plan, chunk, prefix)

    for store in plan.post_stores:  # 后置语句：chunk 循环之后，每个外层迭代一次
        _emit_outer_store(store, emitter)
    post_lets = emitter.lets
    body = _seq([*pre_lets, *main_lets, *post_lets])
    for axis in reversed(plan.axes):
        if axis.var.same_as(plan.vector_axis.var):
            continue
        if any(axis.var.same_as(h.var) for h in hoisted):
            continue
        body = tirx.For(axis.var, 0, axis.extent, tirx.ForKind.SERIAL, body)
    hoisted_lets: list = []
    if invariant_pre:  # 与所有轴无关的 pre 写：整段提到最外层，只执行一次
        pre_emitter = _Emitter(plan, _int(0), f"{prefix}hopre_")
        for store in invariant_pre:
            _emit_outer_store(store, pre_emitter)
        emitter.prologue.extend(pre_emitter.prologue)
        hoisted_lets = pre_emitter.lets
    return [*emitter.prologue, *main_prologue, *hoisted_lets], body


def _emit_parallel_axis_parts(
    plan: VectorPlan, chunk: tirx.Var, prefix: str, hoisted: list[Axis], single: bool = False
) -> tuple[list, tirx.Stmt]:
    """并行向量轴 + ``pre``/``post``：``for chunk: pre; (serial 轴) main; post``（方案 §6 第二步）。

    ``pre``/``post`` 在源程序里是 serial 循环的兄弟语句，沿并行轴向量化后它们与 serial
    循环同级——每个 chunk 执行一次，**不能**被 serial 轴再包一层（否则 `s[i]=0` 这类
    初值会被每个 serial 迭代重置）。因此这里手工组织三段，而不是走
    :func:`_emit_segment_stmts` 的「所有非向量轴包住整段」结构。

    每个 block 用独立 :class:`_Emitter`（SSA 名不撞车），``prologue``（``pset`` 全 lane
    掩码）一并返回给调用方提到最外层。
    """
    assert plan.reduce is None, "a parallel vector axis with pre/post must go through the elementwise path"
    inner = [axis for axis in plan.axes if axis.index > plan.vector_axis.index and not any(axis.var.same_as(h.var) for h in hoisted)]
    prologue: list = []
    written_keys = {_buffer_key(store.spec.buffer) for store in (*plan.pre_stores, *plan.stores, *plan.post_stores)}
    invariant_pre = [store for store in plan.pre_stores if single and _store_is_loop_invariant(store, plan, written_keys)]
    invariant_ids = {id(store) for store in invariant_pre}
    loop_pre = [store for store in plan.pre_stores if id(store) not in invariant_ids]

    def finish(body):
        # 与所有轴无关的 pre 写：整段提到最外层，只执行一次（典型 d11 的 c[b[0]]=1.0）。
        if invariant_pre:
            pre_emitter = _Emitter(plan, _int(0), f"{prefix}hopre_")
            for store in invariant_pre:
                _emit_outer_store(store, pre_emitter)
            prologue.extend(pre_emitter.prologue)
            return [*prologue, *pre_emitter.lets], body
        return prologue, body

    def emit_body(chunk_var, masked_tail: bool, body_prefix: str) -> list:
        body_emitter = _Emitter(plan, chunk_var, body_prefix, masked_tail=masked_tail)
        for store in loop_pre:
            _emit_store(plan, store, body_emitter)
        pre_lets = body_emitter.lets
        body_emitter.lets = []
        # pre 在外层、主体被 inner 轴包住：转发不能跨循环边界（否则 j 的每轮都会
        # 复用 pre 写下的那个值）。
        body_emitter.clear_forward()
        for store in plan.stores:
            _emit_store(plan, store, body_emitter)
        main = _seq(body_emitter.lets)
        body_emitter.lets = []
        # 主体被 inner 轴包住后，循环体内定义的 SSA 不能在 post 里复用。
        body_emitter.clear_forward()
        for store in plan.post_stores:
            _emit_store(plan, store, body_emitter)
        post_lets = body_emitter.lets
        prologue.extend(body_emitter.prologue)
        for axis in reversed(inner):
            main = tirx.For(axis.var, 0, axis.extent, tirx.ForKind.SERIAL, main)
        return [*pre_lets, main, *post_lets]

    if plan.chunks is None:
        extent = plan.vector_axis.extent
        full_count = tirx.FloorDiv(extent, _int(plan.lanes))
        full_block = tirx.For(chunk, 0, full_count, tirx.ForKind.SERIAL, _seq(emit_body(chunk, False, prefix)))
        tail_chunk = tirx.FloorDiv(extent, _int(plan.lanes))
        tail_body = tirx.IfThenElse(
            tirx.FloorMod(extent, _int(plan.lanes)) > 0,
            _seq(emit_body(tail_chunk, True, f"{prefix}tail_")),
            None,
        )
        return finish(_seq([full_block, tail_body]))

    if not plan.tail:
        if plan.chunks == 1:
            return finish(_seq(emit_body(_int(0), False, prefix)))
        return finish(tirx.For(chunk, 0, _int(plan.chunks), tirx.ForKind.SERIAL, _seq(emit_body(chunk, False, prefix))))

    blocks: list = []
    full = plan.chunks - 1
    if full > 1:
        blocks.append(tirx.For(chunk, 0, _int(full), tirx.ForKind.SERIAL, _seq(emit_body(chunk, False, prefix))))
    elif full == 1:
        blocks.append(_seq(emit_body(_int(0), False, prefix)))
    blocks.append(_seq(emit_body(_int(full), True, f"{prefix}tail_")))
    return finish(_seq(blocks) if len(blocks) > 1 else blocks[0])


def _emit_program(program: VectorProgram) -> tirx.Stmt:
    """发射 :class:`VectorProgram`：逐段（源序）发射，``program.axes`` 统一包裹一次。

    ``program.axes`` 必须只是**并行轴的提层前缀**（``plan._outer_axes`` / 多段路径的
    共享并行轴）：它们既被这里包一次，又被 :func:`_emit_segment_stmts` 按身份跳过；
    若其中混入了非并行轴或某段的向量轴，就会出现重复包裹 / 漏包，属非法 IR。
    """
    assert all(axis.is_parallel for axis in program.axes), "program.axes may only hold hoisted T.Parallel axes"
    prologue: list = []
    bodies: list = []
    for index, segment in enumerate(program.segments):
        seg_chunk = tirx.Var(f"chunk_s{index}", "int32")
        seg_prologue, body = _emit_segment_stmts(
            segment, seg_chunk, hoisted=program.axes, prefix=f"s{index}_", single=len(program.segments) == 1
        )
        prologue.extend(seg_prologue)
        bodies.append(body)
    body = _seq(bodies)
    for axis in reversed(program.axes):
        body = tirx.For(axis.var, 0, axis.extent, tirx.ForKind.SERIAL, body)
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
    # ``max(..., 1)``：保证常量 extent 进入循环路径时至少 2 个 chunk，于是「主循环
    # ``chunks-1`` 轮 + 尾块」里主循环非空（否则单个整块会被静默跳过）。
    if chunks is not None and chunks <= max(_REDUCE_UNROLL_LIMIT, 1):
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
    """短归约：编译期展开 chunk 循环 + ``min(_REDUCE_CHAINS, chunks)`` 条累加链。

    链首**直接取第一个 chunk 的部分归约结果**，因此不需要 ``vdup`` 单位元；链内累加完
    后按二叉树合并，把跨 lane 归约的串行延迟从 ``chunks`` 级压到 ``ceil(log2 K)`` 级。
    ``chunks=1`` 时退化成「一次 vcadd + 回写」，不再有中间的 ``vadd``。
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
    chains: list = [None] * min(_REDUCE_CHAINS, chunks)
    for ck in range(chunks):
        emitter.lane_base = _int(ck * plan.lanes)
        emitter.memo = {}  # 地址随 chunk 变化，缓存必须按 chunk 失效
        # 尾块谓词与 fill 是否存在无关：Add 也要把尾块掩码交给 ``vcadd``（跳过填充）。
        emitter.masked_tail = bool(plan.tail) and ck == last_chunk
        reduced = _emit_reduce_chunk(plan, store, emitter, reduce, bits, mask, fill)
        slot = ck % len(chains)
        chains[slot] = (
            reduced
            if chains[slot] is None
            else emitter.bind(getattr(simd, element_op)(chains[slot], reduced, emitter.lane0_mask(bits), _MODE), emitter.lets)
        )
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

    def build_chunk(chunk, masked_tail: bool, body_prefix: str) -> list:
        body = _Emitter(plan, chunk, body_prefix, masked_tail=masked_tail)
        reduced = _emit_reduce_chunk(plan, store, body, reduce, bits, mask, fill)
        # 累加器只保留 lane 0（归约结果的有效车道），循环内的累加也就只算一个车道。
        lane0 = emitter.lane0_mask(bits)
        total = body.bind(getattr(simd, element_op)(tirx.BufferLoad(acc, [0]), reduced, lane0, _MODE), body.lets)
        body.lets.append(tirx.BufferStore(acc, total, [0]))
        emitter.prologue.extend(body.prologue)
        return body.lets

    def emit_chunk_loop(start, count, masked_tail: bool, body_prefix: str) -> None:
        # 单次循环（常量 count == 1）直接展开：把 chunk 变量换成常量 lane 基址。
        single = isinstance(count, tirx.IntImm) and int(count.value) == 1
        chunk = _int(start) if single else tirx.Var(f"{body_prefix}ck", "int32")
        lets = build_chunk(chunk, masked_tail, body_prefix)
        # ``tirx.For(var, min, extent)`` 的第三个参数是 **extent**（不是 end，见 P0 缺陷）。
        blocks.append(_seq(lets) if single else tirx.For(chunk, start, count, tirx.ForKind.SERIAL, _seq(lets)))

    blocks: list = []
    if plan.chunks is None:
        # 符号 extent：整块主循环（满 lane）+ `if E % lanes` 的 peel 尾块。
        extent = plan.vector_axis.extent
        emit_chunk_loop(_int(0), tirx.FloorDiv(extent, _int(plan.lanes)), False, f"{emitter.prefix}full")
        tail_chunk = tirx.FloorDiv(extent, _int(plan.lanes))
        tail_lets = build_chunk(tail_chunk, True, f"{emitter.prefix}tail")
        blocks.append(tirx.IfThenElse(tirx.FloorMod(extent, _int(plan.lanes)) > 0, _seq(tail_lets), None))
    else:
        full = plan.chunks - 1
        if full > 0:
            emit_chunk_loop(_int(0), _int(full), False, f"{emitter.prefix}main")
        if plan.tail:
            # 尾块恒为单次：展开成直线代码，带尾块谓词 + 单位元填充。
            emit_chunk_loop(_int(full), _int(1), True, f"{emitter.prefix}tail")
    emitter.lets.extend(blocks)

    final = emitter.bind(tirx.BufferLoad(acc, [0]), emitter.lets)
    _emit_reduce_tail(store, emitter, reduce, elem, bits, final)
