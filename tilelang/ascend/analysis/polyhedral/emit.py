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

归约（§2 Case II）按 C++ pass 的定型方案**在编译期展开** chunk 循环，用 SSA 形式
串起累加器（跨迭代的寄存器值无法用可变的 ``Var`` 表达）::

    mask = pset(bits, "PAT_ALL")
    acc  = vdup(identity, elem, mask)
    for ck:                                  # 展开
        red = vcadd(vld(src_addr(ck), "NORM"), mask)
        acc = vadd(acc, red, mask)
    if 累加写: old = vld(dst_addr, "BRC_B{bits}"); acc = vadd(acc, old, mask)
    vsts(dst_addr, acc, mask, "ONEPT_B{bits}")
"""

from __future__ import annotations

from tvm import arith, tirx

from tilelang.ascend.language import simd
from tilelang.ascend.language.simd import _mask_bits

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


def _unsigned_index(reg):
    """把有符号整数索引向量按位重解释为无符号（``vgather2`` / ``vscatter`` 需要）。"""
    dtype = str(reg.dtype)
    if dtype.startswith("int"):
        return tirx.reinterpret(dtype.replace("int", "uint", 1), reg)
    return reg


def _value_key(node: ValueNode):
    """SSA 复用键：同一次访存（同 buffer、同下标）只发一条装载。

    ``build_value`` 对每个 ``BufferLoad`` 都新建 :class:`ValueNode`，所以
    ``if a[i, j] != 0: c[i, j] = a[i, j]`` 里的两次读是**两个节点**；按 ``id(node)``
    缓存会让它们各自发一条 ``vld``/``vgather2``。无副作用的装载叶子按结构复用。
    """
    if node.recipe in ("vload", "vload_strided", "uniform_load") and node.access is not None:
        access = node.access
        return (node.recipe, id(access.buffer), tuple(str(index) for index in access.indices), node.detail)
    return id(node)


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

    def data_mask(self, dtype_like):
        """数据向量的执行掩码：元素类型 → 谓词宽度 → 全 lane 谓词（按宽度缓存）。"""
        return self.all_mask(_mask_bits(dtype_like))

    def tail_mask(self, bits: int):
        """尾块谓词：只有在本发射块需要屏蔽尾部车道时非空（整 chunk 块返回 ``None``）。

        常量 extent（``plan.tail`` 已知）用常量车道数；符号 extent 则在每个 chunk 运行时
        算 ``min(extent - chunk*lanes, lanes)``——``update_mask`` 只保证 ``[0, lanes]``
        内有意义，超宽车道数不做假设（缺陷 A5）。

        掩码按宽度缓存：同一个块内只绑定一次，避免每条 ``vsts`` 前重复绑定。
        """
        if not self.masked_tail:
            return None
        key = ("tail", bits)
        cached = self._mask_vars.get(key)
        if cached is not None:
            return cached
        if self.plan.tail is not None:
            count = _int(self.plan.tail)
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
    def value(self, node: ValueNode):
        key = _value_key(node)
        if key in self.memo:
            return self.memo[key]
        expr = self._build(node)
        self.memo[key] = expr
        return expr

    def vector_operand(self, node: ValueNode):
        """把一个 lane 内均匀的标量提升成向量寄存器（``vdup``），否则走 ``value``。"""
        if node.recipe in _SCALAR_ONLY_RECIPES:
            elem = _elem_dtype(node.dtype)
            return self.bind(simd.vdup(self.scalar(node), elem, self.data_mask(elem), _MODE), self.lets)
        return self.value(node)

    def _build(self, node: ValueNode):
        recipe = node.recipe
        args = node.args

        if recipe == "imm":
            return node.scalar
        if recipe == "lane_index":
            # 向量轴的绝对取值 = chunk 基址 + lane id（用于谓词比较，如 ``i < j``）。
            return self.bind(simd.vci(self.lane_base, _elem_dtype(node.dtype)), self.lets)
        if recipe == "vload":
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
            elem = _elem_dtype(operand.dtype)
            return self.bind(simd.vdup(self.scalar(operand), elem, self.data_mask(elem), _MODE), self.lets)
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
            return

    # PERMUTE / PIECEWISE / INDIRECT / 负步长 STRIDED：绝对元素偏移索引向量 + 缓冲区起点基址。
    if store.index is None:  # 防御：这条通道必须带每 lane 索引向量
        raise Unsupported(f"no per-lane index vector for a {store.kind.value} store")
    index = _unsigned_index(emitter.value(store.index))
    emitter.lets.append(tirx.Evaluate(simd.vscatter(value, emitter.index_addr(store.access), index, mask)))


def _emit_outer_store(store: StorePlan, emitter: _Emitter) -> None:
    """发射一条与 serial 循环**同级**的语句（pre/post，方案 §6 第二步）。

    这类语句不含 serial 向量轴变量，沿该轴 ``Δ==0``，因此每个外层迭代只执行一次——
    发射在 chunk 循环**之外**，用 ``ONEPT`` 单点写（值经 ``vdup`` 广播）。
    """
    if store.kind != AccessClass.UNIFORM:
        raise Unsupported(f"a statement outside the serial loop must be uniform along the vector axis (got {store.kind.value})")
    elem = _elem_dtype(store.spec.buffer.dtype)
    bits = _mask_bits(elem)
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
        count = tirx.ceildiv(plan.vector_axis.extent, _int(plan.lanes))
        blocks = [tirx.For(chunk, 0, count, tirx.ForKind.SERIAL, _seq(emit_body(chunk, True, "")))]
        return prologue, blocks

    if not plan.tail:
        blocks = [tirx.For(chunk, 0, _int(plan.chunks), tirx.ForKind.SERIAL, _seq(emit_body(chunk, False, "")))]
        return prologue, blocks

    blocks: list = []
    full = plan.chunks - 1
    if full > 0:
        blocks.append(tirx.For(chunk, 0, _int(full), tirx.ForKind.SERIAL, _seq(emit_body(chunk, False, ""))))
    # 尾块是**单次**循环（``[chunks-1, chunks)``）：只有这一块带尾块谓词，整 chunk 保留全 lane。
    # ``tirx.For(var, min, extent)`` 的第三个参数是 **extent**（不是 end）：写成 ``plan.chunks``
    # 会让尾块从 ``full`` 起跑 ``chunks`` 次（越界读写，P0）。
    tail_chunk = tirx.Var(f"{prefix}tailchunk", "int32")
    blocks.append(
        tirx.For(
            tail_chunk,
            _int(full),
            _int(1),
            tirx.ForKind.SERIAL,
            _seq(emit_body(tail_chunk, True, f"{prefix}tail_")),
        )
    )
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
    prologue, body = _emit_segment_stmts(plan, tirx.Var("chunk", "int32"), hoisted=[])
    if prologue:
        return tirx.SeqStmt([*prologue, body])
    return body


def _emit_segment_stmts(plan: VectorPlan, chunk: tirx.Var, *, hoisted: list[Axis], prefix: str = "") -> tuple[list, tirx.Stmt]:
    """发射**一段** :class:`VectorPlan` 的语句体，返回 ``(prologue, body)``。

    ``prologue`` 是 chunk 无关的掩码绑定（调用方负责提到最前）；``body`` 已按非向量轴
    包裹成嵌套 serial 循环。``hoisted`` 是**已由调用方提到最外层**的轴（多段 program 的
    共享并行轴前缀，见 ``plan._outer_axes``）：按**变量身份**跳过它们，而不是笼统跳过所有
    并行轴——否则当向量轴就是某根外层并行轴时，夹在其后的并行轴会被漏包。
    """
    emitter = _Emitter(plan, chunk, prefix)

    for store in plan.pre_stores:  # 前置语句：chunk 循环之前，每个外层迭代一次
        _emit_outer_store(store, emitter)
    pre_lets = emitter.lets
    emitter.lets = []

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
    return [*emitter.prologue, *main_prologue], body


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
        seg_prologue, body = _emit_segment_stmts(segment, seg_chunk, hoisted=program.axes, prefix=f"s{index}_")
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
    """发射单条归约 store：展开 chunk、SSA 串累加器、``ONEPT`` 回写。

    先要 ``vcadd/vcmax/vcmin`` 把每个 chunk 压成 lane0 的部分结果，再用
    ``vadd/vmax/vmin`` 累加；累加写（``dst op= x``）末尾再并入目标里原有的值
    （``vld(..., BRC_B{bits})`` 广播）。``extent % lanes != 0`` 时最后一个 chunk 是尾块：
    ``update_mask`` 造尾块谓词，``vsel`` 把无效车道填成归约单位元（改进 6），再按全 lane
    掩码归约。逐 lane 取值与旧值已由 ``plan._plan_reduce`` 在 build 期拆好。

    结果写回 ``emitter.lets``；外层轴与 ``pre``/``post`` 的包裹交给 :func:`emit_plan`。
    """
    reduce = store.reduce
    elem = _elem_dtype(store.spec.buffer.dtype)
    bits = _mask_bits(elem)
    mask = emitter.all_mask(bits)
    recipe = reduce.recipe
    element_op = _REDUCE_ELEMENT[recipe]
    identity = _reduce_identity(recipe, elem)
    acc = emitter.bind(simd.vdup(identity, elem, mask, _MODE), emitter.lets)

    # 尾块（``extent % lanes != 0``，改进 6）：最后一个 chunk 只有 ``tail`` 个有效车道，
    # 其余车道在 ``vld`` 里读到的是越界/脏数据。尾块装载带谓词（不再整 lane 越界读，
    # 缺陷 D10），再用 ``vsel(loaded, fill, tail_mask)`` 把无效车道替换成归约单位元
    # （add→0，max→-inf，min→+inf），最后按全 lane 掩码归约，结果正确。
    fill = emitter.bind(simd.vdup(identity, elem, mask, _MODE), emitter.lets) if plan.tail else None

    varying = reduce.varying
    chunks = plan.chunks
    assert chunks is not None, "a reduction requires a compile-time chunk count (see plan._build_plan_for_axis)"
    last_chunk = chunks - 1
    for ck in range(chunks):
        lane_base = _int(ck * plan.lanes)
        emitter.lane_base = lane_base
        emitter.memo = {}  # 地址随 chunk 变化，缓存必须按 chunk 失效
        emitter.masked_tail = fill is not None and ck == last_chunk
        loaded = emitter.value(varying)
        if emitter.masked_tail:
            loaded = emitter.bind(simd.vsel(loaded, fill, emitter.tail_mask(bits)), emitter.lets)
        reduced = emitter.bind(getattr(simd, recipe)(loaded, mask, _MODE), emitter.lets)
        acc = emitter.bind(getattr(simd, element_op)(acc, reduced, mask, _MODE), emitter.lets)
    emitter.masked_tail = False

    if reduce.accumulate:
        # ``dst = dst op x``：末尾并入目标里原有的值（读到的是本 chunk 之前的结果）。
        old = emitter.bind(simd.vld(emitter.addr(store.access, uniform=True), f"BRC_B{bits}"), emitter.lets)
        acc = emitter.bind(getattr(simd, element_op)(acc, old, mask, _MODE), emitter.lets)

    dst = emitter.addr(store.access, uniform=True)
    emitter.lets.append(tirx.Evaluate(simd.vsts(dst, acc, mask, f"ONEPT_B{bits}")))
