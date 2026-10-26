"""手动驱动脚本：逐场景跑 ``build_plan`` / ``emit_plan`` 并打印计划与发射 TIR。

用法（在仓库根目录）::

    python tilelang/ascend/analysis/test_drive_plan.py [--tir]

文件以 ``test_`` 前缀命名以便与实现同目录检索；它**不是 pytest 用例**——
所有场景只在直接运行时执行，被导入时不产生任何副作用。
"""

import sys

if __name__ == "__main__":
    # 直接运行本脚本时，绕过 editable-install 的 finder 重定向以加载本仓库实现。
    sys.meta_path = [f for f in sys.meta_path if type(f).__name__ != "ScikitBuildRedirectingFinder"]
    sys.path.insert(0, "/home/cjl/workspace/tilelang")

import tilelang  # noqa: E402, F401
import tilelang.ascend.language as T  # noqa: E402
from tvm import tirx  # noqa: E402

from tilelang.ascend.analysis.polyhedral import (  # noqa: E402
    Unsupported,
    build_plan,
    emit_plan,
)

N = 128
SHOW_TIR = "--tir" in sys.argv


def first_parallel(func):
    # post_order_visit 先访问内层再访问外层，故外层循环是**最后**被收集的：
    # build_plan 需要 nest 的头节点才能通过 _split_nest 恢复全部轴。
    found = []

    def visit(n):
        if isinstance(n, tirx.For) and n.kind == tirx.ForKind.PARALLEL:
            found.append(n)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return found[-1] if found else None


def _show_plan(plan, indent="       "):
    print(f"{indent}axis={plan.vector_axis.name} L={plan.lanes} Q={plan.chunks} tail={plan.tail} reduce={plan.reduce}")
    for st in plan.stores:
        print(
            f"{indent}store {st.spec.buffer.name}{list(st.spec.indices)} kind={st.kind.value} "
            f"value={st.value.recipe} mask={'y' if st.mask else 'n'}"
        )
        _dump(st.value, len(indent) + 2)
        if st.mask is not None:
            print(f"{indent}mask:")
            _dump(st.mask, len(indent) + 2)
    for note in plan.diagnostics:
        print(f"{indent}diag · {note}")


def run(name, func, lanes=None):
    nest = first_parallel(func)
    print(f"{name}:  **********************\n{func}\n************************")
    try:
        plan = build_plan(nest, lanes=lanes)
        # build_plan 恒返回 VectorProgram（1..N 段）：单段时按既有 plan 形态打印，
        # 多段时逐段打印并标出共享的提层并行轴。
        if len(plan.segments) == 1:
            print(f"[OK ] {name}, plan:")
            _show_plan(plan.segments[0])
        else:
            print(f"[OK ] {name}, program: axes={[a.name for a in plan.axes]} segments={len(plan.segments)}")
            for index, segment in enumerate(plan.segments):
                print(f"     -- segment {index} --")
                _show_plan(segment, indent="       ")
        if SHOW_TIR:
            stmt = emit_plan(plan)
            print("--- emitted TIR ---")
            print("       " + str(stmt).replace("\n", "\n       "))
            print("*" * 50, "\n")
    except Unsupported as err:
        print(f"[REJ] {name}: {err}")
        for note in getattr(err, "diagnostics", []):
            print(f"       diag · {note}")
        print()
    except Exception as err:  # noqa: BLE001
        import traceback

        print(f"[ERR] {name}: {type(err).__name__}: {err}")
        traceback.print_exc()


def _dump(node, indent, depth=0):
    if depth > 4:
        return
    print(" " * indent + f"- {node.recipe} ({node.dtype}) {node.detail}")
    for arg in node.args:
        _dump(arg, indent + 2, depth + 1)


def c0_copy():
    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i]

    return k


def c0_broadcast():
    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[0]

    return k


def c1_ifelse():
    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if a[i] == 0.0:
                    c[i] = 0.0
                else:
                    c[i] = T.exp(b[i])

    return k


def c1_single_arm():
    @T.prim_func
    def k(A: T.Tensor((8, 8), "float32"), C: T.Tensor((8, 8), "float32")):
        with T.Kernel(1):
            c = T.alloc_shared((8, 8), "float32")
            for i, j in T.Parallel(8, 8):
                if i < j:
                    c[i, j] = 0.0

    return k


def c2_rank():
    M, K = 16, 32

    @T.prim_func
    def k(S: T.Tensor((K,), "float32"), C: T.Tensor((M, K), "float32")):
        with T.Kernel(1):
            g = T.alloc_shared((K,), "float32")
            d = T.alloc_shared((M, K), "float32")
            for i, j in T.Parallel(M, K):
                d[i, j] = g[i] - g[j]

    return k


def c2_reduce():
    M, K = 4, 256

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            acc = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                acc[i] = acc[i] + a[i, j]

    return k


def c2_reduce_tail():
    """§2 Case II 归约 + 尾块（改进 6）：K=32 < 64 lanes → tail=32。
    尾块用 update_mask + vsel 把无效车道填成归约单位元后仍能正确发射。"""
    M, K = 4, 32

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            acc = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                acc[i] = acc[i] + a[i, j]

    return k


def r_long_reduce():
    """长归约（常量 ``chunks > _REDUCE_UNROLL_LIMIT``）：不再展开，走真实 chunk 循环 +
    ``local.var`` 可写向量寄存器累加器。

    ``(2, 512)`` / 64 lanes → 8 个 chunk、无尾块：发射成「主循环 7 轮（全 lane 掩码）
    + 循环外一次并入旧值 + ONEPT 回写」，展开会线性膨胀 8×3 条指令且关键路径仍是 8 级。
    """
    M, K = 2, 512

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), S: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            s = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                s[i] = s[i] + a[i, j]

    return k


def r_long_reduce_tail():
    """长归约 + 尾块：``(2, 552)`` → 9 个 chunk、``tail = 40``。

    覆盖长归约的**尾块分块**分支：主循环跑 ``chunks-1 = 8`` 个整块，尾块单独一个单次
    循环块，带 ``update_mask(40)`` + ``vsel`` 单位元填充，再累加进同一个 ``local.var``
    累加器（展开路径的尾块是内联在最后一个展开块里的，两者不同）。
    """
    M, K = 2, 552

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), S: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            s = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                s[i] = s[i] + a[i, j]

    return k


def c2_axis_pick():
    """改进 2：写侧主序的联合打分。转置写 c[j,i]=a[i,j] 里，内层轴 j 对写是 strided、
    对读是 contiguous；外层轴 i 对写是 contiguous、对读是 strided。加权后选**外层轴 i**
    （让写连续），发成 vsts 写 + vgather2 读，diagnostics 同时给出两轴打分。"""
    M = 64

    @T.prim_func
    def k(A: T.Tensor((M, M), "float32"), C: T.Tensor((M, M), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, M), "float32")
            c = T.alloc_shared((M, M), "float32")
            for i, j in T.Parallel(M, M):
                c[j, i] = a[i, j]

    return k


def c1_ternary():
    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = T.if_then_else(a[i] > b[i], a[i], b[i])

    return k


def c3_multi():
    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32"), D: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            d = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i]
                d[i] = a[i] * 2.0

    return k


def c3_scalars():
    S, K = 16, 8

    @T.prim_func
    def k(A: T.Tensor((S, K), "float32"), C: T.Tensor((S, K), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((S, K), "float32")
            c = T.alloc_shared((S, K), "float32")
            for i, j in T.Parallel(S, K):
                acc: T.float32 = a[i, j] * 2.0
                c[i, j] = acc + 1.0

    return k


def c3_value_pow2():
    """§3 Case II 值算子（改进 3）：FloorDiv / FloorMod 作用在**取值**上，除数 4 / 8 是
    2 的幂 → strength-reduce 成 vshrs（算术右移）/ vand（按位与）。"""

    @T.prim_func
    def k(A: T.Tensor((N,), "int32"), C: T.Tensor((N,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "int32")
            c = T.alloc_shared((N,), "int32")
            for i in T.Parallel(N):
                c[i] = a[i] + i // 4 + i % 8

    return k


def c3_topk():
    """§3 Case I 多行语句：一次 T.Parallel 里写两个输出（topk_gates / topk_indices），
    只固定尾维 k 写一列。（按原文形状：单轴 + strided store）"""
    BM, K = 64, 8
    kk = 0

    @T.prim_func
    def k(
        Gates: T.Tensor((BM, K), "float32"),
        Idx: T.Tensor((BM, K), "int32"),
        MaxVal: T.Tensor((BM,), "float32"),
        MaxIdx: T.Tensor((BM,), "int32"),
    ):
        with T.Kernel(1):
            gates = T.alloc_shared((BM, K), "float32")
            idx = T.alloc_shared((BM, K), "int32")
            max_val = T.alloc_shared((BM,), "float32")
            max_idx = T.alloc_shared((BM,), "int32")
            for i in T.Parallel(BM):
                gates[i, kk] = max_val[i]
                idx[i, kk] = max_idx[i]

    return k


def c3_im2col():
    """§3 Case II 多行语句：复杂标量运算（FloorDiv/FloorMod）计算访存下标，
    再接一条真实 store。（按原文表达式，配一组 conv im2col 常量）"""
    N, C, H, W = 1, 4, 8, 8
    KH, KW = 3, 3
    S, D, P = 1, 1, 1
    OH = (H + 2 * P - KH) // S + 1
    OW = (W + 2 * P - KW) // S + 1
    block_M, block_K = 16, 12
    k_iter, by = 0, 0

    @T.prim_func
    def k(Data: T.Tensor((N, H, W, C), "float32"), DataShared: T.Tensor((block_M, block_K), "float32")):
        with T.Kernel(1):
            data = T.alloc_shared((N, H, W, C), "float32")
            data_shared = T.alloc_shared((block_M, block_K), "float32")
            for i, j in T.Parallel(block_M, block_K):
                kk = k_iter * block_K + j
                m = by * block_M + i
                access_h = m % (OH * OW) // OW * S + kk // (KW * C) * D - P
                access_w = m % OW * S + kk // C % KW * D - P
                in_bound = (access_h >= 0) and (access_w >= 0) and (access_h < H) and (access_w < W)
                data_shared[i, j] = T.if_then_else(in_bound, data[m // (OH * OW), access_h, access_w, kk % C], 0.0)

    return k


def c4_flat():
    @T.prim_func
    def k(H: T.Tensor((8, 16), "float32"), D: T.Tensor((8, 16), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            h = T.alloc_shared((8, 16), "float32")
            d = T.alloc_shared((8, 16), "float32")
            c = T.alloc_shared((128,), "float32")
            for i_kv in T.Parallel(128):
                c[i_kv] = h[i_kv // 16, i_kv % 16] * d[i_kv // 16, i_kv % 16]

    return k


def c4_tiled():
    BM, BN, MX, MY = 16, 16, 4, 4

    @T.prim_func
    def k(CS: T.Tensor((BM // MX, BN // MY, MX, MY), "float32"), C: T.Tensor((BM, BN), "float32")):
        with T.Kernel(1):
            cs = T.alloc_shared((BM // MX, BN // MY, MX, MY), "float32")
            c = T.alloc_shared((BM, BN), "float32")
            for i, j in T.Parallel(BM, BN):
                c[i, j] = cs[i // MX, j // MY, i % MX, j % MY]

    return k


def c4_strided():
    @T.prim_func
    def k(A: T.Tensor((256,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((256,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[2 * i]

    return k


def c4_reverse():
    """§4 逆序访存（改进 1 + 4）：a[N-1-i] 的 Δ == -1 → PERMUTE（双射逆序），
    走每 lane 索引向量通道发成 vgather2。"""

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[N - 1 - i]

    return k


def c4_true_random():
    """§4 无结构下标（改进 4）：a[i//3] 的 Δ 非常数且除数 3 非 2 的幂 → 每 lane 索引
    向量构造不出 → TRUE_RANDOM，明确拒绝。"""

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i // 3]

    return k


def c_gather():
    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "int32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "int32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[b[i]]

    return k


def c6_complex():
    S, KD = 16, 16

    @T.prim_func
    def k(A: T.Tensor((S, KD), "float32"), C: T.Tensor((S, KD), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((S, KD), "float32")
            c = T.alloc_shared((S, KD), "float32")
            cnt = T.alloc_shared((1,), "int32")
            for tm in T.Parallel(S):
                cnt[0] = 0
                for g_i in T.serial(4):
                    if a[tm, g_i] != 0.0:
                        cnt[0] += 1
                        c[tm, g_i] = a[tm, g_i]

    return k


def c_gm():
    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = A[i]

    return k


# --- §6 第一步：T.Parallel 内嵌**直接内嵌且干净**的 T.serial 循环（作为额外迭代轴） ---

MS, KS = 64, 8


def s1_serial_copy():
    """S1（独立拷贝）：serial 体内无循环携带依赖。serial 轴 j 因"写连续"打分最高被选为
    向量轴 → 连续 vld/vsts（tail=8，用 update_mask 收尾）。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0

    return k


def s1_serial_predicated():
    """S1 + 谓词：serial 体内的 `if` 降成谓词化 store（掩码随 serial 向量轴逐 lane 变化）。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    if a[i, j] != 0.0:
                        c[i, j] = a[i, j]

    return k


def s3_serial_reduce():
    """S3（归约到随外层轴变化的位置）：serial 轴 j 上 `s[i] += a[i,j]` 是累积归约 →
    选为向量轴后走 vcadd 跨 lane 归约，末尾并入 s[i] 原值（vld BRC + ONEPT 回写）。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] += a[i, j]

    return k


def s2_serial_scalar():
    """S2（迭代私有**标量**累加器）：`acc=0.0; for j: acc+=a[i,j]; c[i]=acc`。
    acc[0] 是 local.var 单元素、lane 均匀的标量；init/store 把 serial 循环夹在中间。
    归约沿 serial 轴 j 走 vcadd，pre（init）与 post（写回 c[i]）在 chunk 循环外各执行一次。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                acc = T.alloc_var("float32")
                acc = 0.0
                for j in T.serial(KS):
                    acc += a[i, j]
                c[i] = acc

    return k


def s_serial_stencil():
    """边界：serial 体内的跨迭代读（stencil `c[i,j-1]`）不是累积自读 → 拒绝吸收，
    退回"nested serial loop"明确拒绝（宁拒绝勿错发）。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = c[i, j - 1] + a[i, j]

    return k


# --- §6 第二步（已支持）：T.Parallel 体内**多条语句**且其中夹着 T.serial ---
# body 是 SeqStmt 时，若其中**恰好含一个干净的 serial 循环**，则把 serial 吸收为
# 候选轴，其前置/后置兄弟语句作为 pre/post 段，在 serial 轴**外层**、每个并行迭代
# 各执行一次，发射在 chunk 循环**之外**（顺序：pre → 主体 → post）。


def m_serial_after_init():
    """M2 前置语句 + serial：`s[i]=0.0; for j in serial: s[i]+=a[i,j]`。
    几何上等价 `s[i]=Σ_j a[i,j]`，但 init 语句给出了归约**初值**（0），不能用旧值累加。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                s[i] = 0.0
                for j in T.serial(KS):
                    s[i] += a[i, j]

    return k


def m_serial_before_post():
    """M3 serial + 后置语句：`for j in serial: s[i]+=a[i,j]; s[i]=s[i]*2`。
    归约结果再顺序加工（乘 2），需要同一 parallel 迭代内的语句顺序组合。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] += a[i, j]
                s[i] = s[i] * 2.0

    return k


def m_serial_middle():
    """前置 + serial + 后置（夹心）：`s[i]=0.0; for j: s[i]+=a[i,j]; s[i]=s[i]*2`。
    同时含"归约初值"与"归约后处理"，是 M2 与 M3 的叠加。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                s[i] = 0.0
                for j in T.serial(KS):
                    s[i] += a[i, j]
                s[i] = s[i] * 2.0

    return k


def m_serial_indep_group():
    """M1 独立 store 组 + serial：`c[i,0]=1.0; for j in serial: c[i,j]=a[i,j]*2`。
    两组写互不相关（一组沿轴 uniform，一组沿轴连续），几何上各自可向量化。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                c[i, 0] = 1.0
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0

    return k


# ---------------------------------------------------------------------------
# §6 第三步：T.Parallel 体内有**多个并列**的干净 T.serial 循环（多段 program）。
# 逐段独立向量化 + 源序发射：段间依赖必须沿各自向量轴均匀（如 m[i]/s[i]）。
# ---------------------------------------------------------------------------


def p_two_serial_independent():
    """§6 第三步：两个并列的干净 serial 循环，互不相关。每段独立向量化，
    共享外层并行轴 i（只在最外层包裹一次），两段按源码顺序发射。"""

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        B: T.Tensor((MS, KS), "float32"),
        C: T.Tensor((MS, KS), "float32"),
        D: T.Tensor((MS, KS), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            d = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    d[i, l] = b[i, l] + 1.0

    return k


def p_serial_reduce_then_use():
    """§6 第三步：先沿 j 归约到 s[i]，再用 s[i] 广播写第二段 c[i,l]。
    段间依赖 s[i] 沿第二段的向量轴 l 均匀 → 逐段向量化 + 源序发射语义正确。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] += a[i, j]
                for l in T.serial(KS):
                    c[i, l] = s[i] * 2.0

    return k


def p_serial_compose():
    """§6 第三步：element-wise 组合——后段读前段写的 c[i,l]（同址、沿向量轴等值）。
    源序发射保证「先算 c 再算 d」，等价于标量语义。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32"), D: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            d = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    d[i, l] = c[i, l] + 1.0

    return k


def p_serial_leading_stmt():
    """§6 第三步：前置语句 + 两个 serial。首段之前语句组归属首段 ``pre``。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                s[i] = 0.0
                for j in T.serial(KS):
                    s[i] += a[i, j]
                for l in T.serial(KS):
                    c[i, l] = s[i] * 2.0

    return k


def p_serial_one_unclean():
    """§6 边界：并列 serial 中有一个跨迭代读（stencil ``c[i,l-1]``）→ 不能吸收该段，
    整体回落到既有单段路径并按「nested serial loop」拒绝（不做静默降级）。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    c[i, l] = c[i, l - 1] + a[i, l]

    return k


# ---------------------------------------------------------------------------
# 已知缺陷复现（探针 A–M）—— 这些用例**分析期多半会打印 [OK ]**，但发射结果错误 /
# 崩溃 / 契约不一致。对应的契约断言与 xfail 基线见
# ``testing/ascend/analysis/test_polyhedral_vectorize.py`` 的「已知缺陷回归基线」小节。
# ---------------------------------------------------------------------------


def d1_product_reduce():
    """D1：乘积归约 ``s[i] *= a[i,j]`` 被 _reduce_op(L1472) 默认成 vadd → 静默求和。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] * a[i, j]

    return k


def d2_plain_overwrite():
    """D2（已修）：``s[i] = a[i,j]`` 是覆写，旧判据曾判成归约 → 静默求和；现显式拒绝。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i, j in T.Parallel(MS, KS):
                s[i] = a[i, j]

    return k


def d3_int_max_reduce():
    """D3：int32 max 归约，_reduce_identity(L1713) 造 FloatImm(int32) → 发射期 ValueError。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "int32"), S: T.Tensor((MS,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "int32")
            s = T.alloc_shared((MS,), "int32")
            for i, j in T.Parallel(MS, KS):
                s[i] = T.max(s[i], a[i, j])

    return k


def d3_f16_max_reduce():
    """D3：f16 max 归约，单位元 ±3.4e38 超出 f16 值域 → 发射期 ValueError。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float16"), S: T.Tensor((MS,), "float16")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float16")
            s = T.alloc_shared((MS,), "float16")
            for i, j in T.Parallel(MS, KS):
                s[i] = T.max(s[i], a[i, j])

    return k


def d4_uniform_addend():
    """D4（已修）：``s[i] = s[i] + a[i,j] + 1.0`` 的常量加数曾被 _varying_operand 丢弃；现保留。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] + a[i, j] + 1.0

    return k


def d6_two_reduces():
    """D6（已修）：同一 nest 内两条归约，旧 _emit_reduce 只取第一条 uniform store → s2 丢失。"""

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"), B: T.Tensor((MS, KS), "float32"), S1: T.Tensor((MS,), "float32"), S2: T.Tensor((MS,), "float32")
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            s1 = T.alloc_shared((MS,), "float32")
            s2 = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s1[i] = s1[i] + a[i, j]
                    s2[i] = s2[i] + b[i, j]

    return k


def d6b_sum_max_reduce():
    """D6/L：sum 与 max 混合归约，_reduce_op 取第一个 → s2 的 vmax 归约丢失。"""

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"), B: T.Tensor((MS, KS), "float32"), S1: T.Tensor((MS,), "float32"), S2: T.Tensor((MS,), "float32")
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            s1 = T.alloc_shared((MS,), "float32")
            s2 = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s1[i] = s1[i] + a[i, j]
                    s2[i] = T.max(s2[i], b[i, j])

    return k


def d7_uniform_store_mixed():
    """D7（探针 M）：is_uniform(L693) 的 all([]) 真空真 → vload_strided 被误判均匀，
    build_plan 成功但 emit_plan 抛 Unsupported。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), D: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            d = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] + a[i, j]
                    d[i, j] = s[i]

    return k


def d8_negative_stride():
    """D8（已修）：Δ=-2 负步长改走索引向量 vgather2（下标 126..0，源程序合法）。"""
    M = 64

    @T.prim_func
    def k(A: T.Tensor((128,), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((128,), "float32")
            c = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                c[i] = a[126 - 2 * i]

    return k


def d8b_negative_stride_store():
    """D8 写侧（已修）：Δ=-2 负步长写同样走索引向量 vscatter。"""
    M = 64

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((128,), "float32")
            for i in T.Parallel(M):
                c[126 - 2 * i] = a[i]

    return k


def d9_symbolic_extent():
    """D9（已修）：符号 extent 走运行时 chunk 循环 ceildiv(n, lanes)，尾块掩码运行时夹紧。"""
    n = tirx.Var("n", "int32")

    @T.prim_func
    def k(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float32")
            c = T.alloc_shared((n,), "float32")
            for i in T.Parallel(n):
                c[i] = a[i]

    return k


def d9b_symbolic_extent_reduce():
    """D9 归约分支（已支持）：符号 extent 走「整块主循环 + peel 尾块」，
    ``local.var`` 向量累加器 + 运行期 ``update_mask`` 夹紧尾块。"""

    n = tirx.Var("n", "int32")

    @T.prim_func
    def k(A: T.Tensor((n,), "float32"), S: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float32")
            s = T.alloc_shared((1,), "float32")
            for i in T.Parallel(n):
                s[0] = s[0] + a[i]

    return k


def d10_one_dimensional_tail():
    """D10 一维形态（已修）：``(100,)`` 上尾块 ``vld(a[64], 64)`` 会越过分配范围。"""
    M = 100

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                c[i] = a[i]

    return k


def d15_bool_store():
    """D11 同族（已修）：``bool`` 型 store → build 期 Unsupported（不再 emit 期 ValueError）。"""
    M = 128

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((M,), "bool")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((M,), "bool")
            for i in T.Parallel(M):
                c[i] = a[i] > 0.0

    return k


def d11b_lane_invariant_indirect_load():
    """D11 同族（已支持）：``a[b[0]]`` 沿轴不变 → lane 内均匀标量读（``vdup`` 广播）。"""
    M = 128

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), B: T.Tensor((1,), "int32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            b = T.alloc_shared((1,), "int32")
            c = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                c[i] = a[b[0]] * 2.0

    return k


def d11_indirect_pre():
    """D11（已支持）：pre 的间接地址沿 serial 轴不变（Δ==0）→ lane 内均匀标量地址，
    发一次 ``ONEPT`` 写；serial 主体照常向量化（部分向量化）。"""

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), B: T.Tensor((1,), "int32"), D: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((1,), "int32")
            d = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                c[b[0]] = 1.0
                for j in T.serial(KS):
                    d[i, j] = a[i, j] * 2.0

    return k


def d12_masked_branch():
    """D12：if/else 同址合并遇上带谓词的臂 → normalize_body(L615) 的未声明拒绝路径。"""

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if a[i] > 0.0:
                    c[i] = 1.0
                else:
                    if b[i] > 0.0:
                        c[i] = 2.0

    return k


def d14_partial_reduce():
    """A0（已修）：吸收的 serial 轴上的「部分归约」→ 明确拒绝（原先静默只累加 2/128 项）。"""
    M, K = 8, 128

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), U: T.Tensor((M, K), "float32"), S: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            u = T.alloc_shared((M, K), "float32")
            s = T.alloc_shared((16,), "float32")
            for i in T.Parallel(M):
                for j in T.serial(K):
                    s[2 * i] = s[2 * i] + a[i, j]
                    u[i, j] = s[2 * i]
                for j in T.serial(K):
                    u[i, j] = a[i, j] * 2.0

    return k


def d14b_partial_reduce_fallback():
    """A0（已修）：单段路径上 serial 轴被拒后回退到并行轴，结果仍然正确。"""
    M, K = 8, 128

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), U: T.Tensor((M, K), "float32"), S: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            u = T.alloc_shared((M, K), "float32")
            s = T.alloc_shared((16,), "float32")
            for i in T.Parallel(M):
                for j in T.serial(K):
                    s[2 * i] = s[2 * i] + a[i, j]
                    u[i, j] = s[2 * i]

    return k


def d16_branch_load_oob():
    """D16（已修）：语句级 if/else 的臂内访存是**惰性**的，装载带条件守卫。

    ``if i >= 32: c[i] = a[i-32]`` 规范化成谓词化 store（``mask = i >= 32``），
    store 谓词同时作为取值表达式的装载守卫；``a[i-32]`` 的偏移下界为 -32（不可证
    安全），于是 chunk 0 发带掩码的 ``vgather2``（lane 0..31 被屏蔽，不再越界），
    而守卫在 chunk 1（``i ∈ [64,128)`` 恒真）已可证 → 回到普通 ``vld``。
    双臂同址 if/else 合并成惰性 ``if_then_else``（而不是急切 ``Select``），同理。
    """

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if i >= 32:
                    c[i] = a[i - 32]

    return k


def d16b_prefix_branch_load():
    """D16 镜像（已修）：**前缀**恒真守卫同样按 chunk 边界回退 ``vld``。

    ``if i < 96: c[i] = a[i+32]`` 的 ``a[i+32]`` 只在 i<96 时安全（i≥96 会越界到
    a[128..159]）；守卫在 chunk 0（``i ∈ [0,64)``）恒真 → 发普通 ``vld``，
    chunk 1 才需要带掩码的 ``vgather2``。前缀/后缀共用同一套守卫边界与分块发射。
    """

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if i < 96:
                    c[i] = a[i + 32]

    return k


# ---------------------------------------------------------------------------
# 设计层审计回归（R1–R7）——「统一模型 / 发射契约」层面的问题，多为 build 成功但
# 语义错误（静默错发）或 build 成功、emit 崩溃。修复后：拒绝，或发射等价形态。
# ---------------------------------------------------------------------------


def r1_predicate_width_f16():
    """R1（已修）：f16 尾块 + 条件使 store 掩码 = 用户谓词 ∧ 尾块谓词。

    旧实现用固定 b32 做 ``pand``（只覆盖 64 lane），f16 的 lane 64..127 被清零 →
    尾块只写一半。现按数据元素宽度 b16 取掩码。"""
    M = 100

    @T.prim_func
    def k(A: T.Tensor((M,), "float16"), B: T.Tensor((M,), "float16"), C: T.Tensor((M,), "float16")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float16")
            b = T.alloc_shared((M,), "float16")
            c = T.alloc_shared((M,), "float16")
            for i in T.Parallel(M):
                if a[i] > b[i]:
                    c[i] = a[i]

    return k


def r2_uniform_condition():
    """R2（已修）：循环不变条件（``a[0] > 0``）沿向量轴均匀，是 bool 标量。

    旧实现把它 ``vdup`` 成数据向量 → emit 期 ``Cannot infer default mask width for
    dtype 'bool'``；现走 ``predicate()`` 的 ``update_mask(cond ? lanes : 0)``。"""
    M = 100

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                if a[0] > 0.0:
                    c[i] = a[i]

    return k


def r3_masked_reduce():
    """R3（已修，改为拒绝）：条件归约 ``if i < 64: s[0] += a[i]`` 的用户谓词会被
    跨 lane 归约静默丢弃（对全部 lane 求和）→ build 期显式拒绝。"""
    M = 100

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), S: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            s = T.alloc_shared((1,), "float32")
            for i in T.Parallel(M):
                if i < 64:
                    s[0] = s[0] + a[i]

    return k


def r4_repeated_old_value():
    """R4（已修，改为拒绝）：``s[0] = s[0] + (s[0] + a[i,j])`` 的旧值出现两次，
    单一初值表达不了（会丢掉一份旧值）→ build 期显式拒绝。"""
    M, K = 4, 128

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), S: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            s = T.alloc_shared((1,), "float32")
            for i in T.Parallel(M):
                for j in T.serial(K):
                    s[0] = s[0] + (s[0] + a[i, j])

    return k


def r5_serial_nonzero_min():
    """R5（已修，改为拒绝）：``T.serial(1, KS)`` 的下界被迭代域建模丢弃（发射成
    从 0 开始的循环）→ 明确拒绝。"""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(1, KS):
                    c[i, j] = a[i, j] * 2.0

    return k


def r6_serial_waw():
    """R6（已修，改为拒绝）：同一 serial 体内 ``c[i,2*j]`` 与 ``c[i,j]`` 跨迭代写到
    同一地址，标量的"后写覆盖"在向量化后不成立 → 拒绝吸收该 serial 轴。"""
    MS, KS = 4, 128

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, 2 * KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, 2 * KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, 2 * j] = a[i, j]
                    c[i, j] = 2.0

    return k


def r7_reduce_parallel_pre_fallback():
    """R7（已修）：沿并行轴归约 + 非均匀 pre（``t[i] = 1.0``）在 build 期拒绝该轴，
    回退到 serial 轴 ``j``（正确的归约），不再出现 build OK / emit Unsupported。"""
    M, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((KS, M), "float32"), Tt: T.Tensor((M,), "float32"), S: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((KS, M), "float32")
            t = T.alloc_shared((M,), "float32")
            s = T.alloc_shared((1,), "float32")
            for i in T.Parallel(M):
                t[i] = 1.0
                for j in T.serial(KS):
                    s[0] = s[0] + a[j, i]

    return k


CASES = [
    # --- 向量化成功（OK）的用例 ---
    ("c0_copy", c0_copy),
    ("c0_broadcast", c0_broadcast),
    ("c1_ifelse", c1_ifelse),
    ("c1_ternary", c1_ternary),
    ("c1_single_arm", c1_single_arm),
    ("c2_rank", c2_rank),
    ("c2_reduce", c2_reduce),
    ("c2_reduce_tail", c2_reduce_tail),  # 改进 6：归约尾块
    ("r_long_reduce", r_long_reduce),  # 长归约（8 chunks）：真实循环 + local.var 累加器
    ("r_long_reduce_tail", r_long_reduce_tail),  # 长归约 + 尾块（9 chunks, tail=40）：尾块单独成块
    ("d9b_symbolic_extent_reduce", d9b_symbolic_extent_reduce),  # D9：符号 extent 归约（主循环 + peel 尾块）
    ("c2_axis_pick", c2_axis_pick),  # 改进 2：联合打分选轴
    ("c3_multi", c3_multi),
    ("c3_scalars", c3_scalars),
    ("c3_value_pow2", c3_value_pow2),  # 改进 3：值算子 strength-reduction
    ("c3_topk", c3_topk),  # 改进 1：strided store → vscatter
    ("c4_flat", c4_flat),
    ("c4_strided", c4_strided),
    ("c4_reverse", c4_reverse),  # 改进 1+4：PERMUTE → vgather2
    ("c4_tiled", c4_tiled),  # 改进 1+4：PIECEWISE → vgather2
    ("c_gather", c_gather),  # 改进 1+4：INDIRECT → vgather2
    ("c3_im2col", c3_im2col),  # 改进 2+7：最优轴 kk 因非 2 的幂除数被拒 → 回退到轴 m（diagnostics 保留两轴打分与卡点）
    ("s1_serial_copy", s1_serial_copy),  # §6 第一步 S1：serial 轴被吸收并向量化
    ("s1_serial_predicated", s1_serial_predicated),  # §6 第一步 S1 + 谓词
    ("s3_serial_reduce", s3_serial_reduce),  # §6 第一步 S3：serial 轴上的累积归约
    ("s2_serial_scalar", s2_serial_scalar),  # §6 第二步：私有标量累加器（pre init + post 写回）
    ("m_serial_after_init", m_serial_after_init),  # §6 第二步 M2：前置语句 + serial（归约初值）
    ("m_serial_before_post", m_serial_before_post),  # §6 第二步 M3：serial + 后置语句（归约后处理）
    ("m_serial_middle", m_serial_middle),  # §6 第二步：前置 + serial + 后置（夹心）
    ("m_serial_indep_group", m_serial_indep_group),  # §6 第二步 M1：独立 store 组 + serial
    ("p_two_serial_independent", p_two_serial_independent),  # §6 第三步：两个并列干净 serial（互不相关）
    ("p_serial_reduce_then_use", p_serial_reduce_then_use),  # §6 第三步：归约段 + 广播使用段
    ("p_serial_compose", p_serial_compose),  # §6 第三步：element-wise 组合（后段读前段写）
    ("p_serial_leading_stmt", p_serial_leading_stmt),  # §6 第三步：前置语句 + 两个 serial
    ("d8_negative_stride", d8_negative_stride),  # D8 已修：负步长 → 索引向量 vgather2
    ("d8b_negative_stride_store", d8b_negative_stride_store),  # D8 已修：负步长写 → vscatter
    ("d9_symbolic_extent", d9_symbolic_extent),  # D9 已修：符号 extent → 整块主循环 + peel 尾块
    ("d10_one_dimensional_tail", d10_one_dimensional_tail),  # D10 已修：一维尾块不再越界读
    ("d14b_partial_reduce_fallback", d14b_partial_reduce_fallback),  # A0 已修：回退到并行轴
    ("d11_indirect_pre", d11_indirect_pre),  # D11 已支持：pre 间接地址沿轴不变 → 外提为一次标量写
    ("d11b_lane_invariant_indirect_load", d11b_lane_invariant_indirect_load),  # D11 同族：vdup 广播
    # --- 明确拒绝（REJ）的用例，统一放在尾部 ---
    ("c6_complex", c6_complex),  # 共享标量 cnt[0] + 非直接内嵌 serial
    ("s_serial_stencil", s_serial_stencil),  # §6 边界：serial 体跨迭代读（stencil）
    ("p_serial_one_unclean", p_serial_one_unclean),  # §6 第三步边界：并列 serial 中有一段不干净
    ("c_gm", c_gm),  # GM 操作数
    ("c4_true_random", c4_true_random),  # 改进 4：TRUE_RANDOM
    ("d14_partial_reduce", d14_partial_reduce),  # A0：部分归约拒绝
    ("d15_bool_store", d15_bool_store),  # D11 同族：bool store 在 build 期拒绝
    # --- 已修缺陷复现（语义正确即可：显式拒绝，或安全地沿并行轴向量化）---
    ("d1_product_reduce", d1_product_reduce),  # D1：乘积语义保留（逐 lane vmul 读改写）
    ("d2_plain_overwrite", d2_plain_overwrite),  # D2：覆写按逐 lane 赋值发射（非归约）
    ("d3_int_max_reduce", d3_int_max_reduce),  # D3：int32 max 归约
    ("d3_f16_max_reduce", d3_f16_max_reduce),  # D3：f16 max 归约
    ("d4_uniform_addend", d4_uniform_addend),  # D4：均匀加数保留
    ("d6_two_reduces", d6_two_reduces),  # D6：两条归约各自发射
    ("d6b_sum_max_reduce", d6b_sum_max_reduce),  # D6：混合归约各自 recipe
    ("d7_uniform_store_mixed", d7_uniform_store_mixed),  # D7：build+emit 均 OK
    ("d12_masked_branch", d12_masked_branch),  # D12：未声明的拒绝路径（特征化）
    # --- 设计层审计回归（R1–R7）：修复后 = 拒绝 / 发射等价形态 ---
    ("r1_predicate_width_f16", r1_predicate_width_f16),  # R1：谓词运算掩码跟随元素宽度（b16）
    ("r2_uniform_condition", r2_uniform_condition),  # R2：均匀 bool 条件走 update_mask
    ("r3_masked_reduce", r3_masked_reduce),  # R3：条件归约拒绝
    ("r4_repeated_old_value", r4_repeated_old_value),  # R4：旧值重复出现拒绝
    ("r5_serial_nonzero_min", r5_serial_nonzero_min),  # R5：非 0 下界拒绝
    ("r6_serial_waw", r6_serial_waw),  # R6：跨迭代写写别名拒绝
    ("r7_reduce_parallel_pre_fallback", r7_reduce_parallel_pre_fallback),  # R7：轴回退到 serial 轴
    # --- 已修：语句级分支的臂内访存越界（装载守卫 + chunk 边界回退 vld） ---
    ("d16_branch_load_oob", d16_branch_load_oob),  # D16 后缀：chunk 0 带掩码，chunk 1 回退 vld
    ("d16b_prefix_branch_load", d16b_prefix_branch_load),  # D16 前缀：chunk 0 回退 vld，chunk 1 带掩码
]


def main() -> None:
    for name, fn in CASES:
        run(name, fn())


if __name__ == "__main__":
    main()
