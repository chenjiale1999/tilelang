import tilelang
import tilelang.ascend.language as T
import tvm
from tilelang.ascend import transform as at


def _module_entry(func):
    """Wrap *func* in a module and return ``(module, entry_name)``."""
    mod = tvm.IRModule.from_expr(func)
    names = [gv.name_hint for gv in mod.get_global_vars()]
    assert len(names) == 1, names
    return mod, names[0]


def _run_SimdVFallocFragment(func):
    """P2: plan the P1 groups (body unchanged)."""
    mod, name = _module_entry(func)
    return at.AscendSimdVFallocFragment()(mod)[name]


def test_simdvf_two_fragments_copy_in_and_out_of_loop():
    @T.prim_func
    def main(
        A: T.Tensor((96,), "float32"),
        D: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "float32"),
        E: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((96,), "float32")
            d_ub = T.alloc_shared((64,), "float32")
            c_ub = T.alloc_shared((64,), "float32")
            e_ub = T.alloc_shared((64,), "float32")

            T.copy(A, a_ub)
            T.copy(D, d_ub)

            with T.SimdVF():
                # 用户在 SimdVF 内手动分配两个 fragment（寄存器级）
                frag  = T.alloc_fragment((64,), "float32")
                frag2 = T.alloc_fragment((64,), "float32")

                # ---- 第 1 个 Parallel：初始化 frag ----
                for i in T.Parallel(64):
                    frag[i] = a_ub[i + 32]

                # ---- Parallel 后的顶层语句：继续操作 frag ----
                for i in T.serial(64):
                    frag[i] = frag[i] * 2.0

                # ---- 第 2 个 Parallel：同时使用 frag 与 frag2 ----
                # 注意：这里 frag 在循环体内逐元素写回 c_ub，
                #       即 "其中一个 fragment 在循环里复制给 ub"
                for i in T.Parallel(64):
                    frag[i]  = frag[i] + d_ub[i]
                    c_ub[i]  = frag[i]              # 循环内 → 写入 UB
                    frag2[i] = frag[i] - 1.0

                # ---- 循环外：frag2 通过 T.copy 一次性写回 e_ub ----
                # 即 "另一个 fragment 在循环外复制给 ub"
                T.copy(frag2, e_ub)

            # SimdVF 外：把两个 UB 结果搬回 GM
            T.copy(c_ub, C)
            T.copy(e_ub, E)

    mod, name = _module_entry(main)
    return at.AscendSimdVFallocFragment()(mod)[name]

@tilelang.jit
def parallel_fa_compile_only():
    """Compile-only entry for ``run_lower_trace.sh`` (first JIT kernel is used)."""
    return parallel_flash_attention_fwd()


if __name__ == "__main__":
    import tilelang
    tilelang.testing.main()