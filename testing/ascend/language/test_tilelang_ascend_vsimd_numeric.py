"""Device numeric regression for the P1→P2→P3 ``T.Parallel`` vectorization.

Every case here is a ``T.SimdVF`` kernel whose ``T.Parallel`` nest is lifted by
P1, planned by P2 and emitted by P3 (the result is checked to really contain the
vector intrinsic, so a silent fallback to the legacy scalar path cannot make a
numeric test pass).  The kernel then runs on an Ascend NPU and is compared with
a golden computed from the *scalar* semantics of the source program.

Covered ingredients:

* ``T.max`` / ``T.min`` NaN and ``±0`` ordering — the pinned contract is
  ``max(a, b) = a > b ? a : b`` and ``min(a, b) = a < b ? a : b`` (the same
  contract the PTO branch materializes as an explicit ``Select``), so a NaN
  operand selects the *other* operand and equal values (``+0``/``-0``) select
  the second one;
* element-wise tail chunk (``update_mask`` driven);
* single-arm ``if`` (predicated store) and two-arm divergent stores (one
  complementary predicated store per arm);
* mask ``==`` / ``!=`` (predicate truth table, ``pxor``/``pnot``);
* a constant ``T.serial`` shell (last write wins, unrolled at plan time) and an
  accumulating shell (absorbed as a reduction axis, ``vcadd``);
* a strided load (per-lane index vector, ``vgather2``) and an indirect gather
  (runtime offsets, owner-wide access declaration);
* narrowing cast stores: ``f32→bf16``, ``f16→fp8`` (``vcvt`` + ``PK`` store) and
  integer narrowing ``i32→u8`` / ``i16→u8``;
* same-width signedness (``i32→u32``): a register reinterpret with **no**
  ``vcvt`` instruction;
* widening cast stores: ``f16→f32`` / ``i16→i32`` (``UNPK_B16`` load) and
  ``i8→i32`` / ``fp8→f32`` (``UNPK4_B8`` load).
"""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T

F32 = "float32"


def _compile(func, out_idx=-1):
    """Compile a kernel and return it together with its generated CCE source.

    ``out_idx=None`` keeps every parameter as a caller-provided buffer, which is
    how an in-place update (and the read-back of an untouched lane) is expressed.
    """
    kernel = tilelang.compile(func, target="ascend", out_idx=out_idx)
    return kernel, kernel.get_kernel_source()


def _assert_close(actual, expected, atol=0.0, rtol=0.0):
    torch.testing.assert_close(actual.cpu(), expected, equal_nan=True, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# T.max / T.min: NaN operand position and ±0 ordering
# ---------------------------------------------------------------------------

NAN = float("nan")
INF = float("inf")

# Every row is one operand pair; the mixed NaN/±0 rows are the contract.
_MAXMIN_A = [
    NAN, 1.0, NAN, 1.0,          # NaN left / right / both
    0.0, -0.0, 0.0, -0.0,        # ±0 in both orders
    INF, -INF, 1.0, -1.0,        # infinities and ordinary values
    0.0, 1.0, NAN, NAN,
]
_MAXMIN_B = [
    1.0, NAN, NAN, 1.0,
    -0.0, 0.0, 0.0, -0.0,
    -INF, INF, 1.0, -1.0,
    0.0, 0.0, 2.0, NAN,
]


def _maxmin_kernel(n, kind):
    @T.prim_func
    def main(A: T.Tensor((n,), F32), B: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            b_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if kind == "max":
                        c_ub[i] = T.max(a_ub[i], b_ub[i])
                    else:
                        c_ub[i] = T.min(a_ub[i], b_ub[i])
            T.copy(c_ub, C)

    return main


@pytest.mark.parametrize("kind", ["max", "min"])
@tilelang.testing.requires_ascend
def test_parallel_max_min_nan_and_signed_zero(kind):
    n = 128
    a = torch.tensor(_MAXMIN_A * (n // len(_MAXMIN_A)), dtype=torch.float32)
    b = torch.tensor(_MAXMIN_B * (n // len(_MAXMIN_B)), dtype=torch.float32)

    kernel, source = _compile(_maxmin_kernel(n, kind))
    assert f"simd_inst::v{kind}(" in source

    actual = kernel(a.npu(), b.npu()).cpu()
    # Pinned contract: ``a > b ? a : b`` (max) / ``a < b ? a : b`` (min).  A NaN
    # operand makes the comparison false, so the *other* operand wins; on equal
    # operands the second one wins, which fixes the sign of a zero result.
    expected = torch.where(a > b, a, b) if kind == "max" else torch.where(a < b, a, b)

    _assert_close(actual, expected)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros])), (
        "the sign of ±0 must follow the (a > b) / (a < b) contract"
    )


# ---------------------------------------------------------------------------
# tail chunk
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_elementwise_tail():
    n = 1000  # 15 full float32 chunks (64 lanes) + a 40-lane tail

    @T.prim_func
    def main(A: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "update_mask_b32" in source, "the tail chunk must be predicated"

    a = torch.arange(n, dtype=torch.float32)
    _assert_close(kernel(a.npu()), a * 2.0)


@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@tilelang.testing.requires_ascend
def test_parallel_eight_bit_elementwise_tail(dtype):
    """An 8-bit tail is gathered into 16-bit lanes and written with ``PK_B16``.

    A ``NORM_B8`` store would reinterpret those 16-bit lanes as interleaved
    bytes and overrun the buffer (the row is shorter than one byte register).
    """
    n = 100  # one 256-byte chunk, so the whole vector is a 100-lane tail
    torch_dtype = getattr(torch, dtype)

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), dtype)
            c_ub = T.alloc_shared((n,), dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2(" in source, "the byte tail load is a widening gather"
    assert "vsts_pack_b16" in source, "the widened value is narrowed back by the pack store"

    if dtype == "uint8":
        a = (torch.arange(n, dtype=torch.int32) % 256).to(torch_dtype)
    else:
        a = (torch.arange(n, dtype=torch.int32) % 251 - 125).to(torch_dtype)
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.uint8), a.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("src_dtype", ["uint8", "int8"])
@tilelang.testing.requires_ascend
def test_parallel_eight_bit_same_width_reinterpret_tail(src_dtype):
    """``i8 <-> u8`` keeps the byte gather: bit identity through the pack store."""
    n = 100
    dst_dtype = "int8" if src_dtype == "uint8" else "uint8"
    src_torch = getattr(torch, src_dtype)

    @T.prim_func
    def main(A: T.Tensor((n,), src_dtype), C: T.Tensor((n,), dst_dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), src_dtype)
            c_ub = T.alloc_shared((n,), dst_dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast(dst_dtype, a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2(" in source
    assert "vsts_pack_b16" in source

    a = (torch.arange(n, dtype=torch.int32) % 256).to(src_torch)
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.uint8), a.view(torch.uint8), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# guarded stores
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_predicated_single_arm_store():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(C, c_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if a_ub[i] > 0.0:
                        c_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main, out_idx=None)

    # NaN / -0 / +0 / infinities around the boundary.
    a = torch.tensor(
        [NAN, -1.0, -0.0, 0.0, 1.0, -2.0, INF, -INF, 3.0, NAN] * (n // 10),
        dtype=torch.float32,
    )
    c_in = torch.arange(n, dtype=torch.float32) * 0.5
    c_npu = c_in.npu()
    kernel(a.npu(), c_npu)
    assert "vcmps_gt" in source, "the guard must be a vector compare"
    # The store is masked, so inactive lanes keep the old destination value.
    expected = torch.where(a > 0.0, a * 2.0, c_in)
    _assert_close(c_npu, expected)


@tilelang.testing.requires_ascend
def test_parallel_divergent_two_arm_store():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), C: T.Tensor((n,), F32), D: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            d_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(C, c_ub)
            T.copy(D, d_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if a_ub[i] > 0.0:
                        c_ub[i] = a_ub[i]
                    else:
                        d_ub[i] = 1.0
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    kernel, source = _compile(main, out_idx=None)
    assert "simd_inst::pnot(" in source, "each arm keeps its own complementary mask"

    a = torch.tensor([NAN, -1.0, 0.0, 1.0, 2.0, -3.0] * (n // 6), dtype=torch.float32)
    c_in = torch.full((n,), -7.0, dtype=torch.float32)
    d_in = torch.full((n,), -9.0, dtype=torch.float32)
    c_npu, d_npu = c_in.npu(), d_in.npu()
    kernel(a.npu(), c_npu, d_npu)

    branch = a > 0.0
    _assert_close(c_npu, torch.where(branch, a, c_in))
    _assert_close(d_npu, torch.where(branch, d_in, torch.ones_like(d_in)))


# ---------------------------------------------------------------------------
# mask == / !=
# ---------------------------------------------------------------------------


def _mask_eq_kernel(n, op):
    @T.prim_func
    def main(A: T.Tensor((n,), F32), B: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            b_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(C, c_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if op == "eq":
                        if (a_ub[i] > 0.0) == (b_ub[i] > 0.0):
                            c_ub[i] = 1.0
                    else:
                        if (a_ub[i] > 0.0) != (b_ub[i] > 0.0):
                            c_ub[i] = 1.0
            T.copy(c_ub, C)

    return main


@pytest.mark.parametrize("op", ["eq", "ne"])
@tilelang.testing.requires_ascend
def test_parallel_mask_equality(op):
    n = 128
    kernel, source = _compile(_mask_eq_kernel(n, op), out_idx=None)
    assert "simd_inst::pxor(" in source, "mask ==/!= is emitted through the predicate truth table"
    assert "simd_inst::vcmp(" not in source

    a = torch.tensor([NAN, 1.0, -1.0, 0.0, 2.0, -0.0, NAN, 3.0] * (n // 8), dtype=torch.float32)
    b = torch.tensor([1.0, NAN, 1.0, -0.0, -2.0, 0.0, NAN, 0.0] * (n // 8), dtype=torch.float32)
    c_in = torch.full((n,), -5.0, dtype=torch.float32)
    c_npu = c_in.npu()
    kernel(a.npu(), b.npu(), c_npu)

    same = (a > 0.0) == (b > 0.0)
    if op == "ne":
        same = ~same
    _assert_close(c_npu, torch.where(same, torch.ones_like(c_in), c_in))


# ---------------------------------------------------------------------------
# constant serial shells
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_serial_shell_last_write_wins():
    rows, steps = 64, 4

    @T.prim_func
    def main(A: T.Tensor((rows, steps), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, steps), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for k in T.serial(steps):
                        c_ub[i] = a_ub[i, k]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert source.count("simd_inst::vsts_norm(") == steps, "the shell is unrolled iteration by iteration"

    a = torch.arange(rows * steps, dtype=torch.float32).reshape(rows, steps)
    _assert_close(kernel(a.npu()), a[:, -1])


@tilelang.testing.requires_ascend
def test_parallel_serial_shell_reduction():
    rows, k = 8, 64

    @T.prim_func
    def main(A: T.Tensor((rows, k), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, k), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            T.fill(c_ub, 0.0)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for j in T.serial(k):
                        c_ub[i] = c_ub[i] + a_ub[i, j]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcadd(" in source, "the shell accumulator is a cross-lane reduction"

    # Small integers keep the sequential and vector sums bit-identical.
    a = (torch.arange(rows * k, dtype=torch.int32).reshape(rows, k) % 7).to(torch.float32)
    _assert_close(kernel(a.npu()), a.sum(dim=1))


@tilelang.testing.requires_ascend
def test_parallel_serial_shell_kept_as_a_scalar_loop():
    """A shell that cannot be absorbed stays a real loop around the vector work.

    ``c[i] = a[i, k]`` writes one address per lane, so ``k`` is neither a vector
    axis nor a reduction; the plan keeps it as a scalar shell instead of falling
    back (and does not unroll it, unlike the constant short shell above).
    """
    rows, steps = 64, 16

    @T.prim_func
    def main(A: T.Tensor((rows, steps), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, steps), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for k in T.serial(steps):
                        c_ub[i] = a_ub[i, k]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    # One store per chunk inside the shell loop: 16 unrolled stores would mean
    # the shell was replicated instead of kept as a loop.
    assert source.count("simd_inst::vsts_norm(") == 1, source
    assert "simd_inst::vgather2(" in source

    a = torch.arange(rows * steps, dtype=torch.float32).reshape(rows, steps)
    _assert_close(kernel(a.npu()), a[:, -1])


@tilelang.testing.requires_ascend
def test_parallel_under_a_serial_shell():
    """A ``T.Parallel`` nested under a serial loop inside the VF is vectorized.

    The serial loop stays scalar and keeps its counter; the vector code replaces
    only the Parallel body, so the address may reference ``k``.
    """
    rows, steps = 128, 4

    @T.prim_func
    def main(A: T.Tensor((steps, rows), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((steps, rows), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            T.fill(c_ub, 0.0)
            with T.SimdVF():
                for k in T.serial(steps):
                    for i in T.Parallel(rows):
                        c_ub[i] = c_ub[i] + a_ub[k, i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vadd" in source
    assert "simd_inst::vlds_norm(" in source

    a = (torch.arange(steps * rows, dtype=torch.int32).reshape(steps, rows) % 5).to(torch.float32)
    _assert_close(kernel(a.npu()), a.sum(dim=0))


# ---------------------------------------------------------------------------
# strided load
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_strided_load():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((2 * n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2 * n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[2 * i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2(" in source, "a stride-2 load uses the per-lane index vector"

    a = torch.arange(2 * n, dtype=torch.float32)
    _assert_close(kernel(a.npu()), a[0::2])


# ---------------------------------------------------------------------------
# narrowing cast stores
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_narrowing_cast_store():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), C: T.Tensor((n,), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), "bfloat16")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast("bfloat16", a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcvt" in source and "PART_EVEN" in source
    assert "vsts_pack_b32" in source, "64 bf16 results are written through the packed form"

    # Exact in both float32 and bfloat16: no rounding-mode dependence.
    a = (torch.arange(n, dtype=torch.float32) % 8) - 4.0
    _assert_close(kernel(a.npu()), a.to(torch.bfloat16))


@pytest.mark.parametrize("src_dtype,pack", [("int32", "vsts_pack_quarter"), ("int16", "vsts_pack_b16")])
@tilelang.testing.requires_ascend
def test_parallel_integer_narrowing_cast_store(src_dtype, pack):
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), src_dtype), C: T.Tensor((n,), "uint8")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), src_dtype)
            c_ub = T.alloc_shared((n,), "uint8")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast("uint8", a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcvt" in source and pack in source

    # Every value is inside the uint8 range, so the expected result is the same
    # whether the narrowing saturates or wraps; only the conversion is checked.
    a = (torch.arange(n, dtype=torch.int32) % 256).to(getattr(torch, src_dtype))
    golden = a.to(torch.uint8)
    _assert_close(kernel(a.npu()), golden)


@tilelang.testing.requires_ascend
def test_parallel_fp16_to_fp8_store():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), "float16"), C: T.Tensor((n,), "float8_e4m3fn")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float16")
            c_ub = T.alloc_shared((n,), "float8_e4m3fn")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast("float8_e4m3fn", a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcvt" in source and "vsts_pack_b16" in source

    # All values are exactly representable in e4m3 (3 mantissa bits).
    values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 8.0]
    a = torch.tensor(values * (n // len(values)), dtype=torch.float16)
    golden = a.to(torch.float8_e4m3fn)
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.uint8), golden.view(torch.uint8), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# same-width signedness (register reinterpret) and widening casts
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_same_width_signedness_reinterpret():
    """``i32 -> u32`` is bit-identical: a reinterpret, never a ``vcvt``."""
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), "int32"), C: T.Tensor((n,), "uint32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "int32")
            c_ub = T.alloc_shared((n,), "uint32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast("uint32", a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcvt" not in source, "a same-width sign change costs no instruction"
    assert "simd_inst::vld" in source and "simd_inst::vsts" in source

    # Every bit pattern must survive, including the sign bit and INT32_MIN.
    a = torch.tensor(
        [-2147483648, -1, -12345, 0, 1, 12345, 2147483647, -7] * (n // 8), dtype=torch.int32
    )
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.int32), a, rtol=0, atol=0)


_WIDENING_CASES = (
    ("float16", "float32", "vlds_unpack", "PART_EVEN"),
    ("int16", "int32", "vlds_unpack", "PART_EVEN"),
    ("int8", "int32", "vlds_unpack4", "PART_P0"),
    ("int8", "int16", "vlds_unpack", "PART_EVEN"),
    ("uint8", "uint16", "vlds_unpack", "PART_EVEN"),
    ("uint8", "float16", "vlds_unpack", "PART_EVEN"),
    ("int8", "float16", "vlds_unpack", "PART_EVEN"),
    ("float8_e4m3fn", "float32", "vlds_unpack4", "PART_P0"),
)

_FP8_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 8.0]


@pytest.mark.parametrize(
    "src_dtype,dst_dtype,load,part", _WIDENING_CASES, ids=[case[0] + "->" + case[1] for case in _WIDENING_CASES]
)
@tilelang.testing.requires_ascend
def test_parallel_widening_cast_store(src_dtype, dst_dtype, load, part):
    """A widening store reads its source through the ``UNPK_*`` load layout."""
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), src_dtype), C: T.Tensor((n,), dst_dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), src_dtype)
            c_ub = T.alloc_shared((n,), dst_dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast(dst_dtype, a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert f"simd_inst::{load}" in source, "the widening load must spread the source elements"
    assert part in source, "the conversion reads the spread sub-slots"

    if src_dtype == "float8_e4m3fn":
        a = torch.tensor(_FP8_VALUES * (n // len(_FP8_VALUES)), dtype=torch.float32).npu()
        a = a.to(torch.float8_e4m3fn)
        golden = a.cpu().to(torch.float32)
        _assert_close(kernel(a), golden)
        return

    if src_dtype == "float16":
        # 0..127 are exact in float16, so the widening result is bit-exact too.
        a = torch.arange(n, dtype=torch.float16)
        golden = a.to(torch.float32)
    elif src_dtype == "int16":
        a = (torch.arange(n, dtype=torch.int32) % 401 - 200).to(torch.int16)
        golden = golden_tensor(a, dst_dtype)
    elif src_dtype == "uint8":
        a = (torch.arange(n, dtype=torch.int32) % 256).to(torch.uint8)
        golden = golden_tensor(a, dst_dtype)
    else:
        a = (torch.arange(n, dtype=torch.int32) % 251 - 125).to(torch.int8)
        golden = golden_tensor(a, dst_dtype)
    _assert_close(kernel(a.npu()), golden)


def golden_tensor(values, dst_dtype):
    # 8-bit values are exactly representable in int16/uint16/float16.
    return values.to(getattr(torch, dst_dtype))


@tilelang.testing.requires_ascend
def test_parallel_indirect_gather():
    """A runtime index declares the whole owner; the values must still be exact."""
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), B: T.Tensor((n,), "int32"), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            b_ub = T.alloc_shared((n,), "int32")
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[b_ub[i]]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2" in source

    a = torch.arange(n, dtype=torch.float32)
    b = torch.randperm(n).to(torch.int32)
    _assert_close(kernel(a.npu(), b.npu()), a[b.long()])


if __name__ == "__main__":
    tilelang.testing.main()
