"""Shared constants and helpers for the ``T.Parallel`` device regressions.

Split out of the former single-file suite; the cases live in
``test_tilelang_ascend_simdvf_parallel*.py``.
"""

import torch

import tilelang
import tilelang.testing

F32 = "float32"
NAN = float("nan")
INF = float("inf")

M, N = 8, 128


def _compile(func, out_idx=-1):
    """Compile a kernel and return it together with its generated CCE source.

    ``out_idx=None`` keeps every parameter as a caller-provided buffer, which is
    how an in-place update (and the read-back of an untouched lane) is expressed.
    """
    kernel = tilelang.compile(func, target="ascend", out_idx=out_idx)
    return kernel, kernel.get_kernel_source()


def _assert_close(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected, equal_nan=True, rtol=0, atol=0)


def _cycle(values, n, dtype=torch.float32):
    """Repeat ``values`` to exactly ``n`` elements (no length divisibility assumption)."""
    return torch.tensor([values[i % len(values)] for i in range(n)], dtype=dtype)


def _nz_physical(values, c0):
    """Logical ``(rows, cols)`` -> compact-NZ physical order ``[cols/C0, rows, C0]``.

    A UB buffer annotated with ``make_ascend_compact_nz_layout`` stores its
    elements in that order, and the UB->GM copy is a raw contiguous DMA, so the
    device output holds the physical permutation of the logical result.
    """
    rows, cols = values.shape
    return values.reshape(rows, cols // c0, c0).permute(1, 0, 2).reshape(rows, cols)
