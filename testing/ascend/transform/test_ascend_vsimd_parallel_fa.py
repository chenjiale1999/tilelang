"""Structural tests for the parallel-softmax FA closed-loop tile (PR-E)."""

import pathlib
import sys

import tilelang
from tilelang.ascend import transform as at

_REPO = pathlib.Path(__file__).resolve().parents[3]
_FA_DIR = _REPO / "examples/ascend/flash_attention"
if str(_FA_DIR) not in sys.path:
    sys.path.insert(0, str(_FA_DIR))

from parallel_fa import parallel_flash_attention_fwd  # noqa: E402


def _report(func):
    return str(at.AscendAnalyzeVSimdRegionReport(func))


def test_parallel_fa_direct_nz_store_lifts():
    func = parallel_flash_attention_fwd()
    report = _report(func)

    assert "status=LIFTED" in report
    assert "domain: i in [0, 128) j in [0, 128)" in report
    assert "W P_nz_ub[i, j] = T.Cast(" in report
    assert "layout=annotated" in report
    assert "layouts:" in report
    assert "verify: OK" in report
    # The P_nz store must be in the same region as the P_ub store.
    assert "W P_ub[i, j] = T.exp" in report


def test_parallel_fa_scope_has_layout():
    func = parallel_flash_attention_fwd()
    infos = at.AscendAnalyzeSimdVFCaptureInfo(func)
    nz_regions = [
        info
        for info in infos
        if any("P_nz_ub" in str(name) for name in info["captures"])
    ]
    assert nz_regions
    assert any("P_nz_ub" in lane for lane in nz_regions[0]["layouts"])


if __name__ == "__main__":
    tilelang.testing.main()
