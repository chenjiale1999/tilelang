#!/usr/bin/env bash
# P1/P2/P3 Parallel vectorization checks: build, offline structural tests and
# NPU device tests (locally or on a remote server).
#
# Usage:
#   ./run_parallel_vectorize_checks.sh
#       Build the project and run the offline P1/P2/P3 test suites.
#
#   ./run_parallel_vectorize_checks.sh --device
#       ... and run the device suites on the local NPU.
#
#   ./run_parallel_vectorize_checks.sh --remote user@host:/abs/path/to/tilelang
#       ... and run the device suites on a remote NPU server (repo must already
#       be synced there; the remote tree is not copied by this script).
#
#   ./run_parallel_vectorize_checks.sh --skip-build --device
#       Reuse the existing build directory.
#
# Environment:
#   JOBS=8                 build parallelism (default: nproc)
#   PYTHON=python3         interpreter used for pytest
#   CANN_ENV=...           optional set_env.sh used on the remote side
#
# Exit codes: 0 all passed, 1 a test/build failed, 2 usage or environment error.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DO_BUILD=1
RUN_DEVICE=0
REMOTE_SPEC=""
JOBS="${JOBS:-$(nproc)}"
PYTHON="${PYTHON:-python3}"
export PV_REPO_ROOT="$ROOT"

OFFLINE_TESTS=(
  testing/ascend/transform/test_ascend_parallel_vectorize_delta.py
  testing/ascend/transform/test_ascend_parallel_vectorize_plan.py
  testing/ascend/transform/test_ascend_parallel_vectorize_emit_dtypes.py
  testing/ascend/transform/test_ascend_parallel_vectorize_nz.py
  testing/ascend/transform/test_ascend_parallel_vectorize_pipeline.py
  testing/ascend/transform/test_ascend_vsimd_lift.py
  testing/ascend/transform/test_ascend_vsimd_ir.py
  testing/ascend/transform/test_ascend_vsimd_region.py
  testing/ascend/transform/test_ascend_vsimd_capture_scope.py
  testing/ascend/transform/test_ascend_vsimd_memory_legality.py
  testing/ascend/transform/test_ascend_vsimd_group_passes.py
)

DEVICE_TESTS=(
  testing/ascend/language/test_tilelang_ascend_simdvf_parallel.py
  testing/ascend/language/test_tilelang_ascend_simdvf_parallel_guards.py
  testing/ascend/language/test_tilelang_ascend_simdvf_parallel_reduce_nz.py
)

usage() {
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() {
  echo "error: $*" >&2
  exit 2
}

# Run pytest against the in-tree package. Some dev machines have an editable
# install pointing at another checkout; that finder wins over PYTHONPATH, so
# drop it and assert the imported tilelang really lives in this repo.
pytest_here() {
  "$PYTHON" - "$@" <<'PY'
import os
import sys

root = os.environ["PV_REPO_ROOT"]
sys.path.insert(0, root)
sys.meta_path = [
    m
    for m in sys.meta_path
    if "skbc_tilelang" not in getattr(m, "__module__", "") and "skbc_tilelang" not in type(m).__module__
]

import tilelang  # noqa: E402

resolved = os.path.realpath(tilelang.__file__)
if not resolved.startswith(os.path.realpath(root) + os.sep):
    raise SystemExit(f"tilelang resolved to {resolved}, expected a path under {root}")

import pytest  # noqa: E402

sys.exit(pytest.main(sys.argv[1:]))
PY
}

run_build() {
  echo "==> build (build.sh -j${JOBS})"
  ./build.sh -j"$JOBS"
}

run_offline() {
  echo "==> offline P1/P2/P3 tests"
  TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/tmp/tilelang_cache}" \
    pytest_here "${OFFLINE_TESTS[@]}" -q --no-header -p no:cacheprovider
  echo "==> analysis tests"
  TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/tmp/tilelang_cache}" \
    pytest_here testing/ascend/analysis -q --no-header -p no:cacheprovider
}

run_device_local() {
  echo "==> device tests (local NPU)"
  "$PYTHON" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch_npu") is None:
    sys.exit("torch_npu is not importable; use --remote <user@host:/repo> for device tests")
PY
  local cann_env
  for cann_env in \
    "${CANN_ENV:-}" \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/cann-*/set_env.sh; do
    if [[ -n "$cann_env" && -f "$cann_env" ]]; then
      # shellcheck disable=SC1090
      source "$cann_env"
      break
    fi
  done
  TILELANG_DISABLE_CACHE=1 \
    pytest_here "${DEVICE_TESTS[@]}" -q --no-header -p no:cacheprovider
}

run_device_remote() {
  local spec="$1" host path
  host="${spec%%:*}"
  path="${spec#*:}"
  [[ "$host" != "$spec" && -n "$path" ]] || die "--remote expects user@host:/abs/path/to/tilelang"
  echo "==> device tests on $host:$path"
  ssh "$host" bash -s -- "$path" <<'REMOTE'
set -euo pipefail
repo="$1"
cd "$repo"

for f in "${CANN_ENV:-}" /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann-*/set_env.sh; do
  if [[ -n "$f" && -f "$f" ]]; then
    # shellcheck disable=SC1090
    source "$f"
    break
  fi
done

export PV_REPO_ROOT="$repo"
export TILELANG_DISABLE_CACHE=1
python - testing/ascend/language/test_tilelang_ascend_simdvf_parallel*.py -q --no-header -p no:cacheprovider <<'PY'
import os
import sys

root = os.environ["PV_REPO_ROOT"]
sys.path.insert(0, root)
sys.meta_path = [
    m
    for m in sys.meta_path
    if "skbc_tilelang" not in getattr(m, "__module__", "") and "skbc_tilelang" not in type(m).__module__
]

import tilelang  # noqa: E402

resolved = os.path.realpath(tilelang.__file__)
if not resolved.startswith(os.path.realpath(root) + os.sep):
    raise SystemExit(f"tilelang resolved to {resolved}, expected a path under {root}")
print(f"tilelang: {resolved}")

import pytest  # noqa: E402

sys.exit(pytest.main(sys.argv[1:]))
PY
REMOTE
}

for arg in "$@"; do
  case "$arg" in
    --device) RUN_DEVICE=1 ;;
    --skip-build) DO_BUILD=0 ;;
    --remote) REMOTE_SPEC="__NEXT__" ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [[ "$REMOTE_SPEC" == "__NEXT__" ]]; then
        REMOTE_SPEC="$arg"
        RUN_DEVICE=0
      else
        die "unknown argument: $arg"
      fi
      ;;
  esac
done
[[ "$REMOTE_SPEC" == "__NEXT__" ]] && die "--remote expects user@host:/abs/path/to/tilelang"

[[ -f "$ROOT/build.sh" ]] || die "build.sh not found at $ROOT"

if [[ "$DO_BUILD" == 1 ]]; then
  run_build
else
  echo "==> build skipped"
fi

run_offline

if [[ -n "$REMOTE_SPEC" ]]; then
  run_device_remote "$REMOTE_SPEC"
elif [[ "$RUN_DEVICE" == 1 ]]; then
  run_device_local
else
  echo "==> device tests skipped (pass --device or --remote <user@host:/abs/path>)"
fi

echo "==> all requested checks passed"
