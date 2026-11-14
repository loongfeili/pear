#!/usr/bin/env bash
# Draft: no remote upload or GPU execution until root explicitly coordinates it.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd -P)
if [[ ${1:-} != --run ]]; then
  exec python3 "$HERE/h20_gdn_regression.py" --describe
fi
: "${CUDA_VISIBLE_DEVICES:?Select one currently idle GPU by full UUID}"
[[ $CUDA_VISIBLE_DEVICES == GPU-* && $CUDA_VISIBLE_DEVICES != *,* ]] || exit 2
ROOT=${PEAR_ROOT:-/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114}
source "$ROOT/launch/runtime_paths.sh"
VERL_ENV=$(cd -- "$VERL_ENV" && pwd -P)
export VERL_ENV
source "$ROOT/repo/scripts/gpu_process_variables.sh" "$VERL_ENV"
SECONDS_LIMIT=${PEAR_GDN_REGRESSION_TIMEOUT_S:-600}
[[ $SECONDS_LIMIT =~ ^[1-9][0-9]*$ ]] && (( SECONDS_LIMIT <= 900 )) || exit 2
export PEAR_GDN_REGRESSION_BOUNDED=1
exec timeout --signal=TERM --kill-after=15s "${SECONDS_LIMIT}s" "$VERL_ENV/bin/python" "$HERE/h20_gdn_regression.py" "$@"
