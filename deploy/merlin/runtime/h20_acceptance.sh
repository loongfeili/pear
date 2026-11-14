#!/usr/bin/env bash
# The unchanged upstream result and strict FLA compatibility are separate evidence.
set -euo pipefail
ROOT=${PEAR_ROOT:-/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114}
LAUNCH=$ROOT/launch
LOG=$ROOT/preflight/logs
source "$LAUNCH/runtime_paths.sh"
[[ ! -e "$LOG/h20_compatibility_result.json" ]] || { echo 'Archive previous acceptance evidence before retrying'; exit 2; }
[[ ! -e "$LOG/upstream_gpu_suite.xml" ]] || { echo 'Archive the previous upstream XML before retrying'; exit 2; }
if bash "$LAUNCH/h20_smoke.sh" > "$LOG/h20_smoke.log" 2>&1; then
  suite_status=0
else
  suite_status=$?
fi
printf 'UPSTREAM_GPU_SUITE_EXIT %s\n' "$suite_status"
(( suite_status <= 1 )) || exit "$suite_status"
[[ -s "$LOG/upstream_gpu_suite.xml" ]] || exit 2

source "$ROOT/repo/scripts/gpu_process_variables.sh" "$VERL_ENV"
GPU_UUID=$("$VERL_ENV/bin/python" - <<'PYGPU'
import subprocess, torch
assert torch.cuda.device_count() == 8
uuid = str(torch.cuda.get_device_properties(7).uuid)
uuid = uuid if uuid.startswith('GPU-') else 'GPU-' + uuid
known = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
assert uuid in known
print(uuid)
PYGPU
)
# Check after the Torch UUID-probe process exits, so its own context cannot be mistaken for a worker.
python3 - <<'PYIDLE'
import subprocess
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
assert not apps.strip(), 'The full-suite GPU workers have not exited'
PYIDLE
CONTROL=$(mktemp -d "$LOG/gdn_regression.XXXXXX")
RESULTS=$CONTROL/results
FIXTURES=$ROOT/preflight/diagnostics/trial451841980/hopper_evidence_20261114T134839Z
if CUDA_VISIBLE_DEVICES=$GPU_UUID bash "$LAUNCH/h20_gdn_regression.sh" --run \
    --repo "$ROOT/repo" --fixtures-dir "$FIXTURES" --out-dir "$RESULTS" --backend original \
    > "$CONTROL/console.log" 2>&1; then
  regression_status=0
else
  regression_status=$?
fi
printf 'STRICT_GDN_PROCESS_EXIT %s (includes unchanged original assertions)\n' "$regression_status"
(( regression_status <= 1 )) || exit "$regression_status"
python3 "$LAUNCH/validate_h20_acceptance.py" \
  --profile "$LAUNCH/h20_acceptance_profile.json" --repo "$ROOT/repo" \
  --junit "$LOG/upstream_gpu_suite.xml" --regression-dir "$RESULTS" \
  --harness "$LAUNCH/h20_gdn_regression.py" --plan "$LAUNCH/h20_gdn_regression_plan.json" \
  --suite-exit "$suite_status" --regression-exit "$regression_status" \
  --output "$LOG/h20_compatibility_result.json"
printf '%s\n' "$RESULTS" > "$LOG/gdn_regression_path.txt"
