#!/usr/bin/env bash
# Dedicated eight-H20 task: restore only, then correctness/simulator/one-step GRPO validation.
set -euo pipefail
LAUNCH=$(cd "$(dirname "$0")" && pwd)
export PEAR_ROOT=${PEAR_ROOT:-/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114}
LOG=$PEAR_ROOT/preflight/logs
mkdir -p "$LOG"
exec > >(tee -a "$LOG/preflight_v2.log") 2>&1
rm -f "$LOG/PASSED" "$LOG/FAILED" "$PEAR_ROOT/work_dir/stage1_environment_ready.json"
trap 's=$?; if ((s)); then printf "%s\n" "$s" > "$LOG/FAILED"; fi' EXIT
bash "$LAUNCH/restore_runtime.sh"
source "$LAUNCH/runtime_paths.sh"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export HF_HUB_OFFLINE=1
[[ -s "$WORK_DIR/models/Qwen3.8-27B/.pear_model" ]] || { echo 'Model download is not complete'; exit 1; }
cd "$PEAR_ROOT/repo"
python3 "$LAUNCH/preflight_stage1.py" --root "$PEAR_ROOT" --check-source-only
PEAR_CHECK_ONLY=1 bash scripts/install.sh libero_run > "$LOG/libero_run.log" 2>&1
bash "$LAUNCH/h20_acceptance.sh" > "$LOG/h20_acceptance.log" 2>&1
bash "$LAUNCH/single_node_grpo_smoke.sh" > "$LOG/single_node_grpo_smoke.log" 2>&1
printf 'PEAR_H20_PREFLIGHT_PASSED profile=h20_fla_reference_compat_v1 %s\n' "$(date -Is)" > "$LOG/PASSED"
python3 "$LAUNCH/preflight_stage1.py" --root "$PEAR_ROOT" --certify
printf 'PEAR_H20_PREFLIGHT_V2_CERTIFIED %s\n' "$(date -Is)"
