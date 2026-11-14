#!/usr/bin/env bash
# Public entry point: exactly eight GPU workers, with one owner per RUN and node.
set -euo pipefail
umask 077
ROOT=${PEAR_ROOT:-/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
RANK=${ARNOLD_ID:?ARNOLD_ID is required}
TRIAL_ID=${ARNOLD_TRIAL_ID:?ARNOLD_TRIAL_ID is required}
[[ ${ARNOLD_ROLE:-} == worker ]] || { echo 'Run only on GPU worker pods' >&2; exit 2; }
[[ "$RANK" =~ ^[0-9]+$ ]] && (( 10#$RANK < 8 )) || { echo 'Exactly eight workers (ranks 0..7) are supported' >&2; exit 2; }
[[ "$TRIAL_ID" =~ ^[0-9]+$ ]] || exit 2
LAUNCH_ID=${PEAR_LAUNCH_ID:-merlin-trial-${TRIAL_ID}-stage1-64}
[[ "$LAUNCH_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
RESUME_ARGS=()
case ${PEAR_RESUME:-0} in
  0) ;;
  1) : "${PEAR_RUN:?Resuming requires an explicit PEAR_RUN}"; RESUME_ARGS=(--resume) ;;
  *) echo 'PEAR_RESUME must be 0 or 1' >&2; exit 2 ;;
esac
RUN=${PEAR_RUN:-$ROOT/work_dir/runs/$LAUNCH_ID}
# Claim ownership before selecting a network, restoring files, or writing failure.
exec python3 "$SCRIPT_DIR/launcher_control.py" \
  --root "$ROOT" --run "$RUN" --launch-id "$LAUNCH_ID" \
  --rank "$((10#$RANK))" --nodes 8 \
  --node-lock "${PEAR_NODE_LOCK:-/var/tmp/pear-merlin-$(id -u).active}" \
  "${RESUME_ARGS[@]}" -- bash "$SCRIPT_DIR/worker_entry.sh"
