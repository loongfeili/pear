#!/usr/bin/env bash
# Public entry point: 64 or 128 GPUs, with one owner per RUN and node.
set -euo pipefail
umask 077
ROOT=${PEAR_ROOT:-/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
TOPOLOGY_ENV=$(python3 "$SCRIPT_DIR/topology.py" --shell)
eval "$TOPOLOGY_ENV"
RANK=${ARNOLD_ID:?ARNOLD_ID is required}
TRIAL_ID=${ARNOLD_TRIAL_ID:?ARNOLD_TRIAL_ID is required}
[[ ${ARNOLD_ROLE:-} == worker ]] || { echo 'Run only on GPU worker pods' >&2; exit 2; }
[[ "$RANK" =~ ^[0-9]+$ ]] && (( 10#$RANK < PEAR_NNODES )) || { echo "Expected $PEAR_NNODES workers (ranks 0..$((PEAR_NNODES - 1)))" >&2; exit 2; }
[[ "$TRIAL_ID" =~ ^[0-9]+$ ]] || exit 2
LAUNCH_ID=${PEAR_LAUNCH_ID:-merlin-trial-${TRIAL_ID}-stage1-${PEAR_TOTAL_GPUS}}
[[ "$LAUNCH_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
RESUME_ENABLED=0
case ${PEAR_RESUME:-0} in
  0) ;;
  1) : "${PEAR_RUN:?Resuming requires an explicit PEAR_RUN}"; RESUME_ENABLED=1 ;;
  *) echo 'PEAR_RESUME must be 0 or 1' >&2; exit 2 ;;
esac
RUN=${PEAR_RUN:-$ROOT/work_dir/runs/$LAUNCH_ID}
CONTROL_ARGS=(--root "$ROOT" --run "$RUN" --launch-id "$LAUNCH_ID"
              --rank "$((10#$RANK))" --nodes "$PEAR_NNODES"
              --node-lock "${PEAR_NODE_LOCK:-/var/tmp/pear-merlin-$(id -u).active}")
if (( RESUME_ENABLED )); then CONTROL_ARGS+=(--resume); fi
# Claim ownership before selecting a network, restoring files, or writing failure.
exec python3 "$SCRIPT_DIR/launcher_control.py" "${CONTROL_ARGS[@]}" -- bash "$SCRIPT_DIR/worker_entry.sh"
