#!/usr/bin/env bash
# H20 Stage 1: either 8 x 8 GPUs (2 trainer/6 rollout) or 16 x 8 (4/12).
# Reuse the frozen eight-GPU environment gate; it does not certify multi-node training.
set -euo pipefail
[[ ${PEAR_SUPERVISED:-} == 1 ]] || { echo "Use merlin_stage1_entry.sh" >&2; exit 2; }
# Remove the idle NERO task's Python/source-check contamination only.
unset PYTHONPATH EXPECTED_GIT_COMMIT

ROOT=${PEAR_ROOT:?The owning controller must set PEAR_ROOT}
LAUNCH=$(cd "$(dirname "$0")" && pwd -P)
TOPOLOGY_ENV=$(python3 "$LAUNCH/topology.py" --shell)
eval "$TOPOLOGY_ENV"
RANK=${PEAR_NODE_RANK:-}
HEAD_ADDRESS=
LAUNCH_ID=
TOKEN_FILE=
STEPS=1000
TIMEOUT=1200
while [ "$#" -gt 0 ]; do
  case "$1" in
    --rank) RANK=${2:?}; shift 2 ;;
    --head-address) HEAD_ADDRESS=${2:?}; shift 2 ;;
    --launch-id) LAUNCH_ID=${2:?}; shift 2 ;;
    --root) ROOT=${2:?}; shift 2 ;;
    --token-file) TOKEN_FILE=${2:?}; shift 2 ;;
    --steps) STEPS=${2:?}; shift 2 ;;
    --timeout) TIMEOUT=${2:?}; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$RANK" =~ ^[0-9]+$ ]] && (( 10#$RANK < PEAR_NNODES )) || { echo "Set --rank (or PEAR_NODE_RANK) to 0..$((PEAR_NNODES - 1))"; exit 2; }
RANK=$((10#$RANK))
[[ "$HEAD_ADDRESS" =~ ^[^:[:space:]]+:[0-9]+$ ]] || { echo '--head-address must be an explicit IPv4-or-hostname:port'; exit 2; }
[[ "$LAUNCH_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo '--launch-id is required; use a NEW value for each cluster launch'; exit 2; }
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] && [[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] || exit 2
command -v setsid >/dev/null
export PEAR_ROOT=$ROOT
source "$LAUNCH/network_env.sh"
BASELINE_LAUNCH=$ROOT/launch
HELPER=$BASELINE_LAUNCH/preflight_stage1.py
[[ -f "$HELPER" ]] || { echo "Frozen baseline environment gate missing: $HELPER" >&2; exit 2; }
PEAR_ROOT=$ROOT bash "$BASELINE_LAUNCH/restore_runtime.sh"
REPO=$ROOT/repo
export WORK_DIR=$ROOT/work_dir
# Keep venv prefixes local even when the published compatibility aliases use NAS.
VERL_ENV=$(cd -- "${VERL_ENV:-${PEAR_ENVS_DIR:-$WORK_DIR/envs}/verl}" && pwd -P)
LIBERO_ENV=$(cd -- "${PEAR_ENVS_DIR:-$WORK_DIR/envs}/libero" && pwd -P)
LIBERO_MESA_DIR=$(cd -- "${PEAR_ENVS_DIR:-$WORK_DIR/envs}/mesa" && pwd -P)
export VERL_ENV LIBERO_MESA_DIR
export LIBERO_PYTHON=$LIBERO_ENV/bin/python LIBERO_CONFIG_PATH=$LIBERO_ENV/etc/libero
export RUN=${PEAR_RUN:?The owning controller must select and lock RUN}
export MODEL=${MODEL:-$WORK_DIR/models/Qwen3.8-27B}
export GPU=h100 STEPS
# Keep the reviewed full Stage 1 algorithm sizes; no smoke overrides leak in.
export BATCH=128 N=8 TP=4 MAX_PROMPT=4096 MAX_RESPONSE=160000 SESSIONS_PER_REPLICA=64
export EPOCHS=100
export PEAR_CUDA_COMPAT_DIR=${PEAR_CUDA_COMPAT_DIR:-$ROOT/compat/cuda-compat-13-0_580.196.07/usr/local/cuda-13.0/compat}
[[ -e "$PEAR_CUDA_COMPAT_DIR/libcuda.so.1" ]] || { echo 'Prepared CUDA compatibility library missing' >&2; exit 2; }
export LD_LIBRARY_PATH=$PEAR_CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# Downloads finish before this job; proxy tunnels must not intercept cluster communication.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export RAY_ADDRESS=$HEAD_ADDRESS PORT=${HEAD_ADDRESS##*:}
export LOCAL_DIR=/var/tmp/${USER:-pear}/pear_${LAUNCH_ID}_rank${RANK}
export NUM_CPUS=120 OBJECT_STORE_GB=64 LIBERO_SERVERS_PER_NODE=16
export RAY_AUTH_MODE=token RAY_AUTH_TOKEN_PATH=${TOKEN_FILE:?--token-file is required; create one shared owner-only token before submission}
[[ -s "$TOKEN_FILE" ]] || { echo 'Ray token file missing/empty'; exit 2; }
python3 - "$TOKEN_FILE" <<'PY'
import os, stat, sys
p=sys.argv[1]
if os.stat(p).st_mode & 0o077:
    raise SystemExit('Ray token file must not be accessible to group/others (chmod 600)')
PY

CONTROL=$ROOT/launches/$LAUNCH_ID
mkdir -p "$CONTROL/logs" "$LOCAL_DIR"
RAY_PID= TRAIN_PID= TRACKING_PID= FINISHED=0
fail() { printf '%s rank=%s %s\n' "$(date -u +%FT%TZ)" "$RANK" "$*" > "$CONTROL/failure.$RANK"; echo "$*" >&2; exit 1; }
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$FINISHED" != 1 && ! -f "$CONTROL/failure.$RANK" ]]; then
    printf 'rank=%s exited status=%s\n' "$RANK" "$status" > "$CONTROL/failure.$RANK"
  fi
  if [[ -n "$TRACKING_PID" ]]; then
    # Give the independent metrics bridge a bounded opportunity to drain JSONL
    # and finish its SDK Run. Its failure never changes the training exit code.
    printf '{"exit_code":%s}\n' "$status" > "$CONTROL/tracking.stop.json.partial" &&
      mv "$CONTROL/tracking.stop.json.partial" "$CONTROL/tracking.stop.json" || true
    for (( attempt = 0; attempt < 350; attempt++ )); do
      kill -0 "$TRACKING_PID" 2>/dev/null || break
      sleep 0.1
    done
  fi
  # The outer Linux subreaper owns the complete process tree, including orphaned
  # and setsid descendants. It performs verified cleanup even after leaders exit.
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if (( RANK == 0 )) && [[ ${PEAR_TRACKING_ENABLED:-0} == 1 ]]; then
  # Use the image's byted-wandb, not the frozen training venv's official wandb.
  env -u PYTHONPATH "${PEAR_TRACKING_PYTHON:?Set the image Python for Tracking}" \
    "$LAUNCH/tracking_bridge.py" --run "$RUN" --control "$CONTROL" \
    --launch-id "$LAUNCH_ID" --project "${PEAR_TRACKING_PROJECT:-pear-h20}" \
    > "$CONTROL/logs/tracking.log" 2>&1 &
  TRACKING_PID=$!
fi
check_failure() {
  local files=("$CONTROL"/failure.*)
  [[ ! -e "${files[0]}" ]] || fail "Another node failed; inspect $CONTROL/failure.*"
  [[ -z "$RAY_PID" ]] || kill -0 "$RAY_PID" 2>/dev/null || fail 'This node Ray process exited'
}
wait_file() {
  local start=$SECONDS
  until [[ -f "$1" ]]; do
    check_failure
    (( SECONDS - start < TIMEOUT )) || fail "Timed out waiting for $1"
    sleep 3
  done
}

# Exclusive rank claim refuses accidental reuse or two launchers on one rank.
( set -o noclobber; printf '%s\n' "$(hostname)" > "$CONTROL/claim.$RANK" ) 2>/dev/null || { FINISHED=1; echo 'Rank already claimed: use a new launch ID' >&2; exit 2; }
python3 "$HELPER" --root "$ROOT" --model-dir "$MODEL" > "$CONTROL/logs/preflight.$RANK.json" 2> "$CONTROL/logs/preflight.$RANK.err" || fail 'Node/environment preflight failed; see per-rank log'
# The original certificate deliberately retains candidate.nodes=32. It proves
# the frozen runtime/source and eight-GPU functional run, not this topology.
printf 'BASELINE_ENVIRONMENT_GATE_PASSED rank=%s scope=eight_gpu_functional_test requested_topology=%sx8\n' "$RANK" "$PEAR_NNODES"
if (( RANK == 0 )); then
  python3 - "$ROOT" "$LAUNCH" "$CONTROL" <<'PYCONFIG'
import hashlib, json, os, pathlib, sys
root, launch, control = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(launch))
from topology import topology_from_environment
topology = topology_from_environment()
certificate = root / 'work_dir/stage1_environment_ready.json'
saved = json.loads(certificate.read_text())
record = {
    'record_type': 'stage1_launch_configuration',
    'requested_topology': topology.as_dict(),
    'deployment_commit': os.environ.get('PEAR_DEPLOY_COMMIT'),
    'algorithm': {'batch_groups': 128, 'rollouts_per_group': 8, 'tp': 4, 'fsdp_size': 8, 'sp': 8,
                  'max_prompt_tokens': 4096, 'max_response_tokens': 160000, 'sessions_per_replica': 64},
    'baseline_certificate_path': str(certificate),
    'baseline_certificate_sha256': hashlib.sha256(certificate.read_bytes()).hexdigest(),
    'baseline_candidate_record_unchanged': saved['candidate'],
    'source_sha256': saved['source_sha256'], 'runtime_bundle_sha256': saved['runtime_bundle']['sha256'],
    'launcher_sha256': {name: hashlib.sha256((launch / name).read_bytes()).hexdigest()
                       for name in ['stage1_node.sh', 'merlin_stage1_entry.sh', 'worker_entry.sh', 'launcher_control.py', 'process_supervisor.py', 'network_env.sh', 'start_ray_node.sh', 'validate_cluster64.py', 'topology.py', 'job_entrypoint.sh', 'tracking_bridge.py', 'pear_merlin_ports.py']},
    'validation_scope': 'Frozen source/runtime and one actual 8-GPU GRPO step; not a multi-node training certificate.',
    'multi_node_training_verified': False,
}
(control / f'deployment{topology.total_gpus}.json').write_text(json.dumps(record, indent=2) + '\n')
PYCONFIG
fi
printf 'ready\n' > "$CONTROL/preflight.$RANK"
for (( rank = 0; rank < PEAR_NNODES; rank++ )); do wait_file "$CONTROL/preflight.$rank"; done
# Ray adds a timestamped session and socket name: the longer LOCAL_DIR is for
# compilation caches only. Every invocation gets its own short owner-only path.
RAY_TMPDIR=$(mktemp -d /var/tmp/pray.XXXXXX)
export RAY_TMPDIR
chmod 700 "$RAY_TMPDIR"
python3 - "$RAY_TMPDIR" <<'PY'
import os, stat, sys
path = sys.argv[1]
info = os.stat(path)
assert info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700
# Ten PID digits and a socket collision suffix conservatively cover Ray 2.58.
bound = path + "/session_9999-12-31_23-59-59_999999_2147483647/sockets/plasma_store.9999"
size = len(bound.encode())
assert size <= 107, f"Ray AF_UNIX socket path bound {size} exceeds 107 bytes"
print(f"RAY_SOCKET_PATH_BOUND bytes={size} limit=107 temp_dir={path}", flush=True)
PY
ROLE=rollout
(( RANK >= TRAINER_NNODES )) || ROLE=trainer
cd "$REPO"
if (( RANK == 0 )); then
  setsid bash "$LAUNCH/start_ray_node.sh" "$ROLE" > "$CONTROL/logs/ray.$RANK.log" 2>&1 &
  RAY_PID=$!
  # Announce only after the authenticated head is responsive.
  start=$SECONDS
  until "$VERL_ENV/bin/ray" status --address "$HEAD_ADDRESS" >/dev/null 2>&1; do
    check_failure
    (( SECONDS - start < TIMEOUT )) || fail 'Ray head readiness timed out'
    sleep 3
  done
  printf 'ready\n' > "$CONTROL/head.ready"
else
  wait_file "$CONTROL/head.ready"
  setsid bash "$LAUNCH/start_ray_node.sh" "$ROLE" "$HEAD_ADDRESS" > "$CONTROL/logs/ray.$RANK.log" 2>&1 &
  RAY_PID=$!
fi

if (( RANK == 0 )); then
  start=$SECONDS
  while ! "$VERL_ENV/bin/python" - "$HEAD_ADDRESS" "$LAUNCH" > "$CONTROL/logs/cluster.json" 2>&1 <<'PY'
import json, ray, sys
sys.path.insert(0, sys.argv[2])
from topology import topology_from_environment
topology = topology_from_environment()
ray.init(address=sys.argv[1], log_to_driver=False)
nodes=[n for n in ray.nodes() if n['Alive']]
resources=ray.cluster_resources()
errors = topology.resource_errors(nodes, resources)
print(json.dumps({'alive_nodes':len(nodes),'resources':resources,'errors':errors}, sort_keys=True))
ray.shutdown()
sys.exit(1 if errors else 0)
PY
  do
    check_failure
    (( SECONDS - start < TIMEOUT )) || fail "Expected $PEAR_NNODES nodes / $PEAR_TOTAL_GPUS GPUs with $TRAINER_NNODES trainer and $ROLLOUT_NNODES rollout nodes; see cluster.json"
    sleep 5
  done
  check_failure
  # This separately reviewed validation releases its per-node GPU actors before
  # training; a runtime certificate alone does not validate cross-node transport.
  "$VERL_ENV/bin/python" "$LAUNCH/validate_cluster64.py" --address "$HEAD_ADDRESS" --gpus "$PEAR_TOTAL_GPUS" \
    > "$CONTROL/logs/cluster_validation.json" 2>&1 || fail "$PEAR_TOTAL_GPUS-GPU cluster network/collective validation failed"
  check_failure
  # Reduced node count and host-cache capacity are deployment differences. Keep
  # context, batch, group size, FSDP8/SP8, TP4 and algorithm at upstream Stage 1 values.
  setsid bash examples/models/qwen3_8_27b_pear/train/run_stage1.sh \
    'actor_rollout_ref.rollout.custom.lmcache.l1_size_gb=600' \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='pear.rollout.server,pear.models.qwen3_5.tool_parser,pear_merlin_ports'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='$LAUNCH:$REPO/pear/utils/sitecustomize:$REPO:$REPO/3rdparty/verl/src'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.PEAR_RESERVED_MASTER_PORTS='1'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.PEAR_LAUNCH_ID=$LAUNCH_ID" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PEAR_CUDA_COMPAT_DIR=$PEAR_CUDA_COMPAT_DIR" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME='$NCCL_SOCKET_IFNAME'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_FAMILY=$NCCL_SOCKET_FAMILY" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_PYTHON=$LIBERO_PYTHON" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_MESA_DIR=$LIBERO_MESA_DIR" \
    > "$CONTROL/logs/training.log" 2>&1 &
  TRAIN_PID=$!
  while kill -0 "$TRAIN_PID" 2>/dev/null; do check_failure; sleep 5; done
  wait "$TRAIN_PID" || fail 'Training exited unsuccessfully; see training.log'
  TRAIN_PID=
  printf 'success\n' > "$CONTROL/training.done"
else
  # Training can run longer than readiness timeout; failures still propagate.
  until [[ -f "$CONTROL/training.done" ]]; do check_failure; sleep 5; done
fi
FINISHED=1
echo "Stage 1 process completed; inspect training metrics and checkpoints under $RUN"
