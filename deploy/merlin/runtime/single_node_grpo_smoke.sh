#!/usr/bin/env bash
# One isolated 8-GPU H20 allocation: real Qwen3.8-27B + LIBERO + one GRPO update.
# Run after h20_smoke.sh's GPU correctness tests. This never alters the 32-node recipe.
set -euo pipefail

export PEAR_ROOT=${PEAR_ROOT:-/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114}
PEAR_REPO=${PEAR_REPO:-$PEAR_ROOT/repo}
export WORK_DIR=${WORK_DIR:-$PEAR_ROOT/work_dir}
# Canonicalize environment directories while preserving each venv's bin/python.
VERL_ENV=$(cd -- "${VERL_ENV:-${PEAR_ENVS_DIR:-$WORK_DIR/envs}/verl}" && pwd -P)
LIBERO_ENV=$(cd -- "${PEAR_ENVS_DIR:-$WORK_DIR/envs}/libero" && pwd -P)
LIBERO_MESA_DIR=$(cd -- "${PEAR_ENVS_DIR:-$WORK_DIR/envs}/mesa" && pwd -P)
export VERL_ENV LIBERO_MESA_DIR
export LIBERO_PYTHON=$LIBERO_ENV/bin/python LIBERO_CONFIG_PATH=$LIBERO_ENV/etc/libero
export MODEL=${MODEL:-$WORK_DIR/models/Qwen3.8-27B}
: "${PEAR_CUDA_COMPAT_DIR:?Set the verified cuda-compat-13-0 directory}"
[[ -e "$PEAR_CUDA_COMPAT_DIR/libcuda.so.1" ]] || exit 2
[[ -x "$VERL_ENV/bin/python" && -x "$VERL_ENV/bin/ray" ]] || exit 2
command -v setsid >/dev/null

# A fresh run cannot resume a 4-rank checkpoint into the eventual 64-rank actor.
mkdir -p "$WORK_DIR/runs" "${LOCAL_DIR:-/var/tmp/${USER:-pear}/pear_h20_smoke}"
SMOKE_LOCAL_BASE=${LOCAL_DIR:-/var/tmp/${USER:-pear}/pear_h20_smoke}
export RUN
RUN=$(mktemp -d "$WORK_DIR/runs/grpo_smoke_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
SMOKE_CONTROL=$(mktemp -d "$SMOKE_LOCAL_BASE/grpo_control_XXXXXX")
export LOCAL_DIR=$SMOKE_LOCAL_BASE
# Ray's session/socket suffix cannot fit beneath the longer cache/control path.
# Keep compilation caches unchanged and use a unique owner-only local Ray path.
RAY_TMPDIR=$(mktemp -d /var/tmp/pray.XXXXXX)
export RAY_TMPDIR
chmod 700 "$RAY_TMPDIR"
python3 - "$RAY_TMPDIR" <<'PY'
import os, stat, sys
path = sys.argv[1]
info = os.stat(path)
assert info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700
bound = path + "/session_9999-12-31_23-59-59_999999_2147483647/sockets/plasma_store.9999"
size = len(bound.encode())
assert size <= 107, f"Ray AF_UNIX socket path bound {size} exceeds 107 bytes"
print(f"RAY_SOCKET_PATH_BOUND bytes={size} limit=107 temp_dir={path}", flush=True)
PY
export RAY_ADDRESS=127.0.0.1:23879
# Ray 2.58 otherwise rewrites a loopback --node-ip-address to a routable IP.
# This process-scoped upstream switch keeps its single-node address on loopback,
# including on Linux; no worker node joins this smoke cluster.
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0
export LD_LIBRARY_PATH=$PEAR_CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_ARCH=9.0
export GPU=h100 TRAINER_NNODES=1 ROLLOUT_NNODES=1
export BATCH=4 N=2 STEPS=1 EPOCHS=1
export MAX_PROMPT=4096 MAX_RESPONSE=12288 SESSIONS_PER_REPLICA=4
export LOGGER='["console","file"]'
export PEAR_LOG_DIR=$RUN/sim PEAR_LOG_TURNS=1
export HF_HUB_OFFLINE=1
export NCCL_LAUNCH_ORDER_IMPLICIT=1 NCCL_MNNVL_ENABLE=0
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset RAY_AUTH_MODE RAY_AUTH_TOKEN RAY_AUTH_TOKEN_PATH
export no_proxy=127.0.0.1,localhost,.byted.org NO_PROXY=127.0.0.1,localhost,.byted.org
SMOKE_TIMEOUT_S=${SMOKE_TIMEOUT_S:-10800}
SMOKE_RAY_TIMEOUT_S=${SMOKE_RAY_TIMEOUT_S:-300}
SMOKE_NUM_CPUS=${SMOKE_NUM_CPUS:-120}
export SMOKE_MIN_FREE_GIB=${SMOKE_MIN_FREE_GIB:-400}
[[ "$SMOKE_TIMEOUT_S" =~ ^[1-9][0-9]*$ && "$SMOKE_RAY_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$SMOKE_NUM_CPUS" =~ ^[1-9][0-9]*$ ]] || exit 2
source "$PEAR_REPO/scripts/gpu_process_variables.sh" "$VERL_ENV"
cd "$PEAR_REPO"

RAY_PID= TRAIN_PID=
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  # Only descendants/session members of the two session leaders this script starts.
  # Ray workers can start their own process groups: collect the tree before stopping it.
  "$VERL_ENV/bin/python" - "$RAY_PID" "$TRAIN_PID" <<'PY' || true
import os, sys, time
import psutil
roots = [int(p) for p in sys.argv[1:] if p]
owned = {}
for pid in roots:
    try:
        p = psutil.Process(pid)
        owned[p.pid] = p
        for child in p.children(recursive=True):
            owned[child.pid] = child
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
for p in psutil.process_iter():
    try:
        if os.getsid(p.pid) in roots:
            owned[p.pid] = p
    except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
        pass
owned.pop(os.getpid(), None)

def is_alive(p):
    # Keep the saved Process object: is_running/terminate/kill check PID reuse.
    try:
        return p.is_running() and p.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # conservatively retain it; signalling still checks identity

def wait_owned(processes, timeout):
    # This container rejects pidfd_open with EINVAL; avoid psutil.wait_procs/wait.
    deadline = time.monotonic() + timeout
    alive = list(processes)
    while alive:
        alive = [p for p in alive if is_alive(p)]
        remaining = deadline - time.monotonic()
        if not alive or remaining <= 0:
            break
        time.sleep(min(0.2, remaining))
    return alive

for p in owned.values():
    try:
        if is_alive(p): p.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
alive = wait_owned(owned.values(), timeout=8)
for p in alive:
    try:
        if is_alive(p): p.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
alive = wait_owned(alive, timeout=3)
if alive:
    print('GRPO cleanup: owned processes remain after bounded TERM/KILL:', [p.pid for p in alive], file=sys.stderr)
PY
  printf '%s\n' "$status" > "$RUN/exit_status.txt"
  printf 'GRPO smoke exit=%s; run=%s\n' "$status" "$RUN"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python - <<'PY' > "$RUN/preflight.json"
import json, os, pathlib, shutil, socket
import torch
model = pathlib.Path(os.environ['MODEL'])
record = (model / '.pear_model').read_text().splitlines()
assert record[:2] == [
    'repository Qwen/Qwen3.8-27B',
    'revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
], 'The prepared model record is not the pinned official base model'
for line in record[2:]:
    size, name = line.split(maxsplit=1)
    path = model / name
    assert path.is_file() and path.stat().st_size == int(size), f'Model file mismatch: {name}'
index = json.loads((model / 'model.safetensors.index.json').read_text())
shards = set(index['weight_map'].values())
assert len(shards) == 18 and all((model / name).is_file() for name in shards)
assert torch.cuda.device_count() == 8, 'This smoke requires exactly 8 visible GPUs'
devices = []
for i in range(8):
    p = torch.cuda.get_device_properties(i)
    assert (p.major, p.minor) == (9, 0), (i, p.name, p.major, p.minor)
    devices.append({'index': i, 'name': p.name, 'memory_gib': p.total_memory / 2**30})
# Refuse an occupied head port; never stop an existing cluster to claim it.
with socket.socket() as s:
    s.bind(('127.0.0.1', 23879))
free = shutil.disk_usage(os.environ['RUN']).free
assert free >= int(os.environ['SMOKE_MIN_FREE_GIB']) * 2**30, 'Insufficient free space for a full model+optimizer checkpoint'
print(json.dumps({'devices': devices, 'model': str(model), 'revision': record[1],
                  'free_checkpoint_gib': free / 2**30, 'ray_address': os.environ['RAY_ADDRESS']}, indent=2))
PY
bash 3rdparty/verl/scripts/checkout_verl.sh --check > "$RUN/verl_check.txt"

# One Ray node offers both role labels. Each placement-group bundle reserves one
# actual GPU; the role labels are only placement selectors, not GPU partitions.
# 4 training bundles + 4 rollout bundles therefore occupy disjoint GPUs.
setsid env PYTHONPATH="$PEAR_REPO/pear/utils/sitecustomize" \
  "$VERL_ENV/bin/ray" start --head --block --disable-usage-stats \
  --node-ip-address=127.0.0.1 --port=23879 --include-dashboard=false \
  --ray-client-server-port=23882 \
  --min-worker-port=24000 --max-worker-port=24999 \
  --num-cpus="$SMOKE_NUM_CPUS" --num-gpus=8 \
  --resources='{"trainer":4,"rollout":4,"libero":4}' \
  --object-store-memory=8589934592 --temp-dir="$RAY_TMPDIR" \
  > "$RUN/ray.log" 2>&1 &
RAY_PID=$!
start=$SECONDS
until "$VERL_ENV/bin/ray" status --address "$RAY_ADDRESS" > "$RUN/ray_status.txt" 2>&1; do
  kill -0 "$RAY_PID" 2>/dev/null || { echo "Smoke Ray exited; see $RUN/ray.log" >&2; exit 1; }
  (( SECONDS - start < SMOKE_RAY_TIMEOUT_S )) || { echo 'Smoke Ray readiness timeout' >&2; exit 1; }
  sleep 3
done
python - <<'PY' > "$RUN/cluster.json"
import json, os, ray
ray.init(address=os.environ['RAY_ADDRESS'], log_to_driver=False)
nodes = [n for n in ray.nodes() if n['Alive']]
r = ray.cluster_resources()
assert len(nodes) == 1, nodes
assert nodes[0]['NodeManagerAddress'] == '127.0.0.1', nodes[0]['NodeManagerAddress']
assert all(r.get(k) == v for k, v in {'GPU':8, 'trainer':4, 'rollout':4, 'libero':4}.items()), r
print(json.dumps({'alive_nodes':len(nodes), 'resources':r}, indent=2))
ray.shutdown()
PY

# Keep the real LIBERO success reward, GRPO, score centering, TP4 Hopper kernels,
# activation offload, and streamed AdamW. Only the smoke sizes and group filter
# change. Both LMCache tiers use 32 GB, above LMCache's 20-GB initial allocation,
# so the sidecar emits the completed-growth event PEAR waits for during startup.
# SP4 gives DP=4/4=1; batch=4*2=8 is divisible by lcm(DP, PPO mini-batch * N)=8.
# Four sessions -> eight decode slots -> one agent worker (the default divisor is 8).
setsid bash examples/models/qwen3_8_27b_pear/train/run_stage1.sh \
  trainer.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=4 \
  actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=false \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  algorithm.filter_groups.enable=false \
  actor_rollout_ref.rollout.custom.lmcache.l1_size_gb=32 \
  transfer_queue.backend.SimpleStorage.num_data_storage_units=4 \
  trainer.critic_warmup=0 trainer.save_freq=1 trainer.test_freq=-1 \
  trainer.val_before_train=false trainer.resume_mode=disable \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' \
  "+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.PEAR_CUDA_COMPAT_DIR=$PEAR_CUDA_COMPAT_DIR" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.PEAR_LOG_TURNS="1"' \
  "++ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_PYTHON=$LIBERO_PYTHON" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_MESA_DIR=$LIBERO_MESA_DIR" \
  > "$RUN/launcher.log" 2>&1 &
TRAIN_PID=$!
start=$SECONDS
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  kill -0 "$RAY_PID" 2>/dev/null || { echo 'Smoke Ray died during training' >&2; exit 1; }
  (( SECONDS - start < SMOKE_TIMEOUT_S )) || { echo "GRPO smoke timeout; see $RUN/launcher.log" >&2; exit 1; }
  sleep 5
done
wait "$TRAIN_PID" || { echo "GRPO smoke training failed; see $RUN/launcher.log" >&2; exit 1; }

# Training's final step includes the NCCL update_weights call after checkpointing.
# Query the still-running isolated cluster before cleanup removes its detached actors.
python - <<'PY' > "$RUN/rollout_summary.json"
import json, os, ray
ray.init(address=os.environ['RAY_ADDRESS'], log_to_driver=False)
actor = ray.get_actor('pear_progress', namespace='pear')
summary = ray.get(actor.stats.remote(2))
print(json.dumps(summary, indent=2))
ray.shutdown()
PY

python - <<'PY'
import json, math, os
from pathlib import Path
run = Path(os.environ['RUN'])
def lines(paths):
    for p in paths:
        for line in p.read_text().splitlines():
            if line.strip():
                yield json.loads(line)
metrics = [r for r in lines(sorted(run.glob('metrics_*.jsonl'))) if r.get('step') == 1]
assert metrics, 'No completed step-1 metrics'
m = metrics[-1]['data']
assert 'actor/grad_norm' in m and math.isfinite(float(m['actor/grad_norm'])), 'Missing/nonfinite backward gradient norm'
loss_keys = [k for k in m if k.startswith('actor/') and 'loss' in k]
assert loss_keys and all(math.isfinite(float(m[k])) for k in loss_keys), 'Missing/nonfinite actor loss'
for name in ('update_actor', 'update_weights'):
    key = f'timing_s/{name}'
    assert key in m and math.isfinite(float(m[key])) and float(m[key]) > 0, f'Missing completed {name} timing'
ckpt = run / 'ckpt'
assert (ckpt / 'latest_checkpointed_iteration.txt').read_text().strip() == '1'
actor = ckpt / 'global_step_1' / 'actor'
files = []
for rank in range(4):
    for kind in ('model', 'optim', 'extra_state'):
        p = actor / f'{kind}_world_size_4_rank_{rank}.pt'
        assert p.is_file() and p.stat().st_size > 0, f'Missing checkpoint shard: {p.name}'
        files.append({'name':p.name, 'bytes':p.stat().st_size})
summary = json.loads((run / 'rollout_summary.json').read_text())
assert summary['finished'] >= 8 and summary['judged'] >= 4, 'Fewer than four real two-trajectory groups completed'
turns = list(lines(sorted((run / 'sim').glob('turns_*.jsonl'))))
assert any(r.get('text', '').strip() for r in turns), 'No model-generated rollout text'
actions = list(lines(sorted((run / 'sim').glob('actions_*.jsonl'))))
executed = [r for r in actions if r.get('action') is not None and int(r.get('executed', 0)) > 0]
assert executed, 'Model loaded/trained but no generated action executed in LIBERO; interaction smoke incomplete'
result = {
    'passed':True, 'completed_training_steps':1, 'trainer_gpus':4, 'rollout_gpus':4,
    'batch_groups':4, 'trajectories_per_group':2, 'context_tokens':16384,
    'dapo_filter_enabled':False, 'reward_source':'unchanged LIBERO binary task success',
    'rollout_summary':summary, 'executed_tool_calls':len(executed),
    'actor_grad_norm':m['actor/grad_norm'], 'losses':{k:m[k] for k in loss_keys},
    'update_timings':{name:m[f'timing_s/{name}'] for name in ('update_actor','update_weights')},
    'checkpoint_files':files, 'checkpoint_bytes':sum(p['bytes'] for p in files),
    'claim':'Real model load, generated LIBERO interaction, backward/optimizer step, weight sync, and checkpoint completed. Zero-variance rewards do not demonstrate learning effectiveness.',
}
(run / 'smoke_result.json').write_text(json.dumps(result, indent=2) + '\n')
(Path(os.environ['PEAR_ROOT']) / 'preflight/logs/grpo_result_path.txt').write_text(str(run / 'smoke_result.json') + '\n')
print(json.dumps(result, indent=2))
PY
