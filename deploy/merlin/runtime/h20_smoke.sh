#!/usr/bin/env bash
# Run only inside the user's dedicated H20 preflight allocation, after environment preparation.
set -euo pipefail
# A CUDA polling kernel or a spawned TP rank can hang below pytest's Python
# stack. Bound the complete probe/test process group and preserve its real status.
if [[ ${_PEAR_H20_SMOKE_BOUNDED:-0} != 1 ]]; then
  H20_SMOKE_TIMEOUT_S=${H20_SMOKE_TIMEOUT_S:-7200}
  H20_SMOKE_KILL_AFTER_S=${H20_SMOKE_KILL_AFTER_S:-60}
  [[ "$H20_SMOKE_TIMEOUT_S" =~ ^[1-9][0-9]*$ && "$H20_SMOKE_KILL_AFTER_S" =~ ^[1-9][0-9]*$ ]] || exit 2
  command -v timeout >/dev/null || { echo 'GNU timeout is required for bounded GPU tests' >&2; exit 2; }
  printf 'H20_SMOKE_BEGIN timeout_seconds=%s kill_after_seconds=%s time=%s\n' \
    "$H20_SMOKE_TIMEOUT_S" "$H20_SMOKE_KILL_AFTER_S" "$(date -Is)"
  if timeout --signal=TERM --kill-after="${H20_SMOKE_KILL_AFTER_S}s" "${H20_SMOKE_TIMEOUT_S}s" \
      env _PEAR_H20_SMOKE_BOUNDED=1 PYTHONUNBUFFERED=1 bash "$0" "$@"; then
    smoke_status=0
  else
    smoke_status=$?
  fi
  printf 'H20_SMOKE_END exit_code=%s time=%s\n' "$smoke_status" "$(date -Is)"
  if (( smoke_status == 124 || smoke_status == 137 )); then
    echo 'H20 GPU checks exceeded their wall-clock limit; retain this log and inspect the last named pytest case' >&2
  fi
  exit "$smoke_status"
fi
PEAR_ROOT=${PEAR_ROOT:-/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114}
PEAR_REPO=${PEAR_REPO:-$PEAR_ROOT/repo}
export WORK_DIR=${WORK_DIR:-$PEAR_ROOT/work_dir}
# Resolve the venv directory, not bin/python (which links to managed base Python).
# Starting through the NAS alias makes sys.prefix and package scans use NAS paths.
VERL_ENV=$(cd -- "${VERL_ENV:-${PEAR_ENVS_DIR:-$WORK_DIR/envs}/verl}" && pwd -P)
export VERL_ENV
export LOCAL_DIR=${LOCAL_DIR:-/var/tmp/${USER:-pear}/pear_h20_smoke}
: "${PEAR_CUDA_COMPAT_DIR:?Set the verified extracted cuda-compat-13-0 directory}"
export LD_LIBRARY_PATH=$PEAR_CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_ARCH=9.0
mkdir -p "$LOCAL_DIR" "$PEAR_ROOT/preparation_logs"
source "$PEAR_REPO/scripts/gpu_process_variables.sh" "$VERL_ENV"
cd "$PEAR_REPO"
python - <<'PY'
import ctypes, json, os
import torch
lib=ctypes.CDLL('libcuda.so.1')
version=ctypes.c_int()
assert lib.cuDriverGetVersion(ctypes.byref(version)) == 0
assert version.value >= 13000, version.value
assert lib.cuInit(0) == 0, 'CUDA compatibility initialization failed'
maps=[line for line in open('/proc/self/maps') if '/libcuda.so' in line]
assert any(os.environ['PEAR_CUDA_COMPAT_DIR'] in line for line in maps), 'Process did not load the requested compatibility libcuda'
assert torch.cuda.is_available() and torch.cuda.device_count() >= 4
props=[]
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    assert (p.major,p.minor)==(9,0), (i,p.name,p.major,p.minor)
    with torch.cuda.device(i):
        x=torch.randn(512,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        (x@x).float().square().mean().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(x.grad).all()
    props.append({'device':i,'name':p.name,'sms':p.multi_processor_count,'memory_gib':round(p.total_memory/2**30,2)})
print(json.dumps({'driver_api':version.value,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'devices':props,'loaded_libcuda':maps},indent=2))
PY
# Covers CUDA dependencies, Hopper custom kernels, TP4 collectives, and distributed training attention.
# Use the upstream correctness checks; preserve original H100 CTA constants for the first run.
python -m pytest -vv -ra --durations=20 \
  --junitxml="$PEAR_ROOT/preflight/logs/upstream_gpu_suite.xml" \
  tests/test_training_environment_on_gpu.py \
  tests/models/test_qwen3_5_rollout_kernels_hopper_on_gpu.py \
  tests/models/test_qwen3_5_training_kernels_on_gpu.py \
  tests/models/test_qwen3_5_training_kernels_ulysses_on_gpu.py \
  tests/models/test_qwen3_5_training_kernels_head_on_gpu.py
