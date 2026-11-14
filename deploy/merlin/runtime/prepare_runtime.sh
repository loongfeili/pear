#!/usr/bin/env bash
# CPU-only task: prepare the full CUDA/Hopper runtime locally, then atomically publish it.
set -euo pipefail
LAUNCH=$(cd "$(dirname "$0")" && pwd)
source "$LAUNCH/runtime_paths.sh"
LOG=$PEAR_ROOT/runtime_build/logs
mkdir -p "$LOG" "$PEAR_ROOT/runtime_bundles"
exec > >(tee -a "$LOG/prepare.log") 2>&1
LOCK=$PEAR_ROOT/runtime_build/preparing.lock.d
mkdir "$LOCK" || { echo 'Another preparation owns the directory lock; inspect before retrying' >&2; exit 1; }
LIBERO_PID= VERL_PID=
cleanup() {
  s=$?
  trap - EXIT INT TERM
  for p in "$LIBERO_PID" "$VERL_PID"; do [[ -z "$p" ]] || kill "$p" 2>/dev/null || true; done
  if (( s )); then printf '%s\n' "$s" > "$LOG/FAILED"; fi
  rmdir "$LOCK" 2>/dev/null || true
  exit "$s"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
rm -f "$LOG/FAILED"
printf 'PREPARE_RUNTIME_START %s\n' "$(date -Is)"
# Publication of the previous successful build remains untouched until the new build is complete.
[[ ! -f "$PEAR_RUNTIME/.restored_bundle" ]] || { echo 'Refusing to rebuild a restored runtime in place'; exit 1; }
printf 'pear-local-runtime-20261114\n' > "$PEAR_RUNTIME/.pear_local_runtime"
bash "$LAUNCH/fetch_cuda_compat.sh"
cd "$PEAR_ROOT/repo"
for patch in cuda_compat_install.patch local_runtime.patch; do
  if git apply --reverse --check "$LAUNCH/$patch" >/dev/null 2>&1; then
    printf 'PATCH_ALREADY_APPLIED %s\n' "$patch"
  elif git apply --check "$LAUNCH/$patch" >/dev/null 2>&1; then
    git apply "$LAUNCH/$patch"
    printf 'PATCH_APPLIED %s\n' "$patch"
  else
    git apply --check "$LAUNCH/$patch" >&2 || true
    echo "Unexpected source state for $patch" >&2; exit 1
  fi
done
python3 "$LAUNCH/preflight_stage1.py" --root "$PEAR_ROOT" --check-source-only
bash "$LAUNCH/install_offline_git.sh"
source "$LAUNCH/offline_git.sh"
# Reuse this task's previously installed tools; Python keeps its upstream pins.
EXPECTED_UV_VERSION=$(sed -n 's/^UV_VERSION=//p' scripts/install/versions.env)
if [[ ! -x "$PEAR_BIN_DIR/uv" && -x "$WORK_DIR/bin/uv" && -x "$WORK_DIR/bin/uvx" ]] &&
   [[ $("$WORK_DIR/bin/uv" --version | awk '{print $2}') == "$EXPECTED_UV_VERSION" ]] &&
   [[ $("$WORK_DIR/bin/uvx" --version | awk '{print $2}') == "$EXPECTED_UV_VERSION" ]]; then
  for tool in uv uvx; do
    cp "$WORK_DIR/bin/$tool" "$PEAR_BIN_DIR/$tool.bootstrap.$$"
    chmod 755 "$PEAR_BIN_DIR/$tool.bootstrap.$$"
    mv "$PEAR_BIN_DIR/$tool.bootstrap.$$" "$PEAR_BIN_DIR/$tool"
  done
  printf 'BOOTSTRAP_UV_FROM_TASK_CACHE %s\n' "$EXPECTED_UV_VERSION"
fi
# Bootstrap uv once; simultaneous environments then use uv's LOCAL filesystem locks.
bash -c 'source scripts/install/common.sh; install_uv'
bash scripts/install.sh libero > "$LOG/libero.log" 2>&1 &
LIBERO_PID=$!
bash scripts/install.sh verl > "$LOG/verl.log" 2>&1 &
VERL_PID=$!
failed=0
wait "$LIBERO_PID" || failed=1
LIBERO_PID=
wait "$VERL_PID" || failed=1
VERL_PID=
(( failed == 0 )) || { echo 'CPU preparation failed; inspect libero.log and verl.log'; exit 1; }
# Always include numactl in the portable bundle even if this CPU image already has it.
bash -c 'source scripts/install/common.sh; install_conda_prefix "$ENVS/numactl" "$INSTALL_DIR/numactl/versions.txt"; "$ENVS/numactl/bin/numactl" --version' > "$LOG/numactl.log" 2>&1
# No NO_GPU=1: CUDA toolkit, FA2/FA3, LMCache and CPU verification must all have completed.
for env in verl libero mesa numactl; do [[ -d "$PEAR_ENVS_DIR/$env" ]] || exit 1; done
for name in flash_attn_2 flash_attn_3 lmcache; do [[ -s "$PEAR_ENVS_DIR/verl/.pear_inputs_$name" ]] || { echo "Build record missing: $name"; exit 1; }; done
python3 "$LAUNCH/preflight_stage1.py" --root "$PEAR_ROOT" --check-source-only > "$PEAR_RUNTIME/source_identity.json"
BUNDLE=runtime-$(date -u +%Y%m%dT%H%M%SZ)-$$
DEST=$PEAR_ROOT/runtime_bundles/$BUNDLE
mkdir "$DEST"
# Environments use copied package files; managed Python lives under envs/uv-python.
# Mutable download/JIT caches are not needed to execute the resulting runtime.
tar -C "$PEAR_RUNTIME" -cf - envs bin .pear_local_runtime source_identity.json | gzip -1 > "$DEST/runtime.tar.gz.partial"
mv "$DEST/runtime.tar.gz.partial" "$DEST/runtime.tar.gz"
( cd "$DEST"; sha256sum runtime.tar.gz > SHA256SUMS.partial; mv SHA256SUMS.partial SHA256SUMS )
python3 - "$DEST" "$PEAR_RUNTIME" <<'PY'
import hashlib,json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); filename='runtime.tar.gz'
digest=(p/'SHA256SUMS').read_text().split()[0]
data={'bundle':p.name,'archive':filename,'sha256':digest,'size':(p/filename).stat().st_size,'prefix':sys.argv[2],
      'source_identity':json.loads((pathlib.Path(sys.argv[2])/'source_identity.json').read_text()),'created_at':time.time()}
(p/'READY.partial').write_text(json.dumps(data,indent=2)+'\n'); (p/'READY.partial').replace(p/'READY')
PY
# Shared aliases always target the identical per-node absolute prefix. Preserve old failed
# NAS environments, but never touch uv-python/cache/uv used by the independent model download.
mkdir -p "$WORK_DIR/envs"
for env in verl libero mesa numactl; do
  target=$PEAR_ENVS_DIR/$env
  link=$WORK_DIR/envs/$env
  if [[ -L "$link" && $(readlink "$link") == "$target" ]]; then continue; fi
  if [[ -e "$link" || -L "$link" ]]; then mv "$link" "$link.before_local_runtime.$BUNDLE"; fi
  ln -s "$target" "$link"
done
cp "$DEST/READY" "$PEAR_ROOT/runtime_bundles/READY.partial"
mv "$PEAR_ROOT/runtime_bundles/READY.partial" "$PEAR_ROOT/runtime_bundles/READY"
rm -f "$LOG/FAILED"
printf 'PREPARE_RUNTIME_PUBLISHED %s\n' "$BUNDLE"
