#!/usr/bin/env bash
# Restore one completed CPU-built artifact to the exact prefix used by its shebangs/RPATHs.
set -euo pipefail
PEAR_ROOT=${PEAR_ROOT:-/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114}
PREFIX=/opt/tiger/pear_runtime_20261114
READY=$PEAR_ROOT/runtime_bundles/READY
[[ -s "$READY" ]] || { echo 'Runtime is not published: READY missing' >&2; exit 1; }
read -r BUNDLE SHA SIZE < <(python3 - "$READY" "$PREFIX" <<'PY'
import json,re,sys
r=json.load(open(sys.argv[1]))
assert re.fullmatch(r'runtime-[a-zA-Z0-9_-]+',r['bundle']) and r['archive']=='runtime.tar.gz'
assert re.fullmatch(r'[a-f0-9]{64}',r['sha256']) and r['prefix']==sys.argv[2]
print(r['bundle'],r['sha256'],r['size'])
PY
)
[[ -n "${BUNDLE:-}" && -n "${SHA:-}" ]] || exit 1
check_aliases() {
  for env in verl libero mesa numactl; do
    [[ -L "$PEAR_ROOT/work_dir/envs/$env" && $(readlink "$PEAR_ROOT/work_dir/envs/$env") == "$PREFIX/envs/$env" ]] || { echo "Shared alias differs: $env"; return 1; }
  done
}
check_runtime() {
  check_aliases || return 1
  [[ -f "$PREFIX/runtime_bundle.json" && -x "$PREFIX/envs/verl/bin/python" ]] || return 1
  "$PREFIX/envs/verl/bin/python" -c 'import platform; print("Runtime restored; Python",platform.python_version())'
}
if [[ -f "$PREFIX/.restored_bundle" ]]; then
  [[ $(cat "$PREFIX/.restored_bundle") == "$SHA" ]] || { echo 'Different runtime already present; refusing in-place replacement'; exit 1; }
  check_runtime
  exit 0
fi
if [[ -e "$PREFIX" ]] && [[ -n $(ls -A "$PREFIX") ]]; then
  echo "Nonempty unverified prefix $PREFIX; refusing to overwrite" >&2; exit 1
fi
LOCK=/opt/tiger/.pear_runtime_20261114.restore.lock.d
mkdir "$LOCK" || { echo 'Another restore owns the local lock'; exit 1; }
TEMP=$(mktemp -d /opt/tiger/pear_runtime_restore.XXXXXX)
trap 's=$?; rm -rf "$TEMP"; rmdir "$LOCK" 2>/dev/null || true; exit "$s"' EXIT
SOURCE=$PEAR_ROOT/runtime_bundles/$BUNDLE
[[ -s "$SOURCE/READY" && -s "$SOURCE/SHA256SUMS" ]] || { echo 'Incomplete runtime bundle'; exit 1; }
cp "$SOURCE/runtime.tar.gz" "$TEMP/runtime.tar.gz"
printf '%s  %s\n' "$SHA" "$TEMP/runtime.tar.gz" | sha256sum --check -
[[ $(stat -c %s "$TEMP/runtime.tar.gz") == "$SIZE" ]] || exit 1
mkdir "$TEMP/extracted"
tar -xzf "$TEMP/runtime.tar.gz" -C "$TEMP/extracted"
[[ -f "$TEMP/extracted/.pear_local_runtime" ]] || exit 1
for env in verl libero mesa numactl; do [[ -d "$TEMP/extracted/envs/$env" ]] || exit 1; done
# Only an empty directory may be removed; an existing environment is never overwritten.
if [[ -d "$PREFIX" ]]; then rmdir "$PREFIX"; fi
mv "$TEMP/extracted" "$PREFIX"
cp "$SOURCE/READY" "$PREFIX/runtime_bundle.json"
mkdir -p "$PREFIX/cache/tmp" "$PREFIX/cache/runtime" "$PREFIX/cache/environment_test"
check_runtime
printf '%s\n' "$SHA" > "$PREFIX/.restored_bundle.partial.$$"
mv "$PREFIX/.restored_bundle.partial.$$" "$PREFIX/.restored_bundle"
