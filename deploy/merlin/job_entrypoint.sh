#!/usr/bin/env bash
# Platform entry: one automatic launch, then keep the allocation for inspection.
set -uo pipefail
umask 077
CHILD=
stop() {
  trap - INT TERM
  if [[ -n "$CHILD" ]]; then
    kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
  fi
  exit 143
}
trap stop INT TERM

ROOT=${PEAR_ROOT:?PEAR_ROOT is required}
RELEASE=$(cd "$(dirname "$0")/../.." && pwd -P) || exit 2
# Resolve in the image before activating the training venv. Different images
# place this executable under /usr/bin or /usr/local/bin; stale job overrides
# must not prevent release verification or writing the failure status.
REQUESTED_PYTHON=${PEAR_TRACKING_PYTHON:-python3}
if ! IMAGE_PYTHON=$(command -v "$REQUESTED_PYTHON") || [[ ! -x "$IMAGE_PYTHON" ]]; then
  echo "Unavailable image Python $REQUESTED_PYTHON; resolving python3 from PATH" >&2
  IMAGE_PYTHON=$(command -v python3) || exit 127
fi
"$IMAGE_PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' || exit 127
export PEAR_TRACKING_PYTHON=$IMAGE_PYTHON
export PEAR_TOTAL_GPUS=${PEAR_TOTAL_GPUS:-128}
export PEAR_LAUNCH_ID=${PEAR_LAUNCH_ID:-merlin-trial-${ARNOLD_TRIAL_ID:?}-stage1-${PEAR_TOTAL_GPUS:-128}-v1}
export PEAR_RUN=${PEAR_RUN:-$ROOT/work_dir/runs/$PEAR_LAUNCH_ID}
ENTRY_DIR=$ROOT/entrypoints/$PEAR_LAUNCH_ID/rank-${ARNOLD_ID:?}-$(date -u +%Y%m%dT%H%M%S)-$$
mkdir -p "$ENTRY_DIR" || exit 2
exec > >(tee -a "$ENTRY_DIR/entrypoint.log") 2>&1
printf 'PEAR_AUTOSTART launch=%s rank=%s commit=%s run=%s\n' \
  "$PEAR_LAUNCH_ID" "$ARNOLD_ID" "${PEAR_DEPLOY_COMMIT:-missing}" "$PEAR_RUN"

launch() {
  # The publisher verifies the archive before an atomic NAS rename. Every worker
  # independently verifies all published files before executing the launcher.
  "$PEAR_TRACKING_PYTHON" - "$RELEASE" <<'PY' || return
import hashlib, json, os, pathlib, re, sys
release = pathlib.Path(sys.argv[1])
commit = os.environ.get('PEAR_DEPLOY_COMMIT', '')
if not re.fullmatch(r'[a-f0-9]{40}', commit):
    raise SystemExit('A fixed Git commit is required')
manifest = json.loads((release / 'release.json').read_text())
if manifest.get('git_commit') != commit:
    raise SystemExit('Release commit does not match job configuration')
required = {'deploy/merlin/job_entrypoint.sh', 'deploy/merlin/merlin_stage1_entry.sh'}
if not isinstance(manifest.get('files'), dict) or not required.issubset(manifest['files']):
    raise SystemExit('Release manifest omits an entry point')
for name, expected in manifest['files'].items():
    path = release / name
    if pathlib.Path(name).is_absolute() or '..' in pathlib.Path(name).parts:
        raise SystemExit('Invalid path in release manifest')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Release differs: {name}')
print('PEAR_RELEASE_VERIFIED', commit, 'files=', len(manifest['files']), flush=True)
PY
  bash "$RELEASE/deploy/merlin/merlin_stage1_entry.sh" &
  CHILD=$!
  wait "$CHILD"
  local status=$?
  CHILD=
  return "$status"
}

launch
STATUS=$?
"$PEAR_TRACKING_PYTHON" - "$ENTRY_DIR/status.json" "$STATUS" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
status = int(sys.argv[2])
record = {'state': 'completed_keepalive' if status == 0 else 'failed_keepalive',
          'training_exit_code': status, 'time': time.time(),
          'launch_id': os.environ['PEAR_LAUNCH_ID'], 'rank': os.environ['ARNOLD_ID'],
          'run': os.environ['PEAR_RUN'], 'commit': os.environ.get('PEAR_DEPLOY_COMMIT'),
          'automatic_retry': False}
temporary = path.with_suffix('.partial')
temporary.write_text(json.dumps(record, indent=2) + '\n')
temporary.replace(path)
PY
printf 'PEAR_TRAINING_EXIT status=%s; entering sleep infinity; status_record=%s/status.json\n' "$STATUS" "$ENTRY_DIR"
# A RUN lock retained by the controller must be inspected, never deleted here.
# Repair explicitly with a new Git release and launch ID; no automatic restart.
trap - INT TERM
exec sleep infinity
