#!/usr/bin/env bash
# Invoke only on the selected GPU worker pods; leave the separate CPU head idle.
# GPU worker rank 0 is the Ray head. No Robust restart.
set -euo pipefail
[[ ${PEAR_SUPERVISED:-} == 1 ]] || { echo "Use merlin_stage1_entry.sh, not this internal worker entry" >&2; exit 2; }
# The reused idle task carries NERO settings unrelated to this PEAR run.
unset PYTHONPATH EXPECTED_GIT_COMMIT
umask 077

ROOT=${PEAR_ROOT:?The owning controller must set PEAR_ROOT}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
TOPOLOGY_ENV=$(python3 "$SCRIPT_DIR/topology.py" --shell)
eval "$TOPOLOGY_ENV"
RANK=${ARNOLD_ID:?ARNOLD_ID must identify this worker}
TRIAL_ID=${ARNOLD_TRIAL_ID:?ARNOLD_TRIAL_ID is required}
[[ ${ARNOLD_ROLE:-} == worker ]] || { echo 'Expected ARNOLD_ROLE=worker' >&2; exit 2; }
[[ "$RANK" =~ ^[0-9]+$ ]] && (( 10#$RANK < PEAR_NNODES )) || { echo "ARNOLD_ID must be 0..$((PEAR_NNODES - 1))" >&2; exit 2; }
[[ "$TRIAL_ID" =~ ^[0-9]+$ ]] || { echo 'ARNOLD_TRIAL_ID must be numeric' >&2; exit 2; }
RANK=$((10#$RANK))
export PEAR_ROOT=$ROOT
source "$SCRIPT_DIR/network_env.sh"
LAUNCH_ID=${PEAR_LAUNCH_ID:-merlin-trial-${TRIAL_ID}-stage1-${PEAR_TOTAL_GPUS}}
[[ "$LAUNCH_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'PEAR_LAUNCH_ID must contain only letters, digits, underscore or hyphen' >&2; exit 2; }
CONTROL=$ROOT/launches/$LAUNCH_ID
SHARED_CONTROL=$CONTROL/sharedcontrol
HEAD_FILE=$SHARED_CONTROL/head_address
TOKEN_FILE=$SHARED_CONTROL/ray.token
TIMEOUT=1200

mkdir -p "$SHARED_CONTROL"
chmod 700 "$SHARED_CONTROL"
fail() {
  printf '%s rank=%s %s\n' "$(date -u +%FT%TZ)" "$RANK" "$*" > "$CONTROL/failure.$RANK"
  echo "$*" >&2
  exit 1
}

# Restores the prepared runtime to /opt/tiger/pear_runtime_20261114 on this node.
# No package installation or shared-cluster cleanup belongs in this wrapper.
BASELINE_LAUNCH=$ROOT/launch
[[ -f "$BASELINE_LAUNCH/restore_runtime.sh" ]] || fail 'Frozen restore_runtime.sh is missing'
PEAR_ROOT=$ROOT bash "$BASELINE_LAUNCH/restore_runtime.sh" || fail 'Local runtime restoration failed'
# Resolve the venv directory only; bin/python points into managed base Python.
VERL_ENV=$(cd -- "${VERL_ENV:-${PEAR_ENVS_DIR:-$ROOT/work_dir/envs}/verl}" && pwd -P)
export VERL_ENV
PYTHON=$VERL_ENV/bin/python
[[ -x "$PYTHON" ]] || fail 'Prepared training Python is unavailable after runtime restoration'

if (( RANK == 0 )); then
  # A trial is a single launch. Refuse a second publisher rather than rotating a
  # token underneath live workers. A new trial gets a new control directory.
  mkdir "$SHARED_CONTROL/publisher.claim" 2>/dev/null || { echo "Head publisher already exists; refusing duplicate" >&2; exit 75; }
  "$PYTHON" - "$HEAD_FILE" "$TOKEN_FILE" <<'PY' || fail 'Head address/token publication failed'
import ipaddress
import os
from pathlib import Path
import secrets
import sys


head_path, token_path = map(Path, sys.argv[1:])
address = ipaddress.ip_address(os.environ["PEAR_NODE_IP"])
if address.version != 4 or address.is_loopback or address.is_unspecified or address.is_multicast:
    raise SystemExit("The selected worker interface must provide a non-loopback IPv4 address")
if head_path.exists() or token_path.exists():
    raise SystemExit("Trial head/token already exists; a new trial is required")

def publish(path, text):
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

# Publish the address last: it is the readiness marker for both complete files.
publish(token_path, secrets.token_urlsafe(48) + "\n")
publish(head_path, f"{address}:6379\n")
print(f"PEAR_HEAD_PUBLISHED {address}:6379", flush=True)
PY
else
  start=$SECONDS
  until [[ -s "$HEAD_FILE" && -s "$TOKEN_FILE" ]]; do
    failures=("$CONTROL"/failure.*)
    [[ ! -e "${failures[0]}" ]] || fail 'Another node failed during runtime/head preparation'
    (( SECONDS - start < TIMEOUT )) || fail 'Timed out waiting for rank 0 head address/token'
    sleep 3
  done
fi

# Do not log the token. Every worker validates the same atomically published pair.
HEAD_ADDRESS=$("$PYTHON" - "$HEAD_FILE" "$TOKEN_FILE" <<'PY'
import ipaddress
import os
from pathlib import Path
import stat
import sys

head_path, token_path = map(Path, sys.argv[1:])
for path in (head_path, token_path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise SystemExit(f"Control file must be a regular file owned by this user: {path.name}")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise SystemExit(f"Control file must have mode 600: {path.name}")
address, port = head_path.read_text().strip().rsplit(":", 1)
ip = ipaddress.ip_address(address)
if ip.version != 4 or ip.is_loopback or ip.is_unspecified or ip.is_multicast or port != "6379":
    raise SystemExit("Invalid published Ray head address")
if not token_path.read_text().strip():
    raise SystemExit("Empty Ray authentication token")
print(f"{ip}:{port}")
PY
) || fail 'Published head address/token validation failed'

printf 'PEAR_STAGE1_WORKER gpus=%s nodes=%s rank=%s trial=%s head=%s\n' "$PEAR_TOTAL_GPUS" "$PEAR_NNODES" "$RANK" "$TRIAL_ID" "$HEAD_ADDRESS"
exec bash "$SCRIPT_DIR/stage1_node.sh" \
  --root "$ROOT" \
  --rank "$RANK" \
  --head-address "$HEAD_ADDRESS" \
  --launch-id "$LAUNCH_ID" \
  --token-file "$TOKEN_FILE"
