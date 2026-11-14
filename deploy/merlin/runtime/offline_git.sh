# Source in this preparation process only. Does not modify global Git configuration.
export PEAR_ROOT=${PEAR_ROOT:-/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114}
PEAR_GIT_MIRRORS=$PEAR_ROOT/offline_git_mirrors
[[ -s "$PEAR_GIT_MIRRORS/OFFLINE_READY" ]] || { echo 'Verified offline Git mirrors are not ready' >&2; return 1; }
PEAR_GIT_CONFIG_FILE=$(mktemp "${TMPDIR:-/tmp}/pear_offline_git.XXXXXX")
if ! python3 - "$PEAR_GIT_MIRRORS" > "$PEAR_GIT_CONFIG_FILE" <<'PY'
import json,pathlib,shlex,subprocess,sys
root=pathlib.Path(sys.argv[1]);ready=json.loads((root/'OFFLINE_READY').read_text())
assert len(ready['repositories'])==6
for entry in ready['repositories']:
    repo=root/(entry['name']+'.git')
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','refs/heads/pinned'],text=True).strip()
    assert commit==entry['commit'],f"Mirror commit mismatch: {entry['name']}"
    print('pear_git_option '+shlex.quote('url.'+repo.as_uri()+'.insteadOf')+' '+shlex.quote(entry['url']))
PY
then rm -f "$PEAR_GIT_CONFIG_FILE"; return 1; fi
pear_git_option() {
  local index=${GIT_CONFIG_COUNT:-0}
  export "GIT_CONFIG_KEY_$index=$1" "GIT_CONFIG_VALUE_$index=$2"
  export GIT_CONFIG_COUNT=$((index + 1))
}
# Submodule update permits the same explicitly verified local-file transport.
pear_git_option protocol.file.allow always
source "$PEAR_GIT_CONFIG_FILE"
rm -f "$PEAR_GIT_CONFIG_FILE"
unset -f pear_git_option
unset PEAR_GIT_CONFIG_FILE
export GIT_TERMINAL_PROMPT=0
printf 'OFFLINE_GIT_ENABLED %s\n' "$PEAR_GIT_MIRRORS"
