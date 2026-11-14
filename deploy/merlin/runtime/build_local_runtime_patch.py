"""Generate the reviewed local-prefix patch/manifest from the pinned local upstream checkout."""
from pathlib import Path
import difflib,hashlib,json,subprocess
here=Path(__file__).resolve().parent
repo=here.parent/'pear'
compat_add='''# Opt-in, task-local forward-compatibility libraries, extracted without changing the host driver.
if [ -n "${PEAR_CUDA_COMPAT_DIR:-}" ]; then
  if [ ! -e "$PEAR_CUDA_COMPAT_DIR/libcuda.so.1" ]; then
    echo "missing CUDA compatibility libcuda at $PEAR_CUDA_COMPAT_DIR" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH=$PEAR_CUDA_COMPAT_DIR
fi
'''
paths=['scripts/install/common.sh','scripts/install/verl/install.sh','scripts/install/sim/libero_run/install.sh']
original={p:(repo/p).read_text() for p in paths}
before=dict(original)
before[paths[0]]=before[paths[0]].replace('unset CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV\n','unset CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV\n'+compat_add)
after=dict(before)
s=after[paths[0]]
changes={
'ENVS=$WORK_DIR/envs':'ENVS=${PEAR_ENVS_DIR:-$WORK_DIR/envs}\nINSTALL_BIN=${PEAR_BIN_DIR:-$WORK_DIR/bin}\nINSTALL_CACHE=${PEAR_CACHE_DIR:-$WORK_DIR/cache}\nmkdir -p "$ENVS" "$INSTALL_BIN" "$INSTALL_CACHE"',
'export PATH=$WORK_DIR/bin:$PATH':'export PATH=$INSTALL_BIN:$PATH',
'export UV_PYTHON_INSTALL_DIR=$ENVS/uv-python':'export UV_PYTHON_INSTALL_DIR=${PEAR_PYTHON_INSTALL_DIR:-$ENVS/uv-python}',
'export UV_CACHE_DIR=$WORK_DIR/cache/uv':'export UV_CACHE_DIR=${PEAR_UV_CACHE_DIR:-$INSTALL_CACHE/uv}',
'export UV_TOOL_DIR=$WORK_DIR/cache/uv_tools':'export UV_TOOL_DIR=${PEAR_UV_TOOL_DIR:-$INSTALL_CACHE/uv_tools}',
'export UV_CREDENTIALS_DIR=$WORK_DIR/cache/uv_credentials':'export UV_CREDENTIALS_DIR=${PEAR_UV_CREDENTIALS_DIR:-$INSTALL_CACHE/uv_credentials}',
'export PIP_CACHE_DIR=$WORK_DIR/cache/pip':'export PIP_CACHE_DIR=${PEAR_PIP_CACHE_DIR:-${PIP_CACHE_DIR:-$INSTALL_CACHE/pip}}',
'local micromamba_home=$WORK_DIR/cache/micromamba':'local micromamba_home=${PEAR_MAMBA_HOME:-$INSTALL_CACHE/micromamba}',
'"$WORK_DIR/bin/uv"':'"$INSTALL_BIN/uv"',
'"$WORK_DIR/cache/$archive.tar.gz"':'"$INSTALL_CACHE/$archive.tar.gz"',
'-C "$WORK_DIR/bin"':'-C "$INSTALL_BIN"',
}
for old,new in changes.items():
    assert old in s,old
    s=s.replace(old,new)
after[paths[0]]=s
for p in paths[1:]:
    after[p]=after[p].replace('LOCAL_DIR=$WORK_DIR/cache/environment_test','LOCAL_DIR=${PEAR_TEST_CACHE_DIR:-$WORK_DIR/cache/environment_test}')
p=paths[2]
needle='# for the targets below: an environment\'s files are copies of those in uv\'s cache, not links to them\n'
assert needle in after[p]
after[p]=after[p].replace(needle,'''# Restored runtimes can run the official GPU and simulator checks without reinstalling anything.
if [ "${PEAR_CHECK_ONLY:-0}" = 1 ]; then
  check_environments
  exit 0
fi

'''+needle)
patch=''.join(''.join(difflib.unified_diff(before[p].splitlines(True),after[p].splitlines(True),fromfile='a/'+p,tofile='b/'+p)) for p in paths)
(here/'local_runtime.patch').write_text(patch)
manifest=json.loads((here/'source_manifest.json').read_text())
for p in paths:
    hashes={hashlib.sha256(after[p].encode()).hexdigest()}
    if before[p]!=original[p]:hashes.add(hashlib.sha256(before[p].encode()).hexdigest())
    manifest['allowed_patched_files'][p]=sorted(hashes)
manifest['approved_patches']['local_runtime.patch']=hashlib.sha256(patch.encode()).hexdigest()
(here/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
for p in paths:
    output=here/'review_sources'/p
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(after[p])
print('Wrote local_runtime.patch, full expected sources and updated manifest')
