#!/usr/bin/env bash
# Materialize verified bundles as local bare mirrors; never changes original upstream URLs.
set -euo pipefail
PEAR_ROOT=${PEAR_ROOT:-/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114}
INPUT=${PEAR_GIT_INPUT:-$PEAR_ROOT/offline_git_inputs}
MIRRORS=$PEAR_ROOT/offline_git_mirrors
python3 - "$INPUT" "$MIRRORS" <<'PY'
import hashlib,json,os,pathlib,subprocess,sys
source,dest=map(pathlib.Path,sys.argv[1:]);manifest=source/'manifest.json'
expected=(source/'manifest.sha256').read_text().split()[0]
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==expected,'Manifest checksum mismatch'
data=json.loads(manifest.read_text());assert data['complete'] and len(data['repositories'])==6

def git(*args,**kw):
    return subprocess.check_output(['git',*map(str,args)],text=True,**kw).strip()

dest.mkdir(parents=True,exist_ok=True)
for entry in data['repositories']:
    name=entry['name'];assert name.replace('_','').isalnum()
    bundle=source/entry['bundle'];assert bundle.name==entry['bundle']
    h=hashlib.sha256()
    with bundle.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==entry['sha256'],f'Bundle checksum mismatch: {name}'
    repo=dest/(name+'.git')
    if repo.exists():
        assert git('-C',repo,'rev-parse','refs/heads/pinned')==entry['commit'],f'Existing mirror differs: {name}'
    else:
        temp=dest/(name+'.git.partial.'+str(os.getpid()))
        subprocess.run(['git','init','--bare','--quiet',str(temp)],check=True)
        subprocess.run(['git','-C',str(temp),'bundle','verify',str(bundle.resolve())],check=True)
        subprocess.run(['git','-C',str(temp),'fetch','--quiet',str(bundle.resolve()),'refs/heads/pinned:refs/heads/pinned'],check=True)
        subprocess.run(['git','-C',str(temp),'symbolic-ref','HEAD','refs/heads/pinned'],check=True)
        subprocess.run(['git','-C',str(temp),'remote','add','origin',entry['url']],check=True)
        assert git('-C',temp,'rev-parse','refs/heads/pinned')==entry['commit']
        temp.rename(repo)
    print('OFFLINE_MIRROR_VERIFIED',name,entry['commit'])
ready={'manifest_sha256':expected,'repositories':data['repositories']}
p=dest/'OFFLINE_READY.partial';p.write_text(json.dumps(ready,indent=2)+'\n');p.replace(dest/'OFFLINE_READY')
PY
