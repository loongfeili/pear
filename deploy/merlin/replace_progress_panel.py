#!/usr/bin/env python3
"""Replace only a verified CPU progress panel with a direct cluster connection."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from process_supervisor import ProcessIdentity, _process
from recover_owned_launch import cluster_helper_environment
from tracking_bridge import publish


def validate_panel(argv, run):
    if argv[1:3] != ['-m', 'pear.monitor.progress_panel']:
        raise ValueError('Refusing to signal a non-panel process')
    if '--run' not in argv or argv.index('--run') + 1 == len(argv) or argv[argv.index('--run') + 1] != str(run):
        raise ValueError('Progress panel belongs to another run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-rank', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    directory, run = args.recovery_rank.resolve(), args.run.resolve()
    adoption = json.loads((directory/'adoption.json').read_text())
    journal = json.loads((directory/'processes/process_supervisor.json').read_text())
    if adoption['trial'] != os.environ.get('ARNOLD_TRIAL_ID') or adoption['rank'] != int(os.environ.get('ARNOLD_ID', '-1')):
        raise ValueError('Wrong recovery node')
    if journal['phase'] != 'recovered_running' or not 0 <= time.time()-journal['updated_at'] < 120:
        raise ValueError('Guardian is not healthy')
    root = _process(adoption['root_identity']['pid'])
    if root is None or root.identity != ProcessIdentity(**adoption['root_identity']) or root.state == 'Z':
        raise ValueError('Training root is not live')
    pid = json.loads((directory/'helpers.json').read_text())['pids'][1]
    identity = ProcessIdentity(**next(i for i in journal['owned_processes_seen'] if i['pid']==pid))
    current = _process(pid)
    if current is None or current.identity != identity or current.state == 'Z':
        raise ValueError('Original CPU panel is no longer live; no automatic takeover')
    proc = Path('/proc')/str(pid)
    argv = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    validate_panel(argv, run)
    environment = dict(x.decode().split('=',1) for x in (proc/'environ').read_bytes().split(b'\0') if b'=' in x)
    if environment.get('PEAR_LAUNCH_ID') != adoption['launch_id']:
        raise ValueError('Panel invocation marker differs')
    cwd = proc.joinpath('cwd').resolve()
    claim = directory/'progress_replacement.claim'
    claim.mkdir()  # Explicit once-only operation. Never start duplicate writers.
    audit = {'old_identity': identity.__dict__, 'launch_id': adoption['launch_id'], 'run': str(run),
             'proxy_keys_removed': sorted(k for k in environment if k.lower() in ('http_proxy','https_proxy','all_proxy')),
             'started_at': time.time(), 'gpu_processes_restarted': 0}
    publish(claim/'audit.json', audit)
    if _process(pid).identity != identity:
        raise ValueError('Panel birth identity changed before signal')
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic()+10
    while time.monotonic()<deadline:
        current = _process(pid)
        if current is None or current.identity != identity or current.state=='Z':
            break
        time.sleep(.1)
    else:
        raise RuntimeError('CPU panel did not exit; no replacement started')
    environment = cluster_helper_environment(environment)
    # This external CPU replacement cannot be reparented under the guardian.
    # Its original invocation markers let that guardian safely discover its
    # birth identity, just like the already-existing original training roots.
    environment['PEAR_PROCESS_OWNER'] = adoption['old_owner']
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    with (directory/'progress.recovered.log').open('ab') as log:
        child = subprocess.Popen(argv, env=environment, cwd=cwd, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    new = _process(child.pid)
    if new is None:
        raise RuntimeError('Replacement panel exited immediately')
    audit.update(new_identity=new.identity.__dict__, replaced_at=time.time(), kernel_reparenting=False)
    publish(directory/'progress_replacement.json', audit)
    print('PEAR_CPU_PROGRESS_PANEL_REPLACED='+json.dumps(audit), flush=True)


if __name__=='__main__':
    main()
