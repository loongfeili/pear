#!/usr/bin/env python3
"""Monitor an existing launch after its original supervisor died.

No training command is started and original RUN/node locks are never released.
Existing orphan processes cannot be adopted for kernel reaping; they are only
eligible for signaling with their verified birth identity and ownership record.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import sys
import time
import uuid

from launcher_control import configure_descriptor_limit, publish
from process_supervisor import ProcessIdentity, _Supervisor, _process, _snapshot, _subreaper


def same_identity(identity):
    info = _process(identity.pid)
    return info is not None and info.identity == identity


def validate_adoption(record, rank_owner, run_owner, *, launch_id, rank, uid, host):
    if record.get('phase') != 'running' or record.get('requested_signal') is not None:
        raise ValueError('Original supervisor was already stopping; refusing live recovery')
    if rank_owner.get('launch_id') != launch_id or rank_owner.get('rank') != rank or rank_owner.get('host') != host:
        raise ValueError('Original rank identity does not match this node')
    if rank_owner.get('pid') != record.get('supervisor_pid') or record.get('uid') != uid:
        raise ValueError('Original supervisor identity differs')
    if run_owner.get('launch_id') != launch_id or not re.fullmatch(r'[a-f0-9]{32}', record.get('owner_id', '')):
        raise ValueError('Original launch ownership is invalid')
    if _process(record['supervisor_pid']) is not None:
        raise ValueError('Original supervisor PID still exists; no takeover attempted')
    identities = [ProcessIdentity(**item) for item in record['owned_processes_seen']]
    if any(identity.uid != uid for identity in identities):
        raise ValueError('Original record includes another user')
    roots = [identity for identity in identities if identity.pid == record['root_pid']]
    if len(roots) != 1 or not same_identity(roots[0]) or _process(roots[0].pid).state == 'Z':
        raise ValueError('Original training root birth identity is no longer live')
    return roots[0], identities


class RecoverySupervisor(_Supervisor):
    def __init__(self, directory, record, trial, launch_id, term_grace=10, kill_grace=120):
        self.previous_owner = record['owner_id']
        self.trial, self.launch_id = trial, launch_id
        self.helper_children = []
        super().__init__(directory, term_grace, kill_grace)

    def record(self, phase, **extra):
        if 'cleanup_complete' in extra:
            extra['observed_cleanup_complete'] = extra.pop('cleanup_complete')
        extra.update(kernel_reparenting=False, original_locks_retained=True,
                     original_launch_cleanup_certified=False,
                     ownership_scope='verified_birth_identities_and_invocation_markers_plus_new_children')
        return super().record(phase, **extra)

    def reap(self):
        protected = {child.pid for child in self.helper_children}
        for child in self.helper_children:
            child.poll()
        for pid, info in self.discover().items():
            if info.ppid == self.pid and pid not in protected:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except (ChildProcessError, ProcessLookupError):
                    pass

    def discover(self):
        # The old parent is gone. Include orphans missed by its last snapshot
        # only when all three explicit invocation markers and UID match. Their
        # birth identity is checked again after reading /proc/environ.
        for pid, info in _snapshot().items():
            if info.identity.uid != self.uid or pid in self.seen:
                continue
            try:
                raw = (Path('/proc') / str(pid) / 'environ').read_bytes()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            fields = dict(item.split(b'=', 1) for item in raw.split(b'\0') if b'=' in item)
            expected = {b'PEAR_PROCESS_OWNER': self.previous_owner.encode(),
                        b'ARNOLD_TRIAL_ID': self.trial.encode(),
                        b'PEAR_LAUNCH_ID': self.launch_id.encode()}
            if all(fields.get(key) == value for key, value in expected.items()) and same_identity(info.identity):
                self.seen[pid] = info.identity
        return super().discover()


def start_helpers(args, supervisor, control, run, directory):
    claim = control / 'tracking-bridge.claim'
    retired = None
    if claim.exists():
        owner = json.loads((claim / 'owner.json').read_text())
        info = _process(owner['pid'])
        if owner.get('launch_id') != args.launch_id or not re.fullmatch(r'[a-f0-9]{32}', owner.get('nonce', '')):
            raise ValueError('Tracking claim belongs to a different or invalid launch')
        if info is not None and (info.state != 'Z' or supervisor.seen.get(info.identity.pid) != info.identity):
            raise ValueError('Tracking claim still has a live or different owner')
        if not args.outputs:
            retired = control / ('tracking-bridge.operator-retired-quota.' + owner['nonce'])
            if retired.exists():
                raise ValueError('Tracking retirement record already exists')
            claim.rename(retired)
    if (control / 'tracking.stop.json').exists():
        raise ValueError('Training already requested Tracking to stop')
    env = dict(os.environ, PEAR_LAUNCH_ID=args.launch_id, PEAR_PROCESS_OWNER=supervisor.owner)
    env.pop('PYTHONPATH', None)
    helper_control, panel_run = control, run
    if args.outputs:
        output = Path(args.outputs).resolve()
        helper_control, panel_run = output / 'control', output / 'runs' / args.launch_id
        helper_control.mkdir(parents=True, exist_ok=True)
        panel_run.mkdir(parents=True, exist_ok=True)
        state = control / 'tracking_state.json'
        if not (helper_control / state.name).exists():
            shutil.copyfile(state, helper_control / state.name)
        # SDK binary queues stay local; durable metrics and recovery records use HDFS.
        sdk = helper_control / 'tracking_sdk'
        if not sdk.exists():
            local = Path('/var/tmp') / f'pear-tracking-recovered-{args.trial}-{supervisor.owner}'
            local.mkdir(mode=0o700)
            sdk.symlink_to(local, target_is_directory=True)
        publish(helper_control / 'storage_migration.json', {'original_control': str(control),
                'original_run': str(run), 'outputs': str(output), 'sdk_id_preserved': True,
                'old_tracking_claim_retained': str(claim), 'time': time.time()})
        mirror_outputs(run, panel_run)
        checkpoint = run / 'ckpt'
        target = panel_run / 'ckpt'
        target.mkdir(exist_ok=True)
        if not checkpoint.exists():
            checkpoint.symlink_to(target, target_is_directory=True)
        elif checkpoint.is_dir() and not checkpoint.is_symlink() and not any(checkpoint.iterdir()):
            checkpoint.rmdir()
            checkpoint.symlink_to(target, target_is_directory=True)
    bridge = [args.image_python, str(Path(args.original_release) / 'deploy/merlin/tracking_bridge.py'),
              '--run', str(run), '--control', str(helper_control), '--launch-id', args.launch_id,
              '--project', 'pear-h20']
    children = supervisor.helper_children
    with (directory / 'tracking.recovered.log').open('ab') as log:
        children.append(subprocess.Popen(bridge, env=env, stdout=log, stderr=subprocess.STDOUT))
    repo = Path(args.root) / 'repo'
    compat = Path(args.root) / 'compat/cuda-compat-13-0_580.196.07/usr/local/cuda-13.0/compat'
    env.update(PYTHONPATH=f'{repo}/pear/utils/sitecustomize:{repo}:{repo}/3rdparty/verl/src',
               RAY_ADDRESS=(control / 'sharedcontrol/head_address').read_text().strip(),
               RAY_AUTH_MODE='token', RAY_AUTH_TOKEN_PATH=str(control / 'sharedcontrol/ray.token'),
               CUDA_VISIBLE_DEVICES='', LD_LIBRARY_PATH=str(compat) + ':' + env.get('LD_LIBRARY_PATH', ''))
    panel = [args.training_python, '-m', 'pear.monitor.progress_panel', '--target', '1024',
             '--group', '8', '--run', str(panel_run), '--json']
    with (directory / 'progress.recovered.log').open('ab') as log:
        children.append(subprocess.Popen(panel, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT))
    if args.outputs:
        redirect = [args.training_python, str(Path(__file__).with_name('reroute_agent_logs.py')),
                    '--directory', str(panel_run / 'sim'), '--trial', args.trial, '--launch-id', args.launch_id]
        with (directory / 'diagnostic_reroute.log').open('ab') as log:
            children.append(subprocess.Popen(redirect, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT))
    publish(directory / 'helpers.json', {'pids': [child.pid for child in children],
                                         'retired_tracking_claim': str(retired) if retired else None})
    return children


_mirror_checked = {}


def mirror_outputs(source, destination):
    """Follow existing open legacy log/metrics descriptors into durable HDFS files."""
    for path in [source / 'train.log', *source.glob('metrics_*.jsonl')]:
        if not path.is_file():
            continue
        target = destination / path.name
        offset = target.stat().st_size if target.exists() else 0
        info = path.stat()
        if info.st_size < offset:
            raise ValueError('Legacy output was truncated; refusing a false mirror offset')
        previous = _mirror_checked.get(str(target))
        if previous is not None and previous != (info.st_ino, offset):
            raise ValueError('Legacy source or mirrored output changed unexpectedly')
        if previous is None and offset:
            expected, actual, remaining = hashlib.sha256(), hashlib.sha256(), offset
            with path.open('rb') as original, target.open('rb') as copied:
                while remaining:
                    size = min(remaining, 1024 * 1024)
                    a, b = original.read(size), copied.read(size)
                    if len(a) != size or len(b) != size:
                        raise ValueError('Cannot verify existing mirror prefix')
                    expected.update(a)
                    actual.update(b)
                    remaining -= size
            if expected.digest() != actual.digest():
                raise ValueError('Existing HDFS mirror does not match the original output prefix')
        with path.open('rb') as original, target.open('ab') as output:
            original.seek(offset)
            shutil.copyfileobj(original, output, 1024 * 1024)
        _mirror_checked[str(target)] = (info.st_ino, target.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--launch-id', required=True)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--recovery-root', required=True)
    parser.add_argument('--original-release', required=True)
    parser.add_argument('--image-python', default='/usr/bin/python3')
    parser.add_argument('--training-python', default='/opt/tiger/pear_runtime_20261114/envs/verl/bin/python')
    parser.add_argument('--outputs', help='Durable output directory (HDFS mount); existing open descriptors are mirrored')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.launch_id) or not args.trial.isdigit():
        parser.error('Invalid launch/trial identity')
    if os.environ.get('ARNOLD_TRIAL_ID') != args.trial or os.environ.get('ARNOLD_ID') != str(args.rank):
        parser.error('This process is not in the requested trial/rank')
    os.umask(0o077)
    root = Path(args.root).resolve()
    control, run = root / 'launches' / args.launch_id, root / 'work_dir/runs' / args.launch_id
    old_path = control / 'processes' / str(args.rank) / 'process_supervisor.json'
    record = json.loads(old_path.read_text())
    rank_owner = json.loads((control / 'ranks' / str(args.rank) / 'owner.json').read_text())
    run_owner = json.loads((run / '.pear-active/owner.json').read_text())
    root_identity, identities = validate_adoption(record, rank_owner, run_owner,
        launch_id=args.launch_id, rank=args.rank, uid=os.getuid(), host=socket.gethostname())
    if list(control.glob('failure.*')) or (control / 'training.done').exists():
        raise ValueError('Launch already ended or failed; refusing live recovery')
    limits = configure_descriptor_limit()
    if any(info.ppid == os.getpid() for info in _snapshot().values()):
        raise ValueError('Run recovery in a dedicated process without existing children')
    directory = Path(args.recovery_root).resolve() / f'rank-{args.rank}'
    directory.mkdir(parents=True)  # Exclusive operator-authorized recovery, no stale takeover.
    _subreaper(True)
    supervisor = RecoverySupervisor(directory / 'processes', record, args.trial, args.launch_id)
    for identity in identities:
        if same_identity(identity):
            supervisor.seen[identity.pid] = identity
    adoption = {'launch_id': args.launch_id, 'trial': args.trial, 'rank': args.rank,
                'old_owner': record['owner_id'], 'root_identity': asdict(root_identity),
                'guardian_pid': os.getpid(), 'guardian_owner': supervisor.owner,
                'original_locks_retained': True, 'kernel_reparenting': False,
                'created_at': time.time(), 'limits': limits}
    publish(directory / 'adoption.json', adoption)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: setattr(supervisor, 'requested_signal', number))
    helpers = supervisor.helper_children
    if args.rank == 0:
        try:
            start_helpers(args, supervisor, control, run, directory)
        except Exception as error:
            try:
                publish(directory / 'helper_start_error.json', {'type': type(error).__name__, 'message': str(error)})
            except OSError:
                pass  # The following local journal and cleanup must still run.
    failures, success, observer_error = [], False, None
    mirror_at = 0.0
    recovery_root = Path(args.recovery_root)
    try:
        supervisor.record('recovered_running', adoption=adoption)
        while True:
            info = _process(root_identity.pid)
            root_live = info is not None and info.identity == root_identity and info.state != 'Z'
            try:
                failures = [*control.glob('failure.*'), *recovery_root.glob('failure.*')]
                observer_error = None
            except OSError as error:
                observer_error = error.errno  # Local process monitoring continues during a NAS outage.
            if supervisor.requested_signal is not None or failures or not root_live:
                break
            supervisor.reap()
            if args.rank == 0 and args.outputs and time.monotonic() - mirror_at >= 5:
                try:
                    mirror_outputs(run, Path(args.outputs) / 'runs' / args.launch_id)
                    mirror_at = time.monotonic()
                except (OSError, ValueError) as error:
                    observer_error = getattr(error, 'errno', None) or type(error).__name__
            if time.monotonic() - supervisor.last_report >= 1:
                supervisor.record('recovered_running', adoption=adoption, control_read_errno=observer_error,
                                  helper_exit_codes={str(child.pid): child.poll() for child in helpers})
            time.sleep(0.2)
        success = not failures and (control / 'training.done').exists() and supervisor.requested_signal is None
    finally:
        try:
            if not success and not failures:
                publish(recovery_root / f'failure.{args.rank}', {'launch_id': args.launch_id, 'rank': args.rank,
                        'message': 'Recovered training root ended or received a stop request', 'time': time.time()})
            if args.rank == 0:
                helper_control = Path(args.outputs) / 'control' if args.outputs else control
                if not (helper_control / 'tracking.stop.json').exists():
                    publish(helper_control / 'tracking.stop.json', {'exit_code': 0 if success else 1})
                deadline = time.monotonic() + 35
                while helpers and helpers[0].poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
        except OSError:
            pass  # A second quota failure cannot bypass owned process cleanup.
        supervisor.cleanup()
    # Supplemental evidence only. The old RUN/node locks require an explicit
    # operator audit; an external guardian cannot certify kernel child adoption.
    supervisor.record('recovered_finished', cleanup_complete=True, original_locks_retained=True)
    publish(directory / 'recovery.finished.json', {'success': success, 'time': time.time(),
            'original_locks_retained': True, 'root_identity': asdict(root_identity)})
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
