#!/usr/bin/env python3
"""CPU-only bridge from PEAR's real JSONL metrics to internal byted-wandb.

Use the image Python, never install or import this SDK in the certified verl
venv. The bridge never writes into RUN and never signals the training process.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from urllib.parse import parse_qs, urlparse
import uuid


def publish(path, value):
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def merlin_identity(environment):
    # Exact precedence/fields from byted-wandb 0.13.98 tracking_cloud.py:119-175.
    if environment.get('SEED_INSTANCE_ID') or environment.get('SEED_ARENA_EVALUATION_ID'):
        raise ValueError('Evaluation identity would override the Merlin job association')
    job, trial = environment.get('RH2_JOB_RUN_ID'), environment.get('ARNOLD_TRIAL_ID')
    if not job or not trial or not trial.isdigit():
        raise ValueError('RH2_JOB_RUN_ID and numeric ARNOLD_TRIAL_ID are required')
    return {'outer_type': 'merlin', 'job_id': job, 'trial_id': trial,
            'robust_run_id': environment.get('ROBUST_RUN_ID'),
            'is_seed': 'seed' in environment.get('RH2_HOST', '')}


def actual_url_record(url):
    if not url:
        return {}
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if parsed.scheme not in ('https', 'http') or not parsed.hostname:
        raise ValueError('SDK returned an invalid Tracking URL')
    project, run = query.get('Id', []), query.get('selectedTrial', [])
    if not project or not run:
        raise ValueError('SDK Tracking URL lacks actual project/run identifiers')
    return {'url': url, 'project_id': project[0], 'run_id': run[0]}


class MetricsTail:
    """Durably buffer complete JSONL lines; submit once per monotonically increasing step.

    Offsets plus pending data are persisted together. A returned SDK log call is
    recorded as submitted, not as a server acknowledgement (see TRACKING.md).
    """
    def __init__(self, run, state_path, launch_id, project, metrics_since=0, settle_seconds=2):
        self.run, self.path = Path(run).resolve(), Path(state_path)
        if not self.run.is_dir():
            raise ValueError('Controller must acquire/create RUN before the bridge starts')
        self.settle_seconds = settle_seconds
        self.source = {'run': str(self.run), 'launch_id': launch_id, 'project': project,
                       'metrics_since': metrics_since}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['source'] != self.source:
                raise ValueError('Bridge state belongs to a different launch/source')
        else:
            self.state = {'schema_version': 1, 'source': self.source, 'files': {}, 'pending': {},
                          'submitted': {}, 'last_submitted_step': -1, 'late_conflicts': 0,
                          'duplicates': 0, 'non_scalar_values': 0,
                          'sdk_unique_id': 'pear-' + fingerprint(self.source)[:24]}
            self.save()

    def save(self):
        publish(self.path, self.state)

    def collect(self, now=None):
        now = time.time() if now is None else now
        count = 0
        for path in sorted(self.run.glob('metrics_*.jsonl')):
            info = path.stat()
            entry = self.state['files'].get(path.name)
            if entry is None:
                entry = {'offset': 0, 'inode': info.st_ino,
                         'excluded': info.st_mtime < self.source['metrics_since']}
                self.state['files'][path.name] = entry
            if entry['excluded']:
                continue
            if entry['inode'] != info.st_ino or info.st_size < entry['offset']:
                raise ValueError('Metrics file replaced or truncated; refusing to invent a replay position')
            with path.open('rb') as stream:
                stream.seek(entry['offset'])
                while True:
                    line = stream.readline(16 * 1024 * 1024 + 1)
                    if not line:
                        break
                    if len(line) > 16 * 1024 * 1024:
                        raise ValueError('Metrics line exceeds 16 MiB')
                    if not line.endswith(b'\n'):
                        break  # writer has not completed this JSONL record yet
                    row = json.loads(line)
                    step, data = row['step'], row['data']
                    if isinstance(step, bool) or not isinstance(step, int) or step < 0 or not isinstance(data, dict):
                        raise ValueError('Expected a nonnegative integer step and data object')
                    scalar = {}
                    for key, value in data.items():
                        if isinstance(key, str) and isinstance(value, (int, float)) and math.isfinite(value):
                            scalar[key] = value
                        else:
                            self.state['non_scalar_values'] += 1
                    if scalar:
                        key = str(step)
                        if step <= self.state['last_submitted_step']:
                            if self.state['submitted'].get(key) == fingerprint(scalar):
                                self.state['duplicates'] += 1
                            else:
                                self.state['late_conflicts'] += 1
                        else:
                            pending = self.state['pending'].setdefault(key, {'data': {}, 'updated_at': now})
                            pending['data'].update(scalar)
                            pending['updated_at'] = now
                    entry['offset'] = stream.tell()
                    count += 1
        self.save()
        return count

    def flush(self, backend, *, final=False, now=None):
        now = time.time() if now is None else now
        steps = sorted(map(int, self.state['pending']))
        sent = 0
        for step in steps:
            key = str(step)
            pending = self.state['pending'][key]
            if not final and step == steps[-1] and now - pending['updated_at'] < self.settle_seconds:
                break
            # One combined dictionary and one commit per original training step.
            backend.log(pending['data'], step=step, commit=True)
            self.state['last_submitted_step'] = step
            self.state['submitted'][key] = fingerprint(pending['data'])
            del self.state['pending'][key]
            self.save()
            sent += 1
        return sent


class WandbBackend:
    def __init__(self, project, name, unique_id, directory, config):
        if os.environ.get('WANDB_OFFICIAL') == '1':
            raise ValueError('Official external W&B mode is not permitted for this bridge')
        if os.environ.get('WANDB_MODE') in ('offline', 'disabled') or os.environ.get('WANDB_DISABLED', '').lower() in ('1', 'true'):
            raise ValueError('Tracking is disabled/offline in the process environment')
        self.sdk_version = importlib.metadata.version('byted-wandb')
        if self.sdk_version != '0.13.98':
            raise ValueError('Expected the reviewed image byted-wandb 0.13.98')
        import wandb
        if not getattr(wandb, '_IS_TRACKING', False):
            raise ValueError('Loaded wandb is not the internal Tracking SDK')
        self.wandb = wandb
        directory.mkdir(parents=True, exist_ok=True)
        os.environ['WANDB_DIR'] = str(directory)
        os.environ.setdefault('WANDB_CONSOLE', 'off')
        self.run = wandb.init(project=project, name=name, id=unique_id, resume='allow',
                              dir=str(directory), config=config)
        if self.run is None:
            raise RuntimeError('SDK did not initialize a run')

    def log(self, data, step, commit):
        self.wandb.log(data, step=step, commit=commit)

    def url_record(self):
        try:
            url = self.run.get_url()
        except (AttributeError, FileNotFoundError):
            return {}  # SDK tracking_info.json may still be getting published
        return actual_url_record(url)

    def finish(self, exit_code):
        self.wandb.finish(exit_code=exit_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--launch-id', default=os.environ.get('PEAR_LAUNCH_ID'))
    parser.add_argument('--project', default='pear-h20')
    parser.add_argument('--metrics-since', type=float, default=0,
                        help='Only discover files whose mtime is at/after this launch epoch; preserve on restart')
    parser.add_argument('--poll', type=float, default=1)
    parser.add_argument('--settle', type=float, default=2)
    parser.add_argument('--finish-timeout', type=float, default=30)
    args = parser.parse_args()
    if not args.launch_id or min(args.poll, args.settle, args.finish_timeout) <= 0 or args.metrics_since < 0:
        parser.error('A launch ID and positive timeouts are required')
    run, control = Path(args.run).resolve(), Path(args.control).resolve()
    if not run.is_dir() or not control.is_dir():
        parser.error('Controller-owned RUN and CONTROL must already exist')
    os.umask(0o077)
    claim = control / 'tracking-bridge.claim'
    try:
        claim.mkdir()
    except FileExistsError:
        print('Tracking bridge already claimed; no automatic stale-owner takeover', file=sys.stderr)
        return 75
    owner = {'nonce': uuid.uuid4().hex, 'pid': os.getpid(), 'launch_id': args.launch_id}
    publish(claim / 'owner.json', owner)
    status = {'launch_id': args.launch_id, 'project': args.project, 'state': 'starting',
              'run_path': str(run), 'created_at': time.time(), 'metrics_since': args.metrics_since}
    target = control / 'tracking.json'
    backend, tail, exit_code, stopped = None, None, 0, {'signal': None}
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in handlers:
        signal.signal(sig, lambda number, frame: stopped.update(signal=number))
    try:
        identity = merlin_identity(os.environ)
        status['association'] = identity
        publish(target, status)
        tail = MetricsTail(run, control / 'tracking_state.json', args.launch_id, args.project,
                           args.metrics_since, args.settle)
        backend = WandbBackend(args.project, args.launch_id, tail.state['sdk_unique_id'],
                               control / 'tracking_sdk', {**identity, 'source': tail.source,
                               'metrics_origin': 'unmodified PEAR JSONL training output'})
        while True:
            tail.collect()
            status.update(backend.url_record())
            stop_file = control / 'tracking.stop.json'
            if stop_file.exists():
                value = json.loads(stop_file.read_text())
                exit_code = int(value['exit_code'])
                if exit_code < 0 or exit_code > 255:
                    raise ValueError('Invalid training exit_code in stop file')
                status['training_exit_code'] = exit_code
                break
            if stopped['signal'] is not None:
                exit_code = 128 + stopped['signal']
                status['stop_signal'] = stopped['signal']
                break
            tail.flush(backend)
            status.update(state='streaming' if tail.state['submitted'] else 'waiting_for_metrics',
                          last_submitted_step=tail.state['last_submitted_step'],
                          submitted_steps=len(tail.state['submitted']),
                          late_conflicts=tail.state['late_conflicts'], updated_at=time.time())
            publish(target, status)
            time.sleep(args.poll)
        # Stop file is written only after training has exited; consume its final
        # complete lines and the buffered last step before requesting SDK flush.
        tail.collect()
        tail.flush(backend, final=True)
        status.update(backend.url_record())
        status.update(state='finished' if exit_code == 0 else 'stopped',
                      last_submitted_step=tail.state['last_submitted_step'],
                      submitted_steps=len(tail.state['submitted']))
    except Exception as error:
        exit_code = 1
        # SDK exceptions can embed request headers; record the type, not secrets.
        status.update(state='failed', error_type=type(error).__name__)
        print(f'Tracking bridge failed ({type(error).__name__}); training is unaffected', file=sys.stderr)
    finally:
        if backend is not None:
            previous = signal.getsignal(signal.SIGALRM)
            def finish_timeout(number, frame):
                raise TimeoutError('Tracking SDK finish exceeded its deadline')
            signal.signal(signal.SIGALRM, finish_timeout)
            signal.setitimer(signal.ITIMER_REAL, args.finish_timeout)
            try:
                backend.finish(exit_code=exit_code)
                status['sdk_finish_completed'] = True
            except Exception as error:
                status.update(state='failed', finish_error_type=type(error).__name__, sdk_finish_completed=False)
                exit_code = 1
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
        status.update(updated_at=time.time(), bridge_exit_code=exit_code)
        publish(target, status)
        if json.loads((claim / 'owner.json').read_text()) == owner:
            claim.rename(control / ('tracking-bridge.released.' + owner['nonce']))
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
