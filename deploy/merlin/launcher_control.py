#!/usr/bin/env python3
"""Own a worker invocation and serialize all launches writing the same RUN.

Locks use mkdir/rename on the shared filesystem, never flock or a stale-lock TTL.
Only rank zero releases the RUN lock, after every rank confirms process cleanup.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
import uuid

from process_supervisor import supervise


class DuplicateLaunch(RuntimeError):
    pass


def publish(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def release_owned_lock(path, owner, destination):
    """Only the one owner/releaser may call this; never delete a successor lock."""
    if read_json(path / 'owner.json') != owner:
        raise RuntimeError(f'Lock ownership changed; retained {path}')
    # Keep the retired directory as an audit record. No recursive deletion can
    # race with a successor acquiring the original lock pathname.
    path.rename(destination)


class Launch:
    def __init__(self, root, run, launch_id, rank, nodes, node_lock, timeout=1200,
                 cleanup_timeout=120, resume=False, term_grace=10):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', launch_id):
            raise ValueError('Invalid launch ID')
        if nodes < 1 or rank < 0 or rank >= nodes:
            raise ValueError('Invalid rank/node count')
        self.root = Path(root).resolve()
        self.run = Path(run).resolve()
        self.launch_id, self.rank, self.nodes = launch_id, rank, nodes
        self.control = self.root / 'launches' / launch_id
        self.run_lock = self.run / '.pear-active'
        self.node_lock = Path(node_lock).resolve()
        self.timeout, self.cleanup_timeout = timeout, cleanup_timeout
        self.resume, self.term_grace = resume, term_grace
        self.owns_rank = self.owns_run = self.owns_node = False
        self.ready = False
        self.record = self.run_owner = self.node_owner = None
        self.interrupted = None

    def failure(self, message):
        if self.owns_rank:
            publish(self.control / f'failure.{self.rank}', {
                'rank': self.rank, 'launch_id': self.launch_id, 'message': message,
                'rank_owner': self.rank_owner, 'time': time.time()})

    def check_wait(self):
        if self.interrupted is not None:
            raise InterruptedError(f'Received signal {self.interrupted}')
        if list(self.control.glob('failure.*')):
            raise RuntimeError('Another worker failed; inspect this launch failure records')

    def claim_rank(self):
        # Refuse old-format claims as well. An unowned duplicate must not create
        # a failure record in a healthy launch, even during initial preparation.
        if (self.control / f'claim.{self.rank}').exists() or (
                self.rank == 0 and (self.control / 'sharedcontrol/publisher.claim').exists()):
            raise DuplicateLaunch('Existing launch claim; choose a new launch ID')
        claims = self.control / 'ranks'
        claims.mkdir(parents=True, exist_ok=True)
        try:
            (claims / str(self.rank)).mkdir()
        except FileExistsError as error:
            raise DuplicateLaunch('This launch/rank already has an owner') from error
        self.owns_rank = True
        self.rank_owner = uuid.uuid4().hex
        publish(claims / str(self.rank) / 'owner.json', {
            'launch_id': self.launch_id, 'rank': self.rank, 'nonce': self.rank_owner,
            'pid': os.getpid(), 'host': socket.gethostname(), 'created_at': time.time()})

    def acquire_run(self):
        self.run.mkdir(parents=True, exist_ok=True)
        try:
            self.run_lock.mkdir()
        except FileExistsError as error:
            raise RuntimeError(f'RUN is already locked: {self.run_lock}; no automatic stale-lock removal') from error
        self.owns_run = True
        self.run_owner = {'launch_id': self.launch_id, 'run': str(self.run),
                          'nodes': self.nodes, 'nonce': uuid.uuid4().hex}
        publish(self.run_lock / 'owner.json', self.run_owner)
        if not self.resume and any(p.name != '.pear-active' for p in self.run.iterdir()):
            raise RuntimeError('RUN contains previous output; choose a fresh RUN or explicitly enable resume')
        self.check_wait()
        self.record = self.run_owner
        # Workers may start only after this complete launch record is published.
        publish(self.control / 'launch.json', self.record)
        self.ready = True

    def await_run(self):
        deadline = time.monotonic() + self.timeout
        while True:
            self.check_wait()
            record = read_json(self.control / 'launch.json')
            if record is not None:
                expected = {'launch_id': self.launch_id, 'run': str(self.run), 'nodes': self.nodes}
                if any(record.get(key) != value for key, value in expected.items()):
                    raise RuntimeError('Launch workers disagree on RUN or node count')
                if read_json(self.run_lock / 'owner.json') != record:
                    raise RuntimeError('RUN lock does not belong to this launch')
                self.record, self.ready = record, True
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting for rank zero to acquire RUN')
            time.sleep(0.1)

    def acquire_node(self):
        self.node_lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.node_lock.mkdir()
        except FileExistsError as error:
            raise RuntimeError(f'This worker already has an active or uncleared launch: {self.node_lock}') from error
        self.owns_node = True
        self.node_owner = {**self.record, 'rank': self.rank, 'rank_owner': self.rank_owner}
        publish(self.node_lock / 'owner.json', self.node_owner)

    def all_cleaned(self):
        for rank in range(self.nodes):
            value = read_json(self.control / f'cleaned.{rank}.json')
            if value is None or value.get('run_owner') != self.record or value.get('rank') != rank:
                return False
            claim = read_json(self.control / 'ranks' / str(rank) / 'owner.json')
            if claim is None or value.get('rank_owner') != claim.get('nonce'):
                return False
        return True

    def finish(self, cleanup_ok):
        if self.owns_node and cleanup_ok:
            release_owned_lock(self.node_lock, self.node_owner,
                               self.node_lock.with_name(self.node_lock.name + '.released.' + self.rank_owner))
        if self.record is not None and cleanup_ok:
            publish(self.control / f'cleaned.{self.rank}.json', {
                'rank': self.rank, 'run_owner': self.record,
                'rank_owner': self.rank_owner, 'time': time.time()})
        if self.rank != 0 or not self.owns_run:
            return True
        if not cleanup_ok:
            return False
        if self.ready:
            deadline = time.monotonic() + self.cleanup_timeout
            while not self.all_cleaned():
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.1)
        # If ready was never published, no peer could have started a child.
        release_owned_lock(self.run_lock, self.run_owner,
                           self.control / ('released-run-lock.' + self.run_owner['nonce']))
        return True

    def execute(self, command):
        os.umask(0o077)
        status, cleanup_ok = 1, True
        handlers = {}
        try:
            self.claim_rank()
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, lambda number, frame: setattr(self, 'interrupted', number))
            if self.rank == 0:
                self.acquire_run()
            else:
                self.await_run()
            self.check_wait()
            self.acquire_node()
            environment = os.environ.copy()
            environment.update({'PEAR_ROOT': str(self.root), 'PEAR_RUN': str(self.run),
                                'PEAR_LAUNCH_ID': self.launch_id, 'PEAR_SUPERVISED': '1',
                                'PEAR_RANK_OWNER': self.rank_owner})
            # An exception from the supervisor means cleanup is not certified.
            # It must fail closed: neither node lock nor RUN is released.
            cleanup_ok = False
            status = supervise(command, env=environment,
                               ownership_dir=self.control / 'processes' / str(self.rank),
                               term_grace=self.term_grace,
                               stop_signal=lambda: self.interrupted)
            cleanup_ok = True
            if status != 0:
                self.failure(f'Worker command exited with status {status}')
        except DuplicateLaunch as error:
            print(f'Launch rejected: {error}', file=sys.stderr)
            return 75
        except Exception as error:
            self.failure(str(error))
            print(f'Launch failed: {error}', file=sys.stderr)
            status = 1
        finally:
            if self.owns_rank:
                try:
                    if not self.finish(cleanup_ok):
                        self.failure('Cleanup is incomplete; RUN lock retained for operator inspection')
                        status = 1
                except Exception as error:
                    self.failure(f'Could not finalize ownership: {error}')
                    print(f'Lock retained: {error}', file=sys.stderr)
                    status = 1
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--launch-id', required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--node-lock', default=f'/var/tmp/pear-merlin-{os.getuid()}.active')
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('--cleanup-timeout', type=float, default=120)
    parser.add_argument('--term-grace', type=float, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A worker command is required')
    for name in ('timeout', 'cleanup_timeout', 'term_grace'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    return Launch(args.root, args.run, args.launch_id, args.rank, args.nodes,
                  args.node_lock, args.timeout, args.cleanup_timeout,
                  args.resume, args.term_grace).execute(command)


if __name__ == '__main__':
    sys.exit(main())
