#!/usr/bin/env python3
"""Supervise one Linux launch tree, including detached and orphaned descendants.

Run this in a dedicated process, not a service which owns other subprocesses.
The supervisor becomes a Linux child subreaper before starting the command. A
descendant which calls setsid(), double-forks or clears its environment remains
part of this tree and is adopted here when its parent exits. No process-name or
machine-wide kill operation is used.
"""

import argparse
import ctypes
import errno
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


class SupervisorCleanupError(RuntimeError):
    """The owned launch tree could not be confirmed empty."""


CleanupError = SupervisorCleanupError


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    uid: int


@dataclass(frozen=True)
class ProcessInfo:
    identity: ProcessIdentity
    ppid: int
    state: str
    comm: str


def _process(pid):
    directory = Path('/proc') / str(pid)
    try:
        raw = (directory / 'stat').read_text()
        end = raw.rfind(')')
        fields = raw[end + 2:].split()
        identity = ProcessIdentity(pid, int(fields[19]), directory.stat().st_uid)
        return ProcessInfo(identity, int(fields[1]), fields[0], raw[raw.find('(') + 1:end])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def _snapshot():
    result = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            process = _process(int(path.name))
            if process is not None:
                result[process.identity.pid] = process
    return result


def _subreaper(enable=None):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('The process supervisor requires Linux /proc and PR_SET_CHILD_SUBREAPER')
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if enable is not None and libc.prctl(36, int(enable), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return previous.value


class _Supervisor:
    def __init__(self, ownership_dir, term_grace, kill_grace):
        self.pid = os.getpid()
        self.uid = os.geteuid()
        self.owner = uuid.uuid4().hex
        self.seen = {}
        self.child = None
        self.requested_signal = None
        self.term_grace = term_grace
        self.kill_grace = kill_grace
        self.directory = Path(ownership_dir)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.stat().st_uid != self.uid:
            raise RuntimeError('Ownership directory belongs to another user')
        self.state_path = self.directory / 'process_supervisor.json'
        # A launch attempt is immutable. Never overwrite an earlier owner's state.
        descriptor = os.open(self.state_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        self.events_path = self.directory / 'process_events.jsonl'
        self.local_directory = Path(tempfile.mkdtemp(prefix=f'pear-owner-{self.uid}-{self.owner}-', dir='/var/tmp'))
        self.local_state_path = self.local_directory / 'process_supervisor.json'
        self.mirror_error = None
        self.last_mirror_warning = 0.0
        self.last_report = 0.0

    def discover(self):
        snapshot = _snapshot()
        selected = {self.pid}
        selected.update(pid for pid, identity in self.seen.items()
                        if pid in snapshot and snapshot[pid].identity == identity)
        while True:
            extra = {pid for pid, info in snapshot.items() if info.ppid in selected}
            if extra.issubset(selected):
                break
            selected.update(extra)
        current = {}
        for pid in selected - {self.pid}:
            if pid not in snapshot:
                continue
            info = snapshot[pid]
            # A dedicated supervisor has no other children. Direct children here
            # are the launch root or descendants adopted by the subreaper.
            self.seen[pid] = info.identity
            current[pid] = info
        return current

    def reap(self):
        # Popen must reap its own root so the original command's exit code is kept.
        if self.child is not None:
            self.child.poll()
        for pid, info in self.discover().items():
            if info.ppid == self.pid and (self.child is None or pid != self.child.pid):
                try:
                    os.waitpid(pid, os.WNOHANG)
                except (ChildProcessError, ProcessLookupError):
                    pass

    def live(self):
        self.reap()
        return {pid: info for pid, info in self.discover().items() if info.state != 'Z'}

    def has_kernel_children(self):
        """Use the kernel child table to certify emptiness, without reaping.

        A /proc walk is not atomic: an unseen child may be read just before its
        parent exits and is then absent from that snapshot's reachable tree.
        As a dedicated subreaper, ECHILD is the stable proof that no descendant
        remains. WNOWAIT preserves the root command's status for Popen.poll().
        """
        try:
            os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            return False
        return True

    def empty(self):
        self.reap()
        return not self.discover() and not self.has_kernel_children()

    def record(self, phase, **extra):
        current = self.discover()
        state = {
            'schema_version': 1,
            'owner_id': self.owner,
            'supervisor_pid': self.pid,
            'uid': self.uid,
            'phase': phase,
            'updated_at': time.time(),
            'root_pid': self.child.pid if self.child is not None else None,
            'requested_signal': self.requested_signal,
            'owned_processes_seen': [asdict(identity) for identity in self.seen.values()],
            'current_processes': [asdict(info) for info in current.values()],
            'local_state_path': str(self.local_state_path),
            'nas_mirror_error': self.mirror_error,
            **extra,
        }
        def write_state(path):
            temporary = path.with_name('.' + path.name + '.' + self.owner + '.tmp')
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(state, stream, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)

        # A NAS quota error must not strand the live process tree. The private
        # local journal is authoritative; the shared copy remains discoverable
        # and explicitly reports its local journal path before any outage.
        write_state(self.local_state_path)
        try:
            write_state(self.state_path)
            # No command line or environment contents are persisted.
            descriptor = os.open(self.events_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(descriptor, 'a') as stream:
                stream.write(json.dumps({'time': state['updated_at'], 'phase': phase, **extra}) + '\n')
            if self.mirror_error is not None:
                self.mirror_error = None
                state['nas_mirror_error'] = None
                write_state(self.local_state_path)
        except OSError as error:
            self.mirror_error = {'errno': error.errno, 'time': time.time()}
            state['nas_mirror_error'] = self.mirror_error
            write_state(self.local_state_path)
            if time.monotonic() - self.last_mirror_warning >= 30:
                try:
                    print(f'PEAR_OWNER_NAS_MIRROR_FAILED errno={error.errno} local_state={self.local_state_path}', file=sys.stderr, flush=True)
                except OSError:
                    pass  # stderr may itself be on the quota-exhausted NAS.
                self.last_mirror_warning = time.monotonic()
        self.last_report = time.monotonic()

    def send(self, identity, sig):
        current = _process(identity.pid)
        if current is None or current.identity != identity or current.state == 'Z':
            return
        if identity.uid != self.uid:
            raise CleanupError(f'Owned descendant {identity.pid} changed user; refusing cross-user signal')
        # pidfds close the PID-reuse window where supported. Some runtime kernels
        # reject pidfd_open with EINVAL, so identity-checked /proc is the fallback.
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            descriptor = None
            try:
                descriptor = os.pidfd_open(identity.pid)
                current = _process(identity.pid)
                if current is not None and current.identity == identity:
                    signal.pidfd_send_signal(descriptor, sig)
                return
            except ProcessLookupError:
                return
            except OSError as exc:
                if exc.errno not in (errno.ENOSYS, errno.EINVAL, errno.EPERM):
                    raise
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        current = _process(identity.pid)
        if current is not None and current.identity == identity:
            try:
                os.kill(identity.pid, sig)
            except ProcessLookupError:
                pass

    def cleanup(self):
        def record_cleanup(phase, **extra):
            try:
                self.record(phase, **extra)
                return True
            except OSError:
                return False  # Signaling must still run if even local storage fails.

        record_cleanup('terminating')
        for sig, grace in ((signal.SIGTERM, self.term_grace), (signal.SIGKILL, self.kill_grace)):
            deadline = time.monotonic() + grace
            signaled = set()
            while True:
                self.reap()
                current = self.discover()
                if not current and not self.has_kernel_children():
                    if not record_cleanup('cleaned', cleanup_complete=True):
                        raise CleanupError('Process tree is empty but cleanup evidence could not be persisted')
                    return
                live = {pid: info for pid, info in current.items() if info.state != 'Z'}
                for info in live.values():
                    if info.identity not in signaled:
                        self.send(info.identity, sig)
                        signaled.add(info.identity)
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        self.reap()
        remaining = self.discover()
        kernel_children = self.has_kernel_children()
        if not remaining and not kernel_children:
            if not record_cleanup('cleaned', cleanup_complete=True):
                raise CleanupError('Process tree is empty but cleanup evidence could not be persisted')
            return
        record_cleanup('cleanup_failed', cleanup_complete=False, remaining_pids=sorted(remaining),
                       kernel_children_pending=kernel_children)
        raise CleanupError(f'Owned launch processes remain after SIGKILL: {sorted(remaining)}')


def supervise(command, env, ownership_dir, term_grace=8.0, kill_grace=5.0, stop_signal=None):
    """Run a command, return its exit code only after its entire owned tree exits.

    Call in a dedicated, single-threaded Linux process with no existing children.
    SIGTERM/SIGINT requests stop the tree and return 128 + signal. Even a successful
    leader exit triggers descendant cleanup. Residual processes raise CleanupError;
    callers must retain their RUN lock rather than publish a cleaned acknowledgement.
    stop_signal may be a callback returning a pending SIGTERM/SIGINT number or
    None. This carries cancellation from a caller's handler across our takeover.
    """
    if not command or term_grace < 0 or kill_grace <= 0:
        raise ValueError('A command and nonnegative TERM/positive KILL grace periods are required')
    _subreaper()  # Fail before creating ownership state on unsupported systems.
    existing = [pid for pid, info in _snapshot().items() if info.ppid == os.getpid()]
    if existing:
        raise RuntimeError(f'Use a dedicated supervisor process; existing children: {existing}')
    supervisor = _Supervisor(ownership_dir, term_grace, kill_grace)
    previous_subreaper = _subreaper(True)
    previous_handlers = {}

    def request_stop(signum, _frame):
        if supervisor.requested_signal is None:
            supervisor.requested_signal = signum

    def consume_stop():
        if supervisor.requested_signal is None and stop_signal is not None:
            pending = stop_signal()
            if pending is not None:
                if pending not in (signal.SIGTERM, signal.SIGINT):
                    raise ValueError('stop_signal must return SIGTERM, SIGINT or None')
                request_stop(int(pending), None)
        return supervisor.requested_signal

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, request_stop)
        consume_stop()
        child_env = dict(os.environ if env is None else env)
        child_env['PEAR_PROCESS_OWNER'] = supervisor.owner
        # A signal caught by the caller before handler installation must prevent
        # command startup. Signals after this check are retained by request_stop.
        if consume_stop() is None:
            supervisor.child = subprocess.Popen(command, env=child_env, start_new_session=True)
            root = _process(supervisor.child.pid)
            if root is not None:
                supervisor.seen[root.identity.pid] = root.identity
            supervisor.record('running')
            while consume_stop() is None and supervisor.child.poll() is None:
                supervisor.reap()
                if time.monotonic() - supervisor.last_report >= 1.0:
                    supervisor.record('running')
                time.sleep(0.1)
        return_code = supervisor.child.poll() if supervisor.child is not None else 0
        supervisor.cleanup()
        if consume_stop() is not None:
            return_code = 128 + supervisor.requested_signal
        elif return_code is None:
            return_code = supervisor.child.wait()
        elif return_code < 0:
            return_code = 128 - return_code
        supervisor.record('finished', cleanup_complete=True, return_code=return_code)
        return return_code
    except BaseException as exc:
        # Covers failed exec, an exception while watching, or an interrupted setup.
        # A cleanup failure must propagate: no caller may interpret it as safe exit.
        if not isinstance(exc, SupervisorCleanupError) and not supervisor.empty():
            supervisor.cleanup()
        if supervisor.child is None and isinstance(exc, FileNotFoundError):
            return_code = 128 + consume_stop() if consume_stop() is not None else 127
            supervisor.record('finished', cleanup_complete=True, return_code=return_code)
            return return_code
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if supervisor.empty():
            _subreaper(previous_subreaper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ownership-dir', required=True)
    parser.add_argument('--term-grace', type=float, default=8.0)
    parser.add_argument('--kill-grace', type=float, default=5.0)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        return supervise(command, os.environ, args.ownership_dir, args.term_grace, args.kill_grace)
    except Exception as exc:
        print(f'PEAR process supervisor failed: {exc}', file=sys.stderr)
        return 125


if __name__ == '__main__':
    raise SystemExit(main())
