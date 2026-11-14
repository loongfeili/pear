"""Linux CPU-only integration regressions for launch ownership and RUN locking.

Run with: python3 -m unittest discover -s deploy/merlin/tests -p test_launcher_control.py -v
The real controller and process supervisor run in child processes.  Only the
worker command is a harmless, bounded file-sentinel fixture; no GPU/Ray imports.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


CONTROLLER = Path(__file__).resolve().parents[1] / 'launcher_control.py'
WORKER = r'''
import json, os, pathlib, sys, time
ready, gate = map(pathlib.Path, sys.argv[1:3])
raw = pathlib.Path('/proc/self/stat').read_text()
fields = raw[raw.rfind(')') + 2:].split()
record = {'pid': os.getpid(), 'start_ticks': int(fields[19]),
          'launch_id': os.environ['PEAR_LAUNCH_ID'],
          'run': os.environ['PEAR_RUN'], 'supervised': os.environ['PEAR_SUPERVISED']}
temporary = ready.with_suffix('.partial')
temporary.write_text(json.dumps(record))
temporary.replace(ready)
if sys.argv[3] == 'hold':
    deadline = time.monotonic() + 30
    while not gate.exists():
        if time.monotonic() >= deadline:
            raise SystemExit('fixture worker exceeded its 30-second limit')
        time.sleep(0.01)
'''


def process_identity(pid):
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = raw[raw.rfind(')') + 2:].split()
    return int(fields[19]), fields[0]


@unittest.skipUnless(sys.platform.startswith('linux'), 'requires Linux /proc and child subreaper')
class LauncherControlIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='pear-launch-control-test-')
        self.base = Path(self.temporary.name)
        self.root = self.base / 'root'
        self.run = self.base / 'run'
        self.attempts = []
        self.addCleanup(self._cleanup_fixtures)

    def _cleanup_fixtures(self):
        # First allow all fixture commands to finish normally. Every PID below
        # was created by this test; never use a name-based or machine-wide kill.
        for attempt in self.attempts:
            attempt['gate'].touch()
        for attempt in self.attempts:
            process = attempt['process']
            try:
                process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=3)
        # A failing supervisor must not leak even these harmless test workers.
        # Recheck start_ticks before each fallback signal to avoid PID reuse.
        for attempt in self.attempts:
            if not attempt['ready'].exists():
                continue
            record = json.loads(attempt['ready'].read_text())
            for sig in (signal.SIGTERM, signal.SIGKILL):
                identity = process_identity(record['pid'])
                if identity is None or identity[0] != record['start_ticks'] or identity[1] == 'Z':
                    break
                try:
                    os.kill(record['pid'], sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    current = process_identity(record['pid'])
                    if current is None or current[0] != record['start_ticks'] or current[1] == 'Z':
                        break
                    time.sleep(0.01)
        self.temporary.cleanup()

    def _start(self, launch, *, rank=0, nodes=1, run=None, node=None,
               hold=True, resume=False, cleanup_timeout=2, start_gate=None, controller_bootstrap=None):
        number = len(self.attempts)
        ready = self.base / f'worker-{number}.json'
        gate = self.base / f'finish-{number}'
        chosen_run = self.run if run is None else Path(run)
        node_lock = self.base / f'node-{rank}.active' if node is None else Path(node)
        args = [sys.executable, str(CONTROLLER), '--root', str(self.root),
                '--run', str(chosen_run), '--launch-id', launch, '--rank', str(rank),
                '--nodes', str(nodes), '--node-lock', str(node_lock), '--timeout', '2',
                '--cleanup-timeout', str(cleanup_timeout), '--term-grace', '0.1']
        if resume:
            args.append('--resume')
        args += ['--', sys.executable, '-c', WORKER, str(ready), str(gate),
                 'hold' if hold else 'exit']
        if controller_bootstrap is not None:
            args = [sys.executable, '-c', controller_bootstrap, str(CONTROLLER), *args[2:]]
        if start_gate is not None:
            rendezvous = ("import os,pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
                          "deadline=time.monotonic()+30\n"
                          "while not p.exists():\n"
                          "    if time.monotonic()>deadline: raise SystemExit(90)\n"
                          "    time.sleep(0.01)\n"
                          "os.execv(sys.argv[2],sys.argv[2:])\n")
            args = [sys.executable, '-c', rendezvous, str(start_gate), *args]
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        attempt = {'process': process, 'ready': ready, 'gate': gate,
                   'node_lock': node_lock, 'launch': launch}
        self.attempts.append(attempt)
        return attempt

    def _wait_for(self, predicate, message, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.01)
        self.fail(message)

    def _ready(self, attempt):
        def ready_or_exited():
            if attempt['ready'].exists():
                return True
            if attempt['process'].poll() is not None:
                out, err = attempt['process'].communicate()
                self.fail(f"Controller exited before worker readiness: {out}\n{err}")
            return False
        self._wait_for(ready_or_exited, 'Worker command did not reach readiness')
        record = json.loads(attempt['ready'].read_text())
        self.assertEqual(record['launch_id'], attempt['launch'])
        self.assertEqual(record['supervised'], '1')
        return record

    def _exit(self, attempt, expected=0, timeout=6):
        try:
            out, err = attempt['process'].communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.fail(f"Controller did not exit within {timeout}s: {attempt['launch']}")
        if expected is None:
            self.assertNotEqual(attempt['process'].returncode, 0, (out, err))
        else:
            self.assertEqual(attempt['process'].returncode, expected, (out, err))
        return out, err

    def _control(self, launch):
        return self.root / 'launches' / launch

    def _owner(self):
        return json.loads((self.run / '.pear-active/owner.json').read_text())

    @staticmethod
    def _snapshot(directory):
        return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in directory.rglob('*') if path.is_file()}

    def test_duplicate_rank_rejected_without_poisoning_healthy_launch(self):
        first = self._start('same-rank')
        child = self._ready(first)
        before = self._snapshot(self._control('same-rank') / 'ranks')
        owner = self._owner()
        duplicate = self._start('same-rank')
        self._exit(duplicate, 75)
        self.assertFalse(duplicate['ready'].exists(), 'Rejected call ran its command')
        self.assertFalse(list(self._control('same-rank').glob('failure.*')))
        self.assertEqual(self._snapshot(self._control('same-rank') / 'ranks'), before)
        self.assertEqual(self._owner(), owner)
        self.assertIsNone(first['process'].poll())
        self.assertEqual(process_identity(child['pid'])[0], child['start_ticks'])
        first['gate'].touch()
        self._exit(first)
        self.assertFalse((self.run / '.pear-active').exists())

    def test_duplicate_preserves_existing_failure_bytes_and_mtime(self):
        first = self._start('existing-evidence')
        self._ready(first)
        failure = self._control('existing-evidence') / 'failure.0'
        failure.write_bytes(b'original-owner-evidence\n')
        before = failure.read_bytes(), failure.stat().st_mtime_ns
        duplicate = self._start('existing-evidence')
        self._exit(duplicate, 75)
        self.assertEqual((failure.read_bytes(), failure.stat().st_mtime_ns), before)
        self.assertFalse(duplicate['ready'].exists())
        self.assertIsNone(first['process'].poll())
        first['gate'].touch()
        self._exit(first)

    def test_different_launch_same_run_rejected_before_command(self):
        first = self._start('run-owner')
        self._ready(first)
        owner = self._owner()
        second = self._start('run-contender', node=self.base / 'other-node.active')
        self._exit(second, expected=None)
        self.assertFalse(second['ready'].exists())
        self.assertEqual(self._owner(), owner)
        self.assertFalse(list(self._control('run-owner').glob('failure.*')))
        self.assertIsNone(first['process'].poll())
        first['gate'].touch()
        self._exit(first)

    def test_all_ranks_clean_before_release_then_explicit_resume(self):
        first = self._start('two-ranks', rank=0, nodes=2, cleanup_timeout=5)
        second = self._start('two-ranks', rank=1, nodes=2)
        self._ready(first)
        self._ready(second)
        owner = self._owner()
        checkpoint = self.run / 'checkpoint-sentinel'
        checkpoint.write_text('preserve-existing-training-output')
        first['gate'].touch()
        cleaned0 = self._control('two-ranks') / 'cleaned.0.json'
        self._wait_for(cleaned0.exists, 'Rank zero did not finish local cleanup')
        self.assertEqual(self._owner(), owner, 'RUN released while rank one was active')
        self.assertIsNone(first['process'].poll())
        self.assertIsNone(second['process'].poll())
        self.assertFalse(first['node_lock'].exists())
        second['gate'].touch()
        self._exit(second)
        self._exit(first)
        self.assertFalse((self.run / '.pear-active').exists())
        self.assertFalse(second['node_lock'].exists())
        for rank in (0, 1):
            cleaned = json.loads((self._control('two-ranks') / f'cleaned.{rank}.json').read_text())
            claim = json.loads((self._control('two-ranks') / 'ranks' / str(rank) / 'owner.json').read_text())
            self.assertEqual(cleaned['run_owner'], owner)
            self.assertEqual(cleaned['rank_owner'], claim['nonce'])
        resumed = self._start('resumed-launch', resume=True, hold=False)
        self._exit(resumed)
        self.assertTrue(resumed['ready'].exists())
        self.assertEqual(checkpoint.read_text(), 'preserve-existing-training-output')
        self.assertFalse((self.run / '.pear-active').exists())

    def test_missing_rank_cleanup_retains_run_lock(self):
        first = self._start('missing-rank', rank=0, nodes=2, hold=False)
        self._exit(first, expected=None)
        owner = self._owner()
        control = self._control('missing-rank')
        self.assertTrue((control / 'cleaned.0.json').exists())
        self.assertFalse((control / 'cleaned.1.json').exists())
        self.assertIn('Cleanup is incomplete', (control / 'failure.0').read_text())
        self.assertFalse(first['node_lock'].exists(), 'Finished local node should be released')
        contender = self._start('after-incomplete', resume=True, hold=False)
        self._exit(contender, expected=None)
        self.assertFalse(contender['ready'].exists())
        self.assertEqual(self._owner(), owner)

    def test_legacy_claims_reject_without_failure_or_worker(self):
        for suffix, relative_claim in [('publisher', 'sharedcontrol/publisher.claim'),
                                       ('rank', 'claim.0')]:
            with self.subTest(claim=relative_claim):
                launch = f'legacy-{suffix}'
                control = self._control(launch)
                path = control / relative_claim
                path.parent.mkdir(parents=True, exist_ok=True)
                if suffix == 'publisher':
                    path.mkdir()
                    (path / 'evidence').write_text('original publisher')
                else:
                    path.write_text('original rank owner')
                before = self._snapshot(control)
                duplicate = self._start(launch)
                self._exit(duplicate, 75)
                self.assertFalse(duplicate['ready'].exists())
                self.assertEqual(self._snapshot(control), before)
                self.assertFalse(list(control.glob('failure.*')))
                self.assertFalse((self.run / '.pear-active').exists())

    def test_nonempty_run_requires_explicit_resume(self):
        self.run.mkdir()
        output = self.run / 'checkpoint-sentinel'
        output.write_text('must not be changed')
        rejected = self._start('nonempty-rejected', hold=False)
        self._exit(rejected, expected=None)
        self.assertFalse(rejected['ready'].exists())
        self.assertFalse((self.run / '.pear-active').exists(), 'Unpublished own lock should be released')
        resumed = self._start('nonempty-resumed', resume=True, hold=False)
        self._exit(resumed)
        self.assertTrue(resumed['ready'].exists())
        self.assertEqual(output.read_text(), 'must not be changed')
        self.assertFalse((self.run / '.pear-active').exists())

    def test_canonical_run_alias_cannot_bypass_lock(self):
        first = self._start('canonical-owner')
        self._ready(first)
        alias = self.base / 'run-alias'
        alias.symlink_to(self.run, target_is_directory=True)
        owner = self._owner()
        contender = self._start('alias-contender', run=alias, hold=False,
                                node=self.base / 'different-node.active')
        self._exit(contender, expected=None)
        self.assertFalse(contender['ready'].exists())
        self.assertEqual(self._owner(), owner)
        first['gate'].touch()
        self._exit(first)

    def test_pending_sigterm_at_supervisor_handoff_prevents_worker_start(self):
        # Inject the real signal after node ownership is acquired but before
        # supervise installs its handlers. Only the timing is overridden; both
        # controller ownership and supervision use their production code.
        bootstrap = r"""
import os, pathlib, signal, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]).parent))
import launcher_control

class SignalAtHandoff(launcher_control.Launch):
    def acquire_node(self):
        super().acquire_node()
        os.kill(os.getpid(), signal.SIGTERM)

launcher_control.Launch = SignalAtHandoff
sys.argv = sys.argv[1:]
raise SystemExit(launcher_control.main())
"""
        attempt = self._start('pending-sigterm', hold=False, controller_bootstrap=bootstrap)
        self._exit(attempt, 143)
        self.assertFalse(attempt['ready'].exists(), 'Cancelled worker command executed')
        self.assertFalse(attempt['node_lock'].exists())
        self.assertFalse((self.run / '.pear-active').exists())
        control = self._control('pending-sigterm')
        self.assertTrue((control / 'cleaned.0.json').exists())
        state = json.loads((control / 'processes/0/process_supervisor.json').read_text())
        self.assertEqual(state['phase'], 'finished')
        self.assertTrue(state['cleanup_complete'])
        self.assertEqual(state['return_code'], 143)
        self.assertEqual(state['requested_signal'], signal.SIGTERM)
        self.assertIsNone(state['root_pid'], 'Supervisor spawned a child after pending cancellation')
        self.assertEqual(state['owned_processes_seen'], [])

    def test_simultaneous_duplicate_has_exactly_one_owner(self):
        # Both controllers compete using the actual atomic mkdir operation.
        barrier = self.base / 'concurrent-start'
        attempts = [self._start('concurrent-rank', start_gate=barrier) for _ in range(2)]
        barrier.touch()
        self._wait_for(lambda: any(a['ready'].exists() for a in attempts),
                       'Neither concurrent controller started its worker')
        winners = [a for a in attempts if a['ready'].exists()]
        self.assertEqual(len(winners), 1)
        winner = winners[0]
        loser = next(a for a in attempts if a is not winner)
        self._exit(loser, 75)
        self.assertFalse(loser['ready'].exists())
        self.assertFalse(list(self._control('concurrent-rank').glob('failure.*')))
        self.assertIsNone(winner['process'].poll())
        winner['gate'].touch()
        self._exit(winner)
        self.assertFalse((self.run / '.pear-active').exists())


if __name__ == '__main__':
    unittest.main()
