"""Real Linux process-tree tests. No Ray, CUDA or training process is started."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SUPERVISOR = Path(__file__).resolve().parents[1] / 'process_supervisor.py'


def process_identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return {'pid': pid, 'start_ticks': int(fields[19]), 'state': fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process_alive(identity):
    current = process_identity(identity['pid'])
    return current is not None and current['start_ticks'] == identity['start_ticks'] and current['state'] != 'Z'


@unittest.skipUnless(sys.platform.startswith('linux') and Path('/proc').is_dir(), 'Requires Linux subreaper and /proc')
class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='pear-supervisor-test-')
        self.root = Path(self.temporary.name)
        self.supervisors = []
        self.fixture_identities = []

    def tearDown(self):
        for process in self.supervisors:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=3)
            elif process.stderr is not None:
                process.stderr.close()
        # Only fixture PIDs with a matching birth identity are eligible here.
        for identity in self.fixture_identities:
            if same_process_alive(identity):
                os.kill(identity['pid'], signal.SIGKILL)
        self.temporary.cleanup()

    def launch(self, code=None, command=None, term_grace=0.2):
        directory = self.root / f'owner-{len(self.supervisors)}'
        command = command or [sys.executable, '-c', code]
        process = subprocess.Popen(
            [sys.executable, str(SUPERVISOR), '--ownership-dir', str(directory),
             '--term-grace', str(term_grace), '--kill-grace', '2', '--', *command],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        self.supervisors.append(process)
        return process, directory

    def wait_file(self, path, process):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if path.exists():
                try:
                    return json.loads(path.read_text())
                except json.JSONDecodeError:
                    pass
            if process.poll() is not None:
                self.fail(f'Supervisor exited before fixture was ready: {process.communicate()[1]}')
            time.sleep(0.02)
        self.fail('Fixture did not become ready')

    def assert_finished_cleanly(self, process, directory, code):
        _, stderr = process.communicate(timeout=12)
        self.assertEqual(process.returncode, code, stderr)
        state = json.loads((directory / 'process_supervisor.json').read_text())
        self.assertEqual(state['phase'], 'finished', state)
        self.assertTrue(state['cleanup_complete'], state)
        self.assertEqual(state['current_processes'], [], state)
        self.assertEqual(state['return_code'], code, state)
        return state

    def test_leader_exit_still_kills_same_group_term_resistant_child(self):
        ready = self.root / 'same-group.json'
        code = f'''
import json, os, pathlib, signal, time
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raw = pathlib.Path('/proc/self/stat').read_text()
    pathlib.Path({str(ready)!r}).write_text(json.dumps({{'pid': os.getpid(), 'start_ticks': int(raw[raw.rfind(')')+2:].split()[19])}}))
    while True: time.sleep(1)
while not pathlib.Path({str(ready)!r}).exists(): time.sleep(0.01)
os._exit(7)
'''
        process, directory = self.launch(code)
        identity = self.wait_file(ready, process)
        self.fixture_identities.append(identity)
        state = self.assert_finished_cleanly(process, directory, 7)
        self.assertFalse(same_process_alive(identity))
        self.assertIn(identity['pid'], [item['pid'] for item in state['owned_processes_seen']])

    def test_double_fork_setsid_and_empty_environment_are_still_owned(self):
        ready = self.root / 'detached.json'
        grandchild = f'''
import json, os, pathlib, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
raw = pathlib.Path('/proc/self/stat').read_text()
pathlib.Path({str(ready)!r}).write_text(json.dumps({{'pid': os.getpid(), 'start_ticks': int(raw[raw.rfind(')')+2:].split()[19]), 'marker_present': 'PEAR_PROCESS_OWNER' in os.environ}}))
while True: time.sleep(1)
'''
        code = f'''
import os, pathlib, sys, time
middle = os.fork()
if middle == 0:
    os.setsid()
    if os.fork() == 0:
        os.execve(sys.executable, [sys.executable, '-c', {grandchild!r}], {{}})
    os._exit(0)
while not pathlib.Path({str(ready)!r}).exists(): time.sleep(0.01)
os._exit(0)
'''
        process, directory = self.launch(code)
        identity = self.wait_file(ready, process)
        self.fixture_identities.append(identity)
        self.assertFalse(identity['marker_present'])
        self.assert_finished_cleanly(process, directory, 0)
        self.assertFalse(same_process_alive(identity))

    def test_sigterm_cleans_detached_children_and_preserves_unrelated_process(self):
        ready = self.root / 'signal-child.json'
        code = f'''
import json, os, pathlib, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if os.fork() == 0:
    os.setsid()
    raw = pathlib.Path('/proc/self/stat').read_text()
    pathlib.Path({str(ready)!r}).write_text(json.dumps({{'pid': os.getpid(), 'start_ticks': int(raw[raw.rfind(')')+2:].split()[19])}}))
while True: time.sleep(1)
'''
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            process, directory = self.launch(code)
            identity = self.wait_file(ready, process)
            self.fixture_identities.append(identity)
            process.send_signal(signal.SIGTERM)
            self.assert_finished_cleanly(process, directory, 143)
            self.assertFalse(same_process_alive(identity))
            self.assertIsNone(unrelated.poll(), 'Supervisor signaled an unrelated process')
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=3)

    def test_missing_command_returns_clean_127(self):
        process, directory = self.launch(command=[str(self.root / 'does-not-exist')])
        self.assert_finished_cleanly(process, directory, 127)

    def launch_api(self, body, directory):
        """Exercise the API in its own process, never in the test runner."""
        script = f'''
import json, os, pathlib, signal, sys, time
sys.path.insert(0, {str(SUPERVISOR.parent)!r})
import process_supervisor as module
{body}
'''
        process = subprocess.Popen([sys.executable, '-c', script], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self.supervisors.append(process)
        return process, directory

    def test_pending_stop_during_handler_handoff_does_not_start_command(self):
        directory = self.root / 'handoff-owner'
        started = self.root / 'must-not-start'
        command = [sys.executable, '-c', f'from pathlib import Path; Path({str(started)!r}).touch()']
        body = f'''
pending = [None]
def caller_handler(number, frame):
    pending[0] = number
signal.signal(signal.SIGTERM, caller_handler)
os.kill(os.getpid(), signal.SIGTERM)
result = module.supervise({command!r}, None, {str(directory)!r}, term_grace=0.1,
                          stop_signal=lambda: pending[0])
assert signal.getsignal(signal.SIGTERM) is caller_handler
raise SystemExit(result)
'''
        process, directory = self.launch_api(body, directory)
        self.assert_finished_cleanly(process, directory, 143)
        self.assertFalse(started.exists(), 'Pending stop was lost during handler takeover')

    def test_stop_callback_is_consumed_while_command_is_running(self):
        directory = self.root / 'callback-owner'
        ready = self.root / 'callback-child.json'
        command_code = f'''
import json, os, pathlib, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
raw = pathlib.Path('/proc/self/stat').read_text()
pathlib.Path({str(ready)!r}).write_text(json.dumps({{'pid': os.getpid(), 'start_ticks': int(raw[raw.rfind(')')+2:].split()[19])}}))
while True: time.sleep(1)
'''
        command = [sys.executable, '-c', command_code]
        body = f'''
result = module.supervise({command!r}, None, {str(directory)!r}, term_grace=0.1,
                          stop_signal=lambda: signal.SIGINT if pathlib.Path({str(ready)!r}).exists() else None)
raise SystemExit(result)
'''
        process, directory = self.launch_api(body, directory)
        identity = self.wait_file(ready, process)
        self.fixture_identities.append(identity)
        self.assert_finished_cleanly(process, directory, 130)
        self.assertFalse(same_process_alive(identity))

    def test_transient_empty_proc_snapshot_cannot_certify_live_orphan_cleaned(self):
        directory = self.root / 'snapshot-owner'
        ready = self.root / 'snapshot-orphan.json'
        command_code = f'''
import json, os, pathlib, signal, time
if os.fork() == 0:
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raw = pathlib.Path('/proc/self/stat').read_text()
    pathlib.Path({str(ready)!r}).write_text(json.dumps({{'pid': os.getpid(), 'start_ticks': int(raw[raw.rfind(')')+2:].split()[19])}}))
    while True: time.sleep(1)
while not pathlib.Path({str(ready)!r}).exists(): time.sleep(0.01)
os._exit(7)
'''
        command = [sys.executable, '-c', command_code]
        # Model a non-atomic /proc walk missing the newly adopted orphan. The
        # processes and kernel child table are real; only those first snapshots
        # are suppressed, so this deterministically exercises the race window.
        body = f'''
original_discover = module._Supervisor.discover
original_cleanup = module._Supervisor.cleanup
def transient_miss(self):
    remaining = getattr(self, 'missing_scans', 0)
    if remaining:
        self.missing_scans -= 1
        return {{}}
    return original_discover(self)
def inject_cleanup(self):
    self.missing_scans = 5
    return original_cleanup(self)
module._Supervisor.discover = transient_miss
module._Supervisor.cleanup = inject_cleanup
result = module.supervise({command!r}, None, {str(directory)!r}, term_grace=0.2, kill_grace=2)
raise SystemExit(result)
'''
        process, directory = self.launch_api(body, directory)
        identity = self.wait_file(ready, process)
        self.fixture_identities.append(identity)
        self.assert_finished_cleanly(process, directory, 7)
        self.assertFalse(same_process_alive(identity), 'An empty /proc snapshot hid a live orphan')

    def test_rejects_process_that_already_has_other_children(self):
        ready = self.root / 'existing-child.json'
        code = f'''
import importlib.util, json, pathlib, subprocess, sys
spec = importlib.util.spec_from_file_location('pear_test_supervisor', {str(SUPERVISOR)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
try:
    try:
        module.supervise([sys.executable, '-c', 'pass'], None, {str(self.root / 'invalid-owner')!r})
    except RuntimeError as exc:
        pathlib.Path({str(ready)!r}).write_text(json.dumps({{'rejected': 'dedicated' in str(exc), 'other_child_alive': child.poll() is None}}))
    else:
        raise AssertionError('Expected dedicated-process guard')
finally:
    child.terminate()
    child.wait(timeout=3)
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=8)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(ready.read_text()), {'rejected': True, 'other_child_alive': True})


if __name__ == '__main__':
    unittest.main()
