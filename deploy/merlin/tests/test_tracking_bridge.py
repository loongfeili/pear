"""CPU regressions. SDK calls use an explicit test double; no platform Run is created."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'tracking_bridge.py'
spec = importlib.util.spec_from_file_location('pear_tracking_bridge', SOURCE)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def log(self, data, step, commit):
        self.calls.append((dict(data), step, commit))


class MetricsTailTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pear-tracking-test-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.run = self.base / 'run'
        self.run.mkdir()
        self.state = self.base / 'state.json'
        self.metrics = self.run / 'metrics_20261115.jsonl'
        self.backend = RecordingBackend()

    def tail(self, **kwargs):
        return bridge.MetricsTail(self.run, self.state, 'test-launch', 'test-project', **kwargs)

    def append(self, step, data):
        with self.metrics.open('a') as stream:
            stream.write(json.dumps({'step': step, 'data': data}) + '\n')

    def test_partial_line_is_not_consumed_and_same_step_is_combined(self):
        self.append(1, {'actor/loss': 0.5})
        with self.metrics.open('a') as stream:
            stream.write('{"step":1,"data":{"actor/grad_norm":2')
        tail = self.tail()
        self.assertEqual(tail.collect(now=10), 1)
        first_offset = tail.state['files'][self.metrics.name]['offset']
        self.assertLess(first_offset, self.metrics.stat().st_size)
        tail.flush(self.backend, now=11)
        self.assertEqual(self.backend.calls, [])
        with self.metrics.open('a') as stream:
            stream.write('}}\n')
        tail.collect(now=11)
        tail.flush(self.backend, final=True)
        self.assertEqual(self.backend.calls, [({'actor/loss': 0.5, 'actor/grad_norm': 2}, 1, True)])

    def test_next_step_flushes_previous_but_settle_holds_latest(self):
        self.append(4, {'loss': 4})
        self.append(5, {'loss': 5})
        tail = self.tail(settle_seconds=2)
        tail.collect(now=10)
        tail.flush(self.backend, now=10)
        self.assertEqual([item[1] for item in self.backend.calls], [4])
        tail.flush(self.backend, now=12)
        self.assertEqual([item[1] for item in self.backend.calls], [4, 5])

    def test_restart_preserves_pending_and_avoids_replaying_submitted_lines(self):
        self.append(1, {'loss': 1})
        self.append(2, {'loss': 2})
        tail = self.tail()
        tail.collect(now=10)
        tail.flush(self.backend, now=10)
        resumed = self.tail()
        self.assertEqual(resumed.collect(now=11), 0)
        resumed.flush(self.backend, final=True)
        self.assertEqual([item[1] for item in self.backend.calls], [1, 2])
        resumed_again = self.tail()
        resumed_again.collect()
        resumed_again.flush(self.backend, final=True)
        self.assertEqual(len(self.backend.calls), 2)

    def test_late_conflicts_are_reported_without_rewriting_old_steps(self):
        self.append(1, {'loss': 1})
        tail = self.tail()
        tail.collect()
        tail.flush(self.backend, final=True)
        self.append(1, {'loss': 1})
        self.append(1, {'new_metric': 2})
        tail.collect()
        tail.flush(self.backend, final=True)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(tail.state['duplicates'], 1)
        self.assertEqual(tail.state['late_conflicts'], 1)

    def test_failed_sdk_submit_does_not_advance_durable_step(self):
        self.append(3, {'loss': 2})
        tail = self.tail()
        tail.collect()
        class FailingBackend:
            def log(self, *args, **kwargs):
                raise RuntimeError('fixture SDK failure')
        with self.assertRaises(RuntimeError):
            tail.flush(FailingBackend(), final=True)
        resumed = self.tail()
        self.assertEqual(resumed.state['last_submitted_step'], -1)
        resumed.flush(self.backend, final=True)
        self.assertEqual(self.backend.calls, [({'loss': 2}, 3, True)])

    def test_source_filter_skips_old_files_and_never_writes_to_run(self):
        self.append(1, {'old': 1})
        os.utime(self.metrics, (100, 100))
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.run.iterdir()}
        tail = self.tail(metrics_since=200)
        tail.collect()
        tail.flush(self.backend, final=True)
        self.assertEqual(self.backend.calls, [])
        self.assertEqual({p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.run.iterdir()}, before)

    def test_missing_run_is_not_created(self):
        absent = self.base / 'not-yet-owned-run'
        with self.assertRaises(ValueError):
            bridge.MetricsTail(absent, self.state, 'launch', 'project')
        self.assertFalse(absent.exists())
        self.assertFalse(self.state.exists())

    def test_truncation_is_rejected(self):
        self.append(1, {'loss': 1})
        tail = self.tail()
        tail.collect()
        self.metrics.write_bytes(b'')
        with self.assertRaises(ValueError):
            tail.collect()

    def test_duplicate_bridge_refuses_without_overwriting_original_status(self):
        control = self.base / 'control'
        control.mkdir()
        (control / 'tracking-bridge.claim').mkdir()
        status = control / 'tracking.json'
        status.write_text('{"state":"original-owner"}')
        before = status.read_bytes(), status.stat().st_mtime_ns
        result = subprocess.run([sys.executable, '-B', str(SOURCE), '--run', str(self.run),
                                 '--control', str(control), '--launch-id', 'duplicate'],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertEqual((status.read_bytes(), status.stat().st_mtime_ns), before)
        self.assertFalse((control / 'tracking_state.json').exists())

    def test_official_sdk_merlin_fields_and_priority_are_checked(self):
        env = {'RH2_JOB_RUN_ID': 'real-job', 'ARNOLD_TRIAL_ID': '123', 'RH2_HOST': 'seed.bytedance.net'}
        self.assertEqual(bridge.merlin_identity(env)['job_id'], 'real-job')
        self.assertTrue(bridge.merlin_identity(env)['is_seed'])
        with self.assertRaises(ValueError):
            bridge.merlin_identity({**env, 'SEED_INSTANCE_ID': 'different-outer'})
        with self.assertRaises(ValueError):
            bridge.merlin_identity({'ARNOLD_TRIAL_ID': '123'})

    def test_url_is_taken_from_sdk_and_not_synthesized(self):
        url = 'https://tracking.example.invalid/detail?Id=test-project&selectedTrial=test-run'
        self.assertEqual(bridge.actual_url_record(url),
                         {'url': url, 'project_id': 'test-project', 'run_id': 'test-run'})
        self.assertEqual(bridge.actual_url_record(None), {})
        with self.assertRaises(ValueError):
            bridge.actual_url_record('https://wandb.ai/not-an-internal-run')


FAKE_BACKEND_BOOTSTRAP = r'''
import importlib.util, json, pathlib, sys
source = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('test_bridge', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class FakeBackend:
    def __init__(self, project, name, unique_id, directory, config):
        self.path = pathlib.Path(directory).parent / 'fixture_sdk_calls.jsonl'
    def log(self, data, step, commit):
        with self.path.open('a') as stream:
            stream.write(json.dumps({'kind':'log','step':step,'data':data,'commit':commit})+'\n')
    def url_record(self):
        return {'url':'https://tracking.example.invalid/?Id=fixture&selectedTrial=fixture',
                'project_id':'fixture','run_id':'fixture'}
    def finish(self, exit_code):
        with self.path.open('a') as stream:
            stream.write(json.dumps({'kind':'finish','exit_code':exit_code})+'\n')
module.WandbBackend = FakeBackend
sys.argv = sys.argv[1:]
raise SystemExit(module.main())
'''


@unittest.skipUnless(hasattr(signal, 'SIGALRM'), 'requires POSIX signal support')
class BridgeShutdownTests(unittest.TestCase):
    def test_sigterm_flushes_pending_original_metrics_then_finishes(self):
        self._exercise_stop(True)

    def test_normal_stop_file_flushes_then_finishes_successfully(self):
        self._exercise_stop(False)

    def _exercise_stop(self, use_signal):
        with tempfile.TemporaryDirectory(prefix='pear-tracking-signal-') as directory:
            base = Path(directory)
            run, control = base / 'run', base / 'control'
            run.mkdir()
            control.mkdir()
            env = dict(os.environ, RH2_JOB_RUN_ID='fixture-job', ARNOLD_TRIAL_ID='123',
                       RH2_HOST='seed.bytedance.net', PYTHONDONTWRITEBYTECODE='1')
            env.pop('SEED_INSTANCE_ID', None)
            env.pop('SEED_ARENA_EVALUATION_ID', None)
            process = subprocess.Popen([sys.executable, '-B', '-c', FAKE_BACKEND_BOOTSTRAP, str(SOURCE),
                '--run', str(run), '--control', str(control), '--launch-id', 'fixture-launch',
                '--poll', '0.02', '--settle', '30', '--finish-timeout', '2'], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 5
                while not (control / 'tracking_state.json').exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(str(process.communicate()))
                    time.sleep(0.01)
                self.assertTrue((control / 'tracking_state.json').exists())
                metrics = run / 'metrics_current.jsonl'
                metrics.write_text('{"step":7,"data":{"actor/loss":0.125}}\n')
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    state = json.loads((control / 'tracking_state.json').read_text())
                    if '7' in state['pending']:
                        break
                    time.sleep(0.01)
                self.assertIn('7', state['pending'])
                expected_exit = 143 if use_signal else 0
                if use_signal:
                    process.send_signal(signal.SIGTERM)
                else:
                    bridge.publish(control / 'tracking.stop.json', {'exit_code': 0})
                out, err = process.communicate(timeout=5)
                self.assertEqual(process.returncode, expected_exit, (out, err))
                calls = [json.loads(line) for line in (control / 'fixture_sdk_calls.jsonl').read_text().splitlines()]
                self.assertEqual(calls, [{'kind':'log','step':7,'data':{'actor/loss':0.125},'commit':True},
                                         {'kind':'finish','exit_code':expected_exit}])
                status = json.loads((control / 'tracking.json').read_text())
                self.assertEqual(status['last_submitted_step'], 7)
                self.assertTrue(status['sdk_finish_completed'])
                self.assertFalse((control / 'tracking-bridge.claim').exists())
                self.assertEqual(list(run.iterdir()), [metrics], 'Bridge wrote extra files into RUN')
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=3)


if __name__ == '__main__':
    unittest.main()
