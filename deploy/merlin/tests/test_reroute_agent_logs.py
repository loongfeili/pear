"""Diagnostic-only routing boundaries; no Ray service or GPU is started."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reroute_agent_logs import reroute_worker


class AgentLoopWorkerTQ:
    def __init__(self, path):
        self.tools = [SimpleNamespace(name='libero', log_path=path)]


class DiagnosticRoutingTests(unittest.TestCase):
    def test_verified_actor_updates_only_diagnostic_references(self):
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder) / f'actions_{os.getpid()}.jsonl'
            old.write_text('original diagnostic history\n')
            worker = AgentLoopWorkerTQ(old)
            worker.episode_state = object()
            episode = worker.episode_state
            with patch.dict(os.environ, {'ARNOLD_TRIAL_ID': '123', 'PEAR_LAUNCH_ID': 'test-launch'}):
                record = reroute_worker(worker, str(Path(folder) / 'new'), '123', 'test-launch')
                self.assertEqual(os.environ['PEAR_LOG_DIR'], record['destination'])
            self.assertEqual(old.read_text(), 'original diagnostic history\n')
            self.assertEqual(worker.tools[0].log_path.parent, Path(record['destination']))
            self.assertIs(worker.episode_state, episode)

    def test_wrong_invocation_is_rejected_without_mutating_actor(self):
        worker = AgentLoopWorkerTQ(Path('/unused') / f'actions_{os.getpid()}.jsonl')
        previous = worker.tools[0].log_path
        with patch.dict(os.environ, {'ARNOLD_TRIAL_ID': '456', 'PEAR_LAUNCH_ID': 'another'}):
            with self.assertRaisesRegex(ValueError, 'another training invocation'):
                reroute_worker(worker, '/unused/new', '123', 'test-launch')
        self.assertEqual(worker.tools[0].log_path, previous)


if __name__ == '__main__':
    unittest.main()
