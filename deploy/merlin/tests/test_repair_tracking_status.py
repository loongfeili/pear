import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))
spec = importlib.util.spec_from_file_location('repair_tracking_status', DEPLOY/'repair_tracking_status.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class StatusRepairTests(unittest.TestCase):
    def setUp(self):
        self.expected = dict(project_id='project', run_id='run', launch_id='launch', job_id='job', trial_id='123')
        self.run = dict(ProjectId='project', RunId='run', Name='launch', OuterType='merlin', OuterId='job',
                        OuterInfo={'arnold_trial_id':'123'}, Status='crashed', Step=1, Runtime=600)
        self.calls = []
        parent = self
        class Client:
            def get_run(self, *, run_id, project_id):
                parent.calls.append(('get', run_id, project_id))
                return dict(parent.run) if parent.run else None
            def update_run_heartbeat_time(self, *args): parent.calls.append(('heartbeat', *args))
            def update_run_status(self, *args):
                parent.calls.append(('status', *args));parent.run['Status'] = args[-1]
        self.client = Client()

    def test_repairs_crashed_label_without_writing_points_or_steps(self):
        report = m.restore_status(self.client, self.expected)
        self.assertEqual(report['after'], dict(Status='running', Step=1, Runtime=600, UpdateTime=None))
        self.assertEqual(report['scalar_points_written'], 0)
        self.assertEqual([c[0] for c in self.calls], ['get','heartbeat','status','get'])

    def test_mismatched_trial_cannot_be_repaired(self):
        self.run['OuterInfo'] = {'arnold_trial_id':'456'}
        with self.assertRaises(ValueError): m.restore_status(self.client, self.expected)
        self.assertEqual(len(self.calls),1)

    def test_failed_lookup_does_not_mutate(self):
        self.run = None
        with self.assertRaises(ValueError): m.restore_status(self.client, self.expected)
        self.assertEqual(len(self.calls),1)

    def test_completed_run_cannot_be_revived(self):
        for status in ['finished','failed','stopped']:
            self.run['Status'] = status
            with self.assertRaises(ValueError): m.restore_status(self.client, self.expected)
        self.assertTrue(all(c[0]=='get' for c in self.calls))

    def test_swallowed_sdk_update_error_is_detected_by_readback(self):
        self.client.update_run_status = lambda *args: None
        with patch.object(m.time, 'monotonic', side_effect=[0, 30]):
            with self.assertRaises(RuntimeError): m.restore_status(self.client, self.expected)

    def test_stale_read_after_update_is_retried_until_server_acknowledges(self):
        before = dict(self.run)
        after = dict(self.run, Status='running')
        with patch.object(self.client, 'get_run', side_effect=[before, before, after]), patch.object(m.time, 'sleep') as sleep, patch.object(m.time, 'monotonic', side_effect=[0, 1]):
            report = m.restore_status(self.client, self.expected)
        self.assertEqual(report['after']['Status'], 'running')
        sleep.assert_called_once_with(1)

    def test_running_run_does_not_receive_a_status_write(self):
        self.run['Status'] = 'running'
        m.restore_status(self.client, self.expected)
        self.assertEqual([c[0] for c in self.calls],['get','heartbeat','get'])


if __name__ == '__main__': unittest.main()
