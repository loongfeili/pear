"""CPU-only tests for the platform's one-shot launch and keepalive behavior."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ENTRY = Path(__file__).resolve().parents[1] / 'job_entrypoint.sh'


@unittest.skipUnless(sys.platform.startswith('linux'), 'requires Linux sleep infinity')
class JobEntrypointTests(unittest.TestCase):
    def run_fixture(self, corrupt=False, optimize=False, missing_python=False):
        with tempfile.TemporaryDirectory(prefix='pear-entrypoint-test-') as temporary:
            base = Path(temporary)
            release = base / 'release'
            scripts = release / 'deploy/merlin'
            scripts.mkdir(parents=True)
            shutil.copyfile(ENTRY, scripts / ENTRY.name)
            launcher = scripts / 'merlin_stage1_entry.sh'
            sentinel = base / 'executions'
            launcher.write_text('"$PEAR_TRACKING_PYTHON" -c "import sys" || exit 99\nprintf "executed:%s\\n" "$PEAR_TOTAL_GPUS" >> "$FIXTURE_SENTINEL"\nexit 7\n')
            manifest = {'git_commit': 'a' * 40, 'files': {
                str(path.relative_to(release)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in scripts.iterdir()}}
            (release / 'release.json').write_text(json.dumps(manifest))
            if corrupt:
                launcher.write_text(launcher.read_text() + '# changed after publication\n')
            env = dict(os.environ, PEAR_ROOT=str(base / 'root'),
                       PEAR_TRACKING_PYTHON=sys.executable, PEAR_DEPLOY_COMMIT='a' * 40,
                       PEAR_LAUNCH_ID='test-keepalive', ARNOLD_ID='0', ARNOLD_TRIAL_ID='1',
                       FIXTURE_SENTINEL=str(sentinel))
            env.pop('PEAR_TOTAL_GPUS', None)  # Platform wrapper must default consistently to 128.
            if missing_python:
                env['PEAR_TRACKING_PYTHON'] = str(base / 'nonexistent/bin/python3')
            if optimize:
                env['PYTHONOPTIMIZE'] = '1'
            process = subprocess.Popen(['bash', str(scripts / ENTRY.name)], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                deadline = time.monotonic() + 5
                records = []
                while time.monotonic() < deadline:
                    records = list((base / 'root/entrypoints').glob('**/status.json'))
                    if records or process.poll() is not None:
                        break
                    time.sleep(0.02)
                self.assertEqual(len(records), 1, 'No persistent exit status was written')
                state = json.loads(records[0].read_text())
                self.assertEqual(state['state'], 'failed_keepalive')
                self.assertFalse(state['automatic_retry'])
                self.assertIsNone(process.poll(), 'Allocation exited instead of staying alive')
                time.sleep(0.15)
                if corrupt:
                    self.assertNotEqual(state['training_exit_code'], 0)
                    self.assertFalse(sentinel.exists(), 'Corrupt release was executed')
                else:
                    self.assertEqual(state['training_exit_code'], 7)
                    self.assertEqual(sentinel.read_text(), 'executed:128\n', 'Wrong topology or automatic retry')
            finally:
                process.terminate()
                try:
                    process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=3)

    def test_failed_training_keeps_allocation_without_automatic_retry(self):
        self.run_fixture()

    def test_corrupt_release_is_rejected_and_kept_for_inspection(self):
        self.run_fixture(corrupt=True)

    def test_optimized_python_cannot_bypass_release_verification(self):
        self.run_fixture(corrupt=True, optimize=True)

    def test_missing_job_python_override_falls_back_and_records_failure(self):
        self.run_fixture(missing_python=True)


if __name__ == '__main__':
    unittest.main()
