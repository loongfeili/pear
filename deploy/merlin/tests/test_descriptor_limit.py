"""Real Linux subprocess coverage for inherited Ray/gRPC descriptor capacity."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

LAUNCHER = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform.startswith('linux'), 'requires Linux launch supervisor')
class DescriptorLimitTests(unittest.TestCase):
    def invoke(self, soft=1024, hard=None, requested=None):
        with tempfile.TemporaryDirectory(prefix='pear-nofile-test-') as temporary:
            base = Path(temporary)
            child = r'''
import json, os, pathlib, resource, sys
descriptors = [os.open(os.devnull, os.O_RDONLY) for _ in range(2048)]
pathlib.Path(sys.argv[1]).write_text(json.dumps({'limit': resource.getrlimit(resource.RLIMIT_NOFILE), 'opened': len(descriptors)}))
for descriptor in descriptors: os.close(descriptor)
'''
            caller = r'''
import resource, sys
sys.path.insert(0, sys.argv[1])
from launcher_control import Launch
hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1] if sys.argv[4] == 'None' else int(sys.argv[4])
resource.setrlimit(resource.RLIMIT_NOFILE, (int(sys.argv[3]), hard))
base = sys.argv[2]
raise SystemExit(Launch(base+'/root', base+'/run', 'nofile', 0, 1, base+'/node-lock').execute([sys.executable, '-c', sys.argv[5], base+'/child.json']))
'''
            env = os.environ.copy()
            env.pop('PEAR_NOFILE', None)
            if requested is not None:
                env['PEAR_NOFILE'] = str(requested)
            result = subprocess.run([sys.executable, '-c', caller, str(LAUNCHER), str(base),
                                     str(soft), str(hard), child], env=env, capture_output=True,
                                    text=True, timeout=15)
            record = json.loads((base/'child.json').read_text()) if (base/'child.json').exists() else None
            evidence = (base/'root/launches/nofile/failure.0')
            return result, record, evidence.read_text() if evidence.exists() else None

    def test_ssh_soft_limit_is_raised_and_child_can_open_over_1024_files(self):
        result, record, _ = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(record['limit'][0], 65536)
        self.assertEqual(record['opened'], 2048)

    def test_existing_higher_soft_limit_is_preserved(self):
        result, record, _ = self.invoke(soft=327680)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record['limit'][0], 327680)

    def test_low_hard_limit_refuses_to_start_child(self):
        result, record, evidence = self.invoke(hard=4096)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(record)
        self.assertIn('hard limit 4096', evidence)

    def test_unsafe_override_refuses_to_start_child(self):
        result, record, evidence = self.invoke(requested=1024)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(record)
        self.assertIn('PEAR_NOFILE', evidence)


if __name__ == '__main__':
    unittest.main()
