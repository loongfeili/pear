"""Eligibility checks for an external recovery monitor; no training is started."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recover_owned_launch as recovery
from process_supervisor import ProcessIdentity, ProcessInfo


class RecoveryIdentityTests(unittest.TestCase):
    def setUp(self):
        self.identity = ProcessIdentity(42002, 12345, os.getuid())
        self.record = {'owner_id': 'a' * 32, 'supervisor_pid': 42001,
                       'root_pid': 42002, 'uid': os.getuid(),
                       'phase': 'running', 'requested_signal': None,
                       'owned_processes_seen': [self.identity.__dict__]}
        self.rank = {'launch_id': 'test-recovery', 'rank': 0, 'host': 'worker-0', 'pid': 42001}
        self.run = {'launch_id': 'test-recovery'}

    def validate(self, record=None, rank=None, lookup=None):
        current = ProcessInfo(self.identity, 1, 'S', 'python')
        with patch.object(recovery, '_process', side_effect=lookup or (lambda pid: current if pid == 42002 else None)):
            return recovery.validate_adoption(record or self.record, rank or self.rank, self.run,
                launch_id='test-recovery', rank=0, uid=os.getuid(), host='worker-0')

    def test_matching_birth_identity_is_required(self):
        root, identities = self.validate()
        self.assertEqual(root, self.identity)
        self.assertEqual(identities, [self.identity])

    def test_live_original_supervisor_is_never_taken_over(self):
        old = ProcessInfo(ProcessIdentity(42001, 999, os.getuid()), 1, 'S', 'python')
        with self.assertRaisesRegex(ValueError, 'still exists'):
            self.validate(lookup=lambda pid: old if pid == 42001 else None)

    def test_reused_root_pid_is_rejected(self):
        replacement = ProcessInfo(ProcessIdentity(42002, 12346, os.getuid()), 1, 'S', 'python')
        with self.assertRaisesRegex(ValueError, 'birth identity'):
            self.validate(lookup=lambda pid: replacement if pid == 42002 else None)

    def test_zombie_root_is_not_a_live_recovery(self):
        zombie = ProcessInfo(self.identity, 1, 'Z', 'python')
        with self.assertRaisesRegex(ValueError, 'no longer live'):
            self.validate(lookup=lambda pid: zombie if pid == 42002 else None)

    def test_wrong_rank_host_or_launch_cannot_be_adopted(self):
        for key, value in [('rank', 1), ('host', 'worker-1'), ('launch_id', 'other')]:
            with self.subTest(key=key):
                rank = {**self.rank, key: value}
                with self.assertRaisesRegex(ValueError, 'rank identity'):
                    self.validate(rank=rank)

    def test_foreign_uid_in_historical_record_is_rejected(self):
        record = copy.deepcopy(self.record)
        record['owned_processes_seen'][0]['uid'] += 1
        with self.assertRaisesRegex(ValueError, 'another user'):
            self.validate(record=record)

    def test_pending_stop_is_never_resumed(self):
        for change in ({'phase': 'terminating'}, {'requested_signal': 15}):
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, 'already stopping'):
                    self.validate(record={**self.record, **change})

    def test_output_mirror_preserves_exact_prefix_and_appends(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source', Path(directory) / 'target'
            source.mkdir(); target.mkdir()
            (source / 'train.log').write_bytes(b'first\n')
            recovery.mirror_outputs(source, target)
            with (source / 'train.log').open('ab') as stream:
                stream.write(b'second\n')
            recovery.mirror_outputs(source, target)
            self.assertEqual((target / 'train.log').read_bytes(), b'first\nsecond\n')
            self.assertEqual((source / 'train.log').read_bytes(), b'first\nsecond\n')

    def test_conflicting_existing_mirror_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source', Path(directory) / 'target'
            source.mkdir(); target.mkdir()
            (source / 'train.log').write_bytes(b'original')
            (target / 'train.log').write_bytes(b'conflict')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                recovery.mirror_outputs(source, target)
            self.assertEqual((target / 'train.log').read_bytes(), b'conflict')

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Requires Linux process identities and subreaping')
    def test_external_guard_cleans_owned_orphans_and_preserves_unrelated_process(self):
        module_dir = Path(recovery.__file__).parent
        with tempfile.TemporaryDirectory(prefix='pear-recovery-process-test-') as directory:
            script = r'''
import json, os, pathlib, signal, socket, subprocess, sys, time
sys.path.insert(0, MODULE_DIR)
import process_supervisor as ps
ps._subreaper(True)
base = pathlib.Path(TEST_DIR); launch = 'recover-fixture'; trial = '123'
root = base/'root'; control = root/'launches'/launch; run = root/'work_dir/runs'/launch
for p in [control/'processes/1', control/'ranks/1', run/'.pear-active']:
 p.mkdir(parents=True)
env = {**os.environ, 'ARNOLD_TRIAL_ID':trial,'ARNOLD_ID':'1',
       'PEAR_LAUNCH_ID':launch,'PEAR_PROCESS_OWNER':'a'*32}
old_code = """
import json,os,pathlib,subprocess,sys,time
children=[subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True) for _ in range(2)]
identities=[]
for p in children:
 t=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text();identities.append({'pid':p.pid,'start_ticks':int(t[t.rfind(')')+2:].split()[19]),'uid':os.getuid()})
pathlib.Path(os.environ['FIXTURE_IDENTITIES']).write_text(json.dumps({'supervisor_pid':os.getpid(),'identities':identities}))
"""
env['FIXTURE_IDENTITIES']=str(base/'identities.json')
old = subprocess.Popen([sys.executable,'-c',old_code],env=env)
old.wait(timeout=5)
data=json.loads((base/'identities.json').read_text()); identities=data['identities']
record={'owner_id':'a'*32,'supervisor_pid':data['supervisor_pid'],'root_pid':identities[0]['pid'],
        'uid':os.getuid(),'phase':'running','requested_signal':None,
        'owned_processes_seen':[identities[0]]}  # Second orphan missed the old snapshot.
(control/'processes/1/process_supervisor.json').write_text(json.dumps(record))
(control/'ranks/1/owner.json').write_text(json.dumps({'launch_id':launch,'rank':1,'host':socket.gethostname(),'pid':data['supervisor_pid']}))
owner={'launch_id':launch,'nonce':'run-owner'}
(run/'.pear-active/owner.json').write_text(json.dumps(owner))
unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
guardian=None
try:
 guardian=subprocess.Popen([sys.executable,str(pathlib.Path(MODULE_DIR)/'recover_owned_launch.py'),
  '--root',str(root),'--launch-id',launch,'--trial',trial,'--rank','1',
  '--recovery-root',str(base/'recovery'),'--original-release',str(base/'unused')],
  env={**os.environ,'ARNOLD_TRIAL_ID':trial,'ARNOLD_ID':'1'},stderr=subprocess.PIPE,text=True)
 ready=base/'recovery/rank-1/processes/process_supervisor.json'; deadline=time.monotonic()+8
 while time.monotonic()<deadline:
  if guardian.poll() is not None:raise RuntimeError(guardian.communicate()[1])
  try:
   state=json.loads(ready.read_text())
   if state.get('phase')=='recovered_running' and all(i['pid'] in [x['identity']['pid'] for x in state['current_processes']] for i in identities):break
  except (FileNotFoundError,json.JSONDecodeError):pass
  time.sleep(.02)
 else:raise RuntimeError('Recovery did not discover both owned orphans')
 (control/'failure.0').write_text('fixture failure')
 deadline=time.monotonic()+10; reaped=set()
 while guardian.poll() is None and time.monotonic()<deadline:
  for identity in identities:
   if identity['pid'] not in reaped:
    pid,_=os.waitpid(identity['pid'],os.WNOHANG)
    if pid:reaped.add(pid)
  time.sleep(.02)
 _,stderr=guardian.communicate(timeout=3)
 assert guardian.returncode==1,stderr
 assert all(ps._process(i['pid']) is None for i in identities)
 assert unrelated.poll() is None,'Unrelated process was signaled'
 assert json.loads((run/'.pear-active/owner.json').read_text())==owner
 final=json.loads((base/'recovery/rank-1/recovery.finished.json').read_text())
 assert final['original_locks_retained'] is True and final['success'] is False
 print('EXTERNAL_RECOVERY_CPU_INTEGRATION_PASSED')
finally:
 if guardian and guardian.poll() is None:guardian.terminate();guardian.wait(timeout=15)
 unrelated.terminate();unrelated.wait(timeout=3)
 for identity in identities:
  current=ps._process(identity['pid'])
  if current and current.identity.start_ticks==identity['start_ticks']:
   if current.state!='Z':os.kill(identity['pid'],signal.SIGKILL)
   try:os.waitpid(identity['pid'],0)
   except ChildProcessError:pass
'''
            script = script.replace('MODULE_DIR', repr(str(module_dir))).replace('TEST_DIR', repr(directory))
            result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('EXTERNAL_RECOVERY_CPU_INTEGRATION_PASSED', result.stdout)


if __name__ == '__main__':
    unittest.main()
