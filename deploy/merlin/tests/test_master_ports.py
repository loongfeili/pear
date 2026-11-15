import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from pear_merlin_ports import allocate_master_port


@unittest.skipUnless(sys.platform.startswith('linux'), 'requires Linux ephemeral-port metadata')
class MasterPortTests(unittest.TestCase):
    def test_concurrent_processes_keep_unique_leases_before_services_bind(self):
        with tempfile.TemporaryDirectory() as directory:
            code = ("import sys;sys.path.insert(0,sys.argv[1]);"
                    "from pear_merlin_ports import allocate_master_port;"
                    "print(allocate_master_port('127.0.0.1','parallel',sys.argv[2],(21000,21016))[1])")
            env = os.environ.copy()
            env.pop('PEAR_RESERVED_MASTER_PORTS', None)
            def allocate(_):
                result = subprocess.run([sys.executable, '-c', code, str(MODULE), directory],
                                        capture_output=True, text=True, env=env, check=True, timeout=10)
                return int(result.stdout.strip())
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                ports = list(pool.map(allocate, range(16)))
            self.assertEqual(len(set(ports)), 16)
            # No listener existed when leases were allocated; actual delayed
            # services can now bind every reserved port without colliding.
            listeners = []
            try:
                for port in ports:
                    listener = socket.socket()
                    listener.bind(('127.0.0.1', port))
                    listener.listen()
                    listeners.append(listener)
                with self.assertRaisesRegex(RuntimeError, 'No unused master ports'):
                    allocate_master_port('127.0.0.1', 'parallel', directory, (21000,21016))
            finally:
                for listener in listeners:listener.close()

    def test_existing_listener_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            listener.bind(('127.0.0.1', 21100))
            listener.listen()
            for _ in range(3):
                _, port = allocate_master_port('127.0.0.1', 'occupied', directory, (21100,21104))
                self.assertNotEqual(int(port), 21100)

    def test_ephemeral_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first, last = map(int, Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
            with self.assertRaisesRegex(ValueError, 'outside the OS ephemeral'):
                allocate_master_port('127.0.0.1', 'bad-range', directory, (first,last))

    def test_new_launch_can_reuse_released_service_ports(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(allocate_master_port('127.0.0.1','first',directory,(21200,21201))[1], '21200')
            self.assertEqual(allocate_master_port('127.0.0.1','next',directory,(21200,21201))[1], '21200')


if __name__ == '__main__':
    unittest.main()
