"""CPU-only topology and entry-point checks; no Ray/GPU/training is launched."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))
from topology import topology_for_gpus, topology_from_environment


def cluster_nodes(topology):
    result = []
    for rank in range(topology.nodes):
        role = topology.check_rank(rank)
        resources = {'GPU': 8, role: 8}
        if role == 'rollout':
            resources['libero'] = 16
        result.append({'NodeID': f'node-{rank}', 'NodeManagerAddress': f'10.1.0.{rank + 1}',
                       'Resources': resources, 'Alive': True})
    return result


class TopologyTests(unittest.TestCase):
    def test_supported_layouts_preserve_algorithm_and_per_node_limits(self):
        for gpus, nodes, trainers, rollouts, replicas, slots in (
                (64, 8, 2, 6, 12, 96), (128, 16, 4, 12, 24, 192)):
            with self.subTest(gpus=gpus):
                topology = topology_for_gpus(gpus)
                record = topology.as_dict()
                self.assertEqual((topology.nodes, topology.trainer_nodes, topology.rollout_nodes),
                                 (nodes, trainers, rollouts))
                self.assertEqual(record['rollout_replicas'], replicas)
                self.assertEqual(record['libero_server_slots'], slots)
                self.assertEqual(record['max_inflight_sessions'], replicas * 64)
                self.assertEqual(record['agent_workers'], replicas * 16)
                self.assertEqual([record[k] for k in ('batch_groups', 'rollouts_per_group',
                                                     'rollout_tp', 'fsdp_size', 'sp',
                                                     'lmcache_gb_per_tier')], [128, 8, 4, 8, 8, 600])
                self.assertEqual(topology.check_rank(trainers - 1), 'trainer')
                self.assertEqual(topology.check_rank(trainers), 'rollout')
                self.assertEqual(topology.check_rank(nodes - 1), 'rollout')
                with self.assertRaises(ValueError):
                    topology.check_rank(nodes)
                self.assertEqual(topology.resource_errors(cluster_nodes(topology),
                                                           topology.expected_resources), [])

    def test_default_stays_64_and_unreviewed_layouts_are_rejected(self):
        self.assertEqual(topology_from_environment({}).total_gpus, 64)
        self.assertEqual(topology_from_environment({'PEAR_TOTAL_GPUS': '128'}).total_gpus, 128)
        for invalid in ('256', '32', '129', '', '128; touch unsafe', '-128'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                topology_for_gpus(invalid)

    def test_incomplete_or_misassigned_128_cluster_is_rejected(self):
        topology = topology_for_gpus(128)
        nodes = cluster_nodes(topology)
        self.assertTrue(topology.resource_errors(nodes[:-1], topology.expected_resources))
        wrong_resources = dict(topology.expected_resources, libero=96)
        self.assertTrue(topology.resource_errors(nodes, wrong_resources))
        nodes[3]['Resources'] = {'GPU': 8, 'trainer': 8, 'rollout': 8, 'libero': 16}
        self.assertTrue(topology.resource_errors(nodes, topology.expected_resources))

    def test_public_entry_passes_128_node_count_and_rejects_out_of_range_rank(self):
        with tempfile.TemporaryDirectory(prefix='pear-entry-topology-') as temporary:
            directory = Path(temporary)
            for name in ('merlin_stage1_entry.sh', 'topology.py'):
                shutil.copy2(DEPLOY / name, directory / name)
            # The only invoked controller is a harmless argument recorder. This
            # tests the real public shell entry without claiming locks or GPUs.
            (directory / 'launcher_control.py').write_text(
                'import json,os,sys\n'
                'print(json.dumps({"args":sys.argv[1:],"nodes":os.environ["PEAR_NNODES"],'
                '"trainers":os.environ["TRAINER_NNODES"],"rollouts":os.environ["ROLLOUT_NNODES"]}))\n')
            environment = dict(os.environ, ARNOLD_ROLE='worker', ARNOLD_TRIAL_ID='123',
                               PEAR_ROOT=str(directory / 'root'))
            for key in ('PEAR_LAUNCH_ID', 'PEAR_RUN', 'PEAR_RESUME', 'PEAR_TOTAL_GPUS'):
                environment.pop(key, None)
            for total, rank, expected_nodes, expected_roles in (
                    ('64', '7', '8', ('2', '6')), ('128', '15', '16', ('4', '12'))):
                with self.subTest(total=total):
                    environment.update(PEAR_TOTAL_GPUS=total, ARNOLD_ID=rank)
                    result = subprocess.run(['bash', str(directory / 'merlin_stage1_entry.sh')],
                                            env=environment, text=True, capture_output=True, timeout=5)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    record = json.loads(result.stdout)
                    self.assertEqual(record['nodes'], expected_nodes)
                    self.assertEqual((record['trainers'], record['rollouts']), expected_roles)
                    args = record['args']
                    self.assertEqual(args[args.index('--nodes') + 1], expected_nodes)
                    self.assertEqual(args[args.index('--launch-id') + 1], f'merlin-trial-123-stage1-{total}')
                    environment['ARNOLD_ID'] = expected_nodes
                    rejected = subprocess.run(['bash', str(directory / 'merlin_stage1_entry.sh')],
                                              env=environment, text=True, capture_output=True, timeout=5)
                    self.assertEqual(rejected.returncode, 2)
                    self.assertEqual(rejected.stdout, '')

    def test_validator_schedules_one_probe_per_node_and_releases_selected_capacity(self):
        # Import the real driver against a tiny Ray protocol double. This checks
        # its world size, placement and release gate without importing real Ray.
        for gpus in (64, 128):
            with self.subTest(gpus=gpus):
                topology = topology_for_gpus(gpus)
                scheduled, killed, calls = [], [], []
                ray = types.ModuleType('ray')
                ray.__path__ = []
                ray.remote = lambda **kwargs: lambda cls: cls
                ray.init = lambda **kwargs: None
                ray.shutdown = lambda: calls.append('shutdown')
                ray.nodes = lambda: cluster_nodes(topology)
                ray.cluster_resources = lambda: topology.expected_resources
                ray.available_resources = lambda: {'GPU': topology.total_gpus - len(scheduled) + len(killed)}
                ray.get = lambda value, **kwargs: value
                ray.kill = lambda actor, **kwargs: killed.append(actor)
                strategy = types.ModuleType('ray.util.scheduling_strategies')
                strategy.NodeAffinitySchedulingStrategy = lambda node, soft: (node, soft)
                util = types.ModuleType('ray.util')
                util.__path__ = []
                with patch.dict(sys.modules, {'ray': ray, 'ray.util': util,
                                               'ray.util.scheduling_strategies': strategy}):
                    spec = importlib.util.spec_from_file_location('pear_validator_test', DEPLOY / 'validate_cluster64.py')
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    class Probes:
                        @staticmethod
                        def options(**options):
                            def remote(rank, world_size):
                                self.assertEqual(world_size, topology.nodes)
                                expected_node = sorted(cluster_nodes(topology), key=lambda n: n['NodeManagerAddress'])[rank]
                                self.assertEqual(options['scheduling_strategy'], (expected_node['NodeID'], False))
                                self.assertNotIn('VLLM_HOST_IP', options['runtime_env']['env_vars'])
                                actor = types.SimpleNamespace(
                                    open_store=types.SimpleNamespace(remote=lambda: ('10.1.0.1', 12345)),
                                    reduce=types.SimpleNamespace(remote=lambda head, port: {'rank': rank, 'passed': True}))
                                scheduled.append(actor)
                                return actor
                            return types.SimpleNamespace(remote=remote)

                    module.NodeProbe = Probes
                    output = io.StringIO()
                    with patch.object(sys, 'argv', ['validator', '--address', 'head:6379', '--gpus', str(gpus)]), \
                            contextlib.redirect_stdout(output):
                        module.main()
                    result = json.loads(output.getvalue())
                    self.assertEqual(len(scheduled), topology.nodes)
                    self.assertEqual(len(killed), topology.nodes)
                    self.assertEqual(result['total_gpus'], gpus)
                    self.assertEqual(result['probe_gpus'], topology.nodes)
                    self.assertTrue(result['all_gpu_slots_released'])
                    self.assertEqual(calls, ['shutdown'])

    def test_actual_probe_uses_selected_collective_world_size_and_sum(self):
        for world_size, expected_sum in ((8, 36.0), (16, 136.0)):
            with self.subTest(world_size=world_size):
                ray = types.ModuleType('ray')
                ray.__path__ = []
                ray.remote = lambda **kwargs: lambda cls: cls
                ray.util = types.SimpleNamespace(get_node_ip_address=lambda: '10.1.0.2')
                strategy = types.ModuleType('ray.util.scheduling_strategies')
                strategy.NodeAffinitySchedulingStrategy = lambda *args, **kwargs: None
                torch = types.ModuleType('torch')
                torch.__path__ = []
                torch.cuda = types.SimpleNamespace(device_count=lambda: 1, set_device=lambda device: None)

                class Tensor:
                    def __init__(self, values, **kwargs):
                        self.value = values[0]

                    def item(self):
                        return self.value

                torch.tensor = Tensor
                dist = types.ModuleType('torch.distributed')
                torch.distributed = dist
                state = {}
                def store(*args, **kwargs):
                    state['store_world_size'] = kwargs['world_size']
                    return object()
                def initialize(**kwargs):
                    state['world_size'] = kwargs['world_size']
                    state['initialized'] = True
                def reduce(tensor, **kwargs):
                    size = state['world_size']
                    tensor.value = size * (size + 1) / 2
                dist.TCPStore = store
                dist.init_process_group = initialize
                dist.all_reduce = reduce
                dist.new_group = lambda **kwargs: object()
                dist.is_initialized = lambda: state.get('initialized', False)
                dist.destroy_process_group = lambda: state.update(initialized=False)
                environment = {'PEAR_NETWORK_INTERFACE': 'eth1', 'VLLM_HOST_IP': '10.1.0.2',
                               'NCCL_SOCKET_IFNAME': '=eth1', 'NCCL_SOCKET_FAMILY': 'AF_INET',
                               'GLOO_SOCKET_IFNAME': 'eth1', 'PEAR_RUNTIME_ENV_PROBE': '1'}
                records = json.dumps([{'addr_info': [{'local': '10.1.0.2', 'family': 'inet', 'scope': 'global'}]}]).encode()
                modules = {'ray': ray, 'ray.util': types.ModuleType('ray.util'),
                           'ray.util.scheduling_strategies': strategy, 'torch': torch,
                           'torch.distributed': dist}
                with patch.dict(sys.modules, modules), patch.dict(os.environ, environment), \
                        patch('subprocess.check_output', return_value=records):
                    spec = importlib.util.spec_from_file_location('pear_probe_test', DEPLOY / 'validate_cluster64.py')
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    probe = module.NodeProbe(1, world_size)
                    result = probe.reduce('10.1.0.1', 12345)
                self.assertEqual(state['world_size'], world_size)
                self.assertEqual(state['store_world_size'], world_size)
                self.assertEqual(result['nccl_sum'], expected_sum)
                self.assertEqual(result['gloo_sum'], expected_sum)
                self.assertFalse(state['initialized'])


if __name__ == '__main__':
    unittest.main()
