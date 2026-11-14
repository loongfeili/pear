"""Bounded eight-node communication check before starting PEAR training."""
import argparse
from datetime import timedelta
import ipaddress
import json
import os
import socket
import subprocess
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=1, num_gpus=1, max_restarts=0)
class NodeProbe:
    def __init__(self, rank):
        import torch
        self.rank = rank
        self.store = None
        self.ip = ray.util.get_node_ip_address()
        interface = os.environ['PEAR_NETWORK_INTERFACE']
        records = json.loads(subprocess.check_output(['ip', '-j', '-4', 'addr', 'show', 'dev', interface]))
        ips = [a['local'] for record in records for a in record['addr_info']
               if a['family'] == 'inet' and a['scope'] == 'global']
        assert ips == [self.ip], (ips, self.ip)
        assert ipaddress.ip_address(self.ip).version == 4
        assert os.environ.get('VLLM_HOST_IP') == self.ip, ('VLLM_HOST_IP', os.environ.get('VLLM_HOST_IP'), self.ip)
        for key, value in [('NCCL_SOCKET_IFNAME', '=' + interface), ('NCCL_SOCKET_FAMILY', 'AF_INET'),
                           ('GLOO_SOCKET_IFNAME', interface), ('PEAR64_RUNTIME_ENV_PROBE', '1')]:
            assert os.environ.get(key) == value, (key, os.environ.get(key))
        assert torch.cuda.device_count() == 1
        torch.cuda.set_device(0)

    def open_store(self):
        import torch.distributed as dist
        assert self.rank == 0
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((self.ip, 0))
            port = probe.getsockname()[1]
        self.store = dist.TCPStore(self.ip, port, world_size=8, is_master=True,
                                   timeout=timedelta(seconds=120), wait_for_workers=False)
        return self.ip, port

    def reduce(self, head, port):
        import torch
        import torch.distributed as dist
        if self.store is None:
            self.store = dist.TCPStore(head, port, world_size=8, is_master=False,
                                       timeout=timedelta(seconds=120))
        try:
            dist.init_process_group(backend='nccl', store=self.store, rank=self.rank,
                                    world_size=8, timeout=timedelta(seconds=120))
            gpu = torch.tensor([self.rank + 1.0], device='cuda')
            dist.all_reduce(gpu)
            gpu_sum = gpu.item()
            assert gpu_sum == 36.0, gpu_sum
            cpu_group = dist.new_group(backend='gloo', timeout=timedelta(seconds=120))
            cpu = torch.tensor([self.rank + 1.0])
            dist.all_reduce(cpu, group=cpu_group)
            cpu_sum = cpu.item()
            assert cpu_sum == 36.0, cpu_sum
            return {'rank': self.rank, 'ip': self.ip, 'runtime_env_verified': True,
                    'nccl_sum': gpu_sum, 'gloo_sum': cpu_sum, 'passed': True}
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            self.store = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', required=True)
    args = parser.parse_args()
    ray.init(address=args.address, log_to_driver=True)
    actors = []
    started = time.monotonic()
    result = None
    released = False
    try:
        nodes = sorted((n for n in ray.nodes() if n['Alive']), key=lambda n: n['NodeManagerAddress'])
        assert len(nodes) == 8, len(nodes)
        assert len({n['NodeManagerAddress'] for n in nodes}) == 8
        assert sum(n['Resources'].get('GPU', 0) for n in nodes) == 64
        interface = os.environ.get('PEAR_NETWORK_INTERFACE', 'eth1')
        environment = {'PEAR64_RUNTIME_ENV_PROBE': '1', 'PEAR_NETWORK_INTERFACE': interface,
                       'NCCL_SOCKET_IFNAME': '=' + interface,
                       'NCCL_SOCKET_FAMILY': 'AF_INET', 'GLOO_SOCKET_IFNAME': interface}
        for key in ('LD_LIBRARY_PATH', 'PEAR_CUDA_COMPAT_DIR'):
            if key in os.environ:
                environment[key] = os.environ[key]
        # Deliberately do not broadcast the driver's VLLM_HOST_IP to other nodes.
        for rank, node in enumerate(nodes):
            actors.append(NodeProbe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node['NodeID'], soft=False),
                runtime_env={'env_vars': environment}).remote(rank))
        head, port = ray.get(actors[0].open_store.remote(), timeout=120)
        checks = ray.get([a.reduce.remote(head, port) for a in actors], timeout=240)
        assert len(checks) == 8 and all(c['passed'] for c in checks)
        result = {'passed': True, 'scope': 'eight-node runtime_env and small NCCL/Gloo collectives',
                  'seconds': time.monotonic() - started, 'checks': checks}
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        # Let resource accounting catch up before the full training pools allocate GPUs.
        deadline = time.monotonic() + 30
        while actors and ray.available_resources().get('GPU', 0) < 64 and time.monotonic() < deadline:
            time.sleep(0.5)
        released = ray.available_resources().get('GPU', 0) == 64
        ray.shutdown()
    assert released, 'Probe actors did not release all 64 GPU slots before training'
    result['all_gpu_slots_released'] = True
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
