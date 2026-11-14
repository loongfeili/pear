#!/usr/bin/env python3
"""The two supported H20 layouts; algorithm and per-node capacities stay fixed."""

import argparse
from dataclasses import dataclass
import json
import os


@dataclass(frozen=True)
class Topology:
    total_gpus: int
    trainer_nodes: int
    rollout_nodes: int
    gpus_per_node: int = 8
    libero_servers_per_node: int = 16

    @property
    def nodes(self):
        return self.trainer_nodes + self.rollout_nodes

    @property
    def expected_resources(self):
        return {'GPU': self.total_gpus,
                'trainer': self.trainer_nodes * self.gpus_per_node,
                'rollout': self.rollout_nodes * self.gpus_per_node,
                'libero': self.rollout_nodes * self.libero_servers_per_node}

    def check_rank(self, rank):
        if not 0 <= rank < self.nodes:
            raise ValueError(f'Worker rank must be in 0..{self.nodes - 1}')
        return 'trainer' if rank < self.trainer_nodes else 'rollout'

    def resource_errors(self, nodes, resources):
        errors = []
        if len(nodes) != self.nodes:
            errors.append(f'Expected {self.nodes} nodes, found {len(nodes)}')
        if len({node['NodeManagerAddress'] for node in nodes}) != self.nodes:
            errors.append('Expected a distinct node IP for every GPU worker')
        for name, expected in self.expected_resources.items():
            if resources.get(name, 0) != expected:
                errors.append(f'Expected {name}={expected}, found {resources.get(name, 0)}')
        for node in nodes:
            offered = node['Resources']
            if offered.get('GPU', 0) != self.gpus_per_node:
                errors.append(f"Node {node['NodeManagerAddress']} does not offer eight GPUs")
            trainer = offered.get('trainer', 0) == self.gpus_per_node
            rollout = offered.get('rollout', 0) == self.gpus_per_node
            if trainer == rollout:
                errors.append(f"Node {node['NodeManagerAddress']} must have exactly one GPU role")
            if offered.get('libero', 0) != (self.libero_servers_per_node if rollout else 0):
                errors.append(f"Node {node['NodeManagerAddress']} has incorrect LIBERO slots")
        return errors

    def as_dict(self):
        replicas = self.expected_resources['rollout'] // 4
        return {
            'nodes': self.nodes, 'gpus_per_node': self.gpus_per_node,
            'total_gpus': self.total_gpus, 'trainer_nodes': self.trainer_nodes,
            'rollout_nodes': self.rollout_nodes,
            'trainer_world_size': self.expected_resources['trainer'],
            'dp_size': self.expected_resources['trainer'] // 8,
            'fsdp_size': 8, 'sp': 8, 'rollout_tp': 4,
            'rollout_replicas': replicas, 'batch_groups': 128,
            'rollouts_per_group': 8, 'trajectories_per_step': 1024,
            'max_model_len': 164096, 'tokens_per_gpu': 20512,
            'sessions_per_replica': 64, 'max_inflight_sessions': replicas * 64,
            # base.yaml: two decoding slots/session, eight slots/agent worker.
            'agent_workers': replicas * 64 * 2 // 8,
            'libero_server_slots': self.expected_resources['libero'],
            'lmcache_gb_per_tier': 600, 'lmcache_tiers_per_rollout_node': 2,
        }

    def shell_environment(self):
        # All emitted values are validated integers, never user-provided code.
        values = {'PEAR_TOTAL_GPUS': self.total_gpus, 'PEAR_NNODES': self.nodes,
                  'TRAINER_NNODES': self.trainer_nodes, 'ROLLOUT_NNODES': self.rollout_nodes,
                  'LIBERO_SERVERS_PER_NODE': self.libero_servers_per_node}
        return '\n'.join(f'export {name}={value}' for name, value in values.items())


def topology_for_gpus(value):
    if str(value) == '64':
        return Topology(64, 2, 6)
    if str(value) == '128':
        return Topology(128, 4, 12)
    raise ValueError('PEAR_TOTAL_GPUS must be exactly 64 or 128')


def topology_from_environment(environment=None):
    environment = os.environ if environment is None else environment
    return topology_for_gpus(environment.get('PEAR_TOTAL_GPUS', '64'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default=os.environ.get('PEAR_TOTAL_GPUS', '64'))
    parser.add_argument('--shell', action='store_true')
    args = parser.parse_args()
    try:
        topology = topology_for_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))
    print(topology.shell_environment() if args.shell else json.dumps(topology.as_dict(), sort_keys=True))


if __name__ == '__main__':
    main()
