#!/usr/bin/env python3
"""Explicitly reroute this live invocation's diagnostic logs, without restarting actors."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import time


def reroute_worker(worker, directory, trial, launch):
    # Shipped by value to the existing async CPU actors. No model, sampler,
    # reward, or episode state is changed.
    import json
    import os
    from pathlib import Path
    import socket
    import time
    if os.environ.get('ARNOLD_TRIAL_ID') != trial or os.environ.get('PEAR_LAUNCH_ID') != launch:
        raise ValueError('Actor belongs to another training invocation')
    if not any(cls.__name__ == 'AgentLoopWorkerTQ' for cls in type(worker).__mro__):
        raise ValueError('Only PEAR agent-loop diagnostic writers may be rerouted')
    folder = Path(directory) / socket.gethostname()
    folder.mkdir(parents=True, exist_ok=True)
    if folder.stat().st_uid != os.getuid():
        raise ValueError('Diagnostic directory belongs to another user')
    tools = list(worker.tools.values()) if isinstance(worker.tools, dict) else list(worker.tools)
    previous = {}
    for index, tool in enumerate(tools):
        if tool.log_path.name != f'actions_{os.getpid()}.jsonl':
            raise ValueError('Unexpected diagnostic filename')
        previous[getattr(tool, 'name', str(index))] = str(tool.log_path)
    record = {'pid': os.getpid(), 'host': socket.gethostname(), 'trial': trial,
              'launch_id': launch, 'previous': previous, 'destination': str(folder), 'time': time.time()}
    # Validate all tools before changing any references.
    for tool in tools:
        tool.log_path = folder / f'actions_{os.getpid()}.jsonl'
    os.environ['PEAR_LOG_DIR'] = str(folder)
    (folder / f'reroute_{os.getpid()}.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--launch-id', required=True)
    args = parser.parse_args()
    if os.environ.get('ARNOLD_TRIAL_ID') != args.trial:
        parser.error('Wrong training trial')
    import ray
    ray.init(address='auto', log_to_driver=False)
    actors = [entry for entry in ray.util.list_named_actors(all_namespaces=True)
              if re.fullmatch(r'agent_loop_worker_\d+_[a-f0-9]{8}', entry['name'])]
    if not actors:
        raise ValueError('No live agent-loop actors were found')
    requests = [ray.get_actor(entry['name'], namespace=entry['namespace']).__ray_call__.remote(
                reroute_worker, args.directory, args.trial, args.launch_id) for entry in actors]
    records = ray.get(requests, timeout=120)
    path = Path(args.directory) / 'reroute_manifest.json'
    path.write_text(json.dumps({'trial': args.trial, 'launch_id': args.launch_id,
                     'actors': records, 'count': len(records), 'finished_at': time.time()}, indent=2) + '\n')
    print('PEAR_DIAGNOSTIC_REROUTE_COMPLETE', len(records), str(path), flush=True)
    ray.shutdown()


if __name__ == '__main__':
    main()
