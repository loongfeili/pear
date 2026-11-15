#!/usr/bin/env python3
"""Repair the stale crashed label of a verified, live recovered Tracking run.

Uses the pinned internal SDK client without initializing another producer or
writing any scalar points. Never repairs a finished/failed run automatically.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

from process_supervisor import ProcessIdentity, _process
from tracking_bridge import merlin_identity, publish


def check_run(record, expected):
    if not isinstance(record, dict):
        raise ValueError('Tracking lookup failed')
    fields = {'ProjectId': expected['project_id'], 'RunId': expected['run_id'],
              'Name': expected['launch_id'], 'OuterType': 'merlin', 'OuterId': expected['job_id']}
    if any(record.get(k) != v for k, v in fields.items()) or str(record.get('OuterInfo', {}).get('arnold_trial_id')) != expected['trial_id']:
        raise ValueError('Tracking record belongs to a different invocation')
    if record.get('Status') not in ('crashed', 'running'):
        raise ValueError('Refusing to revive a finished or failed run')


def restore_status(client, expected):
    def get():
        record = client.get_run(run_id=expected['run_id'], project_id=expected['project_id'])
        check_run(record, expected)
        return record
    before = get()
    client.update_run_heartbeat_time(expected['project_id'], expected['run_id'])
    if before['Status'] == 'crashed':
        client.update_run_status(expected['project_id'], expected['run_id'], 'running')
    # Status reads can briefly lag an acknowledged write. SDK wrappers also
    # swallow errors, so require a bounded server readback rather than trusting
    # the update call's return value.
    deadline = time.monotonic() + 30
    while True:
        after = get()
        if after['Status'] == 'running':
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Tracking status repair was not acknowledged')
        time.sleep(1)
    return {'before': {k: before.get(k) for k in ('Status', 'Step', 'Runtime', 'UpdateTime')},
            'after': {k: after.get(k) for k in ('Status', 'Step', 'Runtime', 'UpdateTime')},
            'scalar_points_written': 0}


def require_live(identity):
    info = _process(identity['pid'])
    if info is None or info.identity != ProcessIdentity(**identity) or info.state == 'Z':
        raise ValueError('Recovered process identity is no longer live')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--recovery-rank', type=Path, required=True)
    args = parser.parse_args()
    identity = merlin_identity(os.environ)
    control, recovery = args.control.resolve(), args.recovery_rank.resolve()
    status = json.loads((control / 'tracking.json').read_text())
    adoption = json.loads((recovery / 'adoption.json').read_text())
    journal = json.loads((recovery / 'processes/process_supervisor.json').read_text())
    owner = json.loads((control / 'tracking-bridge.claim/owner.json').read_text())
    if adoption['trial'] != identity['trial_id'] or adoption['launch_id'] != status['launch_id'] or adoption['rank'] != 0:
        raise ValueError('Wrong recovery invocation')
    if status['association']['job_id'] != identity['job_id'] or status['association']['trial_id'] != identity['trial_id']:
        raise ValueError('Wrong Tracking association')
    now = time.time()
    if status['state'] != 'streaming' or not 0 <= now - status['updated_at'] < 120:
        raise ValueError('Require fresh live metric streaming before repairing status')
    if journal['phase'] != 'recovered_running' or not 0 <= now - journal['updated_at'] < 120:
        raise ValueError('Recovery guardian is not healthy')
    if owner['launch_id'] != adoption['launch_id'] or owner['pid'] not in [p['pid'] for p in journal['owned_processes_seen']]:
        raise ValueError('Tracking producer ownership differs')
    for info in [adoption['root_identity'], next(i for i in journal['owned_processes_seen'] if i['pid'] == owner['pid'])]:
        require_live(info)
    if (control / 'tracking.stop.json').exists() or list(recovery.parent.glob('failure.*')):
        raise ValueError('Recovered run is stopping or failed')
    if importlib.metadata.version('byted-wandb') != '0.13.98' or os.environ.get('WANDB_OFFICIAL') == '1':
        raise ValueError('Require the reviewed internal SDK')
    import wandb
    if not getattr(wandb, '_IS_TRACKING', False):
        raise ValueError('Require internal Tracking')
    from wandb.sdk.internal.tracking_internal_api import Api
    expected = {k: status[k] for k in ('project_id', 'run_id', 'launch_id')}
    expected.update(job_id=identity['job_id'], trial_id=identity['trial_id'])
    audit = {'expected': expected, 'started_at': now, 'mode': 'metadata_only'}
    publish(control / 'tracking_status_repair.json', audit)
    audit.update(restore_status(Api().client, expected), finished_at=time.time())
    publish(control / 'tracking_status_repair.json', audit)
    print('PEAR_TRACKING_STATUS_REPAIRED=' + json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
