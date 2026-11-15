"""Deployment-only veRL master-port allocator, outside OS ephemeral ports."""
import errno
import os
from pathlib import Path
import random
import re
import socket


def allocate_master_port(address, launch_id, directory=None, port_range=None):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', launch_id):
        raise ValueError('A valid PEAR_LAUNCH_ID is required for port ownership')
    first, last = map(int, (port_range or (21000, 30000)))
    ephemeral_first, ephemeral_last = map(int, Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
    if not 1024 < first < last <= 65536 or not (last <= ephemeral_first or first > ephemeral_last):
        raise ValueError('Master ports must be unprivileged and outside the OS ephemeral range')
    leases = Path(directory or f'/var/tmp/pear-master-ports-{os.getuid()}') / launch_id
    leases.mkdir(parents=True, mode=0o700, exist_ok=True)
    if leases.stat().st_uid != os.getuid():
        raise RuntimeError('Port lease directory belongs to another user')
    ports = list(range(first, last))
    random.SystemRandom().shuffle(ports)
    family = socket.AF_INET6 if ':' in address else socket.AF_INET
    for port in ports:
        claim = leases / str(port)
        try:
            claim.mkdir(mode=0o700)
        except FileExistsError:
            continue
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.bind((address, port))
        except OSError as error:
            claim.rmdir()  # Only this process can own this newly created claim.
            if error.errno == errno.EADDRINUSE:
                continue
            raise
        # Keep the claim after closing the probe: other concurrent worker groups
        # cannot reuse the port before rank zero eventually binds TCPStore.
        # The OS cannot allocate it to an outbound ephemeral connection.
        return address, str(port)
    raise RuntimeError(f'No unused master ports in [{first}, {last}) for this launch')


def install():
    launch = os.environ.get('PEAR_LAUNCH_ID')
    if not launch:
        raise RuntimeError('Port allocation plugin requires the real launch identity')
    import ray
    import verl.single_controller.ray.base as base
    if getattr(base, '_pear_reserved_master_ports', False):
        return

    @ray.remote
    def get_master_addr_port(master_port_range=None):
        return allocate_master_port(ray.util.get_node_ip_address().strip('[]'),
                                    os.environ['PEAR_LAUNCH_ID'], port_range=master_port_range)

    base.get_master_addr_port = get_master_addr_port
    base._pear_reserved_master_ports = True


# veRL's documented external-module hook runs in the controller and its Ray
# workers. Normal imports in CPU tests do not install Ray or mutate veRL.
if os.environ.get('PEAR_RESERVED_MASTER_PORTS') == '1':
    install()
