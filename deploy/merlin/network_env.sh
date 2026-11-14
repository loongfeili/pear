#!/usr/bin/env bash
# Source on EACH GPU worker. No default route and no rank-0 IP propagation.
export PEAR_NETWORK_INTERFACE=${PEAR_NETWORK_INTERFACE:-eth1}
PEAR_NODE_IP=$(python3 - "$PEAR_NETWORK_INTERFACE" <<'PYNET'
import ipaddress, json, subprocess, sys
interface = sys.argv[1]
rows = json.loads(subprocess.check_output(['ip', '-j', '-4', 'addr', 'show', 'dev', interface], text=True))
addresses = {item['local'] for row in rows for item in row.get('addr_info', [])
             if item.get('family') == 'inet' and item.get('scope') == 'global'}
if len(addresses) != 1:
    raise SystemExit(f'{interface} must have exactly one global-scope IPv4 address; found {len(addresses)}')
address = ipaddress.ip_address(addresses.pop())
if address.version != 4 or address.is_loopback or address.is_unspecified or address.is_multicast:
    raise SystemExit('Selected network address is not a usable IPv4 address')
print(address)
PYNET
) || { echo 'Cannot select this worker network address' >&2; return 2; }
export PEAR_NODE_IP
# This value belongs to this node's raylet/worker descendants. Never add it to
# driver-wide Ray runtime_env.env_vars, which would copy rank 0's IP everywhere.
export VLLM_HOST_IP=$PEAR_NODE_IP
export NCCL_SOCKET_IFNAME="=$PEAR_NETWORK_INTERFACE"
export GLOO_SOCKET_IFNAME=$PEAR_NETWORK_INTERFACE
export NCCL_SOCKET_FAMILY=AF_INET
