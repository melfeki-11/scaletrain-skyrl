"""Choose the job interface explicitly, including IPv6-only pod networks."""
import ipaddress
import json
import socket
import subprocess
from pathlib import Path


def routable(value):
    ip = ipaddress.ip_address(value.strip('[]'))
    if ip.is_link_local or ip.is_loopback or ip.is_unspecified or ip.is_multicast:
        raise ValueError(f'Unroutable cluster address: {value}')
    return str(ip)


def interface_addresses(interface):
    try:
        rows = json.loads(subprocess.check_output(['ip', '-j', 'addr', 'show', 'dev', interface], text=True))
        values = [a['local'] for row in rows for a in row['addr_info'] if a.get('scope') == 'global']
    except FileNotFoundError:
        values = []
        for line in Path('/proc/net/if_inet6').read_text().splitlines():
            raw, _, _, scope, _, name = line.split()
            if name == interface and scope == '00':
                values.append(str(ipaddress.IPv6Address(int(raw, 16))))
        import fcntl
        import struct
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            try:
                data = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack('256s', interface.encode()[:15]))
                values.append(socket.inet_ntoa(data[20:24]))
            except OSError:
                pass
    valid = []
    for value in values:
        try: valid.append(routable(value))
        except ValueError: pass
    if not valid:
        raise RuntimeError(f'No routable address on {interface}')
    return valid


def select_addresses(local, resolved):
    local = [routable(value) for value in local]
    resolved = [routable(value) for value in resolved]
    for version in (6, 4):
        ours = [value for value in local if ipaddress.ip_address(value).version == version]
        leaders = [value for value in resolved if ipaddress.ip_address(value).version == version]
        if ours and leaders:
            return ours[0], leaders[0]
    raise RuntimeError('Leader and job interface have no common routable address family')


def head_network(interface, port):
    """Single node: Ray head on the job interface's routable address (no rendezvous)."""
    addresses = interface_addresses(interface)
    local, _ = select_addresses(addresses, addresses)
    return local, endpoint(local, port)


def endpoint(host, port):
    host = routable(host)
    return f'[{host}]:{port}' if ':' in host else f'{host}:{port}'


def ray_start_args(rank, address, port, local_ip, cpus, gpus=8):
    args = ['/opt/skyrl-venv/bin/ray', 'start', f'--node-ip-address={routable(local_ip)}',
            f'--num-gpus={gpus}', f'--num-cpus={cpus}', '--disable-usage-stats']
    return args + (['--head', f'--port={port}'] if rank == 0 else [f'--address={address}'])
