"""Cluster probe before a multi-node training run (ported from q35/probe.py and extended).

Per node: hostname, /dev/shm, cgroup memory limit, GPU count, EFA evidence (/dev/infiniband, aws-ofi-nccl plugin, fi_info).
Across nodes: distinct physical hosts, bidirectional TCP to a listener on every node, NCCL all-reduce correctness, and 1 GiB bf16
all-gather bus bandwidth intra-node and cross-node, with the NCCL transport read from NCCL_DEBUG=INFO (NET/OFI|efa, NET/IB,
NET/Socket). EFA absence never fails the probe: it is returned (efa_available, transport) so plan/monitor can warn.
CPU mode (dry runs): node facts + distinct hosts + TCP only.

    python -m strl.probe --nodes 2 --gpus-per-node 8 [--cpu] [--output artifacts/network.json]
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from .network import routable

GIB = 1 << 30
OFI_PLUGIN_GLOBS = ('/opt/amazon/ofi-nccl/lib*/libnccl-net*.so*', '/opt/aws-ofi-nccl/lib*/libnccl-net*.so*',
                    '/opt/amazon/aws-ofi-nccl/lib*/libnccl-net*.so*', '/usr/local/lib/libnccl-net*.so*',
                    '/usr/lib/x86_64-linux-gnu/libnccl-net*.so*', '/opt/amazon/efa/lib*/libnccl-net*.so*')
EFA_ENV = {'FI_PROVIDER': 'efa', 'FI_EFA_USE_DEVICE_RDMA': '1'}


def efa_facts(root: str = '/', environ=None, which=shutil.which, run=subprocess.run) -> dict:
    """Evidence that this container can use EFA: verbs devices, the aws-ofi-nccl NCCL net plugin, and fi_info's efa provider."""
    environ = os.environ if environ is None else environ
    base = Path(root)
    devices = sorted(Path(p).name for p in glob.glob(str(base / 'dev/infiniband/uverbs*')))
    patterns = [str(base / p.lstrip('/')) for p in OFI_PLUGIN_GLOBS]
    patterns += [str(Path(d) / 'libnccl-net*.so*') for d in environ.get('LD_LIBRARY_PATH', '').split(':') if d]
    plugins = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    fi_info = which('fi_info') or next((p for p in glob.glob(str(base / 'opt/amazon/efa/bin/fi_info'))), None)
    provider = None
    if fi_info:
        try:
            provider = run([fi_info, '-p', 'efa', '-t', 'FI_EP_RDM'], capture_output=True, text=True, timeout=15).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            provider = False
    return {'efa_available': bool(devices) and bool(plugins) and provider is not False, 'devices': devices, 'ofi_plugins': plugins,
            'fi_info': fi_info, 'fi_info_efa_provider': provider}


def configure_efa(environ=None, facts: dict | None = None) -> dict:
    """Set FI_PROVIDER=efa / FI_EFA_USE_DEVICE_RDMA=1 (NCCL_PROTO left at its default) when EFA is usable; never fails."""
    environ = os.environ if environ is None else environ
    facts = facts if facts is not None else efa_facts(environ=environ)
    applied = {}
    if facts['efa_available']:
        for key, value in EFA_ENV.items():
            if key not in environ:
                environ[key] = applied[key] = value
    return dict(facts, applied=applied)


def classify_transport(log: str) -> str:
    """NCCL_DEBUG=INFO text -> 'efa' | 'ofi' | 'ib' | 'socket' | 'nvlink' (intra-node only) | 'unknown'."""
    lines = [line for line in log.splitlines() if 'NCCL INFO' in line or 'NET/' in line]
    text = '\n'.join(lines)
    chosen = [line for line in lines if 'Using network' in line]
    probe = '\n'.join(chosen) or text
    if 'Libfabric' in probe or 'NET/OFI' in probe or 'OFI' in probe:
        return 'efa' if 'efa' in text.lower() else 'ofi'
    if 'Using network IB' in probe or 'NET/IB' in probe:
        return 'ib'
    if 'Using network Socket' in probe or 'NET/Socket' in probe:
        return 'socket'
    if any(marker in text for marker in ('via P2P', 'P2P/', 'NVLS', 'via SHM')):
        return 'nvlink'
    return 'unknown'


def bus_bandwidth(nbytes: int, seconds: float, world: int) -> dict:
    """All-gather of an ``nbytes`` output: algorithm bandwidth and bus bandwidth (x (n-1)/n, comparable to nccl-tests)."""
    algbw = nbytes / seconds / 1e9 if seconds > 0 else 0.0
    return {'algbw_gbps': round(algbw, 2), 'busbw_gbps': round(algbw * (world - 1) / world, 2) if world > 1 else round(algbw, 2),
            'seconds': round(seconds, 4), 'bytes': nbytes, 'world': world}


def shm_bytes(path: str = '/dev/shm') -> int | None:
    try:
        stat = os.statvfs(path)
        return stat.f_frsize * stat.f_blocks
    except OSError:
        return None


def cgroup_memory_limit(v2: str = '/sys/fs/cgroup/memory.max', v1: str = '/sys/fs/cgroup/memory/memory.limit_in_bytes') -> int | None:
    for path in (v2, v1):
        try:
            value = Path(path).read_text().strip()
        except OSError:
            continue
        if value == 'max':
            return None
        try:
            limit = int(value)
        except ValueError:
            continue
        return None if limit >= 1 << 60 else limit   # cgroup v1 reports "unlimited" as ~2^63
    return None


def gpu_count(run=subprocess.run) -> int:
    try:
        out = run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=30)
        return sum(line.startswith('GPU ') for line in out.stdout.splitlines()) if out.returncode == 0 else 0
    except (OSError, subprocess.TimeoutExpired):
        return 0


def node_facts() -> dict:
    return {'hostname': socket.gethostname(), 'shm_bytes': shm_bytes(), 'cgroup_memory_limit': cgroup_memory_limit(),
            'gpus': gpu_count(), 'cpus': os.cpu_count(), 'efa': efa_facts(),
            'fi_provider': os.environ.get('FI_PROVIDER'), 'nccl_socket_ifname': os.environ.get('NCCL_SOCKET_IFNAME')}


class Rank:
    """One probe actor per GPU (GPU mode) or per node (CPU mode)."""

    def location(self):
        import ray
        return {'hostname': socket.gethostname(), 'node_ip': routable(ray.util.get_node_ip_address()),
                'node_id': ray.get_runtime_context().get_node_id(), 'gpu': ray.get_gpu_ids()}

    def facts(self):
        return node_facts()

    def listen(self):
        import ray
        import threading
        ip = routable(ray.util.get_node_ip_address())
        self.listener = socket.socket(socket.AF_INET6 if ':' in ip else socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind((ip, 0))
        self.listener.listen(64)
        self.listener.settimeout(1)
        self.listening = True

        def accept():
            while self.listening:
                try:
                    conn, _ = self.listener.accept()
                    with conn:
                        conn.sendall(b'OK')
                except socket.timeout:
                    pass
                except OSError:
                    break
        self.thread = threading.Thread(target=accept, daemon=True)
        self.thread.start()
        return ip, self.listener.getsockname()[1]

    def connect(self, host, port):
        with socket.create_connection((routable(host), port), timeout=5) as conn:
            conn.settimeout(5)
            if conn.recv(2) != b'OK':
                raise RuntimeError('Cross-node TCP rendezvous failed')
        return {'hostname': socket.gethostname(), 'tcp_reachable': True}

    def stop_listener(self):
        self.listening = False
        self.listener.close()
        self.thread.join(timeout=2)

    def free_port(self):
        import ray
        ip = routable(ray.util.get_node_ip_address())
        with socket.socket(socket.AF_INET6 if ':' in ip else socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((ip, 0))
            return ip, sock.getsockname()[1]

    def collective(self, rank, world, per_node, host, port, nbytes=GIB, iters=3):
        log = f'/tmp/strl-nccl-probe-{rank}.log'
        os.environ.update(NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,NET', NCCL_DEBUG_FILE=log)
        import torch
        import torch.distributed as dist
        torch.cuda.set_device(0)
        host = f'[{host}]' if ':' in host else host
        dist.init_process_group('nccl', init_method=f'tcp://{host}:{port}', rank=rank, world_size=world,
                                timeout=datetime.timedelta(seconds=180))
        value = torch.full((1024 * 1024,), float(rank + 1), device='cuda')
        dist.all_reduce(value)
        torch.cuda.synchronize()
        if not bool(torch.all(value == world * (world + 1) / 2).item()):
            raise RuntimeError('Incorrect cross-node all-reduce')
        nodes = world // per_node
        intra = [dist.new_group([n * per_node + i for i in range(per_node)]) for n in range(nodes)]
        cross = [dist.new_group([n * per_node + i for n in range(nodes)]) for i in range(per_node)]

        def timed(group, size):
            out = torch.empty(nbytes // 2, dtype=torch.bfloat16, device='cuda')
            inp = torch.ones(nbytes // 2 // size, dtype=torch.bfloat16, device='cuda')
            for _ in range(2):
                dist.all_gather_into_tensor(out, inp, group=group)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            start = time.perf_counter()
            for _ in range(iters):
                dist.all_gather_into_tensor(out, inp, group=group)
            torch.cuda.synchronize()
            return bus_bandwidth(nbytes, (time.perf_counter() - start) / iters, size)
        result = {'intra_node': timed(intra[rank // per_node], per_node) if per_node > 1 else None,
                  'cross_node': timed(cross[rank % per_node], nodes) if nodes > 1 else None,
                  'global': timed(None, world)}
        dist.destroy_process_group()
        text = Path(log).read_text(errors='replace') if Path(log).is_file() else ''
        return dict(self.location(), rank=rank, world_size=world, all_reduce_passed=True, transport=classify_transport(text),
                    nccl_net_lines=[line for line in text.splitlines() if 'NET/' in line or 'Using network' in line][:8], **result)


def _median(values):
    values = sorted(v for v in values if v is not None)
    return values[len(values) // 2] if values else None


def summarize(nodes: list, facts: list, ranks: list | None, cpu: bool) -> dict:
    """Probe results -> the network.json dict (pure: unit-tested)."""
    efa = all(f['efa']['efa_available'] for f in facts) if facts else False
    out = {'passed': True, 'mode': 'cpu' if cpu else 'gpu', 'nodes': len(nodes), 'hostnames': sorted({f['hostname'] for f in facts}),
           'node_facts': facts, 'tcp': 'bidirectional', 'efa_available': efa, 'transport': 'not_measured' if cpu else 'unknown',
           'bandwidth_gbps': {'intra_node': None, 'cross_node': None, 'global': None}, 'warnings': []}
    if ranks:
        transports = sorted({r['transport'] for r in ranks})
        net = [t for t in transports if t not in {'nvlink', 'unknown'}]
        out['transport'] = net[0] if len(net) == 1 else ('/'.join(net) if net else transports[0])
        for scope in ('intra_node', 'cross_node', 'global'):
            out['bandwidth_gbps'][scope] = _median([(r.get(scope) or {}).get('busbw_gbps') for r in ranks])
        out['ranks'] = ranks
    if len(nodes) > 1 and not efa:
        out['warnings'].append('EFA not available in this image/pod: cross-node NCCL falls back to TCP sockets')
    if out['transport'] == 'socket':
        out['warnings'].append('NCCL uses NET/Socket across nodes: prefer LoRA or HSDP (fsdp_size = gpus_per_node) for full fine-tuning')
    return out


def probe_cluster(nodes: int, gpus_per_node: int, cpu: bool = False, timeout: float = 600) -> dict:
    """Run on the head after every node joined Ray (ray.init already called). Returns the network.json dict."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    alive = sorted([n for n in ray.nodes() if n['Alive']], key=lambda n: n['NodeID'])
    if len(alive) != nodes:
        raise RuntimeError(f'Probe needs {nodes} live Ray nodes, found {len(alive)}')
    if len({n['NodeManagerAddress'] for n in alive}) != nodes:
        raise RuntimeError('Ray nodes share one node address (replicas co-located on one host)')
    actor_cls = ray.remote(Rank)
    per_node = 1 if cpu else gpus_per_node
    actors = []
    try:
        for node in alive:
            pin = NodeAffinitySchedulingStrategy(node_id=node['NodeID'], soft=False)
            for _ in range(per_node):
                actors.append(actor_cls.options(num_cpus=0 if cpu else 1, num_gpus=0 if cpu else 1, scheduling_strategy=pin).remote())
        firsts = actors[::per_node]
        facts = ray.get([a.facts.remote() for a in firsts], timeout=120)
        locations = ray.get([a.location.remote() for a in actors], timeout=120)
        if len({r['hostname'] for r in locations}) != nodes:
            raise RuntimeError('Probe ranks did not span distinct hosts')
        for listener in firsts:
            host, port = ray.get(listener.listen.remote(), timeout=15)
            ray.get([a.connect.remote(host, port) for a in actors], timeout=30)
            ray.get(listener.stop_listener.remote(), timeout=10)
        ranks = None
        if not cpu:
            host, port = ray.get(actors[0].free_port.remote(), timeout=15)
            ranks = ray.get([a.collective.remote(r, len(actors), per_node, host, port) for r, a in enumerate(actors)], timeout=timeout)
        return summarize(alive, facts, ranks, cpu)
    finally:
        for actor in actors:
            ray.kill(actor)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nodes', type=int, required=True)
    p.add_argument('--gpus-per-node', type=int, required=True)
    p.add_argument('--cpu', action='store_true', help='dry runs: node facts + TCP, no NCCL')
    p.add_argument('--output', default='artifacts/network.json')
    a = p.parse_args(argv)
    import ray
    ray.init(address=os.environ['RAY_ADDRESS'])
    try:
        result = probe_cluster(a.nodes, a.gpus_per_node, cpu=a.cpu)
    finally:
        ray.shutdown()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ('passed', 'nodes', 'hostnames', 'transport', 'efa_available', 'bandwidth_gbps', 'warnings')}))


if __name__ == '__main__':
    main()
