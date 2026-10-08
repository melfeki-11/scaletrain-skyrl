from __future__ import annotations

from pathlib import Path


PATCHES = {
    "skyrl/train/utils/vllm_metrics_scraper.py": [
        ('        if not ip or not port:\n', '        if not ip or not isinstance(port, int) or not 0 < port < 65536:\n'),
        ('        urls.append(f"http://{ip}:{port}/metrics")\n',
         '        host = f"[{ip}]" if ":" in ip and not ip.startswith("[") else ip\n'
         '        urls.append(f"http://{host}:{port}/metrics")\n'),
    ],
    "skyrl/backends/skyrl_train/inference_servers/common.py": [
        (
            '        return f"http://{self.ip}:{self.port}"\n',
            '        host = f"[{self.ip}]" if ":" in self.ip and not self.ip.startswith("[") else self.ip\n'
            '        return f"http://{host}:{self.port}"\n',
        ),
        (
            "def find_and_reserve_port(start_port: int) -> Tuple[int, socket.socket]:\n",
            "def find_and_reserve_port(start_port: int, host: str | None = None) -> Tuple[int, socket.socket]:\n",
        ),
        (
            "    sock: socket.socket | None = None\n",
            '    family = socket.AF_INET6 if host is not None and ":" in host else socket.AF_INET\n'
            '    bind_host = "::" if family == socket.AF_INET6 else ""\n'
            "    sock: socket.socket | None = None\n",
        ),
        (
            '            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n'
            '            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n'
            '            sock.bind(("", port))\n',
            '            sock = socket.socket(family, socket.SOCK_STREAM)\n'
            '            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n'
            '            if family == socket.AF_INET6:\n'
            '                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)\n'
            '            sock.bind((bind_host, port))\n',
        ),
    ],
    "skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py": [
        (
            "        self._port, self._port_reservation = find_and_reserve_port(start_port)\n",
            "        self._port, self._port_reservation = find_and_reserve_port(start_port, host=self._ip)\n",
        ),
        ('        self._cli_args.host = "0.0.0.0"\n', '        self._cli_args.host = "::" if ":" in self._ip else "0.0.0.0"\n'),
        ('        url = f"http://{self._ip}:{self._port}/health"\n', '        url = f"{self.get_server_info().url}/health"\n'),
    ],
    "skyrl/backends/skyrl_train/inference_servers/vllm_router.py": [
        (
            "from skyrl.backends.skyrl_train.inference_servers.common import (\n"
            "    find_and_reserve_port,\n"
            "    get_node_ip,\n"
            ")\n",
            "from skyrl.backends.skyrl_train.inference_servers.common import (\n"
            "    ServerInfo,\n"
            "    find_and_reserve_port,\n"
            "    get_node_ip,\n"
            ")\n",
        ),
        (
            "        reserved_port, self._port_reservation = find_and_reserve_port(self._router_args.port)\n",
            "        self._ip = get_node_ip()\n"
            '        self._router_args.host = "::" if ":" in self._ip else "0.0.0.0"\n'
            "        reserved_port, self._port_reservation = find_and_reserve_port(self._router_args.port, host=self._ip)\n",
        ),
        (
            "        reserved_prom_port, self._prometheus_port_reservation = find_and_reserve_port(prometheus_start)\n",
            "        reserved_prom_port, self._prometheus_port_reservation = find_and_reserve_port(prometheus_start, host=self._ip)\n",
        ),
        (
            '        ip = get_node_ip()\n        router_url = f"http://{ip}:{self._router_args.port}"\n',
            "        router_url = ServerInfo(ip=self._ip, port=self._router_args.port).url\n",
        ),
    ],
}


def patch_tree(root: str | Path, patches: dict | None = None) -> None:
    """Apply exact-source replacements (idempotent; refuses when the pinned source changed). Default: the IPv6 patches."""
    root = Path(root)
    for relative, replacements in (PATCHES if patches is None else patches).items():
        path = root / relative
        text = path.read_text(encoding="utf-8")
        for before, after in replacements:
            if after in text:
                continue
            if text.count(before) != 1:
                raise RuntimeError(f"Exact-source patch no longer applies cleanly to {relative}")
            text = text.replace(before, after, 1)
        path.write_text(text, encoding="utf-8")
