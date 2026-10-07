"""Export selected MiMo-V2.6-RL-oss rows to Harbor task dirs. See strl/tasks/mimo/__init__.py
for the full contract (task dir layout, strl.json fields, verifier outputs).

``export_task`` is deterministic: the same ``row`` always produces a byte-identical task dir
(no timestamps, no random ids; every JSON file is dumped with ``sort_keys=True``). The two
build-time artifacts the contract calls for (cyber's pristine binary.tar.gz, code's base_ref)
need a throwaway Modal sandbox of the task's own image; that is still deterministic PER IMAGE
(same image -> same artifact), but is slow and costs real money at full-dataset scale, so it is
gated behind ``MIMO_EXPORT_OFFLINE`` for fast, networkless, Modal-less unit tests -- see
``_cyber_capture_fn`` / ``_code_base_ref_fn`` below. Production exports (``data.py::prepare``,
Track A) must run with that unset.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import urllib.request
from pathlib import Path
from typing import Any

from .graders import cyber as cyber_grader
from .graders import code as code_grader
from .graders import general as general_grader
from .graders import music as music_grader
from .graders import webdev as webdev_grader

DOMAINS = ("music", "cyber", "code", "general", "webdev")

# music rows carry no docker_image (extra_info is just {src_id, lang, tag, length,
# nvoice_want, bpm, meter}): there is no MiMo image for this domain, so every music task runs
# on a plain, non-MiMo base we control (needs abcmidi for the scorer; tmux kept for symmetry).
MUSIC_IMAGE = "python:3.12-slim"

_RAW_DIR = Path(__file__).resolve().parents[3] / "assets" / "mimo" / "raw"
_IMAGE_MAP_PATH = _RAW_DIR / "image-mapping.jsonl"

_HF_REPO = "XiaomiMiMo/MiMo-V2.6-RL-oss"
_HF_REV = "639865fd3374018d6cb29b9fb82dd531406fcf5f"
_HF_API = f"https://huggingface.co/api/datasets/{_HF_REPO}/tree/{_HF_REV}/"
_HF_RESOLVE = f"https://huggingface.co/datasets/{_HF_REPO}/resolve/{_HF_REV}/"

OFFLINE_ENV_VAR = "MIMO_EXPORT_OFFLINE"
PLACEHOLDER_ARTIFACT_NOTE = "MIMO_EXPORT_OFFLINE=1: build-time Modal artifact not computed; run export again without it."


# --------------------------------------------------------------------------------- image map


def _load_image_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(_IMAGE_MAP_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mapping[row["dataset_image"]] = row["dockerhub_image"]
    return mapping


_IMAGE_MAP: dict[str, str] | None = None


def image_ref(domain: str, instance_json: dict) -> str:
    """``docker.io/xiaomimimo/mimo-v2.6-rl-oss:<tag>`` for this row's ``docker_image``, via
    ``assets/mimo/raw/image-mapping.jsonl``. Raises ``KeyError`` with the unmapped tag if the
    image isn't in the mapping (export must fail loudly, not silently fall back)."""
    global _IMAGE_MAP
    if _IMAGE_MAP is None:
        _IMAGE_MAP = _load_image_map()
    dataset_image = instance_json.get("docker_image")
    if not dataset_image:
        raise KeyError(f"{domain}: row has no docker_image in instance_json/extra_info: {instance_json!r}")
    try:
        return _IMAGE_MAP[dataset_image]
    except KeyError:
        raise KeyError(f"{domain}: docker_image {dataset_image!r} not found in {_IMAGE_MAP_PATH}") from None


# ------------------------------------------------------------------------------- domain specs


class DomainSpec:
    def __init__(
        self,
        *,
        agent_timeout_sec: float,
        verifier_timeout_sec: float,
        cpus: int,
        memory_mb: int,
        storage_mb: int,
        max_turns: int,
        max_tokens: int,
        agent_name: str | None,
        agent_import_path: str | None,
        artifacts: list[str],
        judge_kind: str | None,
    ) -> None:
        self.agent_timeout_sec = agent_timeout_sec
        self.verifier_timeout_sec = verifier_timeout_sec
        self.cpus = cpus
        self.memory_mb = memory_mb
        self.storage_mb = storage_mb
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.agent_name = agent_name
        self.agent_import_path = agent_import_path
        self.artifacts = artifacts
        self.judge_kind = judge_kind


DOMAIN_SPECS: dict[str, DomainSpec] = {
    "music": DomainSpec(
        agent_timeout_sec=300, verifier_timeout_sec=120, cpus=1, memory_mb=2048, storage_mb=4096,
        max_turns=1, max_tokens=3072, agent_name=None,
        agent_import_path="strl.tasks.mimo.agents:SingleShotAgent",
        artifacts=[music_grader.ANSWER_PATH], judge_kind=None,
    ),
    "webdev": DomainSpec(
        agent_timeout_sec=900, verifier_timeout_sec=300, cpus=2, memory_mb=4096, storage_mb=10240,
        max_turns=12, max_tokens=2048, agent_name="terminus-2", agent_import_path=None,
        artifacts=[webdev_grader.SCREENSHOT_PATH], judge_kind="vision",
    ),
    "general": DomainSpec(
        agent_timeout_sec=900, verifier_timeout_sec=180, cpus=2, memory_mb=4096, storage_mb=10240,
        max_turns=16, max_tokens=2048, agent_name="terminus-2", agent_import_path=None,
        artifacts=[general_grader.ANSWER_PATH], judge_kind="rubric",  # terminal_bench rows override to None below
    ),
    "cyber": DomainSpec(
        agent_timeout_sec=900, verifier_timeout_sec=300, cpus=2, memory_mb=4096, storage_mb=10240,
        max_turns=16, max_tokens=2048, agent_name="terminus-2", agent_import_path=None,
        artifacts=[cyber_grader.POC_PATH], judge_kind=None,
    ),
    "code": DomainSpec(
        agent_timeout_sec=1200, verifier_timeout_sec=900, cpus=4, memory_mb=8192, storage_mb=20480,
        max_turns=20, max_tokens=2048, agent_name="terminus-2", agent_import_path=None,
        artifacts=["/logs/verifier/patch.diff"], judge_kind=None,
    ),
}


# --------------------------------------------------------------------------------- Dockerfile

# Built from source in an Alpine stage with --enable-static against static libevent/ncurses:
# the result has ZERO dynamic dependencies, so a single COPY --from=... works on every one of
# the five task base images regardless of libc (glibc Ubuntu/Debian, musl Alpine, ...) -- unlike
# `apk add tmux` (dynamically links musl + libevent + ncurses) or a third-party prebuilt binary
# (sha256 would have to be re-pinned by hand whenever it moves). Network for THIS stage comes
# from the Modal build step, not the task sandbox -- the task sandbox itself stays
# network-blocked (see strl/tasks/mimo/__init__.py and _create_sandbox's block_network).
_TMUX_BUILD_STAGE = """FROM alpine:3.20 AS mimo-tmux-static
RUN apk add --no-cache build-base automake autoconf pkgconfig bison \\
        ncurses-static ncurses-dev libevent-static libevent-dev linux-headers \\
    && wget -q https://github.com/tmux/tmux/releases/download/3.4/tmux-3.4.tar.gz -O /tmp/tmux.tar.gz \\
    && tar -xzf /tmp/tmux.tar.gz -C /tmp \\
    && cd /tmp/tmux-3.4 \\
    && LDFLAGS="-static" ./configure --enable-static \\
    && make -j"$(nproc)"
"""

_TMUX_COPY = "COPY --from=mimo-tmux-static /tmp/tmux-3.4/tmux /usr/local/bin/tmux\n"


def _dockerfile(image: str, *, extra_build_stages: str = "", extra_lines: str = "") -> str:
    return (
        _TMUX_BUILD_STAGE
        + extra_build_stages
        + f"\nFROM {image}\n"
        + _TMUX_COPY
        + "RUN chmod +x /usr/local/bin/tmux\n"
        + extra_lines
    )


# Proven in a real Modal build (2026-10-06): the published webdev-rl-opensource image already
# ships node 20 + a working playwright + chromium (`sync_playwright().start().chromium.launch()`
# succeeds out of the box). Re-installing playwright unconditionally risked a python-package /
# already-downloaded-browser version mismatch for no benefit, so this only installs anything
# when the probe fails -- keeping the Dockerfile correct for OTHER webdev images that might not
# have it baked in, without touching a working preinstall.
_WEBDEV_PLAYWRIGHT_LINES = (
    "RUN python3 -c \"from playwright.sync_api import sync_playwright as s; b=s().start().chromium.launch(); "
    "b.close()\" 2>/dev/null || ( \\\n"
    "    (command -v node >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends "
    "nodejs npm && rm -rf /var/lib/apt/lists/*)) \\\n"
    "    && python3 -m pip install --no-cache-dir playwright==1.47.0 Pillow \\\n"
    "    && python3 -m playwright install --with-deps chromium)\n"
)

_GENERAL_AGENT_DOCKERFILE_TAIL = """RUN useradd -m -u 5055 mcpsvc 2>/dev/null || adduser -D -u 5055 mcpsvc 2>/dev/null || true
COPY general_env/system /opt/mimo_general/system
COPY general_env/tools /opt/mimo_general/tools
COPY general_env/mcp_http.py /opt/mimo_general/mcp_http.py
COPY general_env/sidecar.py /opt/mimo_general/sidecar.py
# The published general-agent-env-0 image ships `mcp` 2.2.0, whose `mcp.server.fastmcp.FastMCP`
# was renamed/restructured to `mcp.server.mcpserver.MCPServer` (confirmed in a real Modal
# sandbox, 2026-10-06); the fetched env dir's mcp_http.py (from MiMo's own HF dataset repo)
# still imports `FastMCP`, so it crashes on import under 2.x. Downgrading to the last 1.x
# release fixes the SERVER side; the CLIENT side (mcp_cli.py) uses
# `mcp.client.streamable_http.streamable_http_client`, which exists under BOTH 1.x and 2.x
# (verified in the same sandbox), so one pinned version satisfies both.
RUN pip install --no-cache-dir -q "mcp<2"
# `pip install mcp` also installs the PACKAGE's OWN `mcp` console-script (needs `typer`, which
# this pin doesn't pull in) at /usr/local/bin/mcp -- our wrapper must be copied (and win that
# path) AFTER the pip install, not before (confirmed clobbered in a real sandbox otherwise).
COPY general_env/mcp_cli.py /usr/local/bin/mcp
RUN chmod +x /usr/local/bin/mcp && chown -R mcpsvc:mcpsvc /opt/mimo_general
ENTRYPOINT ["/bin/sh", "-c", "su mcpsvc -s /bin/sh -c 'python3 /opt/mimo_general/sidecar.py' & exec sleep infinity"]
"""


_MUSIC_ABCMIDI_LINE = "RUN apt-get update && apt-get install -y --no-install-recommends abcmidi && rm -rf /var/lib/apt/lists/*\n"


def build_dockerfile(domain: str, image: str) -> str:
    if domain == "music":
        return _dockerfile(image, extra_lines=_MUSIC_ABCMIDI_LINE)
    if domain == "webdev":
        return _dockerfile(image, extra_lines=_WEBDEV_PLAYWRIGHT_LINES)
    if domain == "general":
        return _dockerfile(image, extra_lines=_GENERAL_AGENT_DOCKERFILE_TAIL)
    return _dockerfile(image)


# -------------------------------------------------------------------------------- task.toml


def build_task_toml(domain: str) -> str:
    spec = DOMAIN_SPECS[domain]
    return (
        'schema_version = "1.3"\n\n'
        "[metadata]\n"
        'source = "mimo export"\n\n'
        "[verifier]\n"
        f"timeout_sec = {spec.verifier_timeout_sec}\n\n"
        "[agent]\n"
        f"timeout_sec = {spec.agent_timeout_sec}\n\n"
        "[environment]\n"
        'network_mode = "no-network"\n'
        "build_timeout_sec = 1800.0\n"
        'os = "linux"\n'
        f"cpus = {spec.cpus}\n"
        f"memory_mb = {spec.memory_mb}\n"
        f"storage_mb = {spec.storage_mb}\n"
    )


# ------------------------------------------------------------------------------ instruction.md

_INSTRUCTIONS = {
    "music": (
        "{prompt}\n\n---\nWrite the complete piece in ABC notation. Reply with the tune in a single "
        "```abc\n...\n``` fenced code block (starting with an `X:` tune header). This is a single-shot "
        "task: your entire reply is graded as-is, there is no terminal.\n"
    ),
    "cyber": (
        "{prompt}\n\n---\nYou are in a Linux terminal.\n"
        "- Resources: binary and libraries at /home/agent/binary/, source code at /home/agent/src/.\n"
        f"- Save your final proof-of-concept input at `{cyber_grader.POC_PATH}` before finishing.\n"
        "- Local testing: `bash /home/agent/binary/run.sh <your_poc_file>`.\n"
        "- Submit for feedback (crash/match, without revealing the exact function): "
        "`bash /home/agent/submit.sh <your_poc_file>`.\n"
        "- You may submit multiple times. The file at the path above when you finish is what gets graded, "
        "independently of anything submit.sh reported.\n"
    ),
    "code": (
        "{prompt}\n\n---\nYou are in a Linux terminal with a checkout of the repository at `{cwd}`. "
        "Make the minimal changes needed to resolve the issue above. Do not modify any test files. "
        "Finish as soon as your fix is in place; a hidden test suite will be applied and run.\n"
    ),
    "general": (
        "{prompt}\n\n---\nYou are in a Linux terminal. Local MCP-style tool servers for this task are "
        "reachable only through the `mcp` command-line wrapper (you do not have a native MCP client): \n"
        "    mcp <server> <tool> '<json-arguments>'\n"
        "Available servers: {servers}\n"
        f"Write your final answer to `{general_grader.ANSWER_PATH}` before finishing.\n"
    ),
    "webdev": (
        "{prompt}\n\n---\nYou are in a Linux terminal. Build the requested site as static files "
        f"(HTML/CSS/JS, optionally a framework build) under `{webdev_grader.WORKSPACE_DIR}/`, with "
        f"`{webdev_grader.WORKSPACE_DIR}/{webdev_grader.ENTRY_PAGE}` as the page that gets rendered and "
        "graded. Use mock content only -- there is no backend and the sandbox has no network access. "
        "Finish as soon as the site is complete.\n"
    ),
}


def _row_prompt(raw: dict) -> str:
    prompt = raw.get("prompt") or []
    return prompt[0]["content"] if prompt else ""


def _instance(raw: dict) -> dict:
    extra_info = raw.get("extra_info") or {}
    instance_json = extra_info.get("instance_json")
    if isinstance(instance_json, str):
        return json.loads(instance_json)
    return dict(extra_info)


def build_instruction(domain: str, row: dict, *, mcp_server_names: list[str] | None = None) -> str:
    raw = row.get("raw") or {}
    prompt = _row_prompt(raw)
    instance = _instance(raw)
    template = _INSTRUCTIONS[domain]
    if domain == "code":
        return template.format(prompt=prompt, cwd=instance.get("cwd") or "/testbed")
    if domain == "general":
        servers = ", ".join(mcp_server_names or [])
        return template.format(prompt=prompt, servers=servers)
    return template.format(prompt=prompt)


# ------------------------------------------------------------------------------------ strl.json


def build_strl_json(domain: str, row: dict, image: str, *, judge_kind: str | None = None) -> dict:
    spec = DOMAIN_SPECS[domain]
    agent: dict[str, Any] = {"import_path": spec.agent_import_path} if spec.agent_import_path else {"name": spec.agent_name}
    out = {
        "domain": domain,
        "instance_id": row["instance_id"],
        "group": row.get("group", domain),
        "image": image,
        "agent": agent,
        "max_turns": spec.max_turns,
        "max_tokens": spec.max_tokens,
        "artifacts": list(spec.artifacts),
        "judge": None if judge_kind is None else {"kind": judge_kind},
    }
    return out


# ------------------------------------------------------------------- general_agent env fetch


def _hf_list(path: str) -> list[dict]:
    with urllib.request.urlopen(_HF_API + path, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_general_agent_env_dir(instance_id: str, env_task_dir: str, dest: Path) -> Path:
    """Recursively download ``general/<env_task_dir>/`` from the HF dataset repo into
    ``dest/<instance_id>/``. Network-dependent; injectable in ``export_task`` via
    ``general_env_dir_fn`` for offline tests."""
    root = f"general/{env_task_dir}"
    out_dir = dest / instance_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def _walk(hf_path: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        for entry in _hf_list(hf_path):
            rel = entry["path"][len(root) + 1 :]
            local_path = out_dir / rel
            if entry["type"] == "directory":
                _walk(entry["path"], out_dir / rel)
            else:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(_HF_RESOLVE + entry["path"], timeout=60) as resp:
                    local_path.write_bytes(resp.read())

    _walk(root, out_dir)
    return out_dir


_SIDECAR_PY = '''#!/usr/bin/env python3
"""Our own minimal general_agent sidecar: launches one mcp_http.py per tools/<name>.py file,
on ports 39101, 39102, ... (sorted), waits for all to be listening, then idles forever.

Deliberately NOT a vendored copy of MiMo-Agent's sidecar_entrypoint.py: that script hardcodes
a venv path (/opt/openai-agents-venv/bin/python) specific to MiMo's own training image, which
we have not verified exists in the published general-agent-env-0 image; this uses the system
python3 (mcp 2.2.0 is preinstalled there per probing) instead.
"""
import socket
import subprocess
import sys
import time
from pathlib import Path

TOOLS_DIR = Path("/opt/mimo_general/tools")
MCP_HTTP = "/opt/mimo_general/mcp_http.py"
PORT_BASE = 39101


def main() -> None:
    tools = sorted(p.stem for p in TOOLS_DIR.glob("*.py") if not p.name.startswith("_"))
    procs = []
    for i, name in enumerate(tools):
        port = PORT_BASE + i
        proc = subprocess.Popen(
            [sys.executable, MCP_HTTP, "--tools-path", str(TOOLS_DIR / f"{name}.py"), "--server-name", name, "--port", str(port)],
            stdout=sys.stderr, stderr=sys.stderr,
        )
        procs.append((name, port, proc))
        print(f"[sidecar] started {name} on :{port}", file=sys.stderr)

    deadline = time.time() + 120
    while time.time() < deadline:
        if all(_listening(port) for _, port, _ in procs):
            break
        time.sleep(0.5)
    print("[sidecar] ready" if all(_listening(p) for _, p, _ in procs) else "[sidecar] NOT all ports ready", file=sys.stderr)

    while True:
        time.sleep(5)
        for name, port, proc in procs:
            if proc.poll() is not None:
                print(f"[sidecar] {name} exited rc={proc.returncode}", file=sys.stderr)


def _listening(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


if __name__ == "__main__":
    main()
'''


def _write_general_env_assets(task_dir: Path, env_dir: Path, mcp_server_names: list[str]) -> None:
    environment_dir = task_dir / "environment" / "general_env"
    for sub in ("system", "tools"):
        src = env_dir / sub
        if src.is_dir():
            shutil.copytree(src, environment_dir / sub, dirs_exist_ok=True)
    mcp_http_src = env_dir / "mcp_http.py"
    if mcp_http_src.is_file():
        (environment_dir / "mcp_http.py").write_text(mcp_http_src.read_text())
    (environment_dir / "sidecar.py").write_text(_SIDECAR_PY)
    port_map = {name: 39101 + i for i, name in enumerate(mcp_server_names)}
    (environment_dir / "mcp_cli.py").write_text(general_grader.render_mcp_cli(port_map))


# --------------------------------------------------------------------------------- export_task


def _cyber_capture_fn(image: str):
    if os.environ.get(OFFLINE_ENV_VAR):
        return (PLACEHOLDER_ARTIFACT_NOTE.encode(), "0" * 64)
    return cyber_grader.capture_pristine_binary(image)


def _code_base_ref_fn(image: str, cwd: str) -> str:
    if os.environ.get(OFFLINE_ENV_VAR):
        return "0" * 40
    return code_grader.compute_base_ref(image, cwd)


def _default_general_env_dir_fn(dest_root: Path):
    def _fn(instance_id: str, env_task_dir: str) -> Path:
        return fetch_general_agent_env_dir(instance_id, env_task_dir, dest_root)

    return _fn


def export_task(row: dict, dest: Path, *, split: str, general_env_dir_fn=None) -> dict:
    """Export one row to ``<dest>/<split>/<domain>-<instance_id>/``. Deterministic for a fixed
    row (see module docstring re: ``MIMO_EXPORT_OFFLINE``). Returns the manifest entry."""
    domain = row["domain"]
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    instance_id = row["instance_id"]
    raw = row.get("raw") or {}
    instance = _instance(raw)

    # music has no MiMo docker image at all (extra_info carries no docker_image field): per
    # the contract's background, it runs on a plain python:3.12-slim + abcmidi + tmux image.
    image = MUSIC_IMAGE if domain == "music" else image_ref(domain, instance)

    task_dir = dest / split / f"{domain}-{instance_id}"
    if task_dir.exists():
        shutil.rmtree(task_dir)
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)

    mcp_server_names: list[str] = []
    judge_kind = DOMAIN_SPECS[domain].judge_kind

    if domain == "music":
        music_grader.build(row, task_dir)
    elif domain == "cyber":
        cyber_grader.build(row, task_dir, capture_fn=_cyber_capture_fn)
    elif domain == "code":
        code_grader.build(row, task_dir, base_ref_fn=_code_base_ref_fn)
    elif domain == "webdev":
        webdev_grader.build(row, task_dir)
    elif domain == "general":
        if general_grader.dataset_sub_kind(row) == "terminal_bench":
            general_grader.build_terminal_bench(row, task_dir)
            judge_kind = None
        else:
            if general_env_dir_fn is None:
                scratch = task_dir.parent / ".general_env_cache"
                general_env_dir_fn = _default_general_env_dir_fn(scratch)
            env_task_dir = instance.get("env_task_dir")
            env_dir = general_env_dir_fn(instance_id, env_task_dir)
            general_grader.build_general_agent(row, task_dir, env_dir=env_dir)
            manifest = json.loads((env_dir / "manifest.json").read_text()) if (env_dir / "manifest.json").is_file() else {}
            mcp_server_names = [s["name"] for s in manifest.get("mcp_servers", [])]
            _write_general_env_assets(task_dir, env_dir, mcp_server_names)

    (task_dir / "environment" / "Dockerfile").write_text(build_dockerfile(domain, image))
    (task_dir / "task.toml").write_text(build_task_toml(domain))
    (task_dir / "instruction.md").write_text(build_instruction(domain, row, mcp_server_names=mcp_server_names))
    (task_dir / "strl.json").write_text(json.dumps(build_strl_json(domain, row, image, judge_kind=judge_kind), sort_keys=True))

    return {
        "task_dir": str(task_dir),
        "domain": domain,
        "instance_id": instance_id,
        "group": row.get("group", domain),
        "image": image,
        "tree_sha256": _tree_sha256(task_dir),
    }


def _tree_sha256(task_dir: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    for path in sorted(task_dir.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(task_dir)).encode()
            h.update(rel)
            h.update(path.read_bytes())
    return h.hexdigest()
