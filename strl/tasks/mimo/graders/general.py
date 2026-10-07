"""general grader: two sub-kinds selected by ``extra_info.instance_json["dataset_type"]``.

* ``general_agent`` (925 tasks, image ``general-agent-env-0``): deterministic ``verifier_meta.json``
  items (``method == "rule"``) run IN-SANDBOX via a port of XiaomiMiMo/MiMo-Agent-oss's
  ``general_agent`` sidecar ``verify.py::_run_rule_sc`` / ``_grade_sc`` scoring algebra (the
  genenv reference sample under ``scratchpad/genenv/verify.py`` -- not a separately licensed,
  importable package, so this is a documented re-implementation of its scoring rules rather
  than a vendored file). ``method == "llm"`` items become ``judge_request.json`` entries per
  the ``strl/tasks/mimo/__init__.py`` contract; the judge itself runs OUTSIDE the sandbox.
* ``terminal_bench`` (64 tasks): ``tests_files`` (a JSON-object string whose values are
  base64-encoded file contents -- ``test.sh``/``test_outputs.py``/``anti_hack_guard.py``/
  ``fixtures/*``) ships inside ``extra_info.instance_json``; we write it into ``tests/``
  byte-for-byte and run it verbatim --
  it already writes ``/logs/verifier/reward.txt`` itself (after its own ``anti_hack_guard``), so
  there is no judge for this sub-kind.

Scoring algebra for ``general_agent`` (ported from the reference's ``_grade_sc``, see that
file's inline comments for the Chinese original):

  * Items carrying a truthy ``"gate"`` key are hard gates: if ANY gate's rule score is < 0.999,
    the rule-visible portion of the task is dead -- upstream forces the WHOLE score to 0 and
    skips the judge entirely. We cannot silently drop llm items (the judge call happens
    outside our sandbox), so instead we set ``details.json["gate_failed"] = True`` /
    ``judge_request.json["force_zero"] = True`` and still ship judge_request.json unchanged;
    **this is a note for judge.py (Track A): a true ``force_zero`` must make the combined
    reward 0 regardless of the judge's per-item scores**, matching upstream parity. Gates do
    NOT count toward the weight denominator (upstream: "gate 不进分母").
  * Every other item (rule or llm) is an "atom" with ``weight = item.get("weight", 1.0)``
    (upstream default). Rule atoms are summed directly into ``base_score``/``base_weight``
    (computed in-sandbox); llm atoms go to ``judge_request.json`` with their own weight and
    count toward ``total_weight`` but not ``base_weight``.
  * ``reward.txt`` (the in-sandbox-only portion, already divided by the TOTAL weight per the
    top-level contract) = ``base_score / total_weight`` (0 when there is nothing to grade, or
    when a gate failed).
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ANSWER_PATH = "/app/answer.md"
MCP_CLI_PATH = "/usr/local/bin/mcp"

# ----------------------------------------------------------------- general_agent: rule items


def run_rule_item(*, check_code: str, fn: str, helpers_src: str, workspace: Path, timeout: int = 300) -> dict[str, Any]:
    """Execute one deterministic ``verifier_meta.json`` rule item: ``fn(workspace_path)`` with
    ``helpers_src`` (the env dir's ``_helpers.py``) and ``check_code`` (``verifier_meta.json
    ["check_code"]``) available in scope. Subprocess-isolated, matching the reference's
    ``_run_rule_sc``. Returns ``{"score": float, "detail": str}``.

    ``random``/``random.`` in ``check_code`` is rejected (matches upstream: a rule check must
    be deterministic, since it is the piece we trust without a judge).
    """
    if "import random" in check_code or "random." in check_code:
        return {"score": 0.0, "detail": "check_code uses random: rejected (rule items must be deterministic)"}

    raw = (
        "from pathlib import Path\n"
        + helpers_src
        + "\n"
        + check_code
        + "\n"
        + "import json as _j, sys as _s\n"
        + f"print('___R___' + _j.dumps({fn}(Path(_s.argv[1])), ensure_ascii=False, default=str))\n"
    )
    # `from __future__ import ...` must be the first statement; helpers/check_code may carry
    # one (commonly `annotations`). Hoist all such lines to the top, de-duplicated, order kept.
    lines = raw.split("\n")
    futures = list(dict.fromkeys(l for l in lines if l.lstrip().startswith("from __future__ import")))
    rest = [l for l in lines if not l.lstrip().startswith("from __future__ import")]
    src = "\n".join(futures + rest)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src)
        script_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, script_path, str(workspace)], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"score": 0.0, "detail": f"rule {fn} timed out after {timeout}s"}
    finally:
        Path(script_path).unlink(missing_ok=True)

    out = ""
    for line in (result.stdout or "").splitlines():
        if line.startswith("___R___"):
            out = line[len("___R___") :]
    if not out:
        return {"score": 0.0, "detail": f"rule {fn} produced no output: {(result.stderr or '')[-200:]}"}
    try:
        data = json.loads(out)
        return {"score": float(data.get("score", 0)), "detail": str(data.get("detail", ""))[:200]}
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "detail": f"rule {fn} produced invalid output: {e}"}


def grade_general_agent(
    *, verifier_meta: dict, workspace: Path, answer_text: str = "", rule_timeout: int = 300
) -> dict[str, Any]:
    """Run every rule item in-sandbox, build the judge hand-off for every llm item. Returns
    ``{"reward": float, "details": {...}, "judge_request": dict | None}``."""
    items = verifier_meta.get("items") or []
    check_code = verifier_meta.get("check_code", "")
    helpers_src = verifier_meta.get("_helpers_src", "")

    gates = [it for it in items if it.get("gate")]
    atoms = [it for it in items if not it.get("gate")]

    rule_results: dict[str, dict] = {}
    gate_failed: list[str] = []
    for it in gates:
        fn = it.get("fn")
        r = run_rule_item(check_code=check_code, fn=fn, helpers_src=helpers_src, workspace=workspace, timeout=rule_timeout) if fn else {"score": 0.0, "detail": "gate missing fn"}
        rule_results[it["id"]] = r
        if float(r.get("score", 0.0) or 0.0) < 0.999:
            gate_failed.append(it["id"])

    base_score = base_weight = total_weight = 0.0
    judge_items: list[dict] = []
    per_item: list[dict] = []
    for it in atoms:
        weight = float(it.get("weight", 1.0))
        if it.get("method") == "rule":
            fn = it.get("fn")
            r = run_rule_item(check_code=check_code, fn=fn, helpers_src=helpers_src, workspace=workspace, timeout=rule_timeout) if fn else {"score": 0.0, "detail": "missing fn"}
            rule_results[it["id"]] = r
            score = float(r.get("score", 0.0) or 0.0)
            base_score += weight * score
            base_weight += weight
            total_weight += weight
            per_item.append({"id": it["id"], "method": "rule", "weight": weight, "score": score, "detail": r.get("detail")})
        elif it.get("method") == "llm":
            total_weight += weight
            judge_items.append(
                {"id": it["id"], "weight": weight, "question": it.get("question", ""), "anchor": it.get("pass_anchor", ""), "answer": answer_text}
            )
            per_item.append({"id": it["id"], "method": "llm", "weight": weight})
        # Other methods ("agent", vision-judge) are out of our dataset's scope; skipped.

    force_zero = bool(gate_failed)
    reward = 0.0 if (force_zero or total_weight == 0) else round(base_score / total_weight, 6)

    details = {
        "domain": "general",
        "sub_kind": "general_agent",
        "gate_failed": force_zero,
        "gate_failed_ids": gate_failed,
        "gate_results": {k: v for k, v in rule_results.items() if k in {g["id"] for g in gates}},
        "base_score": base_score,
        "base_weight": base_weight,
        "total_weight": total_weight,
        "items": per_item,
        "infra_error": False,
    }

    judge_request = None
    if judge_items:
        judge_request = {
            "kind": "rubric",
            "base_score": base_score,
            "base_weight": base_weight,
            "total_weight": total_weight,
            "items": judge_items,
        }
        if force_zero:
            judge_request["force_zero"] = True

    return {"reward": reward, "details": details, "judge_request": judge_request}


# ----------------------------------------------------------------------- MCP CLI (exported)

MCP_CLI_SRC = '''#!/usr/bin/env python3
"""/usr/local/bin/mcp <server> <tool> '<json-args>' -- stdlib-only MCP tool-call CLI.

Terminus-2 only has a terminal; this wraps the official `mcp` client SDK's streamable-http
transport so the agent can call one tool on one of the task's local MCP servers with a single
shell command, documented verbatim in each general_agent task's instruction.md.
"""
import asyncio
import json
import sys

PORTS = {{{server_ports}}}


async def _call(server: str, tool: str, args: dict) -> dict:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    port = PORTS.get(server)
    if port is None:
        return {{"error": f"unknown server {{server!r}}; known: {{sorted(PORTS)}}"}}
    url = f"http://127.0.0.1:{{port}}/mcp"
    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return {{
                "is_error": bool(getattr(result, "isError", False)),
                "content": [getattr(c, "text", str(c)) for c in (result.content or [])],
            }}


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: mcp <server> <tool> ['<json-args>']", file=sys.stderr)
        return 2
    server, tool = sys.argv[1], sys.argv[2]
    args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {{}}
    try:
        out = asyncio.run(_call(server, tool, args))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({{"error": f"{{type(e).__name__}}: {{e}}"}}))
        return 1
    print(json.dumps(out, ensure_ascii=False))
    return 1 if out.get("is_error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def render_mcp_cli(server_ports: dict[str, int]) -> str:
    """Render the ``/usr/local/bin/mcp`` CLI wrapper, baked into general_agent task images by
    export.py, with this task's ``{server_name: port}`` map."""
    ports_literal = ", ".join(f"{json.dumps(name)}: {port}" for name, port in sorted(server_ports.items()))
    return MCP_CLI_SRC.format(server_ports=ports_literal)


# --------------------------------------------------------------------------------- export API

_RUN_GRADE_PY = """import json
from pathlib import Path
import sys
sys.path.insert(0, "/tests")
from mimo_general_grader import grade_general_agent

verifier_meta = json.load(open("/tests/verifier_meta.json"))
verifier_meta["_helpers_src"] = open("/tests/_helpers.py").read() if Path("/tests/_helpers.py").is_file() else ""
try:
    answer_text = open({answer_path!r}).read()
except FileNotFoundError:
    answer_text = ""
workspace = Path("__MIMO_WORKSPACE__")
result = grade_general_agent(verifier_meta=verifier_meta, workspace=workspace, answer_text=answer_text)
json.dump(result["details"], open("/logs/verifier/details.json", "w"))
if result["judge_request"] is not None:
    json.dump(result["judge_request"], open("/logs/verifier/judge_request.json", "w"))
open("/logs/verifier/reward.txt", "w").write(str(result["reward"]))
print(f"[general] reward={{result['reward']}} gate_failed={{result['details']['gate_failed']}}")
""".format(answer_path=ANSWER_PATH)
# `workspace` is substituted with a plain string replace (not a second .format() call): the
# script body above already contains literal `{`/`}` (f-strings, dict literals) that survived
# the ANSWER_PATH .format() call precisely because they were double-braced for it; running
# .format() a second time would try (and fail) to substitute THOSE as fields too.

_TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
/usr/bin/python3 /tests/run_grade.py
exit 0
"""


def build_general_agent(row: dict, task_dir: Path, *, env_dir: Path) -> None:
    """Write tests/ for one general_agent task. ``env_dir`` is a local directory already
    holding the HF env dir's files (``verifier_meta.json``, ``_helpers.py``, ``manifest.json``)
    -- fetched by export.py's ``fetch_general_agent_env_dir`` at export time; kept as a
    parameter here so this function has no network dependency and is directly unit-testable."""
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    verifier_meta = json.loads((env_dir / "verifier_meta.json").read_text())
    (tests_dir / "verifier_meta.json").write_text(json.dumps(verifier_meta))
    helpers = env_dir / "_helpers.py"
    if helpers.is_file():
        (tests_dir / "_helpers.py").write_text(helpers.read_text())

    manifest = json.loads((env_dir / "manifest.json").read_text()) if (env_dir / "manifest.json").is_file() else {}
    workspace = manifest.get("cwd", "/work/workspace")

    (tests_dir / "mimo_general_grader.py").write_text(Path(__file__).read_text())
    (tests_dir / "run_grade.py").write_text(_RUN_GRADE_PY.replace("__MIMO_WORKSPACE__", str(workspace)))
    (tests_dir / "test.sh").write_text(_TEST_SH)
    (tests_dir / "test.sh").chmod(0o755)


def build_terminal_bench(row: dict, task_dir: Path) -> None:
    """Write tests/ for one terminal_bench task: ``tests_files`` (base64 JSON dict in
    ``extra_info.instance_json``) decoded byte-for-byte, test.sh run verbatim (it already
    writes ``/logs/verifier/reward.txt`` after its own ``anti_hack_guard``, so there is no
    judge for this sub-kind)."""
    raw = row.get("raw") or {}
    extra_info = raw.get("extra_info") or {}
    instance_json = extra_info.get("instance_json")
    instance = json.loads(instance_json) if isinstance(instance_json, str) else (extra_info or {})
    tests_files_raw = instance.get("tests_files") or extra_info.get("tests_files")
    # Observed row shape: tests_files is a plain JSON-object STRING whose values are
    # base64-encoded file contents (not a base64-encoded JSON blob at the top level).
    tests_files = json.loads(tests_files_raw) if isinstance(tests_files_raw, str) else (tests_files_raw or {})

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, content_b64 in tests_files.items():
        dest = tests_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode(content_b64))
    test_sh = tests_dir / "test.sh"
    if test_sh.is_file():
        test_sh.chmod(0o755)


def dataset_sub_kind(row: dict) -> str:
    raw = row.get("raw") or {}
    extra_info = raw.get("extra_info") or {}
    instance_json = extra_info.get("instance_json")
    instance = json.loads(instance_json) if isinstance(instance_json, str) else (extra_info or {})
    return instance.get("dataset_type") or extra_info.get("dataset_type") or "general_agent"


def build(row: dict, task_dir: Path, *, env_dir_fn=None) -> None:
    """Dispatch to ``build_general_agent`` or ``build_terminal_bench`` by sub-kind.
    ``env_dir_fn(instance_id) -> Path`` supplies the fetched HF env dir for ``general_agent``
    rows (see ``strl/tasks/mimo/export.py::fetch_general_agent_env_dir``); injectable so this
    function has no network dependency in unit tests."""
    if dataset_sub_kind(row) == "terminal_bench":
        build_terminal_bench(row, task_dir)
        return
    if env_dir_fn is None:
        raise ValueError("general_agent rows need env_dir_fn(instance_id) -> local HF env dir path")
    raw = row.get("raw") or {}
    extra_info = raw.get("extra_info") or {}
    instance_id = extra_info.get("instance_id")
    build_general_agent(row, task_dir, env_dir=env_dir_fn(instance_id))
