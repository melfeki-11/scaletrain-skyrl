"""webdev grader: in-sandbox Playwright/Chromium render + a vision judge hand-off.

Render command and rubric prompt are vendored byte-for-byte from XiaomiMiMo/verl @ a2ad9f61
``recipes/design/webdev/{shot.py,eval_rubric.py}`` (Apache-2.0; see
``graders/_vendor/mimo_webdev/NOTICE``). This module is our own glue: it runs the vendored
shot script in-sandbox (python3, no network egress required -- ``file://`` or a loopback
server serves the agent's own workspace), writes ``/logs/verifier/screenshot.png``, and builds
the judge hand-off.

In-sandbox score contribution is **zero** -- there is no deterministic (rule) part of this
grader, matching the contract ("in-sandbox base_score 0 / base_weight 0 / total_weight per
eval_rubric"): ``base_score = 0.0``, ``base_weight = 0.0``, and the WHOLE score comes from one
vision judge item of weight ``total_weight = 1.0`` whose question is the vendored rubric
prompt (``eval_rubric.build_prompt()``, formatted with this task's original request). The
judge (outside the sandbox, see ``strl/tasks/mimo/__init__.py``) must parse the vision model's
JSON reply with the vendored ``eval_rubric.parse_verdict`` and use its
``score = (visual + query_fulfillment + premium_assets) / 3`` as the single item's score --
**note for judge.py (Track A)**: this is a one-item judge_request with a non-trivial
multi-dimension parse, not a plain scalar question/anchor pair; ``kind == "vision"`` is the
dispatch signal.

A render failure (``render_failed`` / no screenshot / Chromium missing) is reported as
``details.json["render_failed"] = True`` with ``infra_error`` left ``False`` -- "the model
produced nothing renderable" is still a real, scoreable 0, per the contract -- UNLESS the
sandbox exec itself errored (process spawn failure, no Chromium in the image at all), which
is reported as ``infra_error = True`` so the trainer masks a broken environment instead of
training on a false 0.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_VENDOR_CANDIDATES = (_HERE / "_vendor", _HERE)

SCREENSHOT_PATH = "/logs/verifier/screenshot.png"
WORKSPACE_DIR = "/workspace"
ENTRY_PAGE = "index.html"


def _vendored():
    for candidate in _VENDOR_CANDIDATES:
        if (candidate / "mimo_webdev" / "__init__.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break
    from mimo_webdev import eval_rubric, shot  # type: ignore

    return eval_rubric, shot


def render(workspace_dir: str | Path, *, entry_page: str = ENTRY_PAGE, timeout_sec: int = 150) -> dict[str, Any]:
    """Run the vendored shot script against ``<workspace_dir>/<entry_page>``. Returns
    ``{"ok": bool, "screenshot_png": bytes | None, "console_errors": list, "render_env": dict,
    "error": str | None}``. Real subprocess execution -- used at verifier time, not directly by
    offline unit tests (which call ``grade_from_shot`` with a canned shot result instead)."""
    _eval_rubric, shot = _vendored()
    url = f"file://{str(workspace_dir).rstrip('/')}/{entry_page.lstrip('/')}"
    cmd = shot._build_shot_cmd(url)
    try:
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return {"ok": False, "screenshot_png": None, "console_errors": [], "render_env": {}, "error": "shot timed out"}
    if proc.returncode != 0 and "SHOT_B64:" not in proc.stdout:
        return {"ok": False, "screenshot_png": None, "console_errors": [], "render_env": {}, "error": proc.stderr[-2000:]}

    shot_b64, console_errors = shot._parse_shot_b64({"output": proc.stdout})
    render_env = shot._parse_render_env({"output": proc.stdout})
    if not shot_b64:
        return {"ok": False, "screenshot_png": None, "console_errors": json.loads(console_errors), "render_env": render_env, "error": "no screenshot produced"}
    return {
        "ok": True,
        "screenshot_png": base64.b64decode(shot_b64),
        "console_errors": json.loads(console_errors),
        "render_env": render_env,
        "error": None,
    }


def grade_from_shot(shot_result: dict[str, Any], *, query: str) -> dict[str, Any]:
    """Pure function: build the verifier-contract output from an already-produced ``render()``
    result, so it is directly unit-testable without Playwright/Chromium. Returns
    ``{"reward", "details", "judge_request", "screenshot_png"}``."""
    eval_rubric, _shot = _vendored()

    details: dict[str, Any] = {"domain": "webdev", "infra_error": False}
    if shot_result.get("error") and not shot_result.get("ok"):
        if shot_result["error"] in ("shot timed out",) or "no screenshot produced" in str(shot_result.get("error", "")):
            details.update(render_failed=True, render_error=shot_result["error"])
            return {"reward": 0.0, "details": details, "judge_request": None, "screenshot_png": None}
        # exec itself failed (e.g. Chromium missing from the image): a genuine infra fault.
        details.update(infra_error=True, render_failed=True, render_error=shot_result["error"])
        return {"reward": 0.0, "details": details, "judge_request": None, "screenshot_png": None}

    render_env = shot_result.get("render_env") or {}
    if render_env.get("proxy_failed"):
        details.update(render_failed=True, render_env=render_env)
        return {"reward": 0.0, "details": details, "judge_request": None, "screenshot_png": None}

    details.update(
        render_failed=False,
        console_errors=(shot_result.get("console_errors") or [])[:15],
        render_env_recovered=bool(render_env.get("recovered")),
    )

    prompt = eval_rubric.build_prompt().format(query=(query or "")[:1500])
    judge_request = {
        "kind": "vision",
        "base_score": 0.0,
        "base_weight": 0.0,
        "total_weight": 1.0,
        "items": [{"id": "webdev_rubric", "weight": 1.0, "question": prompt, "anchor": ""}],
        "images": [SCREENSHOT_PATH],
        "query": query,
        "rubric": eval_rubric.RUBRIC_ID,
    }
    # reward.txt is the in-sandbox-only portion: base_score / total_weight = 0 / 1 = 0. The
    # judge supplies the whole score for this domain.
    return {"reward": 0.0, "details": details, "judge_request": judge_request, "screenshot_png": shot_result.get("screenshot_png")}


# --------------------------------------------------------------------------------- export API

_RUN_GRADE_PY = """import json
from pathlib import Path
import sys
sys.path.insert(0, "/tests")
from mimo_webdev_grader import render, grade_from_shot

query = json.load(open("/tests/query.json"))["query"]
shot_result = render({workspace!r}, entry_page={entry_page!r})
result = grade_from_shot(shot_result, query=query)
if result["screenshot_png"]:
    open({screenshot!r}, "wb").write(result["screenshot_png"])
json.dump(result["details"], open("/logs/verifier/details.json", "w"))
if result["judge_request"] is not None:
    json.dump(result["judge_request"], open("/logs/verifier/judge_request.json", "w"))
open("/logs/verifier/reward.txt", "w").write(str(result["reward"]))
print(f"[webdev] reward={{result['reward']}} render_failed={{result['details'].get('render_failed')}}")
""".format(workspace=WORKSPACE_DIR, entry_page=ENTRY_PAGE, screenshot=SCREENSHOT_PATH)

_TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
/usr/bin/python3 /tests/run_grade.py
exit 0
"""


def build(row: dict, task_dir: Path) -> None:
    """Write tests/ for one webdev task dir."""
    raw = row.get("raw") or {}
    extra_info = raw.get("extra_info") or {}
    instance_json = extra_info.get("instance_json")
    instance = json.loads(instance_json) if isinstance(instance_json, str) else (extra_info or {})
    query = instance.get("problem_statement") or extra_info.get("problem_statement") or ""
    if not query:
        prompt = raw.get("prompt") or []
        query = prompt[0]["content"] if prompt else ""

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    vendor_dst = tests_dir / "mimo_webdev"
    if vendor_dst.exists():
        shutil.rmtree(vendor_dst)
    shutil.copytree(_HERE / "_vendor" / "mimo_webdev", vendor_dst)

    (tests_dir / "mimo_webdev_grader.py").write_text(Path(__file__).read_text())
    (tests_dir / "query.json").write_text(json.dumps({"query": query}))
    (tests_dir / "run_grade.py").write_text(_RUN_GRADE_PY)
    (tests_dir / "test.sh").write_text(_TEST_SH)
    (tests_dir / "test.sh").chmod(0o755)
