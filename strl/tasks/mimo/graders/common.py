"""The verifier entrypoint every mimo task shares (written by export.export_task after the domain grader built its files).

Why one shared `test.sh` instead of five copies: the review of the first merged version found problems that five copies had each:
  * `/usr/bin/python3` does not exist in python:3.12-slim (music) or general-agent-env-0: the grader never ran and the no-op gate
    passed vacuously (reward 0 is also what a correct grader gives a no-op). The interpreter is now found on PATH, and a grader that
    produced no `details.json` is an INFRA error (masked), never a silent 0.
  * Harbor prefers `/logs/verifier/reward.json` over `reward.txt`, and the agent may be root: anything it left in the verifier dir
    (reward.json, a stale judge_request.json) is wiped first.
  * `/tests` is overlaid, not replaced: an agent-planted `/tests/json.py` would shadow the stdlib in the grader. Files not listed in
    `MANIFEST.txt` are deleted before the grader runs.
  * The artifacts named in strl.json (answer.abc, poc, answer.md, ...) were never copied into the verifier dir, so W&B tables,
    `invalid_answer_rate` and the offline re-grade in `strl verify` all saw nothing. They are copied (size-capped) here.
Terminal-bench tasks ship their own `test.sh`; it is kept as `tb_test.sh` and run inside this wrapper.
"""
from __future__ import annotations

from pathlib import Path

# ``{entry}`` is the shell command that runs the domain grader.
_TEST_SH = """#!/bin/bash
# Shared mimo verifier entrypoint (strl/tasks/mimo/graders/common.py).
V=/logs/verifier
mkdir -p "$V"
# 1. wipe whatever the agent left in the verifier dir (Harbor prefers reward.json over reward.txt); keep Harbor's own capture files
find "$V" -mindepth 1 ! -name 'test-stdout.txt' ! -name 'test-stderr.txt' -delete 2>/dev/null
# 2. reward 0 first: a crashed or killed verifier can never leave a stale pass behind
echo 0 > "$V/reward.txt"
# 3. only the files we shipped may exist in /tests (an agent with a root shell can plant modules that shadow the stdlib)
if [ -f /tests/MANIFEST.txt ]; then
  find /tests -type f | while IFS= read -r f; do grep -qxF -- "${{f#/tests/}}" /tests/MANIFEST.txt || rm -f -- "$f"; done
fi
# 4. the images differ: python:3.12-slim and general-agent-env-0 only have /usr/local/bin/python3
PY=$(command -v python3 || command -v /usr/local/bin/python3 || command -v /usr/bin/python3)
if [ -z "$PY" ]; then
  echo '{{"infra_error": true, "error": "no python3 in the verifier image"}}' > "$V/details.json"
  exit 0
fi
{entry}
"$PY" /tests/mimo_post.py
exit 0
"""

# stdlib only; runs after the grader in the same sandbox.
POST_PY = '''"""Normalize the grader's output and copy the agent's artifacts next to it (see graders/common.py)."""
import json
import os
import shutil

V = "/logs/verifier"
CAP = 2 * 1024 * 1024


def number(path):
    try:
        return float(open(path).read().strip())
    except Exception:
        return None


reward = number(V + "/reward.txt")
if os.environ.get("MIMO_TB") and not os.path.isfile(V + "/details.json"):
    # terminal_bench ships its own tests, which write reward.txt only. They ran if pytest left its CTRF report (the agent cannot have
    # planted it: the verifier dir was wiped first) or if their anti-hack guard rejected the workspace (reward 0, a real failure).
    try:
        output = open(V + "/tb-output.txt", errors="replace").read()
    except Exception:
        output = ""
    guard = "anti_hack_guard: REJECT" in output
    if guard or os.path.isfile(V + "/ctrf.json"):
        body = {"metric": 0.0 if guard else (reward or 0.0), "components": {"tests": reward or 0.0}, "guard_rejected": guard}
    else:
        body = {"infra_error": True, "error": "terminal_bench tests did not run (no ctrf.json and no guard verdict): " + output[-300:]}
    json.dump(body, open(V + "/details.json", "w"))
try:
    details = json.load(open(V + "/details.json"))
except FileNotFoundError:
    details = {"infra_error": True, "error": "the grader produced no details.json (it crashed or never ran)"}
except Exception as error:
    details = {"infra_error": True, "error": "unreadable details.json: %s" % error}
if not isinstance(details, dict):
    details = {"infra_error": True, "error": "details.json is not an object"}
details.setdefault("infra_error", False)
details.setdefault("components", {})
if details.get("metric") is None:
    details["metric"] = reward if reward is not None else 0.0
details.setdefault("passed", bool(details["metric"] is not None and float(details["metric"]) >= 0.999))
try:
    for line in open("/tests/artifacts.txt"):
        path = line.strip()
        if path and os.path.isfile(path) and os.path.getsize(path) <= CAP:
            shutil.copyfile(path, os.path.join(V, os.path.basename(path)))
except Exception:
    pass
diff = details.get("agent_diff")
if isinstance(diff, str) and diff:
    open(V + "/patch.diff", "w").write(diff[:200000])
json.dump(details, open(V + "/details.json", "w"))
'''

DEFAULT_ENTRY = '"$PY" /tests/run_grade.py'
TERMINAL_BENCH_ENTRY = 'export MIMO_TB=1\nbash /tests/tb_test.sh 2>&1 | tail -c 30000 > "$V/tb-output.txt"'


def test_sh(entry: str = DEFAULT_ENTRY) -> str:
    return _TEST_SH.format(entry=entry)


def finalize_tests(task_dir: Path, artifacts: list[str], *, terminal_bench: bool = False) -> None:
    """Install the shared entrypoint into ``task_dir/tests`` (call last: the manifest lists every file that exists then)."""
    tests = task_dir / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    entry = DEFAULT_ENTRY
    if terminal_bench:   # their test.sh (it calls their own anti_hack_guard) runs inside our wrapper
        own = tests / "test.sh"
        if own.is_file():
            own.replace(tests / "tb_test.sh")
            (tests / "tb_test.sh").chmod(0o755)
        entry = TERMINAL_BENCH_ENTRY
    (tests / "test.sh").write_text(test_sh(entry))
    (tests / "test.sh").chmod(0o755)
    (tests / "mimo_post.py").write_text(POST_PY)
    (tests / "artifacts.txt").write_text("".join(f"{p}\n" for p in artifacts))
    (tests / "MANIFEST.txt").write_text("")   # listed below, once every file exists
    files = sorted(str(p.relative_to(tests)) for p in tests.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    (tests / "MANIFEST.txt").write_text("".join(f"{f}\n" for f in files))
