"""code grader: SWE-style ``test_patch`` + ``test_command``, exit-code verdict.

A faithful port of XiaomiMiMo/MiMo-Agent @467f0a19
``src/mimoagent/environments/datasets/opensource_code.py`` (MIT) + the shared
``build_reset_test_files_cmd`` / ``get_patch_touched_files`` helpers from
``src/mimoagent/environments/datasets/base.py`` (MIT). See each function's docstring for the
exact provenance.

Reward contract (order matters, exactly as upstream):
  1. reset every path ``test_patch`` touches back to the base ref
  2. ``git apply test_patch``
  3. run ``test_command``; reward = 1.0 iff it exits 0

Deviation from upstream (required by our export-time-only contract -- see
``strl/tasks/mimo/__init__.py``): upstream captures ``base_ref`` via ``git rev-parse HEAD`` at
ROLLOUT-SETUP time, inside the same long-lived pod that later computes the reward. Our
contract computes the verifier from a tests/ directory that is materialised fresh at verify
time (Harbor's model), so ``base_ref`` is captured once at EXPORT time instead, in a one-shot
Modal sandbox of the task's image (see ``compute_base_ref``), and baked into
``tests/base_ref.txt``. A testbed-corruption failure (reset or apply fails) is reported as
``details.json["error_category"] == REWARD_TESTBED_CORRUPTED`` with reward 0, matching
upstream's masking convention (the RL trainer masks this trial instead of training on a false
zero) rather than conflating it with "the agent's patch did not pass tests".
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

REWARD_TESTBED_CORRUPTED = "reward/testbed_corrupted"


def get_patch_touched_files(patch: str) -> list[str]:
    """All file paths a git patch touches, old and new side. Ported from
    ``base.py::DatasetEnvironment.get_patch_touched_files``."""
    files: list[str] = []
    for line in patch.split("\n"):
        m = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if m:
            for fp in (m.group(1), m.group(2)):
                if fp not in files:
                    files.append(fp)
    return files


def build_reset_test_files_cmd(test_patch: str, base_commit: str) -> str | None:
    """Bash that resets every ``test_patch``-touched path to a clean, base-aligned starting
    point before the gold ``test_patch`` is applied. Ported verbatim from
    ``base.py::DatasetEnvironment.build_reset_test_files_cmd``:

      exists in base   -> ``git checkout <base> -- <path>``
      absent from base -> gold-added (possibly agent-planted): unstage + rm

    Returns ``None`` when there is nothing to reset (no touched files, or no base commit).
    """
    touched = get_patch_touched_files(test_patch)
    if not (touched and base_commit):
        return None
    files_blob = "\n".join(touched)
    return (
        "while IFS= read -r tf; do\n"
        '  [ -z "$tf" ] && continue\n'
        f'  if git cat-file -e {base_commit}:"$tf" 2>/dev/null; then\n'
        f'    git checkout {base_commit} -- "$tf" 2>/dev/null || true\n'
        "  else\n"
        '    git rm -f --cached "$tf" >/dev/null 2>&1 || true\n'
        '    rm -f "$tf"\n'
        "  fi\n"
        "done <<'EOF_RESET_TEST_FILES'\n"
        f"{files_blob}\n"
        "EOF_RESET_TEST_FILES\n"
    )


# --------------------------------------------------------------------------------- grading


def grade(
    *,
    repo_path: str | Path,
    test_patch: str,
    test_command: str,
    base_ref: str,
    timeout_sec: float = 1800,
) -> dict[str, Any]:
    """Run the three-step reward recipe against a real git checkout at ``repo_path`` (used by
    unit tests against a throwaway repo, and by ``run_grade.py`` inside the sandbox at verify
    time). Returns ``{"reward", "details"}``; never raises -- a reset/apply failure is reported
    as ``REWARD_TESTBED_CORRUPTED`` rather than propagating."""
    repo_path = Path(repo_path)

    reset_cmd = build_reset_test_files_cmd(test_patch, base_ref)
    if reset_cmd is not None:
        reset = subprocess.run(["bash", "-c", reset_cmd], cwd=repo_path, capture_output=True, text=True)
        if reset.returncode != 0:
            return {
                "reward": 0.0,
                "details": {
                    "domain": "code",
                    "error_category": REWARD_TESTBED_CORRUPTED,
                    "error": "reset_tests_failed",
                    "output": reset.stdout + reset.stderr,
                },
            }

    patch_file = repo_path / ".mimo_test_patch.diff"
    patch_file.write_text(test_patch)
    try:
        applied = subprocess.run(
            ["git", "apply", "--verbose", str(patch_file)], cwd=repo_path, capture_output=True, text=True
        )
    finally:
        patch_file.unlink(missing_ok=True)
    if applied.returncode != 0:
        return {
            "reward": 0.0,
            "details": {
                "domain": "code",
                "error_category": REWARD_TESTBED_CORRUPTED,
                "error": "apply_test_patch_failed",
                "output": applied.stdout + applied.stderr,
            },
        }

    start = time.time()
    try:
        result = subprocess.run(
            ["bash", "-c", test_command], cwd=repo_path, capture_output=True, text=True, timeout=timeout_sec
        )
        rc, output, timed_out = result.returncode, result.stdout + result.stderr, False
    except subprocess.TimeoutExpired as e:
        rc = 124
        output = (e.stdout or "") + (e.stderr or "") + f"\n[timeout after {timeout_sec}s]"
        timed_out = True
    duration = time.time() - start

    diff = subprocess.run(["git", "diff"], cwd=repo_path, capture_output=True, text=True).stdout

    reward = 1.0 if rc == 0 else 0.0
    return {
        "reward": reward,
        "details": {
            "domain": "code",
            "verifier_returncode": rc,
            "resolved": reward == 1.0,
            "timed_out": timed_out,
            "test_duration": duration,
            "test_command": test_command,
            "test_output_tail": output[-4000:],
            "agent_diff": diff,
            "infra_error": False,
        },
    }


# ---------------------------------------------------------------------- Modal-backed helpers


def compute_base_ref(image_ref: str, cwd: str, attempts: int = 3) -> str:
    """``git rev-parse HEAD`` in a fresh sandbox of ``image_ref`` -- the task's immutable base
    commit, captured once at export time (see ``build`` below). Requires Modal; not used by
    unit tests, which pass ``base_ref`` directly to ``grade``.

    The sandbox lifetime counts from creation and these images are ~3 GB: with a short lifetime the pull alone consumed it and the
    first exec failed with "Sandbox is shutting down" (seen on real code tasks), so the lifetime covers the pull and a failed
    attempt is retried."""
    import modal

    app = modal.App.lookup("mimo-export-capture", create_if_missing=True)
    last: Exception | None = None
    for _ in range(attempts):
        sb = None
        try:
            image = modal.Image.from_registry(image_ref)
            sb = modal.Sandbox.create('sleep', 'infinity', app=app, image=image, cpu=1, memory=4096, timeout=1200, block_network=True)   # explicit entrypoint: the image's own CMD exits (status 128)
            proc = sb.exec("git", "rev-parse", "HEAD", workdir=cwd)
            out = proc.stdout.read().strip()
            proc.wait()
            if proc.returncode != 0 or len(out) != 40:
                raise RuntimeError(f"compute_base_ref: 'git rev-parse HEAD' failed in {image_ref}:{cwd}: {out!r}")
            return out
        except Exception as error:   # transient sandbox lifecycle errors (ConflictError, shutting down) are retried
            last = error
        finally:
            if sb is not None:
                try:
                    sb.terminate()
                except Exception:
                    pass
    raise RuntimeError(f"compute_base_ref failed after {attempts} attempts for {image_ref}: {type(last).__name__}: {last}")


def screen_task(row: dict, *, run_fn) -> dict[str, Any]:
    """Reject-at-export-time check (contract item 4): run the task's ``test_command`` against
    the UNMODIFIED repo (no agent, no model patch) and confirm it does NOT already pass -- a
    "nop must fail" sanity check on the task itself, not on any agent. ``run_fn(image_ref, cwd,
    command, timeout) -> returncode`` is injected (a real Modal sandbox run in production,
    a canned stub in unit tests) so this function has no hard Modal dependency.

    Returns ``{"ok": bool, "returncode": int}``; ``ok=False`` means the row should be excluded
    from the exported set and reported, per the contract ("Reject ... report, do not silently
    pass").
    """
    raw = row.get("raw") or {}
    extra_info = raw.get("extra_info") or {}
    instance_json = extra_info.get("instance_json")
    instance = json.loads(instance_json) if isinstance(instance_json, str) else (extra_info or {})
    from ..export import image_ref

    image = image_ref("code", instance if instance else extra_info)
    cwd = instance.get("cwd") or extra_info.get("cwd")
    test_command = instance.get("test_command") or extra_info.get("test_command")
    timeout_sec = float(instance.get("verifier_timeout_sec") or extra_info.get("verifier_timeout_sec") or 1800)

    rc = run_fn(image, cwd, test_command, timeout_sec)
    # The task is usable only if the UNMODIFIED repo (no test_patch applied) fails its own
    # hidden test command -- otherwise every agent (including nop) would score 1.0.
    return {"ok": rc != 0, "returncode": rc}


# --------------------------------------------------------------------------------- export API

_RUN_GRADE_PY = """import json
from pathlib import Path
import sys
sys.path.insert(0, "/tests")
from mimo_code_grader import grade

cfg = json.load(open("/tests/task.json"))
result = grade(
    repo_path=cfg["cwd"],
    test_patch=open("/tests/test_patch.diff").read(),
    test_command=cfg["test_command"],
    base_ref=open("/tests/base_ref.txt").read().strip(),
    timeout_sec=cfg["verifier_timeout_sec"],
)
json.dump(result["details"], open("/logs/verifier/details.json", "w"))
open("/logs/verifier/reward.txt", "w").write(str(result["reward"]))
print(f"[code] reward={result['reward']} infra_error={result['details'].get('infra_error', result['details'].get('error_category') is not None)}")
"""

_TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
/usr/bin/python3 /tests/run_grade.py
exit 0
"""


def build(row: dict, task_dir: Path, *, base_ref_fn=compute_base_ref) -> None:
    """Write tests/ for one code task dir. ``base_ref_fn`` is injectable so tests don't need
    Modal."""
    raw = row.get("raw") or {}
    extra_info = raw.get("extra_info") or {}
    instance_json = extra_info.get("instance_json")
    instance = json.loads(instance_json) if isinstance(instance_json, str) else (extra_info or {})

    # instance["docker_image"] is MiMo's internal short tag (e.g. "format-code-task-001457:
    # latest"), not a pullable reference -- resolve it the same way export.py does (lazy
    # import to avoid a circular import: export.py imports this module at its own top level).
    from ..export import image_ref

    image = image_ref("code", instance if instance else extra_info)
    cwd = instance.get("cwd") or extra_info.get("cwd") or "/testbed"
    test_patch = instance.get("test_patch") or extra_info.get("test_patch") or ""
    test_command = instance.get("test_command") or extra_info.get("test_command") or ""
    verifier_timeout_sec = float(instance.get("verifier_timeout_sec") or extra_info.get("verifier_timeout_sec") or 1800)

    base_ref = base_ref_fn(image, cwd)

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_patch.diff").write_text(test_patch)
    (tests_dir / "base_ref.txt").write_text(base_ref + "\n")
    (tests_dir / "task.json").write_text(
        json.dumps({"cwd": cwd, "test_command": test_command, "verifier_timeout_sec": verifier_timeout_sec}, sort_keys=True)
    )
    (tests_dir / "mimo_code_grader.py").write_text(Path(__file__).read_text())
    (tests_dir / "run_grade.py").write_text(_RUN_GRADE_PY)
    (tests_dir / "test.sh").write_text(_TEST_SH)
    (tests_dir / "test.sh").chmod(0o755)
