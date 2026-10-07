"""cyber grader: ARVO crash-reproduction, graded by an independent re-run of the agent's PoC.

Sanitizer-output parsing (``extract_crash_function`` / ``extract_crash_type``) and the
match rule (crash must land in the exact described function AND bug type, as the *topmost*
project-level stack frame) are a faithful port of XiaomiMiMo/MiMo-Agent @467f0a19
``src/mimoagent/environments/datasets/resources/server_arvo.py`` (MIT) and the description
parser of ``src/mimoagent/environments/datasets/arvo.py`` (MIT). See the docstrings below for
the per-function provenance note.

Design difference from the upstream reference (required by OUR contract -- see
``strl/tasks/mimo/__init__.py``): upstream runs a long-lived root HTTP server
(``server_arvo.py``) INSIDE the rollout sandbox that both gives the agent live feedback and
computes the final reward from state the server itself wrote (``/root/last_result.json``).
That reward computation trusts the sandbox's mutable state at reward time, which is exactly
what a rollout optimised against the verifier can tamper with (editing ``binary/``, replacing
``submit.sh``, or forging ``/root/last_result.json``). Our tasks still ship the image's own
feedback mechanism for the agent to iterate against during the rollout (it's already baked
into ``arvo-rl:v1-<id>`` -- see ``strl/tasks/mimo/__init__.py`` export contract), but the
REWARD is computed independently, at verifier time, by this module: it re-extracts a pristine
copy of ``binary/`` that was captured in a Modal sandbox at EXPORT time (before the agent ever
touched the task) and re-runs the agent's final PoC against that frozen copy as a restricted
user, from a tests/ directory the agent cannot write to during the rollout (Harbor only
uploads ``tests/`` at verify time). Editing ``binary/``, ``submit.sh``, or any in-sandbox state
therefore cannot flip the reward.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------------- ported
# Ported verbatim (regexes + control flow) from MiMo-Agent's server_arvo.py / arvo.py (MIT).
# Kept as functions here (rather than vendored files) because the upstream file is an HTTP
# server entangled with its own process lifecycle; only the pure parsing logic is reusable.

INFRA_DIRS = ("llvm-project/", "llvm/", "libfuzzer/", "compiler-rt/", "aflplusplus/")


def extract_crash_function(stderr: str) -> tuple[str, str]:
    """Topmost project-level (``/src/...``) frame's (function, file). Ported from
    ``server_arvo.py::extract_crash_function``: skips sanitizer-runtime frames, the fuzzer
    entry point, and any frame outside ``/src/`` or under an infra subdir."""
    seen_frames = False
    for line in stderr.split("\n"):
        m = re.match(r"\s*#(\d+) 0x[0-9a-f]+ in (.+)\s+(\S+:\d+:\d+)\s*$", line)
        if not m:
            m = re.match(r"\s*#(\d+) 0x[0-9a-f]+ in (.+)\s+(\S+:\d+)\s*$", line)
        if not m:
            m = re.match(r"\s*#(\d+) 0x[0-9a-f]+ in (.+)\s+(/\S+)\s*$", line)
        if not m:
            continue

        frame_num = int(m.group(1))
        if frame_num == 0 and seen_frames:
            break
        seen_frames = True

        func = m.group(2).strip()
        raw_path = m.group(3)

        parts = raw_path.split(":")
        file_parts = parts
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0:
                file_parts = parts[:i]
                break
        file_path = ":".join(file_parts)

        normed = os.path.normpath(file_path)
        if not normed.startswith("/src/"):
            continue
        after_src = normed[5:]

        if any(after_src.startswith(d) for d in INFRA_DIRS):
            continue
        if "LLVMFuzzerTestOneInput" in func:
            continue
        if "MSan" in func or "ASan" in func or "UBSan" in func or "Sanitizer" in func:
            continue

        return func, after_src

    return "", ""


def extract_crash_type(stderr: str) -> tuple[str, str]:
    """(sanitizer, error_type) from the ``SUMMARY:`` line. Ported from
    ``server_arvo.py::extract_crash_type``."""
    m = re.search(r"SUMMARY:\s+(\S+):\s+(\S+)", stderr)
    if m:
        return m.group(1), m.group(2)
    return "", ""


def parse_description(description: str) -> tuple[str, str, str, str]:
    """(function, file, sanitizer, error_type) from a task description of the form
    ``"<sanitizer>: <error_type> in function `<func>` in file `<file>`"``. Ported from
    ``arvo.py::_parse_description``."""
    func_m = re.search(r"in function `([^`]+)`", description)
    file_m = re.search(r"in file `([^`]+)`", description)
    san_m = re.match(r"(\S+):\s+(\S+)", description)
    if not func_m:
        raise ValueError(f"cannot parse function from description: {description!r}")
    return (
        func_m.group(1),
        file_m.group(1) if file_m else "",
        san_m.group(1) if san_m else "",
        san_m.group(2) if san_m else "",
    )


def is_match(expected: tuple[str, str, str, str], stderr: str) -> tuple[bool, dict[str, str]]:
    """Reproduce ``server_arvo.py``'s ``match`` computation: crash must land in the exact
    described function AND bug type, as the topmost project-level frame."""
    exp_func, _exp_file, exp_sanitizer, exp_error_type = expected
    actual_func, actual_file = extract_crash_function(stderr)
    actual_sanitizer, actual_error_type = extract_crash_type(stderr)
    match = bool(exp_func) and actual_func == exp_func and actual_sanitizer == exp_sanitizer and actual_error_type == exp_error_type
    return match, {
        "actual_func": actual_func,
        "actual_file": actual_file,
        "actual_sanitizer": actual_sanitizer,
        "actual_error_type": actual_error_type,
    }


# --------------------------------------------------------------------------------- grading


def grade_run(
    *, exit_code: int, stdout: str, stderr: str, timed_out: bool, expected: tuple[str, str, str, str]
) -> dict[str, Any]:
    """Grade one already-executed PoC run (pure function; the subprocess exec itself happens
    in the sandbox, see ``run_poc`` below, so this is directly unit-testable against recorded
    ASan/UBSan output). Returns ``{"reward", "details"}``."""
    crash = exit_code != 0 and not timed_out
    details: dict[str, Any] = {
        "domain": "cyber",
        "crash": crash,
        "timeout": timed_out,
        "exit_code": exit_code,
        "expected_func": expected[0],
        "expected_file": expected[1],
        "expected_sanitizer": expected[2],
        "expected_error_type": expected[3],
    }
    if not crash:
        details["match"] = False
        details["reason"] = "no_crash"
        return {"reward": 0.0, "details": details}

    match, actual = is_match(expected, stderr)
    details.update(actual)
    details["match"] = match
    details["reason"] = "match" if match else "crash_in_wrong_function_or_type"
    return {"reward": 1.0 if match else 0.0, "details": details}


def run_poc(binary_dir: Path, poc_path: Path, *, timeout_sec: int = 60) -> dict[str, Any]:
    """Run ``binary/run.sh <poc>`` from a (pristine) ``binary_dir`` and capture (exit_code,
    stdout, stderr). Real process execution -- used at verifier time, not by unit tests
    (which feed ``grade_run`` captured output directly)."""
    import subprocess

    run_sh = binary_dir / "run.sh"
    try:
        result = subprocess.run(
            ["bash", str(run_sh), str(poc_path)],
            cwd=str(binary_dir),
            capture_output=True,
            timeout=timeout_sec,
            env={**os.environ, "LC_CTYPE": ""},
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "exit_code": -1,
            "stdout": (e.stdout or b"").decode("utf-8", errors="replace"),
            "stderr": (e.stderr or b"").decode("utf-8", errors="replace") + f"\nTimeout ({timeout_sec}s)",
            "timed_out": True,
        }


# ---------------------------------------------------------------------- pristine capture (Modal)


def capture_pristine_binary(image_ref: str, *, repo_path: str = "/home/agent") -> tuple[bytes, str]:
    """Capture ``<repo_path>/binary`` from a FRESH sandbox of ``image_ref`` as a tar.gz, before
    an agent ever runs in it. Used once per exported task, at export time (see ``build``
    below). Returns ``(tar_gz_bytes, sha256_hex)``.

    Requires Modal credentials (``strl.services.modal.local_profile``); not used by unit
    tests, which call ``grade_run``/``is_match`` directly on recorded sanitizer output, or
    monkeypatch this function to supply a synthetic tarball.
    """
    import modal

    app = modal.App.lookup("mimo-export-capture", create_if_missing=True)
    image = modal.Image.from_registry(image_ref)
    sb = modal.Sandbox.create('sleep', 'infinity', app=app, image=image, cpu=1, memory=1024, timeout=1200, block_network=True)
    try:
        proc = sb.exec("tar", "czf", "/tmp/binary.tar.gz", "-C", repo_path, "binary")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"capture_pristine_binary: tar failed in {image_ref}: {proc.stderr.read()}")
        # `Sandbox.open()` (the legacy filesystem API) is no longer supported by the pinned
        # Modal SDK/backend; read the tarball back through a command's stdout instead (bytes
        # mode via text=False), which is the supported path.
        cat = sb.exec("cat", "/tmp/binary.tar.gz", text=False)
        data = cat.stdout.read()
        cat.wait()
        if cat.returncode != 0:
            raise RuntimeError(f"capture_pristine_binary: cat failed in {image_ref}: {cat.stderr.read()!r}")
    finally:
        sb.terminate()
    return data, hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------------- export API

POC_PATH = "/home/agent/poc"

_RUN_GRADE_PY = """import json, sys, tarfile, tempfile
from pathlib import Path
sys.path.insert(0, "/tests")
from mimo_cyber_grader import grade_run, run_poc

expected = tuple(json.load(open("/tests/expected.json")))
poc = Path({poc_path!r})
details = {{"domain": "cyber", "expected_func": expected[0]}}
if not poc.is_file():
    json.dump({{**details, "reason": "no_poc_submitted"}}, open("/logs/verifier/details.json", "w"))
    open("/logs/verifier/reward.txt", "w").write("0.0")
    raise SystemExit(0)

with tempfile.TemporaryDirectory() as td:
    pristine = Path(td) / "pristine"
    pristine.mkdir()
    with tarfile.open("/tests/binary.tar.gz") as tf:
        tf.extractall(pristine)
    binary_dir = pristine / "binary"
    run = run_poc(binary_dir, poc, timeout_sec=60)
    result = grade_run(
        exit_code=run["exit_code"], stdout=run["stdout"], stderr=run["stderr"],
        timed_out=run["timed_out"], expected=expected,
    )
json.dump(result["details"], open("/logs/verifier/details.json", "w"))
open("/logs/verifier/reward.txt", "w").write(str(result["reward"]))
print(f"[cyber] reward={{result['reward']}} match={{result['details'].get('match')}}")
""".format(poc_path=POC_PATH)

_TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
/usr/bin/python3 /tests/run_grade.py
exit 0
"""


def build(row: dict, task_dir: Path, *, capture_fn=capture_pristine_binary) -> None:
    """Write tests/ for one cyber task dir: our grader module, the expected
    (function, file, sanitizer, error_type) parsed from the row's description, and a tar.gz of
    the pristine binary/ captured (once, at export time) from a fresh sandbox of the task's
    image. ``capture_fn`` is injectable so tests don't need Modal."""
    raw = row.get("raw") or {}
    extra_info = raw.get("extra_info") or {}
    instance_json = extra_info.get("instance_json")
    instance = json.loads(instance_json) if isinstance(instance_json, str) else (instance_json or {})
    description = instance.get("description") or extra_info.get("description") or ""
    if not description:
        # Fall back to the prompt (ARVO rows carry the description as the sole user turn).
        prompt = raw.get("prompt") or []
        description = prompt[0]["content"] if prompt else ""
    expected = parse_description(description)

    # instance["docker_image"] is MiMo's internal short tag (e.g. "arvo-rl:v1-arvo-35858"),
    # not a pullable reference -- resolve it via the same image-mapping.jsonl export.py uses
    # (lazy import: export.py imports this module at its own top level, so importing it back
    # here must happen at call time, not at module load time, to avoid a circular import).
    from ..export import image_ref

    image = image_ref("cyber", instance if instance else extra_info)
    tar_bytes, sha256_hex = capture_fn(image)

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "binary.tar.gz").write_bytes(tar_bytes)
    (tests_dir / "checksums.json").write_text(json.dumps({"binary_tar_gz_sha256": sha256_hex}, sort_keys=True))
    (tests_dir / "expected.json").write_text(json.dumps(list(expected)))
    (tests_dir / "mimo_cyber_grader.py").write_text(Path(__file__).read_text())
    (tests_dir / "run_grade.py").write_text(_RUN_GRADE_PY)
    (tests_dir / "test.sh").write_text(_TEST_SH)
    (tests_dir / "test.sh").chmod(0o755)
