"""music grader: single-shot ABC-notation generation scored against the MiMo human-likeness baseline.

The scorer itself (``graders/_vendor/mimo_music_scorer/``) is a byte-for-byte vendor of
XiaomiMiMo/verl @ a2ad9f61 ``recipes/design/music/scorer/*`` (Apache-2.0, see NOTICE in that
directory). This module is our own: it (a) extracts the agent's ABC answer the way the MiMo
recipe does, (b) calls the vendored scorer's ``pipeline.do`` with the task's ``extra_info``
constraints threaded through, (c) clamps to a reward in [0, 1] exactly as MiMo's
``compute_score`` does (``total / 100``, with the reject gate applying first), and (d) writes
the Harbor-contract verifier outputs (``reward.txt`` / ``details.json``).

Reward = MiMo's own human-likeness score. Constraint adherence (bpm / meter / nvoice_want /
length / tag) is NOT part of the reward -- MiMo's scorer does not grade it either -- but we
report it in ``details.json["constraint_adherence"]`` as auxiliary, informational-only signal.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
# Dev tree: graders/_vendor/mimo_music_scorer. Exported task dir (build() below copies the
# vendored package as a sibling of this very file, tests/mimo_music_scorer): tests/.
_VENDOR_CANDIDATES = (_HERE / "_vendor", _HERE)

_FENCE_RE = re.compile(r"```(?:abc|ABC)?\s*\n(.*?)```", re.DOTALL)
_XA_RE = re.compile(r"X\s*:\s*\d+")

# Length bands (MiMo "length" extra_info field) in approximate bar counts; used only for the
# auxiliary constraint-adherence report, never for the reward.
_LENGTH_BANDS = {"short": (0, 72), "medium": (48, 160), "long": (120, 10_000)}

LENGTH_TAG_COUNT = "length_tag_count"


def _vendored_scorer():
    """Import the vendored scorer package: graders/_vendor/mimo_music_scorer in the dev tree,
    or tests/mimo_music_scorer next to this file once ``build()`` has copied it into an
    exported task dir."""
    for candidate in _VENDOR_CANDIDATES:
        if (candidate / "mimo_music_scorer" / "__init__.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break
    from mimo_music_scorer import pipeline  # type: ignore

    return pipeline


def extract_abc(text: str | None) -> str | None:
    """Pull the ABC body out of the agent's reply, exactly as the MiMo recipe does:
    prefer a fenced ```abc block containing an ``X:<n>`` tune header; otherwise fall back to
    everything from the last ``X:<n>`` line onward."""
    pipeline = _vendored_scorer()
    return pipeline.extract_abc(text)


def _count_abc_voices(abc: str) -> int:
    """Count declared voices (``V:`` headers) as a rough nvoice proxy; falls back to 1."""
    voices = {m.group(1) for m in re.finditer(r"^V:\s*(\S+)", abc, re.MULTILINE)}
    return len(voices) if voices else 1


def _constraint_adherence(abc: str, constraints: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    """Auxiliary, non-reward comparison of the produced piece against the task's constraints."""
    out: dict[str, Any] = {}
    bpm_want = constraints.get("bpm")
    if bpm_want:
        bpm_got = measured.get("bpm")
        out["bpm_want"] = bpm_want
        out["bpm_got"] = bpm_got
        out["bpm_within_10pct"] = bool(bpm_got and abs(bpm_got - bpm_want) <= 0.1 * bpm_want)
    meter_want = constraints.get("meter")
    if meter_want:
        meter_got = _abc_meter(abc)
        out["meter_want"] = meter_want
        out["meter_got"] = meter_got
        out["meter_match"] = meter_got == meter_want
    nvoice_want = constraints.get("nvoice_want")
    if nvoice_want:
        n_chan_got = measured.get("n_chan") or _count_abc_voices(abc)
        out["nvoice_want"] = nvoice_want
        out["nvoice_got"] = n_chan_got
        out["nvoice_match"] = n_chan_got == nvoice_want
    length_want = constraints.get("length")
    if length_want and length_want in _LENGTH_BANDS:
        lo, hi = _LENGTH_BANDS[length_want]
        bars_got = _abc_bar_count(abc)
        out["length_want"] = length_want
        out["bars_got"] = bars_got
        out["length_within_band"] = lo <= bars_got <= hi
    if constraints.get("tag"):
        out["tag"] = constraints["tag"]
    return out


def _abc_meter(abc: str) -> str | None:
    m = re.search(r"^M:\s*(\S+)", abc, re.MULTILINE)
    return m.group(1) if m else None


def _abc_bar_count(abc: str) -> int:
    body_lines = [ln for ln in abc.splitlines() if ln and not re.match(r"^[A-Zw]:", ln)]
    body = "\n".join(body_lines)
    return body.count("|")


class MusicGradeResult(dict):
    """{"reward": float, "details": {...}}"""


def grade(
    answer_text: str | None,
    *,
    constraints: dict[str, Any] | None = None,
    abc2midi_bin: str | None = None,
) -> MusicGradeResult:
    """Grade one agent reply. Pure function: no filesystem/Harbor dependency beyond invoking
    ``abc2midi`` (resolved via ``abc2midi_bin`` or ``$ABC2MIDI_BIN``/``$PATH``), so it is
    directly unit-testable.

    Returns ``{"reward": float in [0, 1], "details": {...}}``. ``abc2midi`` missing from PATH
    raises ``Abc2MidiMissing`` (an infra fault, not a 0 -- see the vendored pipeline's
    docstring) rather than returning a score, so the caller can tell "bad music" from
    "toolchain missing".
    """
    import os

    pipeline = _vendored_scorer()
    constraints = constraints or {}
    if abc2midi_bin:
        os.environ.setdefault("ABC2MIDI_BIN", abc2midi_bin)

    abc = extract_abc(answer_text)
    details: dict[str, Any] = {"domain": "music", "extracted": bool(abc), "constraints": constraints}
    if not abc:
        details["validity"] = "no_abc_found"
        return MusicGradeResult(reward=0.0, details=details)

    rec = {
        "key": "mimo_task",
        "id": 0,
        "rep": 0,
        "abc": abc,
        "tag": constraints.get("tag"),
        "lang": constraints.get("lang"),
        "abc_len": len(abc),
        "nvoice": int(constraints.get("nvoice_want") or 0),
        "latency": None,
    }
    r = pipeline.do(rec)  # Abc2MidiMissing propagates: infra fault, not a score.

    details["scorer"] = {k: v for k, v in r.items() if not k.startswith("_")}
    reject = bool(r.get("skip") or r.get("reject", 0))
    details["validity"] = "rejected" if reject else ("skipped:" + str(r["skip"]) if r.get("skip") else "ok")
    details["constraint_adherence"] = _constraint_adherence(abc, constraints, r)

    if reject or "total" not in r:
        return MusicGradeResult(reward=0.0, details=details)

    reward = max(0.0, min(1.0, float(r["total"]) / 100.0))
    details["groups"] = r.get("groups")
    return MusicGradeResult(reward=reward, details=details)


# --------------------------------------------------------------------------------- export API

ANSWER_PATH = "/app/answer.abc"

_TEST_SH = """#!/bin/bash
# reward 0 first: a crashed or killed verifier can never leave a stale pass behind
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
/usr/bin/python3 /tests/run_grade.py
exit 0
"""

_RUN_GRADE_PY = """import json, sys
sys.path.insert(0, "/tests")
from mimo_music_grader import grade

constraints = json.load(open("/tests/constraints.json"))
try:
    answer = open({answer_path!r}).read()
except FileNotFoundError:
    answer = ""
result = grade(answer, constraints=constraints)
json.dump(result["details"], open("/logs/verifier/details.json", "w"))
open("/logs/verifier/reward.txt", "w").write(str(result["reward"]))
print(f"[music] reward={{result['reward']:.4f}} validity={{result['details'].get('validity')}}")
""".format(answer_path=ANSWER_PATH)


def build(row: dict, task_dir: Path) -> None:
    """Write tests/ for one music task dir: the vendored scorer + our grader module +
    constraints.json (from row['raw']['extra_info']) + test.sh."""
    import shutil

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    vendor_dst = tests_dir / "mimo_music_scorer"
    if vendor_dst.exists():
        shutil.rmtree(vendor_dst)
    shutil.copytree(_HERE / "_vendor" / "mimo_music_scorer", vendor_dst)

    grader_src = Path(__file__).read_text()
    (tests_dir / "mimo_music_grader.py").write_text(grader_src)

    extra_info = (row.get("raw") or {}).get("extra_info") or {}
    constraints = {
        "src_id": extra_info.get("src_id"),
        "lang": extra_info.get("lang"),
        "tag": extra_info.get("tag"),
        "length": extra_info.get("length"),
        "nvoice_want": int(extra_info.get("nvoice_want") or 0) or None,
        "bpm": int(extra_info.get("bpm") or 0) or None,
        "meter": extra_info.get("meter"),
    }
    (tests_dir / "constraints.json").write_text(json.dumps(constraints, sort_keys=True))
    (tests_dir / "run_grade.py").write_text(_RUN_GRADE_PY)
    (tests_dir / "test.sh").write_text(_TEST_SH)
    (tests_dir / "test.sh").chmod(0o755)
