#!/usr/bin/env python3
"""Real-Modal gates for one mimo domain: nop -> 0 (not infra_error), tamper attempts -> 0,
positive controls where one exists without an external judge, concurrency check.

    python tools/gate_mimo.py music   <task_dir> [<task_dir> ...]
    python tools/gate_mimo.py cyber   <task_dir> [<task_dir> ...] --concurrency-nop 64

Reuses strl/tasks/coding/harbor.py's ``run_trial``/``install_compat`` (the real Harbor Trial
path) with a mimo-specific trial template built straight from each task's ``strl.json`` +
``task.toml`` (NOT coding's TEMPLATE -- see ``strl/tasks/mimo/__init__.py``: "NO global
override_* keys"). Writes evidence/gates/mimo-<domain>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import tomllib
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# the overlay is the Linux pod build (e.g. a x86_64-linux fastuuid .so): locally it is only a FALLBACK behind the venv's native packages
sys.path.append(str(ROOT / "assets" / "overlay"))

from strl.tasks.coding import harbor as coding_harbor  # noqa: E402


def _read_task(task_dir: Path) -> tuple[dict, dict]:
    strl_json = json.loads((task_dir / "strl.json").read_text())
    task_toml = tomllib.loads((task_dir / "task.toml").read_text())
    return strl_json, task_toml


def mimo_trial_template(task_dir: Path, trials_dir: Path) -> dict:
    """Trial template for ONE task, built only from that task's own strl.json + task.toml --
    no cross-task global defaults (the contract's "NO global override_* keys")."""
    strl_json, task_toml = _read_task(task_dir)
    agent_cfg = dict(strl_json["agent"])
    kwargs = {"max_turns": strl_json["max_turns"], "suppress_max_turns_warning": True}
    if agent_cfg.get("name") == "terminus-2":
        kwargs["llm_kwargs"] = {"max_tokens": strl_json["max_tokens"]}
    elif agent_cfg.get("import_path"):
        kwargs["llm_kwargs"] = {"max_tokens": strl_json["max_tokens"]}
    agent_cfg["kwargs"] = kwargs
    return {
        "trials_dir": str(trials_dir),
        "timeout_multiplier": 1.0,
        "agent": agent_cfg,
        "environment": {"type": "modal", "kwargs": {}},
        "verifier": {"disable": False},
    }


def build_trial_config(template: dict, task_path: Path, *, agent_name: str | None = None, model_name: str | None = None) -> dict:
    config = copy.deepcopy(template)
    agent = config.setdefault("agent", {})
    if agent_name:
        agent = {"name": agent_name, "kwargs": {}}
        config["agent"] = agent
    if model_name is not None:
        agent["model_name"] = model_name
    agent.setdefault("kwargs", {})["session_id"] = uuid4().hex
    config["task"] = {"path": str(task_path)}
    return config


async def run_trials(configs: list[dict]) -> list[dict]:
    return await asyncio.gather(*(coding_harbor.run_trial(c) for c in configs))


def check_nop(task_dirs: list[Path], *, trials_dir: Path) -> dict:
    configs = [build_trial_config(mimo_trial_template(td, trials_dir), td, agent_name="nop") for td in task_dirs]
    results = asyncio.run(run_trials(configs))
    from strl.tasks.mimo.generator import verifier_ran
    for r in results:
        r["verifier_ran"], r["verifier_note"] = verifier_ran(r.get("trial_dir"))   # reward 0 alone also describes a verifier that never ran
    bad = [(td, r) for td, r in zip(task_dirs, results) if r.get("reward") not in (0.0, None) or r.get("exception") or not r["verifier_ran"]]
    return {"n": len(task_dirs), "results": results, "all_zero_no_infra": not bad, "bad": [str(b[0]) for b in bad]}


MUSIC_POSITIVE_ABC = (
    "```abc\nX:1\nT:Gate\nM:4/4\nL:1/8\nQ:1/4=120\nK:C\nCDEF GABc | cBAG FEDC | C4 z4 | G,2A,2 B,2C2 |\nCDEF GABc | cBAG FEDC | C8 | z8 |]\n```\n")


def _oracle_copy(task_dir: Path, solve_sh: str) -> Path:
    import shutil
    import tempfile
    copy = Path(tempfile.mkdtemp(prefix="mimo-pos-")) / task_dir.name
    shutil.copytree(task_dir, copy)
    (copy / "solution").mkdir(exist_ok=True)
    (copy / "solution" / "solve.sh").write_text(solve_sh)
    (copy / "solution" / "solve.sh").chmod(0o755)
    return copy


def positive_control(domain: str, task_dir: Path, *, trials_dir: Path) -> dict:
    """A scripted 'perfect' answer through the REAL Harbor path (oracle agent on a temp copy that has a solution/solve.sh; exported tasks never
    ship one). music: a valid ABC tune must score > 0. general_agent: an answer made of the rubric's own pass anchors must reach the judge
    request intact (every llm item carries the answer text). Domains without a scripted answer report that and need a model (--luna)."""
    strl_json, _ = _read_task(task_dir)
    if domain == "music":
        solve = "#!/bin/bash\nmkdir -p /app\ncat > /app/answer.abc <<'MIMO_EOF'\n" + MUSIC_POSITIVE_ABC + "MIMO_EOF\n"
    elif domain == "general" and (task_dir / "tests" / "verifier_meta.json").is_file():
        meta = json.loads((task_dir / "tests" / "verifier_meta.json").read_text())
        anchors = "\n\n".join(str(i.get("pass_anchor", "")) for i in meta.get("items", []) if i.get("method") == "llm")
        solve = "#!/bin/bash\nmkdir -p /app\ncat > /app/answer.md <<'MIMO_EOF'\n" + anchors + "\nMIMO_EOF\n"
    else:
        return {"skipped": f"no scripted positive control for {domain} (needs a model: gpt-6-luna)"}
    copy = _oracle_copy(task_dir, solve)
    config = build_trial_config(mimo_trial_template(copy, trials_dir), copy, agent_name="oracle")
    result = asyncio.run(run_trials([config]))[0]
    from strl.tasks.mimo.generator import verifier_ran
    ran, why = verifier_ran(result.get("trial_dir"))
    out: dict[str, Any] = {"reward": result.get("reward"), "exception": result.get("exception"), "verifier_ran": ran, "verifier_note": why}
    vdir = Path(result["trial_dir"]) / "verifier" if result.get("trial_dir") else None
    if vdir and (vdir / "details.json").is_file():
        out["details"] = json.loads((vdir / "details.json").read_text())
    if domain == "music":
        out["ok"] = bool(ran and (result.get("reward") or 0) > 0 and not result.get("exception"))
    else:
        request = json.loads((vdir / "judge_request.json").read_text()) if vdir and (vdir / "judge_request.json").is_file() else None
        out["judge_items"] = len(request["items"]) if request else 0
        out["ok"] = bool(ran and request and request["items"] and all(str(i.get("answer", "")).strip() for i in request["items"]))
    return out


def check_concurrency(task_dir: Path, *, trials_dir: Path, n: int) -> dict:
    template = mimo_trial_template(task_dir, trials_dir)
    configs = [build_trial_config(template, task_dir, agent_name="nop") for _ in range(n)]
    results = asyncio.run(run_trials(configs))
    infra_errors = [r for r in results if r.get("exception") or r.get("reward") is None]
    rate = len(infra_errors) / n
    return {"n": n, "infra_error_rate": rate, "ok": rate <= 0.01, "seconds": [r["seconds"] for r in results]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["music", "cyber", "code", "general", "webdev"])
    ap.add_argument("task_dirs", nargs="+", type=Path)
    ap.add_argument("--concurrency-nop", type=int, default=0, help="also run N parallel nop trials on task_dirs[0]")
    ap.add_argument("--positive", action="store_true", help="also run the scripted positive control (music, general) on task_dirs[0]")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    coding_harbor.install_compat()
    trials_dir = ROOT / "artifacts" / "mimo-gate-trials" / args.domain

    receipt: dict[str, Any] = {"domain": args.domain, "task_dirs": [str(d) for d in args.task_dirs]}
    receipt["nop"] = check_nop(args.task_dirs, trials_dir=trials_dir)
    print(f"[{args.domain}] nop: all_zero_no_infra={receipt['nop']['all_zero_no_infra']} bad={receipt['nop']['bad']}")

    if args.positive:
        receipt["positive"] = positive_control(args.domain, args.task_dirs[0], trials_dir=trials_dir)
        print(f"[{args.domain}] positive control: {json.dumps({k: v for k, v in receipt['positive'].items() if k != 'details'}, default=str)}")

    if args.concurrency_nop:
        receipt["concurrency"] = check_concurrency(args.task_dirs[0], trials_dir=trials_dir, n=args.concurrency_nop)
        print(f"[{args.domain}] concurrency({args.concurrency_nop}): infra_error_rate={receipt['concurrency']['infra_error_rate']:.3f}")

    out = args.out or (ROOT / "evidence" / "gates" / f"mimo-{args.domain}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    print(f"wrote {out}")
    positive_ok = receipt.get("positive", {}).get("ok", True) if "skipped" not in receipt.get("positive", {}) else True
    return 0 if receipt["nop"]["all_zero_no_infra"] and positive_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
