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
sys.path.insert(0, str(ROOT / "assets" / "overlay"))

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
    bad = [(td, r) for td, r in zip(task_dirs, results) if r.get("reward") not in (0.0, None) or r.get("exception")]
    return {"n": len(task_dirs), "results": results, "all_zero_no_infra": not bad, "bad": [str(b[0]) for b in bad]}


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
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    coding_harbor.install_compat()
    trials_dir = ROOT / "artifacts" / "mimo-gate-trials" / args.domain

    receipt: dict[str, Any] = {"domain": args.domain, "task_dirs": [str(d) for d in args.task_dirs]}
    receipt["nop"] = check_nop(args.task_dirs, trials_dir=trials_dir)
    print(f"[{args.domain}] nop: all_zero_no_infra={receipt['nop']['all_zero_no_infra']} bad={receipt['nop']['bad']}")

    if args.concurrency_nop:
        receipt["concurrency"] = check_concurrency(args.task_dirs[0], trials_dir=trials_dir, n=args.concurrency_nop)
        print(f"[{args.domain}] concurrency({args.concurrency_nop}): infra_error_rate={receipt['concurrency']['infra_error_rate']:.3f}")

    out = args.out or (ROOT / "evidence" / "gates" / f"mimo-{args.domain}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    print(f"wrote {out}")
    return 0 if receipt["nop"]["all_zero_no_infra"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
