"""In-loop paired result for environments whose agents cannot run in the offline evaluator (``evaluation.final_eval: false``, e.g. mimo).

The held-out evaluation happens INSIDE training (EvidenceCallback.on_eval_end appends every eval's records to
``<export_path>/evaluation_samples.jsonl`` with a ``step``). Base = the step-0 eval, trained = the LAST eval, run by the same engines,
agent and chat template. There is no checkpoint selection: the final step is the trained model. If the run stopped before a second eval
exists the result records ``paired: null`` with a reason; a pair is never fabricated.

Outputs (the file names and shapes `strl verify` / `strl compare` already read): final_eval/{base.jsonl, candidate.jsonl, result.json,
README.md}. result.json carries 'test', 'by_source', 'flips' where the environment can compute them, plus 'domains', 'macro', 'dropped'.
"""
from __future__ import annotations

import json
from pathlib import Path

from .finalize import read_jsonl, write_jsonl


def eval_rows_by_step(path) -> dict[int, list[dict]]:
    """evaluation_samples.jsonl -> {step: rows}. A (step, sample_id) written twice (a resumed run) keeps the last row."""
    by_step: dict[int, dict[str, dict]] = {}
    if not Path(path).is_file():
        return {}
    for row in read_jsonl(path):
        by_step.setdefault(int(row['step']), {})[str(row['sample_id'])] = row
    return {step: [rows[k] for k in sorted(rows)] for step, rows in sorted(by_step.items())}


def select_pair(by_step: dict[int, list]) -> tuple[int | None, int | None, str | None]:
    """(base_step, trained_step, reason). Base must be the step-0 eval, trained the largest later eval."""
    steps = sorted(by_step)
    if not steps:
        return None, None, 'no evaluation rows were recorded'
    if steps[0] != 0:
        return None, steps[-1], f'no step-0 (base) evaluation: the first recorded eval is step {steps[0]}'
    if len(steps) < 2:
        return 0, None, 'training stopped before a second evaluation: only the step-0 (base) eval exists'
    return 0, steps[-1], None


def in_loop_result(cfg: dict, env_spec, evaluation_samples, eval_dir, *, optimizer_steps: int) -> dict:
    """Pair the first and last evals, write the final_eval files and return the result dict (``paired`` = the step pair used, or None with ``reason``;
    the environment's statistics are merged at the top level: test, by_source, flips, domains, macro, dropped)."""
    by_step = eval_rows_by_step(evaluation_samples)
    base_step, trained_step, reason = select_pair(by_step)
    result: dict = {'mode': 'in_loop', 'environment': env_spec.name, 'metric': env_spec.metric, 'experiment': cfg['name'],
                    'eval_steps': sorted(by_step), 'optimizer_steps': optimizer_steps, 'base_step': base_step, 'trained_step': trained_step,
                    'selected_step': trained_step, 'selection': 'last_step: no checkpoint selection, the final step is the trained model',
                    'evaluation': 'in-loop (same engines, agent and template for base and trained)', 'paired': None, 'reason': reason}
    Path(eval_dir).mkdir(parents=True, exist_ok=True)
    if reason is None:
        base, trained = by_step[base_step], by_step[trained_step]
        try:
            stats = env_spec.paired_result(base, trained, cfg)
        except Exception as error:   # never fabricate and never break completion: record why there is no pair
            stats, result['reason'] = None, f'paired statistics failed: {type(error).__name__}: {error}'[:300]
        if stats is not None:
            write_jsonl(Path(eval_dir) / 'base.jsonl', base)
            write_jsonl(Path(eval_dir) / 'candidate.jsonl', trained)
            result.update(stats)
            result['paired'] = {'base_step': base_step, 'trained_step': trained_step, 'base_rows': len(base), 'trained_rows': len(trained)}
    (Path(eval_dir) / 'result.json').write_text(json.dumps(result, indent=2, default=str))
    (Path(eval_dir) / 'README.md').write_text(summary_markdown(result) + '\n')
    return result


def _fmt(value) -> str:
    return '-' if value is None else f'{value:.4f}'


def _delta(row: dict) -> str:
    return f"{row['delta']:+.4f} [{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}]"


def summary_markdown(result: dict) -> str:
    """One-screen table of an in-loop result (per domain when the environment reports domains)."""
    head = (f"in-loop paired result: {result.get('experiment')} ({result.get('environment')}), base = step {result.get('base_step')} eval, "
            f"trained = step {result.get('trained_step')} eval; selection: {result.get('selection')}")
    if not result.get('paired'):
        return f"{head}\n\nNO PAIRED RESULT: {result.get('reason')}"
    lines = [head, '']
    domains = result.get('domains') or {}
    if domains:
        lines += ['| domain | base | trained | delta [CI] | tasks | trials | dropped | masked base/trained |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
        rates = result.get('masked_rates') or {}
        for name, d in domains.items():
            r = rates.get(name) or {}
            lines.append(f"| {name} | {_fmt(d['baseline'])} | {_fmt(d['candidate'])} | {_delta(d)} | {d['tasks']} | {d['trials']} | {d['dropped']} | "
                         f"{_fmt(r.get('base'))}/{_fmt(r.get('trained'))} |")
        macro = result.get('macro')
        if macro:
            lines.append(f"| **macro** | {_fmt(macro['baseline'])} | {_fmt(macro['candidate'])} | {_delta(macro)} | | | "
                         f"{(result.get('dropped') or {}).get('total', 0)} | |")
    else:
        t = result['test']
        lines += ['| base | trained | delta [CI] |', '| --- | --- | --- |',
                  f"| {_fmt(t['baseline_accuracy'])} | {_fmt(t['candidate_accuracy'])} | {t['accuracy_delta']:+.4f} [{t['ci_lower']:+.4f}, {t['ci_upper']:+.4f}] |"]
    dropped = result.get('dropped') or {}
    if dropped.get('total'):
        lines.append(f"\ndropped {dropped['total']} of {dropped.get('of')} trial ids (masked or missing on either side): {dropped.get('by_domain')}")
    return '\n'.join(lines)


def wandb_scalars(result: dict) -> dict[str, float]:
    """The W&B summary keys: test/<domain>/{base,trained,delta,ci_lower,ci_upper}, test/macro/*, test/dropped (+ the legacy test/* keys)."""
    out: dict[str, float] = {}
    if not result.get('paired'):
        return out
    for name, d in {**(result.get('domains') or {}), **({'macro': result['macro']} if result.get('macro') else {})}.items():
        out.update({f'test/{name}/base': d['baseline'], f'test/{name}/trained': d['candidate'], f'test/{name}/delta': d['delta'],
                    f'test/{name}/ci_lower': d['ci_lower'], f'test/{name}/ci_upper': d['ci_upper']})
    dropped = result.get('dropped') or {}
    if 'total' in dropped:
        out['test/dropped'] = float(dropped['total'])
    for key, value in (result.get('test') or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.setdefault(f'test/{key}', value)
    return out
