"""mimo results: the in-loop paired result (worker) and the independent verification (`strl verify`).

Worker side: ``paired_result`` = honest_power (strl/monitor/compare.py) per domain and macro over base (step-0 eval) vs trained (last
eval), plus dropped-ID counts per domain, masked rates, repeats and the legacy ``test`` / ``flips`` / ``by_source`` shapes.

Verify side (``verify_attempt``) does NOT call the worker's code for the numbers: ``independent_stats`` is a second implementation
(pure python, stdlib RNG) of the same estimand; point estimates must agree exactly, CIs within Monte-Carlo tolerance. It also checks
coverage against ``final_eval_ids`` (task x rep) and re-grades what persisted artifacts allow offline:
  music    the vendored scorer on the saved ABC (graders.music, imported lazily; skipped when absent / abc2midi missing)
  general  the LLM judge on the persisted rubric request (N records; variance recorded; skipped without a LiteLLM key)
  webdev   not re-gradable offline (the screenshot lives in the sandbox trial dir): reported as skipped
  cyber/code  need the sandbox: reported as skipped
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Callable

from . import data

TOLERANCE = {'point': 1e-6, 'ci_abs': 0.02, 'ci_rel': 0.25, 'regrade': 0.01}


# ------------------------------------------------------------------------------------------------------------ worker side
def masked_rates(base: list[dict], trained: list[dict]) -> dict[str, dict[str, float | None]]:
    """Per domain: share of trials that were masked (infrastructure / judge failure) in the base and the trained eval."""
    out: dict[str, dict[str, float | None]] = {}
    for domain in data.DOMAINS:
        rate = {}
        for label, rows in (('base', base), ('trained', trained)):
            mine = [r for r in rows if r.get('domain') == domain]
            rate[label] = sum(bool(r.get('masked')) for r in mine) / len(mine) if mine else None
        if rate['base'] is not None or rate['trained'] is not None:
            out[domain] = rate
    return out


def pass_flips(base: list[dict], trained: list[dict]) -> dict[str, int]:
    """fixed / broken / both_pass / both_fail over the trial ids usable on both sides (``passed``)."""
    b, c = {r['sample_id']: r for r in base}, {r['sample_id']: r for r in trained}
    out = {'fixed': 0, 'broken': 0, 'both_correct': 0, 'both_wrong': 0}
    for i in sorted(set(b) & set(c)):
        if b[i].get('masked') or c[i].get('masked'):
            continue
        x, y = bool(b[i].get('passed')), bool(c[i].get('passed'))
        out['both_correct' if x and y else 'broken' if x else 'fixed' if y else 'both_wrong'] += 1
    return out


def paired_result(spec: Any, base: list[dict], trained: list[dict], cfg: dict) -> dict:
    """honest_power per domain + macro, with the accounting the result needs to be read honestly (see module docstring)."""
    from ...monitor.compare import honest_power
    e = cfg['evaluation']
    stats = honest_power(base, trained, samples=e['bootstrap_samples'], seed=e['bootstrap_seed'], level=e['ci_level'])
    macro = stats.get('macro')
    if macro is None:
        raise ValueError('no domain has a usable paired trial (everything was masked or missing)')
    stats['test'] = {'examples': sum(d['tasks'] for d in stats['domains'].values()), 'baseline_accuracy': macro['baseline'],
                     'candidate_accuracy': macro['candidate'], 'accuracy_delta': macro['delta'], 'ci_level': e['ci_level'],
                     'ci_lower': macro['ci_lower'], 'ci_upper': macro['ci_upper'], 'quality_passed': macro['quality_passed']}
    stats.update(eval_repeats=data.mimo_config(cfg)['eval_repeats'], masked_rates=masked_rates(base, trained), flips=pass_flips(base, trained),
                 by_source=spec.metric_by_source(base, trained), metric='macro_task_metric', ci_level=e['ci_level'])
    return stats


# ------------------------------------------------------------------------------------------------------------ independent stats
def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default) in pure python."""
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def independent_stats(base: list[dict], trained: list[dict], *, samples: int = 4000, seed: int = 20260606, level: float = 0.95) -> dict:
    """Second implementation of the per-domain / macro paired estimand from raw per-trial records (not the worker's number).

    A trial id is usable iff it is present on both sides, neither is masked, and both have a task_metric. Per task the repeats are
    averaged; the bootstrap resamples tasks within a domain; the macro CI resamples every domain jointly (stdlib ``random``)."""
    b = {r['sample_id']: r for r in base}
    c = {r['sample_id']: r for r in trained}
    per_task: dict[str, dict[str, list[tuple[float, float]]]] = {}
    dropped: dict[str, int] = {}
    for i in sorted(set(b) | set(c)):
        row = b.get(i) or c.get(i)
        domain = str(row.get('domain'))
        x, y = b.get(i), c.get(i)
        ok = x is not None and y is not None and not x.get('masked') and not y.get('masked') and x.get('task_metric') is not None \
            and y.get('task_metric') is not None
        if not ok:
            dropped[domain] = dropped.get(domain, 0) + 1
            continue
        per_task.setdefault(domain, {}).setdefault(str(row.get('task') or i), []).append((float(x['task_metric']), float(y['task_metric'])))
    rng = random.Random(seed)
    alpha = (1.0 - level) / 2.0
    out: dict = {'domains': {}, 'dropped': {'total': sum(dropped.values()), 'by_domain': dropped}}
    diffs: dict[str, list[float]] = {}
    for domain in sorted(per_task):
        tasks = list(per_task[domain].values())
        bm = [sum(p[0] for p in t) / len(t) for t in tasks]
        cm = [sum(p[1] for p in t) / len(t) for t in tasks]
        diffs[domain] = [y - x for x, y in zip(bm, cm)]
        n = len(tasks)
        draws = sorted(sum(diffs[domain][rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
        out['domains'][domain] = {'tasks': n, 'trials': sum(len(t) for t in tasks), 'dropped': dropped.get(domain, 0), 'baseline': sum(bm) / n,
                                  'candidate': sum(cm) / n, 'delta': sum(diffs[domain]) / n, 'ci_lower': _quantile(draws, alpha),
                                  'ci_upper': _quantile(draws, 1.0 - alpha)}
    if diffs:
        macro_draws = []
        for _ in range(samples):
            macro_draws.append(sum(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for d in diffs.values()) / len(diffs))
        macro_draws.sort()
        rows = out['domains'].values()
        out['macro'] = {'domains': sorted(diffs), 'baseline': sum(d['baseline'] for d in rows) / len(diffs), 'candidate': sum(d['candidate'] for d in rows) / len(diffs),
                        'delta': sum(d['delta'] for d in rows) / len(diffs), 'ci_lower': _quantile(macro_draws, alpha), 'ci_upper': _quantile(macro_draws, 1.0 - alpha)}
    return out


def stats_disagreements(worker: dict | None, independent: dict, tol: dict | None = None) -> list[str]:
    """Human-readable differences between the worker's result and the independent recomputation ([] = agree)."""
    tol = {**TOLERANCE, **(tol or {})}
    if not worker:
        return ['worker result has no paired statistics']
    problems: list[str] = []

    def compare(label: str, w: dict | None, i: dict | None, counts: tuple[str, ...] = ()) -> None:
        if w is None or i is None:
            problems.append(f'{label}: present on one side only (worker={w is not None}, independent={i is not None})')
            return
        for key in counts:
            if w.get(key) != i.get(key):
                problems.append(f'{label}.{key}: worker {w.get(key)} vs independent {i.get(key)}')
        for key in ('baseline', 'candidate', 'delta'):
            if abs(float(w[key]) - float(i[key])) > tol['point']:
                problems.append(f'{label}.{key}: worker {w[key]:.6f} vs independent {i[key]:.6f}')
        width = float(w['ci_upper']) - float(w['ci_lower'])
        allowed = max(tol['ci_abs'], tol['ci_rel'] * width)
        for key in ('ci_lower', 'ci_upper'):
            if abs(float(w[key]) - float(i[key])) > allowed:
                problems.append(f'{label}.{key}: worker {w[key]:.4f} vs independent {i[key]:.4f} (tolerance {allowed:.4f})')

    wd, idom = worker.get('domains') or {}, independent['domains']
    for domain in sorted(set(wd) | set(idom)):
        compare(f'domain {domain}', wd.get(domain), idom.get(domain), counts=('tasks', 'trials', 'dropped'))
    compare('macro', worker.get('macro'), independent.get('macro'))
    if (worker.get('dropped') or {}).get('total') != independent['dropped']['total']:
        problems.append(f"dropped total: worker {(worker.get('dropped') or {}).get('total')} vs independent {independent['dropped']['total']}")
    return problems


def coverage(expected: list[str], base: list[dict], trained: list[dict]) -> dict:
    """final_eval_ids (task x rep) versus the persisted rows: missing, unexpected and duplicated ids per side."""
    want = set(expected)
    out: dict[str, Any] = {'expected': len(expected)}
    for label, rows in (('base', base), ('trained', trained)):
        ids = [str(r['sample_id']) for r in rows]
        out[label] = {'rows': len(ids), 'missing': sorted(want - set(ids))[:20], 'n_missing': len(want - set(ids)),
                      'unexpected': sorted(set(ids) - want)[:20], 'n_unexpected': len(set(ids) - want), 'n_duplicated': len(ids) - len(set(ids))}
    out['complete'] = all(out[k]['n_missing'] == 0 and out[k]['n_unexpected'] == 0 and out[k]['n_duplicated'] == 0 for k in ('base', 'trained'))
    return out


# ------------------------------------------------------------------------------------------------------------ offline re-grading
def _evenly(rows: list[dict], n: int) -> list[dict]:
    if len(rows) <= n:
        return rows
    return [rows[int(i * len(rows) / n)] for i in range(n)]


def regrade_music(cfg: dict, rows: list[dict], limit: int = 40, grade: Callable | None = None) -> dict:
    """Re-run the vendored music scorer on the saved ABC (record ``extra.artifact_full`` else ``artifact``). ``grade`` is injectable."""
    if grade is None:
        try:
            from .graders import music   # lazily: Track B's vendored scorer may be absent in this checkout
            grade = music.grade
        except Exception as error:
            return {'skipped': f'graders.music unavailable ({type(error).__name__}: {error})'[:200]}
    mine = [r for r in rows if r.get('domain') == 'music' and not r.get('masked')]
    if not mine:
        return {'skipped': 'no music records'}
    checked, bad, errors, unconstrained = 0, [], 0, 0
    for row in _evenly(sorted(mine, key=lambda r: r['sample_id']), limit):
        answer = ((row.get('extra') or {}).get('artifact_full')) or row.get('artifact')
        if not answer:
            continue
        constraints_file = Path(cfg['run']['data_dir']) / 'tasks' / 'test' / str(row.get('task')) / 'tests' / 'constraints.json'
        constraints = json.loads(constraints_file.read_text()) if constraints_file.is_file() else None
        unconstrained += constraints is None
        try:
            reward = float(grade(answer, constraints=constraints)['reward'])
        except Exception as error:
            if type(error).__name__ == 'Abc2MidiMissing':
                return {'skipped': 'abc2midi is not installed locally (vendored scorer needs it)'}
            errors += 1
            continue
        checked += 1
        if abs(reward - float(row.get('task_metric') or 0.0)) > TOLERANCE['regrade']:
            bad.append({'sample_id': row['sample_id'], 'worker': row.get('task_metric'), 'regraded': reward})
    return {'checked': checked, 'errors': errors, 'disagreements': len(bad), 'examples': bad[:10], 'without_constraints_json': unconstrained}


def regrade_judge(cfg: dict, rows: list[dict], n: int = 8, judge: Any = None) -> dict:
    """Re-score N general (rubric) records twice with the judge from their persisted request; report variance and disagreements.
    Skipped when the key is unavailable (strl.services.litellm resolution as is) or no request was persisted."""
    import asyncio
    mine = [r for r in rows if r.get('domain') == 'general' and not r.get('masked') and ((r.get('extra') or {}).get('judge_request'))]
    if not mine:
        return {'skipped': 'no general record carries a persisted judge_request (needs the eval-phase judge_request persistence)'}
    if judge is None:
        from . import judge as judge_module
        from ...services import litellm
        try:
            litellm.resolve_key()
        except Exception as error:
            return {'skipped': f'no LiteLLM key: {type(error).__name__}'}
        judge = judge_module.Judge(cfg)
    results = []
    for row in _evenly(sorted(mine, key=lambda r: r['sample_id']), n):
        try:
            noise = asyncio.run(judge.rescore_variance(row['extra']['judge_request'], n=2))
        except Exception as error:
            results.append({'sample_id': row['sample_id'], 'error': f'{type(error).__name__}: {error}'[:160]})
            continue
        mean = sum(noise['finals']) / len(noise['finals'])
        results.append({'sample_id': row['sample_id'], 'worker': row.get('task_metric'), 'rescored': mean, 'variance': noise['variance'],
                        'abs_diff': abs(mean - float(row.get('task_metric') or 0.0)), 'cost_usd': noise['cost_usd']})
    ok = [r for r in results if 'error' not in r]
    return {'checked': len(ok), 'errors': len(results) - len(ok), 'mean_abs_diff': sum(r['abs_diff'] for r in ok) / len(ok) if ok else None,
            'max_abs_diff': max((r['abs_diff'] for r in ok), default=None), 'mean_variance': sum(r['variance'] for r in ok) / len(ok) if ok else None,
            'disagreements_gt_0p25': sum(r['abs_diff'] > 0.25 for r in ok), 'cost_usd': sum(r.get('cost_usd', 0.0) for r in ok), 'records': results}


# ------------------------------------------------------------------------------------------------------------ verify
def verify_attempt(cfg: dict, attempt: str, local: Path, spec: Any, *, regrade: bool = True, music: Callable | None = None, judge: Any = None) -> dict:
    """Independent verification of an in-loop mimo attempt from the downloaded final_eval files (see module docstring)."""
    from ...finalize import read_jsonl
    read = lambda name: read_jsonl(local / name) if (local / name).is_file() else []   # noqa: E731
    base, trained = read('base.jsonl'), read('candidate.jsonl')
    result = json.loads((local / 'result.json').read_text()) if (local / 'result.json').is_file() else {}
    out: dict[str, Any] = {'attempt': attempt, 'experiment': cfg['name'], 'experiment_sha': cfg['_meta']['sha'], 'environment': spec.name,
                           'mode': result.get('mode') or 'in_loop', 'selected_step': result.get('selected_step'), 'selection': result.get('selection'),
                           'base_step': result.get('base_step'), 'trained_step': result.get('trained_step'), 'optimizer_steps': result.get('optimizer_steps')}
    reasons: list[str] = []
    if not base or not trained or not result.get('paired'):
        out.update(verified=False, reasons=[f"no paired result to verify: {result.get('reason') or 'final_eval files missing'}"], paired=None)
        (local / 'verified-result.json').write_text(json.dumps(out, indent=2, default=str))
        return out
    ids = spec.final_eval_ids(cfg)
    out['coverage'] = coverage(ids, base, trained)
    if not out['coverage']['complete']:
        reasons.append('coverage: persisted trial ids differ from final_eval_ids (task x rep)')
    e = cfg['evaluation']
    independent = independent_stats(base, trained, level=e['ci_level'])
    differences = stats_disagreements(result, independent)
    out.update(examples=len(ids), independent=independent, worker={k: result.get(k) for k in ('domains', 'macro', 'dropped')}, stat_disagreements=differences, stats_agree=not differences,
               flips=pass_flips(base, trained), masked_rates=masked_rates(base, trained))
    reasons += [f'stats: {d}' for d in differences[:5]]
    if regrade:
        out['regrade'] = {'music': regrade_music(cfg, base + trained, grade=music), 'judge': regrade_judge(cfg, base + trained, judge=judge),
                          'webdev': {'skipped': 'screenshots stay in the sandbox trial dir; not re-gradable offline'},
                          'cyber': {'skipped': 'needs the sandbox'}, 'code': {'skipped': 'needs the sandbox'}}
        if (out['regrade']['music'].get('disagreements') or 0) > 0:
            reasons.append(f"music re-grade: {out['regrade']['music']['disagreements']} record(s) disagree with the vendored scorer")
    out['reasons'] = reasons
    out['verified'] = not reasons
    (local / 'verified-result.json').write_text(json.dumps(out, indent=2, default=str))
    return out
