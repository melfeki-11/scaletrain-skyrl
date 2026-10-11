"""Compare finished attempts on the SAME held-out task ids (paired), from S3 (the source of truth, not W&B).

    python -m strl compare experiments/a.yaml:attempt-a experiments/b.yaml:attempt-b [--wandb]

For each attempt: base vs trained paired bootstrap (task metric), per-source breakdown, flips, steps, node-hours. For every pair
of attempts: trained(A) vs trained(B) paired bootstrap on the intersection of ids, so a reward-design or model change is
judged by the yardstick, never by its own training reward. Also summarises training curves from metrics.jsonl
(learning-signal fraction, reward components, critic failure modes).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..experiment import load
from ..stats import paired_bootstrap


def _download(cfg: dict, attempt: str, dest: Path) -> Path:
    import boto3
    st = cfg['storage']
    s3 = boto3.Session(profile_name=st['profile']).client('s3', region_name=st['region'])
    base = f"{st['prefix']}/attempts/{attempt}/artifacts"
    dest.mkdir(parents=True, exist_ok=True)
    for name, key in {'base.jsonl': 'final_eval/base.jsonl', 'candidate.jsonl': 'final_eval/candidate.jsonl', 'result.json': 'final_eval/result.json',
                      'metrics.jsonl': f'{attempt}/metrics.jsonl', 'runtime.json': 'runtime.json', 'network.json': 'network.json',
                      'completion.json': 'completion.json', 'evaluation_timing.jsonl': f'{attempt}/evaluation_timing.jsonl'}.items():
        target = dest / name
        try:   # always refresh: a file fetched while the run was live (metrics, completion) would otherwise stay stale forever
            s3.download_file(st['bucket'], f'{base}/{key}', str(target))
        except Exception:
            pass
    return dest


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def summarize_curves(metrics: list[dict]) -> dict:
    train = [m for m in metrics if 'policy/policy_loss' in m]
    def series(key):
        return [m[key] for m in metrics if isinstance(m.get(key), (int, float))]
    out = {'optimizer_steps': len(train)}
    for key in ('train/learning_signal_fraction', 'reward/avg_raw_reward', 'eval/task_metric', 'train/pass_at_1', 'timing/step'):
        values = series(key)
        if values:
            out[key] = {'first': values[0], 'last': values[-1], 'max': max(values)}
    comps = sorted({k for m in metrics for k in m if k.startswith('environment/reward_components/')})
    out['reward_components'] = {k.split('/')[-1]: {'first': series(k)[0], 'last': series(k)[-1]} for k in comps if series(k)}
    critic = {k.split('/')[-1].removesuffix('_frac'): v for k, v in (metrics[-1] if metrics else {}).items() if k.startswith('critic/') and k.endswith('_frac')}
    out['critic_modes'] = dict(sorted(critic.items(), key=lambda kv: -kv[1])[:5])
    return out


def compare(specs: list[str], workdir: str | Path = 'evidence/compare', samples: int = 10000) -> dict:
    runs = []
    for spec in specs:
        path, _, attempt = spec.rpartition(':')
        cfg = load(path)
        local = _download(cfg, attempt, Path(workdir) / attempt)
        base, cand = _rows(local / 'base.jsonl'), _rows(local / 'candidate.jsonl')
        result = json.loads((local / 'result.json').read_text()) if (local / 'result.json').is_file() else {}
        runs.append({'attempt': attempt, 'experiment': cfg['name'], 'sha': cfg['_meta']['sha'], 'base': base, 'trained': cand, 'result': result,
                     'curves': summarize_curves(_rows(local / 'metrics.jsonl')),
                     'reward': ' + '.join(f"{c.get('weight', 1)}*{c['fn']}" for c in cfg['reward']['components']), 'model': cfg['model']['id'],
                     'domainful': cfg['environment']['name'] == 'mimo'})
    report = {'runs': [], 'pairs': []}
    for r in runs:
        entry = {k: r[k] for k in ('attempt', 'experiment', 'sha', 'reward', 'model', 'curves')}
        if r['base'] and r['trained'] and r['domainful']:   # per-domain environments: honest_power (drops masked/missing ids, resamples tasks)
            stats = honest_power(r['base'], r['trained'], samples=samples)
            if stats.get('macro'):
                m = stats['macro']
                entry['test'] = {'baseline_accuracy': m['baseline'], 'candidate_accuracy': m['candidate'], 'accuracy_delta': m['delta'],
                                 'ci_lower': m['ci_lower'], 'ci_upper': m['ci_upper']}
            entry.update(domains=stats['domains'], macro=stats.get('macro'), dropped=stats['dropped'], by_source=r['result'].get('by_source'),
                         selected_step=r['result'].get('selected_step'), selection=r['result'].get('selection'))
        elif r['base'] and r['trained']:
            entry['test'] = paired_bootstrap([x['correct'] for x in r['base']], [x['correct'] for x in r['trained']], samples=samples)
            entry['by_source'] = r['result'].get('by_source')
        entry['optimizer_steps'] = r['result'].get('optimizer_steps')
        entry['elapsed_seconds'] = r['result'].get('elapsed_seconds')
        report['runs'].append(entry)
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            unfair = fairness(runs[i], runs[j])
            if runs[i]['domainful'] and runs[j]['domainful']:   # trained vs trained per domain + macro on the shared trial ids
                stats = honest_power(runs[i]['trained'], runs[j]['trained'], samples=samples)
                if stats.get('domains'):
                    report['pairs'].append({'a': runs[i]['attempt'], 'b': runs[j]['attempt'], 'examples': sum(d['trials'] for d in stats['domains'].values()),
                                            'domains': stats['domains'], 'macro': stats.get('macro'), 'dropped': stats['dropped'], 'unfair': unfair})
                continue
            a = {x['sample_id']: x['correct'] for x in runs[i]['trained']}
            b = {x['sample_id']: x['correct'] for x in runs[j]['trained']}
            ids = sorted(set(a) & set(b))
            if ids:
                report['pairs'].append({'a': runs[i]['attempt'], 'b': runs[j]['attempt'], 'examples': len(ids), 'unfair': unfair,
                                        'b_minus_a': paired_bootstrap([a[k] for k in ids], [b[k] for k in ids], samples=samples)})
    return report


def fairness(a: dict, b: dict) -> list[str]:
    """Why an arm-vs-arm delta may not isolate the studied axis: unequal optimizer steps (the predictive deadline or a slower shape stopped
    one arm early), a different placement (a --race-shape / fallback winner), or different base scores on the same ids (different data,
    prompt or model)."""
    out = []
    ra, rb = a['result'], b['result']
    if ra.get('optimizer_steps') != rb.get('optimizer_steps'):
        out.append(f"optimizer steps differ ({ra.get('optimizer_steps')} vs {rb.get('optimizer_steps')})")
    if (ra.get('nodes'), ra.get('gpus')) != (rb.get('nodes'), rb.get('gpus')):
        out.append(f"placement differs ({ra.get('nodes')}x{ra.get('gpus')} vs {rb.get('nodes')}x{rb.get('gpus')} GPUs)")
    base_a = {x['sample_id']: x['correct'] for x in a['base'] if 'sample_id' in x}
    base_b = {x['sample_id']: x['correct'] for x in b['base'] if 'sample_id' in x}
    shared = set(base_a) & set(base_b)
    if shared and any(base_a[k] != base_b[k] for k in shared):
        out.append('base evaluations disagree on shared ids (different data, prompt or model)')
    return out


def markdown(report: dict) -> str:
    pct = lambda x: f'{100 * x:.2f}%'  # noqa: E731
    lines = ['| attempt | model | reward | steps | base | trained | delta [95% CI] |', '| --- | --- | --- | --- | --- | --- | --- |']
    for r in report['runs']:
        t = r.get('test')
        lines.append(f"| {r['attempt']} | {r['model']} | {r['reward']} | {r.get('optimizer_steps')} | "
                     + (f"{pct(t['baseline_accuracy'])} | {pct(t['candidate_accuracy'])} | {100 * t['accuracy_delta']:+.2f} "
                        f"[{100 * t['ci_lower']:+.2f}, {100 * t['ci_upper']:+.2f}] |" if t else '- | - | - |'))
    curves = [r for r in report['runs'] if r.get('curves')]
    if curves:   # training dynamics: did the reward move the yardstick, or only itself?
        lines += ['', '| attempt | learning signal first -> last | train reward last (max) | val task metric last (max) | '
                  'reward components last | step s (last) | top critic modes |', '| --- | --- | --- | --- | --- | --- | --- |']
        for r in curves:
            c = r['curves']
            get = lambda k, f='last': (c.get(k) or {}).get(f)  # noqa: E731
            num = lambda x: '-' if x is None else f'{x:.3f}'  # noqa: E731
            comps = ', '.join(f"{k} {v.get('last', 0):.2f}" for k, v in (c.get('reward_components') or {}).items())
            modes = ', '.join(f'{k} {v:.0%}' for k, v in sorted((c.get('critic_modes') or {}).items(), key=lambda kv: -kv[1])[:3])
            lines.append(f"| {r['attempt']} | {num(get('train/learning_signal_fraction', 'first'))} -> {num(get('train/learning_signal_fraction'))} | "
                         f"{num(get('reward/avg_raw_reward'))} ({num(get('reward/avg_raw_reward', 'max'))}) | "
                         f"{num(get('eval/task_metric'))} ({num(get('eval/task_metric', 'max'))}) | {comps or '-'} | "
                         f"{num(get('timing/step'))} | {modes or '-'} |")
    lines += domain_tables(report)
    for p in report['pairs']:
        if 'b_minus_a' not in p:
            continue
        d = p['b_minus_a']
        lines.append(f"\n**{p['b']} vs {p['a']}** (trained vs trained, {p['examples']} paired tasks): {100 * d['accuracy_delta']:+.2f} points "
                     f"[{100 * d['ci_lower']:+.2f}, {100 * d['ci_upper']:+.2f}]")
        if p.get('unfair'):
            lines.append(f"! NOT A CLEAN A/B: {'; '.join(p['unfair'])}")
    return '\n'.join(lines)


def _delta(d: dict) -> str:
    return f"{d['delta']:+.4f} [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]"


def domain_tables(report: dict) -> list[str]:
    """mimo-style runs: per-domain base / trained / delta [CI] / n tasks / dropped (+ macro) for every run, then run-vs-run trained-vs-trained
    paired deltas per domain and macro. Empty for single-metric environments (their tables are unchanged)."""
    lines: list[str] = []
    for r in report['runs']:
        if not r.get('domains'):
            continue
        selection = f" (trained = step {r['selected_step']}: {r['selection']})" if r.get('selected_step') is not None else ''
        lines += ['', f"**{r['attempt']}** per domain{selection}", '', '| domain | base | trained | delta [95% CI] | n tasks | dropped |', '| --- | --- | --- | --- | --- | --- |']
        for name, d in r['domains'].items():
            lines.append(f"| {name} | {d['baseline']:.4f} | {d['candidate']:.4f} | {_delta(d)} | {d['tasks']} | {d['dropped']} |")
        m = r.get('macro')
        if m:
            lines.append(f"| **macro** | {m['baseline']:.4f} | {m['candidate']:.4f} | {_delta(m)} | {sum(d['tasks'] for d in r['domains'].values())} | "
                         f"{(r.get('dropped') or {}).get('total', 0)} |")
    for p in report['pairs']:
        if 'domains' not in p:
            continue
        lines += ['', f"**{p['b']} vs {p['a']}** (trained vs trained, paired per task; delta = {p['b']} - {p['a']})", '',
                  '| domain | A | B | B - A [95% CI] | n tasks | dropped |', '| --- | --- | --- | --- | --- | --- |']
        for name, d in p['domains'].items():
            lines.append(f"| {name} | {d['baseline']:.4f} | {d['candidate']:.4f} | {_delta(d)} | {d['tasks']} | {d['dropped']} |")
        m = p.get('macro')
        if m:
            lines.append(f"| **macro** | {m['baseline']:.4f} | {m['candidate']:.4f} | {_delta(m)} | | {p['dropped']['total']} |")
        if p.get('unfair'):
            lines.append(f"! NOT A CLEAN A/B: {'; '.join(p['unfair'])}")
    return lines


def honest_power(base: list[dict], candidate: list[dict], *, samples: int = 10000, seed: int = 29, level: float = 0.95,
                 domain_key: str = 'domain', value_key: str = 'task_metric') -> dict:
    """Per-domain paired bootstrap + macro average over domains, with honest accounting of what was dropped.

    Rows are eval records ({sample_id, task, domain, task_metric, masked, ...}; ``sample_id`` = '<task>#r<rep>' identifies a trial).
    A trial id is DROPPED when it is masked/invalid (``masked`` or a missing metric) for either model or absent from one of them;
    drops are reported per domain, never silently turned into zeros. Pairs are aggregated per task (repeats of one task are
    correlated) and the bootstrap resamples TASKS, within each domain; the macro CI resamples every domain jointly."""
    import numpy as np
    b = {r['sample_id']: r for r in base}
    c = {r['sample_id']: r for r in candidate}
    ids = sorted(set(b) | set(c))
    usable = lambda r: r is not None and not r.get('masked') and r.get(value_key) is not None  # noqa: E731
    dropped = {i: ('missing' if i not in b or i not in c else 'masked_or_invalid') for i in ids if not (usable(b.get(i)) and usable(c.get(i)))}
    per_task: dict[str, dict[str, list[tuple[float, float]]]] = {}
    drops_by_domain: dict[str, int] = {}
    for i in ids:
        row = b.get(i) or c.get(i)
        domain = str(row.get(domain_key))
        if i in dropped:
            drops_by_domain[domain] = drops_by_domain.get(domain, 0) + 1
            continue
        per_task.setdefault(domain, {}).setdefault(str(row.get('task') or i), []).append((float(b[i][value_key]), float(c[i][value_key])))
    rng = np.random.default_rng(seed)
    alpha = (1.0 - level) / 2.0
    arrays: dict[str, tuple] = {}
    out: dict = {'domains': {}}
    for domain in sorted(per_task):
        tasks = list(per_task[domain].values())
        base_means = np.array([np.mean([p[0] for p in t]) for t in tasks])
        cand_means = np.array([np.mean([p[1] for p in t]) for t in tasks])
        arrays[domain] = (base_means, cand_means)
        delta = cand_means - base_means
        means = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(axis=1)
        lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
        out['domains'][domain] = {'tasks': len(tasks), 'trials': sum(len(t) for t in tasks), 'dropped': drops_by_domain.get(domain, 0),
                                  'baseline': float(base_means.mean()), 'candidate': float(cand_means.mean()), 'delta': float(delta.mean()),
                                  'ci_lower': float(lower), 'ci_upper': float(upper), 'quality_passed': bool(lower > 0.0)}
    if arrays:
        draws = np.zeros(samples)
        for base_means, cand_means in arrays.values():
            delta = cand_means - base_means
            draws += delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(axis=1)
        draws /= len(arrays)
        lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
        out['macro'] = {'domains': sorted(arrays), 'baseline': float(np.mean([out['domains'][d]['baseline'] for d in arrays])),
                        'candidate': float(np.mean([out['domains'][d]['candidate'] for d in arrays])),
                        'delta': float(np.mean([out['domains'][d]['delta'] for d in arrays])),
                        'ci_lower': float(lower), 'ci_upper': float(upper), 'ci_level': level, 'quality_passed': bool(lower > 0.0)}
    out['dropped'] = {'total': len(dropped), 'by_domain': drops_by_domain, 'ids': sorted(dropped)[:50], 'of': len(ids)}
    return out
