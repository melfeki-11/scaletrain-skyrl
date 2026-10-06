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
                      'metrics.jsonl': f'{attempt}/metrics.jsonl', 'runtime.json': 'runtime.json'}.items():
        target = dest / name
        if not target.is_file():
            try:
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
                     'reward': ' + '.join(f"{c.get('weight', 1)}*{c['fn']}" for c in cfg['reward']['components']), 'model': cfg['model']['id']})
    report = {'runs': [], 'pairs': []}
    for r in runs:
        entry = {k: r[k] for k in ('attempt', 'experiment', 'sha', 'reward', 'model', 'curves')}
        if r['base'] and r['trained']:
            entry['test'] = paired_bootstrap([x['correct'] for x in r['base']], [x['correct'] for x in r['trained']], samples=samples)
            entry['by_source'] = r['result'].get('by_source')
        entry['optimizer_steps'] = r['result'].get('optimizer_steps')
        entry['elapsed_seconds'] = r['result'].get('elapsed_seconds')
        report['runs'].append(entry)
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a = {x['sample_id']: x['correct'] for x in runs[i]['trained']}
            b = {x['sample_id']: x['correct'] for x in runs[j]['trained']}
            ids = sorted(set(a) & set(b))
            if ids:
                report['pairs'].append({'a': runs[i]['attempt'], 'b': runs[j]['attempt'], 'examples': len(ids),
                                        'b_minus_a': paired_bootstrap([a[k] for k in ids], [b[k] for k in ids], samples=samples)})
    return report


def markdown(report: dict) -> str:
    pct = lambda x: f'{100 * x:.2f}%'  # noqa: E731
    lines = ['| attempt | model | reward | steps | base | trained | delta [95% CI] |', '| --- | --- | --- | --- | --- | --- | --- |']
    for r in report['runs']:
        t = r.get('test')
        lines.append(f"| {r['attempt']} | {r['model']} | {r['reward']} | {r.get('optimizer_steps')} | "
                     + (f"{pct(t['baseline_accuracy'])} | {pct(t['candidate_accuracy'])} | {100 * t['accuracy_delta']:+.2f} "
                        f"[{100 * t['ci_lower']:+.2f}, {100 * t['ci_upper']:+.2f}] |" if t else '- | - | - |'))
    for p in report['pairs']:
        d = p['b_minus_a']
        lines.append(f"\n**{p['b']} vs {p['a']}** (trained vs trained, {p['examples']} paired tasks): {100 * d['accuracy_delta']:+.2f} points "
                     f"[{100 * d['ci_lower']:+.2f}, {100 * d['ci_upper']:+.2f}]")
    return '\n'.join(lines)
