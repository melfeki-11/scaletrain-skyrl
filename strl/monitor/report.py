"""`strl report <exp>:<attempt> [...]`: one markdown report per W&B group, from S3 (the source of truth), also pushed to W&B.

    python -m strl report experiments/mimo-a.yaml:mimo-a1 [experiments/mimo-b.yaml:mimo-b1] [--no-wandb]

evidence/results/<group>-report.md holds: the headline table (macro base / trained / delta [CI]) and per-domain tables (compare.py),
run-vs-run trained-vs-trained deltas, learning curves per domain (training pass rate / reward by step), the learning signal, masked
fractions, step-time phases, token share per domain, infrastructure (network.json bandwidth/transport, peak GPU memory), judge cost and
the critic's failure-mode shares. Training curves come from the complete scalar mirror ``<attempt>/metrics.jsonl`` that
AxisTracking writes next to every W&B log call (so no W&B round trip is needed to read them back).

The same content is logged to W&B as a run-level summary table plus a markdown artifact on a small ``report`` run in the same group;
a W&B Report is also built when ``wandb_workspaces`` happens to be importable. Every W&B step is best-effort: failures never break the CLI.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..experiment import load
from . import compare as cmp

PHASES = ('timing/step', 'timing/generate', 'timing/fwd_logprobs_values_reward', 'timing/policy_train', 'timing/train_critic_and_policy',
          'timing/sync_weights', 'timing/save_checkpoints', 'timing/save_hf_model', 'timing/eval')
CURVE_POINTS = 10


def by_step(rows: list[dict]) -> dict[int, dict]:
    """Scalar mirror rows -> {optimizer_step: merged scalars} (a step is logged in several calls: train, eval, tables)."""
    out: dict[int, dict] = {}
    for row in rows:
        step = row.get('optimizer_step')
        if isinstance(step, (int, float)):
            out.setdefault(int(step), {}).update(row)
    return dict(sorted(out.items()))


def domains_in(steps: dict[int, dict]) -> list[str]:
    seen = {m.group(1) for row in steps.values() for key in row if (m := re.match(r'^generate/([a-z]+)/pass_rate$', key))}
    order = ('music', 'cyber', 'code', 'general', 'webdev')
    return [d for d in order if d in seen] + sorted(seen - set(order))


def pick(steps: list[int], n: int = CURVE_POINTS) -> list[int]:
    if len(steps) <= n:
        return steps
    return sorted({steps[round(i * (len(steps) - 1) / (n - 1))] for i in range(n)})


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def num(value, spec: str = '{:.3f}') -> str:
    return '-' if value is None else spec.format(value)


def series(steps: dict[int, dict], key: str) -> list[tuple[int, float]]:
    return [(s, float(r[key])) for s, r in steps.items() if isinstance(r.get(key), (int, float))]


def curve_table(steps: dict[int, dict], domains: list[str], suffix: str, title: str) -> list[str]:
    """Rows = a handful of steps, columns = domains: the training value of ``generate/<domain>/<suffix>``."""
    shown = pick([s for s in steps if any(f'generate/{d}/{suffix}' in steps[s] for d in domains)])
    if not shown:
        return []
    lines = ['', f'{title}', '', '| step | ' + ' | '.join(domains) + ' |', '| --- | ' + ' | '.join('---' for _ in domains) + ' |']
    for s in shown:
        lines.append(f'| {s} | ' + ' | '.join(num(steps[s].get(f'generate/{d}/{suffix}'), '{:.2f}') for d in domains) + ' |')
    return lines


def training_sections(steps: dict[int, dict], network: dict | None) -> list[str]:
    """Everything the report says about HOW the run trained (the headline says what it achieved)."""
    if not steps:
        return ['', '_no training metrics were found (metrics.jsonl missing)_']
    domains = domains_in(steps)
    lines: list[str] = []
    if domains:
        lines += curve_table(steps, domains, 'pass_rate', 'Training pass rate by step (generate/<domain>/pass_rate)')
        lines += curve_table(steps, domains, 'reward_mean', 'Training reward by step (generate/<domain>/reward_mean)')
        shown = pick([s for s in steps if any(f'eval/{d}/task_metric' in steps[s] for d in domains)])
        if shown:
            lines += ['', 'Held-out task metric at every eval (eval/<domain>/task_metric)', '', '| step | ' + ' | '.join([*domains, 'macro']) + ' |',
                      '| --- | ' + ' | '.join('---' for _ in range(len(domains) + 1)) + ' |']
            lines += [f'| {s} | ' + ' | '.join(num(steps[s].get(f'eval/{d}/task_metric'), '{:.3f}') for d in domains) + f" | {num(steps[s].get('eval/macro_task_metric'))} |"
                      for s in shown]
    signal = [v for _, v in series(steps, 'train/learning_signal_fraction')]
    lines += ['', '| learning signal (groups with reward variance) | first | mean | last |', '| --- | --- | --- | --- |',
              f'| train/learning_signal_fraction | {num(signal[0] if signal else None)} | {num(mean(signal))} | {num(signal[-1] if signal else None)} |']
    if domains:
        lines += ['', '| domain | masked groups mean | masked groups max | tokens share | judge cost (USD) |', '| --- | --- | --- | --- | --- |']
        tokens = {d: sum(r.get(f'generate/{d}/tokens', 0.0) * r.get(f'generate/{d}/num_trials', 0.0) for r in steps.values()) for d in domains}
        total = sum(tokens.values()) or 1.0
        for d in domains:
            masked = [v for _, v in series(steps, f'environment/{d}/masked_group_fraction')]
            cost = sum(v for _, v in series(steps, f'generate/{d}/judge_cost_usd'))
            lines.append(f'| {d} | {num(mean(masked))} | {num(max(masked) if masked else None)} | {tokens[d] / total:.1%} | {cost:.3f} |')
    phases = [(k, [v for _, v in series(steps, k)]) for k in PHASES]
    phases = [(k, v) for k, v in phases if v]
    if phases:
        lines += ['', '| step-time phase | mean s | max s | steps |', '| --- | --- | --- | --- |']
        lines += [f'| {k.removeprefix("timing/")} | {mean(v):.1f} | {max(v):.1f} | {len(v)} |' for k, v in phases]
    infra = []
    if network:
        infra.append(f"transport={network.get('transport')} efa={network.get('efa_available')} nodes={network.get('nodes')} "
                     f"bandwidth_gbps={json.dumps(network.get('bandwidth_gbps'))}")
    memory = [v for k in ('gpu/mem_peak_gb', 'gpu/mem_used_gb_max') for _, v in series(steps, k)]
    if memory:
        infra.append(f'peak GPU memory (head node) {max(memory):.1f} GB')
    last = steps[max(steps)]
    cost = last.get('critic/cost_usd')
    if cost is not None:
        infra.append(f'critic cost {cost:.3f} USD')
    lines += ['', 'Infrastructure: ' + ('; '.join(infra) if infra else 'no network.json / memory metric recorded')]
    shares = {k.removeprefix('critic/').removesuffix('_frac'): v for k, v in last.items() if k.startswith('critic/') and k.endswith('_frac')}
    if shares:
        lines.append('Critic failure-mode shares (last analysed step): ' + ', '.join(f'{k} {v:.0%}' for k, v in sorted(shares.items(), key=lambda kv: -kv[1])[:8]))
    return lines


def headline_rows(report: dict) -> list[list]:
    """[attempt, domain, base, trained, delta, ci_lower, ci_upper, tasks, dropped] rows for the W&B summary table."""
    rows = []
    for r in report['runs']:
        for name, d in {**(r.get('domains') or {}), **({'macro': r['macro']} if r.get('macro') else {})}.items():
            rows.append([r['attempt'], name, d['baseline'], d['candidate'], d['delta'], d['ci_lower'], d['ci_upper'], d.get('tasks'),
                         d.get('dropped', (r.get('dropped') or {}).get('total'))])
        if not r.get('domains') and r.get('test'):
            t = r['test']
            rows.append([r['attempt'], 'task_metric', t['baseline_accuracy'], t['candidate_accuracy'], t['accuracy_delta'], t['ci_lower'], t['ci_upper'], t.get('examples'), None])
    return rows


def render(group: str, report: dict, details: dict[str, dict]) -> str:
    lines = [f'# {group}: results report', '',
             'Evidence: base = the step-0 in-loop evaluation, trained = the last in-loop evaluation (same engines, agent and template); '
             'held-out trials only; masked / missing trial ids are dropped on both sides and counted, never scored as zero.', '',
             cmp.markdown(report)]
    for attempt, info in details.items():
        lines += ['', f'## {attempt}: training', *training_sections(by_step(info['metrics']), info.get('network'))]
    return '\n'.join(lines) + '\n'


def log_to_wandb(cfg: dict, group: str, markdown: str, path: Path, rows: list[list], attempts: list[str]) -> dict:
    """A small ``report`` run in the group: summary table, markdown artifact (+ a W&B Report when wandb_workspaces is importable)."""
    receipt: dict = {'wandb': 'skipped'}
    try:
        import wandb
        from ..services.wandb import local_key
        local_key()
        t = cfg['tracking']
        os.environ['WANDB_BASE_URL'] = t['base_url']
        name = re.sub(r'[^A-Za-z0-9_-]', '-', f'{group}-report')[:64]
        run = wandb.init(project=t['project'], entity=t['entity'], group=group, job_type='report', name=name, id=name, resume='allow',
                         config={'attempts': attempts}, reinit=True)
        try:
            columns = ['attempt', 'domain', 'base', 'trained', 'delta', 'ci_lower', 'ci_upper', 'tasks', 'dropped']
            run.log({'report/headline': wandb.Table(columns=columns, data=rows)})
            run.summary['report/markdown'] = markdown[:60000]
            artifact = wandb.Artifact(name, type='report')
            artifact.add_file(str(path))
            run.log_artifact(artifact)
            receipt = {'wandb': 'logged', 'run': run.url}
        finally:
            run.finish()
        try:   # optional: only when the package happens to be installed (never installed by strl)
            import wandb_workspaces.reports.v2 as wr
            report = wr.Report(project=t['project'], entity=t['entity'], title=f'{group}: results report', description=f'attempts: {", ".join(attempts)}',
                               blocks=[wr.MarkdownBlock(text=markdown)])
            report.save()
            receipt['wandb_report'] = getattr(report, 'url', 'saved')
        except ImportError:
            receipt['wandb_report'] = 'wandb_workspaces not installed'
        except Exception as error:
            receipt['wandb_report'] = f'skipped: {type(error).__name__}: {error}'[:200]
    except Exception as error:
        receipt = {'wandb': f'skipped: {type(error).__name__}: {error}'[:300]}
    return receipt


def write_reports(specs: list[str], out_dir: str | Path = 'evidence/results', samples: int = 10000, wandb: bool = True,
                  workdir: str | Path = 'evidence/compare') -> list[dict]:
    """One report per tracking group among ``specs`` (experiment.yaml:attempt). Returns [{group, path, wandb}]."""
    groups: dict[str, list[str]] = {}
    configs: dict[str, dict] = {}
    for spec in specs:
        path, _, attempt = spec.rpartition(':')
        cfg = load(path)
        groups.setdefault(cfg['tracking']['group'], []).append(spec)
        configs.setdefault(cfg['tracking']['group'], cfg)
    out = []
    for group, members in groups.items():
        report = cmp.compare(members, workdir=workdir, samples=samples)
        details = {}
        for spec in members:
            attempt = spec.rpartition(':')[2]
            local = Path(workdir) / attempt
            network = json.loads((local / 'network.json').read_text()) if (local / 'network.json').is_file() else None
            details[attempt] = {'metrics': cmp._rows(local / 'metrics.jsonl'), 'network': network}
        markdown = render(group, report, details)
        target = Path(out_dir) / f'{group}-report.md'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown)
        entry = {'group': group, 'path': str(target), 'attempts': [m.rpartition(':')[2] for m in members]}
        entry['wandb'] = log_to_wandb(configs[group], group, markdown, target, headline_rows(report), entry['attempts']) if wandb else {'wandb': 'disabled'}
        out.append(entry)
    return out
