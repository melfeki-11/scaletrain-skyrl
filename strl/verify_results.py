"""Independent verification of a finished attempt (never trusts the worker's verdicts).

Downloads final_eval/{base,candidate}.jsonl + result.json, checks coverage/order against the LOCAL held-out ids, re-scores every
completion with the environment's own scorer (GSM8K: strict parser vs local ground truth; coding: re-extract + re-execute in the
Modal executor), recomputes the paired bootstrap, and writes evidence/<attempt>/verified-result.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from .finalize import flips
from .monitor.compare import _download, _rows
from .stats import paired_bootstrap
from .tasks import get_environment


def rescore(cfg: dict, env, rows: list[dict]) -> list[bool]:
    if env.name == 'gsm8k':
        import pyarrow.parquet as pq
        truth = {r['extra_info']['sample_id']: r['reward_spec']['ground_truth'] for r in pq.read_table(Path(cfg['run']['data_dir']) / 'test.parquet').to_pylist()}
        return [bool(env.score(r['completion'], truth[r['sample_id']], r['finish_reason'])) for r in rows]
    from .services.modal import local_profile
    from .tasks.coding import data as coding_data, executor
    from .tasks.coding.extract import extract_code
    tasks = {t.sample_id: t for t in coding_data.load_tasks(cfg, 'test')}
    codes = [extract_code(r['completion']) for r in rows]
    idx = [i for i, c in enumerate(codes) if c is not None and rows[i]['finish_reason'] == 'stop']
    with local_profile(((cfg.get('services') or {}).get('modal') or {}).get('environment', 'main')):
        verdicts = dict(zip(idx, executor.score_many([tasks[rows[i]['sample_id']].job(codes[i]) for i in idx])))
    if any(v['failure_class'] == 'infra_error' for v in verdicts.values()):
        raise RuntimeError('executor infra errors during verification; re-run')
    return [bool(verdicts.get(i, {}).get('failure_class') == 'ok') for i in range(len(rows))]


def verify_in_loop(cfg: dict, attempt: str, local: Path, env) -> dict:
    """Generic in-loop result (final_eval: false, single metric): recompute the paired statistics from the persisted base/trained rows
    and compare them with the worker's result.json."""
    base, cand = _rows(local / 'base.jsonl'), _rows(local / 'candidate.jsonl')
    result = json.loads((local / 'result.json').read_text()) if (local / 'result.json').is_file() else {}
    out = {'attempt': attempt, 'experiment': cfg['name'], 'experiment_sha': cfg['_meta']['sha'], 'environment': env.name, 'mode': 'in_loop',
           'selected_step': result.get('selected_step'), 'selection': result.get('selection')}
    if not base or not cand or not result.get('paired'):
        out.update(verified=False, reasons=[f"no paired result to verify: {result.get('reason') or 'final_eval files missing'}"])
    else:
        stats = env.paired_result(base, cand, cfg)
        worker = result.get('test') or {}
        # a statistic the worker omitted is a disagreement (abs(nan - x) > eps is False, which silently "agreed" before)
        diff = {k: worker.get(k) for k in ('baseline_accuracy', 'candidate_accuracy', 'accuracy_delta')
                if worker.get(k) is None or not abs(float(worker[k]) - stats['test'][k]) <= 1e-9}
        out.update(test=stats['test'], worker_test=result.get('test'), flips=stats.get('flips'), disagreements=diff, verified=not diff,
                   reasons=[f'worker {k} differs from the recomputation' for k in diff])
    (local / 'verified-result.json').write_text(json.dumps(out, indent=2, default=str))
    return out


def verify(cfg: dict, attempt: str) -> dict:
    env = get_environment(cfg['environment']['name'])
    local = _download(cfg, attempt, Path('evidence') / attempt)
    if not cfg['evaluation'].get('final_eval', True):   # in-loop paired result (mimo): independent recomputation + offline re-grading
        if env.name == 'mimo':
            from .tasks.mimo.results import verify_attempt
            return verify_attempt(cfg, attempt, local, env)
        return verify_in_loop(cfg, attempt, local, env)
    base, cand = _rows(local / 'base.jsonl'), _rows(local / 'candidate.jsonl')
    result = json.loads((local / 'result.json').read_text())
    ids = env.final_eval_ids(cfg)
    if [r['sample_id'] for r in base] != ids or [r['sample_id'] for r in cand] != ids:
        raise AssertionError('final evaluation coverage/order differs from the local held-out ids')
    out = {'attempt': attempt, 'experiment': cfg['name'], 'experiment_sha': cfg['_meta']['sha'], 'environment': env.name, 'examples': len(ids)}
    disagreements = {}
    for label, rows in (('base', base), ('trained', cand)):
        independent = rescore(cfg, env, rows)
        disagreements[label] = [r['sample_id'] for r, ok in zip(rows, independent) if bool(r['correct']) != ok]
        for r, ok in zip(rows, independent):
            r['correct'] = ok
    out['worker_vs_independent_disagreements'] = {k: len(v) for k, v in disagreements.items()}
    out['disagreement_ids'] = {k: v[:20] for k, v in disagreements.items()}
    stats = paired_bootstrap([r['correct'] for r in base], [r['correct'] for r in cand], samples=cfg['evaluation']['bootstrap_samples'],
                             seed=cfg['evaluation']['bootstrap_seed'], level=cfg['evaluation']['ci_level'])
    out.update(test=stats, flips=flips(base, cand), worker_test=result.get('test'), optimizer_steps=result.get('optimizer_steps'),
               selected_step=result.get('selected_step'), by_source=env.metric_by_source(base, cand) if hasattr(env, 'metric_by_source') else None,
               phases=result.get('phases'), elapsed_seconds=result.get('elapsed_seconds'))
    total = sum(len(v) for v in disagreements.values())
    out['verified'] = total <= max(1, int(0.005 * 2 * len(ids)))   # executor timeouts can flip a handful of borderline programs
    (local / 'verified-result.json').write_text(json.dumps(out, indent=2, default=str))
    return out
