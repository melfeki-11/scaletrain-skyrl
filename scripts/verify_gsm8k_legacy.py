"""Independently verify a finished attempt from S3 and render RESULTS.md.

Downloads final_eval/{base,candidate}.jsonl + result.json + training metrics, re-reads the ground truth from the local
test parquet (not from the worker's rows), re-scores every completion with the project's strict scorer, checks paired
coverage, recomputes the paired bootstrap, and writes evidence/verified-result.json and RESULTS.md.

    python scripts/verify_results.py --attempt q35v2-a1 --job-id <id> --wandb-run <entity/project/run_id>
"""
import argparse
import json
from pathlib import Path
import sys

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from q35.answers import score_answer, parse_final_answer  # noqa: E402
from q35.config import load  # noqa: E402
from q35.finalize import flips, failure_mode  # noqa: E402
from q35.stats import paired_bootstrap  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--attempt', required=True); p.add_argument('--job-id', required=True); p.add_argument('--wandb-run', required=True)
    p.add_argument('--examples', type=int, default=5)
    a = p.parse_args()
    cfg = load(); s3c = cfg['s3']
    s3 = boto3.Session(profile_name=s3c['profile']).client('s3', region_name=s3c['region'])
    base_key = f"{s3c['prefix']}/attempts/{a.attempt}/artifacts"
    out = Path('evidence') / a.attempt; out.mkdir(parents=True, exist_ok=True)
    for name, key in {'base.jsonl': 'final_eval/base.jsonl', 'candidate.jsonl': 'final_eval/candidate.jsonl', 'result.json': 'final_eval/result.json',
                      'metrics.jsonl': f'{a.attempt}/metrics.jsonl', 'eval_samples.jsonl': f'{a.attempt}/exports/evaluation_samples.jsonl',
                      'completion.json': 'completion.json'}.items():
        s3.download_file(s3c['bucket'], f'{base_key}/{key}', str(out / name))
    read = lambda n: [json.loads(l) for l in (out / n).read_text().splitlines()]
    import pyarrow.parquet as pq
    truth = {r['extra_info']['sample_id']: r['reward_spec']['ground_truth'] for r in pq.read_table(Path(cfg['run']['data_dir']) / 'test.parquet').to_pylist()}
    base, cand = read('base.jsonl'), read('candidate.jsonl')
    assert [r['sample_id'] for r in base] == [r['sample_id'] for r in cand] == sorted(truth), 'coverage/order mismatch'
    assert len(base) == cfg['evaluation']['test_size'] == len(set(r['sample_id'] for r in base))
    for rows in (base, cand):
        for r in rows:
            assert r['ground_truth'] == truth[r['sample_id']]
            r['correct'] = bool(score_answer(r['completion'], truth[r['sample_id']], r['finish_reason']))
            r['parsed_answer'] = parse_final_answer(r['completion'])
    stats = paired_bootstrap([r['correct'] for r in base], [r['correct'] for r in cand], samples=cfg['evaluation']['bootstrap_samples'], seed=cfg['evaluation']['bootstrap_seed'])
    counts = flips(base, cand)
    worker = json.loads((out / 'result.json').read_text())
    assert worker['flips'] == counts and abs(worker['test']['candidate_accuracy'] - stats['candidate_accuracy']) < 1e-12, 'worker/independent mismatch'
    completion = json.loads((out / 'completion.json').read_text()); assert completion['status'] == 'completed'
    metrics = read('metrics.jsonl'); steps = [m for m in metrics if m.get('timing/step') and m['optimizer_step'] > 1]
    med = lambda k: sorted(m[k] for m in steps if k in m)[len(steps) // 2]
    val = {}
    for r in read('eval_samples.jsonl'): val.setdefault(r['step'], []).append(r['correct'])
    val = {s: (sum(v), len(v)) for s, v in sorted(val.items())}
    trunc = lambda rows: sum(r['finish_reason'] != 'stop' for r in rows) / len(rows)
    nofinal = lambda rows: sum(r['parsed_answer'] is None for r in rows) / len(rows)
    fixed = [(b, c) for b, c in zip(base, cand) if not b['correct'] and c['correct']]
    by_mode = {}
    for b, c in fixed: by_mode.setdefault(failure_mode(b), []).append((b, c))
    chosen = []
    for mode in ('no_final_answer_line', 'truncated', 'wrong_answer'):
        chosen += by_mode.get(mode, [])[:2]
    chosen = (chosen + [x for x in fixed if x not in chosen])[:a.examples]
    verified = {'attempt': a.attempt, 'job_id': a.job_id, 'wandb_run': a.wandb_run, 'test': stats, 'flips': counts,
                'validation_correct_of': {str(k): v for k, v in val.items()}, 'selected_step': worker['selected_step'],
                'optimizer_steps': worker['optimizer_steps'], 'base_truncation_rate': trunc(base), 'trained_truncation_rate': trunc(cand),
                'base_no_final_answer_rate': nofinal(base), 'trained_no_final_answer_rate': nofinal(cand),
                'median_step_seconds': med('timing/step'), 'median_generate_seconds': med('timing/generate'),
                'median_policy_train_seconds': med('timing/policy_train'), 'median_tokens_per_second_per_gpu': med('trainer/tokens_per_second_per_gpu'),
                'final_train_reward': metrics[-1].get('train/reward'), 'elapsed_seconds': worker['elapsed_seconds'], 'phases': worker['phases']}
    Path('evidence/verified-result.json').write_text(json.dumps(verified, indent=2))
    pct = lambda x: f'{100 * x:.2f}%'
    lines = [f'# Results: {a.attempt}', '',
             f'ScaleTrain job `{a.job_id}` (COMPLETED), W&B `{a.wandb_run}`. Qwen3.5-0.8B, SkyRL GRPO, {worker["optimizer_steps"]} updates on one 8×H100 node; '
             f'checkpoint selected on validation only (step {worker["selected_step"]}).', '',
             '## Official GSM8K test (1,319 paired examples, greedy, identical decoding and scorer)', '',
             '| model | correct | accuracy | truncated | no `#### n` line |', '| --- | --- | --- | --- | --- |',
             f'| base | {sum(r["correct"] for r in base)} | {pct(stats["baseline_accuracy"])} | {pct(trunc(base))} | {pct(nofinal(base))} |',
             f'| trained | {sum(r["correct"] for r in cand)} | {pct(stats["candidate_accuracy"])} | {pct(trunc(cand))} | {pct(nofinal(cand))} |', '',
             f'Delta **{100 * stats["accuracy_delta"]:+.2f} points**, paired 95% bootstrap CI [{100 * stats["ci_lower"]:+.2f}, {100 * stats["ci_upper"]:+.2f}]. '
             f'Flips: {counts["fixed"]} fixed (wrong→right), {counts["broken"]} broken (right→wrong), {counts["both_correct"]} both right, {counts["both_wrong"]} both wrong.', '',
             '## Validation (256 train-derived examples, greedy)', '', '| step | correct | accuracy |', '| --- | --- | --- |']
    lines += [f'| {s} | {c}/{n} | {pct(c / n)} |' for s, (c, n) in val.items()]
    lines += ['', '## Training efficiency (measured)', '',
              f'- median step {verified["median_step_seconds"]:.1f}s (generate {verified["median_generate_seconds"]:.1f}s, policy train {verified["median_policy_train_seconds"]:.1f}s); '
              f'{verified["median_tokens_per_second_per_gpu"]:.0f} tokens/s/GPU; 512 rollouts per step.',
              f'- worker wall time {worker["elapsed_seconds"]:.0f}s (bootstrap ~35s, training+validation ~{(worker["phases"]["training_end"] - worker["phases"]["ray_ready"]):.0f}s, '
              f'final test eval of 2×1,319 examples {(worker["phases"]["eval_end"] - worker["phases"]["eval_start"]):.0f}s); scheduler allocation <=~3,010s (ledger, conservative) of the 3,600s budget.',
              '- the checkpoint is chosen among exported steps only (exports at step 20 and at the deadline step); an unexported step with higher validation accuracy is not eligible.', '',
              '## Examples the base model failed and the trained model solved', '']
    for i, (b, c) in enumerate(chosen, 1):
        lines += [f'### {i}. `{b["sample_id"]}` · base failure: {failure_mode(b)} · truth `{b["ground_truth"]}`', '',
                  '**Question**', '', '> ' + b['prompt'][0]['content'].replace('\n', '\n> '), '',
                  f'**Base** (parsed `{b["parsed_answer"]}`)', '', '```', b['completion'].strip()[:1400], '```', '',
                  f'**Trained** (parsed `{c["parsed_answer"]}`)', '', '```', c['completion'].strip()[:1400], '```', '']
    Path('RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: verified[k] for k in ('test', 'flips', 'selected_step')}, indent=1))


if __name__ == '__main__':
    main()
