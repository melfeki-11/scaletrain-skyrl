"""Live monitor for a running attempt: ScaleTrain state + S3 progress/heartbeat + W&B metrics, one status line per poll.

    python -m strl monitor experiments/<name>.yaml --attempt <attempt> [--job-id <id>] [--interval 60]

Flags (also sent as W&B alerts by the worker itself): STALL (no optimizer step for max(10 min, 3 x median step)), NAN, ENV_FAILURES
(masked groups / infra errors above 5%), BUDGET (node-hours above the attempt allowance), DONE / FAILED (completion.json).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEYS = ['optimizer_step', 'reward/avg_raw_reward', 'eval/task_metric', 'train/learning_signal_fraction', 'train/pass_at_1',
        'environment/infra_error', 'environment/masked_group_fraction', 'timing/step', 'timing/generate', 'policy/grad_norm',
        'policy/policy_loss', 'budget/consumed_node_hours', 'critic/cost_usd', 'generate/avg_num_tokens', 'train/truncation_rate']


def s3_json(session, uri: str):
    bucket, _, key = uri[5:].partition('/')
    try:
        return json.loads(session.client('s3').get_object(Bucket=bucket, Key=key)['Body'].read())
    except Exception:
        return None


def job_state(job_id: str | None) -> str | None:
    if not job_id:
        return None
    out = subprocess.run([sys.executable, str(ROOT / 'tools/scaletrain.py'), 'get', job_id], capture_output=True, text=True, timeout=120)
    try:
        return json.loads(out.stdout)['status']
    except Exception:
        return 'unknown'


def wandb_rows(cfg: dict, run_id: str, last: int = 400) -> list[dict]:
    try:
        import wandb
        from ..services.wandb import local_key
        local_key()
        os.environ['WANDB_BASE_URL'] = cfg['tracking']['base_url']
        api = wandb.Api(overrides={'base_url': cfg['tracking']['base_url']}, timeout=30)
        run = api.run(f"{cfg['tracking']['entity']}/{cfg['tracking']['project']}/{run_id}")
        return [dict(r) for r in run.scan_history(keys=None, page_size=1000)][-last:]
    except Exception:
        return []


def flags(rows: list[dict], progress: dict | None, heartbeat: dict | None, node_hour_allowance: float) -> list[str]:
    out = []
    steps = [r for r in rows if r.get('timing/step')]
    median = sorted(r['timing/step'] for r in steps)[len(steps) // 2] if steps else None
    if progress and progress.get('updated_at'):
        age = time.time() - progress['updated_at']
        if age > max(600.0, 3 * (median or 200.0)):
            out.append(f'STALL(no step for {age:.0f}s)')
    for r in rows[-20:]:
        if any(isinstance(v, float) and not math.isfinite(v) for v in r.values()):
            out.append('NAN')
            break
    latest = rows[-1] if rows else {}
    for key in ('environment/masked_group_fraction', 'environment/infra_error'):
        if (latest.get(key) or 0) > 0.05:
            out.append(f'ENV_FAILURES({key}={latest[key]:.2f})')
    if progress and (progress.get('budget_consumed_node_hours') or 0) > node_hour_allowance:
        out.append('BUDGET')
    return out


def snapshot(cfg: dict, attempt: str, job_id: str | None) -> dict:
    import boto3
    st = cfg['storage']
    session = boto3.Session(profile_name=st['profile'])
    base = f"s3://{st['bucket']}/{st['prefix']}/attempts/{attempt}/artifacts"
    local = Path(ROOT / 'evidence' / attempt / 'stage.json')
    runtime = json.loads(local.read_text()) if local.is_file() else {}
    progress = s3_json(session, f'{base}/{attempt}/progress.json')
    heartbeat = s3_json(session, f'{base}/heartbeat.json')
    completion = s3_json(session, f'{base}/completion.json')
    rows = wandb_rows(cfg, runtime.get('wandb_run_id') or os.environ.get('STRL_WANDB_RUN_ID', ''), 400) if runtime else []
    latest: dict = {}
    for r in rows:
        latest.update({k: v for k, v in r.items() if k in KEYS and v is not None})
    allowance = cfg['compute']['nodes'] * cfg['budget']['attempt_max_seconds'] / 3600 * (cfg['compute']['gpus_per_node'] / 8)
    return {'at': time.strftime('%H:%M:%S'), 'job': job_state(job_id), 'completion': (completion or {}).get('status'),
            'heartbeat_age_s': round(time.time() - heartbeat['at']) if heartbeat else None, 'latest': latest,
            'flags': flags(rows, progress, heartbeat, allowance), 'wandb_run_id': runtime.get('wandb_run_id')}


def line(snap: dict) -> str:
    m = snap['latest']
    fmt = lambda k, f='{:.3f}': (f.format(m[k]) if isinstance(m.get(k), (int, float)) else '-')  # noqa: E731
    return (f"{snap['at']} job={snap['job']} step={fmt('optimizer_step', '{:.0f}')} reward={fmt('reward/avg_raw_reward')} "
            f"task_metric={fmt('eval/task_metric')} signal={fmt('train/learning_signal_fraction')} step_s={fmt('timing/step', '{:.0f}')} "
            f"gen_s={fmt('timing/generate', '{:.0f}')} infra={fmt('environment/infra_error')} masked={fmt('environment/masked_group_fraction')} "
            f"node_h={fmt('budget/consumed_node_hours', '{:.2f}')} critic$={fmt('critic/cost_usd', '{:.2f}')} hb={snap['heartbeat_age_s']}s "
            f"{' '.join(snap['flags'])} {('[' + snap['completion'] + ']') if snap['completion'] else ''}")


def watch(cfg: dict, attempt: str, job_id: str | None, interval: float = 60, once: bool = False) -> dict:
    while True:
        snap = snapshot(cfg, attempt, job_id)
        print(line(snap), flush=True)
        if once or snap['completion'] or snap['job'] in {'COMPLETED', 'FAILED', 'CANCELED', 'CANCELLED'}:
            return snap
        time.sleep(interval)
