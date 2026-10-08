"""Live monitor for a running attempt: ScaleTrain state + S3 progress/heartbeat + W&B metrics, one status line per poll.

    python -m strl monitor experiments/<name>.yaml --attempt <attempt> [--job-id <id>] [--interval 60]

Flags (also sent as W&B alerts by the worker itself): STALL (no optimizer step for max(10 min, 3 x median step)), NAN, ENV_FAILURES
(masked groups / infra errors above 5%), BUDGET (node-hours above the attempt allowance), DONE / FAILED (completion.json);
multi-node: NETWORK (probe found no EFA / NCCL on sockets), PEER_STALE (a peer heartbeat older than 5 min).
Per-domain environments (mimo): the line shows `<domain> <eval task metric>/<train pass rate %>`; DOMAIN_MASKED (any domain with more than
15% of its GRPO groups masked), STEP_SLOW (timing/step above 1.5 x the plan estimate recorded in stage.json / runtime.json), MEM (a GPU
memory metric above 76 GB). DOMAIN_MASKED / STEP_SLOW / MEM also send one W&B alert per run per flag (``notify``).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEYS = ['optimizer_step', 'reward/avg_raw_reward', 'eval/task_metric', 'train/learning_signal_fraction', 'train/pass_at_1',
        'environment/infra_error', 'environment/masked_group_fraction', 'timing/step', 'timing/generate', 'policy/grad_norm',
        'policy/policy_loss', 'budget/consumed_node_hours', 'critic/cost_usd', 'generate/avg_num_tokens', 'train/truncation_rate',
        'generate/num_seq_before_merge', 'generate/num_seq_after_merge']


DOMAIN_MASKED_LIMIT = 0.15
TIMEOUT_LIMIT = 0.30                 # informational: a policy that is out of time is not an environment failure
STEP_SLOW_FACTOR = 1.5
MEM_LIMIT_GB = 76.0
DOMAIN_KEY = re.compile(r'^(eval/([a-z]+)/task_metric|generate/([a-z]+)/pass_rate|environment/([a-z]+)/masked_group_fraction|environment/([a-z]+)/timeout_group_fraction)$')
MERGE_RATIO_LIMIT = 0.8   # sequences after / before step-wise merging; a healthy multi-turn run is far below (Qwen3.8: ~0.26 per turn)
ALERTED = ('DOMAIN_MASKED', 'TIMEOUTS', 'STEP_SLOW', 'MEM', 'MERGE')


def is_memory_key(key: str) -> bool:
    return key.startswith(('gpu/', 'ray/gpu')) and 'mem' in key and 'gb' in key


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


def domain_status(latest: dict) -> dict[str, dict[str, float | None]]:
    """{domain: {task_metric, pass_rate, masked}} from the latest per-domain keys (empty for single-domain environments)."""
    out: dict[str, dict] = {}
    for key, value in latest.items():
        match = DOMAIN_KEY.match(key)
        if match and isinstance(value, (int, float)):
            name = next(g for g in match.groups()[1:] if g)
            out.setdefault(name, {})[key.split('/')[-1]] = float(value)
    return out


def flags(rows: list[dict], progress: dict | None, heartbeat: dict | None, node_hour_allowance: float, plan_step_seconds: float | None = None) -> list[str]:
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
    seen: dict = {}
    for r in rows[-20:]:   # the newest value of each key (per-domain eval keys appear only on eval rows)
        seen.update({k: v for k, v in r.items() if v is not None})
    for name, fraction in sorted(domain_status(seen).items()):
        if (fraction.get('masked_group_fraction') or 0) > DOMAIN_MASKED_LIMIT:
            out.append(f"DOMAIN_MASKED({name}={fraction['masked_group_fraction']:.2f})")
        if (fraction.get('timeout_group_fraction') or 0) > TIMEOUT_LIMIT:
            out.append(f"TIMEOUTS({name}={fraction['timeout_group_fraction']:.2f}: a trial ran out of time (it trains on the verifier's reward if it left tokens, else it is a zero-loss placeholder); raise the caps or shorten turns)")
    if plan_step_seconds and isinstance(seen.get('timing/step'), (int, float)) and seen['timing/step'] > STEP_SLOW_FACTOR * plan_step_seconds:
        out.append(f"STEP_SLOW({seen['timing/step']:.0f}s > {STEP_SLOW_FACTOR} x plan {plan_step_seconds:.0f}s)")
    before, after = seen.get('generate/num_seq_before_merge'), seen.get('generate/num_seq_after_merge')
    if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before >= 40 and after / before > MERGE_RATIO_LIMIT:
        out.append(f'MERGE({after:.0f}/{before:.0f} sequences kept: step-wise turns are not merging, several x more training tokens; strl check-template)')
    memory = {k: v for k, v in seen.items() if is_memory_key(k) and isinstance(v, (int, float))}
    if memory and max(memory.values()) > MEM_LIMIT_GB:
        key = max(memory, key=memory.get)
        out.append(f'MEM({key}={memory[key]:.1f}GB > {MEM_LIMIT_GB:g})')
    return out


def alert_once(cfg: dict, run_id: str | None, found: list[str], sent: set[str], notify=None) -> list[str]:
    """One W&B alert per run per flag for DOMAIN_MASKED / STEP_SLOW / MEM (``sent`` carries the flags already alerted). Returns the new ones."""
    fresh = []
    for flag in found:
        name = flag.split('(')[0]
        if name in ALERTED and name not in sent:
            sent.add(name)
            fresh.append(flag)
            try:
                (notify or notify_wandb)(cfg, run_id, name, flag)
            except Exception:
                pass
    return fresh


def notify_wandb(cfg: dict, run_id: str | None, name: str, text: str) -> None:
    """Send the alert from a SEPARATE monitor run (``<run_id>-monitor``, same group): attaching to the training run itself made a second
    concurrent writer and run.finish() could mark the live run finished. A no-op unless alerts are enabled."""
    if not run_id or not cfg['tracking'].get('alerts', True):
        return
    import wandb
    from ..services.wandb import local_key
    from ..tracking import alert
    local_key()
    run = wandb.init(project=cfg['tracking']['project'], entity=cfg['tracking']['entity'], id=f'{run_id}-monitor', resume='allow', reinit=True,
                     group=cfg['tracking'].get('group'), job_type='monitor', name=f'{run_id}-monitor')
    try:
        alert(f'strl monitor: {name}', f'{cfg["name"]}: {text}', 'WARN')
    finally:
        run.finish()


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
        latest.update({k: v for k, v in r.items() if (k in KEYS or DOMAIN_KEY.match(k) or is_memory_key(k)) and v is not None})
    allowance = cfg['compute']['nodes'] * cfg['budget']['attempt_max_seconds'] / 3600 * (cfg['compute']['gpus_per_node'] / 8)
    found = flags(rows, progress, heartbeat, allowance, runtime.get('plan_step_seconds') or stage_plan_estimate(runtime))
    if cfg['compute']['nodes'] > 1:
        found += cluster_flags(s3_json(session, f'{base}/network.json'),
                               [s3_json(session, f'{base}/peer-{r}-heartbeat.json') for r in range(1, cfg['compute']['nodes'])])
    return {'at': time.strftime('%H:%M:%S'), 'job': job_state(job_id), 'completion': (completion or {}).get('status'),
            'heartbeat_age_s': round(time.time() - heartbeat['at']) if heartbeat else None, 'latest': latest,
            'flags': found, 'wandb_run_id': runtime.get('wandb_run_id')}


def stage_plan_estimate(runtime: dict) -> float | None:
    """stage.json written before plan_step_seconds existed: no estimate (STEP_SLOW stays silent)."""
    value = runtime.get('plan_step_seconds')
    return float(value) if value else None


def cluster_flags(network: dict | None, peers: list[dict | None], now: float | None = None) -> list[str]:
    """Multi-node: NETWORK when the probe found no EFA (or NCCL on sockets), PEER_STALE when a peer heartbeat is > 5 min old."""
    now = time.time() if now is None else now
    out = []
    if network and (not network.get('efa_available') or network.get('transport') == 'socket'):
        cross = (network.get('bandwidth_gbps') or {}).get('cross_node')
        out.append(f"NETWORK(transport={network.get('transport')}, efa={network.get('efa_available')}"
                   + (f', cross_node={cross}GB/s' if cross is not None else '') + ')')
    for rank, beat in enumerate(peers, start=1):
        if beat and now - float(beat['at']) > 300:
            out.append(f'PEER_STALE(rank {rank}, {now - float(beat["at"]):.0f}s)')
    return out


def fmt_dom(value, spec: str, scale: float = 1.0) -> str:
    return '-' if value is None else spec.format(value * scale)


def line(snap: dict) -> str:
    m = snap['latest']
    fmt = lambda k, f='{:.3f}': (f.format(m[k]) if isinstance(m.get(k), (int, float)) else '-')  # noqa: E731
    domains = ' '.join(f"{name} {fmt_dom(d.get('task_metric'), '{:.2f}')}/{fmt_dom(d.get('pass_rate'), '{:.0f}', 100)}%"
                       for name, d in sorted(domain_status(m).items()))
    return (f"{snap['at']} job={snap['job']} step={fmt('optimizer_step', '{:.0f}')} reward={fmt('reward/avg_raw_reward')} "
            f"task_metric={fmt('eval/task_metric')} signal={fmt('train/learning_signal_fraction')} step_s={fmt('timing/step', '{:.0f}')} "
            f"gen_s={fmt('timing/generate', '{:.0f}')} infra={fmt('environment/infra_error')} masked={fmt('environment/masked_group_fraction')} "
            f"node_h={fmt('budget/consumed_node_hours', '{:.2f}')} critic$={fmt('critic/cost_usd', '{:.4f}')} hb={snap['heartbeat_age_s']}s "
            f"{(domains + ' ') if domains else ''}{' '.join(snap['flags'])} {('[' + snap['completion'] + ']') if snap['completion'] else ''}")


def watch(cfg: dict, attempt: str, job_id: str | None, interval: float = 60, once: bool = False, notify=None) -> dict:
    sent: set[str] = set()
    while True:
        snap = snapshot(cfg, attempt, job_id)
        print(line(snap), flush=True)
        alert_once(cfg, snap.get('wandb_run_id'), snap['flags'], sent, notify)
        if once or snap['completion'] or snap['job'] in {'COMPLETED', 'FAILED', 'CANCELED', 'CANCELLED'}:
            return snap
        time.sleep(interval)
