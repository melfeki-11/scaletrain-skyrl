"""strl: one CLI for the whole experiment lifecycle. Run from the repo root with the local control venv (.venv).

  plan      <exp> [--against <exp>] [--set K=V] [--network network.json]
                                           resolve, validate, derive infra + layout, estimate time/cost/memory, print warnings
  fetch     <exp>                          download the pinned model + router wheel, then prepare the environment's data
  prepare   <exp>                          (re)build the environment's data only (coding: validates references in Modal)
  executor  deploy|check                   the Modal code executor used by coding rewards
  overlay                                  build assets/overlay (Modal/Harbor packages for the training image)
  capacity  [--gpus-per-node 8,4,2,1]      read-only ScaleTrain/EKS capacity by team and borrowing route
  stage     <exp> --attempt A --mode dry|canary|train [--set K=V]  upload assets + render job.json (never submits)
  stage-model <exp> --attempt A            render a CPU job that downloads the model in a pod and uploads it to S3 (large models)
  submit    <job.json> [--user-id U]       submit once; receipt written before the POST
  status    <job_id> | --list              job state
  monitor   <exp> --attempt A [--job-id J] live status line + alerts until the attempt finishes
  ledger    <exp> --attempt A --job-id J   record a terminal attempt's allocated interval in the program budget ledger
  verify    <exp> --attempt A              independent re-scoring of the final evaluation from S3
  compare   <exp>:<attempt> ...            paired comparison of finished attempts on the same held-out ids
  eval-llm  <exp> --model M                reference eval of an external LLM with the training prompts/harness
  critic    <exp> [--n 5]                  run the LiteLLM critic on sample failures (no GPUs)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = os.environ.get('STRL_SCALETRAIN_USER', 'mohamed.elfeki')


def _load(path: str, overrides: list[str] | None = None) -> dict:
    from .experiment import ExperimentError, load
    try:
        return load(path, overrides=overrides)
    except ExperimentError as error:
        print(f'experiment {path} is invalid:\n{error}', file=sys.stderr)
        raise SystemExit(2)


def cmd_plan(a):
    from .derive import plan
    from .experiment import diff
    cfg = _load(a.experiment, a.set)
    report = plan(cfg, json.loads(Path(a.network).read_text()) if a.network else None)
    if a.against:
        report['diff_vs'] = {a.against: [{'key': k, 'from': x, 'to': y} for k, x, y in diff(_load(a.against), cfg)]}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
        return
    print(f"experiment {report['experiment']}  sha {report['sha']}\n  chain: {' -> '.join(Path(c).name for c in report['chain'])}")
    for k, v in report['axes'].items():
        print(f'  {k:12s} {v}')
    d, e = report['derived'], report['estimates']
    print(f"  placement    {d['world_size']} GPU(s), {d['num_engines']} vLLM engine(s), {d['rollouts_per_step']} rollouts/step "
          f"({d['rollouts_per_gpu']}/GPU, {d['micro_batches_per_gpu']} micro-batches), max_model_len {d['max_model_len']}, env_class {d['skyrl_env_class']}")
    print(f"  estimates    step ~{e['step_s']}s (generate {e['generate_s']} + train {e['train_s']} + other {e['other_s']}), ~{e['steps_in_attempt']} steps/attempt, "
          f"{e['node_hours']:.2f} node-h; memory train {e['memory']['training_phase_gb']} GB / rollout {e['memory']['rollout_phase_gb']} GB "
          f"(fits={e['memory']['fits']}); from profile {e['profile']}"
          + ("" if e.get('calibrated') else " (UNCALIBRATED: different environment/model; the canary measures the real step time)"))
    lay, mem = report['layout'], e['memory']
    print(f"  layout       {lay['mode']} | FSDP group {lay['fsdp_group']} x {lay['replicas']} replica(s), cpu_offload={lay['cpu_offload']} | "
          f"vLLM TP{lay['tp']} x {lay['engines']} engine(s) | per-GPU train {lay['train_gb_per_gpu']} GB, rollout {lay['rollout_gb_per_gpu']} GB "
          f"(weights {mem['engine_weights_gb']}, KV room {mem['kv_room_gb']})"
          + (f", LoRA {mem['lora_params_m']}M params" if mem['lora_params_m'] else '') + f"; recommended: {lay['recommended']['reason']}")
    print(f"  monitoring   {report['monitoring']['wandb']}  critic={report['monitoring']['critic']}")
    ledger_path = Path(cfg['budget'].get('ledger') or '')
    if ledger_path.is_file():   # the program budget the next stage would draw from (node-seconds)
        from .budget import remaining_seconds
        ledger = json.loads(ledger_path.read_text())
        left = remaining_seconds(ledger['attempts'], ledger['total_seconds'])
        from .budget import node_fraction
        from .stage import CANARY_OVERLAY, canary_overlay_for
        train = int(node_fraction(cfg) * cfg['budget']['attempt_max_seconds'])
        small = canary_overlay_for(cfg) == CANARY_OVERLAY
        canary = int(min(3600, cfg['budget']['canary_max_seconds'] + 600) * (1 / 8 if small else node_fraction(cfg)))
        shape = '1-GPU' if small else '{}x{}-GPU'.format(cfg['compute']['nodes'], cfg['compute']['gpus_per_node'])
        print(f"  budget       {ledger_path}: {left} of {ledger['total_seconds']} node-s left; a train attempt reserves up to "
              f"{train} node-s, a {shape} canary up to {canary} ({len(ledger['attempts'])} attempts recorded)")
    for w in report['warnings']:
        print(f'  ! {w}')
    for k, rows in report.get('diff_vs', {}).items():
        print(f'  diff vs {k}:')
        for row in rows:
            print(f"     {row['key']}: {row['from']!r} -> {row['to']!r}")


def cmd_fetch(a):
    cfg = _load(a.experiment)
    from huggingface_hub import snapshot_download
    snapshot_download(cfg['model']['id'], revision=cfg['model']['revision'], local_dir=cfg['model']['local_path'])
    from .runtime import ROUTER_SHA256, ROUTER_WHEEL
    import hashlib
    runtime = ROOT / 'assets/runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    wheel = runtime / ROUTER_WHEEL
    if not wheel.exists():
        import urllib.request
        urllib.request.urlretrieve('https://github.com/SumanthRH/router/releases/download/0.1.14.post1/' + ROUTER_WHEEL, wheel)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != ROUTER_SHA256:
        raise SystemExit('vllm-router wheel hash mismatch')
    from .tasks import get_environment
    try:
        get_environment(cfg['environment']['name']).assert_manifest(cfg)
        print(f"data already prepared and verified at {cfg['run']['data_dir']}")
    except Exception:
        cmd_prepare(a)


def cmd_prepare(a):
    cfg = _load(a.experiment)
    from .tasks import get_environment
    env = get_environment(cfg['environment']['name'])
    kwargs = {}
    if env.needs_modal:
        from transformers import AutoTokenizer
        from .services.modal import local_profile
        kwargs['tokenizer'] = AutoTokenizer.from_pretrained(cfg['model']['local_path'])
        with local_profile(cfg['services']['modal'].get('environment', 'main')):
            from .tasks.coding import executor
            print(json.dumps(executor.ensure_deployed()))
            manifest = env.prepare_data(cfg, **kwargs)
    else:
        manifest = env.prepare_data(cfg)
    print(json.dumps({k: manifest.get(k) for k in ('counts', 'files', 'prompt_tokens', 'executor_version') if k in manifest} or
                     {'files': manifest.get('files')}, indent=1, default=str))
    if manifest.get('dropped'):
        print(f"dropped {len(manifest['dropped'])} task(s): " + json.dumps(dict(list(manifest['dropped'].items())[:10]), indent=1))


def cmd_executor(a):
    from .services.modal import local_profile
    from .tasks.coding import executor
    with local_profile(a.environment):
        print(json.dumps(executor.deploy() if a.action == 'deploy' else executor.ensure_deployed(auto_deploy=False)))


def cmd_overlay(a):
    subprocess.run([sys.executable, str(ROOT / 'tools/build_overlay.py'), *(['--force'] if a.force else [])], check=True)


def cmd_capacity(a):
    subprocess.run([sys.executable, str(ROOT / 'tools/capacity.py'), '--nodes', str(a.nodes), '--gpus-per-node', a.gpus_per_node,
                    '--output', a.output], check=True, env={**os.environ, 'AWS_PROFILE': os.environ.get('AWS_PROFILE', 'ml-admin')})


def cmd_stage(a):
    from .stage import stage
    receipt = stage(a.experiment, attempt=a.attempt, mode=a.mode, team=a.team, allow_borrowing=a.allow_borrowing, ledger=a.ledger,
                    output=a.output or f'evidence/{a.attempt}/job.json', overrides=a.set)
    Path(f'evidence/{a.attempt}').mkdir(parents=True, exist_ok=True)
    Path(f'evidence/{a.attempt}/stage.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=1))
    print(f"\nnext: python -m strl submit {receipt['job_file']}")


def cmd_stage_model(a):
    from .stage import stage_model
    receipt = stage_model(a.experiment, attempt=a.attempt, team=a.team, output=a.output or f'evidence/{a.attempt}/job.json', overrides=a.set)
    Path(f'evidence/{a.attempt}').mkdir(parents=True, exist_ok=True)
    Path(f'evidence/{a.attempt}/stage.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=1))
    print(f"\nnext: python -m strl submit {receipt['job_file']}  (CPU; then stage the experiment: it reads {receipt['manifest']})")


def _scaletrain(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / 'tools/scaletrain.py'), *args], capture_output=True, text=True)


def cmd_submit(a):
    job = json.loads(Path(a.job).read_text())
    receipt = a.receipt or str(Path(a.job).parent / 'submission.json')
    check = _scaletrain('validate', a.job)
    if check.returncode:
        raise SystemExit(check.stderr or check.stdout)
    out = _scaletrain('submit', a.job, '--user-id', a.user_id, '--receipt', receipt)
    if out.returncode:  # never resubmit blindly: reconcile by exact job name
        listing = json.loads(_scaletrain('list', '--user-id', a.user_id, '--limit', '20').stdout or '[]')
        match = [j for j in listing if j['name'] == job['job_name']]
        print(json.dumps({'submit_error': (out.stderr or out.stdout).strip()[-300:], 'reconciled_by_name': match[:1]}, indent=1))
        return
    print(out.stdout.strip())


def cmd_status(a):
    out = _scaletrain('list', '--user-id', a.user_id, '--limit', '20') if a.list else _scaletrain('get', a.job_id)
    print(out.stdout.strip() or out.stderr.strip())


def cmd_monitor(a):
    from .monitor.live import watch
    watch(_load(a.experiment), a.attempt, a.job_id, interval=a.interval, once=a.once)


def cmd_ledger(a):
    cfg = _load(a.experiment)
    ledger_path = Path(a.ledger) if a.ledger else ROOT / cfg['budget']['ledger']
    ledger = json.loads(ledger_path.read_text())
    if any(x['job_id'] == a.job_id for x in ledger['attempts']):
        raise SystemExit(f'{a.job_id} already recorded')
    job = json.loads(_scaletrain('get', a.job_id).stdout)
    if job['status'] not in {'COMPLETED', 'FAILED', 'CANCELED', 'CANCELLED'}:
        raise SystemExit(f'job is {job["status"]}; record it only after a terminal state')
    stage_receipt = json.loads(Path(f'evidence/{a.attempt}/stage.json').read_text())
    entry = {'job_id': a.job_id, 'attempt': a.attempt, 'status': job['status'], 'allocated_at': str(job['created_at']).replace(' ', 'T')[:19] + 'Z',
             'terminal_at': a.terminal_at or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'gpu_fraction': stage_receipt['gpu_fraction'],
             'nodes': int(stage_receipt.get('nodes', 1)),
             'evidence': 'conservative: job created_at to first terminal observation (includes queue/pod start)'}
    ledger['attempts'].append(entry)
    ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
    from .budget import remaining_seconds
    print(json.dumps({'recorded': entry, 'remaining_node_seconds': remaining_seconds(ledger['attempts'], ledger['total_seconds'])}, indent=1))


def cmd_verify(a):
    from .verify_results import verify
    print(json.dumps(verify(_load(a.experiment), a.attempt), indent=1, default=str))


def cmd_compare(a):
    from .monitor.compare import compare, markdown
    report = compare(a.runs)
    Path('evidence/compare').mkdir(parents=True, exist_ok=True)
    Path('evidence/compare/report.json').write_text(json.dumps(report, indent=2, default=str))
    print(markdown(report))


def cmd_eval_llm(a):
    from . import evals
    cfg = _load(a.experiment)
    project = ((cfg.get('services') or {}).get('litellm') or {}).get('project_id', '696011502a26dc4a64b24f55')
    fn = evals.coding_harbor if cfg['environment']['name'] == 'coding_harbor' or a.harbor else evals.coding_exec
    if fn is evals.coding_exec:
        from .services.modal import local_profile
        with local_profile(cfg['services']['modal'].get('environment', 'main')):
            print(json.dumps(fn(cfg, a.split, a.limit, a.model, a.max_completion_tokens, project), indent=1))
    else:
        print(json.dumps(fn(cfg, a.split, a.limit, a.model, a.max_completion_tokens, project), indent=1))


def cmd_critic(a):
    from .monitor import critic
    from .tasks.coding import data as coding_data
    cfg = _load(a.experiment)
    critic.configure({**cfg['tracking'].get('critic', {}), 'enabled': True})
    tasks = coding_data.load_tasks(cfg, 'canary')[:a.n] if cfg['environment']['name'].startswith('coding') else []
    rows = []
    for task in tasks:
        broken = coding_data.stub(task.reference)
        rows.append(critic.analyse({'source_step': 0, 'idx': 0, 'prompt': task.prompt, 'completion': f'```python\n{broken}\n```',
                                    'env_failure': 'wrong_answer'}))
    print(json.dumps([{k: r[k] for k in ('critic_mode', 'rationale', 'fix_hint', '_cost')} for r in rows], indent=1))
    print(json.dumps({'calls': len(rows), 'cost_usd': round(sum(r['_cost'] for r in rows), 5)}))


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m strl', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('plan'); s.add_argument('experiment'); s.add_argument('--against'); s.add_argument('--json', action='store_true')
    s.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='dotted experiment override, YAML value (repeatable)')
    s.add_argument('--network', help="a probe result (an attempt's artifacts/network.json) for the layout recommendation")
    s = sub.add_parser('fetch'); s.add_argument('experiment')
    s = sub.add_parser('prepare'); s.add_argument('experiment')
    s = sub.add_parser('executor'); s.add_argument('action', choices=['deploy', 'check']); s.add_argument('--environment', default='main')
    s = sub.add_parser('overlay'); s.add_argument('--force', action='store_true')
    s = sub.add_parser('capacity'); s.add_argument('--nodes', type=int, default=1); s.add_argument('--gpus-per-node', default='8,4,2,1')
    s.add_argument('--output', default='evidence/capacity.json')
    s = sub.add_parser('stage'); s.add_argument('experiment'); s.add_argument('--attempt', required=True)
    s.add_argument('--mode', choices=['dry', 'canary', 'train'], required=True); s.add_argument('--team'); s.add_argument('--allow-borrowing', action='store_true')
    s.add_argument('--output'); s.add_argument('--ledger', help='program budget ledger (default: budget.ledger of the experiment)')
    s.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='dotted experiment override, YAML value (repeatable)')
    s = sub.add_parser('stage-model'); s.add_argument('experiment'); s.add_argument('--attempt', required=True); s.add_argument('--team')
    s.add_argument('--output'); s.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    s = sub.add_parser('submit'); s.add_argument('job'); s.add_argument('--user-id', default=USER); s.add_argument('--receipt')
    s = sub.add_parser('status'); s.add_argument('job_id', nargs='?'); s.add_argument('--list', action='store_true'); s.add_argument('--user-id', default=USER)
    s = sub.add_parser('monitor'); s.add_argument('experiment'); s.add_argument('--attempt', required=True); s.add_argument('--job-id')
    s.add_argument('--interval', type=float, default=60); s.add_argument('--once', action='store_true')
    s = sub.add_parser('ledger'); s.add_argument('experiment'); s.add_argument('--attempt', required=True); s.add_argument('--job-id', required=True)
    s.add_argument('--terminal-at', help='ISO time of the terminal observation (default: now)')
    s.add_argument('--ledger', help='program budget ledger (default: budget.ledger of the experiment)')
    s = sub.add_parser('verify'); s.add_argument('experiment'); s.add_argument('--attempt', required=True)
    s = sub.add_parser('compare'); s.add_argument('runs', nargs='+', help='experiment.yaml:attempt')
    s = sub.add_parser('eval-llm'); s.add_argument('experiment'); s.add_argument('--model', default='openai/gpt-6-luna')
    s.add_argument('--split', default='validation'); s.add_argument('--limit', type=int, default=20); s.add_argument('--max-completion-tokens', type=int, default=4096)
    s.add_argument('--harbor', action='store_true', help='use the Harbor/Terminus-2 agent even for coding_exec tasks')
    s = sub.add_parser('critic'); s.add_argument('experiment'); s.add_argument('--n', type=int, default=5)
    a = p.parse_args(argv)
    os.chdir(ROOT)
    {'plan': cmd_plan, 'fetch': cmd_fetch, 'prepare': cmd_prepare, 'executor': cmd_executor, 'overlay': cmd_overlay, 'capacity': cmd_capacity,
     'stage': cmd_stage, 'stage-model': cmd_stage_model, 'submit': cmd_submit, 'status': cmd_status, 'monitor': cmd_monitor, 'ledger': cmd_ledger, 'verify': cmd_verify,
     'compare': cmd_compare, 'eval-llm': cmd_eval_llm, 'critic': cmd_critic}[a.cmd](a)


if __name__ == '__main__':
    main()
