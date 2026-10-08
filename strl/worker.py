"""ScaleTrain worker for any experiment: budget -> services -> Ray -> preflight -> train -> select -> final eval.

Phases and receipts (all synced to ``<artifact_uri>``):
  budget.json (write-once) -> worker-started.json -> artifacts/runtime.json (preflight) -> [dry-run.json] -> training.log,
  heartbeat.json (every 60 s), <run>/progress.json + metrics.jsonl -> final_eval/* -> completion.json (completed | failed)
Modes (STRL_MODE): dry (CPU: every gate except GPU training), canary (2 updates, no final eval), train.
Watchdog: if training makes no progress for budget.stall_seconds, the child is killed, an alert is sent, and the latest
export (if any) is still evaluated, so a stalled run is salvaged instead of silently burning its allocation.

Multi-node (compute.nodes > 1; one pod per node, rank = JOB_COMPLETION_INDEX): every node runs the same budget / services /
preflight / SkyRL patches before Ray. The head (rank 0) publishes leader-network.json, waits for N nodes x G GPUs, runs the
cluster probe (artifacts/network.json), trains with a liveness watcher (aborts when a node is lost), verifies each node's own
checkpoint shards over Ray and evaluates on its own GPUs. Peers join Ray and wait (strl.multinode.peer_wait): exit 0 once
completion.json exists, nonzero when the head goes silent. Single-node jobs take exactly the previous code path.
"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

import boto3
from botocore.exceptions import ClientError

from .experiment import current, finetune, save_optimizer_state
from .network import head_network, ray_start_args
from .resources import allocated_cpus
from .storage import get_json, put_json, split_uri, sync_local
from .tasks import get_environment
from .verify import verify_adapter_change, verify_cluster_shards, verify_export_change, verify_shards

RAY = '/opt/skyrl-venv/bin/ray'
SYNC_EXCLUDE = ('checkpoints/',)  # optimizer shards stay on the node; exports/metrics/evals go to S3


CHILD_HARD_STOP_SECONDS = 480   # the training child is killed this long before the attempt deadline (train mode)


def train_reserve_seconds(cfg: dict, mode: str) -> int:
    """Seconds before the attempt deadline at which training must have FINISHED (GRPO_TRAIN_DEADLINE = deadline - this). With an offline final
    eval it is that eval's reserve. With the in-loop final eval (agentic environments) the forced eval + export run inside the training child,
    which is hard-killed at deadline - 480: the soft stop must leave room for them before that (it was 120 s: a run ending on its deadline
    was killed mid-export, the attempt failed and the paired result was never written)."""
    if mode != 'train':
        return 120
    from .experiment import RETENTION_RESERVE_SECONDS
    retention = RETENTION_RESERVE_SECONDS if cfg['evaluation'].get('retention') else 0
    if cfg['evaluation']['final_eval']:
        return int(cfg['budget']['final_eval_reserve_seconds']) + retention
    return CHILD_HARD_STOP_SECONDS + 120 + retention


class Stalled(RuntimeError):
    pass


def _progress_mtime(progress: Path | None) -> float | None:
    return progress.stat().st_mtime if progress is not None and progress.is_file() else None


def previous_prefix(prefix: str, previous: str) -> str:
    """s3://b/<p>/attempts/<this>/artifacts -> s3://b/<p>/attempts/<previous>/artifacts"""
    head, sep, _ = prefix.rpartition('/attempts/')
    if not sep:
        raise RuntimeError(f'cannot derive the previous attempt prefix from {prefix}')
    return f'{head}/attempts/{previous}/artifacts'


def prepare_resume(prefix: str, run_name: str, previous: str, root: Path) -> dict:
    """Continue ``previous`` (a preempted/stopped attempt with durable checkpoints): point SkyRL at its latest checkpoint (STRL_RESUME_PATH,
    read by strl.overrides in the training child), and carry over its in-loop eval records (the step-0 eval stays the base of the paired
    result) and its canary record (the canary rule ran in the earlier attempt)."""
    import boto3
    old = previous_prefix(prefix, run_name)
    bucket, key = split_uri(f'{old}/checkpoints/latest_ckpt_global_step.txt')
    s3 = boto3.client('s3')
    try:
        step = int(s3.get_object(Bucket=bucket, Key=key)['Body'].read().decode().strip())
    except Exception as error:
        raise RuntimeError(f'run.resume_from={previous}: no durable checkpoint at {old}/checkpoints ({type(error).__name__}); '
                           'only attempts run with evaluation.durable_checkpoints: true can be resumed') from error
    os.environ['STRL_RESUME_PATH'] = f'{old}/checkpoints/global_step_{step}'
    carried = []
    for rel, target in ((f'{previous}/exports/evaluation_samples.jsonl', root / 'exports' / 'evaluation_samples.jsonl'),
                        (f'{previous}/canary.json', root / 'canary.json')):
        b, k = split_uri(f'{old}/{rel}')
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(b, k, str(target))
            carried.append(rel)
        except Exception:
            pass
    return {'attempt': previous, 'step': step, 'checkpoint': os.environ['STRL_RESUME_PATH'], 'carried': carried}


def gather_exports(root, prefix):
    """Multi-node: SkyRL writes HF exports from FSDP rank 0, which is on a random node (bundles are ordered by Ray NodeID). Upload every
    node's exports to S3, then fetch onto the head whatever the head lacks, so selection, merge, final eval and the S3 copy of the trained
    model never depend on where rank 0 landed."""
    from .storage import download_prefix, sync_cluster
    rel = Path(root).relative_to('artifacts')
    target = f"{prefix}/{rel}/exports"
    try:
        report = sync_cluster(str(Path(root).resolve() / 'exports'), target)
    except Exception as error:
        return {'error': f'{type(error).__name__}: {error}'}
    try:
        download_prefix(target, Path(root) / 'exports')
    except Exception as error:
        return {'nodes': report, 'download_error': f'{type(error).__name__}: {error}'}
    return {'nodes': report}


def _background_sync():
    """The periodic artifacts upload. A transient S3 error must never kill training: it is logged and retried on the next cycle."""
    try:
        sync_local('artifacts', os.environ['GRPO_ARTIFACT_URI'], exclude=SYNC_EXCLUDE)
    except Exception as error:
        print(f'[strl] artifacts sync failed (retried next cycle): {type(error).__name__}: {error}', file=sys.stderr, flush=True)


def child(args, log, deadline, canary_path=None, canary_seconds=1200, env=None, progress=None, stall_seconds=None, abort=None):
    """``abort``: optional callable returning a reason to stop (multi-node liveness watcher)."""
    started = time.time()
    with Path(log).open('w') as output:
        process = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        last_sync, sync_thread = 0, None
        try:
            while process.poll() is None:
                now = time.time()
                if now >= deadline:
                    raise TimeoutError('Phase exceeded authorized time budget')
                reason = abort() if abort is not None else None
                if reason:
                    raise RuntimeError(f'Cluster liveness: {reason}')
                if canary_path is not None and now >= started + canary_seconds and not canary_path.exists():
                    raise TimeoutError('Canary did not verify two updates in time')
                seen = _progress_mtime(progress)
                if stall_seconds and seen is not None and now - seen > stall_seconds:
                    raise Stalled(f'No training progress for {now - seen:.0f}s (stall_seconds={stall_seconds})')
                if now - last_sync >= 60:
                    Path('artifacts/heartbeat.json').write_text(json.dumps({
                        'at': now, 'child_alive': True, 'elapsed_seconds': now - started, 'last_progress_at': seen,
                        'log_bytes': Path(log).stat().st_size if Path(log).exists() else 0}))
                    if sync_thread is None or not sync_thread.is_alive():   # never block the deadline / liveness checks on an upload
                        sync_thread = threading.Thread(target=_background_sync, daemon=True)
                        sync_thread.start()
                    last_sync = now
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(f'Worker phase failed with exit {process.returncode}; see {log}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def run_parallel(jobs, deadline):
    """jobs: [(args, log, env)]. All run concurrently; any failure or the deadline kills the rest."""
    procs = []
    try:
        for args, log, env in jobs:
            procs.append((subprocess.Popen(args, stdout=Path(log).open('w'), stderr=subprocess.STDOUT, start_new_session=True, env=env), log))
        while any(p.poll() is None for p, _ in procs):
            if time.time() >= deadline:
                raise TimeoutError('Parallel evaluation exceeded the budget')
            if any(p.poll() not in (None, 0) for p, _ in procs):
                break
            time.sleep(3)
        bad = [log for p, log in procs if p.poll() not in (0,)]
        if bad:
            raise RuntimeError('Evaluation shard failed: ' + ', '.join(map(str, bad)))
    finally:
        for p, _ in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)


def budget(runtime, cfg):
    prefix = runtime['artifact_uri']
    s3 = boto3.client('s3')
    bucket, key = split_uri(prefix + '/budget.json')
    now = time.time()
    seconds, total = int(runtime['remaining_seconds']), int(runtime['total_seconds'])
    if not 0 < seconds <= min(total, int(cfg['budget']['attempt_max_seconds'])):
        raise ValueError('Invalid remaining allocation budget')
    payload = {'started_at': now, 'deadline': now + seconds, 'nodes': cfg['compute']['nodes'], 'seconds': seconds,
               'gpu_fraction': float(runtime.get('gpu_fraction', 1.0)), 'attempt': runtime['run_name'], 'ledger_uri': runtime['budget_ledger_uri']}
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode(), ServerSideEncryption='AES256', IfNoneMatch='*')
    except ClientError as e:
        if e.response['ResponseMetadata']['HTTPStatusCode'] not in {409, 412}:
            raise
    result = get_json(prefix + '/budget.json')
    if result['attempt'] != runtime['run_name'] or result['seconds'] != seconds:
        raise RuntimeError('Attempt budget identity mismatch')
    return result


def setup_services(cfg, runtime, env_spec, mode):
    """W&B always; Modal when the environment needs it; LiteLLM key file when the critic is enabled. Returns a receipt."""
    from .services import wandb as wandb_service
    wandb_service.configure_env(cfg, runtime['wandb_run_id'])
    receipt = {'wandb': 'configured'}
    services = cfg.get('services') or {}
    modal_cfg = services.get('modal') or {}
    if env_spec.needs_modal:
        from .services import modal as modal_service
        path = modal_service.configure_from_locator(modal_cfg['credentials_uri'], Path('artifacts').resolve().parent / '.modal-runtime.toml',
                                                    modal_cfg.get('environment', 'main'))
        receipt['modal'] = modal_service.check(modal_cfg.get('environment', 'main'), modal_cfg.get('sandbox_secret'))
        receipt['modal']['profile_mode'] = oct(path.stat().st_mode & 0o777)
    critic = (cfg['tracking'].get('critic') or {})
    if critic.get('enabled') or env_spec.needs_litellm:
        from .services import litellm
        try:
            key_file = litellm.write_key_file(Path('artifacts').resolve().parent / '.litellm-key',
                                              (services.get('litellm') or {}).get('key_name', litellm.KEY_NAME))
            os.environ['STRL_LITELLM_KEY_FILE'] = str(key_file)
            if critic.get('enabled'):
                os.environ['STRL_CRITIC'] = json.dumps(critic)
                receipt['critic'] = 'enabled'
            if env_spec.needs_litellm:
                receipt['litellm'] = 'key file written (environment judge)'
        except Exception as error:  # the critic is optional: never fail a training run because of it (dry runs report it)
            receipt['critic'] = f'disabled: {type(error).__name__}: {error}'[:200]
            if mode == 'dry' or env_spec.needs_litellm:   # an environment that needs the key (judge) cannot run without it
                raise
    return receipt


def dry_run_gates(cfg, runtime, env_spec, receipt):
    """Everything a GPU run depends on, exercised on CPU: real parser + validate_cfg at the target GPU count, generator and
    reward contract, environment executor, W&B write/readback, one critic call."""
    gates = {}
    env = dict(os.environ, STRL_STAGE='train')
    out = subprocess.run([sys.executable, '-m', 'strl.train', '--validate-only'], capture_output=True, text=True, env=env)
    if out.returncode:
        raise RuntimeError('strl.train --validate-only failed:\n' + out.stdout[-4000:] + out.stderr[-4000:])
    gates['skyrl_validate'] = json.loads(out.stdout.strip().splitlines()[-1])
    gates['environment'] = env_spec.dry_run_checks(cfg)
    from .services import wandb as wandb_service
    gates['wandb'] = wandb_service.smoke(cfg, runtime['wandb_run_id'] + 'd')
    if receipt.get('critic') == 'enabled':
        from .monitor import critic
        critic.configure({**cfg['tracking']['critic'], 'enabled': True})
        row = critic.analyse({'source_step': 0, 'idx': 0, 'prompt': 'Write a function add(a, b) returning a + b.',
                              'completion': 'def add(a, b):\n    return a - b', 'env_failure': 'wrong_answer'})
        gates['critic'] = {k: row[k] for k in ('critic_mode', 'rationale', '_cost')}
    return gates


def evaluate_test_split(cfg, env_spec, base_model, trained_model, eval_dir, deadline, limit=None, resolved=None):
    """Base on GPUs [0, n), trained on GPUs [n, 2n), n shards each, concurrently; reassemble in canonical id order. ``resolved``: a
    different resolved config for the eval processes (the retention environment); ``limit``: the first N held-out examples."""
    from .finalize import merge_shards, read_jsonl
    n = cfg['evaluation']['eval_shards_per_model']
    Path(eval_dir).mkdir(parents=True, exist_ok=True)
    jobs = []
    for offset, (label, model) in enumerate([('base', base_model), ('candidate', trained_model)]):
        for shard in range(n):
            env = os.environ.copy()
            env.pop('RAY_ADDRESS', None)
            env['CUDA_VISIBLE_DEVICES'] = str(offset * n + shard)
            if resolved:
                env['STRL_RESOLVED'] = str(resolved)
            args = [sys.executable, '-m', env_spec.eval_module, '--model', str(model), '--output', str(eval_dir / f'{label}-{shard}.jsonl'),
                    '--rank', str(shard), '--world-size', str(n), *(['--limit', str(limit)] if limit else [])]
            jobs.append((args, eval_dir / f'{label}-{shard}.log', env))
    run_parallel(jobs, deadline)
    ids = env_spec.final_eval_ids(cfg)
    ids = ids[:limit] if limit else ids
    return {label: merge_shards([read_jsonl(eval_dir / f'{label}-{s}.jsonl') for s in range(n)], ids) for label in ('base', 'candidate')}


def latest_export(root: Path) -> Path | None:
    steps = sorted((int(p.name.split('_')[-1]), p / 'policy') for p in (Path(root) / 'exports').glob('global_step_*') if (p / 'policy').is_dir())
    return steps[-1][1] if steps else None


def run_retention(cfg, runtime, root, trained, deadline):
    """evaluation.retention: base vs trained on another environment's held-out set (e.g. GSM8K for a MiMo run). Never raises: the result
    or the reason it is missing goes to artifacts/final_eval/retention/result.json and W&B retention/<env>/*."""
    from .experiment import retention_config
    from .tasks import get_environment
    out_dir = Path('artifacts/final_eval/retention')
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        rcfg = retention_config(cfg)
        if trained is None:
            raise RuntimeError('no trained export to evaluate')
        renv = get_environment(rcfg['environment']['name'])
        rows = evaluate_test_split(rcfg, renv, cfg['model']['local_path'], trained, out_dir, deadline, limit=rcfg.get('_retention_limit'),
                                   resolved=Path('retention-resolved.json').resolve())
        stats = renv.paired_result(rows['base'], rows['candidate'], rcfg)
        result = {'environment': renv.name, 'experiment': cfg['evaluation']['retention']['experiment'], 'trained': str(trained),
                  'examples': len(rows['base']), 'test': stats.get('test'), 'flips': stats.get('flips')}
    except Exception as error:
        traceback.print_exc()
        result = {'environment': None, 'error': f'{type(error).__name__}: {error}'[:400]}
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2, default=str))
    try:
        t = result.get('test') or {}
        if t and os.environ.get('WANDB_API_KEY'):
            import wandb
            run = wandb.init(project=cfg['tracking']['project'], entity=cfg['tracking']['entity'], id=runtime['wandb_run_id'], resume='allow', reinit=True)
            name = result['environment']
            scalars = {f'retention/{name}/base': t.get('baseline_accuracy'), f'retention/{name}/trained': t.get('candidate_accuracy'),
                       f'retention/{name}/delta': t.get('accuracy_delta'), f'retention/{name}/ci_lower': t.get('ci_lower'),
                       f'retention/{name}/ci_upper': t.get('ci_upper'), f'retention/{name}/examples': result['examples']}
            run.summary.update({k: v for k, v in scalars.items() if v is not None})
            run.finish()
    except Exception as error:
        print(f'retention W&B logging skipped: {type(error).__name__}: {error}', flush=True)
    return result


def in_loop_finish(cfg, runtime, env_spec, root, steps):
    """final_eval: false environments (mimo): base = step-0 in-loop eval, trained = last in-loop eval (no checkpoint selection).
    Writes artifacts/final_eval/*, prints a one-screen table, logs test/* to the training W&B run. Never raises: a failure is recorded
    as ``paired: null`` with the reason (the run's training evidence is already complete)."""
    from .inloop import in_loop_result, summary_markdown, wandb_scalars
    eval_dir = Path('artifacts/final_eval')
    try:
        result = in_loop_result(cfg, env_spec, root / 'exports/evaluation_samples.jsonl', eval_dir, optimizer_steps=steps)
    except Exception as error:
        traceback.print_exc()
        return {'paired': None, 'reason': f'in-loop result failed: {type(error).__name__}: {error}'[:300], 'selected_step': None}
    print(summary_markdown(result), flush=True)
    try:
        scalars = wandb_scalars(result)
        if scalars and os.environ.get('WANDB_API_KEY'):
            import wandb
            run = wandb.init(project=cfg['tracking']['project'], entity=cfg['tracking']['entity'], id=runtime['wandb_run_id'], resume='allow', reinit=True)
            run.log({'optimizer_step': steps, **scalars})
            run.summary.update({**scalars, 'test/trained_step': result['trained_step'], 'test/selection': result['selection']})
            from .inloop import FLIP_COLUMNS
            from .finalize import read_jsonl
            for kind in ('fixed', 'broken'):   # the qualitative "what changed" tables, as the offline path logs them
                path = eval_dir / f'{kind}_by_training.jsonl'
                if path.is_file():
                    table = wandb.Table(columns=list(FLIP_COLUMNS))
                    for row in read_jsonl(path):
                        table.add_data(*[row.get(c) for c in FLIP_COLUMNS])
                    run.log({f'qualitative/{kind}_by_training': table})
            artifact = wandb.Artifact(runtime['run_name'] + '-evidence', type='evaluation')
            for name in ('result.json', 'README.md', 'base.jsonl', 'candidate.jsonl', 'fixed_by_training.jsonl', 'broken_by_training.jsonl'):
                if (eval_dir / name).is_file():
                    artifact.add_file(str(eval_dir / name))
            run.log_artifact(artifact)
            run.finish()
    except Exception as error:   # the S3 copy is the source of truth; W&B is a view
        print(f'in-loop W&B logging skipped: {type(error).__name__}: {error}', flush=True)
    return result


def run_probe(nodes, gpus, cpu, deadline):
    """Cluster probe in a child process (bounded; CUDA/NCCL state never leaks into the worker). Returns the network summary."""
    child([sys.executable, '-m', 'strl.probe', '--nodes', str(nodes), '--gpus-per-node', str(gpus), *(['--cpu'] if cpu else [])],
          'artifacts/probe.log', deadline)
    network = json.loads(Path('artifacts/network.json').read_text())
    return {k: network.get(k) for k in ('passed', 'mode', 'nodes', 'hostnames', 'transport', 'efa_available', 'bandwidth_gbps', 'warnings')}


def cluster_shards(root, world_size):
    """Each node lists the checkpoint shards it holds (node-pinned Ray tasks, run before Ray stops); nothing is copied."""
    step_file = root / 'checkpoints/latest_ckpt_global_step.txt'
    if not step_file.is_file():
        return None
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from .verify import local_shards
    task = ray.remote(num_cpus=0)(local_shards)
    step = int(step_file.read_text())
    nodes = [n for n in ray.nodes() if n['Alive']]
    return ray.get([task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n['NodeID'], soft=False)).remote(
        str(root.resolve()), step, world_size) for n in nodes], timeout=300)


def peer(prefix, rank, begin, identity):
    """Peers keep their Ray node (and its GPUs) in the head's cluster until the head finishes or goes silent."""
    from . import multinode
    final = multinode.peer_wait(prefix, rank, begin['deadline'], identity={'hostname': identity['hostname']})
    put_json(f'{prefix}/workers/{rank}/finished.json', dict(identity, finished_at=time.time(), head_status=final.get('status')))
    print(json.dumps({'peer': rank, 'head_status': final.get('status')}), flush=True)


def outcome_status(error: BaseException) -> str:
    """'preempted' when the pod was told to stop (SIGTERM: Ray's handler turns it into SystemExit(15)) - capacity borrowed from another team
    is reclaimed with no warning (g4-mimo-a: 70 s after it started) - else 'failed'. A preempted attempt cannot resume (no optimizer-state
    checkpoints by design, the budget window is anchored at the first start): restage it as a new attempt."""
    return 'preempted' if isinstance(error, SystemExit) and error.code == 15 else 'failed'


def main():
    cfg = current()
    runtime = json.loads(Path('runtime.json').read_text())
    mode = os.environ.get('STRL_MODE', 'dry' if os.environ.get('GRPO_DRY_RUN') else 'train')
    env_spec = get_environment(cfg['environment']['name'])
    from . import models
    profile = models.detect(cfg['model'])[0]
    prefix = runtime['artifact_uri']
    os.environ['GRPO_ARTIFACT_URI'] = prefix
    Path('artifacts').mkdir(exist_ok=True)
    gpus = cfg['compute']['gpus_per_node']
    nodes = int(cfg['compute']['nodes'])
    multi, rank = nodes > 1, int(os.environ.get('JOB_COMPLETION_INDEX', '0'))
    identity = {'hostname': socket.gethostname(), 'started_at': time.time(), 'model': cfg['model']['id'], 'model_revision': cfg['model']['revision'],
                'experiment': cfg['name'], 'experiment_sha': cfg['_meta']['sha'], 'environment': env_spec.name, 'mode': mode,
                'profile': profile.name}
    if multi:
        identity.update(rank=rank, nodes=nodes)
    phases = {'worker_start': time.time()}
    heartbeat = watcher = None
    try:
        if int(os.environ.get('NUM_INSTANCES', '1')) != nodes or not 0 <= rank < nodes:
            raise RuntimeError('Incorrect worker allocation: expected exactly ' + ('one node' if nodes == 1 else f'{nodes} nodes'))
        begin = budget(runtime, cfg)
        identity['budget'] = begin
        reserve = train_reserve_seconds(cfg, mode)
        os.environ.update(
            GRPO_WORKER_BEGAN_EPOCH=str(begin['started_at']), STRL_GPU_FRACTION=str(begin['gpu_fraction']),
            GRPO_TRAIN_DEADLINE=str(begin['deadline'] - reserve), GRPO_FINAL_EVAL_RESERVE_SECONDS=str(reserve),
            GRPO_SAMPLES_PER_PROMPT=str(cfg['algorithm']['n_samples']), GRPO_NODES=str(cfg['compute']['nodes']),
            GRPO_EXPECTED_VALIDATION_SIZE=str(env_spec.validation_size(cfg) if mode != 'canary' else env_spec.canary_size(cfg)),
            GRPO_RUN_NAME=runtime['run_name'], GRPO_CHECKPOINT_MINUTES='100000',
            GRPO_FIXED_QUALITATIVE=str(cfg['evaluation']['fixed_qualitative_samples']),
            GRPO_CANARY_REQUIRE_GRADIENT='1' if env_spec.canary_requires_gradient(cfg) else '0',
            STRL_RESOLVED=str(Path('resolved.json').resolve()), STRL_EXPERIMENT_NAME=cfg['name'], STRL_EXPERIMENT_SHA=cfg['_meta']['sha'],
            STRL_ENVIRONMENT=env_spec.name, STRL_STAGE='canary' if mode == 'canary' else 'train',
            STRL_ALERTS='1' if cfg['tracking'].get('alerts', True) else '0', STRL_TOKENIZER=cfg['model']['local_path'],
            STRL_METRICS_JSONL=str((Path('artifacts') / runtime['run_name'] / 'metrics.jsonl').resolve()))
        from .tracking import failfast_config   # budget.failfast (+ the environment's opt-in defaults); renders nothing when empty
        rules = failfast_config(cfg, env_spec)
        if rules:
            os.environ['STRL_FAILFAST'] = json.dumps(rules)
        if runtime.get('plan_step_seconds'):
            os.environ['STRL_PLAN_STEP_SECONDS'] = str(runtime['plan_step_seconds'])
        interface = os.environ['NETWORK_INTERFACE_NAME']
        port = os.environ.get('LEADER_PORT', '6379')
        if multi:
            from . import multinode
            local_ip, leader_ip, address = multinode.shared_job_network(rank, interface, port, prefix, min(time.time() + 600, begin['deadline']))
            identity['leader_ip'] = leader_ip
        else:
            local_ip, address = head_network(interface, port)
        os.environ['NCCL_SOCKET_IFNAME'] = '=' + interface
        os.environ['GLOO_SOCKET_IFNAME'] = interface
        os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET6' if ':' in local_ip else 'AF_INET'
        os.environ['RAY_ADDRESS'] = address
        cpus = allocated_cpus()
        identity.update(node_ip=local_ip, interface=interface, ray_address=address, cpus=cpus, gpus=gpus)
        put_json(prefix + ('/worker-started.json' if rank == 0 else f'/workers/{rank}/started.json'), identity)
        if multi and rank == 0:
            heartbeat = multinode.Heartbeat(f'{prefix}/{multinode.HEAD_HEARTBEAT}', {'hostname': identity['hostname']})
            heartbeat.start()
        print(json.dumps({'network': {k: identity[k] for k in ('node_ip', 'interface', 'ray_address', 'cpus', 'gpus')}}), flush=True)
        identity['services'] = setup_services(cfg, runtime, env_spec, mode)
        from .runtime import SKYRL_ROOT, patch_scaling, preflight
        Path('artifacts/runtime.json').write_text(json.dumps(preflight(env_spec), indent=2))
        env_spec.assert_manifest(cfg)
        env_spec.patch_runtime(SKYRL_ROOT, cfg)
        if subprocess.check_output(['git', '-C', SKYRL_ROOT, 'rev-parse', 'HEAD'], text=True).strip() != cfg['runtime']['skyrl_commit']:
            raise RuntimeError('SkyRL image revision mismatch')
        patches = patch_scaling(SKYRL_ROOT, cfg)   # every node: Ray workers import SkyRL from their own node's tree
        if patches:
            identity['skyrl_patches'] = patches
        if multi:
            from .probe import configure_efa
            identity['efa'] = configure_efa()      # FI_PROVIDER=efa when usable; inherited by this node's Ray workers
            identity['ray_join_tries'] = multinode.start_ray(rank, address, port, local_ip, cpus, gpus, min(time.time() + 600, begin['deadline']))
            if rank:
                return peer(prefix, rank, begin, identity)
        else:
            subprocess.run([RAY, 'stop', '--force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(ray_start_args(0, address, port, local_ip, cpus, gpus), check=True, timeout=120)
        import ray
        ray.init(address=address)
        if multi:
            identity['cluster'] = multinode.wait_for_cluster(nodes, gpus, max(60, min(900, begin['deadline'] - time.time() - 600)))
        else:
            until = time.time() + 120
            while time.time() < until:
                if len([n for n in ray.nodes() if n['Alive']]) == 1 and int(ray.cluster_resources().get('GPU', 0)) == gpus:
                    break
                time.sleep(3)
            else:
                raise RuntimeError('Ray did not expose the configured GPU count')
        ray.shutdown()
        phases['ray_ready'] = time.time()
        if multi:
            identity['network'] = run_probe(nodes, gpus, mode == 'dry', min(time.time() + 900, begin['deadline'] - 300))
            phases['probe_done'] = time.time()
            print(json.dumps({'probe': identity['network']}), flush=True)
        if mode == 'dry':  # CPU rehearsal of every gate the GPU run depends on; never starts training
            gates = dry_run_gates(cfg, runtime, env_spec, identity['services'])
            per_domain = int((cfg.get('run') or {}).get('ceiling_per_domain') or 0)
            if per_domain and env_spec.name == 'mimo':   # the reference agent's ceiling, run where the LiteLLM key already is (never the laptop)
                from .evals import mimo as ceiling
                try:
                    gates['ceiling'] = ceiling(cfg, 'test', per_domain, (cfg.get('run') or {}).get('ceiling_model', 'openai/gpt-6-luna'), 8192,
                                               ((cfg.get('services') or {}).get('litellm') or {}).get('project_id', '696011502a26dc4a64b24f55'))
                except Exception as error:
                    gates['ceiling'] = {'error': f'{type(error).__name__}: {error}'[:400]}
            dry = dict(identity, status='dry_run_passed', phases=phases, gates=gates, train_deadline=float(os.environ['GRPO_TRAIN_DEADLINE']),
                       skyrl_commit=cfg['runtime']['skyrl_commit'])
            put_json(prefix + '/dry-run.json', dry)
            put_json(prefix + '/completion.json', {'status': 'dry_run_passed', 'experiment_sha': cfg['_meta']['sha'],
                                                   **({'network': identity['network']} if multi else {})})
            print(json.dumps(dry, default=str), flush=True)
            return
        root = Path('artifacts') / runtime['run_name']
        if (cfg.get('run') or {}).get('resume_from'):
            identity['resumed_from'] = prepare_resume(prefix, runtime['run_name'], cfg['run']['resume_from'], root)
            put_json(prefix + '/worker-started.json', identity)
        stalled = None
        if multi:   # head-side liveness: a lost peer aborts training instead of hanging in NCCL until the deadline
            ray.init(address=address)
            watcher = multinode.Watcher(multinode.cluster_check(nodes))
            watcher.start()
        try:
            child([sys.executable, '-m', 'strl.train'], 'artifacts/training.log', begin['deadline'] - (CHILD_HARD_STOP_SECONDS if mode == 'train' else 60),
                  root / 'canary.json', cfg['budget']['canary_max_seconds'], progress=root / 'progress.json',
                  stall_seconds=int(cfg['budget']['stall_seconds']), abort=watcher)
        except Stalled as error:
            stalled = str(error)
            _resume_alert(cfg, runtime, 'Training stalled', stalled)
        phases['training_end'] = time.time()
        shard_reports = None
        if multi:
            watcher.stop()
            shard_reports = cluster_shards(root, nodes * gpus) if save_optimizer_state(cfg) else None
            identity['exports_gathered'] = gather_exports(root, prefix)
            ray.shutdown()
        subprocess.run([RAY, 'stop', '--force'], check=False)
        progress = json.loads((root / 'training_completed.json').read_text()) if (root / 'training_completed.json').is_file() else \
            json.loads((root / 'progress.json').read_text()) if (root / 'progress.json').is_file() else {}
        steps = int(progress.get('optimizer_steps', progress.get('optimizer_step', 0)))
        if steps < 2 or not (root / 'canary.json').is_file():
            raise RuntimeError('Missing successful real training canary' + (f' ({stalled})' if stalled else ''))
        extra = {'nodes': nodes, 'network': identity.get('network')} if multi else {}
        if mode == 'canary' or not cfg['evaluation']['final_eval']:
            final = dict(status='completed', mode=mode, optimizer_steps=steps, stalled=stalled, phases=phases,
                         elapsed_seconds=time.time() - begin['started_at'], gpus=gpus, experiment_sha=cfg['_meta']['sha'], **extra)
            if mode == 'train':   # no offline final eval: pair the first (step-0) and last in-loop evals
                final['paired_result'] = in_loop_finish(cfg, runtime, env_spec, root, steps)
                final['selected_step'] = final['paired_result'].get('selected_step')
                if cfg['evaluation'].get('retention'):
                    final['retention'] = run_retention(cfg, runtime, root, latest_export(root), begin['deadline'] - 60)
            sync_local('artifacts', prefix, exclude=SYNC_EXCLUDE)
            put_json(prefix + '/completion.json', final)
            print(json.dumps(final), flush=True)
            return
        # Select using validation only; the held-out test split is used once per model.
        candidates = {}
        for line in (root / 'exports/evaluation_samples.jsonl').read_text().splitlines():
            row = json.loads(line)
            candidates.setdefault(row['step'], []).append(row['correct'])
        validation = {s: sum(v) / len(v) for s, v in sorted(candidates.items())}
        available = [(a, s, root / 'exports' / f'global_step_{s}' / 'policy') for s, a in validation.items()
                     if s > 0 and (root / 'exports' / f'global_step_{s}' / 'policy').is_dir()]
        if not available:
            raise RuntimeError('No validation-selected checkpoint export')
        accuracy, selected, export = max(available, key=lambda item: (item[0], item[1]))
        from .lora import is_adapter, keep_adapter, merge_adapter
        if is_adapter(export):   # LoRA: keep the adapter (+ its base), evaluate a merged copy with the unchanged evaluator
            extra['adapter'] = keep_adapter(export, root / 'reload' / f'global_step_{selected}', cfg['model']['id'], cfg['model']['revision'],
                                            cfg['model']['local_path'])
            change = verify_adapter_change(export)
            # evaluate the ADAPTER (served by vLLM on the bf16 base, as in training): a bf16 merge rounds a small RL update away
            change['served'] = 'vllm-lora'
        else:
            export = profile.prepare_export_for_vllm(cfg['model']['local_path'], export, root / 'reload' / f'global_step_{selected}')
            change = None
        if not save_optimizer_state(cfg):
            shards = {'skipped': 'budget.save_optimizer_state=false: no FSDP checkpoint (optimizer state) was written'}
        elif multi:
            if shard_reports is None:
                raise RuntimeError('Missing distributed checkpoint (no latest_ckpt_global_step.txt on the head)')
            shards = verify_cluster_shards(shard_reports, nodes * gpus)
        else:
            shard_step = int((root / 'checkpoints/latest_ckpt_global_step.txt').read_text())
            shards = verify_shards(root, shard_step, gpus)
        if change is None:
            change = verify_export_change(cfg['model']['local_path'], export)
        eval_dir = Path('artifacts/final_eval')
        eval_dir.mkdir(exist_ok=True)
        phases['eval_start'] = time.time()
        rows = evaluate_test_split(cfg, env_spec, cfg['model']['local_path'], export, eval_dir, begin['deadline'] - 90)
        phases['eval_end'] = time.time()
        from .finalize import flips, select_examples, write_jsonl
        base, trained = rows['base'], rows['candidate']
        write_jsonl(eval_dir / 'base.jsonl', base)
        write_jsonl(eval_dir / 'candidate.jsonl', trained)
        from .stats import paired_bootstrap
        result = paired_bootstrap([r['correct'] for r in base], [r['correct'] for r in trained], samples=cfg['evaluation']['bootstrap_samples'],
                                  seed=cfg['evaluation']['bootstrap_seed'], level=cfg['evaluation']['ci_level'])
        counts = flips(base, trained)
        fixed = select_examples(base, trained, 'fixed', mode_fn=env_spec.failure_mode)
        broken = select_examples(base, trained, 'broken', 20, mode_fn=env_spec.failure_mode)
        write_jsonl(eval_dir / 'fixed_by_training.jsonl', fixed)
        write_jsonl(eval_dir / 'broken_by_training.jsonl', broken)
        by_source = env_spec.metric_by_source(base, trained) if hasattr(env_spec, 'metric_by_source') else {}
        phases['finished'] = time.time()
        (root / 'phases.json').write_text(json.dumps(phases, indent=2))
        final = dict(status='completed', mode=mode, optimizer_steps=steps, stalled=stalled, selected_step=selected, validation_accuracy=accuracy,
                     validation_by_step=validation, checkpoint=shards, parameter_change=change, test=result, flips=counts, by_source=by_source,
                     phases=phases, elapsed_seconds=time.time() - begin['started_at'], gpus=gpus, experiment=cfg['name'],
                     experiment_sha=cfg['_meta']['sha'], environment=env_spec.name, metric=env_spec.metric, model=cfg['model']['id'], **extra)
        (eval_dir / 'result.json').write_text(json.dumps(final, indent=2))
        import wandb
        try:   # the evaluation is complete and on disk: a W&B outage must not turn a finished attempt into a failed one
            run = wandb.init(project=cfg['tracking']['project'], entity=cfg['tracking']['entity'], id=runtime['wandb_run_id'], resume='must')
            run.log({'optimizer_step': steps, **{'test/' + k: v for k, v in result.items()}, **{'test/flips_' + k: v for k, v in counts.items()},
                     **{f'test/{src}/{k}': v for src, d in by_source.items() for k, v in d.items()},
                     'checkpoint/reload_passed': True, 'distributed/nodes': cfg['compute']['nodes'], 'distributed/gpus': gpus})
            cols = ['sample_id', 'prompt', 'ground_truth', 'base_failure_mode', 'base_completion', 'trained_completion']
            table = wandb.Table(columns=cols)
            for r in fixed:
                table.add_data(r['sample_id'], str(r['prompt']), str(r['ground_truth']), r['base_failure_mode'], r['base_completion'], r['trained_completion'])
            run.log({'qualitative/fixed_by_training': table})
            artifact = wandb.Artifact(runtime['run_name'] + '-evidence', type='evaluation')
            for name in ['result.json', 'base.jsonl', 'candidate.jsonl', 'fixed_by_training.jsonl', 'broken_by_training.jsonl']:
                artifact.add_file(str(eval_dir / name))
            run.log_artifact(artifact)
            run.summary.update({k: v for k, v in final.items() if k != 'validation_by_step'})
            run.finish()
        except Exception as error:
            final['wandb_error'] = f'{type(error).__name__}: {error}'
            print(f'[strl] W&B logging of the final evaluation failed (results are in S3): {final["wandb_error"]}', file=sys.stderr, flush=True)
        sync_local('artifacts', prefix, exclude=SYNC_EXCLUDE)
        put_json(prefix + '/completion.json', final)
        print(json.dumps(final), flush=True)
    except BaseException as error:
        failed = dict(identity, status=outcome_status(error), error_type=type(error).__name__, error=str(error), failed_at=time.time(), phases=phases)
        if rank:   # a peer never writes completion.json (the head owns the attempt's outcome) and never alerts
            Path(f'artifacts/failure-{rank}.log').write_text(traceback.format_exc())
            try:
                put_json(f'{prefix}/workers/{rank}/failed.json', failed)
            except Exception:
                pass
            raise
        Path('artifacts/failure.log').write_text(traceback.format_exc())
        _resume_alert(cfg, runtime, f'Run {failed["status"]}: {type(error).__name__}', str(error))
        try:
            sync_local('artifacts', prefix, exclude=SYNC_EXCLUDE)
            put_json(prefix + '/completion.json', failed)
        except Exception:
            pass
        raise
    finally:
        for thread in (watcher, heartbeat):
            if thread is not None:
                thread.stop()
        subprocess.run([RAY, 'stop', '--force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.environ.pop('WANDB_API_KEY', None)
        for secret_file in ('.modal-runtime.toml', '.litellm-key'):
            Path(secret_file).unlink(missing_ok=True)


def _resume_alert(cfg, runtime, title, text):
    """Alert from the worker process (the training process may be dead): attach to the run and send a W&B alert."""
    if not cfg['tracking'].get('alerts', True) or not os.environ.get('WANDB_API_KEY'):
        return
    try:
        import wandb
        run = wandb.init(project=cfg['tracking']['project'], entity=cfg['tracking']['entity'], id=runtime['wandb_run_id'], resume='allow',
                         reinit=True)
        run.alert(title=title[:64], text=f'{runtime["run_name"]}: {text}'[:1000], level=wandb.AlertLevel.ERROR)
        run.finish()
    except Exception:
        pass


if __name__ == '__main__':
    main()
