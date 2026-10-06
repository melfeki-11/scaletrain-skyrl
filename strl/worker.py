"""Single-node ScaleTrain worker for any experiment: budget -> services -> Ray -> preflight -> train -> select -> final eval.

Phases and receipts (all synced to ``<artifact_uri>``):
  budget.json (write-once) -> worker-started.json -> artifacts/runtime.json (preflight) -> [dry-run.json] -> training.log,
  heartbeat.json (every 60 s), <run>/progress.json + metrics.jsonl -> final_eval/* -> completion.json (completed | failed)
Modes (STRL_MODE): dry (CPU: every gate except GPU training), canary (2 updates, no final eval), train.
Watchdog: if training makes no progress for budget.stall_seconds, the child is killed, an alert is sent, and the latest
export (if any) is still evaluated, so a stalled run is salvaged instead of silently burning its allocation.
"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

import boto3
from botocore.exceptions import ClientError

from .experiment import current
from .network import head_network, ray_start_args
from .resources import allocated_cpus
from .storage import get_json, put_json, split_uri, sync_local
from .tasks import get_environment
from .verify import verify_export_change, verify_shards

RAY = '/opt/skyrl-venv/bin/ray'
SYNC_EXCLUDE = ('checkpoints/',)  # optimizer shards stay on the node; exports/metrics/evals go to S3


class Stalled(RuntimeError):
    pass


def _progress_mtime(progress: Path | None) -> float | None:
    return progress.stat().st_mtime if progress is not None and progress.is_file() else None


def child(args, log, deadline, canary_path=None, canary_seconds=1200, env=None, progress=None, stall_seconds=None):
    started = time.time()
    with Path(log).open('w') as output:
        process = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        last_sync = 0
        try:
            while process.poll() is None:
                now = time.time()
                if now >= deadline:
                    raise TimeoutError('Phase exceeded authorized time budget')
                if canary_path is not None and now >= started + canary_seconds and not canary_path.exists():
                    raise TimeoutError('Canary did not verify two updates in time')
                seen = _progress_mtime(progress)
                if stall_seconds and seen is not None and now - seen > stall_seconds:
                    raise Stalled(f'No training progress for {now - seen:.0f}s (stall_seconds={stall_seconds})')
                if now - last_sync >= 60:
                    Path('artifacts/heartbeat.json').write_text(json.dumps({
                        'at': now, 'child_alive': True, 'elapsed_seconds': now - started, 'last_progress_at': seen,
                        'log_bytes': Path(log).stat().st_size if Path(log).exists() else 0}))
                    sync_local('artifacts', os.environ['GRPO_ARTIFACT_URI'], exclude=SYNC_EXCLUDE)
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
    if critic.get('enabled'):
        from .services import litellm
        try:
            key_file = litellm.write_key_file(Path('artifacts').resolve().parent / '.litellm-key',
                                              (services.get('litellm') or {}).get('key_name', litellm.KEY_NAME))
            os.environ['STRL_LITELLM_KEY_FILE'] = str(key_file)
            os.environ['STRL_CRITIC'] = json.dumps(critic)
            receipt['critic'] = 'enabled'
        except Exception as error:  # the critic is optional: never fail a training run because of it (dry runs report it)
            receipt['critic'] = f'disabled: {type(error).__name__}: {error}'[:200]
            if mode == 'dry':
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


def evaluate_test_split(cfg, env_spec, base_model, trained_model, eval_dir, deadline):
    """Base on GPUs [0, n), trained on GPUs [n, 2n), n shards each, concurrently; reassemble in canonical id order."""
    from .finalize import merge_shards, read_jsonl
    n = cfg['evaluation']['eval_shards_per_model']
    jobs = []
    for offset, (label, model) in enumerate([('base', base_model), ('candidate', trained_model)]):
        for shard in range(n):
            env = os.environ.copy()
            env.pop('RAY_ADDRESS', None)
            env['CUDA_VISIBLE_DEVICES'] = str(offset * n + shard)
            args = [sys.executable, '-m', env_spec.eval_module, '--model', str(model), '--output', str(eval_dir / f'{label}-{shard}.jsonl'),
                    '--rank', str(shard), '--world-size', str(n)]
            jobs.append((args, eval_dir / f'{label}-{shard}.log', env))
    run_parallel(jobs, deadline)
    ids = env_spec.final_eval_ids(cfg)
    return {label: merge_shards([read_jsonl(eval_dir / f'{label}-{s}.jsonl') for s in range(n)], ids) for label in ('base', 'candidate')}


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
    identity = {'hostname': socket.gethostname(), 'started_at': time.time(), 'model': cfg['model']['id'], 'model_revision': cfg['model']['revision'],
                'experiment': cfg['name'], 'experiment_sha': cfg['_meta']['sha'], 'environment': env_spec.name, 'mode': mode,
                'profile': profile.name}
    phases = {'worker_start': time.time()}
    try:
        if int(os.environ.get('NUM_INSTANCES', '1')) != 1 or int(os.environ.get('JOB_COMPLETION_INDEX', '0')) != 0:
            raise RuntimeError('Incorrect worker allocation: expected exactly one node')
        begin = budget(runtime, cfg)
        identity['budget'] = begin
        reserve = int(cfg['budget']['final_eval_reserve_seconds']) if mode == 'train' and cfg['evaluation']['final_eval'] else 120
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
        interface = os.environ['NETWORK_INTERFACE_NAME']
        port = os.environ.get('LEADER_PORT', '6379')
        local_ip, address = head_network(interface, port)
        os.environ['NCCL_SOCKET_IFNAME'] = '=' + interface
        os.environ['GLOO_SOCKET_IFNAME'] = interface
        os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET6' if ':' in local_ip else 'AF_INET'
        os.environ['RAY_ADDRESS'] = address
        cpus = allocated_cpus()
        identity.update(node_ip=local_ip, interface=interface, ray_address=address, cpus=cpus, gpus=gpus)
        put_json(prefix + '/worker-started.json', identity)
        print(json.dumps({'network': {k: identity[k] for k in ('node_ip', 'interface', 'ray_address', 'cpus', 'gpus')}}), flush=True)
        identity['services'] = setup_services(cfg, runtime, env_spec, mode)
        from .runtime import SKYRL_ROOT, preflight
        Path('artifacts/runtime.json').write_text(json.dumps(preflight(env_spec), indent=2))
        env_spec.assert_manifest(cfg)
        env_spec.patch_runtime(SKYRL_ROOT, cfg)
        if subprocess.check_output(['git', '-C', SKYRL_ROOT, 'rev-parse', 'HEAD'], text=True).strip() != cfg['runtime']['skyrl_commit']:
            raise RuntimeError('SkyRL image revision mismatch')
        subprocess.run([RAY, 'stop', '--force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(ray_start_args(0, address, port, local_ip, cpus, gpus), check=True, timeout=120)
        import ray
        ray.init(address=address)
        until = time.time() + 120
        while time.time() < until:
            if len([n for n in ray.nodes() if n['Alive']]) == 1 and int(ray.cluster_resources().get('GPU', 0)) == gpus:
                break
            time.sleep(3)
        else:
            raise RuntimeError('Ray did not expose the configured GPU count')
        ray.shutdown()
        phases['ray_ready'] = time.time()
        if mode == 'dry':  # CPU rehearsal of every gate the GPU run depends on; never starts training
            gates = dry_run_gates(cfg, runtime, env_spec, identity['services'])
            dry = dict(identity, status='dry_run_passed', phases=phases, gates=gates, train_deadline=float(os.environ['GRPO_TRAIN_DEADLINE']),
                       skyrl_commit=cfg['runtime']['skyrl_commit'])
            put_json(prefix + '/dry-run.json', dry)
            put_json(prefix + '/completion.json', {'status': 'dry_run_passed', 'experiment_sha': cfg['_meta']['sha']})
            print(json.dumps(dry, default=str), flush=True)
            return
        root = Path('artifacts') / runtime['run_name']
        stalled = None
        try:
            child([sys.executable, '-m', 'strl.train'], 'artifacts/training.log', begin['deadline'] - (480 if mode == 'train' else 60),
                  root / 'canary.json', cfg['budget']['canary_max_seconds'], progress=root / 'progress.json',
                  stall_seconds=int(cfg['budget']['stall_seconds']))
        except Stalled as error:
            stalled = str(error)
            _resume_alert(cfg, runtime, 'Training stalled', stalled)
        phases['training_end'] = time.time()
        subprocess.run([RAY, 'stop', '--force'], check=False)
        progress = json.loads((root / 'training_completed.json').read_text()) if (root / 'training_completed.json').is_file() else \
            json.loads((root / 'progress.json').read_text()) if (root / 'progress.json').is_file() else {}
        steps = int(progress.get('optimizer_steps', progress.get('optimizer_step', 0)))
        if steps < 2 or not (root / 'canary.json').is_file():
            raise RuntimeError('Missing successful real training canary' + (f' ({stalled})' if stalled else ''))
        if mode == 'canary' or not cfg['evaluation']['final_eval']:
            final = dict(status='completed', mode=mode, optimizer_steps=steps, stalled=stalled, phases=phases,
                         elapsed_seconds=time.time() - begin['started_at'], gpus=gpus, experiment_sha=cfg['_meta']['sha'])
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
        export = profile.prepare_export_for_vllm(cfg['model']['local_path'], export, root / 'reload' / f'global_step_{selected}')
        shard_step = int((root / 'checkpoints/latest_ckpt_global_step.txt').read_text())
        shards = verify_shards(root, shard_step, gpus)
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
                     experiment_sha=cfg['_meta']['sha'], environment=env_spec.name, metric=env_spec.metric, model=cfg['model']['id'])
        (eval_dir / 'result.json').write_text(json.dumps(final, indent=2))
        import wandb
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
        sync_local('artifacts', prefix, exclude=SYNC_EXCLUDE)
        put_json(prefix + '/completion.json', final)
        print(json.dumps(final), flush=True)
    except BaseException as error:
        failed = dict(identity, status='failed', error_type=type(error).__name__, error=str(error), failed_at=time.time(), phases=phases)
        Path('artifacts/failure.log').write_text(traceback.format_exc())
        _resume_alert(cfg, runtime, f'Run failed: {type(error).__name__}', str(error))
        try:
            sync_local('artifacts', prefix, exclude=SYNC_EXCLUDE)
            put_json(prefix + '/completion.json', failed)
        except Exception:
            pass
        raise
    finally:
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
