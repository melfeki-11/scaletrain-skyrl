"""Single-node (8 x H100) GRPO worker: train, select by validation, evaluate base vs trained on official test."""
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
from .config import load
from .storage import get_json, put_json, sync_local, split_uri
from .verify import verify_export_change, verify_shards
from .network import head_network, ray_start_args
from .resources import allocated_cpus

RAY = '/opt/skyrl-venv/bin/ray'
SYNC_EXCLUDE = ('checkpoints/',)  # optimizer shards stay on the node; exports/metrics/evals go to S3


def child(args, log, deadline, canary_path=None, canary_seconds=1200, env=None):
    started = time.time()
    with Path(log).open('w') as output:
        process = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        last = 0
        try:
            while process.poll() is None:
                now = time.time()
                if now >= deadline: raise TimeoutError('Phase exceeded authorized time budget')
                if canary_path is not None and now >= started + canary_seconds and not canary_path.exists():
                    raise TimeoutError('Canary did not verify two updates in time')
                if now - last >= 60:
                    sync_local('artifacts', os.environ['GRPO_ARTIFACT_URI'], exclude=SYNC_EXCLUDE); last = now
                time.sleep(5)
            if process.returncode: raise RuntimeError(f'Worker phase failed with exit {process.returncode}; see {log}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()


def run_parallel(jobs, deadline):
    """jobs: [(args, log, env)]. All run concurrently; any failure or the deadline kills the rest."""
    procs = []
    try:
        for args, log, env in jobs:
            procs.append((subprocess.Popen(args, stdout=Path(log).open('w'), stderr=subprocess.STDOUT, start_new_session=True, env=env), log))
        while any(p.poll() is None for p, _ in procs):
            if time.time() >= deadline: raise TimeoutError('Parallel evaluation exceeded the budget')
            if any(p.poll() not in (None, 0) for p, _ in procs): break
            time.sleep(3)
        bad = [log for p, log in procs if p.poll() not in (0,)]
        if bad: raise RuntimeError('Evaluation shard failed: ' + ', '.join(map(str, bad)))
    finally:
        for p, _ in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)


def budget(runtime):
    prefix = runtime['artifact_uri']
    s3 = boto3.client('s3'); bucket, key = split_uri(prefix + '/budget.json')
    now = time.time()
    seconds = int(runtime['remaining_seconds']); total = int(runtime['total_seconds'])
    if not 0 < seconds <= total <= 3600: raise ValueError('Invalid remaining allocation budget')
    payload = {'started_at': now, 'deadline': now + seconds, 'nodes': 1, 'seconds': seconds,
               'consumed_previous_seconds': total - seconds, 'attempt': runtime['run_name'],
               'ledger_uri': runtime['budget_ledger_uri']}
    try: s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode(), ServerSideEncryption='AES256', IfNoneMatch='*')
    except ClientError as e:
        if e.response['ResponseMetadata']['HTTPStatusCode'] not in {409, 412}: raise
    result = get_json(prefix + '/budget.json')
    if result['attempt'] != runtime['run_name'] or result['seconds'] != seconds:
        raise RuntimeError('Attempt budget identity mismatch')
    return result


def setup_services(cfg, runtime):
    bucket, key = split_uri(cfg['wandb']['credentials_uri'])
    raw = boto3.client('s3').get_object(Bucket=bucket, Key=key)['Body'].read().decode()
    credentials = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    os.environ['WANDB_API_KEY'] = credentials['WANDB_API_KEY'].strip().strip('"\'')
    os.environ['WANDB_BASE_URL'] = cfg['wandb']['base_url']
    os.environ['WANDB_ENTITY'] = cfg['wandb']['entity']
    os.environ['WANDB_PROJECT'] = cfg['wandb']['project']
    os.environ['WANDB_RUN_GROUP'] = cfg['wandb']['group']
    os.environ['WANDB_RUN_ID'] = runtime['wandb_run_id']
    os.environ['WANDB_RESUME'] = 'allow'


def evaluate_test_split(cfg, base_model, trained_model, eval_dir, deadline):
    """Base on GPUs [0, n), trained on GPUs [n, 2n), n shards each, concurrently; reassemble and verify."""
    from .data import read_jsonl
    n = cfg['evaluation']['eval_shards_per_model']
    jobs = []
    for offset, (label, model) in enumerate([('base', base_model), ('candidate', trained_model)]):
        for shard in range(n):
            env = os.environ.copy(); env.pop('RAY_ADDRESS', None); env['CUDA_VISIBLE_DEVICES'] = str(offset * n + shard)
            args = [sys.executable, '-m', 'q35.evaluate', '--model', str(model), '--output', str(eval_dir / f'{label}-{shard}.jsonl'),
                    '--rank', str(shard), '--world-size', str(n)]
            jobs.append((args, eval_dir / f'{label}-{shard}.log', env))
    run_parallel(jobs, deadline)
    from .finalize import merge_shards
    ids = [f"gsm8k-test-{i:05d}" for i in range(cfg['evaluation']['test_size'])]
    return {label: merge_shards([read_jsonl(eval_dir / f'{label}-{s}.jsonl') for s in range(n)], ids) for label in ('base', 'candidate')}


def main():
    cfg = load(); runtime = json.loads(Path('runtime.json').read_text())
    prefix = runtime['artifact_uri']; os.environ['GRPO_ARTIFACT_URI'] = prefix
    Path('artifacts').mkdir(exist_ok=True)
    identity = {'hostname': socket.gethostname(), 'started_at': time.time(), 'model_revision': cfg['model']['revision']}
    phases = {'worker_start': time.time()}
    try:
        if int(os.environ.get('NUM_INSTANCES', '1')) != 1 or int(os.environ.get('JOB_COMPLETION_INDEX', '0')) != 0:
            raise RuntimeError('Incorrect worker allocation: expected exactly one node')
        begin = budget(runtime); identity['budget'] = begin
        reserve = int(cfg['budget']['final_eval_reserve_seconds'])
        os.environ.update(GRPO_WORKER_BEGAN_EPOCH=str(begin['started_at']), GRPO_PREVIOUS_SECONDS=str(begin['consumed_previous_seconds']),
                          GRPO_TRAIN_DEADLINE=str(begin['deadline'] - reserve), GRPO_FINAL_EVAL_RESERVE_SECONDS=str(reserve),
                          GRPO_SAMPLES_PER_PROMPT=str(cfg['training']['grpo']['samples_per_prompt']), GRPO_NODES='1',
                          GRPO_EXPECTED_VALIDATION_SIZE=str(cfg['dataset']['validation_size']), GRPO_RUN_NAME=runtime['run_name'],
                          GRPO_BASE_MODEL=str(Path(cfg['model']['local_path']).resolve()), GRPO_CHECKPOINT_MINUTES='100000')
        interface = os.environ['NETWORK_INTERFACE_NAME']; port = os.environ.get('LEADER_PORT', '6379')
        local_ip, address = head_network(interface, port)
        os.environ['NCCL_SOCKET_IFNAME'] = '=' + interface; os.environ['GLOO_SOCKET_IFNAME'] = interface
        os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET6' if ':' in local_ip else 'AF_INET'
        os.environ['RAY_ADDRESS'] = address
        cpus = allocated_cpus(); gpus = cfg['scale_train']['gpu_count']
        identity.update(node_ip=local_ip, interface=interface, ray_address=address, cpus=cpus, gpus=gpus)
        put_json(prefix + '/worker-started.json', identity)
        print(json.dumps({'network': {k: identity[k] for k in ('node_ip', 'interface', 'ray_address', 'cpus', 'gpus')}}), flush=True)
        setup_services(cfg, runtime)
        from .runtime import preflight
        Path('artifacts/runtime.json').write_text(json.dumps(preflight(), indent=2))
        from .data import assert_manifest
        assert_manifest(cfg)
        from .ipv6_patch import patch_reward_verifier
        patch_reward_verifier('/workspace/vendor/SkyRL')
        if subprocess.check_output(['git', '-C', '/workspace/vendor/SkyRL', 'rev-parse', 'HEAD'], text=True).strip() != cfg['runtime']['skyrl_commit']:
            raise RuntimeError('SkyRL image revision mismatch')
        subprocess.run([RAY, 'stop', '--force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(ray_start_args(0, address, port, local_ip, cpus, gpus), check=True, timeout=120)
        import ray
        ray.init(address=address)
        until = time.time() + 120
        while time.time() < until:
            if len([n for n in ray.nodes() if n['Alive']]) == 1 and int(ray.cluster_resources().get('GPU', 0)) == gpus: break
            time.sleep(3)
        else: raise RuntimeError('Ray did not expose the configured GPU count')
        ray.shutdown()
        phases['ray_ready'] = time.time()
        if os.environ.get('GRPO_DRY_RUN'):  # CPU rehearsal of bootstrap/budget/services/ray; never starts training
            dry = dict(identity, status='dry_run_passed', phases=phases, train_deadline=float(os.environ['GRPO_TRAIN_DEADLINE']),
                       skyrl_commit=cfg['runtime']['skyrl_commit'])
            put_json(prefix + '/dry-run.json', dry); print(json.dumps(dry), flush=True)
            return
        root = Path('artifacts') / runtime['run_name']
        child([sys.executable, '-m', 'q35.train'], 'artifacts/training.log', begin['deadline'] - 600,
              root / 'canary.json', cfg['budget']['canary_max_seconds'])
        phases['training_end'] = time.time()
        subprocess.run([RAY, 'stop', '--force'], check=True)
        progress = json.loads((root / 'training_completed.json').read_text())
        steps = int(progress['optimizer_steps'])
        if steps < 2 or not (root / 'canary.json').is_file(): raise RuntimeError('Missing successful real training canary')
        # Select using validation only; the official test split is used once per model.
        candidates = {}
        for line in (root / 'exports/evaluation_samples.jsonl').read_text().splitlines():
            row = json.loads(line); candidates.setdefault(row['step'], []).append(row['correct'])
        validation = {s: sum(v) / len(v) for s, v in sorted(candidates.items())}
        available = [(a, s, root / 'exports' / f'global_step_{s}' / 'policy') for s, a in validation.items()
                     if s > 0 and (root / 'exports' / f'global_step_{s}' / 'policy').is_dir()]
        if not available: raise RuntimeError('No validation-selected checkpoint export')
        accuracy, selected, export = max(available, key=lambda item: (item[0], item[1]))
        from .export_adapter import adapt
        if json.loads((export / 'config.json').read_text()).get('model_type') == 'qwen3_5_text':
            adapted = root / 'reload' / f'global_step_{selected}'
            adapt(cfg['model']['local_path'], export, adapted); export = adapted
        shard_step = int((root / 'checkpoints/latest_ckpt_global_step.txt').read_text())
        shards = verify_shards(root, shard_step, gpus)
        change = verify_export_change(cfg['model']['local_path'], export)
        eval_dir = Path('artifacts/final_eval'); eval_dir.mkdir(exist_ok=True)
        phases['eval_start'] = time.time()
        rows = evaluate_test_split(cfg, cfg['model']['local_path'], export, eval_dir, begin['deadline'] - 90)
        phases['eval_end'] = time.time()
        from .data import write_jsonl
        base, trained = rows['base'], rows['candidate']
        write_jsonl(eval_dir / 'base.jsonl', base); write_jsonl(eval_dir / 'candidate.jsonl', trained)
        from .stats import paired_bootstrap
        from .finalize import flips, select_examples
        result = paired_bootstrap([r['correct'] for r in base], [r['correct'] for r in trained],
                                  samples=cfg['evaluation']['bootstrap_samples'], seed=cfg['evaluation']['bootstrap_seed'])
        counts = flips(base, trained)
        fixed = select_examples(base, trained, 'fixed'); broken = select_examples(base, trained, 'broken', 20)
        write_jsonl(eval_dir / 'fixed_by_training.jsonl', fixed); write_jsonl(eval_dir / 'broken_by_training.jsonl', broken)
        phases['finished'] = time.time()
        (root / 'phases.json').write_text(json.dumps(phases, indent=2))
        final = dict(status='completed', optimizer_steps=steps, selected_step=selected, validation_accuracy=accuracy,
                     validation_by_step=validation, checkpoint=shards, parameter_change=change, test=result, flips=counts,
                     phases=phases, elapsed_seconds=time.time() - begin['started_at'], gpus=gpus)
        (eval_dir / 'result.json').write_text(json.dumps(final, indent=2))
        import wandb
        run = wandb.init(project=cfg['wandb']['project'], entity=cfg['wandb']['entity'], id=runtime['wandb_run_id'], resume='must')
        run.log({'optimizer_step': steps, **{'test/' + k: v for k, v in result.items()}, **{'test/flips_' + k: v for k, v in counts.items()},
                 'checkpoint/reload_passed': True, 'distributed/nodes': 1, 'distributed/gpus': gpus})
        cols = ['sample_id', 'prompt', 'ground_truth', 'base_failure_mode', 'base_completion', 'trained_completion']
        table = wandb.Table(columns=cols)
        for r in fixed: table.add_data(r['sample_id'], str(r['prompt']), r['ground_truth'], r['base_failure_mode'], r['base_completion'], r['trained_completion'])
        run.log({'qualitative/fixed_by_training': table})
        artifact = wandb.Artifact(runtime['run_name'] + '-evidence', type='evaluation')
        for name in ['result.json', 'base.jsonl', 'candidate.jsonl', 'fixed_by_training.jsonl', 'broken_by_training.jsonl']: artifact.add_file(str(eval_dir / name))
        run.log_artifact(artifact); run.summary.update({k: v for k, v in final.items() if k != 'validation_by_step'}); run.finish()
        sync_local('artifacts', prefix, exclude=SYNC_EXCLUDE)
        put_json(prefix + '/completion.json', final)
        print(json.dumps(final), flush=True)
    except BaseException as error:
        failed = dict(identity, status='failed', error_type=type(error).__name__, error=str(error), failed_at=time.time(), phases=phases)
        Path('artifacts/failure.log').write_text(traceback.format_exc())
        try:
            sync_local('artifacts', prefix, exclude=SYNC_EXCLUDE)
            put_json(prefix + '/completion.json', failed)
        except Exception: pass
        raise
    finally:
        subprocess.run([RAY, 'stop', '--force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.environ.pop('WANDB_API_KEY', None)


if __name__ == '__main__': main()
