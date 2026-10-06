"""Stage an experiment for ScaleTrain: upload immutable assets and render ``job.json``. NEVER submits.

    python -m strl stage experiments/<name>.yaml --attempt <unique-name> --mode dry|canary|train

dry    CPU (m5.2xlarge): every gate the GPU run depends on, no GPU spend.
canary the experiment with the canary overlay (base/canary_overlay.yaml: 1 GPU, small batch, 2 updates, no final eval);
       model, environment and reward are unchanged, so it tests exactly what the real run will use.
train  the experiment as written, time-bounded by the program budget ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import models
from .budget import remaining_seconds
from .experiment import REPO_ROOT, deep_merge, load, save_resolved, validate, derive_placement, experiment_sha
from .tasks import get_environment

POD_ROOT = '/workspace/strl-run'
CANARY_OVERLAY = REPO_ROOT / 'base/canary_overlay.yaml'
SKIP = {'__pycache__', 'packages', '.DS_Store'}


def apply_canary_overlay(cfg: dict) -> dict:
    import yaml
    overlay = yaml.safe_load(CANARY_OVERLAY.read_text())
    out = deep_merge({k: v for k, v in cfg.items() if k not in {'derived', '_meta'}}, overlay)
    out['algorithm']['n_samples'] = min(cfg['algorithm']['n_samples'], overlay['algorithm'].get('n_samples', 4))
    out['name'] = cfg['name'] + '-canary'
    validate(out)
    out['derived'] = derive_placement(out)
    out['_meta'] = dict(cfg['_meta'], sha=experiment_sha(out), canary_of=cfg['_meta']['sha'])
    return out


def preset(cfg: dict) -> str:
    instance, gpus = cfg['compute']['instance_type'], cfg['compute']['gpus_per_node']
    return instance if gpus == 8 or ':' in instance else f'{instance}:{gpus}'


def model_prefix(base_prefix: str, cfg: dict) -> str:
    return f"{base_prefix}/models/{models.served_name(cfg['model']['id'])}@{cfg['model']['revision'][:12]}"


def relative(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def tar_payload(entries: list[tuple[Path, str]]) -> bytes:
    """entries: (source path, archive name). Deterministic: sorted members, zeroed mtimes/owners, gzip mtime 0 (same content =>
    same digest => uploaded once)."""
    import gzip

    def clean(info):
        if any(part in SKIP for part in Path(info.name).parts):
            return None
        info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, '', ''
        return info

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w', format=tarfile.PAX_FORMAT) as tar:
        for source, arcname in entries:
            paths = [source] if source.is_file() else [source, *sorted(p for p in source.rglob('*'))]
            for path in paths:
                name = arcname if path == source else f'{arcname}/{path.relative_to(source)}'
                tar.add(path, arcname=name, recursive=False, filter=clean)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode='wb', mtime=0) as handle:
        handle.write(raw.getvalue())
    return out.getvalue()


def upload_once(s3, bucket: str, key: str, raw: bytes) -> None:
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except Exception:
        s3.put_object(Bucket=bucket, Key=key, Body=raw, ServerSideEncryption='AES256')


def render_command(*, region: str, bucket: str, key: str, digest: str, model_prefix_: str, model_rel: str, mode: str,
                   overlay: tuple[str, str] | None) -> str:
    overlay_fetch = '' if overlay is None else (
        f"o=s3.get_object(Bucket={bucket!r},Key={overlay[0]!r})['Body'].read()\n"
        f"assert hashlib.sha256(o).hexdigest()=={overlay[1]!r}\n"
        "tarfile.open(fileobj=io.BytesIO(o),mode='r:gz').extractall('.',filter='data')\n")
    bootstrap = f'''import boto3,hashlib,io,tarfile,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
s3=boto3.client('s3',region_name={region!r})
b=s3.get_object(Bucket={bucket!r},Key={key!r})['Body'].read()
assert hashlib.sha256(b).hexdigest()=={digest!r}
tarfile.open(fileobj=io.BytesIO(b),mode='r:gz').extractall('.',filter='data')
{overlay_fetch}Path({model_rel!r}).mkdir(parents=True,exist_ok=True)
def fetch(record):
 p=Path({model_rel!r})/record['name']
 s3.download_file({bucket!r},{model_prefix_ + '/'!r}+record['name'],str(p))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 assert h.hexdigest()==record['sha256'],record['name']
with ThreadPoolExecutor(4) as pool:list(pool.map(fetch,json.loads(Path('assets/model-manifest.json').read_text())))
print('Verified pinned payload, model and data staging',flush=True)
'''
    pythonpath = (f'{POD_ROOT}/assets/overlay:' if overlay is not None else '') + f'{POD_ROOT}:/workspace/src:/workspace/vendor/SkyRL:/workspace/vendor/SkyRL/skyrl-gym'
    return ('set -euo pipefail\nexport PATH=/opt/skyrl-venv/bin:$PATH\n'
            f'mkdir -p {POD_ROOT}\ncd {POD_ROOT}\n'
            f'export PYTHONPATH={pythonpath}\n'
            'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SKYRL_PYTHONPATH_EXPORT=1\n'
            'export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=300 SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=600\n'
            'export VLLM_DISABLE_PYNCCL=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0\n' + ('export GRPO_DRY_RUN=1\n' if mode == 'dry' else '') +
            f'export STRL_MODE={mode} STRL_RESOLVED={POD_ROOT}/resolved.json\n'
            '/opt/skyrl-venv/bin/python -c ' + shlex.quote(bootstrap) + '\nexec /opt/skyrl-venv/bin/python -m strl.worker\n')


def stage(experiment_path: str, *, attempt: str, mode: str, ledger: str | None = None, team: str | None = None,
          allow_borrowing: bool = False, output: str = 'job.json', s3=None, now: time.struct_time | None = None) -> dict:
    if mode not in {'dry', 'canary', 'train'}:
        raise ValueError('mode must be dry, canary or train')
    if not (len(attempt) <= 32 and attempt == attempt.lower() and attempt.replace('-', '').isalnum()):
        raise ValueError('attempt (job name) must be lowercase alphanumeric/hyphen, <= 32 characters')
    cfg = load(experiment_path)
    if mode == 'canary':
        cfg = apply_canary_overlay(cfg)
    env = get_environment(cfg['environment']['name'])
    profile = models.detect(cfg['model'])[0]
    ledger_path = Path(ledger or (REPO_ROOT / cfg['budget']['ledger']))
    ledger_data = json.loads(ledger_path.read_text())
    remaining = remaining_seconds(ledger_data['attempts'], ledger_data['total_seconds'])
    reserve = cfg['budget']['final_eval_reserve_seconds']
    gpus = cfg['compute']['gpus_per_node']
    gpu_fraction = 0.0 if mode == 'dry' else gpus / 8.0
    if mode == 'train':
        # the ledger counts node-seconds; a k-GPU slice may run (8/k) wall-seconds per node-second
        affordable = int(remaining / gpu_fraction) if gpu_fraction else remaining
        if affordable <= reserve + 900:
            raise RuntimeError(f'Insufficient remaining budget ({remaining} node-s = {affordable}s at {gpus} GPU(s)) for bootstrap, training and evaluation')
        remaining = min(affordable, cfg['budget']['attempt_max_seconds'])
        worker_seconds = remaining - cfg['budget']['bootstrap_margin_seconds']
    elif mode == 'canary':
        affordable = int(remaining / gpu_fraction) if gpu_fraction else remaining
        # the canary must write canary.json within budget.canary_max_seconds of the child start; the job adds pod start + Ray
        remaining = min(affordable, min(3600, int(cfg['budget']['canary_max_seconds']) + 600))
        if remaining < 900:
            raise RuntimeError(f'Insufficient remaining budget for a canary ({affordable}s at {gpus} GPU(s))')
        worker_seconds = remaining - cfg['budget']['bootstrap_margin_seconds']
    else:
        remaining, worker_seconds = env_dry_seconds(env), env_dry_seconds(env) - 100

    st = cfg['storage']
    bucket, base_prefix = st['bucket'], st['prefix']
    stamp = time.strftime('%y%m%d%H%M%S', now or time.gmtime())
    attempt_prefix = f'{base_prefix}/attempts/{attempt}'
    artifact_uri = f's3://{bucket}/{attempt_prefix}/artifacts'
    wandb_run_id = f'{attempt}-{stamp}'
    if s3 is None:
        import boto3
        s3 = boto3.Session(profile_name=st['profile']).client('s3', region_name=st['region'])
    ledger_key = f'{attempt_prefix}/budget-ledger.json'
    ledger_bytes = json.dumps(ledger_data).encode()
    try:  # write-once per attempt; re-staging an unsubmitted attempt with the SAME ledger is allowed, a different one is not
        s3.put_object(Bucket=bucket, Key=ledger_key, Body=ledger_bytes, ServerSideEncryption='AES256', IfNoneMatch='*')
    except Exception as error:
        if 'PreconditionFailed' not in str(error) and '412' not in str(error):
            raise
        if s3.get_object(Bucket=bucket, Key=ledger_key)['Body'].read() != ledger_bytes:
            raise RuntimeError(f'attempt {attempt} was already staged against a different budget ledger; use a new attempt name') from None

    model_dir = Path(cfg['model']['local_path'])
    if not model_dir.is_dir():
        raise RuntimeError(f'model not staged locally at {model_dir}; run: python -m strl fetch {experiment_path}')
    model_files = sorted(p for p in model_dir.iterdir() if p.is_file() and not p.name.startswith('.') and p.name not in {'README.md', 'LICENSE'})
    mprefix = model_prefix(base_prefix, cfg)

    def upload(path: Path) -> dict:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = f'{mprefix}/{path.name}'
        try:
            prior = s3.head_object(Bucket=bucket, Key=key)
        except Exception:
            prior = {}
        if prior.get('Metadata', {}).get('sha256') != digest:
            s3.upload_file(str(path), bucket, key, ExtraArgs={'ServerSideEncryption': 'AES256', 'Metadata': {'sha256': digest}})
        return {'name': path.name, 'bytes': path.stat().st_size, 'sha256': digest}

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(upload, model_files))
    staging = Path(tempfile.mkdtemp(prefix='strl-stage-'))
    (staging / 'model-manifest.json').write_text(json.dumps(manifest, indent=2))

    runtime = {'artifact_uri': artifact_uri, 'budget_ledger_uri': f's3://{bucket}/{ledger_key}', 'remaining_seconds': worker_seconds,
               'total_seconds': ledger_data['total_seconds'], 'model_uri': f's3://{bucket}/{mprefix}', 'wandb_run_id': wandb_run_id,
               'run_name': attempt, 'mode': mode, 'gpu_fraction': gpu_fraction, 'experiment': cfg['name'], 'experiment_sha': cfg['_meta']['sha']}
    (staging / 'runtime.json').write_text(json.dumps(runtime, indent=2))
    save_resolved(cfg, staging / 'resolved.json', root=POD_ROOT)

    overlay = (REPO_ROOT / 'assets/overlay')
    needs_overlay = env.needs_modal
    if needs_overlay and not (overlay / 'modal' / '__init__.py').is_file():
        raise RuntimeError('this environment needs the Modal/Harbor overlay: run python -m strl overlay')
    entries = [(REPO_ROOT / 'strl', 'strl'), (staging / 'resolved.json', 'resolved.json'), (staging / 'runtime.json', 'runtime.json'),
               (Path(cfg['run']['data_dir']), relative(cfg['run']['data_dir'])), (REPO_ROOT / 'assets/runtime', 'assets/runtime'),
               (staging / 'model-manifest.json', 'assets/model-manifest.json')]
    from .rewards import custom_payload_paths
    entries += [(REPO_ROOT / top, top) for top in custom_payload_paths(cfg['reward']['components'])]   # user 'module:function' code
    overlay_ref = None
    if needs_overlay:  # content-addressed separately: ~125 MB uploaded once, reused by every attempt
        overlay_raw = tar_payload([(overlay, 'assets/overlay')])
        overlay_ref = (f"{base_prefix}/overlays/overlay-{hashlib.sha256(overlay_raw).hexdigest()}.tar.gz", hashlib.sha256(overlay_raw).hexdigest())
        upload_once(s3, bucket, overlay_ref[0], overlay_raw)
    raw = tar_payload(entries)
    digest = hashlib.sha256(raw).hexdigest()
    key = f'{base_prefix}/payloads/payload-{digest}.tar.gz'
    upload_once(s3, bucket, key, raw)
    command = render_command(region=st['region'], bucket=bucket, key=key, digest=digest, model_prefix_=mprefix,
                             model_rel=relative(cfg['model']['local_path']), mode=mode, overlay=overlay_ref)
    job = {'job_type': 'k8s-preset', 'job_name': attempt, 'image': cfg['compute']['image'], 'team': team or cfg['compute']['team'],
           'product': cfg['compute']['product'], 'num_instances': 1, 'preset_instance_type': 'm5.2xlarge' if mode == 'dry' else preset(cfg),
           'active_deadline_seconds': remaining, 'priority': cfg['compute']['priority'],
           'allow_borrowing': allow_borrowing or cfg['compute']['allow_borrowing'], 'command': ['bash', '-c', command]}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(job, indent=2))
    receipt = {'staged': True, 'name': attempt, 'mode': mode, 'experiment': cfg['name'], 'experiment_sha': cfg['_meta']['sha'],
               'payload_sha256': digest, 'payload_megabytes': round(len(raw) / 1e6, 1), 'remaining_seconds': remaining,
               'worker_seconds': worker_seconds, 'preset': job['preset_instance_type'], 'gpu_fraction': gpu_fraction, 'team': job['team'],
               'allow_borrowing': job['allow_borrowing'], 'wandb_run_id': wandb_run_id, 'artifact_uri': artifact_uri, 'job_file': output,
               'profile': profile.name, 'staging_dir': str(staging)}
    return receipt


def env_dry_seconds(env) -> int:
    return int(getattr(env, 'dry_run_seconds', 900))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('experiment')
    p.add_argument('--attempt', required=True, help='unique job/attempt name (<= 32 chars), e.g. mbpp-bin-a1')
    p.add_argument('--mode', choices=['dry', 'canary', 'train'], default='train')
    p.add_argument('--ledger', help='program budget ledger (default: budget.ledger from the experiment)')
    p.add_argument('--team', help='override team (capacity routing result)')
    p.add_argument('--allow-borrowing', action='store_true')
    p.add_argument('--output', default='job.json')
    a = p.parse_args(argv)
    print(json.dumps(stage(a.experiment, attempt=a.attempt, mode=a.mode, ledger=a.ledger, team=a.team, allow_borrowing=a.allow_borrowing,
                           output=a.output)))


if __name__ == '__main__':
    main()
