"""Stage immutable assets to S3 and render the ScaleTrain job (single p5.48xlarge node). Never submits."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile
import time
import boto3
from q35.budget import remaining_seconds
from q35.config import load

parser = argparse.ArgumentParser()
parser.add_argument('--attempt', required=True, help='unique job/attempt name, e.g. q35v2-a1')
parser.add_argument('--ledger', default='evidence/budget-ledger.json')
parser.add_argument('--team', help='override team (capacity routing result)')
parser.add_argument('--allow-borrowing', action='store_true')
parser.add_argument('--output', default='job.json')
args = parser.parse_args()
cfg = load()
root = Path(__file__).resolve().parent
ledger = json.loads(Path(args.ledger).read_text())
remaining = remaining_seconds(ledger['attempts'], ledger['total_seconds'])
reserve = cfg['budget']['final_eval_reserve_seconds']
if remaining <= reserve + 900:
    raise RuntimeError(f'Insufficient remaining budget ({remaining}s) for bootstrap, training and evaluation')
worker_seconds = remaining - cfg['budget']['bootstrap_margin_seconds']

s3cfg = cfg['s3']
bucket, base_prefix = s3cfg['bucket'], s3cfg['prefix']
stamp = time.strftime('%y%m%d%H%M%S', time.gmtime())
attempt_prefix = f"{base_prefix}/attempts/{args.attempt}"
artifact_uri = f"s3://{bucket}/{attempt_prefix}/artifacts"
wandb_run_id = 'q35v2' + stamp
s3 = boto3.Session(profile_name=s3cfg['profile']).client('s3', region_name=s3cfg['region'])
ledger_key = f"{attempt_prefix}/budget-ledger.json"
s3.put_object(Bucket=bucket, Key=ledger_key, Body=json.dumps(ledger).encode(), ServerSideEncryption='AES256', IfNoneMatch='*')

model_files = sorted(p for p in (root / 'assets/model').iterdir()
                     if p.is_file() and not p.name.startswith('.') and p.name not in {'README.md', 'LICENSE'})
def upload(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    key = f"{base_prefix}/model/{path.name}"
    try: prior = s3.head_object(Bucket=bucket, Key=key)
    except Exception: prior = {}
    if prior.get('Metadata', {}).get('sha256') != digest:
        s3.upload_file(str(path), bucket, key, ExtraArgs={'ServerSideEncryption': 'AES256', 'Metadata': {'sha256': digest}})
    return {'name': path.name, 'bytes': path.stat().st_size, 'sha256': digest}
with ThreadPoolExecutor(max_workers=4) as pool: manifest = list(pool.map(upload, model_files))
(root / 'assets/model-manifest.json').write_text(json.dumps(manifest, indent=2))

runtime = {'artifact_uri': artifact_uri, 'budget_ledger_uri': f's3://{bucket}/{ledger_key}', 'remaining_seconds': worker_seconds,
           'total_seconds': ledger['total_seconds'], 'model_uri': f's3://{bucket}/{base_prefix}/model',
           'wandb_run_id': wandb_run_id, 'run_name': args.attempt}
(root / 'runtime.json').write_text(json.dumps(runtime, indent=2))

archive = io.BytesIO()
with tarfile.open(fileobj=archive, mode='w:gz') as tar:
    for relative in ['q35', 'config.yaml', 'runtime.json', 'assets/data', 'assets/runtime', 'assets/model-manifest.json']:
        tar.add(root / relative, arcname=relative,
                filter=lambda x: None if '__pycache__' in x.name or x.name == 'packages' else x)
raw = archive.getvalue(); digest = hashlib.sha256(raw).hexdigest()
key = f"{base_prefix}/payloads/payload-{digest}.tar.gz"
s3.put_object(Bucket=bucket, Key=key, Body=raw, ServerSideEncryption='AES256')

bootstrap = f'''import boto3,hashlib,io,tarfile,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
s3=boto3.client('s3',region_name={s3cfg['region']!r})
b=s3.get_object(Bucket={bucket!r},Key={key!r})['Body'].read()
assert hashlib.sha256(b).hexdigest()=={digest!r}
tarfile.open(fileobj=io.BytesIO(b),mode='r:gz').extractall('.',filter='data')
Path('assets/model').mkdir(parents=True,exist_ok=True)
def fetch(record):
 p=Path('assets/model')/record['name']
 s3.download_file({bucket!r},{base_prefix + '/model/'!r}+record['name'],str(p))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 assert h.hexdigest()==record['sha256'],record['name']
with ThreadPoolExecutor(4) as pool:list(pool.map(fetch,json.loads(Path('assets/model-manifest.json').read_text())))
print('Verified pinned payload, model and data staging',flush=True)
'''
command = ('set -euo pipefail\nexport PATH=/opt/skyrl-venv/bin:$PATH\nmkdir -p /workspace/qwen35-grpo\ncd /workspace/qwen35-grpo\n'
           'export PYTHONPATH=/workspace/qwen35-grpo:/workspace/src:/workspace/vendor/SkyRL:/workspace/vendor/SkyRL/skyrl-gym\n'
           'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SKYRL_PYTHONPATH_EXPORT=1\n'
           'export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=300 SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=600\n'
           'export VLLM_DISABLE_PYNCCL=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0\n'
           '/opt/skyrl-venv/bin/python -c ' + shlex.quote(bootstrap) + '\nexec /opt/skyrl-venv/bin/python -m q35.worker\n')
job = {'job_type': 'k8s-preset', 'job_name': args.attempt, 'image': cfg['runtime']['image'],
       'team': args.team or cfg['scale_train']['team'], 'product': cfg['scale_train']['product'], 'num_instances': 1,
       'preset_instance_type': cfg['scale_train']['instance_type'], 'active_deadline_seconds': remaining,
       'priority': cfg['scale_train']['priority'], 'allow_borrowing': args.allow_borrowing, 'command': ['bash', '-c', command]}
Path(args.output).write_text(json.dumps(job, indent=2))
print(json.dumps({'staged': True, 'name': args.attempt, 'payload_sha256': digest, 'remaining_seconds': remaining,
                  'worker_seconds': worker_seconds, 'team': job['team'], 'allow_borrowing': job['allow_borrowing'],
                  'wandb_run_id': wandb_run_id}))
