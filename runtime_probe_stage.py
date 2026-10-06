"""Render a CPU-only (m5.2xlarge) preflight job in the training image.

It runs the real SkyRL config parser, validate_cfg, the actual generator checks, reward/metrics contracts and the actual
inference/router/FSDP/trainer imports against this project's 8-GPU overrides, before any GPU is requested.
Result JSON is written to S3 under <prefix>/runtime-probes/<name>.json. Never submits.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile
import boto3
from q35.config import load

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--output', default='runtime-check-job.json')
a = p.parse_args()
cfg = load(); s3cfg = cfg['s3']
out = io.BytesIO()
with tarfile.open(fileobj=out, mode='w:gz') as tar:
    for path in ['q35', 'config.yaml', 'assets/runtime']:
        tar.add(path, arcname=path, filter=lambda x: None if '__pycache__' in x.name or x.name == 'packages' else x)
raw = out.getvalue(); h = hashlib.sha256(raw).hexdigest()
key = f"{s3cfg['prefix']}/runtime-probes/{h}.tgz"; result_key = f"{s3cfg['prefix']}/runtime-probes/{a.name}.json"
boto3.Session(profile_name=s3cfg['profile']).client('s3', region_name=s3cfg['region']).put_object(
    Bucket=s3cfg['bucket'], Key=key, Body=raw, ServerSideEncryption='AES256')
code = f'''import boto3,io,tarfile,hashlib,json
s3=boto3.client('s3',region_name={s3cfg['region']!r})
raw=s3.get_object(Bucket={s3cfg['bucket']!r},Key={key!r})['Body'].read()
assert hashlib.sha256(raw).hexdigest()=={h!r}
tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz').extractall('.',filter='data')
from q35.runtime import preflight
result=preflight()
from q35.config import load
from q35.skyrl import overrides
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import validate_cfg
cfg=SkyRLTrainConfig.from_cli_overrides(overrides(load(),stage='train',run_name='runtime-check'))
cfg.trainer.logger='console'
validate_cfg(cfg)
from q35.runtime import validate_generator
result.update(validate_generator(cfg))
result['world_size']=cfg.trainer.placement.policy_num_nodes*cfg.trainer.placement.policy_num_gpus_per_node
result['engines']=cfg.generator.inference_engine.num_engines
result['schema_passed']=True
s3.put_object(Bucket={s3cfg['bucket']!r},Key={result_key!r},Body=json.dumps(result).encode(),ServerSideEncryption='AES256')
print(json.dumps(result))
'''
compile(code, '<runtime-probe>', 'exec')
command = ('set -euo pipefail\nexport PATH=/opt/skyrl-venv/bin:$PATH\nmkdir -p /workspace/runtime-check\ncd /workspace/runtime-check\n'
           'export PYTHONPATH=/workspace/runtime-check:/workspace/src:/workspace/vendor/SkyRL:/workspace/vendor/SkyRL/skyrl-gym\n'
           '/opt/skyrl-venv/bin/python -c ' + shlex.quote(code))
job = {'job_type': 'k8s-preset', 'job_name': a.name, 'image': cfg['runtime']['image'], 'team': cfg['scale_train']['team'],
       'product': cfg['scale_train']['product'], 'num_instances': 1, 'preset_instance_type': 'm5.2xlarge',
       'active_deadline_seconds': 900, 'priority': cfg['scale_train']['priority'], 'allow_borrowing': False,
       'command': ['bash', '-c', command]}
Path(a.output).write_text(json.dumps(job, indent=2))
print(json.dumps({'name': a.name, 'result_key': result_key, 'payload_sha256': h}))
