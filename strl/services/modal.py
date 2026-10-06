"""Modal credentials on ScaleTrain workers: an encrypted S3 object or Secrets Manager id holding MODAL_TOKEN_ID/SECRET is
turned into a mode-600 profile TOML; only MODAL_CONFIG_PATH / MODAL_PROFILE / MODAL_ENVIRONMENT ever reach Ray actors."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _parse(raw: str) -> dict:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    out = {}
    for line in raw.splitlines():
        name, sep, value = line.strip().removeprefix('export ').partition('=')
        if sep and not name.startswith('#'):
            out[name.strip()] = value.strip().strip('"\'')
    return out


def materialize_profile(raw: str, dest: str | Path, profile: str = 'scaleapi') -> Path:
    values = _parse(raw)
    tid, tsec = values.get('MODAL_TOKEN_ID'), values.get('MODAL_TOKEN_SECRET')
    if not (isinstance(tid, str) and tid and isinstance(tsec, str) and tsec) or any(c in tid + tsec for c in '\r\n'):
        raise RuntimeError('Modal credential bundle lacks a valid MODAL_TOKEN_ID/MODAL_TOKEN_SECRET')
    target = Path(dest)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as handle:
        handle.write(f'[{profile}]\ntoken_id = {json.dumps(tid)}\ntoken_secret = {json.dumps(tsec)}\n')
    os.chmod(target, 0o600)
    return target


def configure_from_locator(locator: str, dest: str | Path, environment: str = 'main', profile: str = 'scaleapi') -> Path:
    """Fetch the bundle (s3:// or Secrets Manager id), write the private profile, export the non-secret pointers."""
    import boto3
    if locator.startswith('s3://'):
        from ..storage import split_uri
        bucket, key = split_uri(locator)
        raw = boto3.client('s3').get_object(Bucket=bucket, Key=key)['Body'].read().decode()
    else:
        raw = boto3.client('secretsmanager').get_secret_value(SecretId=locator)['SecretString']
    path = materialize_profile(raw, dest, profile)
    del raw
    os.environ.update(MODAL_CONFIG_PATH=str(path), MODAL_PROFILE=profile, MODAL_ENVIRONMENT=environment)
    os.environ.pop('MODAL_TOKEN_ID', None)
    os.environ.pop('MODAL_TOKEN_SECRET', None)
    return path


def check(environment: str = 'main', secret: str | None = None) -> dict:
    """Modal auth + environment visibility (+ sandbox secret presence) through the CLI, without reading secret values."""
    def cli(*args):
        out = subprocess.run([sys.executable, '-m', 'modal', *args, '--json'], capture_output=True, text=True, timeout=120)
        if out.returncode:
            raise RuntimeError(f'modal {args[0]} {args[1]} failed: {(out.stderr or out.stdout).strip().splitlines()[-1:]}')
        return json.loads(out.stdout)
    envs = [e.get('name') for e in cli('environment', 'list')]
    if environment not in envs:
        raise RuntimeError(f'Modal environment {environment!r} not visible')
    receipt = {'modal_auth': 'ok', 'modal_environment': environment}
    if secret:
        names = [s.get('name') or s.get('Name') for s in cli('secret', 'list', '--env', environment)]
        if secret not in names:
            raise RuntimeError(f'Modal secret {secret!r} missing in {environment!r}')
        receipt['modal_secret'] = secret
    return receipt
