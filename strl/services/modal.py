"""Modal credentials on ScaleTrain workers: an encrypted S3 object or Secrets Manager id holding MODAL_TOKEN_ID/SECRET is
turned into a mode-600 profile TOML; only MODAL_CONFIG_PATH / MODAL_PROFILE / MODAL_ENVIRONMENT ever reach Ray actors."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# The team Modal environment every strl app, sandbox (Harbor's `__harbor__` app) and secret lives in. `main` is the shared workspace
# default and hit its budget cap on 2026-10-12 (Modal stops everything running there at the cap); genai-research is the team
# environment created for this work ($2,000/month; any workspace member can deploy). Experiments set services.modal.environment.
DEFAULT_ENVIRONMENT = 'genai-research'


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


def configure_from_locator(locator: str, dest: str | Path, environment: str = DEFAULT_ENVIRONMENT, profile: str = 'scaleapi') -> Path:
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


def check(environment: str = DEFAULT_ENVIRONMENT, secret: str | None = None) -> dict:
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


DEFAULT_ENV_FILE = Path(os.environ.get('STRL_MODAL_ENV_FILE', Path(__file__).resolve().parents[3] / '.env.modal'))


class local_profile:
    """Context manager for laptop use: materialize a temporary mode-600 profile from a mode-600 MODAL_TOKEN_ID/SECRET dotenv
    (default: ScaleTrain/.env.modal, override with $STRL_MODAL_ENV_FILE). If MODAL_CONFIG_PATH is already set, it is used as is."""

    def __init__(self, environment: str = DEFAULT_ENVIRONMENT, env_file: str | Path | None = None):
        self.environment, self.env_file, self.path = environment, Path(env_file or DEFAULT_ENV_FILE), None

    def __enter__(self):
        os.environ['MODAL_ENVIRONMENT'] = self.environment
        if os.environ.get('MODAL_CONFIG_PATH'):
            os.environ.setdefault('MODAL_PROFILE', 'scaleapi')
            return Path(os.environ['MODAL_CONFIG_PATH'])
        if self.env_file.stat().st_mode & 0o077:
            raise RuntimeError(f'{self.env_file} must be mode 600')
        import tempfile
        fd, name = tempfile.mkstemp(prefix='modal-', suffix='.toml')
        os.close(fd)
        self.path = materialize_profile(self.env_file.read_text(), name)
        os.environ.update(MODAL_CONFIG_PATH=str(self.path), MODAL_PROFILE='scaleapi')
        os.environ.pop('MODAL_TOKEN_ID', None)
        os.environ.pop('MODAL_TOKEN_SECRET', None)
        return self.path

    def __exit__(self, *exc):
        if self.path is not None:
            os.environ.pop('MODAL_CONFIG_PATH', None)
            self.path.unlink(missing_ok=True)
        return False
