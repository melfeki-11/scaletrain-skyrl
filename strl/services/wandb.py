"""W&B service helpers. Key comes from an S3 locator; host/key stay scoped in the process env through init, artifact upload,
finish and API reads (the artifact InternalApi otherwise falls back to the public host and fails 401)."""
from __future__ import annotations

import os
import time

import boto3

from ..storage import split_uri


def configure_env(cfg: dict, run_id: str) -> None:
    t = cfg['tracking']
    bucket, key = split_uri(t['credentials_uri'])
    raw = boto3.client('s3').get_object(Bucket=bucket, Key=key)['Body'].read().decode()
    credentials = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    os.environ['WANDB_API_KEY'] = credentials['WANDB_API_KEY'].strip().strip('"\'')
    os.environ.update(WANDB_BASE_URL=t['base_url'], WANDB_ENTITY=t['entity'], WANDB_PROJECT=t['project'], WANDB_RUN_GROUP=t['group'],
                      WANDB_RUN_ID=run_id, WANDB_RESUME='allow')


def smoke(cfg: dict, run_id: str) -> dict:
    """Write a scalar + an INCREMENTAL table + an artifact to a throwaway run, finish, and read all of it back."""
    import wandb
    t = cfg['tracking']
    run = wandb.init(project=t['project'], entity=t['entity'], id=run_id, group=t['group'] + '-dryrun', job_type='dry-run',
                     settings=wandb.Settings(x_disable_stats=True) if hasattr(wandb.Settings, 'x_disable_stats') else None)
    try:
        try:
            table = wandb.Table(columns=['step', 'value'], log_mode='INCREMENTAL')
            incremental = True
        except TypeError:
            table, incremental = wandb.Table(columns=['step', 'value']), False
        for step in (1, 2):
            table.add_data(step, step * 10)
            run.log({'optimizer_step': step, 'dryrun/value': float(step), 'dryrun/table': table})
        artifact = wandb.Artifact(f'{run_id}-receipt', type='dry-run')
        with artifact.new_file('receipt.txt') as handle:
            handle.write('ok')
        run.log_artifact(artifact).wait()
        run.summary['dryrun_passed'] = 1
    finally:
        run.finish()
    api = wandb.Api(overrides={'base_url': t['base_url']}, timeout=30)
    for _ in range(10):
        remote = api.run(f"{t['entity']}/{t['project']}/{run_id}")
        if remote.state == 'finished' and remote.summary.get('dryrun_passed') == 1:
            history = list(remote.scan_history(keys=['optimizer_step', 'dryrun/value']))
            arts = [a.name for a in remote.logged_artifacts()]
            return {'wandb_write_readback': True, 'history_rows': len(history), 'artifacts': arts, 'incremental_table': incremental,
                    'url': remote.url, 'sdk': wandb.__version__}
        time.sleep(3)
    raise RuntimeError('W&B dry-run readback did not confirm the finished run')


def local_key(env_file: str | None = None) -> None:
    """Laptop W&B reads (monitor/compare/verify): WANDB_API_KEY from $WANDB_API_KEY, else a mode-600 dotenv
    ($STRL_WANDB_ENV_FILE, default ScaleTrain/.env.wandb). Never printed."""
    from pathlib import Path
    if os.environ.get('WANDB_API_KEY'):
        return
    path = Path(env_file or os.environ.get('STRL_WANDB_ENV_FILE') or Path(__file__).resolve().parents[3] / '.env.wandb')
    if not path.is_file():
        raise RuntimeError(f'no W&B key: export WANDB_API_KEY or create mode-600 {path}')
    if path.stat().st_mode & 0o077:
        raise RuntimeError(f'{path} must be mode 600')
    for line in path.read_text().splitlines():
        name, sep, value = line.strip().removeprefix('export ').partition('=')
        if sep and name.strip() == 'WANDB_API_KEY':
            os.environ['WANDB_API_KEY'] = value.strip().strip('"\'')
            return
    raise RuntimeError(f'{path} has no WANDB_API_KEY')
