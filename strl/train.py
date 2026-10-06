"""Render + validate the SkyRL overrides with the real parser, then exec the entrypoint (or stop after validation)."""
import json
import os
import subprocess
import sys
from pathlib import Path

from .experiment import current
from .overrides import overrides


def build(stage: str, run_name: str):
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.utils import validate_cfg
    from .runtime import validate_generator
    experiment = current()
    values = overrides(experiment, stage=stage, run_name=run_name)
    validated = SkyRLTrainConfig.from_cli_overrides(values)
    logger = validated.trainer.logger
    validated.trainer.logger = 'console'   # validation must not require W&B
    validate_cfg(validated)
    validated.trainer.logger = logger
    receipt = validate_generator(validated, experiment)
    return values, validated, receipt


def main():
    name = os.environ['GRPO_RUN_NAME']
    stage = os.environ.get('STRL_STAGE', 'train')
    values, validated, receipt = build(stage, name)
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/schema.json').write_text(json.dumps({'validated': True, 'stage': stage, 'overrides': values,
                                                         'generator': receipt}, indent=2, default=str))
    if '--validate-only' in sys.argv:
        print(json.dumps({'validated': True, 'overrides': len(values), 'generator': receipt}, default=str))
        return
    subprocess.run([sys.executable, '-m', 'strl.entrypoint', *values], check=True)


if __name__ == '__main__':
    main()
