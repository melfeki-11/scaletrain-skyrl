"""Render + validate the SkyRL overrides with the real parser, then exec the entrypoint (or stop after validation)."""
import json
import os
import subprocess
import sys
from pathlib import Path

from .experiment import current
from .overrides import overrides


def build(stage: str, run_name: str):
    from skyrl.train.utils import validate_cfg
    from .runtime import validate_generator
    from .tasks import get_environment
    experiment = current()
    values = overrides(experiment, stage=stage, run_name=run_name)
    validated = get_environment(experiment['environment']['name']).skyrl_config_class().from_cli_overrides(values)
    logger = validated.trainer.logger
    validated.trainer.logger = 'console'   # validation must not require W&B
    validate_cfg(validated)
    validated.trainer.logger = logger
    receipt = validate_generator(validated, experiment)
    from .entrypoint import build_datasets
    receipt['datasets'] = build_datasets(get_environment(experiment['environment']['name']), experiment, validated)
    return values, validated, receipt


def main():
    name = os.environ['GRPO_RUN_NAME']
    stage = os.environ.get('STRL_STAGE', 'train')
    values, validated, receipt = build(stage, name)
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/schema.json').write_text(json.dumps({'validated': True, 'stage': stage, 'overrides': values,
                                                         'generator': receipt}, indent=2, default=str))
    if '--validate-only' in sys.argv:
        layout_keys = ('trainer.placement.', 'trainer.policy.fsdp_config.', 'trainer.policy.model.lora.', 'trainer.strategy',
                       'generator.inference_engine.tensor_parallel_size', 'generator.inference_engine.num_engines',
                       'generator.inference_engine.gpu_memory_utilization', 'trainer.ckpt_path', 'trainer.resume_mode')
        layout = {k: v for k, v in (x.split('=', 1) for x in values) if k.startswith(layout_keys)}   # WHAT was validated, not just how many
        print(json.dumps({'validated': True, 'overrides': len(values), 'layout': layout, 'generator': receipt}, default=str))
        return
    subprocess.run([sys.executable, '-m', 'strl.entrypoint', *values], check=True)


if __name__ == '__main__':
    main()
