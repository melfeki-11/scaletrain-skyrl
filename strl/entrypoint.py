"""SkyRL entrypoint for every experiment: generic tracker/logger/callback + the environment's generator and env registration."""
from __future__ import annotations

import os
import sys

# Forwarded into every Ray actor's runtime env. Raw Modal tokens and LiteLLM keys are NEVER forwarded (pinned SkyRL would
# otherwise copy MODAL_TOKEN_ID/SECRET into runtime_env); actors get file paths (MODAL_CONFIG_PATH, STRL_LITELLM_KEY_FILE).
FORWARD_PREFIXES = ('WANDB_', 'GRPO_', 'NCCL_', 'GLOO_', 'STRL_')
FORWARD_KEYS = {'PYTHONPATH', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'MODAL_CONFIG_PATH', 'MODAL_PROFILE', 'MODAL_ENVIRONMENT'}
NEVER_FORWARD = {'MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET', 'LITELLM_PROXY_API_KEY', 'DAYTONA_API_KEY'}


IGNORE_ROW_ENV_CLASS = '__strl_ignore_row_env_class__'


def prepare_dataset(dataset, env_spec, experiment, split):
    """Make every row use the experiment's env class and refuse silently filtered prompts.

    SkyRL lets a row's ``env_class`` column override ``environment.env_class`` (dataset.py:89, generators/utils.py:536). GSM8K rows
    carry ``env_class: gsm8k``, so a custom reward design would silently run the built-in env; the generic path therefore ignores
    the row column. SkyRL also drops prompts longer than max_prompt_length without failing; environments that declare their row
    counts get a hard check instead.
    """
    if dataset is None:
        return None
    if env_spec.uses_generic_reward(experiment):
        dataset.env_class_key = IGNORE_ROW_ENV_CLASS
    expected = env_spec.expected_rows(experiment, split)
    if expected is not None and len(dataset) != expected:
        raise RuntimeError(f'{split} dataset has {len(dataset)} rows after SkyRL prompt-length filtering, expected {expected}: '
                           'raise algorithm.max_prompt_tokens or re-prepare the data')
    return dataset


def build_datasets(env_spec, experiment, cfg) -> dict:
    """Build the REAL train/validation datasets exactly as the trainer will (used by the CPU dry run / --validate-only), so
    dataset-class mistakes and SkyRL's silent prompt-length filtering surface before any GPU is allocated."""
    out = {}
    for split in ('train', 'validation'):
        dataset = env_spec.build_dataset(experiment, cfg, split)
        if dataset is None:
            files = cfg.data.train_data if split == 'train' else cfg.data.val_data
            if not files:
                continue
            from transformers import AutoTokenizer
            from skyrl.train.dataset import PromptDataset
            tokenizer = AutoTokenizer.from_pretrained(cfg.trainer.policy.model.path)
            dataset = PromptDataset(datasets=files, tokenizer=tokenizer, max_prompt_length=cfg.trainer.max_prompt_length, num_workers=1)
        dataset = prepare_dataset(dataset, env_spec, experiment, split)
        out[split] = {'rows': len(dataset), 'class': type(dataset).__name__}
        if split == 'train' and len(dataset) < cfg.trainer.train_batch_size:
            raise RuntimeError(f'train set has {len(dataset)} rows < train_batch_size {cfg.trainer.train_batch_size}')
    return out


def forwarded_env(environ: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in environ.items()
            if (key.startswith(FORWARD_PREFIXES) or key in FORWARD_KEYS) and key not in NEVER_FORWARD}


def main() -> None:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from skyrl.train.entrypoints.main_base import BasePPOExp
    from skyrl.train.utils import validate_cfg
    import skyrl.train.utils.utils as ray_utils

    from .experiment import current
    from .tasks import get_environment
    from .tracking import make_callback_class, make_tracking_class, make_trajectory_logger_class

    experiment = current()
    env_spec = get_environment(experiment['environment']['name'])
    AxisTracking = make_tracking_class()
    GroupTrajectoryLogger = make_trajectory_logger_class()
    EvidenceCallback = make_callback_class()

    class StrlExp(BasePPOExp):
        def get_tracker(self):
            return AxisTracking(project_name=self.cfg.trainer.project_name, experiment_name=self.cfg.trainer.run_name,
                                backend=self.cfg.trainer.logger, config=self.cfg, tags=self.cfg.trainer.tags)

        def get_train_dataset(self):
            custom = env_spec.build_dataset(experiment, self.cfg, 'train')
            return prepare_dataset(custom if custom is not None else super().get_train_dataset(), env_spec, experiment, 'train')

        def get_eval_dataset(self):
            custom = env_spec.build_dataset(experiment, self.cfg, 'validation')
            return prepare_dataset(custom if custom is not None else super().get_eval_dataset(), env_spec, experiment, 'validation')

        def get_trajectory_logger(self):
            return GroupTrajectoryLogger()

        def get_generator(self, cfg, tokenizer, inference_engine_client):
            custom = env_spec.build_generator(experiment, cfg, tokenizer, inference_engine_client)
            return custom if custom is not None else super().get_generator(cfg, tokenizer, inference_engine_client)

        def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg):
            sampler = env_spec.train_batch_sampler(experiment, train_dataset, cfg) if train_dataset is not None else None
            if sampler is None:   # every environment without a custom sampler: SkyRL's trainer, untouched
                trainer = super().get_trainer(cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg)
            else:
                from skyrl.train.trainer import RayPPOTrainer
                from .tasks.mimo.sampler import make_trainer_class
                trainer = make_trainer_class(RayPPOTrainer, sampler)(
                    cfg=cfg, tracker=tracker, tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                    inference_engine_client=inference_engine_client, generator=generator, colocate_pg=colocate_pg)
            trainer.add_callback(EvidenceCallback(float(os.getenv('GRPO_CHECKPOINT_MINUTES', '15'))))
            return trainer

    @ray.remote(num_cpus=1)
    def run(cfg):
        from .monitor import critic
        env_spec.register(experiment)            # e.g. strl_env for the generic reward path
        critic.configure(critic.settings_from_env(), modes=env_spec.critic_modes(experiment))
        StrlExp(cfg).run()

    cfg = env_spec.skyrl_config_class().from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    original_prepare = ray_utils.prepare_runtime_environment

    def prepare_with_services(config):
        env = original_prepare(config)
        for key in NEVER_FORWARD:
            env.pop(key, None)
        env.update(forwarded_env(dict(os.environ)))
        return env

    ray_utils.prepare_runtime_environment = prepare_with_services
    ray_utils.initialize_ray(cfg)
    ray.get(run.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        node_id=ray.get_runtime_context().get_node_id(), soft=False)).remote(cfg))


if __name__ == '__main__':
    main()
