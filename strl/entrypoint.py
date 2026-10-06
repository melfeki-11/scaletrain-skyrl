"""SkyRL entrypoint for every experiment: generic tracker/logger/callback + the environment's generator and env registration."""
from __future__ import annotations

import os
import sys

# Forwarded into every Ray actor's runtime env. Raw Modal tokens and LiteLLM keys are NEVER forwarded (pinned SkyRL would
# otherwise copy MODAL_TOKEN_ID/SECRET into runtime_env); actors get file paths (MODAL_CONFIG_PATH, STRL_LITELLM_KEY_FILE).
FORWARD_PREFIXES = ('WANDB_', 'GRPO_', 'NCCL_', 'GLOO_', 'STRL_')
FORWARD_KEYS = {'PYTHONPATH', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'MODAL_CONFIG_PATH', 'MODAL_PROFILE', 'MODAL_ENVIRONMENT'}
NEVER_FORWARD = {'MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET', 'LITELLM_PROXY_API_KEY', 'DAYTONA_API_KEY'}


def forwarded_env(environ: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in environ.items()
            if (key.startswith(FORWARD_PREFIXES) or key in FORWARD_KEYS) and key not in NEVER_FORWARD}


def main() -> None:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from skyrl.train.config import SkyRLTrainConfig
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

        def get_trajectory_logger(self):
            return GroupTrajectoryLogger()

        def get_generator(self, cfg, tokenizer, inference_engine_client):
            custom = env_spec.build_generator(experiment, cfg, tokenizer, inference_engine_client)
            return custom if custom is not None else super().get_generator(cfg, tokenizer, inference_engine_client)

        def get_trainer(self, *args, **kwargs):
            trainer = super().get_trainer(*args, **kwargs)
            trainer.add_callback(EvidenceCallback(float(os.getenv('GRPO_CHECKPOINT_MINUTES', '15'))))
            return trainer

    @ray.remote(num_cpus=1)
    def run(cfg):
        from .monitor import critic
        env_spec.register(experiment)            # e.g. strl_env for the generic reward path
        critic.configure(critic.settings_from_env())
        StrlExp(cfg).run()

    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
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
