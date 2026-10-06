from __future__ import annotations

import os
import sys


def main() -> None:
    import ray
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.entrypoints.main_base import BasePPOExp
    from skyrl.train.utils import validate_cfg
    import skyrl.train.utils.utils as ray_utils
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from .tracking import make_callback_class, make_tracking_class, make_trajectory_logger_class

    AxisTracking = make_tracking_class()
    GroupTrajectoryLogger = make_trajectory_logger_class()
    EvidenceCallback = make_callback_class()

    class MiniExp(BasePPOExp):
        def get_tracker(self):
            return AxisTracking(
                project_name=self.cfg.trainer.project_name,
                experiment_name=self.cfg.trainer.run_name,
                backend=self.cfg.trainer.logger,
                config=self.cfg,
                tags=self.cfg.trainer.tags,
            )

        def get_trajectory_logger(self):
            return GroupTrajectoryLogger(columns=GroupTrajectoryLogger.columns)

        def get_trainer(self, *args, **kwargs):
            trainer = super().get_trainer(*args, **kwargs)
            trainer.add_callback(EvidenceCallback(float(os.getenv("GRPO_CHECKPOINT_MINUTES", "15"))))
            return trainer

    @ray.remote(num_cpus=1)
    def run(cfg):
        MiniExp(cfg).run()

    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    original_prepare = ray_utils.prepare_runtime_environment
    def prepare_with_services(config):
        env = original_prepare(config)
        env.update({key: value for key, value in os.environ.items()
                    if key.startswith(("WANDB_", "GRPO_", "NCCL_", "GLOO_"))
                    or key in {"PYTHONPATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}})
        return env
    ray_utils.prepare_runtime_environment = prepare_with_services
    ray_utils.initialize_ray(cfg)
    ray.get(run.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        node_id=ray.get_runtime_context().get_node_id(), soft=False)).remote(cfg))


if __name__ == "__main__":
    main()

