from __future__ import annotations

from pathlib import Path
from typing import Any


def encode(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return repr(value)


def overrides(
    cfg: dict[str, Any],
    *,
    stage: str,
    run_name: str,
    learning_rate: float | None = None,
    max_steps: int | None = None,
) -> list[str]:
    if stage not in {"canary", "train"}:
        raise ValueError("SkyRL stage must be canary or train")
    training = cfg["training"]
    inference = cfg["inference"]
    data_dir = Path(cfg["run"]["data_dir"])
    artifact_dir = Path(cfg["run"]["artifact_dir"]) / run_name
    steps = int(max_steps if max_steps is not None else (2 if stage == "canary" else training["max_steps"]))
    values = {
        "data.train_data": [str(data_dir / "train.parquet")],
        "data.val_data": [str(data_dir / ("canary.parquet" if stage == "canary" else "validation.parquet"))],
        "data.dataloader.num_workers": 0,
        "trainer.algorithm.advantage_estimator": "grpo",
        "trainer.algorithm.loss_reduction": "token_mean",
        "trainer.algorithm.zero_variance_filter": training["grpo"]["zero_variance_filter"],
        "trainer.algorithm.use_kl_loss": training["grpo"]["use_kl_loss"],
        "trainer.algorithm.kl_loss_coef": training["grpo"]["kl_coefficient"],
        "trainer.policy.model.path": cfg["model"]["local_path"],
        "trainer.policy.language_model_only": True,
        "trainer.ref.language_model_only": True,
        "trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer"],
        "trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer"],
        "trainer.placement.colocate_all": True,
        "trainer.strategy": "fsdp",
        "trainer.placement.policy_num_nodes": cfg["scale_train"]["num_instances"],
        "trainer.placement.policy_num_gpus_per_node": cfg['scale_train']['gpus_per_node'],
        "trainer.placement.ref_num_nodes": cfg["scale_train"]["num_instances"],
        "trainer.placement.ref_num_gpus_per_node": cfg['scale_train']['gpus_per_node'],
        "trainer.epochs": 20,
        "trainer.max_training_steps": steps,
        "trainer.seed": cfg["run"]["seed"],
        "generator.batched": False,
        "trainer.train_batch_size": training["train_batch_size"],
        "trainer.policy_mini_batch_size": training["policy_mini_batch_size"],
        "trainer.micro_train_batch_size_per_gpu": training["micro_train_batch_size_per_gpu"],
        "trainer.micro_forward_batch_size_per_gpu": training["micro_forward_batch_size_per_gpu"],
        "trainer.policy.optimizer_config.lr": learning_rate or training["learning_rate"],
        "trainer.update_epochs_per_batch": training["update_epochs_per_batch"],
        "trainer.max_prompt_length": training["max_prompt_tokens"],
        "trainer.bf16": training["bf16"],
        "trainer.remove_microbatch_padding": False,
        "trainer.eval_batch_size": 256,
        "trainer.eval_before_train": True,
        "trainer.eval_interval": 1 if stage == "canary" else cfg["evaluation"]["interval_steps"],
        "trainer.ckpt_interval": 1 if stage == "canary" else cfg["evaluation"]["checkpoint_interval_steps"],
        "trainer.hf_save_interval": 1 if stage == "canary" else cfg["evaluation"]["checkpoint_interval_steps"],
        "trainer.max_ckpts_to_keep": cfg["evaluation"]["max_checkpoints"],
        "trainer.resume_mode": None,
        "trainer.ckpt_path": str(artifact_dir / "checkpoints"),
        "trainer.export_path": str(artifact_dir / "exports"),
        "trainer.log_path": str(artifact_dir / "logs"),
        "trainer.logger": "wandb",
        "trainer.project_name": cfg["wandb"]["project"],
        "trainer.run_name": run_name,
        "trainer.tags": ["scale-train", "grpo", "gsm8k", cfg['scale_train']['team']],
        "trainer.dump_eval_results": True,
        "trainer.print_example_interval": 1,
        "trainer.num_logger_eval_samples": cfg["evaluation"]["fixed_qualitative_samples"],
        "trainer.num_logger_train_samples": training["grpo"]["samples_per_prompt"] * cfg["evaluation"]["training_groups_to_log"],
        "generator.n_samples_per_prompt": training["grpo"]["samples_per_prompt"],
        "generator.eval_n_samples_per_prompt": 1,
        "generator.sampling_params.temperature": training["generation"]["temperature"],
        "generator.sampling_params.top_p": training["generation"]["top_p"],
        "generator.sampling_params.max_generate_length": training["max_response_tokens"],
        "generator.eval_sampling_params.temperature": 0.0,
        "generator.eval_sampling_params.top_p": training["generation"]["top_p"],
        "generator.eval_sampling_params.max_generate_length": training["max_response_tokens"],
        "trainer.enable_ray_gpu_monitor": True,
        "generator.chat_template_kwargs": {"enable_thinking": False},
        "generator.zero_reward_on_non_stop": True,
        "generator.inference_engine.backend": "vllm",
        "generator.inference_engine.run_engines_locally": True,
        "generator.inference_engine.weight_sync_backend": "nccl",
        "generator.inference_engine.num_engines": inference["num_engines"],
        "generator.inference_engine.tensor_parallel_size": inference["tensor_parallel_size"],
        "generator.inference_engine.served_model_name": "Qwen3.5-0.8B",
        "generator.inference_engine.language_model_only": True,
        "generator.inference_engine.gpu_memory_utilization": inference["gpu_memory_utilization"],
        "generator.inference_engine.max_num_seqs": inference["max_num_seqs"],
        "generator.inference_engine.engine_init_kwargs.disable_custom_all_reduce": True,
        "generator.inference_engine.engine_init_kwargs.enforce_eager": inference["enforce_eager"],
        "generator.inference_engine.engine_init_kwargs.gdn_prefill_backend": inference["gdn_prefill_backend"],
        "generator.inference_engine.engine_init_kwargs.language_model_only": True,
        "generator.inference_engine.engine_init_kwargs.max_model_len": training["max_prompt_tokens"] + training["max_response_tokens"],
        "environment.env_class": "gsm8k",
    }
    return [f"{key}={encode(value)}" for key, value in values.items()]
