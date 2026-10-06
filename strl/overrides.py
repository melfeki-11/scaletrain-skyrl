"""Experiment -> SkyRL dotted overrides.

The set is assembled from three independent sources, which is what makes model / environment / reward swappable:
  common (algorithm, placement, batching, eval cadence, logging)  +  model profile  +  environment spec.
For Qwen3.5 x GSM8K x the default reward this reproduces the GPU-verified override set of commit 2916fb9 exactly
(tests/test_gsm8k_golden.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import models
from .tasks import get_environment


def encode(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return 'null'
    if isinstance(value, str):
        return value
    return repr(value)


def skyrl_values(cfg: dict[str, Any], *, stage: str, run_name: str, learning_rate: float | None = None,
                 max_steps: int | None = None, profile: models.ModelProfile | None = None) -> dict[str, Any]:
    if stage not in {'canary', 'train'}:
        raise ValueError('SkyRL stage must be canary or train')
    a, inf, ev, comp = cfg['algorithm'], cfg['inference'], cfg['evaluation'], cfg['compute']
    env = get_environment(cfg['environment']['name'])
    profile = profile or models.detect(cfg['model'])[0]
    artifact_dir = Path(cfg['run']['artifact_dir']) / run_name
    steps = int(max_steps if max_steps is not None else (2 if stage == 'canary' else a['max_steps']))
    placement = cfg['derived']
    train_files, val_files = env.data_files(cfg, stage)
    values: dict[str, Any] = {
        'data.train_data': train_files,
        'data.val_data': val_files,
        'data.dataloader.num_workers': 0,
        'trainer.algorithm.advantage_estimator': a['estimator'],
        'trainer.algorithm.loss_reduction': 'token_mean',
        'trainer.algorithm.zero_variance_filter': a['zero_variance_filter'],
        'trainer.algorithm.use_kl_loss': a['use_kl_loss'],
        'trainer.algorithm.kl_loss_coef': a['kl_coef'],
        'trainer.policy.model.path': cfg['model']['local_path'],
        **profile.trainer_overrides(cfg),
        'trainer.placement.colocate_all': True,
        'trainer.strategy': 'fsdp',
        'trainer.placement.policy_num_nodes': comp['nodes'],
        'trainer.placement.policy_num_gpus_per_node': comp['gpus_per_node'],
        'trainer.placement.ref_num_nodes': comp['nodes'],
        'trainer.placement.ref_num_gpus_per_node': comp['gpus_per_node'],
        'trainer.epochs': 20,
        'trainer.max_training_steps': steps,
        'trainer.seed': a['seed'],
        'generator.batched': False,
        'trainer.train_batch_size': a['prompts_per_step'],
        'trainer.policy_mini_batch_size': a['minibatch_prompts'],
        'trainer.micro_train_batch_size_per_gpu': a['micro_train_per_gpu'],
        'trainer.micro_forward_batch_size_per_gpu': a['micro_forward_per_gpu'],
        'trainer.policy.optimizer_config.lr': learning_rate or a['lr'],
        'trainer.update_epochs_per_batch': a['update_epochs'],
        'trainer.max_prompt_length': a['max_prompt_tokens'],
        'trainer.bf16': a['bf16'],
        'trainer.eval_batch_size': ev['eval_batch_size'],
        'trainer.eval_before_train': True,
        'trainer.eval_interval': 1 if stage == 'canary' else ev['interval_steps'],
        'trainer.ckpt_interval': 1 if stage == 'canary' else ev['checkpoint_interval_steps'],
        'trainer.hf_save_interval': 1 if stage == 'canary' else ev['checkpoint_interval_steps'],
        'trainer.max_ckpts_to_keep': ev['max_checkpoints'],
        'trainer.resume_mode': None,
        'trainer.ckpt_path': str(artifact_dir / 'checkpoints'),
        'trainer.export_path': str(artifact_dir / 'exports'),
        'trainer.log_path': str(artifact_dir / 'logs'),
        'trainer.logger': 'wandb',
        'trainer.project_name': cfg['tracking']['project'],
        'trainer.run_name': run_name,
        'trainer.tags': ['scale-train', a['estimator'], *env.tags(cfg), comp['team'], *cfg['tracking'].get('tags', [])],
        'trainer.dump_eval_results': True,
        'trainer.print_example_interval': 1,
        'trainer.num_logger_eval_samples': ev['fixed_qualitative_samples'],
        'trainer.num_logger_train_samples': a['n_samples'] * ev['training_groups_to_log'],
        'generator.n_samples_per_prompt': a['n_samples'],
        'generator.eval_n_samples_per_prompt': 1,
        'generator.sampling_params.temperature': a['temperature'],
        'generator.sampling_params.top_p': a['top_p'],
        'generator.sampling_params.max_generate_length': a['max_generate_length'],
        'generator.eval_sampling_params.temperature': a['eval_temperature'],
        'generator.eval_sampling_params.top_p': a['top_p'],
        'generator.eval_sampling_params.max_generate_length': a['max_generate_length'],
        'trainer.enable_ray_gpu_monitor': True,
    }
    chat = profile.chat_kwargs(cfg)
    if chat:
        values['generator.chat_template_kwargs'] = chat
    values.update({
        'generator.zero_reward_on_non_stop': a['zero_reward_on_non_stop'],
        'generator.inference_engine.backend': 'vllm',
        'generator.inference_engine.run_engines_locally': True,
        'generator.inference_engine.weight_sync_backend': 'nccl',
        'generator.inference_engine.num_engines': placement['num_engines'],
        'generator.inference_engine.tensor_parallel_size': inf['tensor_parallel_size'],
        'generator.inference_engine.served_model_name': models.served_name(cfg['model']['id']),
        'generator.inference_engine.gpu_memory_utilization': inf['gpu_memory_utilization'],
        'generator.inference_engine.max_num_seqs': inf['max_num_seqs'],
        'generator.inference_engine.engine_init_kwargs.enforce_eager': inf['enforce_eager'],
        'generator.inference_engine.engine_init_kwargs.max_model_len': a['max_prompt_tokens'] + a['max_generate_length'],
        **profile.generator_overrides(cfg),
        **env.skyrl_overrides(cfg, stage),
    })
    return values


def overrides(cfg: dict[str, Any], *, stage: str, run_name: str, learning_rate: float | None = None,
              max_steps: int | None = None, profile: models.ModelProfile | None = None) -> list[str]:
    values = skyrl_values(cfg, stage=stage, run_name=run_name, learning_rate=learning_rate, max_steps=max_steps, profile=profile)
    return [f'{key}={encode(value)}' for key, value in values.items()]
