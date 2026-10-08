"""Qwen3.5 (hybrid Gated-DeltaNet, VLM wrapper). GPU-verified: Qwen3.5-0.8B GRPO on GSM8K, +18.04 points (commit 2916fb9).

Every value here was previously hard-coded in q35/skyrl.py, q35/evaluate.py and q35/worker.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import ModelProfile


class Qwen35Profile(ModelProfile):
    name = 'qwen3_5'
    architectures = ('Qwen3_5ForConditionalGeneration', 'Qwen3_5ForCausalLM')
    verified = 'gpu'
    notes = ('Hybrid GDN layers do not support sample packing (remove_microbatch_padding=false); the VLM wrapper needs '
             'language_model_only for trainer/ref/inference; SkyRL exports a text-only config that vLLM needs wrapped again.')
    thinking_flag = True
    decoder_layer_cls = 'Qwen3_5DecoderLayer'

    def trainer_overrides(self, cfg: dict) -> dict[str, Any]:
        out = {'trainer.policy.language_model_only': True, 'trainer.ref.language_model_only': True}
        out.update(super().trainer_overrides(cfg))
        out['trainer.remove_microbatch_padding'] = False
        return out

    def generator_overrides(self, cfg: dict) -> dict[str, Any]:
        return {
            'generator.inference_engine.language_model_only': True,
            'generator.inference_engine.engine_init_kwargs.disable_custom_all_reduce': True,
            'generator.inference_engine.engine_init_kwargs.gdn_prefill_backend': 'triton',
            'generator.inference_engine.engine_init_kwargs.language_model_only': True,
        }

    def vllm_kwargs(self, cfg: dict) -> dict[str, Any]:
        return {'language_model_only': True, 'gdn_prefill_backend': 'triton'}

    def prepare_export_for_vllm(self, base_path: str, export_dir: Path, target_dir: Path) -> Path:
        if json.loads((export_dir / 'config.json').read_text()).get('model_type') != 'qwen3_5_text':
            return export_dir
        from .qwen3_5_export import adapt
        adapt(base_path, export_dir, target_dir)
        return target_dir
