"""The offline vLLM engine the final and retention evaluations use, for a plain model directory OR a LoRA adapter.

A LoRA adapter is served by vLLM on top of the bf16 base (enable_lora + LoRARequest), exactly as the in-loop evaluation and the training
rollouts see it. Merging the adapter into bf16 weights first (the previous path) rounds most of a small RL update away: with |delta| ~ 5e-5
about 40% of the touched weights did not change at all and the delta had ~39% relative error, so the offline score described a model
closer to the base than the one that was trained.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def lora_rank(adapter: str | Path) -> int:
    return int(json.loads((Path(adapter) / 'adapter_config.json').read_text()).get('r') or 16)


def build(cfg: dict, profile: Any, model: str | Path, *, max_model_len: int, gpu_memory_utilization: float = 0.75) -> tuple[Any, Any]:
    """(engine, lora_request or None). ``model`` is a model directory, or a PEFT adapter directory (served on cfg.model.local_path)."""
    from vllm import LLM
    from .lora import is_adapter
    kwargs = dict(tokenizer=cfg['model']['local_path'], tensor_parallel_size=1, dtype=cfg['model']['dtype'],
                  enforce_eager=cfg['inference']['enforce_eager'], gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                  max_num_seqs=128, disable_log_stats=True, **profile.vllm_kwargs(cfg))
    if is_adapter(model):
        from vllm.lora.request import LoRARequest
        rank = lora_rank(model)
        engine = LLM(model=cfg['model']['local_path'], enable_lora=True, max_lora_rank=max(8, rank), max_loras=1, **kwargs)
        return engine, LoRARequest('trained', 1, str(model))
    return LLM(model=str(model), **kwargs), None


def generate(engine: Any, lora_request: Any, prompts: list[str], params: Any) -> list:
    return engine.generate(prompts, params, lora_request=lora_request) if lora_request is not None else engine.generate(prompts, params)
