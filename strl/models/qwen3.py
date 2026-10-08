"""Qwen3 dense models. Canary-verified: Qwen3-1.7B GRPO through the Harbor/Modal loop on 1 H100 (2 real updates).

Plain FSDP wrapping (SkyRL derives the decoder class from the model), sample packing allowed, thinking toggle supported.
"""
from __future__ import annotations

from . import ModelProfile


class Qwen3Profile(ModelProfile):
    name = 'qwen3'
    architectures = ('Qwen3ForCausalLM',)
    verified = 'canary'
    notes = 'Dense Qwen3: standard FSDP + vLLM. Qwen3 history templates strip <think> blocks, so Harbor step-wise merging falls back to per-turn sequences.'
    thinking_flag = True
