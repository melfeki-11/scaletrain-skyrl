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


class Qwen3MoeProfile(Qwen3Profile):
    """Qwen3 mixture-of-experts (e.g. Qwen3-30B-A3B): the dense Qwen3 settings (thinking off) plus the MoE warning. Unverified."""
    name = 'qwen3_moe'
    architectures = ('Qwen3MoeForCausalLM',)
    verified = 'none'
    moe = True
    notes = ('Qwen3 MoE: thinking is switched off like dense Qwen3; experts are FSDP-sharded as dense weights (no expert parallelism). '
             'Unverified: dry run + canary first.')
