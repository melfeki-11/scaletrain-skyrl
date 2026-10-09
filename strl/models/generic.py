"""Fallback profiles. Unverified: use for CPU dry runs / 1-GPU canaries before a real run, and add a dedicated profile
(see .claude/skills/scaletrain-rl-experiments/references/extending.md, "A new model") once an architecture needs special handling."""
from __future__ import annotations

from . import ModelProfile


class GenericProfile(ModelProfile):
    name = 'generic_causal_lm'
    verified = 'none'


class LlamaProfile(ModelProfile):
    name = 'llama'
    architectures = ('LlamaForCausalLM',)
    verified = 'preflight'
    notes = ('Llama architecture: no thinking toggle. The end-of-turn token comes from the chat template, not the architecture (Llama 3: '
             '<|eot_id|>; SmolLM2: <|im_end|>); the GSM8K scorer strips every known terminal marker (strl/tasks/gsm8k/answers.py).')
    thinking_flag = False


class DenseProfile(ModelProfile):
    """Dense decoder-only families (Qwen2, Mistral, OLMo-2, Phi-3/4, Gemma-2/3 text, Granite, Command-R7B, GLM-4): standard FSDP wrapping and
    vLLM serving. No thinking toggle. Their end-of-turn tokens differ from Qwen's (scorers strip the tokenizer's special tokens)."""
    name = 'dense'
    verified = 'none'
    notes = ('Dense decoder-only model: no architecture-specific overrides. Unverified on this platform: run a dry run and a 1-GPU canary; '
             '`strl check-template` tells whether Harbor step-wise turns merge.')
    thinking_flag = False
