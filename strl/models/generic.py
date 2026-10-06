"""Fallback profiles. Unverified: use for CPU dry runs / 1-GPU canaries before a real run, and add a dedicated profile
(see references/models.md) once an architecture needs special handling."""
from __future__ import annotations

from . import ModelProfile


class GenericProfile(ModelProfile):
    name = 'generic_causal_lm'
    verified = 'none'


class LlamaProfile(ModelProfile):
    name = 'llama'
    architectures = ('LlamaForCausalLM',)
    verified = 'preflight'
    notes = 'Dense Llama: no thinking toggle; terminal EOS is <|eot_id|> (pass it to scorers that strip terminal markers).'
    thinking_flag = False
