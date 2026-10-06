"""The model axis. A ``ModelProfile`` holds everything that depends on the model architecture, so switching the LLM is a
YAML edit (``model.id``): SkyRL trainer/inference overrides, chat-template flags (thinking), offline-vLLM kwargs, export/reload
adaptation, and a parameter-count estimate for planning. The profile is chosen from the model's ``config.json``
(``architectures``), or forced with ``model.profile``.

Verification levels (shown by ``strl plan``; unverified profiles warn):
  gpu        - a full GRPO run completed and held-out improvement was measured on ScaleTrain
  canary     - GRPO steps ran on ScaleTrain, no held-out measurement
  preflight  - SkyRL config parsing + imports validated on CPU only
  none       - generic fallback; run a CPU dry run and a 1-GPU canary before trusting it
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ModelProfile:
    name = 'generic_causal_lm'
    architectures: tuple[str, ...] = ()
    verified = 'none'
    notes = 'Generic causal LM: no architecture-specific overrides. Validate with a dry run and a 1-GPU canary.'
    thinking_flag = False                 # supports chat_template_kwargs enable_thinking
    decoder_layer_cls: str | None = None  # FSDP wrap class; None lets SkyRL pick from _no_split_modules

    # ---- trainer / generator overrides (SkyRL dotted keys)
    def trainer_overrides(self, cfg: dict) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.decoder_layer_cls:
            out['trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap'] = [self.decoder_layer_cls]
            out['trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap'] = [self.decoder_layer_cls]
        return out

    def generator_overrides(self, cfg: dict) -> dict[str, Any]:
        return {}

    def chat_kwargs(self, cfg: dict) -> dict[str, Any]:
        """Extra kwargs for tokenizer.apply_chat_template / the generator (e.g. enable_thinking=False)."""
        return {'enable_thinking': False} if self.thinking_flag else {}

    def vllm_kwargs(self, cfg: dict) -> dict[str, Any]:
        """Extra kwargs for the offline vLLM engine used by the final evaluation."""
        return {}

    def prepare_export_for_vllm(self, base_path: str, export_dir: Path, target_dir: Path) -> Path:
        """SkyRL HF export -> a directory vLLM can reload. Default: the export is already loadable."""
        return export_dir

    def param_estimate_b(self, hf_config: dict) -> float | None:
        """Rough parameter count in billions from config.json (dense transformer formula); None if unknown."""
        text = hf_config.get('text_config', hf_config)
        try:
            layers, hidden, vocab = text['num_hidden_layers'], text['hidden_size'], text['vocab_size']
            inter = text.get('intermediate_size', 4 * hidden)
            kv = text.get('num_key_value_heads', text.get('num_attention_heads', 1)) * text.get('head_dim', hidden // text['num_attention_heads'])
            attn = 2 * hidden * hidden + 2 * hidden * kv
            mlp = 3 * hidden * inter
            embed = vocab * hidden * (1 if text.get('tie_word_embeddings', hf_config.get('tie_word_embeddings', False)) else 2)
            return (layers * (attn + mlp) + embed) / 1e9
        except (KeyError, ZeroDivisionError):
            return None


def served_name(model_id: str) -> str:
    """SkyRL/Harbor require served_model_name without '/'."""
    return model_id.split('/')[-1]


_PROFILES: dict[str, str] = {
    'qwen3_5': 'strl.models.qwen3_5:Qwen35Profile',
    'qwen3': 'strl.models.qwen3:Qwen3Profile',
    'generic_causal_lm': 'strl.models.generic:GenericProfile',
    'llama': 'strl.models.generic:LlamaProfile',
}
_ARCH_TO_PROFILE = {
    'Qwen3_5ForConditionalGeneration': 'qwen3_5', 'Qwen3_5ForCausalLM': 'qwen3_5',
    'Qwen3ForCausalLM': 'qwen3', 'LlamaForCausalLM': 'llama',
}


def register_profile(name: str, target: str, architectures: tuple[str, ...] = ()) -> None:
    """Register a model profile ('module:Class'); route config.json architectures to it."""
    _PROFILES[name] = target
    for arch in architectures:
        _ARCH_TO_PROFILE[arch] = name


def get_profile(name: str) -> ModelProfile:
    import importlib
    if name not in _PROFILES:
        raise KeyError(f'unknown model profile {name!r}; registered: {sorted(_PROFILES)}')
    module, _, cls = _PROFILES[name].partition(':')
    return getattr(importlib.import_module(module), cls)()


def read_hf_config(model_cfg: dict) -> dict | None:
    """config.json from the staged local model, else from the HF hub (revision-pinned). None when unavailable offline."""
    local = Path(model_cfg.get('local_path', '')) / 'config.json'
    if local.is_file():
        return json.loads(local.read_text())
    try:
        from huggingface_hub import hf_hub_download
        return json.loads(Path(hf_hub_download(model_cfg['id'], 'config.json', revision=model_cfg.get('revision') or None)).read_text())
    except Exception:
        return None


def detect(model_cfg: dict, hf_config: dict | None = None) -> tuple[ModelProfile, dict | None]:
    """Choose the profile for a ``model`` section. Returns (profile, hf_config)."""
    forced = model_cfg.get('profile', 'auto')
    hf_config = hf_config if hf_config is not None else read_hf_config(model_cfg)
    if forced != 'auto':
        return get_profile(forced), hf_config
    if hf_config is None:
        raise RuntimeError(f"cannot read config.json for {model_cfg['id']} (offline?); set model.profile explicitly or stage the model")
    for arch in hf_config.get('architectures') or []:
        if arch in _ARCH_TO_PROFILE:
            return get_profile(_ARCH_TO_PROFILE[arch]), hf_config
    return get_profile('generic_causal_lm'), hf_config
