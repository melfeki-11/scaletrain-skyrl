"""Does this model's chat template keep step-wise multi-turn trajectories mergeable?

SkyRL (HarborGenerator, step-wise GRPO) trains one sequence per agent turn unless turn i+1's prompt tokens start with
`prompt_i + response_i` exactly, in which case it merges them into one sequence (generators/utils.py `_merge_single_trajectory`).
Chat templates that render past assistant turns differently from how the generation prompt looked break that: e.g. Qwen3.5 drops
`<think>\\n\\n</think>\\n\\n` from history while the generation prompt ends with it, so nothing ever merges (measured on 811 real
Terminus-2 turns: 811 sequences); Qwen3.8's template keeps it (209 sequences, 86% of merge points merge).

`prefix_property` is the cheap, model-agnostic probe (string level, one synthetic conversation); it is what `strl plan` and
`strl check-template` report. The authoritative number is measured on real transcripts (see references/extending.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def prefix_property(tokenizer: Any, chat_kwargs: dict | None = None) -> dict:
    """{'ok': bool, 'detail': str}: is `prompt_1 + reply + eos` a prefix of the prompt rendered after a user follow-up?"""
    kw = dict(chat_kwargs or {})
    user1 = [{'role': 'user', 'content': 'first question'}]
    reply = '{"analysis": "ok", "commands": []}'
    follow = user1 + [{'role': 'assistant', 'content': reply}, {'role': 'user', 'content': 'terminal output'}]
    try:
        p1 = tokenizer.apply_chat_template(user1, tokenize=False, add_generation_prompt=True, **kw)
        p2 = tokenizer.apply_chat_template(follow, tokenize=False, add_generation_prompt=True, **kw)
    except Exception as error:   # templates that reject the conversation shape (tool-only, system required, ...)
        return {'ok': None, 'detail': f'template could not render the probe conversation: {type(error).__name__}: {str(error)[:120]}'}
    expected = p1 + reply + (getattr(tokenizer, 'eos_token', None) or '')
    if p2.startswith(expected):
        return {'ok': True, 'detail': 'history keeps the generation prompt and reply verbatim: turns can merge'}
    k = next((i for i, (a, b) in enumerate(zip(expected, p2)) if a != b), min(len(expected), len(p2)))
    return {'ok': False, 'detail': f'history diverges from the generated prefix at char {k}: generated {expected[max(0, k - 24):k + 12]!r} '
                                   f'vs rendered {p2[max(0, k - 24):k + 12]!r}'}


def load_tokenizer(model_cfg: dict, *, online: bool = False) -> Any | None:
    """The model's tokenizer from model.local_path, the HF cache, or (online=True) the hub at the pinned revision; None if unavailable."""
    from transformers import AutoTokenizer
    sources = []
    local = Path(model_cfg.get('local_path', ''))
    if (local / 'tokenizer_config.json').is_file():
        sources.append((str(local), {}))
    sources.append((model_cfg['id'], {'revision': model_cfg.get('revision') or None, 'local_files_only': not online}))
    for name, kw in sources:
        try:
            return AutoTokenizer.from_pretrained(name, **kw)
        except Exception:
            continue
    return None


def check(cfg: dict, *, online: bool = False) -> dict | None:
    """prefix_property under the experiment's model profile chat kwargs (e.g. enable_thinking=False); None if no tokenizer is available."""
    from . import detect
    tokenizer = load_tokenizer(cfg['model'], online=online)
    if tokenizer is None:
        return None
    profile = detect(cfg['model'])[0]
    return dict(prefix_property(tokenizer, profile.chat_kwargs(cfg)), model=cfg['model']['id'], profile=profile.name)
