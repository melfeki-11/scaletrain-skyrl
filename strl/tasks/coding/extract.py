"""Extract the candidate program from a completion: strip special tokens, take the LAST fenced python block."""
from __future__ import annotations

import re

SPECIAL = re.compile(r'<\|[a-z_]+\|>')
FENCE = re.compile(r'```(?:python|py|python3)?[ \t]*\n(.*?)```', re.S | re.I)
THINK = re.compile(r'<think>.*?</think>', re.S)


def extract_code(completion: str) -> str | None:
    text = THINK.sub('', SPECIAL.sub('', completion or ''))
    blocks = [b for b in FENCE.findall(text) if b.strip()]
    if not blocks:
        return None
    return blocks[-1].rstrip() + '\n'
