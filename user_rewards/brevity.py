"""Brevity shaping that only pays on a correct answer (used by experiments/r3a-cd-phi4.yaml).

``short_correct_bonus(result, ctx, bonus, max_tokens)`` = bonus * max(0, 1 - tokens / max_tokens) when the scorer marked the answer
``correct``, else 0. Tokens are counted with ``ctx.count_tokens`` (the model's tokenizer in the pod; chars/4 locally). A wrong answer
can never earn it, so the policy cannot trade correctness for length; the maximum training reward is 1 + bonus.
"""
from __future__ import annotations


def short_correct_bonus(result: dict, ctx, bonus: float = 0.1, max_tokens: int | None = None) -> float:
    if not result.get('correct'):
        return 0.0
    limit = float(max_tokens or ctx.max_generate_length)
    tokens = float(ctx.count_tokens(result.get('completion') or ''))
    return float(bonus) * max(0.0, 1.0 - tokens / max(limit, 1.0))
