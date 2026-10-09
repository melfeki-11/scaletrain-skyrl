"""Countdown reward components (``reward.components[].fn``). All consume the scorer result of ``CountdownSpec.score_result``."""
from __future__ import annotations


def countdown_correct(result: dict, ctx) -> float:
    """1.0 iff the last <answer> is a valid expression using each given number at most once and equal to the target (the yardstick)."""
    return 1.0 if result.get('correct') else 0.0


def countdown_closeness(result: dict, ctx, scale: float = 1.0) -> float:
    """Partial credit for a VALID, LEGAL expression that misses: 1 - |value - target| / max(target, 1), floored at 0 (and 0 when correct,
    so a weighted sum never pays a hit twice). Invalid syntax or illegal numbers earn nothing, so it cannot be gamed by ignoring the rules."""
    if result.get('correct') or not result.get('legal_numbers') or result.get('distance') is None:
        return 0.0
    target = float(result.get('target') or 0.0)
    return max(0.0, 1.0 - float(result['distance']) / max(abs(target), 1.0)) * float(scale)


FUNCTIONS = {'countdown_correct': countdown_correct, 'countdown_closeness': countdown_closeness}
