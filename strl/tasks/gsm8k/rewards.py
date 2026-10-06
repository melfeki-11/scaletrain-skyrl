"""GSM8K reward components (``reward.components[].fn``). All consume the scorer result of ``Gsm8kSpec.score_result``."""
from __future__ import annotations


def strict_answer(result: dict, ctx) -> float:
    """1.0 iff the completion ends with exactly one ``#### <number>`` line equal to the reference (the verified reward)."""
    return 1.0 if result.get('correct') else 0.0


def contains_number_answer(result: dict, ctx) -> float:
    """Shaping term: 1.0 when any parseable ``#### <number>`` final line exists, even if wrong (format credit)."""
    return 1.0 if result.get('parsed') is not None else 0.0


FUNCTIONS = {'strict_answer': strict_answer, 'contains_number_answer': contains_number_answer}
