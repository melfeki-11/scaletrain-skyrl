"""math_boxed reward components (``reward.components[].fn``). All consume the scorer result of ``MathBoxedSpec.score_result``."""
from __future__ import annotations


def math_correct(result: dict, ctx) -> float:
    """1.0 iff the LAST \\boxed{} answer equals the reference after normalization (the yardstick; = the task metric)."""
    return 1.0 if result.get('correct') else 0.0


def math_format(result: dict, ctx, single: bool = True) -> float:
    """Format term: 1.0 when the completion commits to a non-empty \\boxed{} answer; with ``single`` (default) only when every box in it
    holds the same answer, so hedging with several boxed candidates earns no format credit. Paid on wrong answers too (that is its
    job: it teaches the output contract), so weight it well below ``math_correct`` (e.g. 0.1): a correct answer always ranks first."""
    if result.get('parsed') is None:
        return 0.0
    if single and not result.get('single_answer'):
        return 0.0
    return 1.0


FUNCTIONS = {'math_correct': math_correct, 'math_format': math_format}
