"""Coding reward components (``reward.components[].fn``). ``result`` is the executor verdict for one rollout:
{failure_class, passed, total, has_code, compiled, suspicious, ...}. The task metric (pass@1 = all tests pass) is reported
separately by the environment, so every design below stays comparable on the same held-out yardstick."""
from __future__ import annotations


def pass_all_tests(result: dict, ctx) -> float:
    """1.0 iff every test passed (the task metric itself; the default reward)."""
    return 1.0 if result.get('failure_class') == 'ok' else 0.0


def partial_tests(result: dict, ctx) -> float:
    """Fraction of tests passed (denser signal; MBPP has 3 tests per task, HumanEval check() counts as one)."""
    total = result.get('total') or 0
    return float(result.get('passed', 0)) / total if total else 0.0


def compiles(result: dict, ctx) -> float:
    """1.0 when a code block was found and it compiled (format/syntax shaping)."""
    return 1.0 if result.get('compiled') else 0.0


def no_suspicious_code(result: dict, ctx, penalty: float = 1.0) -> float:
    """-penalty when the program uses exit/__eq__/builtins/frame tricks (weight it positively to penalise)."""
    return -float(penalty) if result.get('suspicious') else 0.0


FUNCTIONS = {'pass_all_tests': pass_all_tests, 'partial_tests': partial_tests, 'compiles': compiles, 'no_suspicious_code': no_suspicious_code}
