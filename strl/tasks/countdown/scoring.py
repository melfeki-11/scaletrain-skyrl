"""Countdown scorer: parse ``<answer>EXPR</answer>``, evaluate EXPR exactly and safely, check number usage and the target.

Pure functions (no I/O, no eval()): the expression is parsed with ``ast`` and only integer literals, ``+ - * /`` and parentheses
are accepted. Arithmetic is exact (``fractions.Fraction``), so ``7 / 2 * 2`` is 7 and division by zero is an invalid expression.
Each given number may be used at most once (multiset inclusion); unary minus is refused (it would conjure numbers).
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from fractions import Fraction
from typing import Any

from ..gsm8k.answers import strip_terminal_tokens

ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)
ALLOWED_CHARS = re.compile(r'^[0-9+\-*/()\s.]*$')
MAX_EXPR_CHARS = 200
_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}


class InvalidExpression(ValueError):
    pass


def parse_answer(text: str) -> str | None:
    """The content of the LAST ``<answer>...</answer>`` block (stripped), or None when there is none / it is empty."""
    matches = ANSWER_RE.findall(strip_terminal_tokens(text or ''))
    if not matches:
        return None
    expr = matches[-1].strip()
    if expr.endswith('='):           # tolerate a trailing '=' only; "= 24" after the expression is refused below
        expr = expr[:-1].strip()
    return expr or None


def _walk(node: ast.AST, used: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _walk(node.body, used)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        used.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _walk(node.left, used), _walk(node.right, used)
        if isinstance(node.op, ast.Div) and right == 0:
            raise InvalidExpression('division by zero')
        return _OPS[type(node.op)](left, right)
    raise InvalidExpression(f'disallowed syntax: {type(node).__name__}')


def evaluate(expr: str) -> tuple[Fraction, list[int]]:
    """(exact value, integer literals in order). Raises InvalidExpression for anything but integers, + - * /, parentheses."""
    if len(expr) > MAX_EXPR_CHARS:
        raise InvalidExpression('expression too long')
    if not ALLOWED_CHARS.match(expr):
        raise InvalidExpression('disallowed characters')
    if '**' in expr or '//' in expr:
        raise InvalidExpression('disallowed operator')
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as error:
        raise InvalidExpression(f'syntax error: {error.msg}') from None
    used: list[int] = []
    return _walk(tree, used), used


def uses_allowed_numbers(used: list[int], numbers: list[int]) -> bool:
    """Every literal is one of the given numbers and none is used more often than it was given."""
    have = Counter(int(n) for n in numbers)
    return all(have[n] >= c for n, c in Counter(used).items())


def load_truth(truth: Any) -> tuple[list[int], int]:
    """Ground truth is stored as a JSON string ``{"numbers": [...], "target": N}`` (string: survives parquet and SkyRL extras)."""
    if isinstance(truth, str):
        truth = json.loads(truth)
    return [int(n) for n in truth['numbers']], int(truth['target'])


def score_expression(text: str, numbers: list[int], target: int) -> dict:
    """The scorer result consumed by the reward components and the task metric."""
    expr = parse_answer(text)
    out: dict[str, Any] = {'parsed': expr, 'value': None, 'valid': False, 'legal_numbers': False, 'numbers_used': 0,
                           'distance': None, 'correct': False}
    if expr is None:
        return {**out, 'failure_class': 'no_answer'}
    try:
        value, used = evaluate(expr)
    except InvalidExpression:
        return {**out, 'failure_class': 'invalid_expression'}
    legal = uses_allowed_numbers(used, numbers)
    distance = float(abs(value - target))
    out.update(value=float(value), valid=True, legal_numbers=legal, numbers_used=len(used), distance=distance)
    if not used or not legal:
        return {**out, 'legal_numbers': False, 'failure_class': 'illegal_numbers'}
    if value != target:
        return {**out, 'failure_class': 'wrong_value'}
    return {**out, 'correct': True, 'failure_class': 'ok'}
