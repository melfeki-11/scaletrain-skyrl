"""Partial credit for a mathematically EQUIVALENT but non-canonical boxed answer (math_boxed; experiments/r3d-math-granite-equiv.yaml).

``equivalent_answer_credit(result, ctx, credit=0.3)`` pays ``credit`` when the exact-match yardstick said NO but the last boxed answer
evaluates to exactly the same number(s) as the reference: ``\\frac{6}{8}`` for ``\\frac{3}{4}``, ``0.75`` for ``\\frac{3}{4}``,
``\\sqrt{8}`` for ``2\\sqrt{2}``, ``\\frac{1}{\\sqrt{3}}`` for ``\\frac{\\sqrt{3}}{3}``, ``(2, 6/2)`` for ``(2, 3)``.

Why: exact match after Hendrycks normalization can only under-count (strl/tasks/math_boxed/scoring.py), so a 0/1 reward scores
"right math, unsimplified form" the same as a wrong answer. In a group with no canonical hit that leaves zero advantage; this term turns
such groups into signal toward the right mathematics, while a canonical answer (1.0) always ranks well above it (credit < 1).

Anti-gaming by construction:
- never paid on an exact hit (no double pay; the maximum training reward stays 1.0);
- never paid on a hedged completion (several distinct boxed answers: ``single_answer`` False);
- rationals are compared EXACTLY (``fractions.Fraction``): a truncated decimal (0.333) is not 1/3; a terminating decimal never equals an
  irrational value; irrational-vs-irrational values (surds, pi) must agree to 1e-12 relative (float rounding only);
- the parser is a small whitelist recursive-descent evaluator (numbers, + - * / ^, \\frac, \\sqrt, \\pi, \\cdot, \\times, parentheses,
  top-level comma lists): no ``eval``, length- and depth-bounded; anything else (variables, intervals, text) is "not evaluable" -> 0.
"""
from __future__ import annotations

import math
from fractions import Fraction

MAX_CHARS = 120
MAX_DEPTH = 24
MAX_EXPONENT = 12
REL_TOL = 1e-12

Value = Fraction | float


class _NotEvaluable(Exception):
    pass


class _Parser:
    """Evaluates a normalized answer string (spaces removed by ``scoring.normalize``) to a Fraction (exact) or a float (irrational)."""

    def __init__(self, text: str):
        self.s, self.i, self.depth = text, 0, 0

    def peek(self, token: str) -> bool:
        return self.s.startswith(token, self.i)

    def take(self, token: str) -> bool:
        if self.peek(token):
            self.i += len(token)
            return True
        return False

    def expect(self, token: str) -> None:
        if not self.take(token):
            raise _NotEvaluable(token)

    def done(self) -> bool:
        return self.i >= len(self.s)

    def enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise _NotEvaluable('depth')

    # expr := term (('+'|'-') term)*
    def expr(self) -> Value:
        self.enter()
        value = self.term()
        while True:
            if self.take('+'):
                value = value + self.term()
            elif self.peek('-'):
                self.i += 1
                value = value - self.term()
            else:
                break
        self.depth -= 1
        return value

    # term := unary (('*'|'\cdot'|'\times'|'/'|implicit) unary)*
    def term(self) -> Value:
        value = self.unary()
        while True:
            if self.take('*') or self.take('\\cdot') or self.take('\\times'):
                value = value * self.unary()
            elif self.take('/'):
                value = _div(value, self.unary())
            elif self._starts_atom():
                value = value * self.unary()      # implicit product: 2\sqrt{3}, 3\pi, 2(5)
            else:
                return value

    def unary(self) -> Value:
        if self.take('-'):
            return -self.unary()
        if self.take('+'):
            return self.unary()
        return self.power()

    def power(self) -> Value:
        base = self.atom()
        if self.take('^'):
            exponent = self.braced() if self.peek('{') else self.atom()
            return _pow(base, exponent)
        return base

    def _starts_atom(self) -> bool:
        if self.done():
            return False
        c = self.s[self.i]
        return c.isdigit() or c in '.({' or self.peek('\\frac') or self.peek('\\sqrt') or self.peek('\\pi')

    def braced(self) -> Value:
        self.expect('{')
        value = self.expr()
        self.expect('}')
        return value

    def atom(self) -> Value:
        if self.take('\\frac'):
            return _div(self.braced(), self.braced())
        if self.take('\\sqrt'):
            index = None
            if self.take('['):
                index = self.expr()
                self.expect(']')
            return _root(self.braced(), index)
        if self.take('\\pi'):
            return math.pi
        if self.take('('):
            value = self.expr()
            self.expect(')')
            return value
        if self.peek('{'):
            return self.braced()
        start = self.i
        while not self.done() and (self.s[self.i].isdigit() or self.s[self.i] == '.'):
            self.i += 1
        token = self.s[start:self.i]
        if not token or token.count('.') > 1 or token == '.':
            raise _NotEvaluable(token)
        return Fraction(token)


def _div(a: Value, b: Value) -> Value:
    if b == 0:
        raise _NotEvaluable('division by zero')
    return a / b


def _pow(base: Value, exponent: Value) -> Value:
    if isinstance(exponent, Fraction) and exponent.denominator == 1 and abs(exponent) <= MAX_EXPONENT:
        if base == 0 and exponent < 0:
            raise _NotEvaluable('0^-n')
        return base ** int(exponent)
    if isinstance(exponent, Fraction) and exponent.denominator in (2, 3) and abs(exponent) <= MAX_EXPONENT:
        return _pow(_root(base, Fraction(exponent.denominator)), Fraction(exponent.numerator))
    raise _NotEvaluable('exponent')


def _exact_root(n: int, k: int) -> int | None:
    if n < 0:
        return None
    r = round(n ** (1.0 / k))
    for cand in (r - 1, r, r + 1):
        if cand >= 0 and cand ** k == n:
            return cand
    return None


def _root(value: Value, index: Value | None) -> Value:
    k = Fraction(2) if index is None else index
    if not (isinstance(k, Fraction) and k.denominator == 1 and 2 <= k <= 6):
        raise _NotEvaluable('root index')
    k = int(k)
    if isinstance(value, Fraction):
        if value < 0:
            if k % 2 == 0:
                raise _NotEvaluable('complex')
            return -_root(-value, Fraction(k))
        num, den = _exact_root(value.numerator, k), _exact_root(value.denominator, k)
        if num is not None and den is not None:
            return Fraction(num, den)
    v = float(value)
    if v < 0:
        if k % 2 == 0:
            raise _NotEvaluable('complex')
        return -((-v) ** (1.0 / k))
    return v ** (1.0 / k)


def _split_top_level(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for k, c in enumerate(text):
        if c in '({[':
            depth += 1
        elif c in ')}]':
            depth -= 1
        elif c == ',' and depth == 0:
            parts.append(text[start:k])
            start = k + 1
    parts.append(text[start:])
    return parts


def evaluate(answer: str | None) -> tuple[Value, ...] | None:
    """The value(s) of a normalized answer (a top-level comma list or one bracketed ordered pair/tuple), or None if not evaluable."""
    if not answer or len(answer) > MAX_CHARS:
        return None
    text = answer
    if text.startswith('(') and text.endswith(')') and len(_split_top_level(text[1:-1])) > 1:
        text = text[1:-1]                                  # "(2,3)" -> a tuple; "(2)" stays one parenthesized value
    values = []
    for part in _split_top_level(text):
        parser = _Parser(part)
        try:
            value = parser.expr()
        except (_NotEvaluable, ZeroDivisionError, OverflowError, ValueError, RecursionError):
            return None
        if not parser.done():
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        values.append(value)
    return tuple(values)


def _same(a: Value, b: Value) -> bool:
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        return a == b
    if isinstance(a, Fraction) != isinstance(b, Fraction):
        return False        # exact rational vs irrational: never equal (blocks decimal approximations of surds / pi)
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=REL_TOL)


def equivalent(answer: str | None, truth: str | None) -> bool:
    """Both normalized answers evaluate to the same ordered value list."""
    a, b = evaluate(answer), evaluate(truth)
    return a is not None and b is not None and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))


def equivalent_answer_credit(result: dict, ctx, credit: float = 0.3) -> float:
    """``credit`` for an un-hedged, non-exact boxed answer whose value equals the reference's; 0 otherwise (and 0 on an exact hit)."""
    if result.get('correct') or result.get('parsed') is None or not result.get('single_answer'):
        return 0.0
    if not 0.0 <= float(credit) < 1.0:
        raise ValueError('credit must be in [0, 1): an equivalent form must rank below the canonical answer')
    return float(credit) if equivalent(result.get('normalized'), result.get('truth_normalized')) else 0.0
