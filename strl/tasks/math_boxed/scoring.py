"""MATH scorer: extract the LAST ``\\boxed{...}`` answer (brace-matched), normalize it Hendrycks-style and compare exactly.

Pure functions (no I/O, no eval, no sympy): the normalization follows the MATH benchmark's ``is_equiv`` string canonicalization
(``\\dfrac`` -> ``\\frac``, no ``\\left``/``\\right``, no units/degrees/dollars/percent, ``\\frac12`` -> ``\\frac{1}{2}``, ``x = 5`` -> ``5``,
spaces removed) plus numeric canonicalization (``2.0`` == ``2``, ``1,000`` == ``1000``). Symbolic equivalence (``\\frac{2}{4}`` vs
``\\frac{1}{2}``) is deliberately NOT credited: exact match is the yardstick, and it can only under-count, never over-count.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from ..gsm8k.answers import strip_terminal_tokens

BOXED_RE = re.compile(r'\\(boxed|fbox)\b')
MAX_ANSWER_CHARS = 300
_TEXT_RE = re.compile(r'\\(?:text|textbf|textrm|mathrm|mathbf|mbox)\s*\{([^{}]*)\}')
_SPACING_RE = re.compile(r'\\(?:,|;|:|!|quad|qquad|displaystyle)')
_NUMBER_RE = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)')
_THOUSANDS_RE = re.compile(r'[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?')


def _braced(text: str, start: int) -> tuple[str, int] | None:
    """(content, end index after the closing brace) of the brace group opening at ``text[start] == '{'``; None if unbalanced."""
    depth = 0
    for k in range(start, len(text)):
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:k], k + 1
    return None


def boxed_answers(text: str) -> list[str]:
    """Contents of every top-level ``\\boxed{..}`` / ``\\fbox{..}`` (and ``\\boxed 5``) in order. Unbalanced boxes are ignored."""
    text = text or ''
    out: list[str] = []
    pos = 0
    for match in BOXED_RE.finditer(text):
        if match.start() < pos:          # nested inside the previous box
            continue
        j = match.end()
        while j < len(text) and text[j] in ' \t':
            j += 1
        if j < len(text) and text[j] == '{':
            group = _braced(text, j)
            if group is None:
                continue
            out.append(group[0])
            pos = group[1]
        elif match.group(1) == 'boxed' and j > match.end():
            bare = re.match(r'[^\s$]+', text[j:])
            if bare:
                out.append(bare.group(0))
                pos = j + bare.end()
    return out


def last_boxed(text: str) -> str | None:
    """The stripped content of the LAST box, or None when there is none or it is empty."""
    answers = boxed_answers(strip_terminal_tokens(text or ''))
    if not answers:
        return None
    answer = answers[-1].strip()
    return answer or None


def _fix_fracs(s: str) -> str:
    r"""\frac12 -> \frac{1}{2}, \frac1{2} -> \frac{1}{2}, \frac{1}2 -> \frac{1}{2}."""
    parts = s.split('\\frac')
    out = parts[0]
    for part in parts[1:]:
        out += '\\frac'
        if part.startswith('{'):
            group = _braced(part, 0)
            if group is None:
                return s
            num, rest = '{' + group[0] + '}', part[group[1]:]
        elif part:
            num, rest = '{' + part[0] + '}', part[1:]
        else:
            return s
        if rest.startswith('{'):
            out += num + rest
        elif rest:
            out += num + '{' + rest[0] + '}' + rest[1:]
        else:
            return s
    return out


def _fix_sqrt(s: str) -> str:
    r"""\sqrt3 -> \sqrt{3}."""
    return re.sub(r'\\sqrt(?![{\[])\s*([0-9a-zA-Z])', r'\\sqrt{\1}', s)


def _fix_slash(s: str) -> str:
    """a/b with integer a, b -> \\frac{a}{b}."""
    m = re.fullmatch(r'(-?\d+)/(\d+)', s)
    return f'\\frac{{{m.group(1)}}}{{{m.group(2)}}}' if m else s


def _canonical_number(s: str) -> str:
    if _THOUSANDS_RE.fullmatch(s):
        s = s.replace(',', '')
    if not _NUMBER_RE.fullmatch(s):
        return s
    try:
        number = Decimal(s)
    except InvalidOperation:
        return s
    if not number.is_finite():
        return s
    text = format(number.normalize(), 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return '0' if text in {'-0', '+0', ''} else text


def normalize(answer: str | None) -> str | None:
    """Canonical string for exact-match comparison (None for None/empty)."""
    if answer is None:
        return None
    s = str(answer).strip()
    if len(s) > MAX_ANSWER_CHARS:
        s = s[:MAX_ANSWER_CHARS]
    s = s.replace('\n', '').replace('\\\\', '\\')
    s = _SPACING_RE.sub('', s)
    s = s.replace('tfrac', 'frac').replace('dfrac', 'frac')
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('^{\\circ}', '').replace('^\\circ', '').replace('\u00b0', '')
    s = s.replace('\\$', '').replace('$', '')
    if '\\text{ ' in s:                  # trailing units, e.g. "5 \text{ cm}" (MATH convention: a space opens the units)
        s = s.split('\\text{ ')[0]
    prev = None
    while prev != s:                     # unwrap \text{..}, \mathrm{..}, ... (nested ones too)
        prev, s = s, _TEXT_RE.sub(r'\1', s)
    s = s.replace('\\%', '').replace('%', '')
    s = s.replace(' .', ' 0.').replace('{.', '{0.')
    if s.startswith('.'):
        s = '0' + s
    sides = s.split('=')
    if len(sides) == 2 and len(sides[0].strip()) <= 2:   # "x = 5", "k=3": keep the value
        s = sides[1]
    s = _fix_sqrt(s)
    s = s.replace(' ', '')
    s = _fix_fracs(s)
    s = s.rstrip('.')
    if s == '0.5':
        s = '\\frac{1}{2}'
    s = _fix_slash(s)
    s = _canonical_number(s)
    return s or None


def score_completion(text: str, ground_truth: str) -> dict[str, Any]:
    """The scorer result consumed by the reward components and the task metric."""
    clean = strip_terminal_tokens(text or '')
    boxes = [b.strip() for b in boxed_answers(clean) if b.strip()]
    parsed = boxes[-1] if boxes else None
    distinct = len({normalize(b) for b in boxes})
    out: dict[str, Any] = {'parsed': parsed, 'normalized': normalize(parsed), 'truth_normalized': normalize(ground_truth),
                           'boxed_count': len(boxes), 'distinct_answers': distinct, 'single_answer': distinct == 1, 'correct': False}
    if parsed is None:
        return {**out, 'failure_class': 'no_answer'}
    if out['normalized'] is None or out['normalized'] != out['truth_normalized']:
        return {**out, 'failure_class': 'wrong_answer'}
    return {**out, 'correct': True, 'failure_class': 'ok'}


def reference_answer(solution: str) -> str | None:
    """The reference answer: the last box of the dataset's worked solution."""
    answers = [a.strip() for a in boxed_answers(solution or '') if a.strip()]
    return answers[-1] if answers else None
