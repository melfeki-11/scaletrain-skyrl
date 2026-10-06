from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

FINAL_RE = re.compile(r"(?:^|\n)####[ \t]*([-+]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?)[ \t]*\Z")
MARKER_RE = re.compile(r"(?m)^####(?:[ \t]|$)")


def normalize_number(value: str) -> str | None:
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned in {"+", "-", "."}:
        return None
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", "+0", ""} else normalized


def parse_final_answer(text: str) -> str | None:
    # The generator may decode the real terminal EOS marker into the reward text.
    text = text.rstrip()
    for token in ("<|endoftext|>", "<|im_end|>"):
        if text.endswith(token):text = text[:-len(token)].rstrip()
    if len(MARKER_RE.findall(text)) != 1:
        return None
    matches = FINAL_RE.findall(text)
    if len(matches) != 1:
        return None
    return normalize_number(matches[0])


def extract_reference(answer: str) -> str:
    marker = re.search(r"####\s*([^\n]+)\s*$", answer)
    if marker is None:
        raise ValueError("GSM8K reference lacks a terminal #### answer")
    normalized = normalize_number(marker.group(1))
    if normalized is None:
        raise ValueError("GSM8K reference answer is not numeric")
    return normalized


def score_answer(completion: str, ground_truth: str, finish_reason: str | None = None) -> float:
    if finish_reason not in {None, "stop", "eos_token"}:
        return 0.0
    parsed = parse_final_answer(completion)
    expected = normalize_number(ground_truth)
    return float(parsed is not None and expected is not None and parsed == expected)
