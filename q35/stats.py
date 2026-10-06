from __future__ import annotations

import numpy as np


def paired_bootstrap(
    baseline: list[bool], candidate: list[bool], *, samples: int = 10000, seed: int = 29, level: float = 0.95
) -> dict[str, float | int | bool]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("baseline and candidate must have equal nonzero length")
    base = np.asarray(baseline, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    delta = cand - base
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(samples, len(delta)))
    means = delta[draws].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "examples": len(delta),
        "baseline_accuracy": float(base.mean()),
        "candidate_accuracy": float(cand.mean()),
        "accuracy_delta": float(delta.mean()),
        "ci_level": level,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "quality_passed": bool(lower > 0.0),
    }
