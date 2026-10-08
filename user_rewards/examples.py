"""Example custom reward components (see user_rewards/__init__.py and experiments/mbpp-qwen35-custom-reward.yaml)."""
from __future__ import annotations

import ast


def docstring_bonus(result: dict, ctx, bonus: float = 0.1) -> float:
    """+bonus when the extracted code (result['parsed']) has a module/function/class docstring, else 0."""
    code = result.get('parsed')
    if not code:
        return 0.0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0
    nodes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return float(bonus) if any(ast.get_docstring(n) for n in nodes) else 0.0


def turn_efficiency_bonus(result: dict, ctx, bonus: float = 0.05, max_turns: int = 12) -> float:
    """MiMo shaping example: +bonus * (1 - turns / max_turns) when the trial fully passed (native score >= 0.999) in fewer than max_turns turns.

    For `mimo`, ``result`` is {domain, task, reward (the NATIVE grader score), metric, passed, components, turns, tokens, seconds, stop,
    artifact}; the term is added to the TRAINING reward only (evaluation always reports the native task metric). Weight it in YAML.
    """
    if float(result.get('reward', 0.0)) < 0.999:
        return 0.0
    return float(bonus) * max(0.0, 1.0 - int(result.get('turns', max_turns)) / float(max_turns))
