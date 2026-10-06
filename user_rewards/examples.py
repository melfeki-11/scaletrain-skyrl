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
