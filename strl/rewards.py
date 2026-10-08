"""The reward axis: a weighted sum of named components.

    reward:
      components:
        - {fn: pass_all_tests, weight: 1.0}
        - {fn: partial_tests,  weight: 0.5}
        - {fn: length_penalty, weight: 1.0, max_tokens: 400}
        - {fn: user_rewards.my_module:my_reward, weight: 0.2, some_param: 3}   # your own code

A component is ``fn(result: dict, ctx: RewardContext, **params) -> float`` where ``result`` is the environment's scorer
output (coding: tests passed/total, failure class, nonce check...; gsm8k: parsed answer, correct). Every component value is
logged as ``environment/reward_components/<fn>`` (SkyRL prefixes env metrics) so you can see which term moves. The *task metric* (accuracy / pass@1) is computed
separately by the environment spec and is what held-out evaluation reports, so different reward designs stay comparable
and reward hacking (reward up, metric flat) is visible.

Custom ``module:function`` code must live inside this repo (convention: ``user_rewards/``). ``strl stage`` ships the top-level
package of every such module in the payload; ``strl plan`` rejects modules outside the repo, which could not run in the pod.
"""
from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RewardContext:
    ground_truth: Any = None
    extras: dict = field(default_factory=dict)
    max_generate_length: int = 1024
    count_tokens: Callable[[str], int] = staticmethod(lambda text: max(1, len(text) // 4))  # chars/4 unless a tokenizer is wired


def length_penalty(result: dict, ctx: RewardContext, max_tokens: int | None = None, penalty: float = 1.0) -> float:
    """-penalty when the completion exceeds ``max_tokens`` (default: the generation cap), else 0. Weight it positively."""
    limit = max_tokens or ctx.max_generate_length
    return -float(penalty) if ctx.count_tokens(result.get('completion', '')) > limit else 0.0


def format_ok(result: dict, ctx: RewardContext) -> float:
    """1 when the answer was extractable at all (right format), regardless of correctness."""
    return 1.0 if result.get('parsed') is not None or result.get('has_code') else 0.0


COMMON: dict[str, Callable] = {'length_penalty': length_penalty, 'format_ok': format_ok}


def resolve_component(env: Any, name: str) -> Callable:
    """name -> callable. Order: environment components, common components, then 'module:function'."""
    functions = env.reward_functions()
    if name in functions:
        return functions[name]
    if name in COMMON:
        return COMMON[name]
    if ':' in name:
        module, _, attr = name.partition(':')
        try:
            return getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError) as error:
            raise KeyError(f'reward component {name!r} could not be imported: {error}') from error
    raise KeyError(f'unknown reward component {name!r}; {env.name} provides {sorted(functions)}; common: {sorted(COMMON)}; '
                   'or use "module:function"')


def validate_components(env: Any, components: list[dict]) -> list[str]:
    problems = []
    for entry in components:
        name = entry['fn']
        try:
            fn = resolve_component(env, name)
        except KeyError as error:
            problems.append(str(error.args[0]))
            continue
        params = {k for k in entry if k not in {'fn', 'weight'}}
        signature = inspect.signature(fn)
        accepts_var = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        allowed = {n for n, p in signature.parameters.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)} - {'result', 'ctx'}
        unknown = params - allowed
        if unknown and not accepts_var:
            problems.append(f'reward component {name!r} does not accept parameter(s) {sorted(unknown)} (accepts {sorted(allowed)})')
        if ':' in name:
            try:
                custom_payload_paths([entry])
            except ValueError as error:
                problems.append(str(error))
    return problems


def custom_payload_paths(components: list[dict]) -> list[str]:
    """Repo-relative top-level packages (or single-file modules) that 'module:function' components need in the pod payload.
    Raises ValueError for a module that lives outside the repo (e.g. installed in the local venv) and so cannot ship."""
    out: list[str] = []
    for entry in components:
        name = entry['fn']
        if ':' not in name:
            continue
        module_name = name.partition(':')[0]
        source = getattr(importlib.import_module(module_name), '__file__', None)
        path = Path(source).resolve() if source else None
        try:
            rel = path.relative_to(REPO_ROOT) if path else None
        except ValueError:
            rel = None
        if rel is None or rel.parts[0] in {'.venv', 'assets', 'evidence', 'tests'}:
            raise ValueError(f'reward component {name!r}: module file {path} is outside the shippable repo tree; put it under '
                             'user_rewards/ (or another top-level package of this repo) so `strl stage` packages it')
        top = rel.parts[0]
        if top != 'strl' and top not in out:
            out.append(top)
    return out


class Reward:
    """Callable built from the experiment's reward.components."""

    def __init__(self, env: Any, components: list[dict]):
        self.parts: list[tuple[str, float, Callable, dict]] = []
        for entry in components:
            params = {k: v for k, v in entry.items() if k not in {'fn', 'weight'}}
            self.parts.append((entry['fn'], float(entry.get('weight', 1.0)), resolve_component(env, entry['fn']), params))

    def __call__(self, result: dict, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        values: dict[str, float] = {}
        total = 0.0
        for name, weight, fn, params in self.parts:
            value = float(fn(result, ctx, **params))
            values[name] = values.get(name, 0.0) + value
            total += weight * value
        return total, values

    @property
    def names(self) -> list[str]:
        return [p[0] for p in self.parts]
