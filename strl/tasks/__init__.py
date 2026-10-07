"""Environment registry. ``get_environment('gsm8k')`` returns the singleton spec."""
from __future__ import annotations

import importlib

from .base import EnvironmentSpec

_REGISTRY: dict[str, str] = {
    'gsm8k': 'strl.tasks.gsm8k.spec:Gsm8kSpec',
    'coding_exec': 'strl.tasks.coding.spec:CodingExecSpec',
    'coding_harbor': 'strl.tasks.coding.spec:CodingHarborSpec',
    'mimo': 'strl.tasks.mimo.spec:MimoSpec',
}
_CACHE: dict[str, EnvironmentSpec] = {}


class EnvironmentNotFound(KeyError):
    def __str__(self) -> str:
        return self.args[0]


def register_environment(name: str, target: str | EnvironmentSpec) -> None:
    """Register an environment by 'module:Class' path or by instance (user plugins: call this from your own module)."""
    if isinstance(target, EnvironmentSpec):
        _CACHE[name] = target
    else:
        _REGISTRY[name] = target
        _CACHE.pop(name, None)


def list_environments() -> list[str]:
    return sorted(set(_REGISTRY) | set(_CACHE))


def get_environment(name: str) -> EnvironmentSpec:
    if name in _CACHE:
        return _CACHE[name]
    if name not in _REGISTRY:
        raise EnvironmentNotFound(f'unknown environment {name!r}; registered: {list_environments()}')
    module, _, cls = _REGISTRY[name].partition(':')
    spec = getattr(importlib.import_module(module), cls)()
    _CACHE[name] = spec
    return spec
