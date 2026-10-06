"""The generic SkyRL environment: score the completion with the environment spec, then apply the experiment's reward components.

Registered as ``strl_env`` inside the Ray driver task (``register_generic_env``). It is used whenever the reward differs
from the verified default, and for every coding environment. The scorer result and each reward component are recorded so
that (a) the in-loop eval can report the task metric rather than the training reward, and (b) W&B shows every term.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

from . import experiment as experiment_mod
from .rewards import Reward, RewardContext
from .tasks import get_environment

_LOCK = threading.Lock()
_RUNTIME: dict[str, Any] = {}


def _tokenizer_counter(path: str | None):
    if path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(path)
            return lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return None


def runtime() -> dict[str, Any]:
    """Process-wide cache: resolved experiment, environment spec, Reward, token counter (built once, thread-safe)."""
    with _LOCK:
        if not _RUNTIME:
            cfg = experiment_mod.current()
            spec = get_environment(cfg['environment']['name'])
            counter = _tokenizer_counter(os.environ.get('STRL_TOKENIZER') or cfg['model'].get('local_path'))
            _RUNTIME.update(cfg=cfg, spec=spec, reward=Reward(spec, cfg['reward']['components']), counter=counter)
    return _RUNTIME


def reset_runtime() -> None:  # tests
    with _LOCK:
        _RUNTIME.clear()


def compact(result: dict) -> dict:
    """JSON-safe, size-bounded copy of a scorer result for eval dumps and tables."""
    out = {}
    for key, value in result.items():
        if isinstance(value, str) and len(value) > 2000:
            value = value[:2000] + '...<truncated>'
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, dict)):
            try:
                json.dumps(value)
                out[key] = value
            except TypeError:
                out[key] = str(value)[:500]
    return out


def make_env_class():
    from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

    class StrlEnv(BaseTextEnv):
        """One-step environment: completion -> scorer result -> weighted reward."""

        def __init__(self, env_config: Any = None, extras: dict | None = None):
            super().__init__()
            self.extras = extras if extras is not None else {}
            self.rt = runtime()
            self.metrics: dict[str, float] = {}

        def step(self, action: str) -> BaseTextEnvStepOutput:
            rt = self.rt
            result = rt['spec'].score_result(action, self.extras, rt['cfg'])
            result.setdefault('completion', action)
            ctx = RewardContext(ground_truth=(self.extras.get('reward_spec') or {}).get('ground_truth'), extras=self.extras,
                                max_generate_length=int(rt['cfg']['algorithm']['max_generate_length']))
            if rt['counter'] is not None:
                ctx.count_tokens = rt['counter']
            reward, parts = rt['reward'](result, ctx)
            self.extras['strl_result'] = compact({**result, 'reward': reward, 'components': parts})
            metrics = {f'reward_components/{name}': value for name, value in parts.items()}
            metrics['task_metric'] = 1.0 if result.get('correct') else 0.0
            for failure_class in rt['spec'].failure_classes:
                metrics[f'failure/{failure_class}'] = 1.0 if result.get('failure_class') == failure_class else 0.0
            if 'latency_s' in result:
                metrics['exec_latency_s'] = float(result['latency_s'])
            metrics['infra_error'] = 1.0 if result.get('failure_class') == 'infra_error' else 0.0
            self.metrics = metrics
            return BaseTextEnvStepOutput(observations=[], reward=float(reward), done=True, metadata={})

        def get_metrics(self) -> dict[str, Any]:
            return dict(self.metrics)

    return StrlEnv


def register_generic_env(name: str = 'strl_env') -> None:
    """Register ``strl_env`` with skyrl_gym (call inside the Ray driver task; idempotent).

    The entry point must be the class itself: SkyRL's metric aggregation calls ``<entry_point>.aggregate_metrics``.
    """
    from skyrl_gym.envs import register
    from skyrl_gym.envs.registration import registry
    if name in registry:
        return
    register(id=name, entry_point=make_env_class())
