"""Client for the Modal code executor, used from the reward path (one call per finished rollout) and from batch evaluation.

Training path: ``score(job)`` = spawn + get(timeout) with one jittered retry; memoized by sha256(task, code) because GRPO groups
often repeat identical programs. A control-plane hang can therefore never stall a training step: it becomes ``infra_error``,
which the group-masking generator removes from the loss. Evaluation path: ``score_many`` uses ``.map`` and retries harder.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from collections import OrderedDict

from . import harness

APP_NAME = 'strl-code-exec'
FUNCTION = 'run_tests'
_LOCK = threading.Lock()
_FN: list = []
_MEMO: 'OrderedDict[str, dict]' = OrderedDict()
MEMO_MAX = 50_000


def environment() -> str:
    from ...services.modal import DEFAULT_ENVIRONMENT
    return os.environ.get('MODAL_ENVIRONMENT', DEFAULT_ENVIRONMENT)


def function():
    with _LOCK:
        if not _FN:
            import modal
            fn = modal.Function.from_name(APP_NAME, FUNCTION, environment_name=environment())
            fn.hydrate()
            _FN.append(fn)
        return _FN[0]


def remote_version() -> str | None:
    try:
        return function().remote({'ping': True}).get('version')
    except Exception:
        return None


def deploy() -> dict:
    """Deploy (or update) the executor app in $MODAL_ENVIRONMENT; idempotent. Returns the verified remote version."""
    import modal
    from . import modal_app
    with modal.enable_output():
        modal_app.app.deploy(name=APP_NAME, environment_name=environment(), strategy='recreate')  # old containers stop serving
    with _LOCK:
        _FN.clear()
    version = None
    for _ in range(30):  # a warm old container may still answer briefly; never let a stale harness grade a run
        version = remote_version()
        if version == harness.VERSION:
            break
        time.sleep(4)
    if version != harness.VERSION:
        raise RuntimeError(f'deployed executor version {version} != local harness {harness.VERSION}')
    return {'deployed': True, 'version': version, 'environment': environment()}


def ensure_deployed(auto_deploy: bool = True) -> dict:
    version = remote_version()
    if version == harness.VERSION:
        return {'deployed': False, 'version': version, 'environment': environment()}
    if not auto_deploy:
        raise RuntimeError(f'executor version {version} != harness {harness.VERSION}; run: python -m strl executor deploy')
    return deploy()


def prewarm(n: int = 64) -> float:
    """Wake containers with a burst of no-op calls; returns seconds taken."""
    started = time.monotonic()
    fn = function()
    list(fn.map([{'ping': True}] * n, order_outputs=False, return_exceptions=True))
    return time.monotonic() - started


def _key(job: dict) -> str:
    return hashlib.sha256(json.dumps([job.get('task_id'), job['code'], job['tests'], job.get('setup', '')], sort_keys=True).encode()).hexdigest()


def infra_error(detail: str) -> dict:
    return {'passed': 0, 'total': 0, 'failure_class': 'infra_error', 'detail': detail[:300], 'nonce_ok': False, 'compiled': False,
            'latency_s': 0.0}


def score(job: dict, *, timeout: float = 30.0, retries: int = 1, memo: bool = True) -> dict:
    key = _key(job)
    if memo:
        with _LOCK:
            if key in _MEMO:
                _MEMO.move_to_end(key)
                return dict(_MEMO[key], memo_hit=True)
    last = 'unknown'
    started = time.monotonic()
    for attempt in range(retries + 1):
        call = None
        try:
            call = function().spawn({k: v for k, v in job.items() if k != 'task_id'})
            result = call.get(timeout=timeout)
            if result.get('harness_version') not in (None, harness.VERSION):
                return infra_error(f'executor version {result.get("harness_version")} != {harness.VERSION}')
            result['round_trip_s'] = round(time.monotonic() - started, 4)
            if memo:
                with _LOCK:
                    _MEMO[key] = result
                    if len(_MEMO) > MEMO_MAX:
                        _MEMO.popitem(last=False)
            return result
        except Exception as error:  # timeouts and Modal transport errors are infrastructure, never the model's fault
            last = f'{type(error).__name__}: {error}'
            if call is not None:
                try:
                    call.cancel()
                except Exception:
                    pass
            time.sleep(0.5 + random.random())
    return infra_error(last)


def score_many(jobs: list[dict], *, retries: int = 3) -> list[dict]:
    """Batch scoring for evaluation: .map with ordered outputs; failed items retried; remaining failures are infra_error."""
    fn = function()
    payload = [{k: v for k, v in job.items() if k != 'task_id'} for job in jobs]
    results: list = list(fn.map(payload, order_outputs=True, return_exceptions=True))
    for _ in range(retries):
        pending = [i for i, r in enumerate(results) if isinstance(r, BaseException)]
        if not pending:
            break
        retried = list(fn.map([payload[i] for i in pending], order_outputs=True, return_exceptions=True))
        for i, r in zip(pending, retried):
            results[i] = r
    return [infra_error(f'{type(r).__name__}: {r}') if isinstance(r, BaseException) else r for r in results]
