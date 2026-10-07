"""Qualitative critic: an LLM (Scale LiteLLM, e.g. gpt-6-luna) classifies WHY sampled rollouts failed.

Off the training path by construction: the trajectory logger ``submit_failures`` with ``put_nowait`` into a bounded queue
(rows are dropped when full); a daemon thread calls LiteLLM; ``drain`` (called from ``on_log`` on the trainer thread)
returns finished analyses for the ``qualitative/critic`` table and ``critic/*`` scalars. A hard USD cap disables it.

    tracking:
      critic: {enabled: true, model: openai/gpt-6-luna, every_n_steps: 5, max_rows: 8, budget_usd: 5.0,
               project_id: 696011502a26dc4a64b24f55, max_completion_tokens: 1024}
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import Counter
from typing import Any

MODES = ('misread_task', 'wrong_algorithm', 'edge_case', 'syntax_or_runtime_error', 'format_violation', 'truncated_or_incomplete',
         'timeout_or_inefficient', 'arithmetic_error', 'reward_hacking_attempt', 'other')
DEFAULTS = {'enabled': False, 'model': 'openai/gpt-6-luna', 'every_n_steps': 5, 'max_rows': 8, 'budget_usd': 5.0,
            'project_id': '696011502a26dc4a64b24f55', 'max_completion_tokens': 1024, 'timeout_seconds': 60,
            'usd_per_mtok_in': 2.0, 'usd_per_mtok_out': 8.0}
def system_prompt(modes: tuple[str, ...] = MODES) -> str:
    return ('You analyse failed attempts of a small language model being trained with reinforcement learning. Classify the PRIMARY '
            'reason the attempt failed. Respond with a JSON object only: {"failure_mode": one of ' + json.dumps(list(modes)) +
            ', "rationale": "<= 40 words", "fix_hint": "<= 25 words"}.')


SYSTEM = system_prompt()

_S: dict[str, Any] = {}
_LOCK = threading.Lock()


def configure(settings: dict | None, modes: tuple[str, ...] | None = None) -> bool:
    """Start the critic thread if enabled. Idempotent. Returns whether it is active. ``modes``: the environment's failure taxonomy
    (EnvironmentSpec.critic_modes); None keeps the default MODES."""
    cfg = {**DEFAULTS, **(settings or {})}
    with _LOCK:
        if _S.get('thread') or not cfg['enabled']:
            return bool(_S.get('thread'))
        _S.update(cfg=cfg, q=queue.Queue(maxsize=32), results=[], modes=Counter(), cost=0.0, calls=0, errors=0, dropped=0,
                  disabled=None, taxonomy=tuple(modes or MODES))
        thread = threading.Thread(target=_loop, name='strl-critic', daemon=True)
        _S['thread'] = thread
        thread.start()
    return True


def active() -> bool:
    return bool(_S.get('thread')) and not _S.get('disabled')


def submit_failures(step: int, prompts: list, completions: list[str], rewards: list[float], env_metrics: list | None, env: Any) -> int:
    """Queue up to max_rows failed rollouts from this step (every_n_steps). Never blocks; returns rows queued."""
    if not active() or step % int(_S['cfg']['every_n_steps']):
        return 0
    failed = []
    for index, completion in enumerate(completions):
        row = (env_metrics[index] or {}) if env_metrics and index < len(env_metrics) else {}
        metric = row.get('task_metric')
        if 'failed' in row:      # environments with graded (non-binary) metrics say themselves what counts as a failure
            if row['failed']:
                failed.append(index)
        elif (metric == 0.0) if metric is not None else (rewards[index] <= 0.0):
            failed.append(index)
    queued = 0
    for index in failed[: int(_S['cfg']['max_rows'])]:
        info = (env_metrics[index] if env_metrics and index < len(env_metrics) else {}) or {}
        item = {'source_step': step, 'idx': index, 'prompt': (env.display_prompt(prompts[index]) if env else str(prompts[index]))[:3000],
                'completion': completions[index][-3000:], 'env_failure': next((k.split('/', 1)[1] for k, v in info.items()
                                                                               if k.startswith('failure/') and v), None)}
        if info.get('domain'):   # multi-domain environments: the critic sees which domain failed (and the judge's reason, if any)
            item['domain'], item['judge'] = info['domain'], info.get('judge_rationale')
        try:
            _S['q'].put_nowait(item)
            queued += 1
        except queue.Full:
            _S['dropped'] += 1
    return queued


def _estimate_cost(usage: dict) -> float:
    cfg = _S['cfg']
    return (usage.get('prompt_tokens', 0) * cfg['usd_per_mtok_in'] + usage.get('completion_tokens', 0) * cfg['usd_per_mtok_out']) / 1e6


def analyse(item: dict, call=None) -> dict:
    """One critic call (``call`` is injectable for tests)."""
    from ..services import litellm
    cfg = _S.get('cfg', DEFAULTS)
    call = call or litellm.chat
    taxonomy = _S.get('taxonomy') or MODES
    context = (f'Domain: {item["domain"]}\n' if item.get('domain') else '') + (f'Grader/judge note: {item["judge"]}\n' if item.get('judge') else '')
    user = (f'Environment verdict: {item.get("env_failure") or "failed"}\n{context}\nTASK:\n{item["prompt"]}\n\n'
            f'MODEL ATTEMPT (tail):\n{item["completion"]}')
    response = call([{'role': 'system', 'content': system_prompt(taxonomy) if taxonomy is not MODES else SYSTEM}, {'role': 'user', 'content': user}],
                    model=cfg['model'], max_completion_tokens=int(cfg['max_completion_tokens']), project_id=cfg['project_id'],
                    timeout=float(cfg['timeout_seconds']))
    parsed = litellm.parse_json_object(response['text']) or {}
    mode = parsed.get('failure_mode') if parsed.get('failure_mode') in taxonomy else 'other'
    cost = response['cost_usd'] if response.get('cost_usd') is not None else _estimate_cost(response.get('usage') or {})
    return {'source_step': item['source_step'], 'idx': item['idx'], 'env_failure': item.get('env_failure'), 'critic_mode': mode,
            'rationale': str(parsed.get('rationale', ''))[:400], 'fix_hint': str(parsed.get('fix_hint', ''))[:300],
            'prompt': item['prompt'][:1500], 'completion_tail': item['completion'][-1500:], '_cost': float(cost)}


def _loop() -> None:
    q = _S["q"]          # a thread serves only the queue it was started for
    while True:
        item = q.get()
        if item is None:
            return
        if _S.get('disabled'):
            continue
        try:
            row = analyse(item)
            with _LOCK:
                _S['cost'] += row.pop('_cost')
                _S['calls'] += 1
                _S['modes'][row['critic_mode']] += 1
                _S['results'].append(row)
                if _S['cost'] >= float(_S['cfg']['budget_usd']):
                    _S['disabled'] = f'budget {_S["cfg"]["budget_usd"]} USD reached'
        except Exception as error:
            with _LOCK:
                _S['errors'] += 1
                if _S['errors'] >= 10 and _S['calls'] == 0:
                    _S['disabled'] = f'{type(error).__name__}: {error}'[:200]


def drain() -> tuple[dict[str, float], list[dict]]:
    """Finished analyses since the last drain + cumulative critic/* scalars (empty when the critic is off)."""
    if not _S.get('thread'):
        return {}, []
    with _LOCK:
        rows, _S['results'] = _S['results'], []
        total = sum(_S['modes'].values())
        scalars = {f'critic/{mode}_frac': _S['modes'][mode] / total for mode in _S.get('taxonomy') or MODES} if total else {}
        scalars.update({'critic/cost_usd': _S['cost'], 'critic/analyses': float(_S['calls']), 'critic/errors': float(_S['errors']),
                        'critic/dropped': float(_S['dropped']), 'critic/disabled': float(bool(_S.get('disabled')))})
    return scalars, rows


def shutdown(timeout: float = 60.0) -> None:
    thread = _S.get('thread')
    if not thread:
        return
    deadline = time.time() + timeout
    while not _S['q'].empty() and time.time() < deadline:
        time.sleep(0.5)
    try:
        _S['q'].put_nowait(None)
    except queue.Full:
        pass
    thread.join(timeout=max(0.0, deadline - time.time()))


def reset() -> None:  # tests
    thread, q = _S.get('thread'), _S.get('q')
    if thread is not None and q is not None:
        _S['disabled'] = 'reset'
        try:
            q.put_nowait(None)
        except queue.Full:
            pass
        thread.join(timeout=3)
    with _LOCK:
        _S.clear()


def settings_from_env() -> dict | None:
    raw = os.environ.get('STRL_CRITIC')
    return json.loads(raw) if raw else None
