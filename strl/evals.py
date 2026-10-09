"""Reference evaluation of an external LLM (Scale LiteLLM, e.g. gpt-6-luna) with the EXACT training prompts, extractor and harness.

coding_exec: same single-turn prompt -> extract -> Modal executor. coding_harbor: the same Terminus-2 trial template as training,
with only the model route changed (LiteLLM proxy; no vLLM token ids; no session_id; max_completion_tokens for reasoning models).
Use it to calibrate task difficulty (a ceiling) and to sanity-check the environment before spending GPUs.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .services import litellm


def coding_exec(cfg: dict, split: str, limit: int, model: str, max_completion_tokens: int, project_id: str, concurrency: int = 8) -> dict:
    from .tasks.coding import data as coding_data, executor
    from .tasks.coding.extract import extract_code
    tasks = coding_data.load_tasks(cfg, split)[:limit]
    key = litellm.resolve_key()

    def ask(task):
        try:
            return litellm.chat([{'role': 'user', 'content': task.prompt}], model=model, max_completion_tokens=max_completion_tokens,
                                project_id=project_id, json_mode=False, key=key, timeout=180)
        except litellm.LiteLLMError as error:
            return {'text': '', 'error': str(error), 'cost_usd': 0.0, 'usage': {}}

    started = time.monotonic()
    with ThreadPoolExecutor(concurrency) as pool:
        answers = list(pool.map(ask, tasks))
    codes = [extract_code(a['text']) for a in answers]
    runnable = [i for i, c in enumerate(codes) if c]
    verdicts = dict(zip(runnable, executor.score_many([tasks[i].job(codes[i]) for i in runnable])))
    rows = []
    for i, task in enumerate(tasks):
        v = verdicts.get(i) or {'failure_class': 'no_code_block' if not answers[i].get('error') else 'llm_error', 'passed': 0, 'total': len(task.tests)}
        rows.append({'sample_id': task.sample_id, 'source': task.source, 'failure_class': v['failure_class'], 'passed': v.get('passed'),
                     'total': v.get('total'), 'correct': v['failure_class'] == 'ok', 'cost_usd': answers[i].get('cost_usd'),
                     'error': answers[i].get('error')})
    return _summary(rows, model, split, time.monotonic() - started)


def coding_harbor(cfg: dict, split: str, limit: int, model: str, max_completion_tokens: int, project_id: str, concurrency: int = 5) -> dict:
    from .services.modal import local_profile
    from .tasks.coding import harbor
    os.environ['LITELLM_PROXY_API_KEY'] = litellm.resolve_key()
    tasks = sorted((Path(cfg['run']['data_dir']) / 'harbor' / split).iterdir())[:limit]
    template = harbor.trial_template(cfg)
    template['trials_dir'] = str(Path('evidence/reference-evals/harbor-trials').resolve())
    route = model if model.startswith('litellm_proxy/') else f'litellm_proxy/{model}'
    configs = []
    for task in tasks:
        c = harbor.build_trial_config(template, str(task), model_name=route, api_base=litellm.BASE_URL)
        kw = c['agent']['kwargs']
        kw.pop('session_id', None)                     # OpenAI rejects Harbor's vLLM-router affinity field
        kw['collect_rollout_details'] = False          # external APIs return no vLLM token ids
        llm = kw.setdefault('llm_kwargs', {})
        llm.pop('max_tokens', None)
        llm.pop('top_p', None)
        llm.pop('extra_body', None)
        llm.update(max_completion_tokens=max_completion_tokens, user=project_id, extra_body={'metadata': {'tags': [f'project:{project_id}']}})
        configs.append(c)
    started = time.monotonic()
    with local_profile(((cfg.get('services') or {}).get('modal') or {}).get('environment', 'main')):
        harbor.install_compat()

        async def main():
            sem = asyncio.Semaphore(concurrency)

            async def one(c):
                async with sem:
                    return await harbor.run_trial(c)
            return await asyncio.wait_for(asyncio.gather(*(one(c) for c in configs)), timeout=3600)
        results = asyncio.run(main())
    rows = [{'sample_id': t.name, 'source': t.name.split('-')[0], 'failure_class': 'ok' if r['reward'] == 1.0 else (r['exception'] or 'wrong'),
             'correct': r['reward'] == 1.0, 'reward': r['reward'], 'seconds': r['seconds']} for t, r in zip(tasks, results)]
    return _summary(rows, model, split, time.monotonic() - started)


def _summary(rows: list[dict], model: str, split: str, seconds: float) -> dict:
    by_source: dict[str, list] = {}
    for r in rows:
        by_source.setdefault(r['source'], []).append(r['correct'])
    classes: dict[str, int] = {}
    for r in rows:
        classes[r['failure_class']] = classes.get(r['failure_class'], 0) + 1
    out = {'model': model, 'split': split, 'tasks': len(rows), 'pass_at_1': sum(r['correct'] for r in rows) / max(1, len(rows)),
           'by_source': {k: sum(v) / len(v) for k, v in by_source.items()}, 'failure_classes': classes, 'seconds': round(seconds, 1),
           'cost_usd': round(sum(r.get('cost_usd') or 0 for r in rows), 4)}
    Path('evidence/reference-evals').mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%y%m%d%H%M%S')
    Path(f'evidence/reference-evals/{model.replace("/", "_")}-{split}-{stamp}.json').write_text(json.dumps({'summary': out, 'rows': rows}, indent=2))
    return out


def mimo(cfg: dict, split: str, limit: int, model: str, max_completion_tokens: int, project_id: str, concurrency: int = 8) -> dict:
    """The CEILING of a mimo selection: an external model (default gpt-6-luna) as the agent on the same held-out tasks, through the same
    exported tasks, per-domain caps, graders and judge as training (strl.tasks.mimo.generator.run_trajectory in its eval phase); only the
    model route changes. ``limit`` = tasks per domain. Tells whether a domain is solvable at all within the caps before GPUs are spent."""
    from types import SimpleNamespace
    from .services.modal import local_profile
    from .tasks.coding import harbor
    from .tasks.mimo import data, judge as judge_mod
    from .tasks.mimo.generator import DEFAULT_SETTINGS, domain_of, generator_settings, mimo_template, run_trajectory
    os.environ['LITELLM_PROXY_API_KEY'] = litellm.resolve_key()
    selection = data.runtime_selection(cfg)[split]
    tasks = [str(data.task_dir(cfg['run']['data_dir'], split, d, i)) for d in data.DOMAINS if d in selection for i in selection[d][:limit]]
    template = mimo_template(cfg)
    template['trials_dir'] = str(Path('evidence/reference-evals/mimo-trials').resolve())
    template['agent'].update(model_name=model if model.startswith('litellm_proxy/') else f'litellm_proxy/{model}')
    template['agent'].setdefault('kwargs', {})['api_base'] = litellm.BASE_URL
    m = data.mimo_config(cfg)
    caps = {d: {k: v[k] for k in ('max_agent_sec', 'max_verifier_sec', 'max_turns') if v.get(k)} for d, v in m['domains'].items()}
    judged = any(d in selection for d in ('general', 'webdev'))

    async def create_trial(config: dict):
        from harbor.models.trial.config import TrialConfig
        from harbor.trial.trial import Trial
        kw = config['agent'].setdefault('kwargs', {})
        kw.pop('session_id', None)                     # OpenAI rejects the vLLM-router affinity field
        kw['collect_rollout_details'] = False          # external APIs return no vLLM token ids
        llm = kw.setdefault('llm_kwargs', {})
        for key in ('max_tokens', 'top_p', 'extra_body'):
            llm.pop(key, None)
        llm.update(max_completion_tokens=max_completion_tokens, user=project_id, extra_body={'metadata': {'tags': [f'project:{project_id}']}})
        kw.pop('temperature', None)                    # reasoning models reject a temperature
        return await Trial.create(TrialConfig.model_validate(config))

    class _Null:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *exc):
            return False

    async def finish(_session):
        return None

    started = time.monotonic()
    with local_profile(((cfg.get('services') or {}).get('modal') or {}).get('environment', 'main')):
        harbor.install_compat()
        deps = SimpleNamespace(template=template, settings={**DEFAULT_SETTINGS, **generator_settings(cfg)}, rate_limiter=_Null(),
                               judge=judge_mod.Judge(cfg) if judged else None, create_trial=create_trial, finish_session=finish,
                               caps=lambda d: caps.get(d) or {})

        async def main():
            sem = asyncio.Semaphore(concurrency)

            async def one(i, path):
                async with sem:
                    _, summary = await run_trajectory(deps, path, SimpleNamespace(to_string=lambda i=i: f'ceiling_{i}'), phase='eval')
                    return summary
            return await asyncio.wait_for(asyncio.gather(*(one(i, p) for i, p in enumerate(tasks))), timeout=4 * 3600)
        summaries = asyncio.run(main())
    rows = [{'sample_id': Path(p).name, 'source': domain_of(p, None), 'domain': domain_of(p, None), 'correct': bool(s.get('passed')),
             'metric': s.get('metric'), 'masked': s.get('masked'), 'stop': s.get('stop'), 'turns': s.get('turns'),
             'failure_class': s.get('mask_reason') or s.get('stop') or 'complete',
             'cost_usd': ((s.get('judge') or {}).get('meta') or {}).get('cost_usd', 0.0)} for p, s in zip(tasks, summaries)]
    out = _summary(rows, model, f'mimo-{split}', time.monotonic() - started)
    valid = [r for r in rows if not r['masked']]
    out['task_metric_by_domain'] = {d: sum(float(r['metric'] or 0) for r in valid if r['domain'] == d) / max(1, sum(r['domain'] == d for r in valid))
                                    for d in sorted({r['domain'] for r in valid})}
    out['timed_out'] = sum(r['stop'] in ('agent_timeout', 'verifier_timeout') for r in rows)
    out['masked'] = len(rows) - len(valid)
    return out
