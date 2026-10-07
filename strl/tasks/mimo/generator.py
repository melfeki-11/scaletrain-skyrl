"""MimoHarborGenerator: SkyRL's HarborGenerator (pin a9564572) with MiMo semantics, built lazily by ``make_generator_class()``.

What differs from the stock generator, and why (all pure logic lives in module-level functions, unit-tested with fake trials):
  * per-task trial config: base template (no global override_* keys) < strl.json (agent name | import_path, max_turns, max_tokens,
    kwargs) < task.toml (read natively by Harbor: agent/verifier timeouts, cpus/memory) < optional per-domain caps;
  * retry policy: only transport / unknown trial errors are retried (<= max_trial_attempts); agent and verifier timeouts and
    verifier-reported infra errors never are (a timeout is not transient, and a retried 30-minute task costs a step);
  * outcomes: a timed-out agent is SCORED by its verifier (``timeout_policy: score``, default) because a slow policy is a
    failing policy; ``mask`` restores SkyRL's behaviour (mask the group). Infra errors, judge failures and deadline stragglers
    are always MASKED (stop_reason ``error``: SkyRL zeroes the loss of the whole GRPO group), never scored 0;
  * judged domains: the verifier's judge_request.json is scored by judge.py outside the sandbox (key never enters a sandbox);
  * per-step deadline (``step_deadline_sec``): trials still running when it expires are cancelled and masked ('deadline');
  * eval temperature: the batch's own sampling temperature (eval_sampling_params.temperature) reaches the agent;
  * per-domain rollout metrics + in-memory trial summaries (artifact text, judge rationale, components) that tracking reads.
"""
from __future__ import annotations

import asyncio
import json
import time
import tomllib
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from ..coding import harbor as coding_harbor
from . import judge as judge_mod
from .data import DOMAINS, task_domain

DEFAULT_SETTINGS = {'step_deadline_sec': None, 'timeout_policy': 'score', 'max_trial_attempts': 2, 'summary_chars': 12000}
TIMEOUT_POLICIES = ('score', 'mask')
OVERRIDE_KEYS = {'agent': ('override_timeout_sec', 'override_setup_timeout_sec'), 'verifier': ('override_timeout_sec',),
                 'environment': ('override_cpus', 'override_memory_mb', 'override_storage_mb', 'suppress_override_warnings')}
SUMMARY_STORE_LIMIT = 4096
SUMMARIES: 'OrderedDict[str, dict]' = OrderedDict()    # "<phase>:<instance>_<rep>" -> trial summary (bounded, newest kept)


# ------------------------------------------------------------------------------------------------------------ settings / template
def generator_settings(cfg: dict | None) -> dict:
    raw = ((cfg or {}).get('environment') or {}).get('mimo') or {}
    out = {k: raw.get(k, v) for k, v in DEFAULT_SETTINGS.items()}
    if out['step_deadline_sec'] is not None:
        out['step_deadline_sec'] = float(out['step_deadline_sec'])
    return out


def mimo_template(cfg: dict) -> dict:
    """The coding trial template without any global override_* key: timeouts and resources come from each task's task.toml."""
    template = coding_harbor.trial_template(cfg)
    for section, keys in OVERRIDE_KEYS.items():
        for key in keys:
            template.get(section, {}).pop(key, None)
    return template


def read_task(task_dir: str | Path) -> tuple[dict, dict]:
    """(strl.json, task.toml) of an exported task directory."""
    root = Path(task_dir)
    meta = json.loads((root / 'strl.json').read_text()) if (root / 'strl.json').is_file() else {}
    toml = tomllib.loads((root / 'task.toml').read_text()) if (root / 'task.toml').is_file() else {}
    return meta, toml


def domain_of(task_dir: str | Path, env_extra: dict | None = None) -> str:
    domain = (env_extra or {}).get('domain')
    return str(domain) if domain else task_domain(task_dir)


def build_trial_config(template: dict, task_dir: str | Path, *, session_id: str, meta: dict | None = None, toml: dict | None = None,
                       temperature: float | None = None, cache_salt: str | None = None, caps: dict | None = None) -> dict:
    """The Harbor TrialConfig dict of one trial: template < strl.json < task.toml-derived caps (+ sampling temperature, cache salt)."""
    import copy
    meta, toml = meta or {}, toml or {}
    config = copy.deepcopy(template)
    config['task'] = {'path': str(task_dir)}
    agent = config.setdefault('agent', {})
    kwargs = agent.setdefault('kwargs', {})
    kwargs['session_id'] = session_id
    spec = meta.get('agent') or {}
    if spec.get('import_path'):
        agent['import_path'] = spec['import_path']
        agent.pop('name', None)
    elif spec.get('name'):
        agent['name'] = spec['name']
        agent.pop('import_path', None)
    kwargs.update(spec.get('kwargs') or {})
    llm = kwargs.setdefault('llm_kwargs', {})
    if meta.get('max_turns') is not None:
        kwargs['max_turns'] = int(meta['max_turns'])
    if meta.get('max_tokens') is not None:
        llm['max_tokens'] = int(meta['max_tokens'])
        kwargs.setdefault('model_info', {})['max_output_tokens'] = int(meta['max_tokens'])
    if temperature is not None:
        kwargs['temperature'] = float(temperature)
    if cache_salt is not None:
        llm.setdefault('extra_body', {})['cache_salt'] = cache_salt
    caps = caps or {}
    if caps.get('max_agent_sec'):
        agent['override_timeout_sec'] = min(float((toml.get('agent') or {}).get('timeout_sec', caps['max_agent_sec'])), float(caps['max_agent_sec']))
    if caps.get('max_verifier_sec'):
        config.setdefault('verifier', {})['override_timeout_sec'] = min(float((toml.get('verifier') or {}).get('timeout_sec', caps['max_verifier_sec'])),
                                                                        float(caps['max_verifier_sec']))
    return config


# ------------------------------------------------------------------------------------------------------------ verifier files
def trial_dir_of(trial: Any, results: Any) -> Path | None:
    uri = getattr(results, 'trial_uri', None)
    if uri:
        return Path(unquote(urlparse(uri).path))
    config = getattr(trial, 'config', None)
    if config is not None and getattr(config, 'trials_dir', None) and getattr(config, 'trial_name', None):
        return Path(config.trials_dir) / config.trial_name
    return None


def read_verifier(trial_dir: Path | None) -> dict:
    """{'reward', 'details', 'judge_request', 'dir'} from <trial_dir>/verifier (missing files are None / {}); malformed JSON raises."""
    out: dict[str, Any] = {'reward': None, 'details': {}, 'judge_request': None, 'dir': None}
    if trial_dir is None:
        return out
    root = Path(trial_dir) / 'verifier'
    out['dir'] = root
    if (root / 'reward.txt').is_file():
        try:
            out['reward'] = float((root / 'reward.txt').read_text().strip())
        except ValueError:
            out['reward'] = None
    if (root / 'details.json').is_file():
        out['details'] = json.loads((root / 'details.json').read_text())
    if (root / 'judge_request.json').is_file():
        out['judge_request'] = json.loads((root / 'judge_request.json').read_text())
    return out


def verifier_ran(trial_dir: str | Path | None) -> tuple[bool, str]:
    """(ran, why-not): the verifier left a non-empty details.json without an infra flag. A reward of 0 proves nothing about this: a verifier
    that never ran scores exactly like a correct one on a no-op agent (the first music/general graders never ran and the nop gate passed)."""
    if trial_dir is None:
        return False, 'no trial dir'
    details = Path(trial_dir) / 'verifier' / 'details.json'
    if not details.is_file():
        return False, 'no details.json'
    try:
        body = json.loads(details.read_text())
    except ValueError as error:
        return False, f'unreadable details.json: {error}'
    if not isinstance(body, dict) or not body:
        return False, 'empty details.json'
    if body.get('infra_error'):
        return False, f"infra_error: {str(body.get('error'))[:120]}"
    return True, ''


def read_artifacts(verifier_dir: Path | None, names: list[str], cap: int) -> tuple[str | None, str | None]:
    """(first non-empty text artifact truncated to ``cap`` chars, screenshot path) from the host copy of the verifier outputs."""
    if verifier_dir is None:
        return None, None
    text = None
    for name in names:
        path = Path(verifier_dir) / Path(name).name
        if path.is_file() and path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.gif'}:
            body = path.read_text(errors='replace')[:cap]
            if body.strip():
                text = body
                break
    shot = next((str(Path(verifier_dir) / n) for n in ('screenshot.png', 'screenshot.jpg') if (Path(verifier_dir) / n).is_file()), None)
    return text, shot


# ------------------------------------------------------------------------------------------------------------ policy
def exception_type(results: Any) -> str | None:
    info = getattr(results, 'exception_info', None)
    return getattr(info, 'exception_type', None) if info else None


# The policy's own failures: the trial counts (reward 0) instead of being retried or masked. Hiding them would make a policy that
# rambles or runs out of time look better (survivorship) and would drop its negative training signal.
POLICY_LIMIT = ('ContextLengthExceededError', 'OutputLengthExceededError')
ZERO_EXC = POLICY_LIMIT + ('VerifierTimeoutError',)
STOP_NAME = {'ContextLengthExceededError': 'context_length', 'OutputLengthExceededError': 'output_length',
             'AgentTimeoutError': 'agent_timeout', 'VerifierTimeoutError': 'verifier_timeout'}


def decide(results: Any, *, attempt: int, attempts: int, policy: str) -> str:
    """What to do with one finished trial attempt:
      'use'    -> a verifier verdict exists and the trial counts (maybe with reward 0 for context overflow / timeouts)
      'retry'  -> transport or unknown error with attempts left
      'mask'   -> give up: infra or unrecoverable error (the group is masked)
    Agent timeout, verifier timeout, context- and output-length overflow are never retried."""
    exc = exception_type(results)
    has_verdict = bool(getattr(results, 'verifier_result', None) and getattr(results.verifier_result, 'rewards', None))
    if exc in POLICY_LIMIT:
        return 'use'
    if exc in ('AgentTimeoutError', 'VerifierTimeoutError'):
        return 'use' if policy == 'score' and (has_verdict or exc == 'VerifierTimeoutError') else 'mask'
    if has_verdict:
        return 'use'     # like the stock generator: any other exception after the verifier ran does not discard its verdict
    return 'retry' if attempt + 1 < attempts else 'mask'


def has_rollout(results: Any) -> bool:
    rollout = getattr(getattr(results, 'agent_result', None), 'rollout_details', None)
    return bool(rollout and len(rollout) >= 1 and len(rollout[0].get('completion_token_ids', [])) > 0)


def completion_tokens(rollout: Any) -> int:
    try:
        return sum(len(t) for t in rollout[0]['completion_token_ids'])
    except (TypeError, KeyError, IndexError):
        return 0


# ------------------------------------------------------------------------------------------------------------ one trajectory
def _warn(message: str) -> None:
    try:
        from loguru import logger
        logger.warning(message)
    except Exception:
        print(message, flush=True)


def masked_summary(domain: str, task: str, rep: int | None, reason: str, **extra: Any) -> dict:
    return {'domain': domain, 'task': task, 'rep': rep, 'reward': 0.0, 'metric': None, 'passed': False, 'masked': True, 'mask_reason': reason,
            'stop': reason, 'components': {}, 'judge': None, 'artifact': None, 'screenshot': None, 'turns': 0, 'tokens': 0, **extra}


async def run_trajectory(deps: Any, prompt: str, trajectory_id: Any, *, cache_salt: str | None = None, temperature: float | None = None,
                         env_extra: dict | None = None, phase: str = 'train') -> tuple[dict, dict]:
    """One trajectory: run (with the retry policy), read the verifier, judge if asked. Returns (SkyRL output fields, summary).

    ``deps``: .template, .settings, .caps(domain) -> dict, .rate_limiter (async ctx), .judge (or None), .create_trial(config) -> awaitable
    trial with ``await trial.run()``, .finish_session(session_id) -> awaitable."""
    from uuid import uuid4
    started = time.monotonic()
    task_path = str(prompt)
    domain, rep = domain_of(task_path, env_extra), (env_extra or {}).get('rep')
    name = Path(task_path).name
    meta, toml = read_task(task_path)
    settings = deps.settings
    results, trial, verdict = None, None, 'mask'
    for attempt in range(int(settings['max_trial_attempts'])):
        session_id = uuid4().hex
        try:
            config = build_trial_config(deps.template, task_path, session_id=session_id, meta=meta, toml=toml, temperature=temperature,
                                        cache_salt=cache_salt, caps=deps.caps(domain))
            trial = await deps.create_trial(config)
            async with deps.rate_limiter:
                results = await trial.run()
            verdict = decide(results, attempt=attempt, attempts=int(settings['max_trial_attempts']), policy=settings['timeout_policy'])
        except Exception:   # CancelledError (the deadline watchdog) is a BaseException and propagates
            results, verdict = None, 'retry' if attempt + 1 < int(settings['max_trial_attempts']) else 'mask'
        finally:
            try:
                await deps.finish_session(session_id)
            except Exception:
                pass
        if verdict != 'retry':
            break
    seconds = time.monotonic() - started
    if verdict == 'mask' or results is None:
        reason = {'AgentTimeoutError': 'agent_timeout', 'VerifierTimeoutError': 'verifier_timeout'}.get(exception_type(results), 'error') if results is not None else 'error'
        if phase == 'eval' and reason in ('agent_timeout', 'verifier_timeout') and settings['timeout_policy'] == 'score':
            # evaluation needs no tokens: a policy that ran out of time failed the task, it is not an infrastructure loss
            return _masked(trajectory_id, seconds), {**masked_summary(domain, name, rep, reason, seconds=round(seconds, 1)), 'masked': False, 'mask_reason': None, 'metric': 0.0}
        _warn(f'mimo masked {domain}/{name} rep={rep}: {reason} ({exception_type(results) if results is not None else "trial raised"})')
        return _masked(trajectory_id, seconds), masked_summary(domain, name, rep, reason, seconds=round(seconds, 1))
    exc = exception_type(results)
    rollout = getattr(getattr(results, 'agent_result', None), 'rollout_details', None) if has_rollout(results) else None
    if rollout is None and phase != 'eval':   # training needs the tokens; evaluation only needs the verdict
        reason = 'agent_timeout' if exc == 'AgentTimeoutError' else 'no_rollout'
        _warn(f'mimo masked {domain}/{name} rep={rep}: {reason} (exception {exc})')
        return _masked(trajectory_id, seconds), masked_summary(domain, name, rep, reason, seconds=round(seconds, 1))
    files = read_verifier(trial_dir_of(trial, results))
    details = files['details']
    summary = {'domain': domain, 'task': name, 'rep': rep, 'masked': False, 'mask_reason': None, 'components': details.get('components') or {},
               'turns': int((getattr(getattr(results, 'agent_result', None), 'metadata', None) or {}).get('n_episodes', 0) or 0), 'tokens': completion_tokens(rollout) if rollout else 0,
               'seconds': round(seconds, 1), 'judge': None,
               'stop': STOP_NAME.get(exc, 'complete')}
    summary['artifact'], summary['screenshot'] = read_artifacts(files['dir'], meta.get('artifacts') or [], int(settings['summary_chars']))
    if not details and exc is None:   # only a trial that finished cleanly: timeouts/context errors keep their own policy
        # The shared verifier entrypoint always leaves a non-empty details.json. Its absence means the verifier did not run (the
        # first version called /usr/bin/python3, which two images lack: every music/general trial silently scored 0). Mask it, loudly.
        details = {'infra_error': True, 'error': 'verifier_no_details'}
        summary['mask_detail'] = 'verifier produced no details.json'
    if details.get('infra_error'):
        _warn(f'mimo masked {domain}/{name} rep={rep}: infra_error ({str(details.get("error"))[:160]})')
        return _masked(trajectory_id, seconds), {**summary, **masked_summary(domain, name, rep, 'infra_error', seconds=round(seconds, 1)),
                                                 'artifact': summary['artifact'], 'screenshot': summary['screenshot']}
    if exc in ZERO_EXC:
        reward = 0.0
    else:
        verifier = getattr(results, 'verifier_result', None)
        reward = float(verifier.rewards['reward']) if verifier and verifier.rewards.get('reward') is not None else (files['reward'] or 0.0)
    metric = details.get('metric')
    metric = reward if metric is None else float(metric)
    request = files['judge_request']
    if request is not None and request.get('force_zero') and exc not in ZERO_EXC:
        reward = metric = 0.0   # a gate failed in the sandbox: reward 0 without spending a judge call (and never masked)
        summary['judge'] = {'final': 0.0, 'scores': {}, 'rationale': 'force_zero: verifier gate failed', 'meta': {'skipped': True, 'latency_s': 0.0, 'cost_usd': 0.0, 'calls': 0}}
        request = None
    if request is not None and exc not in ZERO_EXC:
        if deps.judge is None:
            return _masked(trajectory_id, seconds), masked_summary(domain, name, rep, 'judge', error='no judge configured', seconds=round(seconds, 1))
        try:
            judged = await deps.judge.score(request, workdir=files['dir'], eval=phase == 'eval')
        except judge_mod.JudgeUnavailable as error:
            return _masked(trajectory_id, seconds), masked_summary(domain, name, rep, 'judge', error=str(error)[:200], seconds=round(seconds, 1))
        reward = metric = judged['final']
        if phase == 'eval' and request.get('kind') == 'rubric':   # persisted so `strl verify` can re-score a sample of judged records offline
            summary['judge_request'] = request
        summary['judge'] = {'final': judged['final'], 'scores': judged['scores'], 'rationale': judged['rationale'], 'meta': judged['meta']}
    summary.update(reward=reward, metric=metric, passed=bool(details.get('passed', metric >= 0.999)))
    return {'rollout_details': rollout, 'reward': float(reward), 'num_turns': summary['turns'],
            'stop_reason': ('context_length' if exc == 'ContextLengthExceededError' else 'output_length' if exc == 'OutputLengthExceededError'
                                    else 'complete') if rollout else 'error', 'e2e_time': seconds, 'trajectory_id': trajectory_id}, summary


def _masked(trajectory_id: Any, seconds: float) -> dict:
    return {'rollout_details': None, 'reward': 0.0, 'num_turns': 0, 'stop_reason': 'error', 'e2e_time': seconds, 'trajectory_id': trajectory_id}


# ------------------------------------------------------------------------------------------------------------ deadline
async def gather_with_deadline(factories: list[Callable[[], Any]], deadline_sec: float | None) -> list[tuple[str, Any]]:
    """Run coroutine factories concurrently. Returns [('done', result) | ('deadline', None) | ('error', exception)] in input order;
    tasks still running at the deadline are cancelled (and awaited, so their sandboxes are torn down) and reported 'deadline'."""
    tasks = [asyncio.ensure_future(f()) for f in factories]
    if tasks:
        _, pending = await asyncio.wait(tasks, timeout=deadline_sec)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    out: list[tuple[str, Any]] = []
    for task in tasks:
        if task.cancelled():
            out.append(('deadline', None))
        elif task.exception() is not None:
            out.append(('error', task.exception()))
        else:
            out.append(('done', task.result()))
    return out


# ------------------------------------------------------------------------------------------------------------ metrics
def rollout_metrics(summaries: list[dict], group_ids: list[str] | None = None, *, phase: str = 'train') -> dict[str, float]:
    """Per-domain generate/<d>/* scalars plus environment/<d>/masked_group_fraction and the overall masked_group_fraction.
    ``group_ids[i]`` is trajectory i's prompt id (a group is masked when any of its trials is masked)."""
    out: dict[str, float] = {}
    groups = group_ids or [str(i) for i in range(len(summaries))]
    for domain in [d for d in DOMAINS if any(s['domain'] == d for s in summaries)]:
        idx = [i for i, s in enumerate(summaries) if s['domain'] == domain]
        rows = [summaries[i] for i in idx]
        valid = [s for s in rows if not s['masked']]
        p = f'generate/{domain}'
        out[f'{p}/num_trials'] = float(len(rows))
        out[f'{p}/num_masked'] = float(len(rows) - len(valid))
        out[f'{p}/num_timeout'] = float(sum(s['stop'] in ('agent_timeout', 'verifier_timeout') for s in rows))
        out[f'{p}/num_deadline'] = float(sum(s.get('mask_reason') == 'deadline' for s in rows))
        out[f'{p}/num_context_exceeded'] = float(sum(s['stop'] == 'context_length' for s in rows))
        out[f'{p}/judge_masked'] = float(sum(s.get('mask_reason') == 'judge' for s in rows))
        if valid:
            out[f'{p}/reward_mean'] = sum(float(s['reward']) for s in valid) / len(valid)
            out[f'{p}/pass_rate'] = sum(bool(s['passed']) for s in valid) / len(valid)
            out[f'{p}/avg_turns'] = sum(s['turns'] for s in valid) / len(valid)
            out[f'{p}/tokens'] = sum(s['tokens'] for s in valid) / len(valid)
        judged = [s['judge'] for s in valid if s.get('judge')]
        if judged:
            out[f'{p}/judge_latency_s'] = sum(j['meta']['latency_s'] for j in judged) / len(judged)
            out[f'{p}/judge_cost_usd'] = sum(j['meta']['cost_usd'] for j in judged)
        gids = {groups[i] for i in idx}
        masked = {groups[i] for i in idx if summaries[i]['masked']}
        out[f'environment/{domain}/masked_group_fraction'] = len(masked) / len(gids) if gids else 0.0
    all_groups = set(groups)
    out['environment/masked_group_fraction'] = len({g for g, s in zip(groups, summaries) if s['masked']}) / len(all_groups) if all_groups else 0.0
    return out


def public_summary(summary: dict) -> dict:
    """JSON-safe, bounded copy for the eval dump (env_extras['mimo']) and the tables: the judge keeps verdict, rationale and cost."""
    judge = summary.get('judge')
    return {**{k: v for k, v in summary.items() if k != 'judge'},
            'judge': judge and {'final': judge['final'], 'scores': judge['scores'], 'rationale': judge['rationale'][:600],
                                'cost_usd': judge['meta']['cost_usd'], 'latency_s': judge['meta']['latency_s']}}


def remember(phase: str, trajectory_id: Any, summary: dict, env_extra: dict | None = None) -> None:
    """Keep the summary in memory (bounded; tracking reads it) and in the batch's env_extras, which SkyRL dumps with the eval rows."""
    SUMMARIES[f'{phase}:{trajectory_id.to_string()}'] = summary
    while len(SUMMARIES) > SUMMARY_STORE_LIMIT:
        SUMMARIES.popitem(last=False)
    if env_extra is not None:
        env_extra['mimo'] = public_summary(summary)


def summaries_for(trajectory_ids: list, phase: str) -> list[dict | None]:
    return [SUMMARIES.get(f'{phase}:{t.to_string()}') for t in trajectory_ids]


# ------------------------------------------------------------------------------------------------------------ the SkyRL subclass
def make_generator_class():
    from examples.train_integrations.harbor.harbor_generator import (HarborGenerator, HarborTrajectoryOutput,
                                                                     build_step_wise_generator_output)
    from loguru import logger
    from tqdm import tqdm

    class MimoHarborGenerator(HarborGenerator):
        def __init__(self, *args: Any, mimo: dict | None = None, judge: Any = None, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._mimo = {**DEFAULT_SETTINGS, **(mimo or {})}
            self._judge = judge
            self._caps = (mimo or {}).get('caps') or {}

        def _deps(self) -> Any:
            async def create_trial(config: dict) -> Any:
                from harbor.models.trial.config import TrialConfig
                from harbor.trial.trial import Trial
                return await Trial.create(TrialConfig.model_validate(config))

            return SimpleNamespace(template=self._harbor_trial_config_template, settings=self._mimo, rate_limiter=self._rate_limiter,
                                   judge=self._judge, create_trial=create_trial, finish_session=self.inference_engine_client.finish_session,
                                   caps=lambda domain: self._caps.get(domain) or {})

        async def _harbor_agent_loop(self, prompt, trajectory_id, cache_salt=None, *, temperature=None, env_extra=None, phase='train'):
            out, summary = await run_trajectory(self._deps(), prompt, trajectory_id, cache_salt=cache_salt, temperature=temperature,
                                                env_extra=env_extra, phase=phase)
            remember(phase, trajectory_id, summary, env_extra)
            output = HarborTrajectoryOutput(**out)
            output.mimo_summary = summary
            return output

        async def generate(self, input_batch, disable_tqdm: bool = False):
            prompts, trajectory_ids = input_batch['prompts'], input_batch['trajectory_ids']
            if trajectory_ids is None:
                raise ValueError('`trajectory_ids` is required in the input batch')
            if len(prompts) != len(trajectory_ids):
                raise ValueError(f'Prompt count ({len(prompts)}) doesn\'t match trajectory_ids count ({len(trajectory_ids)})')
            phase = getattr(input_batch.get('batch_metadata'), 'training_phase', 'train')
            sampling = input_batch.get('sampling_params') or {}
            temperature = sampling.get('temperature')
            if temperature is None and phase == 'eval':
                temperature = getattr(getattr(self.generator_cfg, 'eval_sampling_params', None), 'temperature', None)
            extras = input_batch.get('env_extras') or [None] * len(prompts)
            cache_salt = self._compute_cache_salt()
            progress = tqdm(disable=disable_tqdm, total=len(prompts), desc='Generating Trajectories', miniters=max(1, len(prompts) // 10), mininterval=5)

            def factory(i: int):
                async def run():
                    out = await self._harbor_agent_loop(prompts[i], trajectory_ids[i], cache_salt, temperature=temperature, env_extra=extras[i], phase=phase)
                    progress.update(1)
                    return out
                return run

            started = time.monotonic()
            try:
                finished = await gather_with_deadline([factory(i) for i in range(len(prompts))], self._mimo.get('step_deadline_sec'))
            finally:
                progress.close()
            outputs, summaries = [], []
            for i, (status, value) in enumerate(finished):
                if status == 'done':
                    outputs.append(value)
                    summaries.append(value.mimo_summary)
                    continue
                if status == 'error':
                    logger.warning(f'trajectory {trajectory_ids[i]} raised {type(value).__name__}: {value}')
                summary = masked_summary(domain_of(prompts[i], extras[i]), Path(str(prompts[i])).name, (extras[i] or {}).get('rep'),
                                         'deadline' if status == 'deadline' else 'error', seconds=round(time.monotonic() - started, 1))
                remember(phase, trajectory_ids[i], summary, extras[i])
                outputs.append(HarborTrajectoryOutput(trajectory_id=trajectory_ids[i], rollout_details=None, stop_reason='error', e2e_time=time.monotonic() - started))
                summaries.append(summary)
            output = build_step_wise_generator_output(outputs, overlong_filtering=self.generator_cfg.apply_overlong_filtering)
            output['rollout_metrics'] = {**(output.get('rollout_metrics') or {}),
                                         **rollout_metrics(summaries, [t.instance_id for t in trajectory_ids], phase=phase)}
            return output

    return MimoHarborGenerator
