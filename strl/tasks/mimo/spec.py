"""`mimo`: XiaomiMiMo/MiMo-V2.6-RL-oss as Harbor tasks, one trainer, five domains (music, cyber, code, general, webdev).

Built on the coding_harbor machinery (SkyRL HarborGenerator, Modal sandboxes, step-wise GRPO). What is new: a domain-stratified
batch sampler, an LLM judge for the judged domains (outside the sandbox), per-domain held-out evaluation (every held-out task run
``eval_repeats`` times with the training agent) and the macro task metric. The reward starts from the native grader (verifier_score, exactly once);
extra `module:function` terms shape the TRAINING reward only (evaluation always reports the native task metric). Contract with the exporter/graders: strl/tasks/mimo/__init__.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..coding import harbor as coding_harbor
from ..coding.spec import CodingHarborSpec
from . import data, judge, metrics
from .sampler import DomainBatchSampler, quotas

JUDGED = ('general', 'webdev')
CRITIC_MODES = {
    'music': ('invalid_abc', 'wrong_key_or_meter', 'wrong_length', 'voice_count_mismatch', 'range_or_rule_violation'),
    'cyber': ('no_crash', 'wrong_function', 'wrong_bug_type', 'no_submission', 'invalid_poc'),
    'code': ('no_patch', 'wrong_file', 'incomplete_fix', 'breaks_existing_tests', 'did_not_run_tests'),
    'general': ('missed_rubric_item', 'hallucinated_fact', 'wrong_output_format', 'tool_misuse', 'incomplete_deliverable'),
    'webdev': ('blank_render', 'layout_broken', 'missing_sections', 'no_visual_assets', 'build_error'),
}
BASE_CRITIC_MODES = ('misread_task', 'wrong_algorithm', 'edge_case', 'syntax_or_runtime_error', 'format_violation', 'truncated_or_incomplete',
                     'timeout_or_inefficient', 'reward_hacking_attempt', 'other')


def verifier_score(result: dict, ctx: Any) -> float:
    """The native grader's reward (reward.txt, or the judged final). Present so `strl plan` can validate reward.components."""
    return float(result.get('reward', 0.0))


def mcfg(cfg: dict) -> dict:
    return data.mimo_config(cfg)


def per_step_quota(cfg: dict) -> dict[str, int]:
    m = mcfg(cfg)
    return quotas({d: m['domains'][d]['weight'] for d in data.selected_domains(m)}, cfg['algorithm']['prompts_per_step'])


def steps_per_epoch(cfg: dict) -> int:
    m, q = mcfg(cfg), per_step_quota(cfg)
    return min(m['domains'][d]['train'] // q[d] for d in q if q[d]) if q else 0


class MimoSpec(CodingHarborSpec):
    name = 'mimo'
    generator_kind = 'harbor'
    metric = 'macro_task_metric'
    needs_modal = True
    needs_litellm = True
    uses_coding_executor = False
    default_reward = [{'fn': 'verifier_score', 'weight': 1.0}]
    eval_module = ''
    extra_columns = metrics.EXTRA_COLUMNS
    failure_classes = ('agent_timeout', 'verifier_timeout', 'context_length', 'infra_error', 'judge', 'deadline', 'error', 'no_rollout', 'wrong')
    preflight_imports = ('modal', 'harbor', 'examples.train_integrations.harbor.harbor_generator', 'strl.tasks.mimo.generator')
    dry_run_seconds = 1800

    # ------------------------------------------------------------------ planning
    def validate_experiment(self, cfg: dict) -> list[str]:
        problems = data.config_problems(cfg) + judge.settings_problems(cfg)
        if problems:
            return problems
        m = mcfg(cfg)
        a, e, b = cfg['algorithm'], cfg['evaluation'], cfg['budget']
        if not data.selected_domains(m):
            problems.append('environment.mimo.domains: at least one domain needs train > 0')
        names = [c['fn'] for c in cfg['reward']['components'] if c.get('weight', 1.0)]
        if names.count('verifier_score') != 1:
            problems.append('mimo rewards start from the native graders (verifier reward / LLM judge): reward.components must contain '
                            f'{{fn: verifier_score}} exactly once with a non-zero weight, got {names}')
        extra = [n for n in names if n != 'verifier_score' and ':' not in n]
        if extra:
            problems.append(f'mimo shaping terms must be custom "module:function" components (user_rewards/); got {extra}. They are called as '
                            'fn(result, ctx, **params) with result = {domain, task, reward (native), metric, passed, components, turns, tokens, seconds, stop, artifact}')
        if e.get('final_eval', True):
            problems.append('mimo has no offline final eval (agents run in sandboxes): set evaluation.final_eval: false; the held-out '
                            'per-domain eval runs inside training (eval/<domain>/task_metric, eval/macro_task_metric)')
        services = cfg.get('services') or {}
        if not (services.get('modal') or {}).get('credentials_uri') or not (services.get('modal') or {}).get('sandbox_secret'):
            problems.append('services.modal.credentials_uri and services.modal.sandbox_secret are required (Harbor Modal sandboxes)')
        judged = [d for d in JUDGED if d in m['domains'] and (m['domains'][d]['train'] or m['domains'][d]['test'])]
        if judged and not (services.get('litellm') or {}).get('key_name'):
            problems.append(f'services.litellm.key_name is required: the LLM judge scores {judged}')
        if judged and not (judge.judge_settings(cfg)['model']):
            problems.append('environment.mimo.judge.model must be set when general/webdev are selected')
        if (m['domains'].get('webdev') or {}).get('train') and not (judge.judge_settings(cfg)['vision_model'] or judge.judge_settings(cfg)['model']):
            problems.append('webdev needs a vision-capable judge: set environment.mimo.judge.vision_model')
        hb = cfg['environment'].get('harbor') or {}
        caps = [d['max_prompt_tokens'] for d in m['domains'].values() if d['max_prompt_tokens']]
        if caps and max(caps) > a['max_prompt_tokens']:
            problems.append(f'environment.mimo.domains.*.max_prompt_tokens={max(caps)} exceeds algorithm.max_prompt_tokens={a["max_prompt_tokens"]}')
        need = max(coding_harbor.FIRST_TURN_MAX_TOKENS, a['max_prompt_tokens']) + 2 * a['max_generate_length']
        if coding_harbor.max_seq_len(cfg) < need:
            problems.append(f'environment.harbor.max_seq_len={coding_harbor.max_seq_len(cfg)} is below {need} (= max(first Terminus turn '
                            f'{coding_harbor.FIRST_TURN_MAX_TOKENS}, max_prompt_tokens) + 2 x max_generate_length): trials would overflow their context')
        deadline = m_raw(cfg).get('step_deadline_sec')
        stall = int(b.get('stall_seconds', 600))
        if deadline is not None:
            if deadline + 600 > stall:
                problems.append(f'environment.mimo.step_deadline_sec={deadline} + 600 s (logprobs + training) exceeds budget.stall_seconds={stall}: '
                                'the watchdog would kill a healthy step')
            from .export import DOMAIN_SPECS   # the effective timeouts are the exporter's defaults unless a cap is configured
            for name in data.selected_domains(m):
                d = m['domains'][name]
                agent = d['max_agent_sec'] or DOMAIN_SPECS[name].agent_timeout_sec
                verifier = d['max_verifier_sec'] or DOMAIN_SPECS[name].verifier_timeout_sec
                if agent + verifier + 120 > deadline:
                    problems.append(f'environment.mimo.domains.{name}: agent {agent:g} s + verifier {verifier:g} s + 120 exceeds step_deadline_sec={deadline}: '
                                    'long but healthy trials would be cancelled and masked as "deadline" instead of scored (timeout_policy). Lower '
                                    f'domains.{name}.max_agent_sec / max_verifier_sec or raise step_deadline_sec (and budget.stall_seconds)')
        elif int(hb.get('agent_timeout_sec', 600)) * 2 + 300 > stall:
            problems.append(f'budget.stall_seconds={stall} is below 2 x environment.harbor.agent_timeout_sec + 300; set environment.mimo.step_deadline_sec')
        if int(m_raw(cfg).get('eval_repeats', 1)) > 1 and float(a.get('eval_temperature') or 0.0) == 0.0:
            problems.append('environment.mimo.eval_repeats > 1 with algorithm.eval_temperature 0: greedy repeats are near-identical (the extra eval cost buys '
                            'nothing; small models also loop when greedy). Set algorithm.eval_temperature (e.g. 1.0, the training temperature) or eval_repeats: 1')
        if m_raw(cfg).get('timeout_policy', 'score') not in ('score', 'mask'):
            problems.append("environment.mimo.timeout_policy must be 'score' or 'mask'")
        sel = data.selected_domains(m)
        if sel:
            if a['prompts_per_step'] < len(sel):
                problems.append(f'algorithm.prompts_per_step={a["prompts_per_step"]} is below the {len(sel)} selected domains: some would never be sampled')
            else:
                short = {d: (m['domains'][d]['train'], q) for d, q in per_step_quota(cfg).items() if m['domains'][d]['train'] < q}
                if short:
                    problems.append(f'domains with fewer train tasks than their per-step quota (train, quota): {short}; raise train or lower prompts_per_step')
        return problems

    def reward_functions(self):
        return {'verifier_score': verifier_score}

    def tags(self, cfg: dict) -> list[str]:
        return ['mimo', *[f'domain:{d}' for d in data.selected_domains(mcfg(cfg))]]

    def plan_warnings(self, cfg: dict) -> list[str]:
        m, out = mcfg(cfg), []
        weights = {d: m['domains'][d]['weight'] for d in data.selected_domains(m)}
        out.append(f'MIMO: per-step quota {per_step_quota(cfg)} (weights {weights}); {steps_per_epoch(cfg)} step(s) per epoch before a domain repeats tasks; '
                   f'held-out eval = {self.validation_size(cfg)} trials ({sum(1 for d in m["domains"].values() if d["test"])} domains x test x {m["eval_repeats"]} repeats).')
        judged = [d for d in JUDGED if d in weights]
        if judged:
            out.append(f'JUDGE: {judged} are scored by {judge.judge_settings(cfg)["model"]} outside the sandbox (votes {judge.judge_settings(cfg)["votes"]} train / '
                       f'{judge.judge_settings(cfg)["eval_votes"]} eval); a judge outage masks groups (environment/<domain>/masked_group_fraction), it never scores 0.')
        if cfg['algorithm'].get('grpo_norm_by_std', True):
            out.append('GRPO: grpo_norm_by_std is on: continuous judged rewards with tiny group variance get amplified; MiMo-style runs set it false.')
        if (m_raw(cfg).get('step_deadline_sec') or 0) == 0:
            out.append('DEADLINE: environment.mimo.step_deadline_sec is unset: one hung sandbox can stall a whole step up to budget.stall_seconds.')
        return out

    # ------------------------------------------------------------------ data
    def prepare_data(self, cfg: dict, **kwargs: Any) -> dict:
        return data.prepare(cfg, **{k: v for k, v in kwargs.items() if k in {'exporter', 'screen', 'tokenizer', 'raw', 'screened'}})

    def assert_manifest(self, cfg: dict) -> dict:
        return data.assert_manifest(cfg)

    def _manifest_or_none(self, cfg: dict) -> dict | None:
        try:
            return data.load_manifest(cfg)
        except data.DataError:
            return None

    def data_files(self, cfg: dict, stage: str):
        """Exact task directories of THIS run (from the manifest, truncated to the configured counts). Held-out tasks are listed
        ``eval_repeats`` times (canary: once); HarborTaskDataset keeps the duplicates and ``rep`` counts occurrences. Before
        `strl prepare` (planning) the split directories are returned instead."""
        manifest = self._manifest_or_none(cfg)
        val = 'canary' if stage == 'canary' else 'test'
        if manifest is None:
            root = Path(cfg['run']['data_dir']) / 'tasks'
            return [str(root / 'train')], [str(root / val)]
        repeats = 1 if stage == 'canary' else mcfg(cfg)['eval_repeats']
        return data.split_paths(cfg, 'train', manifest), data.split_paths(cfg, val, manifest, repeats=repeats)

    def validation_size(self, cfg: dict) -> int:
        m = mcfg(cfg)
        return sum(m['domains'][d]['test'] for d in data.eval_domains(m)) * m['eval_repeats']

    def canary_size(self, cfg: dict) -> int:
        m = mcfg(cfg)
        return sum(min(m['canary'], m['domains'][d]['test']) for d in data.eval_domains(m))

    def final_eval_ids(self, cfg: dict) -> list[str]:
        selection = data.runtime_selection(cfg)['test']
        return [f'{data.task_name(d, i)}#r{rep}' for rep in range(mcfg(cfg)['eval_repeats']) for d in data.DOMAINS if d in selection for i in selection[d]]

    # ------------------------------------------------------------------ SkyRL wiring
    def skyrl_overrides(self, cfg: dict, stage: str) -> dict[str, Any]:
        out = coding_harbor.skyrl_overrides(cfg)
        a = cfg['algorithm']
        epochs = max(1, math.ceil(a['max_steps'] / steps_per_epoch(cfg))) if steps_per_epoch(cfg) else 1
        out.update({'trainer.eval_batch_size': max(1, self.canary_size(cfg) if stage == 'canary' else self.validation_size(cfg)),
                    'trainer.epochs': max(20, epochs), 'generator.eval_n_samples_per_prompt': 1})
        if stage == 'canary':    # one eval per generation costs ~8 min at smoke size: eval at step 0 (base) and after the 2nd update only
            out['trainer.eval_interval'] = 2
        return out

    def build_dataset(self, experiment: dict, cfg: Any, split: str) -> Any:
        coding_harbor.config_class()      # puts the SkyRL root on sys.path
        files = cfg.data.train_data if split == 'train' else cfg.data.val_data
        if split != 'train' and not (cfg.trainer.eval_interval > 0 and files):
            return None
        dataset = make_dataset_class()(data_files=list(files))
        if split == 'train' and len(dataset) < cfg.trainer.train_batch_size:
            raise RuntimeError(f'mimo train set has {len(dataset)} tasks < train_batch_size {cfg.trainer.train_batch_size}')
        return dataset

    def train_batch_sampler(self, experiment: dict, dataset: Any, cfg: Any) -> Any:
        m = mcfg(experiment)
        weights = {d: m['domains'][d]['weight'] for d in data.selected_domains(m)}
        domains = [data.task_domain(p) for p in dataset.task_paths]
        return DomainBatchSampler(domains, weights, cfg.trainer.train_batch_size, seed=cfg.trainer.seed)

    def build_generator(self, experiment: dict, cfg: Any, tokenizer: Any, inference_engine_client: Any) -> Any:
        from .generator import generator_settings, make_generator_class, mimo_template, native_weight, reward_terms
        coding_harbor.install_compat()
        m = mcfg(experiment)
        caps = {d: {k: v[k] for k in ('max_agent_sec', 'max_verifier_sec', 'max_turns') if v.get(k)} for d, v in m['domains'].items()}
        judged = any(d in m['domains'] for d in JUDGED)
        return make_generator_class()(
            generator_cfg=cfg.generator, harbor_cfg=mimo_template(experiment), inference_engine_client=inference_engine_client, tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
            mimo={**generator_settings(experiment), 'caps': caps, 'reward_terms': reward_terms(experiment), 'native_weight': native_weight(experiment)},
            judge=judge.Judge(experiment) if judged else None)

    # ------------------------------------------------------------------ evidence
    def generator_contract(self, cfg: dict) -> dict:
        """CPU: for one exported task per domain the per-task trial config validates as a Harbor TrialConfig; the judge contract
        (combine / masking) holds."""
        from harbor.models.trial.config import TrialConfig
        from .generator import build_trial_config, mimo_template, read_task
        template = mimo_template(cfg)
        template['agent'].update(model_name='hosted_vllm/policy')
        template['agent']['kwargs']['api_base'] = 'http://127.0.0.1:8000/v1'
        receipt: dict[str, Any] = {'domains': {}, 'actual_reward_passed': True}
        for domain, task in self._one_per_domain(cfg).items():
            meta, toml = read_task(task)
            caps = {k: v for k, v in data.mimo_config(cfg)['domains'][domain].items() if k in ('max_agent_sec', 'max_verifier_sec', 'max_turns') and v}
            config = build_trial_config(template, task, session_id='contract', meta=meta, toml=toml, temperature=1.0, caps=caps)   # the caps real trials get
            TrialConfig.model_validate(config)
            receipt['domains'][domain] = {'agent': config['agent'].get('name') or config['agent'].get('import_path'), 'max_turns': config['agent']['kwargs']['max_turns'],
                                          'agent_timeout_sec': config['agent'].get('override_timeout_sec') or (toml.get('agent') or {}).get('timeout_sec')}
        request = {'kind': 'rubric', 'base_score': 0.25, 'total_weight': 1.0, 'items': [{'id': 'a', 'weight': 0.75}]}
        if abs(judge.combine(request, {'a': 1.0}) - 1.0) > 1e-9 or abs(judge.combine(request, {'a': 0.0}) - 0.25) > 1e-9:
            raise RuntimeError('judge combine contract failed')
        receipt['judge_combine_passed'] = True
        return receipt

    def _one_per_domain(self, cfg: dict) -> dict[str, str]:
        selection = data.runtime_selection(cfg)
        root, out = Path(cfg['run']['data_dir']), {}
        for domain in data.DOMAINS:
            for split in ('canary', 'train'):
                if selection[split].get(domain):
                    out[domain] = str(data.task_dir(root, split, domain, selection[split][domain][0]))
                    break
        return out

    def dry_run_checks(self, cfg: dict) -> dict:
        """From the pod: a nop agent on one exported task per domain must run its verifier cleanly in a Modal sandbox (reward 0 for the
        rule domains, no infra error), and the judge selftest (one real rubric call, one real vision call) must pass."""
        import asyncio
        coding_harbor.install_compat()
        from .generator import mimo_template
        tasks = self._one_per_domain(cfg)
        template = mimo_template(cfg)
        configs = {d: coding_harbor.build_trial_config(template, t, model_name=None, api_base=None, agent_name='nop') for d, t in tasks.items()}
        results = dict(zip(configs, asyncio.run(coding_harbor._run(list(configs.values())))))
        from .generator import verifier_ran
        ran = {d: verifier_ran(r.get('trial_dir')) for d, r in results.items()}
        receipt: dict[str, Any] = {'nop': {d: {'reward': r['reward'], 'exception': r['exception'], 'seconds': r['seconds'],
                                               'verifier_ran': ran[d][0], 'verifier_note': ran[d][1]} for d, r in results.items()}}
        bad = {d: r for d, r in results.items() if r['reward'] is None or r['exception'] or (d not in JUDGED and r['reward'] != 0.0)}
        if bad:
            raise RuntimeError(f'Harbor nop verifier check failed: {bad}')
        dead = {d: why for d, (ok, why) in ran.items() if not ok}   # reward 0 alone cannot tell a correct verifier from one that never ran
        if dead:
            raise RuntimeError(f'verifier did not run (a nop reward of 0 is not evidence): {dead}')
        selftest = judge.selftest(judge.judge_settings(cfg))
        receipt['judge'] = selftest
        if not all(v['ok'] for v in selftest.values()):
            raise RuntimeError(f'judge selftest failed: {selftest}')
        return receipt

    def eval_record(self, item: dict, trainer: Any, step: int) -> dict:
        extras = item.get('env_extras') or {}
        mimo = extras.get('mimo') or {}
        reward = item.get('score', 0.0)
        reward = float(sum(reward)) if isinstance(reward, list) else float(reward)
        task = Path(extras.get('task_dir') or extras.get('task') or '').name
        rep = int(extras.get('rep', mimo.get('rep') or 0))
        domain = extras.get('domain') or mimo.get('domain') or data.task_domain(task)
        masked = bool(mimo.get('masked')) if mimo else item.get('stop_reason') == 'error'
        judged = mimo.get('judge') or {}
        metric = mimo.get('metric')
        return {'step': step, 'sample_id': f'{task}#r{rep}', 'prompt': task, 'completion': (item.get('output_response') or '')[-4000:], 'ground_truth': task,
                'parsed_answer': (mimo.get('artifact') or '')[:200] or None, 'correct': bool(mimo.get('passed')) if mimo else reward >= 1.0, 'reward': reward,
                'token_count': mimo.get('tokens'), 'finish_reason': item.get('stop_reason'), 'domain': domain, 'rep': rep, 'task': task,
                'task_metric': (reward if metric is None and not masked else metric), 'passed': bool(mimo.get('passed')) if mimo else reward >= 1.0,
                'masked': masked, 'mask_reason': mimo.get('mask_reason'), 'stop': mimo.get('stop'), 'judge_rationale': judged.get('rationale'),
                'judge_variance': None, 'verifier_components': json.dumps(mimo.get('components') or {}, sort_keys=True)[:1000],
                'artifact': (mimo.get('artifact') or '')[:metrics.ARTIFACT_CHARS], 'screenshot': mimo.get('screenshot'), 'turns': 1,
                'extra': {'artifact_full': mimo.get('artifact'), **({'judge_request': mimo['judge_request']} if mimo.get('judge_request') else {})}}   # 'extra' is never a table column (EvidenceCallback.on_eval_end)

    def collapse_eval_records(self, records: list[dict]) -> list[dict]:
        """One record per (task, rep): per-turn rows are contiguous; the verifier reward sits on the last turn."""
        merged: dict[str, dict] = {}
        for record in records:
            current = merged.get(record['sample_id'])
            if current is None:
                merged[record['sample_id']] = dict(record)
            else:
                merged[record['sample_id']] = dict(record, reward=current['reward'] + record['reward'], turns=current['turns'] + 1)
        return list(merged.values())

    def eval_metrics(self, records: list[dict]) -> dict[str, float]:
        out = metrics.domain_metrics(records, 'eval')
        for domain in data.DOMAINS:   # eval/invalid_answer_rate is the all-domain version of this: a trial that left no artifact for the verifier
            rows = [r for r in records if r.get('domain') == domain and not r.get('masked')]
            if rows:
                out[f'eval/{domain}/no_artifact_rate'] = sum(not (r.get('artifact') or r.get('screenshot')) for r in rows) / len(rows)
        if 'eval/macro_task_metric' in out:
            out['eval/task_metric'] = out['eval/macro_task_metric']
        return out

    def media_table(self, records: list[dict]):
        return metrics.media_table(records)

    def metric_by_source(self, base: list[dict], trained: list[dict]) -> dict[str, dict[str, float]]:
        """Per-domain base vs trained task metric (masked rows dropped on both sides) plus the macro average over present domains."""
        out: dict[str, dict[str, float]] = {}
        for domain in data.DOMAINS:
            b = [r for r in base if r.get('domain') == domain and not r.get('masked')]
            t = [r for r in trained if r.get('domain') == domain and not r.get('masked')]
            if b and t:
                bm, tm = (sum(r['task_metric'] for r in b) / len(b), sum(r['task_metric'] for r in t) / len(t))
                out[domain] = {'examples': len(b), 'base_task_metric': bm, 'trained_task_metric': tm, 'delta': tm - bm}
        if out:
            bm, tm = (sum(v['base_task_metric'] for v in out.values()) / len(out), sum(v['trained_task_metric'] for v in out.values()) / len(out))
            out['macro'] = {'examples': sum(v['examples'] for v in out.values()), 'base_task_metric': bm, 'trained_task_metric': tm, 'delta': tm - bm}
        return out

    def paired_result(self, base_rows: list[dict], trained_rows: list[dict], cfg: dict) -> dict:
        """In-loop base (step-0 eval) vs trained (last eval): honest_power per domain + macro, dropped ids, masked rates (see results.py)."""
        from .results import paired_result
        return paired_result(self, base_rows, trained_rows, cfg)

    def failfast_defaults(self, cfg: dict) -> dict:
        """budget.failfast rules mimo opts into: per-GPU memory (GB) at step 1, masked-group fraction per domain, step time vs the plan estimate."""
        return {'mem_gb': 76, 'domain_masked': 0.15, 'step_slow_factor': 1.5, 'min_steps': 1}

    def failure_mode(self, row: dict) -> str:
        if row.get('masked'):
            return str(row.get('mask_reason') or 'masked')
        if row.get('finish_reason') not in (None, 'stop', 'complete'):
            return 'truncated'
        return str(row.get('stop') if row.get('stop') not in (None, 'complete') else 'wrong')

    def train_diagnostics(self, completions, rewards, stop_reasons, group_size, env_metrics=None):
        valid = [(m or {}).get('task_metric') for m in (env_metrics or []) if (m or {}).get('task_metric') is not None]
        return {'train/task_metric': sum(valid) / len(valid)} if valid else {}

    def env_metrics_from(self, generator_output: dict, phase: str) -> list | None:
        from .generator import summaries_for
        ids = generator_output.get('trajectory_ids')
        if not ids:
            return None
        rows = [metrics.env_metrics_row(s) if s is not None else None for s in summaries_for(list(ids), phase)]
        return rows if any(rows) else None

    def extra_values(self, index: int, env_metrics: list | None) -> tuple:
        return metrics.extra_values(index, env_metrics)

    def critic_modes(self, cfg: dict) -> tuple[str, ...] | None:
        selected = data.selected_domains(mcfg(cfg))
        return BASE_CRITIC_MODES[:-1] + tuple(x for d in selected for x in CRITIC_MODES[d]) + BASE_CRITIC_MODES[-1:]

    def display_prompt(self, prompt: Any) -> str:
        return str(prompt)


def m_raw(cfg: dict) -> dict:
    return (cfg.get('environment') or {}).get('mimo') or {}


_DATASET_CLASS: list = []


def make_dataset_class() -> type:
    """HarborTaskDataset whose items carry the domain (env_extras.data_source = domain, so SkyRL's per-source eval metrics and eval
    dumps are per domain), the task dir and the repeat index of duplicated held-out tasks."""
    if _DATASET_CLASS:
        return _DATASET_CLASS[0]
    coding_harbor.config_class()
    from examples.train_integrations.harbor.dataset import HarborTaskDataset

    class MimoTaskDataset(HarborTaskDataset):
        def __init__(self, data_files):
            super().__init__(data_files)
            seen: dict[str, int] = {}
            self.reps = []
            for path in self.task_paths:
                self.reps.append(seen.get(str(path), 0))
                seen[str(path)] = self.reps[-1] + 1
            self.domains = [data.task_domain(str(p)) for p in self.task_paths]

        def _item(self, index: int) -> dict:
            path = str(self.task_paths[index])
            return {'prompt': path, 'env_class': None, 'uid': str(index),
                    'env_extras': {'data_source': self.domains[index], 'domain': self.domains[index], 'task_dir': path, 'task': Path(path).name,
                                   'rep': self.reps[index]}}

        def __getitem__(self, index: int) -> dict:
            if index >= len(self.task_paths):
                raise IndexError(f'Index {index} out of range for dataset of size {len(self.task_paths)}')
            return self._item(index)

        def __iter__(self):
            return (self._item(i) for i in range(len(self.task_paths)))

    _DATASET_CLASS.append(MimoTaskDataset)
    return MimoTaskDataset
