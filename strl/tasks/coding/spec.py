"""Coding environments. ``coding_exec``: single-turn, the model writes a solution, tests run in the Modal executor (Tier A, fast,
token-in/token-out). ``coding_harbor``: the same tasks as Harbor task dirs, solved by the Terminus-2 agent in Modal sandboxes
(Tier B, agentic, multi-turn). Same data, rewards, task metric and qualitative tables."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..base import EnvironmentSpec
from . import data as coding_data
from . import harness
from .extract import extract_code
from .rewards import FUNCTIONS


class CodingExecSpec(EnvironmentSpec):
    name = 'coding_exec'
    generator_kind = 'gym_single'
    verified_env_class = None              # always the generic reward path (strl_env)
    metric = 'pass_at_1'
    invalid_metric = 'train/no_code_block_rate'
    needs_modal = True
    default_reward = [{'fn': 'pass_all_tests', 'weight': 1.0}]
    eval_module = 'strl.tasks.coding.evaluate'
    failure_classes = harness.FAILURE_CLASSES
    extra_columns = ('failure_class', 'tests_passed', 'tests_total')
    preflight_imports = ('modal', 'strl.tasks.coding.generator')
    dry_run_seconds = 1200

    # ------------------------------------------------------------------ planning
    def validate_experiment(self, cfg: dict) -> list[str]:
        problems = []
        sources = cfg['environment'].get('sources') or {}
        for key in ('train', 'validation', 'test'):
            if key not in sources:
                problems.append(f'environment.sources.{key} is required for {self.name}')
        for group in ('train', 'test'):
            for entry in sources.get(group, []):
                if entry.get('dataset') not in coding_data.DATASETS:
                    problems.append(f'environment.sources.{group}: unknown dataset {entry.get("dataset")!r}; known {sorted(coding_data.DATASETS)}')
        size = (sources.get('validation') or {}).get('size')
        if size and cfg['evaluation']['eval_batch_size'] != size:
            problems.append('evaluation.eval_batch_size must equal environment.sources.validation.size (one validation pass per eval)')
        if size and cfg['evaluation']['fixed_qualitative_samples'] > size:
            problems.append('evaluation.fixed_qualitative_samples cannot exceed the validation size')
        modal = (cfg.get('services') or {}).get('modal') or {}
        if not modal.get('credentials_uri'):
            problems.append('services.modal.credentials_uri (encrypted S3 bundle or Secrets Manager id) is required for Modal execution')
        if cfg['inference'].get('max_env_workers', 64) < 16:
            problems.append('inference.max_env_workers < 16 serialises Modal calls; use >= 32')
        return problems

    def reward_functions(self):
        return dict(FUNCTIONS)

    def tags(self, cfg: dict) -> list[str]:
        return ['coding', self.name]

    # ------------------------------------------------------------------ data
    def prepare_data(self, cfg: dict, **kwargs: Any) -> dict:
        return coding_data.prepare(cfg, **kwargs)

    def assert_manifest(self, cfg: dict) -> dict:
        return coding_data.assert_manifest(cfg)

    def _manifest(self, cfg: dict) -> dict:
        return json.loads((Path(cfg['run']['data_dir']) / 'manifest.json').read_text())

    def data_files(self, cfg: dict, stage: str):
        root = Path(cfg['run']['data_dir'])
        return [str(root / 'train.parquet')], [str(root / ('canary.parquet' if stage == 'canary' else 'validation.parquet'))]

    def validation_size(self, cfg: dict) -> int:
        return int(cfg['environment']['sources']['validation']['size'])

    def expected_rows(self, cfg: dict, split: str) -> int | None:
        counts = self._manifest(cfg)['counts']
        if split == 'validation' and os.environ.get('STRL_STAGE') == 'canary':
            return counts['canary']
        return counts[split]

    def final_eval_ids(self, cfg: dict) -> list[str]:
        return list(self._manifest(cfg)['test_ids'])

    # ------------------------------------------------------------------ scoring
    def parse(self, text: str):
        return extract_code(text)

    def score_result(self, action: str, extras: dict, cfg: dict) -> dict:
        from . import executor
        task = coding_data.task_from_extras(extras)
        code = extract_code(action)
        if code is None:
            return {'completion': action, 'parsed': None, 'has_code': False, 'compiled': False, 'passed': 0, 'total': len(task['tests']),
                    'failure_class': 'no_code_block', 'correct': False, 'suspicious': False, 'latency_s': 0.0}
        verdict = executor.score({'task_id': task['task_id'], 'code': code, 'tests': task['tests'], 'setup': task['setup']})
        verdict.update(completion=action, parsed=code[:2000], has_code=True, suspicious=harness.suspicious(code),
                       correct=verdict['failure_class'] == 'ok')
        verdict.setdefault('total', len(task['tests']))
        return verdict

    def score(self, text: str, truth: Any, finish_reason: str | None, **extra: Any) -> bool:
        if finish_reason not in (None, 'stop', 'eos_token'):
            return False
        return bool(self.score_result(text, {'reward_spec': {'ground_truth': truth}, 'extra_info': {}}, {})['correct'])

    def failure_mode(self, row: dict) -> str:
        if row.get('finish_reason') not in (None, 'stop'):
            return 'truncated'
        return row.get('failure_class') or 'wrong_answer'

    def result_metrics(self, result: dict) -> dict[str, float]:
        total = result.get('total') or 0
        return {'suspicious_code': 1.0 if result.get('suspicious') else 0.0, 'partial_pass': (result.get('passed', 0) / total) if total else 0.0,
                'tests_passed': float(result.get('passed', 0)), 'tests_total': float(total), 'memo_hit': 1.0 if result.get('memo_hit') else 0.0}

    def metric_by_source(self, base: list[dict], trained: list[dict]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for label, rows in (('base', base), ('trained', trained)):
            for row in rows:
                source = row['sample_id'].split('-')[0]
                bucket = out.setdefault(source, {'examples': 0, 'base_pass_at_1': 0.0, 'trained_pass_at_1': 0.0})
                if label == 'base':
                    bucket['examples'] += 1
                bucket[f'{label}_pass_at_1'] += float(row['correct'])
        for bucket in out.values():
            n = bucket['examples'] or 1
            bucket['base_pass_at_1'] /= n
            bucket['trained_pass_at_1'] /= n
            bucket['delta'] = bucket['trained_pass_at_1'] - bucket['base_pass_at_1']
        return out

    # ------------------------------------------------------------------ SkyRL wiring
    def skyrl_overrides(self, cfg: dict, stage: str) -> dict[str, Any]:
        return {'environment.env_class': self.generic_env_class,
                'environment.skyrl_gym.max_env_workers': int(cfg['inference'].get('max_env_workers', 64))}

    def register(self, cfg: dict) -> None:
        from ...envs import register_generic_env
        register_generic_env()

    def build_generator(self, experiment: dict, cfg: Any, tokenizer: Any, inference_engine_client: Any) -> Any:
        from skyrl.train.entrypoints.main_base import resolve_policy_model_name
        from .generator import make_generator_class
        return make_generator_class()(generator_cfg=cfg.generator, skyrl_gym_cfg=cfg.environment.skyrl_gym,
                                      inference_engine_client=inference_engine_client, tokenizer=tokenizer,
                                      policy_model_name=resolve_policy_model_name(cfg))

    def generator_contract(self, cfg: dict) -> dict:
        """The REAL reward path from the pod: reference passes, wrong/empty/exit-hack programs fail, components sum correctly."""
        from ...rewards import Reward, RewardContext
        from . import executor
        executor.ensure_deployed()
        task = coding_data.load_tasks(cfg, 'canary')[0]
        extras = coding_data.to_row(task)
        extras = {'reward_spec': extras['reward_spec'], 'extra_info': extras['extra_info']}
        reward = Reward(self, cfg['reward']['components'])
        cases = {'reference': (f'```python\n{task.reference}\n```', True), 'no_code': ('I think the answer is 42.', False),
                 'exit_hack': ('```python\nimport sys\nsys.exit(0)\n```', False), 'stub': (f'```python\n{coding_data.stub(task.reference)}\n```', False)}
        receipt = {'task': task.sample_id, 'cases': {}}
        for name, (text, expected) in cases.items():
            result = self.score_result(text, extras, cfg)
            total, parts = reward(result, RewardContext(ground_truth=None, extras=extras))
            receipt['cases'][name] = {'failure_class': result['failure_class'], 'reward': total, 'components': parts}
            if result['failure_class'] == 'infra_error' or bool(result['correct']) != expected:
                raise RuntimeError(f'coding reward contract failed on {name}: {result["failure_class"]}')
        receipt['actual_reward_passed'] = True
        return receipt

    def dry_run_checks(self, cfg: dict) -> dict:
        """Executor deployed at the right version, then 512 concurrent per-rollout calls exactly like a training step."""
        from . import executor
        receipt = {'deploy': executor.ensure_deployed(), 'prewarm_s': round(executor.prewarm(64), 2)}
        task = coding_data.load_tasks(cfg, 'canary')[0]
        jobs = [task.job(task.reference if i % 2 == 0 else 'def broken(:') for i in range(512)]
        for i, job in enumerate(jobs):
            job['code'] += f'\n# variant {i}\n'  # defeat the memo so every call goes to Modal
        started = time.monotonic()
        with ThreadPoolExecutor(int(cfg['inference'].get('max_env_workers', 64))) as pool:
            results = list(pool.map(lambda j: executor.score(j, memo=False), jobs))
        elapsed = time.monotonic() - started
        trips = sorted(r.get('round_trip_s', 0.0) for r in results)
        receipt.update(calls=len(results), seconds=round(elapsed, 2), calls_per_second=round(len(results) / elapsed, 1),
                       p50_s=trips[len(trips) // 2], p95_s=trips[int(0.95 * (len(trips) - 1))], max_s=trips[-1],
                       infra_errors=sum(r['failure_class'] == 'infra_error' for r in results),
                       classes={c: sum(r['failure_class'] == c for r in results) for c in ('ok', 'syntax_error')})
        if receipt['infra_errors'] or receipt['classes'] != {'ok': 256, 'syntax_error': 256}:
            raise RuntimeError(f'executor load test failed: {receipt}')
        return receipt

    # ------------------------------------------------------------------ monitoring
    def train_diagnostics(self, completions, rewards, stop_reasons, group_size, env_metrics=None):
        out = {self.invalid_metric: sum(extract_code(c) is None for c in completions) / len(completions) if completions else 0.0}
        if env_metrics:
            out['train/pass_at_1'] = sum((m or {}).get('task_metric', 0.0) for m in env_metrics) / len(env_metrics)
            out['train/suspicious_code_rate'] = sum((m or {}).get('suspicious_code', 0.0) for m in env_metrics) / len(env_metrics)
        return out

    def extra_values(self, index: int, env_metrics: list | None) -> tuple:
        m = (env_metrics[index] if env_metrics and index < len(env_metrics) else None) or {}
        failure = next((k.split('/', 1)[1] for k, v in m.items() if k.startswith('failure/') and v), 'ok' if m.get('task_metric') else None)
        return failure, m.get('tests_passed'), m.get('tests_total')

    def eval_record(self, item: dict, trainer: Any, step: int) -> dict:
        extras = item.get('env_extras') or {}
        info = extras.get('extra_info') or {}
        result = extras.get('strl_result') or {}
        completion = item.get('output_response', '')
        tokenizer = trainer.tokenizer
        completion = tokenizer.decode(tokenizer.encode(completion, add_special_tokens=False), skip_special_tokens=True)
        reward = item.get('score', 0.0)
        reward = float(sum(reward)) if isinstance(reward, list) else float(reward)
        if result.get('failure_class') == 'infra_error':
            raise RuntimeError(f'executor infrastructure error during validation ({info.get("sample_id")}); refusing to report a biased metric')
        return {'step': step, 'sample_id': info.get('sample_id'), 'prompt': self.display_prompt(item.get('input_prompt')), 'completion': completion,
                'ground_truth': f"{info.get('source')}:{info.get('task_id')}", 'parsed_answer': result.get('parsed'),
                'correct': bool(result.get('correct')), 'reward': reward, 'token_count': len(tokenizer.encode(completion, add_special_tokens=False)),
                'finish_reason': item.get('stop_reason'), 'failure_class': result.get('failure_class'), 'tests_passed': result.get('passed'),
                'tests_total': result.get('total')}

    def display_prompt(self, prompt: Any) -> str:
        if isinstance(prompt, list) and prompt and isinstance(prompt[-1], dict):
            return str(prompt[-1].get('content', ''))
        return str(prompt)


class CodingHarborSpec(CodingExecSpec):
    """Same tasks, solved by Harbor's Terminus-2 agent in Modal sandboxes (multi-turn, step-wise GRPO)."""
    name = 'coding_harbor'
    generator_kind = 'harbor'
    preflight_imports = ('modal', 'harbor', 'examples.train_integrations.harbor.harbor_generator')
    dry_run_seconds = 1500

    def validate_experiment(self, cfg: dict) -> list[str]:
        problems = super().validate_experiment(cfg)
        modal = (cfg.get('services') or {}).get('modal') or {}
        if not modal.get('sandbox_secret'):
            problems.append('services.modal.sandbox_secret (e.g. modal-proxy-user) is required for Harbor Modal sandboxes')
        names = [c['fn'] for c in cfg['reward']['components'] if c.get('weight', 1.0)]
        if names != ['pass_all_tests']:
            problems.append('coding_harbor rewards come from the sandbox verifier (reward.txt); only [pass_all_tests] is supported here. '
                            'Explore reward designs with coding_exec, which applies arbitrary components.')
        return problems

    def skyrl_config_class(self):
        from .harbor import config_class
        return config_class()

    def data_files(self, cfg: dict, stage: str):
        root = Path(cfg['run']['data_dir']) / 'harbor'
        return [str(root / 'train')], [str(root / ('canary' if stage == 'canary' else 'validation'))]

    def expected_rows(self, cfg: dict, split: str) -> int | None:
        return None  # HarborTaskDataset reads directories; counts are checked by the harbor export manifest

    def canary_requires_gradient(self, cfg: dict) -> bool:
        return False  # a small model may solve no agentic task in 2 steps: all-equal rewards are legitimate here

    def skyrl_overrides(self, cfg: dict, stage: str) -> dict[str, Any]:
        from .harbor import skyrl_overrides
        return skyrl_overrides(cfg)

    def register(self, cfg: dict) -> None:
        from .harbor import install_compat
        install_compat()

    def build_generator(self, experiment: dict, cfg: Any, tokenizer: Any, inference_engine_client: Any) -> Any:
        from .harbor import build_generator
        return build_generator(experiment, cfg, tokenizer, inference_engine_client)

    def generator_contract(self, cfg: dict) -> dict:
        from .harbor import contract
        return contract(cfg)

    def dry_run_checks(self, cfg: dict) -> dict:
        from .harbor import dry_run_checks
        return dry_run_checks(cfg)

    def train_diagnostics(self, completions, rewards, stop_reasons, group_size, env_metrics=None):
        return {}

    def eval_record(self, item: dict, trainer: Any, step: int) -> dict:
        extras = item.get('env_extras') or {}
        reward = item.get('score', 0.0)
        reward = float(sum(reward)) if isinstance(reward, list) else float(reward)
        completion = item.get('output_response', '')
        task_dir = extras.get('data_source') or ''
        return {'step': step, 'sample_id': Path(task_dir).name, 'prompt': task_dir, 'completion': completion[-4000:], 'ground_truth': Path(task_dir).name,
                'parsed_answer': None, 'correct': reward >= 1.0, 'reward': reward, 'token_count': None, 'finish_reason': item.get('stop_reason')}
