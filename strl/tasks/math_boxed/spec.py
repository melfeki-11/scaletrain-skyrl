"""math_boxed environment spec: MATH problems, final answer in \\boxed{}, exact match after normalization (generic ``strl_env`` path)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..base import EnvironmentSpec
from . import data as mb_data, scoring
from .rewards import FUNCTIONS


def _truth(extras: dict) -> Any:
    spec = extras.get('reward_spec') or {}
    info = extras.get('extra_info') or {}
    return spec.get('ground_truth') or info.get('ground_truth')


class MathBoxedSpec(EnvironmentSpec):
    name = 'math_boxed'
    generator_kind = 'gym_single'
    verified_env_class = None                 # no SkyRL built-in: always the generic reward path (strl_env)
    metric = 'accuracy'
    invalid_metric = 'train/invalid_answer_rate'
    needs_modal = False
    uses_coding_executor = False
    default_reward = [{'fn': 'math_correct', 'weight': 1.0}, {'fn': 'math_format', 'weight': 0.1}]
    eval_module = 'strl.tasks.math_boxed.evaluate'
    extra_columns = ('failure_class',)
    failure_classes = ('no_answer', 'wrong_answer')
    expected_base_reward = None               # unknown until a canary measures the base pass rate

    # ------------------------------------------------------------------ planning
    def validate_experiment(self, cfg: dict) -> list[str]:
        problems = []
        raw = (cfg['environment'].get(self.name) or {})
        unknown = set(raw) - set(mb_data.DEFAULTS)
        if unknown:
            problems.append(f'unknown key(s) environment.math_boxed.{sorted(unknown)} (allowed: {sorted(mb_data.DEFAULTS)})')
        s = mb_data.settings(cfg)
        if not s['levels'] or not set(int(x) for x in s['levels']) <= {1, 2, 3, 4, 5}:
            problems.append('environment.math_boxed.levels must be a non-empty subset of [1, 2, 3, 4, 5]')
        if len(str(s['revision'])) != 40:
            problems.append('environment.math_boxed.revision must be a 40-character dataset commit sha (never a branch)')
        for split in ('train', 'validation', 'test'):
            if int(s[f'{split}_size']) < 1:
                problems.append(f'environment.math_boxed.{split}_size must be >= 1')
        if cfg['evaluation']['eval_batch_size'] != int(s['validation_size']):
            problems.append('evaluation.eval_batch_size must equal environment.math_boxed.validation_size (one validation pass per eval)')
        if cfg['evaluation']['fixed_qualitative_samples'] > int(s['validation_size']):
            problems.append('evaluation.fixed_qualitative_samples cannot exceed environment.math_boxed.validation_size')
        if int(s['train_size']) < cfg['algorithm']['prompts_per_step']:
            problems.append('environment.math_boxed.train_size must be >= algorithm.prompts_per_step')
        if '\\boxed' not in s['prompt_suffix']:
            problems.append('environment.math_boxed.prompt_suffix must ask for the answer in \\boxed{} (the scorer reads only boxed answers)')
        return problems

    def reward_functions(self):
        return dict(FUNCTIONS)

    # ------------------------------------------------------------------ data
    def prepare_data(self, cfg: dict, **kwargs: Any) -> dict:
        return mb_data.prepare(cfg, **kwargs)

    def assert_manifest(self, cfg: dict) -> dict:
        return mb_data.assert_manifest(cfg)

    def data_files(self, cfg: dict, stage: str):
        root = Path(cfg['run']['data_dir'])
        return [str(root / 'train.parquet')], [str(root / ('canary.parquet' if stage == 'canary' else 'validation.parquet'))]

    def validation_size(self, cfg: dict) -> int:
        return int(mb_data.settings(cfg)['validation_size'])

    def expected_rows(self, cfg: dict, split: str) -> int | None:
        s = mb_data.settings(cfg)
        if split == 'validation' and os.environ.get('STRL_STAGE') == 'canary':
            return self.canary_size(cfg)
        return int(s['train_size'] if split == 'train' else s['validation_size'])

    def final_eval_ids(self, cfg: dict) -> list[str]:
        return [f'{self.name}-test-{i:05d}' for i in range(int(mb_data.settings(cfg)['test_size']))]

    # ------------------------------------------------------------------ scoring
    def parse(self, text: str):
        return scoring.last_boxed(text)

    def score_result(self, action: str, extras: dict, cfg: dict) -> dict:
        return {'completion': action, **scoring.score_completion(action, _truth(extras))}

    def score(self, text: str, truth: Any, finish_reason: str | None, **extra: Any) -> bool:
        if finish_reason not in (None, 'stop', 'eos_token'):
            return False
        return bool(scoring.score_completion(text, truth)['correct'])

    def failure_mode(self, row: dict) -> str:
        if row.get('finish_reason') not in (None, 'stop'):
            return 'truncated'
        return row.get('failure_class') or 'wrong_answer'

    def result_metrics(self, result: dict) -> dict[str, float]:
        return {'boxed': 1.0 if result.get('parsed') is not None else 0.0, 'single_answer': 1.0 if result.get('single_answer') else 0.0}

    # ------------------------------------------------------------------ SkyRL wiring
    def skyrl_overrides(self, cfg: dict, stage: str) -> dict[str, Any]:
        return {'environment.env_class': self.generic_env_class}

    def register(self, cfg: dict) -> None:
        from ...envs import register_generic_env
        register_generic_env()

    def generator_contract(self, cfg: dict) -> dict:
        """The REAL generic reward path (scorer + the experiment's reward components) on known-good and known-bad completions."""
        from ...rewards import Reward, RewardContext
        truth = '\\frac{3}{4}'
        extras = {'reward_spec': {'ground_truth': truth}}
        cases = {'correct': ('Half of 3/2 is 3/4, so the answer is $\\boxed{\\frac{3}{4}}$.<|im_end|>', True),
                 'correct_dfrac_spaces': ('So $x = \\boxed{ \\dfrac34 }$.', True),
                 'correct_after_hedge': ('Maybe \\boxed{1}. No: \\boxed{\\frac{3}{4}}', True),
                 'wrong': ('The answer is \\boxed{\\frac{4}{3}}.', False), 'unreduced': ('\\boxed{\\frac{6}{8}}', False),
                 'no_box': ('The answer is 3/4.', False), 'empty_box': ('\\boxed{}', False), 'unbalanced': ('\\boxed{\\frac{3}{4}', False),
                 'nested_wrong': ('\\boxed{\\boxed{1}}', False)}
        reward = Reward(self, cfg['reward']['components'])
        receipt: dict[str, Any] = {'cases': {}}
        for name, (text, expected) in cases.items():
            result = self.score_result(text, extras, cfg)
            total, parts = reward(result, RewardContext(ground_truth=truth, extras=extras,
                                                        max_generate_length=int(cfg['algorithm']['max_generate_length'])))
            receipt['cases'][name] = {'failure_class': result['failure_class'], 'reward': total, 'components': parts}
            if bool(result['correct']) != expected:
                raise RuntimeError(f'math_boxed reward contract failed on {name}: {result["failure_class"]} parsed={result["parsed"]!r}')
        best_wrong = max(c['reward'] for n, c in receipt['cases'].items() if not cases[n][1])
        worst_right = min(c['reward'] for n, c in receipt['cases'].items() if cases[n][1])
        if not worst_right > best_wrong:
            raise RuntimeError(f'math_boxed reward does not rank every correct answer above every wrong one: {receipt}')
        receipt['generic_reward_passed'] = True
        return receipt

    # ------------------------------------------------------------------ monitoring
    def train_diagnostics(self, completions, rewards, stop_reasons, group_size, env_metrics=None):
        out = {self.invalid_metric: (sum(scoring.last_boxed(c) is None for c in completions) / len(completions) if completions else 0.0)}
        if env_metrics:
            out['train/single_answer_rate'] = sum((m or {}).get('single_answer', 0.0) for m in env_metrics) / len(env_metrics)
        return out

    def extra_values(self, index: int, env_metrics: list | None) -> tuple:
        if not env_metrics or index >= len(env_metrics):
            return ('',)
        m = env_metrics[index] or {}
        return (next((c for c in self.failure_classes if m.get(f'failure/{c}')), 'ok' if m.get('task_metric') else ''),)

    def eval_record(self, item: dict, trainer: Any, step: int) -> dict:
        extras = item.get('env_extras') or {}
        info = extras.get('extra_info') or {}
        tokenizer = trainer.tokenizer
        completion = item.get('output_response', '')
        completion = tokenizer.decode(tokenizer.encode(completion, add_special_tokens=False), skip_special_tokens=True)
        reward = item.get('score', 0.0)
        reward = float(sum(reward)) if isinstance(reward, list) else float(reward)
        strl = extras.get('strl_result')
        if strl is None:   # rescore from the decoded text (never trust the training reward as the task metric)
            strl = scoring.score_completion(completion, _truth(extras))
        if item.get('stop_reason') not in (None, 'stop', 'eos_token'):   # truncated: wrong, exactly as score() and the offline eval treat it
            strl = {**strl, 'correct': False, 'failure_class': strl.get('failure_class') if not strl.get('correct') else 'truncated'}
        return {
            'step': step, 'sample_id': info.get('sample_id'), 'prompt': item.get('input_prompt'), 'completion': completion,
            'ground_truth': _truth(extras), 'parsed_answer': strl.get('parsed'), 'failure_class': strl.get('failure_class'),
            'correct': bool(strl.get('correct')), 'reward': reward,
            'token_count': len(tokenizer.encode(completion, add_special_tokens=False)), 'finish_reason': item.get('stop_reason'),
        }
