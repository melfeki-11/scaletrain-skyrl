"""Countdown environment spec: single-turn arithmetic with a verifiable reward (generic ``strl_env`` path, no external services)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..base import EnvironmentSpec
from . import data as cd_data, scoring
from .rewards import FUNCTIONS


def _truth(extras: dict) -> Any:
    spec = extras.get('reward_spec') or {}
    info = extras.get('extra_info') or {}
    return spec.get('ground_truth') or info.get('ground_truth')


class CountdownSpec(EnvironmentSpec):
    name = 'countdown'
    generator_kind = 'gym_single'
    verified_env_class = None                 # no SkyRL built-in: always the generic reward path (strl_env)
    metric = 'accuracy'
    invalid_metric = 'train/invalid_answer_rate'
    needs_modal = False
    uses_coding_executor = False
    default_reward = [{'fn': 'countdown_correct', 'weight': 1.0}]
    eval_module = 'strl.tasks.countdown.evaluate'
    extra_columns = ('failure_class',)
    failure_classes = ('no_answer', 'invalid_expression', 'illegal_numbers', 'wrong_value')
    expected_base_reward = None               # unknown until a canary measures the base pass rate

    # ------------------------------------------------------------------ planning
    def validate_experiment(self, cfg: dict) -> list[str]:
        problems = []
        raw = (cfg['environment'].get('countdown') or {})
        unknown = set(raw) - set(cd_data.DEFAULTS)
        if unknown:
            problems.append(f'unknown key(s) environment.countdown.{sorted(unknown)} (allowed: {sorted(cd_data.DEFAULTS)})')
        s = cd_data.settings(cfg)
        if s['prompt'] is not None:   # the prompt is a design axis (a non-reasoning model told to "think briefly" answers in ~15 tokens)
            missing = [k for k in ('{numbers}', '{target}', '<answer>') if k not in str(s['prompt'])]
            if missing:
                problems.append(f'environment.countdown.prompt must contain {missing} (the scorer reads the LAST <answer></answer>)')
        if not 2 <= s['min_numbers'] <= s['max_numbers']:
            problems.append('environment.countdown needs 2 <= min_numbers <= max_numbers')
        for split in cd_data.SPLITS:
            if int(s[f'{split}_size']) < 1:
                problems.append(f'environment.countdown.{split}_size must be >= 1')
        if cfg['evaluation']['eval_batch_size'] != int(s['validation_size']):
            problems.append('evaluation.eval_batch_size must equal environment.countdown.validation_size (one validation pass per eval)')
        if cfg['evaluation']['fixed_qualitative_samples'] > int(s['validation_size']):
            problems.append('evaluation.fixed_qualitative_samples cannot exceed environment.countdown.validation_size')
        if int(s['train_size']) < cfg['algorithm']['prompts_per_step']:
            problems.append('environment.countdown.train_size must be >= algorithm.prompts_per_step')
        return problems

    def reward_functions(self):
        return dict(FUNCTIONS)

    # ------------------------------------------------------------------ data
    def prepare_data(self, cfg: dict, **kwargs: Any) -> dict:
        return cd_data.prepare(cfg, **kwargs)

    def assert_manifest(self, cfg: dict) -> dict:
        return cd_data.assert_manifest(cfg)

    def data_files(self, cfg: dict, stage: str):
        root = Path(cfg['run']['data_dir'])
        return [str(root / 'train.parquet')], [str(root / ('canary.parquet' if stage == 'canary' else 'validation.parquet'))]

    def validation_size(self, cfg: dict) -> int:
        return int(cd_data.settings(cfg)['validation_size'])

    def expected_rows(self, cfg: dict, split: str) -> int | None:
        s = cd_data.settings(cfg)
        if split == 'validation' and os.environ.get('STRL_STAGE') == 'canary':
            return self.canary_size(cfg)
        return int(s['train_size'] if split == 'train' else s['validation_size'])

    def final_eval_ids(self, cfg: dict) -> list[str]:
        return [f'countdown-test-{i:05d}' for i in range(int(cd_data.settings(cfg)['test_size']))]

    # ------------------------------------------------------------------ scoring
    def parse(self, text: str):
        return scoring.parse_answer(text)

    def score_result(self, action: str, extras: dict, cfg: dict) -> dict:
        numbers, target = scoring.load_truth(_truth(extras))
        return {'completion': action, 'target': target, **scoring.score_expression(action, numbers, target)}

    def score(self, text: str, truth: Any, finish_reason: str | None, **extra: Any) -> bool:
        if finish_reason not in (None, 'stop', 'eos_token'):
            return False
        numbers, target = scoring.load_truth(truth)
        return bool(scoring.score_expression(text, numbers, target)['correct'])

    def failure_mode(self, row: dict) -> str:
        if row.get('finish_reason') not in (None, 'stop'):
            return 'truncated'
        return row.get('failure_class') or 'wrong_value'

    def result_metrics(self, result: dict) -> dict[str, float]:
        return {'valid_expression': 1.0 if result.get('valid') else 0.0, 'legal_numbers': 1.0 if result.get('legal_numbers') else 0.0}

    # ------------------------------------------------------------------ SkyRL wiring
    def skyrl_overrides(self, cfg: dict, stage: str) -> dict[str, Any]:
        return {'environment.env_class': self.generic_env_class}

    def register(self, cfg: dict) -> None:
        from ...envs import register_generic_env
        register_generic_env()

    def generator_contract(self, cfg: dict) -> dict:
        """The REAL generic reward path (scorer + the experiment's reward components) on known-good and known-bad completions."""
        from ...rewards import Reward, RewardContext
        truth = '{"numbers": [3, 7, 25, 2], "target": 181}'
        extras = {'reward_spec': {'ground_truth': truth}}
        cases = {'correct': ('7 * 25 = 175, + 3 + 2 = 180... use 7*25+3*2.\n<answer>7 * 25 + 3 * 2</answer><|im_end|>', True),
                 'reuse': ('<answer>25 * 7 + 3 + 3</answer>', False), 'foreign_number': ('<answer>180 + 1</answer>', False),
                 'wrong_value': ('<answer>25 * 7 + 3</answer>', False), 'no_answer': ('The answer is 7*25+3*2.', False),
                 'injection': ('<answer>__import__("os").system("true")</answer>', False), 'div_zero': ('<answer>7 / (3 - 3)</answer>', False)}
        reward = Reward(self, cfg['reward']['components'])
        receipt: dict[str, Any] = {'cases': {}}
        for name, (text, expected) in cases.items():
            result = self.score_result(text, extras, cfg)
            total, parts = reward(result, RewardContext(ground_truth=truth, extras=extras,
                                                        max_generate_length=int(cfg['algorithm']['max_generate_length'])))
            receipt['cases'][name] = {'failure_class': result['failure_class'], 'reward': total, 'components': parts}
            if bool(result['correct']) != expected:
                raise RuntimeError(f'countdown reward contract failed on {name}: {result["failure_class"]}')
        best_wrong = max(c['reward'] for n, c in receipt['cases'].items() if n != 'correct')
        if not receipt['cases']['correct']['reward'] > best_wrong:
            raise RuntimeError(f'countdown reward does not rank the correct answer above every wrong one: {receipt}')
        receipt['generic_reward_passed'] = True
        return receipt

    # ------------------------------------------------------------------ monitoring
    def train_diagnostics(self, completions, rewards, stop_reasons, group_size, env_metrics=None):
        out = {self.invalid_metric: (sum(scoring.parse_answer(c) is None for c in completions) / len(completions) if completions else 0.0)}
        if env_metrics:
            out['train/valid_expression_rate'] = sum((m or {}).get('valid_expression', 0.0) for m in env_metrics) / len(env_metrics)
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
            numbers, target = scoring.load_truth(_truth(extras))
            strl = scoring.score_expression(completion, numbers, target)
        if item.get('stop_reason') not in (None, 'stop', 'eos_token'):   # truncated: wrong, exactly as score() and the offline eval treat it
            strl = {**strl, 'correct': False, 'failure_class': strl.get('failure_class') if not strl.get('correct') else 'truncated'}
        return {
            'step': step, 'sample_id': info.get('sample_id'), 'prompt': item.get('input_prompt'), 'completion': completion,
            'ground_truth': _truth(extras), 'parsed_answer': strl.get('parsed'), 'failure_class': strl.get('failure_class'),
            'correct': bool(strl.get('correct')), 'reward': reward,
            'token_count': len(tokenizer.encode(completion, add_special_tokens=False)), 'finish_reason': item.get('stop_reason'),
        }
