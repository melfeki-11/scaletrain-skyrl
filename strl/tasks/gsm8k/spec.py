"""GSM8K environment spec (the GPU-verified environment)."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from ..base import EnvironmentSpec
from . import answers, data as gsm_data
from .rewards import FUNCTIONS


def _ground_truth(extras: dict) -> str | None:
    spec = extras.get('reward_spec') or {}
    info = extras.get('extra_info') or {}
    return spec.get('ground_truth') or info.get('ground_truth')


class Gsm8kSpec(EnvironmentSpec):
    name = 'gsm8k'
    generator_kind = 'gym_single'
    verified_env_class = 'gsm8k'
    metric = 'accuracy'
    invalid_metric = 'train/invalid_answer_rate'
    needs_modal = False
    default_reward = [{'fn': 'strict_answer', 'weight': 1.0}]
    sample_id_format = 'gsm8k-test-%05d'
    eval_module = 'strl.tasks.gsm8k.evaluate'
    failure_classes = ('no_final_answer_line', 'wrong_answer', 'truncated')
    expected_base_reward = 0.51

    # ------------------------------------------------------------------ planning
    def validate_experiment(self, cfg: dict) -> list[str]:
        problems = []
        dataset = cfg['environment'].get('dataset') or {}
        for key in ('id', 'config', 'revision', 'validation_size', 'split_seed', 'prompt_suffix'):
            if key not in dataset:
                problems.append(f'environment.dataset.{key} is required for gsm8k')
        if 'test_size' not in cfg['environment']:
            problems.append('environment.test_size is required for gsm8k')
        if dataset.get('validation_size') and cfg['evaluation']['eval_batch_size'] != dataset['validation_size']:
            problems.append('evaluation.eval_batch_size must equal environment.dataset.validation_size (one validation pass per eval)')
        if dataset.get('validation_size') and cfg['evaluation']['fixed_qualitative_samples'] > dataset['validation_size']:
            problems.append('evaluation.fixed_qualitative_samples cannot exceed the validation size')
        return problems

    def reward_functions(self):
        return dict(FUNCTIONS)

    # ------------------------------------------------------------------ data
    def prepare_data(self, cfg: dict, **kwargs: Any) -> dict:
        return gsm_data.prepare_dataset(cfg, **kwargs)

    def assert_manifest(self, cfg: dict) -> dict:
        return gsm_data.assert_manifest(cfg)

    def data_files(self, cfg: dict, stage: str):
        root = Path(cfg['run']['data_dir'])
        return [str(root / 'train.parquet')], [str(root / ('canary.parquet' if stage == 'canary' else 'validation.parquet'))]

    def validation_size(self, cfg: dict) -> int:
        return int(cfg['environment']['dataset']['validation_size'])

    # ------------------------------------------------------------------ scoring
    def parse(self, text: str):
        return answers.parse_final_answer(text)

    def score(self, text: str, truth: Any, finish_reason: str | None, **extra: Any) -> bool:
        return bool(answers.score_answer(text, truth, finish_reason))

    def score_result(self, action: str, extras: dict, cfg: dict) -> dict:
        truth = _ground_truth(extras)
        parsed = answers.parse_final_answer(action)
        expected = answers.normalize_number(str(truth)) if truth is not None else None
        correct = parsed is not None and expected is not None and parsed == expected
        failure = 'ok' if correct else ('no_final_answer_line' if parsed is None else 'wrong_answer')
        return {'completion': action, 'parsed': parsed, 'correct': correct, 'failure_class': failure}

    def failure_mode(self, row: dict) -> str:
        if row.get('finish_reason') != 'stop':
            return 'truncated'
        if row.get('parsed_answer') is None:
            return 'no_final_answer_line'
        return 'wrong_answer'

    # ------------------------------------------------------------------ SkyRL wiring
    def skyrl_overrides(self, cfg: dict, stage: str) -> dict[str, Any]:
        generic = self.uses_generic_reward(cfg)
        return {'environment.env_class': self.generic_env_class if generic else self.verified_env_class}

    def register(self, cfg: dict) -> None:
        if self.uses_generic_reward(cfg):
            from ...envs import register_generic_env
            register_generic_env()

    def patch_runtime(self, skyrl_root: str, cfg: dict) -> None:
        from .reward_patch import patch_reward_verifier
        patch_reward_verifier(skyrl_root)

    def generator_contract(self, cfg: dict) -> dict:
        """Exercise the REAL reward path(s) on edge cases: terminal EOS, trailing text, and (generic path) the component sum."""
        receipt = {}
        cases = [('work\n#### 38<|im_end|>', 1.0), ('#### 38<|im_end|> extra', 0.0), ('#### 39', 0.0), ('no answer', 0.0)]
        if not self.uses_generic_reward(cfg):
            reward_utils = importlib.import_module('skyrl_gym.envs.gsm8k.utils')
            importlib.reload(reward_utils)
            from skyrl_gym.envs.gsm8k.env import GSM8kEnv
            env = GSM8kEnv(extras={'reward_spec': {'ground_truth': '38'}})
            for text, expected in cases:
                if env.step(text)['reward'] != expected:
                    raise RuntimeError(f'Actual GSM8K reward contract failed on {text!r}')
            receipt['actual_reward_passed'] = True
        else:
            from ...rewards import Reward, RewardContext
            reward = Reward(self, cfg['reward']['components'])
            for text, expected_correct in cases:
                result = self.score_result(text, {'reward_spec': {'ground_truth': '38'}}, cfg)
                total, parts = reward(result, RewardContext(ground_truth='38'))
                if result['correct'] != bool(expected_correct):
                    raise RuntimeError(f'Generic GSM8K scorer disagrees on {text!r}')
                receipt.setdefault('generic_reward_samples', []).append({'text': text, 'reward': total, 'components': parts})
            receipt['generic_reward_passed'] = True
        return receipt

    # ------------------------------------------------------------------ monitoring
    def train_diagnostics(self, completions, rewards, stop_reasons, group_size, env_metrics=None):
        return {self.invalid_metric: (sum(answers.parse_final_answer(c) is None for c in completions) / len(completions)
                                      if completions else 0.0)}

    def eval_record(self, item: dict, trainer: Any, step: int) -> dict:
        extras = item.get('env_extras') or {}
        info = extras.get('extra_info') or {}
        completion = item.get('output_response', '')
        tokenizer = trainer.tokenizer
        completion = tokenizer.decode(tokenizer.encode(completion, add_special_tokens=False), skip_special_tokens=True)
        reward = item.get('score', 0.0)
        reward = float(sum(reward)) if isinstance(reward, list) else float(reward)
        strl = extras.get('strl_result')
        parsed = answers.parse_final_answer(completion)
        return {
            'step': step, 'sample_id': info.get('sample_id'), 'prompt': item.get('input_prompt'), 'completion': completion,
            'ground_truth': _ground_truth(extras), 'parsed_answer': parsed,
            # task metric: strict correctness. Verified path: reward > 0 (unchanged). Generic path: the scorer's verdict,
            # so a custom reward design never changes what "correct" means.
            'correct': bool(strl['correct']) if strl else bool(reward > 0.0),
            'reward': reward, 'token_count': len(tokenizer.encode(completion, add_special_tokens=False)),
            'finish_reason': item.get('stop_reason'),
        }

    def display_prompt(self, prompt: Any) -> str:
        return str(prompt)
