"""Countdown environment (strl/tasks/countdown): scorer, rewards, deterministic data, manifest, spec wiring, experiment plan."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import copy
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from strl.experiment import load
from strl.rewards import Reward, RewardContext, validate_components
from strl.tasks import get_environment
from strl.tasks.countdown import data as cd_data, scoring
from strl.tasks.countdown.rewards import countdown_closeness, countdown_correct

EXP = 'experiments/countdown-qwen35.yaml'
NUMS, TARGET = [3, 7, 25, 2], 181


def score(text, numbers=NUMS, target=TARGET):
    return scoring.score_expression(text, numbers, target)


class Scorer(unittest.TestCase):
    def test_correct_with_terminal_token_and_reasoning(self):
        r = score('7*25 is 175, plus 3*2.\n<answer>7 * 25 + 3 * 2</answer><|im_end|>')
        self.assertTrue(r['correct']); self.assertEqual(r['failure_class'], 'ok'); self.assertEqual(r['numbers_used'], 4)

    def test_subset_of_numbers_is_allowed(self):
        self.assertTrue(score('<answer>(25 - 7) * 2</answer>', target=36)['correct'])

    def test_last_answer_block_wins(self):
        self.assertTrue(score('<answer>1+1</answer> no wait <answer>7*25+3*2</answer>')['correct'])
        self.assertFalse(score('<answer>7*25+3*2</answer> no wait <answer>7*25</answer>')['correct'])

    def test_number_reuse_and_foreign_numbers_are_illegal(self):
        self.assertEqual(score('<answer>25 * 7 + 3 + 3</answer>')['failure_class'], 'illegal_numbers')
        self.assertEqual(score('<answer>180 + 1</answer>')['failure_class'], 'illegal_numbers')
        self.assertTrue(score('<answer>3 + 3</answer>', numbers=[3, 3, 5], target=6)['correct'])   # duplicates given twice may be used twice

    def test_wrong_value(self):
        r = score('<answer>25 * 7 + 3</answer>')
        self.assertEqual(r['failure_class'], 'wrong_value'); self.assertEqual(r['distance'], 3.0); self.assertFalse(r['correct'])

    def test_exact_division(self):
        self.assertTrue(score('<answer>7 / 2 * 2</answer>', numbers=[7, 2, 2], target=7)['correct'])
        self.assertEqual(scoring.evaluate('7 / 2')[0], Fraction(7, 2))
        self.assertEqual(score('<answer>7 / (3 - 3)</answer>', numbers=[7, 3, 3], target=1)['failure_class'], 'invalid_expression')

    def test_hostile_or_malformed_expressions_are_invalid(self):
        for expr in ('__import__("os").system("true")', '2 ** 10', '7 // 2', '-3 + 7', '3 - -2', '2.5 * 2', '(3 + 7', '3 7',
                     '1' * 300, 'abs(3)', '[3][0]', '3 if 1 else 7', '３ + 7'):
            self.assertEqual(score(f'<answer>{expr}</answer>')['failure_class'], 'invalid_expression', expr)

    def test_no_answer(self):
        for text in ('The answer is 7*25+3*2', '<answer></answer>', '<answer>  </answer>', ''):
            self.assertEqual(score(text)['failure_class'], 'no_answer', text)

    def test_trailing_equals_tolerated_but_not_result(self):
        self.assertTrue(score('<answer>7*25+3*2 =</answer>')['correct'])
        self.assertEqual(score('<answer>7*25+3*2 = 181</answer>')['failure_class'], 'invalid_expression')

    def test_load_truth(self):
        self.assertEqual(scoring.load_truth('{"numbers": [1, 2], "target": 3}'), ([1, 2], 3))
        self.assertEqual(scoring.load_truth({'numbers': [1, 2], 'target': 3}), ([1, 2], 3))


class Rewards(unittest.TestCase):
    def setUp(self):
        self.env = get_environment('countdown')

    def result(self, text):
        return self.env.score_result(text, {'reward_spec': {'ground_truth': json.dumps({'numbers': NUMS, 'target': TARGET})}}, {})

    def test_binary(self):
        self.assertEqual(countdown_correct(self.result('<answer>7*25+3*2</answer>'), None), 1.0)
        self.assertEqual(countdown_correct(self.result('<answer>7*25+3</answer>'), None), 0.0)

    def test_closeness_only_for_legal_misses(self):
        self.assertAlmostEqual(countdown_closeness(self.result('<answer>7*25+3</answer>'), None), 1 - 3 / 181)
        self.assertEqual(countdown_closeness(self.result('<answer>7*25+3*2</answer>'), None), 0.0)       # correct: no double pay
        self.assertEqual(countdown_closeness(self.result('<answer>180+1</answer>'), None), 0.0)          # illegal numbers
        self.assertEqual(countdown_closeness(self.result('<answer>7*25*(3+2)</answer>'), None), 0.0)      # far away: floored
        self.assertEqual(countdown_closeness(self.result('nothing'), None), 0.0)
        self.assertAlmostEqual(countdown_closeness(self.result('<answer>7*25+3</answer>'), None, scale=0.5), 0.5 * (1 - 3 / 181))

    def test_weighted_sum_ranks_correct_first(self):
        reward = Reward(self.env, [{'fn': 'countdown_correct', 'weight': 1.0}, {'fn': 'countdown_closeness', 'weight': 0.1}])
        good, _ = reward(self.result('<answer>7*25+3*2</answer>'), RewardContext())
        miss, _ = reward(self.result('<answer>7*25+3</answer>'), RewardContext())
        self.assertEqual(good, 1.0); self.assertLess(miss, 0.1); self.assertGreater(miss, 0.0)

    def test_validate_components(self):
        self.assertEqual(validate_components(self.env, [{'fn': 'countdown_closeness', 'weight': 0.1, 'scale': 1.0}]), [])
        self.assertTrue(validate_components(self.env, [{'fn': 'countdown_closeness', 'weight': 0.1, 'bogus': 1}]))
        self.assertTrue(validate_components(self.env, [{'fn': 'countdown_nope'}]))

    def test_generator_contract_runs_real_reward_path(self):
        receipt = self.env.generator_contract(load(EXP))
        self.assertTrue(receipt['generic_reward_passed'])
        self.assertEqual(receipt['cases']['injection']['failure_class'], 'invalid_expression')


class Data(unittest.TestCase):
    def cfg(self, **countdown):
        cfg = copy.deepcopy(load(EXP))
        cfg['environment']['countdown'] = {**cfg['environment']['countdown'], **countdown}
        return cfg

    def test_deterministic_sizes_disjoint_and_solvable(self):
        cfg = self.cfg()
        a, b = cd_data.records(cfg), cd_data.records(cfg)
        self.assertEqual(a, b)
        self.assertEqual({k: len(v) for k, v in a.items()}, {'test': 200, 'validation': 100, 'train': 512})
        keys = {s: {(tuple(sorted(r['extra_info']['numbers'])), r['extra_info']['target']) for r in rows} for s, rows in a.items()}
        self.assertFalse(keys['train'] & keys['validation'] or keys['train'] & keys['test'] or keys['validation'] & keys['test'])
        for rows in a.values():
            for r in rows:
                info = r['extra_info']
                self.assertTrue(3 <= len(info['numbers']) <= 5)
                self.assertNotIn(info['solution'], r['prompt'][0]['content'])
                self.assertTrue(score(f"<answer>{info['solution']}</answer>", info['numbers'], info['target'])['correct'], info)
                self.assertEqual(scoring.load_truth(r['reward_spec']['ground_truth']), (info['numbers'], info['target']))

    def test_held_out_sets_independent_of_train_size(self):
        small, big = cd_data.records(self.cfg(train_size=64)), cd_data.records(self.cfg())
        self.assertEqual(small['test'], big['test']); self.assertEqual(small['validation'], big['validation'])
        self.assertNotEqual(cd_data.records(self.cfg(seed=1))['test'], big['test'])

    def test_custom_prompt_is_used_validated_and_part_of_the_data_contract(self):
        template = 'Reach {target} with {numbers}; show your work, then <answer>EXPR</answer>.'
        rows = cd_data.records(self.cfg(prompt=template))
        info = rows['test'][0]['extra_info']
        self.assertEqual(rows['test'][0]['prompt'][0]['content'], template.format(numbers=info['numbers'], target=info['target']))
        self.assertEqual([r['extra_info'] for r in rows['test']], [r['extra_info'] for r in cd_data.records(self.cfg())['test']])  # same problems
        env = get_environment('countdown')
        self.assertTrue(any('prompt must contain' in m for m in env.validate_experiment(self.cfg(prompt='Reach {target}.'))))
        self.assertEqual(env.validate_experiment(self.cfg(prompt=template)), [])
        with tempfile.TemporaryDirectory() as temp:
            cfg = self.cfg(train_size=32, validation_size=16, test_size=8)
            cfg['evaluation']['fixed_qualitative_samples'] = 4
            cfg['run']['data_dir'] = temp
            cd_data.prepare(cfg)
            with self.assertRaises(RuntimeError):       # data prepared with the default prompt does not satisfy a new prompt
                cd_data.assert_manifest(dict(cfg, environment={**cfg['environment'], 'countdown': {**cfg['environment']['countdown'], 'prompt': template}}))

    def test_prepare_and_manifest_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = self.cfg(train_size=32, validation_size=16, test_size=8)
            cfg['evaluation']['fixed_qualitative_samples'] = 4
            cfg['run']['data_dir'] = temp
            manifest = cd_data.prepare(cfg)
            self.assertEqual(manifest['counts'], {'test': 8, 'validation': 16, 'train': 32, 'canary': 4})
            self.assertEqual(manifest['test_ids'], [f'countdown-test-{i:05d}' for i in range(8)])
            cd_data.assert_manifest(cfg)
            other = copy.deepcopy(cfg)
            other['environment']['countdown']['train_size'] = 33
            with self.assertRaises(RuntimeError):
                cd_data.assert_manifest(other)
            (Path(temp) / 'train.parquet').write_bytes(b'tampered')
            with self.assertRaises(RuntimeError):
                cd_data.assert_manifest(cfg)


class Wiring(unittest.TestCase):
    def test_registered_and_experiment_plans(self):
        from strl.derive import plan
        from strl.overrides import overrides
        cfg = load(EXP)
        env = get_environment(cfg['environment']['name'])
        self.assertEqual(env.name, 'countdown')
        self.assertTrue(env.uses_generic_reward(cfg))
        self.assertEqual(env.skyrl_overrides(cfg, 'train'), {'environment.env_class': 'strl_env'})
        plan(cfg)
        rendered = overrides(cfg, stage='train', run_name='t')
        self.assertIn('environment.env_class=strl_env', rendered)
        self.assertEqual(cfg['compute']['gpus_per_node'], 1)
        self.assertEqual(env.final_eval_ids(cfg)[:2], ['countdown-test-00000', 'countdown-test-00001'])
        self.assertEqual(env.expected_rows(cfg, 'train'), 512)

    def test_validation_errors(self):
        from strl.experiment import ExperimentError
        with self.assertRaises(ExperimentError) as ctx:
            load(EXP, overrides=['environment.countdown.validation_size=50'])
        self.assertIn('eval_batch_size', str(ctx.exception))
        with self.assertRaises(ExperimentError) as ctx:
            load(EXP, overrides=['environment.countdown.typo=1'])
        self.assertIn('environment.countdown', str(ctx.exception))

    def test_eval_record_uses_scorer_not_reward(self):
        class Tok:
            def encode(self, text, add_special_tokens=False): return list(text)
            def decode(self, ids, skip_special_tokens=True): return ''.join(ids)
        trainer = type('T', (), {'tokenizer': Tok()})()
        env = get_environment('countdown')
        truth = json.dumps({'numbers': NUMS, 'target': TARGET})
        item = {'env_extras': {'reward_spec': {'ground_truth': truth}, 'extra_info': {'sample_id': 'countdown-validation-00000'}},
                'output_response': '<answer>7*25+3</answer>', 'score': 0.0983, 'stop_reason': 'stop'}
        rec = env.eval_record(item, trainer, 0)
        self.assertFalse(rec['correct']); self.assertEqual(rec['failure_class'], 'wrong_value'); self.assertEqual(rec['sample_id'], 'countdown-validation-00000')


if __name__ == '__main__':
    unittest.main()
