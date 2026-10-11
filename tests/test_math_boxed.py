"""math_boxed environment (strl/tasks/math_boxed): boxed extraction, normalization, scorer, rewards, data selection + manifest (offline,
with a synthetic source dataset), spec wiring and both r3b experiments' plans."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import copy
import json
import tempfile
import unittest
from pathlib import Path

from strl.experiment import load
from strl.rewards import Reward, RewardContext, validate_components
from strl.tasks import get_environment
from strl.tasks.math_boxed import data as mb_data, scoring
from strl.tasks.math_boxed.rewards import math_correct, math_format

EXP = 'experiments/r3b-math-smollm2-2node.yaml'
EXP_1NODE = 'experiments/r3b-math-smollm2-1node.yaml'


class Boxed(unittest.TestCase):
    def test_last_box_brace_matched(self):
        self.assertEqual(scoring.boxed_answers(r'a \boxed{1} b \boxed{\frac{1}{2}} c'), ['1', r'\frac{1}{2}'])
        self.assertEqual(scoring.last_boxed(r'\boxed{\{1, 2\}}'), r'\{1, 2\}')
        self.assertEqual(scoring.last_boxed(r'$\boxed {7}$'), '7')
        self.assertEqual(scoring.last_boxed(r'\fbox{7}'), '7')
        self.assertEqual(scoring.last_boxed(r'so \boxed 5$.'), '5')

    def test_nested_and_unbalanced(self):
        self.assertEqual(scoring.boxed_answers(r'\boxed{\boxed{1}}'), [r'\boxed{1}'])     # the inner box is not a separate answer
        self.assertEqual(scoring.boxed_answers(r'\boxed{1} \boxed{\frac{3}{4}'), ['1'])   # unbalanced trailing box ignored
        self.assertIsNone(scoring.last_boxed(r'\boxed{}'))
        self.assertIsNone(scoring.last_boxed('no box here'))
        self.assertIsNone(scoring.last_boxed(''))

    def test_terminal_tokens_stripped(self):
        self.assertEqual(scoring.last_boxed(r'\boxed{3}<|im_end|>'), '3')


class Normalize(unittest.TestCase):
    def eq(self, a, b):
        self.assertEqual(scoring.normalize(a), scoring.normalize(b), (a, b))

    def ne(self, a, b):
        self.assertNotEqual(scoring.normalize(a), scoring.normalize(b), (a, b))

    def test_review_over_credit_cases_are_not_credited(self):
        self.ne('-12', '-12 \\text{ or } -30')      # "\text{ " that is not trailing units keeps the whole answer
        self.ne('1125', '1, 125')                   # a list is not a thousands-separated number
        self.ne('3', '10 = 3')                      # only a variable on the left is dropped
        self.eq('5', '5\\text{ cm}^2')
        self.eq('1125', '1,125')

    def test_equivalent_spellings(self):
        self.eq(r'\dfrac{3}{4}', r'\frac{3}{4}')
        self.eq(r'\tfrac34', r'\frac{3}{4}')
        self.eq('3/4', r'\frac{3}{4}')
        self.eq('0.5', r'\frac{1}{2}')
        self.eq(r'\left( 1, 2 \right)', '(1,2)')
        self.eq(r'90^\circ', '90')
        self.eq(r'90^{\circ}', '90')
        self.eq(r'\$5', '5')
        self.eq(r'50\%', '50')
        self.eq(r'5 \text{ cm}', '5')
        self.eq(r'\text{(A)}', '(A)')
        self.eq('x = 7', '7')
        self.eq(r'\sqrt3', r'\sqrt{3}')
        self.eq('2.0', '2')
        self.eq('1,000', '1000')
        self.eq('.5', '0.5')
        self.eq('7.', '7')
        self.eq(r'2\,000', '2000')

    def test_different_answers_stay_different(self):
        self.ne(r'\frac{4}{3}', r'\frac{3}{4}')
        self.ne(r'\frac{6}{8}', r'\frac{3}{4}')       # exact match: unreduced forms are not credited
        self.ne('(1,2)', '(2,1)')
        self.ne('1,2', '12')                          # a list is not a thousands-separated number
        self.ne('-3', '3')
        self.ne(r'2\sqrt{2}', r'\sqrt{8}')

    def test_none_and_long(self):
        self.assertIsNone(scoring.normalize(None))
        self.assertIsNone(scoring.normalize('  '))
        self.assertLessEqual(len(scoring.normalize('1' * 5000)), scoring.MAX_ANSWER_CHARS)


class Scorer(unittest.TestCase):
    def test_classes(self):
        ok = scoring.score_completion(r'so $\boxed{\dfrac{3}{4}}$', r'\frac34')
        self.assertTrue(ok['correct']); self.assertEqual(ok['failure_class'], 'ok'); self.assertTrue(ok['single_answer'])
        wrong = scoring.score_completion(r'\boxed{1}', r'\frac34')
        self.assertFalse(wrong['correct']); self.assertEqual(wrong['failure_class'], 'wrong_answer')
        none = scoring.score_completion('3/4', r'\frac34')
        self.assertEqual(none['failure_class'], 'no_answer'); self.assertIsNone(none['parsed'])

    def test_hedging(self):
        r = scoring.score_completion(r'\boxed{1} or \boxed{\frac{3}{4}}', r'\frac{3}{4}')
        self.assertTrue(r['correct']); self.assertFalse(r['single_answer']); self.assertEqual(r['distinct_answers'], 2)
        same = scoring.score_completion(r'\boxed{0.5} hence \boxed{\frac12}', r'\frac{1}{2}')
        self.assertTrue(same['single_answer'])   # the same answer twice is not a hedge

    def test_reference_answer(self):
        self.assertEqual(scoring.reference_answer(r'... so $a+b=\boxed{0}$.'), '0')
        self.assertIsNone(scoring.reference_answer(r'there are $\boxed{}$ primes'))


class Rewards(unittest.TestCase):
    def setUp(self):
        self.env = get_environment('math_boxed')

    def result(self, text, truth=r'\frac{3}{4}'):
        return self.env.score_result(text, {'reward_spec': {'ground_truth': truth}}, {})

    def test_terms(self):
        self.assertEqual(math_correct(self.result(r'\boxed{\frac34}'), None), 1.0)
        self.assertEqual(math_correct(self.result(r'\boxed{1}'), None), 0.0)
        self.assertEqual(math_format(self.result(r'\boxed{1}'), None), 1.0)
        self.assertEqual(math_format(self.result('3/4'), None), 0.0)
        self.assertEqual(math_format(self.result(r'\boxed{1} \boxed{2}'), None), 0.0)
        self.assertEqual(math_format(self.result(r'\boxed{1} \boxed{2}'), None, single=False), 1.0)

    def test_weighted_sum_ranks_correct_first(self):
        reward = Reward(self.env, [{'fn': 'math_correct', 'weight': 1.0}, {'fn': 'math_format', 'weight': 0.1}])
        good, parts = reward(self.result(r'\boxed{\frac{3}{4}}'), RewardContext())
        hedge, _ = reward(self.result(r'\boxed{1}, \boxed{\frac{3}{4}}'), RewardContext())
        miss, _ = reward(self.result(r'\boxed{1}'), RewardContext())
        bare, _ = reward(self.result('3/4'), RewardContext())
        self.assertAlmostEqual(good, 1.1); self.assertEqual(parts, {'math_correct': 1.0, 'math_format': 1.0})
        self.assertGreater(good, hedge); self.assertGreater(hedge, miss); self.assertGreater(miss, bare)

    def test_validate_components(self):
        self.assertEqual(validate_components(self.env, [{'fn': 'math_format', 'weight': 0.1, 'single': False}]), [])
        self.assertTrue(validate_components(self.env, [{'fn': 'math_format', 'weight': 0.1, 'bogus': 1}]))
        self.assertTrue(validate_components(self.env, [{'fn': 'math_nope'}]))

    def test_generator_contract_runs_real_reward_path(self):
        receipt = self.env.generator_contract(load(EXP))
        self.assertTrue(receipt['generic_reward_passed'])
        self.assertEqual(receipt['cases']['no_box']['failure_class'], 'no_answer')
        self.assertEqual(receipt['cases']['unreduced']['failure_class'], 'wrong_answer')


def _fake_source():
    """A synthetic stand-in for the HF dataset (same columns): levels 1-5 x 7 subjects, a few long or box-less rows."""
    subjects = ['Algebra', 'Geometry', 'Number Theory', 'Prealgebra', 'Precalculus', 'Intermediate Algebra', 'Counting & Probability']

    def rows(n, tag):
        out = []
        for i in range(n):
            problem = f'{tag} problem {i}: compute {i} + 1.' + (' x' * 2000 if i % 97 == 0 else '')
            solution = f'It is $\\boxed{{{i + 1}}}$.' if i % 89 else 'It is unknowable: $\\boxed{}$.'
            out.append({'problem': problem, 'level': f'Level {1 + i % 5}', 'solution': solution, 'type': subjects[i % 7]})
        return out
    return {'train': rows(1500, 'train'), 'test': rows(600, 'test')}


class Data(unittest.TestCase):
    def cfg(self, **settings):
        cfg = copy.deepcopy(load(EXP))
        cfg['environment']['math_boxed'] = {**cfg['environment']['math_boxed'], 'train_size': 300, 'validation_size': 64,
                                            'test_size': 50, **settings}
        return cfg

    def test_selection_filters_disjoint_deterministic(self):
        source = _fake_source()
        cfg = self.cfg()
        a, stats = mb_data.records(cfg, source)
        b, _ = mb_data.records(cfg, source)
        self.assertEqual(a, b)
        self.assertEqual({k: len(v) for k, v in a.items()}, {'test': 50, 'validation': 64, 'train': 300})
        picks = stats['source_indices']
        self.assertFalse(set(picks['train']) & set(picks['validation']))
        self.assertGreater(stats['train_dropped']['too_long'], 0); self.assertGreater(stats['train_dropped']['no_reference'], 0)
        for split, rows in a.items():
            for r in rows:
                info = r['extra_info']
                self.assertIn(info['level'], (1, 2, 3))
                self.assertLessEqual(len(r['prompt'][0]['content']), 1500 + 200)
                self.assertNotIn(info['solution'], r['prompt'][0]['content'])
                self.assertIn('\\boxed{}', r['prompt'][0]['content'])
                self.assertEqual(info['source_split'], 'test' if split == 'test' else 'train')
                self.assertTrue(scoring.score_completion(f"\\boxed{{{r['reward_spec']['ground_truth']}}}", r['reward_spec']['ground_truth'])['correct'])

    def test_held_out_sets_independent_of_train_size(self):
        source = _fake_source()
        small, _ = mb_data.records(self.cfg(train_size=100), source)
        big, _ = mb_data.records(self.cfg(), source)
        self.assertEqual(small['test'], big['test']); self.assertEqual(small['validation'], big['validation'])
        other, _ = mb_data.records(self.cfg(split_seed=1), source)
        self.assertNotEqual(other['test'], big['test'])

    def test_too_many_requested_fails(self):
        with self.assertRaises(ValueError):
            mb_data.records(self.cfg(train_size=5000), _fake_source())

    def test_prepare_and_manifest_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = self.cfg(train_size=64, validation_size=16, test_size=8)
            cfg['evaluation']['fixed_qualitative_samples'] = 4
            cfg['evaluation']['eval_batch_size'] = 16
            cfg['run']['data_dir'] = temp
            manifest = mb_data.prepare(cfg, source=_fake_source())
            self.assertEqual(manifest['counts'], {'test': 8, 'validation': 16, 'train': 64, 'canary': 4})
            self.assertEqual(manifest['test_ids'], [f'math_boxed-test-{i:05d}' for i in range(8)])
            mb_data.assert_manifest(cfg)
            other = copy.deepcopy(cfg)
            other['environment']['math_boxed']['revision'] = 'f' * 40
            with self.assertRaises(RuntimeError):
                mb_data.assert_manifest(other)
            (Path(temp) / 'train.parquet').write_bytes(b'tampered')
            with self.assertRaises(RuntimeError):
                mb_data.assert_manifest(cfg)


class Wiring(unittest.TestCase):
    def test_registered_and_both_experiments_plan(self):
        from strl.derive import plan
        from strl.overrides import overrides
        for path, nodes in ((EXP, 2), (EXP_1NODE, 1)):
            cfg = load(path)
            env = get_environment(cfg['environment']['name'])
            self.assertEqual(env.name, 'math_boxed')
            self.assertTrue(env.uses_generic_reward(cfg))
            self.assertEqual(env.skyrl_overrides(cfg, 'train'), {'environment.env_class': 'strl_env'})
            plan(cfg)
            rendered = overrides(cfg, stage='train', run_name='t')
            self.assertIn('environment.env_class=strl_env', rendered)
            self.assertEqual(cfg['compute']['nodes'], nodes)
            self.assertEqual(cfg['finetune']['mode'], 'full')
            self.assertEqual(env.final_eval_ids(cfg)[:2], ['math_boxed-test-00000', 'math_boxed-test-00001'])
            self.assertEqual(env.expected_rows(cfg, 'train'), 3000)

    def test_one_node_differs_only_in_nodes(self):
        a, b = load(EXP), load(EXP_1NODE)
        for key in ('model', 'environment', 'reward', 'algorithm', 'evaluation', 'finetune', 'inference'):
            self.assertEqual(a[key], b[key], key)

    def test_validation_errors(self):
        from strl.experiment import ExperimentError
        for override, needle in (('environment.math_boxed.validation_size=50', 'eval_batch_size'),
                                 ('environment.math_boxed.typo=1', 'environment.math_boxed'),
                                 ('environment.math_boxed.revision=main', 'revision'),
                                 ('environment.math_boxed.levels=[6]', 'levels')):
            with self.assertRaises(ExperimentError, msg=override) as ctx:
                load(EXP, overrides=[override])
            self.assertIn(needle, str(ctx.exception))

    def test_eval_record_uses_scorer_not_reward(self):
        class Tok:
            def encode(self, text, add_special_tokens=False): return list(text)
            def decode(self, ids, skip_special_tokens=True): return ''.join(ids)
        trainer = type('T', (), {'tokenizer': Tok()})()
        env = get_environment('math_boxed')
        item = {'env_extras': {'reward_spec': {'ground_truth': '7'}, 'extra_info': {'sample_id': 'math_boxed-validation-00000'}},
                'output_response': r'\boxed{8}', 'score': 0.1, 'stop_reason': 'stop'}
        rec = env.eval_record(item, trainer, 0)
        self.assertFalse(rec['correct']); self.assertEqual(rec['failure_class'], 'wrong_answer')
        truncated = env.eval_record({**item, 'output_response': r'\boxed{7}', 'stop_reason': 'length'}, trainer, 0)
        self.assertFalse(truncated['correct']); self.assertEqual(truncated['failure_class'], 'truncated')
        self.assertFalse(env.score(r'\boxed{7}', '7', 'length')); self.assertTrue(env.score(r'\boxed{7}', '7', 'stop'))


if __name__ == '__main__':
    unittest.main()
