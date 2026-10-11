"""user_rewards.math_equiv:equivalent_answer_credit (experiments/r3d-math-granite-equiv.yaml): partial credit only for an un-hedged,
non-exact answer whose VALUE equals the reference; never on exact hits, hedges, approximations or non-evaluable answers."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import unittest

from strl.experiment import load
from strl.rewards import Reward, RewardContext, validate_components
from strl.tasks import get_environment
from strl.tasks.math_boxed import scoring
from user_rewards.math_equiv import equivalent, equivalent_answer_credit, evaluate

ARM_A, ARM_B = 'experiments/r3d-math-granite-exact.yaml', 'experiments/r3d-math-granite-equiv.yaml'
FN = 'user_rewards.math_equiv:equivalent_answer_credit'


def credit_for(completion: str, truth: str, credit: float = 0.3) -> float:
    result = {'completion': completion, **scoring.score_completion(completion, truth)}
    return equivalent_answer_credit(result, RewardContext(ground_truth=truth), credit=credit)


class Evaluator(unittest.TestCase):
    def test_values(self):
        from fractions import Fraction
        self.assertEqual(evaluate(scoring.normalize('\\frac{6}{8}')), (Fraction(3, 4),))
        self.assertEqual(evaluate(scoring.normalize('2^{10}-1')), (Fraction(1023),))
        self.assertEqual(evaluate(scoring.normalize('\\sqrt{16}')), (Fraction(4),))
        self.assertEqual(evaluate(scoring.normalize('(1, -2)')), (Fraction(1), Fraction(-2)))
        self.assertAlmostEqual(evaluate(scoring.normalize('2\\sqrt{2}'))[0], 2 * 2 ** 0.5)

    def test_not_evaluable(self):
        for text in ('x+1', '\\text{Monday}', '[1,2)', '\\infty', '1/0', '\\sqrt{-4}', '2^{x}', '9' * 200, '', None, '((((1', '\\frac{1}'):
            self.assertIsNone(evaluate(scoring.normalize(text) if text else text), text)

    def test_equivalences(self):
        pairs = [('\\frac{6}{8}', '\\frac{3}{4}'), ('0.75', '\\frac{3}{4}'), ('\\sqrt{8}', '2\\sqrt{2}'),
                 ('\\frac{1}{\\sqrt{3}}', '\\frac{\\sqrt{3}}{3}'), ('(2, 6/2)', '(2,3)'), ('\\frac{2\\pi}{4}', '\\frac{\\pi}{2}'),
                 ('3\\cdot 4', '12'), ('8^{1/3}', '2')]
        for a, b in pairs:
            self.assertTrue(equivalent(scoring.normalize(a), scoring.normalize(b)), (a, b))

    def test_non_equivalences(self):
        pairs = [('0.333', '\\frac{1}{3}'), ('1.41421356237', '\\sqrt{2}'), ('3.14159265359', '\\pi'), ('\\frac{4}{3}', '\\frac{3}{4}'),
                 ('(3, 2)', '(2, 3)'), ('2, 3, 4', '2, 3'), ('x', 'x')]
        for a, b in pairs:
            self.assertFalse(equivalent(scoring.normalize(a), scoring.normalize(b)), (a, b))


class Credit(unittest.TestCase):
    def test_paid_on_equivalent_miss(self):
        self.assertEqual(credit_for('So the answer is $\\boxed{\\frac{6}{8}}$.<|end_of_text|>', '\\frac{3}{4}'), 0.3)
        self.assertEqual(credit_for('\\boxed{\\sqrt{8}}', '2\\sqrt{2}', credit=0.5), 0.5)

    def test_never_on_exact_hit(self):
        self.assertEqual(credit_for('\\boxed{\\dfrac34}', '\\frac{3}{4}'), 0.0)

    def test_never_on_hedge_wrong_or_missing(self):
        self.assertEqual(credit_for('\\boxed{\\frac{6}{8}} or maybe \\boxed{1}... \\boxed{\\frac{6}{8}}', '\\frac{3}{4}'), 0.0)
        self.assertEqual(credit_for('\\boxed{\\frac{4}{3}}', '\\frac{3}{4}'), 0.0)
        self.assertEqual(credit_for('The answer is 6/8.', '\\frac{3}{4}'), 0.0)
        self.assertEqual(credit_for('\\boxed{0.333}', '\\frac{1}{3}'), 0.0)
        self.assertEqual(credit_for('\\boxed{x+1}', 'x+1'), 0.0)     # exact: paid by math_correct, not here

    def test_credit_must_rank_below_canonical(self):
        with self.assertRaises(ValueError):
            credit_for('\\boxed{\\frac{6}{8}}', '\\frac{3}{4}', credit=1.0)


class Arms(unittest.TestCase):
    def test_arms_differ_only_in_reward_and_b_ranks_correct_first(self):
        a, b = load(ARM_A), load(ARM_B)
        self.assertEqual([c['fn'] for c in a['reward']['components']], ['math_correct'])
        self.assertEqual([c['fn'] for c in b['reward']['components']], ['math_correct', FN])
        strip = lambda cfg: {k: v for k, v in cfg.items() if k not in {'reward', 'tracking', 'name', 'description', 'extends', '_meta'}}
        self.assertEqual(strip(a), strip(b))
        self.assertEqual({k: v for k, v in a['tracking'].items() if k != 'tags'}, {k: v for k, v in b['tracking'].items() if k != 'tags'})
        env = get_environment(b['environment']['name'])
        self.assertEqual(validate_components(env, b['reward']['components']), [])
        reward = Reward(env, b['reward']['components'])
        truth = '\\frac{3}{4}'
        extras = {'reward_spec': {'ground_truth': truth}}
        score = lambda text: reward(env.score_result(text, extras, b), RewardContext(ground_truth=truth, extras=extras))
        right, _ = score('\\boxed{\\frac{3}{4}}<|end_of_text|>')
        equiv, parts = score('\\boxed{\\frac{6}{8}}')
        wrong, _ = score('\\boxed{\\frac{4}{3}}')
        self.assertEqual(right, 1.0)
        self.assertAlmostEqual(equiv, 0.3)
        self.assertEqual(parts[FN], 0.3)
        self.assertEqual(wrong, 0.0)
        self.assertTrue(env.generator_contract(b)['generic_reward_passed'])
        self.assertTrue(env.generator_contract(a)['generic_reward_passed'])


if __name__ == '__main__':
    unittest.main()
