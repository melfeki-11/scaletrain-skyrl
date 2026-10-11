"""user_rewards.brevity:short_correct_bonus (experiments/r3a-cd-phi4.yaml): pays only on correct answers, shorter pays more, bounded."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import unittest

from strl.experiment import load
from strl.rewards import Reward, RewardContext, validate_components
from strl.tasks import get_environment
from strl.tasks.countdown import scoring
from user_rewards.brevity import short_correct_bonus

EXP = 'experiments/r3a-cd-phi4.yaml'
NUMS, TARGET = [3, 7, 25, 2], 181


def ctx(limit=1024):
    return RewardContext(max_generate_length=limit, count_tokens=lambda text: len(text.split()) or 1)


class ShortCorrectBonus(unittest.TestCase):
    def test_zero_when_wrong_even_if_short(self):
        for text in ('<answer>25 * 7 + 3</answer>', '<answer>180 + 1</answer>', 'no answer', ''):
            result = {'completion': text, **scoring.score_expression(text, NUMS, TARGET)}
            self.assertEqual(short_correct_bonus(result, ctx(), bonus=0.1, max_tokens=100), 0.0, text)

    def test_shorter_correct_pays_more_and_is_bounded(self):
        short = {'completion': 'x ' * 10, 'correct': True}
        long_ = {'completion': 'x ' * 80, 'correct': True}
        over = {'completion': 'x ' * 500, 'correct': True}
        b_short, b_long = (short_correct_bonus(r, ctx(), bonus=0.1, max_tokens=100) for r in (short, long_))
        self.assertGreater(b_short, b_long)
        self.assertLessEqual(b_short, 0.1)
        self.assertEqual(short_correct_bonus(over, ctx(), bonus=0.1, max_tokens=100), 0.0)

    def test_default_limit_is_generation_cap(self):
        r = {'completion': 'x ' * 512, 'correct': True}
        self.assertAlmostEqual(short_correct_bonus(r, ctx(1024), bonus=0.1), 0.05)

    def test_experiment_reward_ranks_correct_above_every_wrong(self):
        cfg = load(EXP)
        env = get_environment(cfg['environment']['name'])
        self.assertEqual(validate_components(env, cfg['reward']['components']), [])
        reward = Reward(env, cfg['reward']['components'])
        truth = '{"numbers": [3, 7, 25, 2], "target": 181}'
        extras = {'reward_spec': {'ground_truth': truth}}
        good, _ = reward(env.score_result('<answer>7 * 25 + 3 * 2</answer><|end|>', extras, cfg), RewardContext(ground_truth=truth))
        bad, parts = reward(env.score_result('<answer>25 * 7 + 3</answer>', extras, cfg), RewardContext(ground_truth=truth))
        self.assertGreater(good, 1.0)
        self.assertLessEqual(good, 1.1)
        self.assertEqual(bad, 0.0)
        self.assertEqual(parts['user_rewards.brevity:short_correct_bonus'], 0.0)


if __name__ == '__main__':
    unittest.main()
