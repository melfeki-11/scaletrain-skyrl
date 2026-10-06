"""Coding environment contracts: harness (incl. reward hacks), extractor, rewards, group masking, data rules. No Modal needed."""
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strl.rewards import Reward, RewardContext, validate_components  # noqa: E402
from strl.tasks import get_environment  # noqa: E402
from strl.tasks.coding import data as coding_data  # noqa: E402
from strl.tasks.coding.extract import extract_code  # noqa: E402
from strl.tasks.coding.generator import mask_infra_groups  # noqa: E402
from strl.tasks.coding.harness import run_program, suspicious  # noqa: E402

TESTS = ['assert add(1,2)==3', 'assert add(-1,1)==0', 'assert add(10,5)==15']


def run(code, tests=TESTS, **kw):
    return run_program({'code': code, 'tests': tests, 'setup': ''}, cpu_seconds=2, wall_seconds=5, **kw)


class Harness(unittest.TestCase):
    def test_verdicts(self):
        cases = {
            'def add(a,b):\n    return a+b': ('ok', 3),
            'def add(a,b):\n    return 3 if a==1 else a+b+1': ('wrong_answer', 1),
            'def add(a,b)\n    return a+b': ('syntax_error', 0),
            'import nonexistent_mod\ndef add(a,b): return a+b': ('runtime_error', 0),
            'def add(a,b):\n    raise TypeError("x")': ('runtime_error', 0),
        }
        for code, (klass, passed) in cases.items():
            v = run(code)
            self.assertEqual((v['failure_class'], v['passed']), (klass, passed), code)

    def test_exit_hacks_never_pass(self):
        for code in ['import sys\nsys.exit(0)', 'import os\nos._exit(0)', 'import sys\ndef add(a,b):\n    sys.exit(0)', 'raise SystemExit(0)']:
            v = run(code)
            self.assertNotEqual(v['failure_class'], 'ok', code)
            self.assertEqual(v['passed'], 0, code)

    def test_eq_hack_rejected_but_stdlib_types_accepted(self):
        hack = run('class A:\n    def __eq__(self,o): return True\ndef add(a,b): return A()')
        self.assertEqual(hack['failure_class'], 'wrong_answer')
        sub = run('from collections import Counter\nclass C(Counter):\n    def __eq__(self,o): return True\ndef add(a,b): return C()')
        self.assertEqual(sub['failure_class'], 'wrong_answer')
        counter = run('from collections import Counter\ndef f(x): return Counter(x)', ['assert f([1,1,2])==({1:2,2:1})'])
        self.assertEqual(counter['failure_class'], 'ok')

    def test_stdin_and_timeout(self):
        snoop = run('import sys\ndata=sys.stdin.read()\ndef add(a,b): return len(data)')
        self.assertNotEqual(snoop['failure_class'], 'ok')  # tests/nonce were consumed before the candidate ran
        self.assertEqual(run('while True: pass')['failure_class'], 'timeout')

    def test_humaneval_style_check(self):
        v = run('def add(a,b): return a+b', ['def check(candidate):\n    assert candidate(1,2)==3\n    assert abs(candidate(0.1,0.2)-0.3)<1e-9\ncheck(add)\n'])
        self.assertEqual((v['failure_class'], v['total']), ('ok', 1))

    def test_suspicious(self):
        self.assertTrue(suspicious('import sys\nsys.exit(0)'))
        self.assertTrue(suspicious('def __eq__(self, o): return True'))
        self.assertFalse(suspicious('def add(a, b):\n    return a + b'))


class Extract(unittest.TestCase):
    def test_last_python_block_and_special_tokens(self):
        text = 'plan\n```python\nx=1\n```\nfinal:\n```python\ndef f():\n    return 2\n```<|im_end|>'
        self.assertEqual(extract_code(text), 'def f():\n    return 2\n')
        self.assertIsNone(extract_code('no code here'))
        self.assertIsNone(extract_code('```python\n\n```'))
        self.assertEqual(extract_code('<think>```python\nbad\n```</think>```py\nok=1\n```'), 'ok=1\n')


class Rewards(unittest.TestCase):
    def setUp(self):
        self.env = get_environment('coding_exec')
        self.ok = {'failure_class': 'ok', 'passed': 3, 'total': 3, 'compiled': True, 'has_code': True, 'completion': 'x'}
        self.partial = {'failure_class': 'wrong_answer', 'passed': 1, 'total': 3, 'compiled': True, 'has_code': True, 'completion': 'x'}

    def test_weighted_sum_and_components(self):
        reward = Reward(self.env, [{'fn': 'pass_all_tests', 'weight': 1.0}, {'fn': 'partial_tests', 'weight': 0.5}])
        total, parts = reward(self.partial, RewardContext())
        self.assertAlmostEqual(total, 0.5 / 3)
        self.assertEqual(parts, {'pass_all_tests': 0.0, 'partial_tests': 1 / 3})
        self.assertEqual(reward(self.ok, RewardContext())[0], 1.5)

    def test_length_penalty_and_suspicious(self):
        reward = Reward(self.env, [{'fn': 'pass_all_tests'}, {'fn': 'length_penalty', 'weight': 0.5, 'max_tokens': 2},
                                   {'fn': 'no_suspicious_code', 'weight': 1.0}])
        total, _ = reward(dict(self.ok, completion='a' * 400, suspicious=True), RewardContext(count_tokens=lambda t: len(t)))
        self.assertEqual(total, 1.0 - 0.5 - 1.0)

    def test_validation_catches_typos(self):
        self.assertTrue(validate_components(self.env, [{'fn': 'pass_all_test'}]))           # unknown name
        self.assertTrue(validate_components(self.env, [{'fn': 'length_penalty', 'max_token': 5}]))  # unknown parameter
        self.assertFalse(validate_components(self.env, [{'fn': 'length_penalty', 'max_tokens': 5}, {'fn': 'tests.test_coding:custom_reward'}]))

    def test_user_module_component(self):
        reward = Reward(self.env, [{'fn': 'tests.test_coding:custom_reward', 'weight': 2.0, 'bonus': 0.25}])
        self.assertEqual(reward(self.ok, RewardContext())[0], 2.5)


def custom_reward(result, ctx, bonus: float = 0.0):
    return (1.0 if result.get('failure_class') == 'ok' else 0.0) + bonus


class GroupMasking(unittest.TestCase):
    def test_masks_whole_group_with_infra_error(self):
        ids = [types.SimpleNamespace(instance_id=g) for g in ('a', 'a', 'b', 'b')]
        out = {'loss_masks': [[1, 1], [1], [1, 1, 1], [1]], 'rewards': [1.0, 0.0, [0.0, 1.0], 1.0],
               'env_metrics': [{'infra_error': 0.0}, {'infra_error': 1.0}, {'infra_error': 0.0}, {'infra_error': 0.0}]}
        stats = mask_infra_groups(out, ids)
        self.assertEqual(out['loss_masks'][:2], [[0, 0], [0]])
        self.assertEqual(out['rewards'][:2], [0.0, 0.0])
        self.assertEqual(out['loss_masks'][2:], [[1, 1, 1], [1]])
        self.assertEqual(stats['environment/masked_group_fraction'], 0.5)


class Data(unittest.TestCase):
    def test_stub_and_rows(self):
        stub = coding_data.stub('def f(x):\n    return x + 1\nclass A:\n    def g(self):\n        return 2')
        self.assertIn('return None', stub)
        task = coding_data.mbpp_task({'task_id': 7, 'text': 'Add.', 'code': 'def add(a,b): return a+b', 'test_list': TESTS,
                                      'test_setup_code': '', 'challenge_test_list': []}, 'train')
        self.assertIn(TESTS[0], task.prompt)
        self.assertNotIn(TESTS[1], task.prompt)  # hidden tests are graded but not shown
        row = coding_data.to_row(task)
        self.assertEqual(coding_data.task_from_extras(row)['tests'], TESTS)
        self.assertEqual(row['extra_info']['sample_id'], 'mbpp-train-0007')

    def test_score_result_uses_executor_and_flags(self):
        env = get_environment('coding_exec')
        from strl.tasks.coding import executor
        original = executor.score
        executor.score = lambda job, **kw: run_program(job, cpu_seconds=2, wall_seconds=5)
        try:
            task = coding_data.to_row(coding_data.mbpp_task({'task_id': 1, 'text': 'Add.', 'code': '', 'test_list': TESTS,
                                                             'test_setup_code': '', 'challenge_test_list': []}, 'train'))
            good = env.score_result('```python\ndef add(a,b):\n    return a+b\n```', task, {})
            self.assertTrue(good['correct'])
            self.assertEqual(env.score_result('no code', task, {})['failure_class'], 'no_code_block')
            hack = env.score_result('```python\nimport sys\nsys.exit(0)\n```', task, {})
            self.assertFalse(hack['correct'])
            self.assertTrue(hack['suspicious'])
        finally:
            executor.score = original


if __name__ == '__main__':
    unittest.main()
