"""Monitoring contracts: critic (queue, drain, USD cap, never blocks), live-monitor flags, comparison, planning warnings."""
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strl.monitor import critic  # noqa: E402
from strl.monitor.live import flags  # noqa: E402


class Critic(unittest.TestCase):
    def tearDown(self):
        critic.reset()

    def test_disabled_is_noop(self):
        self.assertFalse(critic.configure({'enabled': False}))
        self.assertEqual(critic.submit_failures(5, ['p'], ['c'], [0.0], None, None), 0)
        self.assertEqual(critic.drain(), ({}, []))

    def test_queue_drain_and_cap(self):
        import strl.monitor.critic as module
        calls = []

        def fake(item):
            calls.append(item)
            return {'source_step': item['source_step'], 'idx': item['idx'], 'env_failure': item['env_failure'], 'critic_mode': 'edge_case',
                    'rationale': 'r', 'fix_hint': 'f', 'prompt': item['prompt'], 'completion_tail': item['completion'], '_cost': 0.6}
        original = module.analyse
        module.analyse = fake
        try:
            critic.configure({'enabled': True, 'every_n_steps': 5, 'max_rows': 3, 'budget_usd': 1.0})
            self.assertEqual(critic.submit_failures(3, ['p'] * 4, ['c'] * 4, [0.0] * 4, None, None), 0)   # not a critic step
            queued = critic.submit_failures(5, ['p'] * 4, ['c'] * 4, [0.0, 1.0, 0.0, 0.0],
                                            [{'task_metric': 0.0, 'failure/wrong_answer': 1.0}, {'task_metric': 1.0}, {'task_metric': 0.0}, {'task_metric': 0.0}], None)
            self.assertEqual(queued, 3)
            deadline = time.time() + 5
            rows = []
            while time.time() < deadline and len(rows) < 2:
                scalars, new = critic.drain()
                rows += new
                time.sleep(0.05)
            scalars, _ = critic.drain()
            self.assertGreaterEqual(len(rows), 2)                 # the third is skipped once the 1 USD cap is hit
            self.assertEqual(rows[0]['env_failure'], 'wrong_answer')
            self.assertEqual(scalars['critic/disabled'], 1.0)
            self.assertAlmostEqual(scalars['critic/edge_case_frac'], 1.0)
        finally:
            module.analyse = original

    def test_full_queue_drops_instead_of_blocking(self):
        import strl.monitor.critic as module
        original = module.analyse
        module.analyse = lambda item: time.sleep(2) or {}      # a slow LLM: the bounded queue fills up
        try:
            critic.configure({'enabled': True, 'every_n_steps': 1, 'max_rows': 100})
            started = time.time()
            for _ in range(5):
                critic.submit_failures(1, ['p'] * 40, ['c'] * 40, [0.0] * 40, None, None)
            self.assertLess(time.time() - started, 0.5)             # the training thread never waits
            self.assertGreater(module._S['dropped'], 0)
        finally:
            module.analyse = original


class LiveFlags(unittest.TestCase):
    def test_stall_nan_env_budget(self):
        rows = [{'timing/step': 40.0}] * 5 + [{'environment/masked_group_fraction': 0.2, 'policy/policy_loss': float('nan')}]
        out = flags(rows, {'updated_at': time.time() - 1200, 'budget_consumed_node_hours': 2.0}, None, 1.0)
        self.assertTrue(any(f.startswith('STALL') for f in out))
        self.assertIn('NAN', out)
        self.assertTrue(any(f.startswith('ENV_FAILURES') for f in out))
        self.assertIn('BUDGET', out)
        self.assertEqual(flags([{'timing/step': 40.0}], {'updated_at': time.time(), 'budget_consumed_node_hours': 0.1}, None, 1.0), [])


class CompareAndPlan(unittest.TestCase):
    def test_curve_summary(self):
        from strl.monitor.compare import summarize_curves
        out = summarize_curves([{'policy/policy_loss': 0.1, 'train/learning_signal_fraction': 0.2, 'environment/reward_components/partial_tests': 0.3},
                                {'policy/policy_loss': 0.1, 'train/learning_signal_fraction': 0.5, 'environment/reward_components/partial_tests': 0.6,
                                 'critic/edge_case_frac': 0.7}])
        self.assertEqual(out['optimizer_steps'], 2)
        self.assertEqual(out['train/learning_signal_fraction'], {'first': 0.2, 'last': 0.5, 'max': 0.5})
        self.assertEqual(out['reward_components']['partial_tests'], {'first': 0.3, 'last': 0.6})
        self.assertEqual(out['critic_modes'], {'edge_case': 0.7})

    def test_plan_reproduces_measured_profile_and_warns(self):
        from strl.derive import plan
        from strl.experiment import load
        report = plan(load('base/gsm8k.yaml'))
        self.assertEqual(report['estimates']['step_s'], 51.2)        # the measured verified run
        partial = plan(load('experiments/mbpp-qwen35-partial.yaml'))
        self.assertTrue(any(w.startswith('REWARD') for w in partial['warnings']))
        self.assertAlmostEqual(partial['estimates']['node_hours'], 0.5)   # 4-GPU slice

    def test_slice_budget_arithmetic(self):
        from strl.budget import remaining_seconds
        attempts = [{'job_id': 'c1', 'status': 'COMPLETED', 'allocated_at': '2026-10-06T10:00:00Z', 'terminal_at': '2026-10-06T10:16:00Z', 'gpu_fraction': 0.125},
                    {'job_id': 'd1', 'status': 'COMPLETED', 'allocated_at': '2026-10-06T10:00:00Z', 'terminal_at': '2026-10-06T10:05:00Z', 'gpu_fraction': 0.0}]
        self.assertEqual(remaining_seconds(attempts, 7200), 7200 - 120)


if __name__ == '__main__':
    unittest.main()


class CanaryEnvironmentGate(unittest.TestCase):
    """can-hb-b: every Harbor trial failed on turn 1 (context too small) and the canary still 'passed'. Never again."""

    def setUp(self):
        import os
        import tempfile
        import types
        callbacks = types.ModuleType('skyrl.train.utils.callbacks')
        callbacks.TrainingCallback = object
        for name in ('skyrl', 'skyrl.train', 'skyrl.train.utils'):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules['skyrl.train.utils.callbacks'] = callbacks
        self.tmp = tempfile.mkdtemp()
        os.environ.update(GRPO_TRAIN_DEADLINE=str(time.time() + 3600), STRL_ALERTS='0', GRPO_CANARY_REQUIRE_GRADIENT='0')
        from strl import tracking
        self.tracking = tracking
        self.callback = tracking.make_callback_class()()

    def trainer(self, all_metrics):
        cfg = types_ns(trainer=types_ns(train_batch_size=8, export_path=f'{self.tmp}/exports', max_training_steps=100))
        return types_ns(cfg=cfg, all_metrics=all_metrics)

    def step(self, step, all_metrics, grad=1.0):
        control = types_ns(should_save=False, should_evaluate=False)
        self.callback.on_step_end(self.trainer(all_metrics), types_ns(global_step=step, metrics={'policy/grad_norm': grad}), control)

    def test_failure_fraction_sources(self):
        f = self.tracking.env_failure_fraction
        self.assertEqual(f({'environment/masked_group_fraction': 0.25}, 8), 0.25)        # coding_exec generator
        self.assertEqual(f({'generate/num_masked_instances': 8}, 8), 1.0)                # Harbor generator (can-hb-b)
        self.assertIsNone(f({'reward/avg_raw_reward': 0.0}, 8))                           # gsm8k: no environment failures

    def test_mostly_masked_canary_fails(self):
        self.step(1, {'generate/num_masked_instances': 8})
        with self.assertRaisesRegex(RuntimeError, 'masked by environment failures'):
            self.step(2, {'generate/num_masked_instances': 8})

    def test_healthy_canary_passes_and_records_fraction(self):
        import json
        self.step(1, {'environment/masked_group_fraction': 0.0})
        self.step(2, {'environment/masked_group_fraction': 0.125})
        canary = json.loads((Path(self.tmp) / 'canary.json').read_text())
        self.assertEqual(canary['env_failure_fraction_steps_1_2'], 0.125)


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


class StepWiseDiagnostics(unittest.TestCase):
    def test_turn_rows_collapse_to_trajectories(self):
        from strl.tracking import per_trajectory, rollout_diagnostics
        # 2 groups x 2 trajectories; trajectory turns: [3, 1 | 2, 2]; one success (reward on its last turn) in group 1
        last = [False, False, True, True, False, True, False, True]
        rewards = [0, 0, 1.0, 0, 0, 0, 0, 0]
        prompts, texts, r, stops = per_trajectory(last, list('abcdefgh'), list('ABCDEFGH'), rewards, ['stop'] * 8)
        self.assertEqual((prompts, texts, r), (['c', 'd', 'f', 'h'], ['C', 'D', 'F', 'H'], [1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(rollout_diagnostics(texts, r, stops, 2)['train/learning_signal_fraction'], 0.5)
        single = per_trajectory(None, ['p'], ['x'], [1.0], ['stop'])          # single-turn output passes through
        self.assertEqual(single, (['p'], ['x'], [1.0], ['stop']))
