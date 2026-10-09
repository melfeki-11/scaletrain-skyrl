"""mimo Track C: in-loop paired result, independent verify, compare/report, live flags, fail-fast rules, predictive deadline.
No network, Modal, GPU, SkyRL or LiteLLM key needed: synthetic eval records, fake trainer/callback inputs, fake W&B/S3."""
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strl import inloop  # noqa: E402
from strl.experiment import ExperimentError, load  # noqa: E402
from strl.monitor import compare, live, report  # noqa: E402
from strl.tasks import get_environment  # noqa: E402
from strl.tasks.mimo import results  # noqa: E402

SMOKE = 'experiments/mimo-qwen35-0p8b-smoke.yaml'


def ns(**kw):
    return types.SimpleNamespace(**kw)


def records(values: dict, step=0, reps=2, masked=(), extra=None):
    """Eval records for {domain: [metric per (task, rep) in order]}; ids '<domain>-t<i>#r<rep>' like MimoSpec.eval_record."""
    out = []
    for domain, vals in values.items():
        for n, v in enumerate(vals):
            i, rep = divmod(n, reps)
            sample_id = f'{domain}-t{i}#r{rep}'
            out.append({'step': step, 'sample_id': sample_id, 'task': f'{domain}-t{i}', 'domain': domain, 'rep': rep, 'task_metric': v,
                        'passed': v >= 0.999, 'correct': v >= 0.999, 'masked': sample_id in masked, 'extra': dict(extra or {})})
    return out


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


BASE = {'music': [0.2] * 8, 'code': [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 'general': [0.4] * 8}
TRAINED = {'music': [0.5, 0.6] * 4, 'code': [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], 'general': [0.5] * 8}


class InLoopResult(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = load(SMOKE)
        self.cfg['evaluation']['bootstrap_samples'] = 400
        self.env = get_environment('mimo')

    def samples(self, steps):
        path = self.tmp / 'exports' / 'evaluation_samples.jsonl'
        rows = []
        for step in steps:
            rows += records(BASE if step == 0 else TRAINED, step=step)
        write_jsonl(path, rows)
        return path

    def test_pairs_first_and_last_eval_and_writes_the_verify_files(self):
        result = inloop.in_loop_result(self.cfg, self.env, self.samples([0, 5, 10]), self.tmp / 'final_eval', optimizer_steps=10)
        self.assertEqual((result['base_step'], result['trained_step'], result['selected_step']), (0, 10, 10))
        self.assertIn('no checkpoint selection', result['selection'])
        self.assertEqual(result['eval_steps'], [0, 5, 10])
        for name in ('base.jsonl', 'candidate.jsonl', 'result.json', 'README.md'):
            self.assertTrue((self.tmp / 'final_eval' / name).is_file(), name)
        self.assertEqual({r['step'] for r in inloop.read_jsonl(self.tmp / 'final_eval' / 'candidate.jsonl')}, {10})
        self.assertEqual({r['step'] for r in inloop.read_jsonl(self.tmp / 'final_eval' / 'base.jsonl')}, {0})
        self.assertEqual(set(result['domains']), {'music', 'code', 'general'})
        self.assertAlmostEqual(result['domains']['music']['delta'], 0.35)
        self.assertEqual(result['macro']['domains'], ['code', 'general', 'music'])
        self.assertEqual((result['eval_repeats'], result['test']['examples']), (1, 12))
        self.assertIn('masked_rates', result)
        self.assertEqual(result['flips']['fixed'], 4)                       # code: 2 tasks flipped fail -> pass over their repeats
        stored = json.loads((self.tmp / 'final_eval' / 'result.json').read_text())
        self.assertEqual(stored['macro']['delta'], result['macro']['delta'])
        self.assertIn('| **macro** |', (self.tmp / 'final_eval' / 'README.md').read_text())

    def test_stopped_before_second_eval_records_null_never_fabricates(self):
        result = inloop.in_loop_result(self.cfg, self.env, self.samples([0]), self.tmp / 'final_eval', optimizer_steps=3)
        self.assertIsNone(result['paired'])
        self.assertIn('second evaluation', result['reason'])
        self.assertNotIn('domains', result)
        self.assertFalse((self.tmp / 'final_eval' / 'base.jsonl').exists())
        self.assertIn('NO PAIRED RESULT', (self.tmp / 'final_eval' / 'README.md').read_text())
        self.assertEqual(inloop.wandb_scalars(result), {})

    def test_missing_step_zero_and_no_rows(self):
        self.assertIn('no step-0', inloop.in_loop_result(self.cfg, self.env, self.samples([5, 10]), self.tmp / 'a', optimizer_steps=10)['reason'])
        self.assertIn('no evaluation rows', inloop.in_loop_result(self.cfg, self.env, self.tmp / 'missing.jsonl', self.tmp / 'b', optimizer_steps=1)['reason'])

    def test_resumed_duplicate_step_keeps_the_last_row(self):
        path = self.tmp / 'e.jsonl'
        old = records({'music': [0.0, 0.0]}, step=0)
        new = records({'music': [1.0, 1.0]}, step=0)
        write_jsonl(path, old + new + records({'music': [1.0, 1.0]}, step=4))
        by_step = inloop.eval_rows_by_step(path)
        self.assertEqual([r['task_metric'] for r in by_step[0]], [1.0, 1.0])

    def test_statistics_failure_is_recorded_not_raised(self):
        env = ns(name='x', metric='m', paired_result=lambda b, t, c: (_ for _ in ()).throw(ValueError('boom')))
        result = inloop.in_loop_result(self.cfg, env, self.samples([0, 3]), self.tmp / 'f', optimizer_steps=3)
        self.assertIsNone(result['paired'])
        self.assertIn('boom', result['reason'])

    def test_wandb_keys_and_table(self):
        result = inloop.in_loop_result(self.cfg, self.env, self.samples([0, 5]), self.tmp / 'f', optimizer_steps=5)
        keys = inloop.wandb_scalars(result)
        for domain in ('music', 'code', 'general', 'macro'):
            for k in ('base', 'trained', 'delta', 'ci_lower', 'ci_upper'):
                self.assertIn(f'test/{domain}/{k}', keys)
        self.assertEqual(keys['test/dropped'], 0.0)
        self.assertAlmostEqual(keys['test/macro/delta'], result['macro']['delta'])
        table = inloop.summary_markdown(result)
        self.assertIn('| music |', table)
        self.assertIn('selection: last_step', table)

    def test_dropped_ids_masked_rates_and_reasons(self):
        base = records(BASE, step=0, masked={'music-t0#r1'})
        trained = records(TRAINED, step=5, masked={'general-t3#r0'})
        stats = self.env.paired_result(base, trained, self.cfg)
        self.assertEqual(stats['dropped']['total'], 2)
        self.assertEqual(stats['dropped']['by_domain'], {'music': 1, 'general': 1})
        self.assertEqual(stats['masked_rates']['music'], {'base': 1 / 8, 'trained': 0.0})
        self.assertEqual(stats['masked_rates']['general']['trained'], 1 / 8)
        self.assertEqual(stats['domains']['music']['dropped'], 1)

    def test_all_masked_raises_inside_spec_and_is_recorded(self):
        everything = {r['sample_id'] for r in records(BASE)}
        with self.assertRaises(ValueError):
            self.env.paired_result(records(BASE, masked=everything), records(TRAINED, step=5), self.cfg)

    def test_default_spec_paired_result_uses_correct(self):
        from strl.tasks.base import EnvironmentSpec
        rows = lambda ok: [{'sample_id': f's{i}', 'correct': v} for i, v in enumerate(ok)]   # noqa: E731
        out = EnvironmentSpec().paired_result(rows([0, 0, 1, 0]), rows([1, 0, 1, 1, ]), self.cfg)
        self.assertEqual(out['test']['examples'], 4)
        self.assertAlmostEqual(out['test']['accuracy_delta'], 0.5)
        self.assertEqual(out['flips'], {'fixed': 2, 'broken': 0, 'both_correct': 1, 'both_wrong': 1})
        self.assertEqual(out['dropped']['total'], 0)


class WorkerInLoopFinish(unittest.TestCase):
    def test_finish_writes_files_prints_table_and_survives_wandb_absence(self):
        from strl import worker
        tmp = Path(tempfile.mkdtemp())
        cfg = load(SMOKE)
        cfg['evaluation']['bootstrap_samples'] = 200
        root = tmp / 'artifacts' / 'att'
        write_jsonl(root / 'exports' / 'evaluation_samples.jsonl', records(BASE, step=0) + records(TRAINED, step=4))
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop('WANDB_API_KEY', None)
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    out = worker.in_loop_finish(cfg, {'wandb_run_id': 'x', 'run_name': 'att'}, get_environment('mimo'), root, steps=4)
        finally:
            os.chdir(cwd)
        self.assertEqual((out['selected_step'], out['trained_step']), (4, 4))
        self.assertIn('| **macro** |', printed.getvalue())      # the one-screen table goes to stdout
        self.assertTrue((tmp / 'artifacts' / 'final_eval' / 'result.json').is_file())
        json.dumps(out)    # completion.json must be serialisable


class JudgeRequestPersistence(unittest.TestCase):
    """`strl verify` re-scores judged records offline, so the eval phase persists the rubric request (training does not: memory)."""

    def test_eval_phase_keeps_the_rubric_request_and_the_record_carries_it(self):
        from strl.tasks.mimo import judge
        from tests import test_mimo as t
        request = {**t.RUBRIC, 'base_score': 0.2}
        with tempfile.TemporaryDirectory() as temp:
            env = t.Env(temp + '/eval', domain='general', judge_obj=judge.Judge(t.fast(eval_votes=1), chat=t.FakeChat([t.verdict('a', 1), t.verdict('b', 1)])))
            env.queue = [t.fake_results(reward=0.2, trial_dir=env.trial_dir(reward_txt='0.2', request=request))]
            _, summary = env.go(phase='eval')
            self.assertEqual(summary['judge_request'], request)
            env = t.Env(temp + '/train', domain='general', judge_obj=judge.Judge(t.fast(), chat=t.FakeChat([t.verdict('a', 1), t.verdict('b', 1)])))
            env.queue = [t.fake_results(reward=0.2, trial_dir=env.trial_dir(reward_txt='0.2', request=request))]
            self.assertNotIn('judge_request', env.go(phase='train')[1])
        spec = get_environment('mimo')
        item = {'score': 1.0, 'env_extras': {'task_dir': '/x/general-g1', 'rep': 0, 'domain': 'general', 'mimo': {'artifact': 'A', 'judge_request': request, 'passed': True}}}
        self.assertEqual(spec.eval_record(item, None, 3)['extra']['judge_request'], request)
        del item['env_extras']['mimo']['judge_request']
        self.assertNotIn('judge_request', spec.eval_record(item, None, 3)['extra'])


class PlanEstimateInRuntime(unittest.TestCase):
    def test_plan_step_seconds_matches_the_plan(self):
        from strl.stage import plan_step_seconds
        self.assertEqual(plan_step_seconds(load('base/gsm8k.yaml')), 51.2)          # the measured verified run
        self.assertIsNone(plan_step_seconds({}))                                      # never raises


# ---------------------------------------------------------------------------------------------------------------- verify
class FakeSpec:
    name = 'mimo'

    def __init__(self, ids):
        self.ids = ids

    def final_eval_ids(self, cfg):
        return self.ids


class Verify(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = load(SMOKE)
        self.cfg['evaluation']['bootstrap_samples'] = 1500
        self.cfg['run']['data_dir'] = str(self.tmp / 'data')
        self.env = get_environment('mimo')
        self.base, self.trained = records(BASE, step=0), records(TRAINED, step=9)
        self.ids = [r['sample_id'] for r in self.base]

    def publish(self, base=None, trained=None):
        base, trained = base or self.base, trained or self.trained
        stats = self.env.paired_result(base, trained, self.cfg)
        result = {'mode': 'in_loop', 'selected_step': 9, 'selection': 'last_step', 'base_step': 0, 'trained_step': 9, 'optimizer_steps': 9,
                  'paired': {'base_step': 0, 'trained_step': 9}, **stats}
        write_jsonl(self.tmp / 'local' / 'base.jsonl', base)
        write_jsonl(self.tmp / 'local' / 'candidate.jsonl', trained)
        (self.tmp / 'local' / 'result.json').write_text(json.dumps(result))
        return result

    def test_independent_statistics_agree_with_the_worker(self):
        self.publish()
        out = results.verify_attempt(self.cfg, 'a1', self.tmp / 'local', FakeSpec(self.ids), regrade=False)
        self.assertEqual(out['stat_disagreements'], [])
        self.assertTrue(out['stats_agree'] and out['coverage']['complete'] and out['verified'])
        self.assertAlmostEqual(out['independent']['domains']['music']['delta'], out['worker']['domains']['music']['delta'])
        self.assertTrue((self.tmp / 'local' / 'verified-result.json').is_file())

    def test_cli_verify_dispatches_in_loop_results_and_writes_the_evidence_file(self):
        from strl import verify_results
        self.publish()
        with patch.object(verify_results, '_download', return_value=self.tmp / 'local'), patch.object(verify_results, 'get_environment', return_value=FakeSpec(self.ids)), \
                patch.object(results, 'regrade_music', return_value={'skipped': 'test'}), patch.object(results, 'regrade_judge', return_value={'skipped': 'test'}):
            out = verify_results.verify(self.cfg, 'a1')
        self.assertTrue(out['verified'])
        self.assertEqual(out['regrade']['music'], {'skipped': 'test'})
        self.assertEqual(json.loads((self.tmp / 'local' / 'verified-result.json').read_text())['attempt'], 'a1')

    def test_generic_in_loop_verify_for_single_metric_environments(self):
        from strl import verify_results
        from strl.tasks.base import EnvironmentSpec
        cfg = load('experiments/gsm8k-qwen35.yaml')
        cfg['evaluation'].update(final_eval=False, bootstrap_samples=200)
        base = [{'sample_id': f's{i}', 'correct': i % 2 == 0} for i in range(10)]
        trained = [{'sample_id': f's{i}', 'correct': i % 3 != 0} for i in range(10)]
        local = self.tmp / 'gsm'
        write_jsonl(local / 'base.jsonl', base)
        write_jsonl(local / 'candidate.jsonl', trained)
        stats = EnvironmentSpec().paired_result(base, trained, cfg)
        (local / 'result.json').write_text(json.dumps({'paired': {'x': 1}, **stats}))
        out = verify_results.verify_in_loop(cfg, 'g1', local, ns(name='gsm8k', paired_result=EnvironmentSpec().paired_result))
        self.assertTrue(out['verified'])
        stats['test']['accuracy_delta'] += 0.05
        (local / 'result.json').write_text(json.dumps({'paired': {'x': 1}, **stats}))
        self.assertFalse(verify_results.verify_in_loop(cfg, 'g1', local, ns(name='gsm8k', paired_result=EnvironmentSpec().paired_result))['verified'])

    def test_independent_bootstrap_is_a_separate_implementation(self):
        stats = results.independent_stats(self.base, self.trained, samples=1500)
        worker = self.env.paired_result(self.base, self.trained, self.cfg)
        self.assertEqual(stats['dropped']['total'], worker['dropped']['total'])
        for name in worker['domains']:
            for key in ('baseline', 'candidate', 'delta', 'tasks', 'trials'):
                self.assertAlmostEqual(stats['domains'][name][key], worker['domains'][name][key])
            self.assertLess(abs(stats['domains'][name]['ci_lower'] - worker['domains'][name]['ci_lower']), 0.1)

    def test_a_tampered_worker_number_is_caught(self):
        result = self.publish()
        result['domains']['music']['delta'] += 0.1
        result['macro']['ci_lower'] -= 0.5
        (self.tmp / 'local' / 'result.json').write_text(json.dumps(result))
        out = results.verify_attempt(self.cfg, 'a1', self.tmp / 'local', FakeSpec(self.ids), regrade=False)
        self.assertFalse(out['verified'])
        self.assertTrue(any('domain music.delta' in d for d in out['stat_disagreements']))
        self.assertTrue(any('macro.ci_lower' in d for d in out['stat_disagreements']))

    def test_coverage_must_match_task_x_rep_ids(self):
        self.publish()
        out = results.verify_attempt(self.cfg, 'a1', self.tmp / 'local', FakeSpec(self.ids + ['music-t9#r0']), regrade=False)
        self.assertFalse(out['verified'])
        self.assertEqual(out['coverage']['base']['missing'], ['music-t9#r0'])
        self.assertFalse(out['coverage']['complete'])
        cov = results.coverage(self.ids, self.base + self.base[:1], self.trained)
        self.assertEqual(cov['base']['n_duplicated'], 1)

    def test_null_pair_is_not_verified(self):
        (self.tmp / 'local').mkdir(parents=True)
        (self.tmp / 'local' / 'result.json').write_text(json.dumps({'paired': None, 'reason': 'training stopped before a second evaluation'}))
        out = results.verify_attempt(self.cfg, 'a1', self.tmp / 'local', FakeSpec([]))
        self.assertFalse(out['verified'])
        self.assertIn('second evaluation', out['reasons'][0])

    def test_music_regrade_uses_the_saved_abc_and_flags_disagreements(self):
        extra = {'artifact_full': 'X:1\nK:C\nCDEF|'}
        rows = records({'music': [0.2, 0.2, 0.4, 0.4]}, extra=extra)
        seen = []

        def grade(text, constraints=None):
            seen.append((text, constraints))
            return {'reward': 0.2}
        out = results.regrade_music(self.cfg, rows, grade=grade)
        self.assertEqual((out['checked'], out['disagreements']), (4, 2))
        self.assertEqual(seen[0][0], 'X:1\nK:C\nCDEF|')
        self.assertEqual(out['without_constraints_json'], 4)
        constraints = self.tmp / 'data' / 'tasks' / 'test' / 'music-t0' / 'tests'
        constraints.mkdir(parents=True)
        (constraints / 'constraints.json').write_text('{"nvoice_want": 3}')
        results.regrade_music(self.cfg, rows[:1], grade=grade)
        self.assertEqual(seen[-1][1], {'nvoice_want': 3})

    def test_music_regrade_skips_when_scorer_unavailable(self):
        rows = records({'music': [0.2, 0.2]}, extra={'artifact_full': 'X:1'})
        self.assertIn('skipped', results.regrade_music(self.cfg, [], grade=lambda *a, **k: {'reward': 0}))
        with patch.dict(sys.modules, {'strl.tasks.mimo.graders.music': None}):
            self.assertIn('unavailable', results.regrade_music(self.cfg, rows)['skipped'])

        class Missing(Exception):
            pass
        Missing.__name__ = 'Abc2MidiMissing'

        def boom(*a, **k):
            raise Missing('no abc2midi')
        self.assertIn('abc2midi', results.regrade_music(self.cfg, rows, grade=boom)['skipped'])

    def test_judge_regrade_variance_and_skips(self):
        request = {'kind': 'rubric', 'items': []}
        rows = records({'general': [0.5, 0.5, 0.6, 0.6]}, extra={'judge_request': request})

        class FakeJudge:
            async def rescore_variance(self, req, *, n=2, workdir=None):
                return {'finals': [0.5, 0.7], 'variance': 0.01, 'max_abs_diff': 0.2, 'cost_usd': 0.001}
        out = results.regrade_judge(self.cfg, rows, n=2, judge=FakeJudge())
        self.assertEqual(out['checked'], 2)
        self.assertAlmostEqual(out['mean_variance'], 0.01)
        self.assertIn('no general record', results.regrade_judge(self.cfg, records({'general': [0.5]}), judge=FakeJudge())['skipped'])
        from strl.services import litellm
        with patch.object(litellm, 'resolve_key', side_effect=litellm.LiteLLMError('no key')):
            self.assertIn('no LiteLLM key', results.regrade_judge(self.cfg, rows)['skipped'])


# ---------------------------------------------------------------------------------------------------------------- compare / report
class CompareReport(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = load(SMOKE)
        for attempt, trained in (('a1', TRAINED), ('a2', {'music': [0.9] * 8, 'code': [1.0] * 8, 'general': [0.5] * 8})):
            local = self.tmp / attempt
            write_jsonl(local / 'base.jsonl', records(BASE))
            write_jsonl(local / 'candidate.jsonl', records(trained, step=5))
            (local / 'result.json').write_text(json.dumps({'optimizer_steps': 5, 'selected_step': 5, 'selection': 'last_step'}))
            steps = []
            for s in range(1, 6):
                steps.append({'optimizer_step': s, 'generate/music/pass_rate': 0.1 * s, 'generate/code/pass_rate': 0.2, 'generate/music/reward_mean': 0.3,
                              'generate/music/tokens': 100.0, 'generate/music/num_trials': 8.0, 'generate/code/tokens': 300.0, 'generate/code/num_trials': 8.0,
                              'environment/music/masked_group_fraction': 0.1 * (s % 2), 'timing/step': 100.0 + s, 'timing/generate': 60.0,
                              'train/learning_signal_fraction': 0.5, 'generate/general/judge_cost_usd': 0.25, 'generate/general/pass_rate': 0.0,
                              'gpu/mem_peak_gb': 70.0})
            steps.append({'optimizer_step': 5, 'critic/no_patch_frac': 0.6, 'critic/cost_usd': 1.2})
            write_jsonl(local / 'metrics.jsonl', steps)
            (local / 'network.json').write_text(json.dumps({'transport': 'efa', 'efa_available': True, 'nodes': 2, 'bandwidth_gbps': {'cross_node': 41.0}}))
        self.patches = [patch.object(compare, 'load', return_value=self.cfg), patch.object(report, 'load', return_value=self.cfg),
                        patch.object(compare, '_download', side_effect=lambda cfg, attempt, dest: Path(dest))]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])

    def test_compare_tables_per_domain_macro_and_run_vs_run(self):
        out = compare.compare(['x.yaml:a1', 'x.yaml:a2'], workdir=self.tmp, samples=300)
        self.assertEqual(set(out['runs'][0]['domains']), {'music', 'code', 'general'})
        self.assertIn('macro', out['runs'][0])
        pair = out['pairs'][0]
        self.assertEqual((pair['a'], pair['b']), ('a1', 'a2'))
        self.assertGreater(pair['domains']['music']['delta'], 0.3)       # a2 music 0.9 vs a1 0.55
        text = compare.markdown(out)
        self.assertIn('| music |', text)
        self.assertIn('**macro**', text)
        self.assertIn('a2 vs a1', text)
        self.assertIn('n tasks', text)
        self.assertIn('trained = step 5', text)

    def test_existing_environments_keep_their_tables(self):
        gsm = {'runs': [{'attempt': 'a', 'model': 'm', 'reward': 'r', 'optimizer_steps': 3, 'test': {'baseline_accuracy': .5, 'candidate_accuracy': .6,
                                                                                                   'accuracy_delta': .1, 'ci_lower': 0.0, 'ci_upper': .2}}], 'pairs': []}
        self.assertEqual(compare.domain_tables(gsm), [])
        self.assertIn('+10.00 [+0.00, +20.00]', compare.markdown(gsm))

    def test_report_file_sections_and_wandb_optional(self):
        written = report.write_reports(['x.yaml:a1', 'x.yaml:a2'], out_dir=self.tmp / 'results', samples=300, wandb=False, workdir=self.tmp)
        self.assertEqual(len(written), 1)
        path = Path(written[0]['path'])
        self.assertEqual(path.name, 'mimo-smoke-report.md')
        text = path.read_text()
        for needle in ('results report', 'Training pass rate by step', 'learning signal', 'masked groups mean', 'tokens share', 'step-time phase',
                       'Infrastructure: transport=efa', 'peak GPU memory (head node) 70.0 GB', 'judge cost', 'critic cost 1.200 USD', 'no_patch 60%',
                       'a2 vs a1'):
            self.assertIn(needle, text, needle)
        self.assertEqual(written[0]['wandb'], {'wandb': 'disabled'})

    def test_report_wandb_failure_never_breaks(self):
        with patch.dict(sys.modules, {'wandb': None}):
            written = report.write_reports(['x.yaml:a1'], out_dir=self.tmp / 'r2', samples=200, wandb=True, workdir=self.tmp)
        self.assertTrue(written[0]['wandb']['wandb'].startswith('skipped'))
        self.assertTrue(Path(written[0]['path']).is_file())

    def test_report_logs_table_and_artifact_to_a_report_run(self):
        calls = {}

        class Run:
            url = 'http://w/run'
            summary = {}

            def log(self, data):
                calls['log'] = data

            def log_artifact(self, artifact):
                calls['artifact'] = artifact

            def finish(self):
                calls['finished'] = True
        fake = types.ModuleType('wandb')
        fake.init = lambda **kw: calls.setdefault('init', kw) and Run()
        fake.Table = lambda columns, data: ('table', columns, len(data))
        fake.Artifact = lambda name, type: ns(name=name, type=type, add_file=lambda p: calls.setdefault('file', p))
        service = types.ModuleType('strl.services.wandb')
        service.local_key = lambda: None
        with patch.dict(sys.modules, {'wandb': fake, 'strl.services.wandb': service}):
            with patch.dict(sys.modules, {'wandb_workspaces': None, 'wandb_workspaces.reports': None, 'wandb_workspaces.reports.v2': None}):
                written = report.write_reports(['x.yaml:a1'], out_dir=self.tmp / 'r3', samples=200, wandb=True, workdir=self.tmp)
        self.assertEqual(written[0]['wandb']['wandb'], 'logged')
        self.assertEqual(calls['init']['group'], 'mimo-smoke')
        self.assertEqual(calls['init']['job_type'], 'report')
        self.assertEqual(calls['log']['report/headline'][2], 4)       # 3 domains + macro
        self.assertTrue(calls['file'].endswith('mimo-smoke-report.md') and calls['finished'])
        self.assertEqual(written[0]['wandb']['wandb_report'], 'wandb_workspaces not installed')


# ---------------------------------------------------------------------------------------------------------------- live monitor
class LiveDomainFlags(unittest.TestCase):
    def test_domain_masked_step_slow_mem(self):
        rows = [{'timing/step': 400.0, 'environment/music/masked_group_fraction': 0.4, 'environment/code/masked_group_fraction': 0.05,
                 'gpu/mem_peak_gb': 77.5, 'ray/gpu.mem_used_gb.avg': 50.0}]
        out = live.flags(rows, {'updated_at': time.time(), 'budget_consumed_node_hours': 0.1}, None, 10.0, plan_step_seconds=200.0)
        self.assertTrue(any(f.startswith('DOMAIN_MASKED(music=0.40') for f in out))
        self.assertFalse(any('code' in f for f in out))
        self.assertTrue(any(f.startswith('STEP_SLOW(400s > 1.5 x plan 200s') for f in out))
        self.assertTrue(any(f.startswith('MEM(gpu/mem_peak_gb=77.5GB') for f in out))
        quiet = live.flags([{'timing/step': 250.0, 'environment/music/masked_group_fraction': 0.15, 'gpu/mem_peak_gb': 76.0}],
                           {'updated_at': time.time(), 'budget_consumed_node_hours': 0.1}, None, 10.0, plan_step_seconds=200.0)
        self.assertEqual(quiet, [])                                 # thresholds are strict: 0.15, 76 GB and 1.5 x are not exceeded
        self.assertFalse(any(f.startswith('STEP_SLOW') for f in live.flags(rows, {'updated_at': time.time()}, None, 10.0)))   # no plan estimate: silent

    def test_timeouts_are_flagged_not_called_environment_failures(self):
        rows = [{'environment/music/timeout_group_fraction': 1.0, 'environment/music/masked_group_fraction': 0.0}]
        out = live.flags(rows, None, None, 10.0)
        self.assertTrue(any(f.startswith('TIMEOUTS(music=1.00') for f in out))
        self.assertFalse(any(f.startswith('DOMAIN_MASKED') for f in out))

    def test_per_domain_values_come_from_different_rows(self):
        rows = [{'eval/music/task_metric': 0.42}, {'environment/music/masked_group_fraction': 0.3}, {'timing/step': 10.0}]
        self.assertTrue(any(f.startswith('DOMAIN_MASKED') for f in live.flags(rows, None, None, 10.0)))

    def test_status_line_shows_domains(self):
        latest = {'optimizer_step': 3, 'eval/music/task_metric': 0.42, 'generate/music/pass_rate': 0.24, 'eval/cyber/task_metric': 0.1}
        text = live.line({'at': '00:00:01', 'job': 'RUNNING', 'completion': None, 'heartbeat_age_s': 5, 'latest': latest, 'flags': ['MEM(x)'], 'wandb_run_id': 'r'})
        self.assertIn('cyber 0.10/-% music 0.42/24%', text)
        self.assertIn('MEM(x)', text)

    def test_each_flag_alerts_once_per_run(self):
        sent, calls = set(), []
        notify = lambda cfg, run_id, name, text: calls.append((run_id, name))   # noqa: E731
        first = live.alert_once({}, 'run1', ['DOMAIN_MASKED(music=0.40)', 'STEP_SLOW(400s)', 'STALL(5s)'], sent, notify)
        again = live.alert_once({}, 'run1', ['DOMAIN_MASKED(music=0.50)', 'MEM(x=80GB)'], sent, notify)
        self.assertEqual([c[1] for c in calls], ['DOMAIN_MASKED', 'STEP_SLOW', 'MEM'])
        self.assertEqual((len(first), len(again)), (2, 1))

    def test_notify_goes_through_tracking_alert(self):
        calls = []
        fake_run = ns(finish=lambda: calls.append('finish'))
        fake = types.ModuleType('wandb')
        fake.init = lambda **kw: fake_run
        service = types.ModuleType('strl.services.wandb')
        service.local_key = lambda: None
        cfg = {'name': 'e', 'tracking': {'project': 'p', 'entity': 'e', 'alerts': True}}
        with patch.dict(sys.modules, {'wandb': fake, 'strl.services.wandb': service}):
            with patch('strl.tracking.alert', side_effect=lambda *a, **k: calls.append(a[0])):
                live.notify_wandb(cfg, 'run1', 'MEM', 'MEM(x)')
                live.notify_wandb(dict(cfg, tracking={**cfg['tracking'], 'alerts': False}), 'run1', 'MEM', 'MEM(x)')
        self.assertEqual(calls, ['strl monitor: MEM', 'finish'])


# ---------------------------------------------------------------------------------------------------------------- fail-fast + predictive deadline
class FailFastBase(unittest.TestCase):
    def setUp(self):
        callbacks = types.ModuleType('skyrl.train.utils.callbacks')
        callbacks.TrainingCallback = object
        for name in ('skyrl', 'skyrl.train', 'skyrl.train.utils'):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules['skyrl.train.utils.callbacks'] = callbacks
        self.tmp = tempfile.mkdtemp()
        self.env = patch.dict(os.environ, {'GRPO_TRAIN_DEADLINE': str(time.time() + 36000), 'STRL_ALERTS': '0', 'GRPO_CANARY_REQUIRE_GRADIENT': '0'})
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop('STRL_FAILFAST', None)
        os.environ.pop('STRL_STAGE', None)
        os.environ.pop('STRL_PLAN_STEP_SECONDS', None)
        from strl import tracking
        self.tracking = tracking
        self.callback = tracking.make_callback_class()()
        self.now = [1000.0]
        self.callback.clock = lambda: self.now[0]
        self.alerts = []
        patcher = patch.object(tracking, 'alert', side_effect=lambda *a, **k: self.alerts.append(a[0]))
        patcher.start()
        self.addCleanup(patcher.stop)

    def trainer(self, all_metrics=None):
        cfg = ns(trainer=ns(train_batch_size=8, export_path=f'{self.tmp}/exports', max_training_steps=100))
        return ns(cfg=cfg, all_metrics=all_metrics or {})

    def step(self, step, all_metrics=None, seconds=None, metrics=None):
        """One optimizer step: on_step_start, ``seconds`` of fake time, on_step_end."""
        control = ns(should_save=False, should_evaluate=False)
        trainer = self.trainer(all_metrics)
        self.callback.on_step_start(trainer, ns(global_step=step), control)
        if seconds is not None:
            self.now[0] += seconds
        self.callback.on_step_end(trainer, ns(global_step=step, metrics={'policy/grad_norm': 1.0, **(metrics or {})}), control)
        return control, trainer


class FailFastRules(FailFastBase):
    def rules(self, **rules):
        self.callback.failfast = rules

    def test_no_rules_nothing_fires_whatever_the_numbers(self):
        self.callback.plan_step_seconds = 1.0
        for s in range(1, 6):
            self.step(s, {'environment/music/masked_group_fraction': 1.0, 'ray/gpu.mem_used_gb.avg': 500.0}, seconds=1e6)
        self.assertEqual(self.alerts, [])

    def test_defaults_render_nothing_unless_the_environment_opts_in(self):
        cfg = load(SMOKE)
        self.assertEqual(self.tracking.failfast_config(load('experiments/gsm8k-qwen35.yaml')), {})
        self.assertEqual(self.tracking.failfast_config(cfg), {'mem_gb': 76, 'domain_masked': 0.15, 'step_slow_factor': 1.5, 'min_steps': 1})
        cfg['budget']['failfast'] = {'mem_gb': None, 'domain_masked': 0.3}
        self.assertEqual(self.tracking.failfast_config(cfg), {'domain_masked': 0.3, 'step_slow_factor': 1.5, 'min_steps': 1})
        gsm = load('experiments/gsm8k-qwen35.yaml')
        gsm['budget']['failfast'] = {'step_slow_factor': 2}
        self.assertEqual(self.tracking.failfast_config(gsm), {'step_slow_factor': 2})
        self.assertEqual(load('experiments/gsm8k-qwen35.yaml')['budget'].get('failfast'), None)     # nothing rendered into existing experiments

    def test_schema_rejects_unknown_or_negative_rules(self):
        with self.assertRaises(ExperimentError):
            load(SMOKE, overrides=['budget.failfast.nonsense=1'])
        with self.assertRaisesRegex(ExperimentError, 'non-negative'):
            load(SMOKE, overrides=['budget.failfast.mem_gb=-3'])

    def test_memory_at_step_one(self):
        self.rules(mem_gb=76, min_steps=1)
        sampler = self.tracking.MemorySampler(query=lambda: 79.5)
        sampler.sample()
        self.callback.memory = sampler
        with self.assertRaisesRegex(RuntimeError, r'Fail-fast \(memory\).*79\.5 GB'):
            self.step(1)
        self.assertEqual(self.alerts, ['Fail-fast: memory'])
        self.assertEqual(json.loads((Path(self.tmp) / 'failfast.json').read_text())['rule'], 'memory')

    def test_memory_below_limit_unmeasurable_and_only_step_one(self):
        self.rules(mem_gb=76)
        ok = self.tracking.MemorySampler(query=lambda: 60.0)
        ok.sample()
        self.callback.memory = ok
        self.step(1)
        ok.peak = 90.0                                              # a later spike is the stall/NaN guards' business, not the step-1 check
        self.step(2)
        self.callback._mem_checked = False
        broken = self.tracking.MemorySampler(query=lambda: (_ for _ in ()).throw(OSError('no nvidia-smi')))
        self.assertIsNone(broken.sample())
        self.assertIsNone(broken.start().peak)
        self.callback.memory = broken
        self.step(3)                                                # not measurable: never fires
        self.assertEqual(self.alerts, [])

    def test_memory_from_ray_metrics_and_peak_helper(self):
        self.assertEqual(self.tracking.memory_gb_from_metrics({'ray/gpu.mem_used_gb.avg': 70.0, 'gpu/mem_peak_gb': 72.0, 'x/mem_gb': 99.0}), 72.0)
        self.assertIsNone(self.tracking.memory_gb_from_metrics({'reward/avg_raw_reward': 1.0}))
        self.rules(mem_gb=76)
        with self.assertRaisesRegex(RuntimeError, 'memory'):
            self.step(1, {'ray/gpu.mem_used_gb.avg': 78.0})

    def test_domain_masked_needs_two_consecutive_steps(self):
        self.rules(domain_masked=0.15)
        self.step(1, {'environment/music/masked_group_fraction': 0.2, 'environment/code/masked_group_fraction': 0.0})
        with self.assertRaisesRegex(RuntimeError, r'Fail-fast \(domain_masked\).*music'):
            self.step(2, {'environment/music/masked_group_fraction': 0.25})
        self.assertEqual(self.alerts, ['Fail-fast: domain_masked'])

    def test_domain_masked_interrupted_or_at_threshold_does_not_fire(self):
        self.rules(domain_masked=0.15)
        for s, f in enumerate([0.2, 0.0, 0.2, 0.15, 0.15, 0.1], start=1):
            self.step(s, {'environment/music/masked_group_fraction': f})
        self.step(7, {'environment/music/masked_group_fraction': 0.5})
        self.step(8, {})                                            # the domain vanished: counter resets
        self.step(9, {'environment/music/masked_group_fraction': 0.5})
        self.assertEqual(self.alerts, [])

    def test_step_slow_two_consecutive_steps_vs_plan(self):
        self.rules(step_slow_factor=1.5)
        self.callback.plan_step_seconds = 100.0
        self.step(1, seconds=160.0)
        with self.assertRaisesRegex(RuntimeError, r'Fail-fast \(step_slow\).*150|Fail-fast \(step_slow\).*x the plan'):
            self.step(2, seconds=170.0)

    def test_uncalibrated_estimate_and_last_step_only_alert(self):
        # v3smk2-fb: an extrapolated 39 s estimate stopped a healthy 2 x 8 run after its last step
        self.rules(step_slow_factor=1.5, domain_masked=0.15)
        self.callback.plan_step_seconds = 39.0
        self.callback.plan_calibrated = False
        for s in (1, 2, 3):
            self.step(s, seconds=250.0)                             # never raises
        self.assertEqual(len([a for a in self.alerts if 'advisory' in str(a)]), 1)   # one advisory alert, not one per step
        self.callback.plan_calibrated = True
        self.callback._slow_steps = []
        self.step(99, seconds=250.0)
        self.step(100, seconds=250.0)                               # max_training_steps=100: the last step only alerts
        self.callback._slow_steps = []
        self.step(10, seconds=250.0)
        with self.assertRaises(RuntimeError):
            self.step(11, seconds=250.0)                            # calibrated and not the last step: stops

    def test_step_slow_not_consecutive_fast_unknown_or_no_plan(self):
        self.rules(step_slow_factor=1.5)
        self.callback.plan_step_seconds = 100.0
        for s, seconds in enumerate([160.0, 100.0, 160.0, 150.0, 150.0], start=1):   # 150 s is exactly 1.5 x: not slower
            self.step(s, seconds=seconds)
        self.callback.plan_step_seconds = None
        self.step(6, seconds=1e6)
        self.step(7, seconds=1e6)
        self.callback.plan_step_seconds = 100.0
        self.step(8)                                                # no measured duration (no on_step_start time): never fires
        self.assertEqual(self.alerts, [])

    def test_canary_runs_skip_masking_and_plan_rules(self):
        self.rules(domain_masked=0.15, step_slow_factor=1.5)
        self.callback.plan_step_seconds = 1.0
        with patch.dict(os.environ, {'STRL_STAGE': 'canary'}):
            self.step(1, {'environment/music/masked_group_fraction': 0.2}, seconds=500)
            self.callback.failfast = {'domain_masked': 0.15, 'step_slow_factor': 1.5}
            self.step(2, {'environment/music/masked_group_fraction': 0.2}, seconds=500)
        self.assertEqual(self.alerts, [])

    def test_min_steps_delays_every_rule(self):
        self.rules(domain_masked=0.15, min_steps=3)
        self.step(1, {'environment/music/masked_group_fraction': 0.9})
        self.step(2, {'environment/music/masked_group_fraction': 0.9})
        with self.assertRaises(RuntimeError):
            self.step(3, {'environment/music/masked_group_fraction': 0.9})
            self.step(4, {'environment/music/masked_group_fraction': 0.9})

    def test_sampler_thread_tracks_the_peak(self):
        values = iter([10.0, 50.0, 30.0] + [20.0] * 50)
        sampler = self.tracking.MemorySampler(interval=0.01, query=lambda: next(values))
        sampler.start()
        deadline = time.time() + 2
        while time.time() < deadline and (sampler.peak or 0) < 50.0:
            time.sleep(0.01)
        sampler.stop()
        self.assertEqual(sampler.peak, 50.0)

    def test_worker_environment_roundtrip(self):
        with patch.dict(os.environ, {'STRL_FAILFAST': json.dumps({'domain_masked': 0.1}), 'STRL_PLAN_STEP_SECONDS': '42.5'}):
            callback = self.tracking.make_callback_class()()
        self.assertEqual((callback.failfast, callback.plan_step_seconds, callback.memory), ({'domain_masked': 0.1}, 42.5, None))


class PredictiveDeadline(FailFastBase):
    def deadline(self, seconds_from_now):
        os.environ['GRPO_TRAIN_DEADLINE'] = str(time.time() + seconds_from_now)

    def write_eval_timing(self, seconds):
        path = Path(self.tmp) / 'evaluation_timing.jsonl'
        path.write_text(json.dumps({'optimizer_step': 0, 'examples': 4, 'eval_generate_seconds': 1.0, 'eval_seconds': seconds}) + '\n')

    def test_pure_rule(self):
        self.assertTrue(self.tracking.predictive_stop(1000.0, [100.0, 100.0], 300.0, 1535.0)[0])       # 1000 + 115 + 300 + 120 = 1535
        self.assertFalse(self.tracking.predictive_stop(1000.0, [100.0, 100.0], 300.0, 1535.1)[0])
        self.assertTrue(self.tracking.predictive_stop(1000.0, [10.0, 300.0, 100.0], 0.0, 1350.0)[0])    # only the last two steps count: 1.15 x 200 + 120
        self.assertFalse(self.tracking.predictive_stop(1000.0, [10.0, 300.0, 100.0], 0.0, 1350.1)[0])
        self.assertEqual(self.tracking.predictive_stop(1000.0, [], 300.0, 1001.0), (False, None))       # first step: plain deadline
        self.assertEqual(self.tracking.predictive_stop(1000.0, [900.0], 300.0, 1001.0), (False, None))  # one cold-start time is no estimate
        self.assertEqual(self.tracking.predictive_stop(1002.0, [], 300.0, 1001.0), (True, None))

    def test_stops_early_forces_one_eval_and_save_and_records_reason(self):
        self.deadline(500.0)
        self.write_eval_timing(300.0)
        control, _ = self.step(1, seconds=100.0)             # one (cold-start) step time is no estimate: plain deadline, 500 s left
        self.assertFalse(control.should_evaluate)
        control, trainer = self.step(2, seconds=100.0)       # now + 1.15 x 100 + 300 + 120 = 535 s > 500 s: this is the last step
        self.assertTrue(control.should_save and control.should_evaluate)
        self.assertEqual((trainer.cfg.trainer.max_training_steps, self.callback.stop_reason), (2, 'predictive_deadline'))
        control, _ = self.step(3, seconds=100.0)             # the stop is forced exactly once
        self.assertFalse(control.should_evaluate or control.should_save)

    def test_keeps_training_while_one_more_step_and_the_eval_fit(self):
        self.deadline(5000.0)
        self.write_eval_timing(300.0)
        self.step(1, seconds=100.0)
        control, trainer = self.step(2, seconds=100.0)
        self.assertFalse(control.should_evaluate or control.should_save)
        self.assertEqual((trainer.cfg.trainer.max_training_steps, self.callback.stop_reason), (100, None))

    def test_first_step_keeps_the_plain_deadline_behaviour(self):
        self.deadline(10.0)
        self.write_eval_timing(5000.0)                      # no step time known yet: the huge eval estimate must not stop step 1
        control, _ = self.step(1)
        self.assertFalse(control.should_evaluate)
        self.deadline(-1.0)
        control, trainer = self.step(2)                     # past the deadline: the old rule
        self.assertTrue(control.should_evaluate and control.should_save)
        self.assertEqual((trainer.cfg.trainer.max_training_steps, self.callback.stop_reason), (2, 'deadline'))
        self.deadline(-1.0)
        self.callback.stop_reason, self.callback.step_times = None, []
        control, _ = self.step(3)                           # unknown step times and past the deadline: still the old rule
        self.assertTrue(control.should_evaluate)

    def test_uses_the_last_measured_eval_and_measures_it_itself(self):
        self.deadline(500.0)
        self.write_eval_timing(10.0)                        # the step-0 eval was short...
        self.callback.on_eval_start(self.trainer(), ns(global_step=2), ns())
        self.now[0] += 400.0                                # ...but the last measured eval took 400 s
        self.callback.eval_seconds = 400.0
        self.step(1, seconds=10.0)
        control, _ = self.step(2, seconds=10.0)             # 1.15 x 10 + 400 + 120 = 531.5 s > 500 s
        self.assertTrue(control.should_evaluate)
        self.assertEqual(self.callback.eval_estimate(self.trainer()), 400.0)
        self.callback.eval_seconds, self.callback.stop_reason = None, None
        self.assertEqual(self.callback.eval_estimate(self.trainer()), 10.0)       # no measurement yet: the step-0 eval from the timing file
        control, _ = self.step(3, seconds=10.0)             # 11.5 + 10 + 120 < 500 s: keeps training
        self.assertFalse(control.should_evaluate)

    def test_eval_estimate_falls_back_to_generation_seconds_and_zero(self):
        trainer = self.trainer()
        self.assertEqual(self.callback.eval_estimate(trainer), 0.0)
        (Path(self.tmp) / 'evaluation_timing.jsonl').write_text(json.dumps({'eval_generate_seconds': 42.0}) + '\n')
        self.assertEqual(self.callback.eval_estimate(trainer), 42.0)

    def test_uses_mean_of_last_two_steps(self):
        self.deadline(100000.0)
        for s, seconds in enumerate([5000.0, 100.0, 100.0], start=1):
            self.step(s, seconds=seconds)
        self.assertEqual(self.callback.step_times[-2:], [100.0, 100.0])
        self.deadline(1.15 * 100 + 120 - 1)                 # remaining < 1.15 x mean(100, 100) + 0 eval + 120 s export
        control, _ = self.step(4, seconds=100.0)
        self.assertTrue(control.should_evaluate)


if __name__ == '__main__':
    unittest.main()


class DeadlineConsistency(unittest.TestCase):
    """Review finding 5: the soft stop (GRPO_TRAIN_DEADLINE) was deadline-120 while the child was hard-killed at deadline-480."""

    def test_in_loop_final_eval_leaves_room_before_the_child_is_killed(self):
        from strl import worker
        cfg = {'evaluation': {'final_eval': False}, 'budget': {'final_eval_reserve_seconds': 900}}
        self.assertEqual(worker.train_reserve_seconds(cfg, 'train'), worker.CHILD_HARD_STOP_SECONDS + 120)
        self.assertGreater(worker.train_reserve_seconds(cfg, 'train'), worker.CHILD_HARD_STOP_SECONDS)
        self.assertEqual(worker.train_reserve_seconds(dict(cfg, evaluation={'final_eval': True}), 'train'), 900)   # offline final eval: unchanged
        self.assertEqual(worker.train_reserve_seconds(cfg, 'canary'), 120)

    def test_predictive_stop_reserves_the_interval_eval_that_still_runs(self):
        from strl.tracking import predictive_stop
        # 1000 s left; steps take 300 s (x1.15 = 345), eval 200, export 120: continuing needs 665 < 1000 -> go on
        self.assertFalse(predictive_stop(0, [300, 300], 200, 1000)[0])
        # at t=100 with a 900 s deadline: continuing projects 100 + 345 + 200 + 120 = 765 (fits) ... but on an eval step this step's own interval
        # eval (200 s) still has to run first: 965 > 900, so the run must stop now. Without the reserve it would overshoot by one eval.
        self.assertFalse(predictive_stop(100, [300, 300], 200, 900, pending_eval_seconds=0)[0])
        self.assertTrue(predictive_stop(100, [300, 300], 200, 900, pending_eval_seconds=200)[0])
        self.assertAlmostEqual(predictive_stop(100, [300, 300], 200, 900, pending_eval_seconds=200)[1], 965.0)


class MergeFlag(unittest.TestCase):
    def test_flags_unmerged_stepwise_training(self):
        from strl.monitor.live import flags
        def rows(before, after):
            return [{'timing/step': 100.0, 'generate/num_seq_before_merge': before, 'generate/num_seq_after_merge': after}]
        self.assertTrue(any(f.startswith('MERGE(') for f in flags(rows(362, 362), None, None, 99)))      # the Qwen3.5 canary: 362 -> 362
        self.assertFalse(any(f.startswith('MERGE(') for f in flags(rows(811, 209), None, None, 99)))     # Qwen3.8 on real transcripts
        self.assertFalse(any(f.startswith('MERGE(') for f in flags(rows(12, 12), None, None, 99)))       # too few sequences to judge


class MultiNodeMonitoring(unittest.TestCase):
    """Compliance audit: no infra/* in W&B, no per-node line, NETWORK / PEER_STALE never alerted."""

    def test_infra_metrics_from_the_probe(self):
        from strl.tracking import infra_metrics
        out = infra_metrics({'nodes': 2, 'efa_available': False, 'transport': 'socket', 'passed': True,
                             'bandwidth_gbps': {'intra_node': 343.7, 'cross_node': 0.69, 'global': 2.39}})
        self.assertEqual((out['infra/nodes'], out['infra/efa_available'], out['infra/nccl_socket_transport']), (2.0, 0.0, 1.0))
        self.assertEqual(out['infra/cross_node_busbw_GBps'], 0.69)

    def test_per_node_line_and_alerted_flags(self):
        snap = {'at': '00:00:00', 'job': 'IN_PROGRESS', 'latest': {}, 'heartbeat_age_s': 12, 'peer_ages_s': [30, None],
                'flags': ['PEER_STALE(rank 2, 400s)'], 'completion': None}
        text = live.line(snap)
        self.assertIn('nodes=head:12s,rank1:30s,rank2:-s', text)
        sent, calls = set(), []
        live.alert_once({'tracking': {}}, 'run', ['PEER_STALE(rank 2, 400s)', 'NETWORK(transport=socket)'], sent, lambda *a: calls.append(a[2]))
        self.assertEqual(sorted(calls), ['NETWORK', 'PEER_STALE'])


class InLoopFlipTables(unittest.TestCase):
    def test_flips_are_joined_by_sample_id_and_skip_masked(self):
        from strl.inloop import flip_examples
        from strl.tasks.mimo.spec import MimoSpec
        base = [{'sample_id': 'a#0', 'domain': 'code', 'passed': False, 'task_metric': 0.0, 'masked': False, 'artifact': 'x'},
                {'sample_id': 'b#0', 'domain': 'music', 'passed': True, 'task_metric': 1.0, 'masked': False},
                {'sample_id': 'c#0', 'domain': 'cyber', 'passed': False, 'task_metric': 0.0, 'masked': True}]
        trained = [{'sample_id': 'c#0', 'passed': True, 'masked': False}, {'sample_id': 'b#0', 'passed': False, 'task_metric': 0.2, 'masked': False},
                   {'sample_id': 'a#0', 'passed': True, 'task_metric': 1.0, 'masked': False, 'artifact': 'y'}]
        fixed, broken = flip_examples(base, trained, MimoSpec())
        self.assertEqual([r['sample_id'] for r in fixed], ['a#0'])
        self.assertEqual([r['sample_id'] for r in broken], ['b#0'])
        self.assertEqual((fixed[0]['base_artifact'], fixed[0]['trained_artifact']), ('x', 'y'))
