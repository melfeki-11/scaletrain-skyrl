"""MiMo environment: data selection, judge, generator policy, sampler, spec, metrics/media, tracking hooks, honest-power compare.
No network, Modal, GPU, SkyRL or LiteLLM key needed: synthetic parquet data, fake exporter, fake Harbor trials, fake chat."""
import asyncio
import copy
import hashlib
import json
import struct
import sys
import tempfile
import types
import unittest
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strl.experiment import ExperimentError, load  # noqa: E402
from strl.tasks import get_environment  # noqa: E402
from strl.tasks.mimo import data, generator, judge, metrics, rubric_prompts, sampler  # noqa: E402

SMOKE = 'experiments/mimo-qwen35-0p8b-smoke.yaml'
LORA = 'experiments/mimo-qwen38-27b-lora.yaml'
FULL = 'experiments/mimo-qwen38-27b-full.yaml'


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------------------------------------------- synthetic data
def _row(source, prompt, info, **extra):
    return {'data_source': source, 'prompt': [{'role': 'user', 'content': prompt}], 'reward_model': {'style': 'rule', 'ground_truth': ''},
            'extra_info': info, **extra}


def synthetic_raw(directory: Path) -> Path:
    """Parquet files with the real schema, small: groups of known size so selection rules are checkable."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    directory.mkdir(parents=True, exist_ok=True)
    cyber = []
    for n in range(120):
        project = f'proj{n % 12}'
        text = f'There is a vuln. **Vulnerability:**\n{["AddressSanitizer", "MemorySanitizer", "UndefinedBehaviorSanitizer"][n % 3]}: x in function `f` in file `{project}/src/a.c`\n'
        cyber.append(_row('arvo', 'short', {'instance_id': f'arvo_{n}', 'instance_json': json.dumps({'instance_id': f'arvo_{n}', 'problem_statement': text,
                                                                                                        'description': text.split('\n')[1]})}))
    code = []
    for n in range(150):
        if n < 90:
            files = f'pkg{n % 9}/tests/test_x.py'
        elif n < 140:
            files = f'usercase-test-coderl/case{n}_test.cc'     # one coarse key shared by 50 tasks: must be split into sub-groups
        else:
            files = ''                                          # no test file: hash-bucket group
        patch_text = (f'diff --git a/{files} b/{files}\n' if files else '') + 'diff --git a/mimo_test_command.sh b/mimo_test_command.sh\n'
        code.append(_row('opensource-code', 'fix the bug ' * (1 + n % 7), {'instance_id': f'task-{n:04d}', 'instance_json': json.dumps(
            {'instance_id': f'task-{n:04d}', 'test_patch': patch_text, 'test_command': 'bash a && bash b' if n % 4 == 0 else 'bash a', 'cwd': '/testbed',
             'problem_statement': 'fix the bug ' * (1 + n % 7)})}))
    general = []
    for n in range(80):
        general.append(_row('mimoagent/general_agent', f'do thing {n}', {'instance_id': f's3k_{n // 2:04d}_legal_en_t2_rl_00{n % 2}', 'instance_json': json.dumps(
            {'instance_id': f's3k_{n // 2:04d}_legal_en_t2_rl_00{n % 2}', 'docker_image': 'general-agent-env-0:oss'})}))
    for n in range(10):
        general.append(_row('mimoagent/terminal_bench', f'tb {n}', {'instance_id': f'candidate-{n:04d}-x', 'instance_json': json.dumps(
            {'instance_id': f'candidate-{n:04d}-x', 'category': 'Software', 'docker_image': f'general-agent-env-{n + 1}:oss'})}))
    music = [_row('music', f'compose {n}', {'src_id': f'm-{n:04d}', 'lang': 'zh' if n % 2 else 'en', 'tag': f'tag{n % 10}', 'length': 'short' if n % 3 else 'long',
                                              'nvoice_want': '3', 'bpm': '120', 'meter': '4/4' if n % 5 else '3/4'}) for n in range(100)]
    web = [_row('blackbox/webdev', f'site {n} ' * (1 + n % 5), {'instance_id': f'dasyn_{n:05d}', 'instance_json': json.dumps(
        {'instance_id': f'dasyn_{n:05d}', 'problem_statement': f'site {n} ' * (1 + n % 5)})}) for n in range(150)]
    for domain, rows in (('cyber', cyber), ('code', code), ('general', general), ('music', music), ('webdev', web)):
        pq.write_table(pa.Table.from_pylist(rows), directory / data.PARQUET[domain])
    return directory


def make_cfg(raw: Path, data_dir: Path, train: int = 6, test: int = 6, **domains) -> dict:
    spec = {d: {'train': train, 'test': test, **domains.get(d, {})} for d in data.DOMAINS}
    return {'environment': {'name': 'mimo', 'mimo': {'revision': 'test', 'seed': 7, 'raw_dir': str(raw), 'domains': spec, 'eval_repeats': 2, 'canary': 2}},
            'run': {'data_dir': str(data_dir)}}


def fake_exporter(row, dest, *, split):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'instruction.md').write_text(f'{row["domain"]} {row["instance_id"]}')
    (dest / 'task.toml').write_text('[agent]\ntimeout_sec = 100\n')
    (dest / 'strl.json').write_text(json.dumps({'domain': row['domain'], 'instance_id': row['instance_id'], 'group': row['group'], 'artifacts': ['/app/answer.md']}))
    return {'task_dir': str(dest), 'domain': row['domain'], 'instance_id': row['instance_id'], 'group': row['group'], 'image': 'img', 'tree_sha256': 'x'}


class DataCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.raw = synthetic_raw(Path(cls.tmp.name) / 'raw')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cfg(self, name='d', train=6, test=6, **domains):
        return make_cfg(self.raw, Path(self.tmp.name) / name, train, test, **domains)


class Data(DataCase):
    def test_group_keys(self):
        cyber = data.load_pool('cyber', self.raw, 'test')
        self.assertEqual({r['group'].split('#')[0] for r in cyber}, {f'cyber:proj{i}' for i in range(12)})   # 10-row projects exceed max(8, pool/40): split by hash
        self.assertTrue(all(r['group'].startswith('music:tag') for r in data.load_pool('music', self.raw, 'test')))
        general = {r['instance_id']: r['group'] for r in data.load_pool('general', self.raw, 'test')}
        self.assertEqual(general['s3k_0003_legal_en_t2_rl_001'], 'ga:s3k_0003')
        self.assertTrue(general['candidate-0001-x'].startswith('tb:Software:general-agent-env-2'))
        music = data.load_pool('music', self.raw, 'test')
        self.assertTrue(music[0]['group'].startswith('music:tag0|3/4'))
        self.assertEqual(music[0]['instance_id'], 'm-0000')
        code = data.load_pool('code', self.raw, 'test')
        groups = {r['instance_id']: r['group'] for r in code}
        self.assertTrue(groups['task-0000'].startswith('code:pkg0'))
        self.assertTrue(groups['task-0145'].startswith('code:hash'))
        coarse = {g for g in groups.values() if g.startswith('code:usercase-test-coderl')}
        self.assertGreater(len(coarse), 3)    # the 50-row coarse group was split by hash
        self.assertTrue(all(g.startswith('webdev:b') for g in (r['group'] for r in data.load_pool('webdev', self.raw, 'test'))))

    def test_test_set_is_independent_of_training_settings(self):
        a = data.select(self.cfg(train=3))
        base = data.selection_ids(a)
        for variant in (dict(code={'train': 12}), dict(general={'filters': {'source': 'general_agent'}}), dict(music={'filters': {'lang': 'zh'}, 'train': 4}),
                        dict(cyber={'filters': {'sanitizer': 'address'}}), dict(webdev={'max_prompt_tokens': 12, 'train': 10})):
            b = data.selection_ids(data.select(self.cfg(**variant)))
            for domain in data.DOMAINS:
                self.assertEqual(b[domain]['test'], base[domain]['test'], (variant, domain))
        sha = lambda cfg: data.selection_shas(data.mimo_config(cfg), data.selection_ids(data.select(cfg)))  # noqa: E731
        self.assertEqual(sha(self.cfg())['test_sha'], sha(self.cfg(code={'train': 9}))['test_sha'])
        self.assertNotEqual(sha(self.cfg())['selection_sha'], sha(self.cfg(code={'train': 9}))['selection_sha'])
        self.assertNotEqual(sha(self.cfg())['test_sha'], sha({**self.cfg(), 'environment': {'mimo': {**self.cfg()['environment']['mimo'], 'seed': 8}}})['test_sha'])

    def test_no_group_leaks_and_group_cap(self):
        chosen = data.select(self.cfg(train=40, test=12))
        for domain, info in chosen['domains'].items():
            test_groups = {r['group'] for r in info['test']}
            self.assertFalse(test_groups & {r['group'] for r in info['train']}, domain)
            self.assertFalse({r['instance_id'] for r in info['test']} & {r['instance_id'] for r in info['train']})
            counts = {}
            for r in info['test']:
                counts[r['group']] = counts.get(r['group'], 0) + 1
            self.assertLessEqual(max(counts.values()), data.TEST_GROUP_CAP, domain)
            self.assertEqual(set(info['test_groups']), test_groups)

    def test_train_prefix_stable_and_deterministic(self):
        big = data.selection_ids(data.select(self.cfg(train=12)))
        small = data.selection_ids(data.select(self.cfg(train=5)))
        for domain in data.DOMAINS:
            self.assertEqual(small[domain]['train'], big[domain]['train'][:5])
        again = data.selection_ids(data.select(self.cfg(train=12)))
        self.assertEqual(big, again)
        self.assertEqual({d: v['canary'] for d, v in big.items()}, {d: v['test'][:2] for d, v in big.items()})

    def test_filters(self):
        chosen = data.select(self.cfg(train=8, general={'filters': {'source': 'terminal_bench'}, 'train': 3}, music={'filters': {'lang': 'zh', 'length': 'long'}},
                                      cyber={'filters': {'sanitizer': 'memory'}}, code={'filters': {'test_kind': 'dual'}}, webdev={'filters': {'max_prompt_chars': 20}}))
        attrs = lambda domain: [data.row_attrs(r) for r in chosen['domains'][domain]['train']]  # noqa: E731
        self.assertTrue(all(a['source'] == 'terminal_bench' for a in attrs('general')))
        self.assertTrue(all(a['lang'] == 'zh' and a['length'] == 'long' for a in attrs('music')))
        self.assertTrue(all(a['sanitizer'] == 'memory' for a in attrs('cyber')))
        self.assertTrue(all(a['test_kind'] == 'dual' for a in attrs('code')))
        self.assertTrue(all(a['prompt_chars'] <= 20 for a in attrs('webdev')))
        with self.assertRaises(data.DataError):     # only 10 terminal_bench rows exist, 6+ of them are test groups
            data.select(self.cfg(general={'filters': {'source': 'terminal_bench'}, 'train': 9}))
        problems = data.config_problems({'environment': {'mimo': {'domains': {'music': {'filters': {'bogus': 1}}, 'nope': {}}, 'zzz': 1}}})
        self.assertEqual(len(problems), 3)

    def test_screening_drops_rows_before_selection_and_is_recorded(self):
        first = data.select(self.cfg(train=4))
        victims = [r['instance_id'] for r in first['domains']['code']['train'][:2]] + [first['domains']['music']['test'][0]['instance_id']]
        chosen = data.select(self.cfg(train=4), screen=lambda row: 'unmodified repo passes' if row['instance_id'] in victims else None)
        ids = data.selection_ids(chosen)
        for victim in victims:
            self.assertEqual(chosen['screened'][victim], 'unmodified repo passes')
            self.assertFalse(any(victim in v[s] for v in ids.values() for s in data.SPLITS))
        self.assertEqual(len(ids['code']['train']), 4)           # refilled from the pool
        self.assertNotEqual(ids['music']['test'], data.selection_ids(first)['music']['test'])   # screening changes the pool, hence the test set
        directory = Path(self.tmp.name) / 'screening-data'
        directory.mkdir(exist_ok=True)
        (directory / 'screening.json').write_text(json.dumps({'code': {victims[0]: 'no tests'}}))
        again = data.select({**self.cfg(train=4), 'run': {'data_dir': str(directory)}})
        self.assertIn(victims[0], again['screened'])

    def test_prepare_assert_manifest_and_tamper(self):
        cfg = self.cfg('prep', train=4, test=4)
        manifest = data.prepare(cfg, exporter=fake_exporter)
        self.assertEqual(manifest['counts'], {'train': 20, 'test': 20, 'canary': 10})
        self.assertEqual(set(manifest['tasks']), set(data.SPLITS))
        self.assertTrue(manifest['prompt_tokens']['code']['n'] > 0 and manifest['prompt_tokens']['method'] == 'chars/4')
        self.assertTrue(all('task_dir' not in e for e in manifest['tasks']['train'].values()))
        self.assertEqual(data.assert_manifest(cfg)['test_sha'], manifest['test_sha'])
        self.assertEqual(data.assert_manifest(cfg, deep=False)['revision'], 'test')
        # fewer tasks than prepared is fine (prefix-stable); more is not
        smaller = copy.deepcopy(cfg)
        smaller['environment']['mimo']['domains']['code']['train'] = 2
        self.assertEqual(data.runtime_selection(smaller)['train']['code'], manifest['domains']['code']['train'][:2])
        data.assert_manifest(smaller)
        bigger = copy.deepcopy(cfg)
        bigger['environment']['mimo']['domains']['code']['train'] = 9
        with self.assertRaisesRegex(data.DataError, 'exceed prepared'):
            data.assert_manifest(bigger)
        filtered = copy.deepcopy(cfg)
        filtered['environment']['mimo']['domains']['music']['filters'] = {'lang': 'en'}
        with self.assertRaisesRegex(data.DataError, 'filters differ'):
            data.assert_manifest(filtered)
        # tamper: edit a test id, delete a task file, wrong strl.json
        path = data.manifest_path(cfg)
        original = path.read_text()
        edited = json.loads(original)
        edited['domains']['code']['test'][0] = 'task-9999'
        path.write_text(json.dumps(edited))
        with self.assertRaisesRegex(data.DataError, 'does not recompute'):
            data.assert_manifest(cfg)
        path.write_text(original)
        victim = data.task_dir(cfg['run']['data_dir'], 'train', 'music', manifest['domains']['music']['train'][0]) / 'strl.json'
        victim.write_text(json.dumps({'domain': 'code', 'instance_id': 'x'}))
        with self.assertRaisesRegex(data.DataError, 'strl.json names'):
            data.assert_manifest(cfg, deep=False)
        victim.unlink()
        with self.assertRaisesRegex(data.DataError, 'missing'):
            data.assert_manifest(cfg, deep=False)
        # re-prepare with a smaller selection removes the stale dirs (a directory scan must never find them)
        small = self.cfg('prep', train=2, test=4)
        data.prepare(small, exporter=fake_exporter)
        self.assertEqual(len(list((Path(small['run']['data_dir']) / 'tasks' / 'train').iterdir())), 10)

    def test_deep_check_detects_changed_parquet(self):
        cfg = self.cfg('deep', train=3, test=3)
        data.prepare(cfg, exporter=fake_exporter)
        data.assert_manifest(cfg, deep=True)
        manifest = json.loads(data.manifest_path(cfg).read_text())
        manifest['domains']['music']['parquet_sha256'] = '0' * 64
        data.manifest_path(cfg).write_text(json.dumps(manifest))
        with self.assertRaisesRegex(data.DataError, 'parquet sha256 changed'):
            data.assert_manifest(cfg, deep=True)

    def test_split_paths_repeat_the_held_out_tasks(self):
        cfg = self.cfg('paths', train=3, test=3)
        manifest = data.prepare(cfg, exporter=fake_exporter)
        test = data.split_paths(cfg, 'test', manifest, repeats=2)
        self.assertEqual(len(test), 30)
        self.assertEqual(test[:15], test[15:])
        self.assertEqual(data.split_paths(cfg, 'canary', manifest).__len__(), 10)
        self.assertEqual(len(data.split_paths(cfg, 'train', manifest)), 15)

    def test_prompt_report_uses_tokenizer_when_given(self):
        rows = {'train': [{'domain': 'code', 'raw': {'prompt': [{'content': 'a b c d'}]}}] * 3}
        self.assertEqual(data.prompt_report(rows, types.SimpleNamespace(encode=lambda t, add_special_tokens=False: t.split()))['code']['max'], 4)
        self.assertEqual(data.prompt_report(rows)['method'], 'chars/4')

    @unittest.skipUnless((data.DEFAULT_RAW_DIR / 'code.parquet').is_file(), 'real MiMo parquet not downloaded')
    def test_real_pool_pins_and_counts(self):
        counts = {'code': 2698, 'cyber': 1000, 'general': 989, 'music': 1000, 'webdev': 2093}
        for domain, n in counts.items():
            self.assertEqual(len(data.load_pool(domain)), n)
        cfg = {'environment': {'mimo': {'domains': {d: {'train': 24, 'test': 24} for d in data.DOMAINS}}}, 'run': {'data_dir': '/nonexistent'}}
        other = {'environment': {'mimo': {'domains': {d: {'train': 8, 'test': 24} for d in data.DOMAINS}}}, 'run': {'data_dir': '/nonexistent'}}
        a, b = (data.selection_ids(data.select(c)) for c in (cfg, other))
        self.assertEqual({d: v['test'] for d, v in a.items()}, {d: v['test'] for d, v in b.items()})
        self.assertTrue(all(len(v['test']) == 24 and len(v['train']) == 24 for v in a.values()))


# ---------------------------------------------------------------------------------------------------------------- judge
class FakeChat:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        reply = self.replies.pop(0) if self.replies else self.calls and self.replies
        if isinstance(reply, Exception):
            raise reply
        return {'text': reply, 'usage': {'prompt_tokens': 100, 'completion_tokens': 10}, 'cost_usd': 0.001}


RUBRIC = {'kind': 'rubric', 'base_score': 0.2, 'base_weight': 0.2, 'total_weight': 1.0, 'items': [
    {'id': 'a', 'weight': 0.5, 'question': 'q1', 'anchor': 'anchor one', 'answer': 'evidence one'},
    {'id': 'b', 'weight': 0.3, 'question': 'q2', 'anchor': 'anchor two', 'answer': 'evidence two'}]}
VISION_OK = json.dumps({'layout_integrity': 0.8, 'typography_hierarchy': 0.6, 'color_harmony': 0.7, 'whitespace': 0.5, 'content_richness': 0.4,
                        'query_fulfillment': 0.9, 'premium_assets': 0.3, 'reason': 'ok'})


def verdict(item, score, why='w'):
    return json.dumps({'results': {item: {'score': score, 'why': why}}})


def fast(**kw):
    return {'backoff_s': 0.0, 'retries': 2, **kw}


class Judge(unittest.TestCase):
    def test_combine_formula(self):
        self.assertAlmostEqual(judge.combine(RUBRIC, {'a': 1.0, 'b': 0.0}), 0.2 + 0.5)
        self.assertAlmostEqual(judge.combine(RUBRIC, {'a': 1.0, 'b': 1.0}), 1.0)
        self.assertEqual(judge.combine({'kind': 'vision', 'base_score': 0, 'total_weight': 1},
                                       {'visual': 0.3, 'query_fulfillment': 0.6, 'premium_assets': 0.9}), 0.6)
        with self.assertRaises(ValueError):
            judge.combine(RUBRIC, {'a': 1.0})
        with self.assertRaises(ValueError):
            judge.combine({**RUBRIC, 'total_weight': 0}, {'a': 1, 'b': 1})

    def test_rubric_prompt_and_per_item_calls(self):
        chat = FakeChat([verdict('a', 1), verdict('b', 0)])
        result = run(judge.Judge(fast(), chat=chat).score(RUBRIC))
        self.assertEqual(result['scores'], {'a': 1.0, 'b': 0.0})
        self.assertAlmostEqual(result['final'], 0.7)
        self.assertEqual(len(chat.calls), 2)                                       # one isolated call per item
        message = chat.calls[0][0][0]['content']
        self.assertIn('anchor one', message)
        self.assertIn('evidence one', message)
        self.assertNotIn('evidence two', message)
        self.assertEqual(chat.calls[0][1]['temperature'], 0)
        self.assertEqual(result['meta']['calls'], 2)
        self.assertAlmostEqual(result['meta']['cost_usd'], 0.002)

    def test_median_of_votes_and_retry_on_garbage(self):
        chat = FakeChat(['not json', verdict('a', 1), verdict('a', 0), verdict('a', 1)])
        result = run(judge.Judge(fast(votes=3, eval_votes=3), chat=chat).score({**RUBRIC, 'items': RUBRIC['items'][:1], 'total_weight': 0.7}, votes=3))
        self.assertEqual(result['scores']['a'], 1.0)                              # median(1, 0, 1); the garbage reply was retried
        self.assertEqual(len(chat.calls), 4)

    def test_failures_raise_judge_unavailable(self):
        for replies in (['x', 'y', 'z'], [judge.litellm.LiteLLMError('LiteLLM HTTP 500')] * 3,
                        [json.dumps({'results': {'a': {'score': 'high'}}})] * 3):
            with self.assertRaises(judge.JudgeUnavailable):
                run(judge.Judge(fast(), chat=FakeChat(replies)).score({**RUBRIC, 'items': RUBRIC['items'][:1]}))
        with self.assertRaises(judge.JudgeUnavailable):
            run(judge.Judge(fast(), chat=FakeChat([])).score({'kind': 'nope'}))

    def test_retry_then_success_and_timeout(self):
        chat = FakeChat([judge.litellm.LiteLLMError('LiteLLM HTTP 429'), verdict('a', 1)])
        result = run(judge.Judge(fast(), chat=chat).score({**RUBRIC, 'items': RUBRIC['items'][:1], 'total_weight': 0.7}))
        self.assertEqual(result['scores']['a'], 1.0)

        def slow(messages, **kw):
            import time
            time.sleep(0.4)
            return {'text': verdict('a', 1), 'usage': {}, 'cost_usd': 0}
        with self.assertRaises(judge.JudgeUnavailable):
            run(judge.Judge(fast(timeout_s=0.01, margin_s=0, retries=0), chat=lambda m, **kw: slow(m, **kw)).score({**RUBRIC, 'items': RUBRIC['items'][:1]}))

    def test_concurrency_is_limited(self):
        import threading
        state = {'now': 0, 'peak': 0}
        lock = threading.Lock()

        def chat(messages, **kw):
            import time
            with lock:
                state['now'] += 1
                state['peak'] = max(state['peak'], state['now'])
            time.sleep(0.05)
            with lock:
                state['now'] -= 1
            return {'text': verdict('a', 1), 'usage': {}, 'cost_usd': 0}

        async def many():
            j = judge.Judge(fast(concurrency=2), chat=chat)
            request = {**RUBRIC, 'items': RUBRIC['items'][:1], 'total_weight': 0.7}
            return await asyncio.gather(*[j.score(request) for _ in range(8)])
        run(many())
        self.assertLessEqual(state['peak'], 2)
        self.assertGreaterEqual(state['peak'], 2)

    def test_vision_prompt_hash_and_verdict(self):
        self.assertEqual(hashlib.sha256(rubric_prompts.build_prompt().encode()).hexdigest(),
                         'a4d3be63029e8fb28b469bf3d188816a7fa238b415749d7ff4aad1ca360b2997')   # the ruler itself: do not edit
        v = judge.parse_vision_verdict(VISION_OK)
        self.assertAlmostEqual(v['visual'], 0.6)
        self.assertIsNone(judge.parse_vision_verdict(json.dumps({'layout_integrity': 1})))   # a missing dimension is not a zero
        self.assertAlmostEqual(judge.vision_final({'visual': 0.6, 'query_fulfillment': 0.9, 'premium_assets': 0.3}), 0.6)

    def test_vision_call_sends_image_and_median(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'screenshot.png').write_bytes(metrics.encode_png([[(1, 2, 3)] * 4] * 4))
            chat = FakeChat([VISION_OK])
            request = {'kind': 'vision', 'base_score': 0.0, 'total_weight': 1.0, 'images': ['/logs/verifier/screenshot.png'], 'query': 'a cafe site'}
            result = run(judge.Judge(fast(vision_model='vision-x'), chat=chat).score(request, workdir=temp))
            content = chat.calls[0][0][0]['content']
            self.assertTrue(content[0]['image_url']['url'].startswith('data:image/png;base64,'))
            self.assertIn('a cafe site', content[1]['text'])
            self.assertEqual(chat.calls[0][1]['model'], 'vision-x')
            self.assertAlmostEqual(result['final'], (0.6 + 0.9 + 0.3) / 3)
            self.assertEqual(set(result['scores']), set(judge.VISION_KEYS))
            with self.assertRaises(judge.JudgeUnavailable):
                run(judge.Judge(fast(), chat=FakeChat([VISION_OK])).score({**request, 'images': ['/logs/verifier/missing.png']}, workdir=temp))
            with self.assertRaises(judge.JudgeUnavailable):
                run(judge.Judge(fast(), chat=FakeChat([VISION_OK])).score({**request, 'images': []}, workdir=temp))

    def test_temperature_fallback_for_models_that_reject_it(self):
        calls = []

        def fake_chat(messages, **kw):
            calls.append(kw)
            if 'temperature' in kw:
                raise judge.litellm.LiteLLMError('LiteLLM HTTP 400')
            return {'text': verdict('a', 1), 'usage': {}, 'cost_usd': None}
        j = judge.Judge(fast())
        j._key = 'k'
        with patch.object(judge.litellm, 'chat', fake_chat):
            result = run(j.score({**RUBRIC, 'items': RUBRIC['items'][:1], 'total_weight': 0.7}))
            run(j.score({**RUBRIC, 'items': RUBRIC['items'][:1], 'total_weight': 0.7}))
        self.assertEqual(result['scores']['a'], 1.0)
        self.assertEqual(['temperature' in c for c in calls], [True, False, False])    # remembered after the first rejection

    def test_rescore_variance(self):
        chat = FakeChat([verdict('a', 1), verdict('a', 0)])
        out = run(judge.Judge(fast(), chat=chat).rescore_variance({**RUBRIC, 'items': RUBRIC['items'][:1], 'total_weight': 0.5, 'base_score': 0.0}, n=2))
        self.assertEqual(sorted(out['finals']), [0.0, 1.0])
        self.assertAlmostEqual(out['variance'], 0.25)
        self.assertEqual(out['max_abs_diff'], 1.0)

    def test_selftest_with_fake_chat_and_failure_receipt(self):
        chat = FakeChat([verdict('sum', 1), VISION_OK])
        receipt = judge.selftest(fast(), chat=chat)
        self.assertTrue(receipt['rubric']['ok'] and receipt['vision']['ok'])
        self.assertEqual(receipt['rubric']['final'], 1.0)
        self.assertEqual(receipt['vision']['model'], 'openai/gpt-6-luna')
        broken = judge.selftest(fast(retries=0), chat=FakeChat([judge.litellm.LiteLLMError('LiteLLM HTTP 500')] * 2))
        self.assertFalse(broken['rubric']['ok'])
        self.assertIn('JudgeUnavailable', broken['rubric']['error'])

    def test_settings_validation(self):
        self.assertEqual(judge.judge_settings(None)['model'], 'openai/gpt-6-luna')
        with self.assertRaises(ValueError):
            judge.judge_settings({'votez': 3})
        cfg = {'environment': {'mimo': {'judge': {'votes': 0, 'timeout_s': -1, 'bogus': 1}}}}
        self.assertEqual(len(judge.settings_problems(cfg)), 3)

    def test_litellm_chat_sends_temperature_and_tag(self):
        from strl.services import litellm
        captured = {}

        class Response:
            headers = {'x-litellm-response-cost': '0.5'}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({'choices': [{'message': {'content': 'hi'}}], 'usage': {'prompt_tokens': 1}}).encode()

        def fake_open(request, timeout=None):
            captured['body'] = json.loads(request.data)
            return Response()
        with patch('urllib.request.urlopen', fake_open):
            out = litellm.chat([{'role': 'user', 'content': 'x'}], key='k', temperature=0, tag='strl-mimo-judge')
            litellm.chat([{'role': 'user', 'content': 'x'}], key='k')
        self.assertEqual(out['cost_usd'], 0.5)
        self.assertEqual(captured['body'].get('temperature', 'absent'), 'absent')       # second call: not sent unless asked
        with patch('urllib.request.urlopen', fake_open):
            litellm.chat([{'role': 'user', 'content': 'x'}], key='k', temperature=0, tag='strl-mimo-judge')
        self.assertEqual(captured['body']['temperature'], 0)
        self.assertIn('strl-mimo-judge', captured['body']['metadata']['tags'])


# ---------------------------------------------------------------------------------------------------------------- generator
@dataclass
class TID:
    instance_id: str
    repetition_id: int = 0

    def to_string(self):
        return f'{self.instance_id}_{self.repetition_id}'


class NullCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def fake_results(*, exc=None, reward=1.0, rollout=True, trial_dir=None, turns=3, verdict_missing=False):
    return types.SimpleNamespace(
        exception_info=types.SimpleNamespace(exception_type=exc) if exc else None,
        verifier_result=None if verdict_missing or reward is None else types.SimpleNamespace(rewards={'reward': reward}),
        agent_result=types.SimpleNamespace(rollout_details=[{'completion_token_ids': [[1, 2, 3], [4, 5]], 'prompt_token_ids': [[0], [0, 1, 2, 3]], 'logprobs': [[0.0] * 3, [0.0] * 2]}] if rollout else None,
                                           metadata={'n_episodes': turns}),
        trial_uri=Path(trial_dir).as_uri() if trial_dir else None)


class Env:
    """Scaffolding for run_trajectory: a task dir, trial dirs with verifier files and a queue of fake trial results."""

    def __init__(self, temp, *, domain='code', strl=None, judge_obj=None, settings=None, caps=None):
        self.temp = Path(temp)
        self.task = self.temp / 'tasks' / f'{domain}-t1'
        self.task.mkdir(parents=True)
        (self.task / 'strl.json').write_text(json.dumps(strl if strl is not None else {'domain': domain, 'artifacts': ['/app/answer.md'], 'agent': {'name': 'terminus-2'}}))
        (self.task / 'task.toml').write_text('[agent]\ntimeout_sec = 300.0\n[verifier]\ntimeout_sec = 60.0\n')
        self.queue, self.configs, self.sessions = [], [], []
        self.deps = types.SimpleNamespace(
            template={'agent': {'name': 'terminus-2', 'kwargs': {'max_turns': 8, 'llm_kwargs': {'max_tokens': 2048}, 'model_info': {}}}, 'environment': {}, 'verifier': {}},
            settings={**generator.DEFAULT_SETTINGS, **(settings or {})}, caps=lambda d: (caps or {}).get(d, {}), rate_limiter=NullCtx(), judge=judge_obj,
            create_trial=self.create_trial, finish_session=self.finish)

    def trial_dir(self, *, reward_txt='1', details=..., request=None, files=None):
        """details=... (default) mirrors a real trial: the shared verifier entrypoint always leaves a non-empty details.json;
        pass details=None to simulate a verifier that did not run."""
        if details is ...:
            details = {'metric': float(reward_txt)} if reward_txt is not None else None
        directory = self.temp / f'trial{len(list(self.temp.glob("trial*")))}'
        (directory / 'verifier').mkdir(parents=True)
        if reward_txt is not None:
            (directory / 'verifier/reward.txt').write_text(reward_txt)
        if details is not None:
            (directory / 'verifier/details.json').write_text(json.dumps(details))
        if request is not None:
            (directory / 'verifier/judge_request.json').write_text(json.dumps(request))
        for name, body in (files or {}).items():
            (directory / 'verifier' / name).write_bytes(body if isinstance(body, bytes) else body.encode())
        return directory

    async def create_trial(self, config):
        self.configs.append(config)
        outcome = self.queue.pop(0)

        class Trial:
            async def run(_self):
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome
        return Trial()

    async def finish(self, session):
        self.sessions.append(session)

    def go(self, **kw):
        return run(generator.run_trajectory(self.deps, str(self.task), TID('7'), **kw))


class Generator(unittest.TestCase):
    def test_trial_config_layering(self):
        template = {'agent': {'name': 'terminus-2', 'kwargs': {'max_turns': 8, 'temperature': 1.0, 'llm_kwargs': {'max_tokens': 2048}, 'model_info': {'max_output_tokens': 2048}}},
                    'environment': {'type': 'modal'}, 'verifier': {}}
        before = copy.deepcopy(template)
        plain = generator.build_trial_config(template, '/t/code-1', session_id='s')
        self.assertEqual(plain['agent']['kwargs']['max_turns'], 8)
        self.assertEqual(plain['task'], {'path': '/t/code-1'})
        single = generator.build_trial_config(template, '/t/music-1', session_id='s', temperature=0.0, cache_salt='salt', meta={
            'agent': {'import_path': 'strl.tasks.mimo.agents:SingleShotAgent', 'kwargs': {'x': 1}}, 'max_turns': 1, 'max_tokens': 4096})
        self.assertEqual(single['agent']['import_path'], 'strl.tasks.mimo.agents:SingleShotAgent')
        self.assertNotIn('name', single['agent'])
        kw = single['agent']['kwargs']
        self.assertEqual((kw['max_turns'], kw['llm_kwargs']['max_tokens'], kw['model_info']['max_output_tokens'], kw['temperature'], kw['x']), (1, 4096, 4096, 0.0, 1))
        self.assertEqual(kw['llm_kwargs']['extra_body']['cache_salt'], 'salt')
        named = generator.build_trial_config(single, '/t/x', session_id='s2', meta={'agent': {'name': 'terminus-2'}})
        self.assertEqual(named['agent']['name'], 'terminus-2')
        self.assertNotIn('import_path', named['agent'])
        self.assertEqual(template, before)                                      # the template is never mutated
        capped = generator.build_trial_config(template, '/t/x', session_id='s', toml={'agent': {'timeout_sec': 1800}, 'verifier': {'timeout_sec': 100}},
                                              caps={'max_agent_sec': 600, 'max_verifier_sec': 200})
        self.assertEqual((capped['agent']['override_timeout_sec'], capped['verifier']['override_timeout_sec']), (600.0, 100.0))
        self.assertNotIn('override_timeout_sec', plain['agent'])

    def test_template_has_no_global_overrides_and_validates_with_harbor(self):
        cfg = load(SMOKE)
        template = generator.mimo_template(cfg)
        for section, keys in generator.OVERRIDE_KEYS.items():
            for key in keys:
                self.assertNotIn(key, template.get(section, {}), (section, key))
        self.assertEqual(template['agent']['kwargs']['max_turns'], 12)
        from harbor.models.trial.config import TrialConfig
        with tempfile.TemporaryDirectory() as temp:
            env = Env(temp, strl={'domain': 'music', 'agent': {'import_path': 'strl.tasks.mimo.agents:SingleShotAgent'}, 'max_turns': 1, 'max_tokens': 2048})
            meta, toml = generator.read_task(env.task)
            for agent_meta in ({'name': 'terminus-2'}, meta['agent']):
                config = generator.build_trial_config({**template, 'agent': {**template['agent'], 'model_name': 'hosted_vllm/m'}}, env.task, session_id='s',
                                                      meta={**meta, 'agent': agent_meta}, toml=toml, temperature=0.5, caps={'max_agent_sec': 60})
                TrialConfig.model_validate(config)

    def test_decide_policy(self):
        d = lambda **kw: generator.decide(fake_results(**kw), attempt=0, attempts=2, policy='score')  # noqa: E731
        self.assertEqual(d(), 'use')
        self.assertEqual(d(exc='ContextLengthExceededError'), 'use')
        self.assertEqual(d(exc='AgentTimeoutError'), 'use')                       # scored by the verifier
        self.assertEqual(d(exc='VerifierTimeoutError', reward=None), 'use')       # reward 0, never retried
        self.assertEqual(d(exc='RuntimeError', reward=None), 'retry')
        self.assertEqual(d(reward=None), 'retry')
        self.assertEqual(generator.decide(fake_results(exc='RuntimeError', reward=None), attempt=1, attempts=2, policy='score'), 'mask')
        self.assertEqual(generator.decide(fake_results(exc='AgentTimeoutError'), attempt=0, attempts=2, policy='mask'), 'mask')
        self.assertEqual(generator.decide(fake_results(exc='AgentTimeoutError', reward=None), attempt=0, attempts=2, policy='score'), 'mask')
        self.assertEqual(generator.decide(fake_results(exc='VerifierTimeoutError', reward=None), attempt=0, attempts=2, policy='mask'), 'mask')

    def test_success_reads_verifier_files_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            env = Env(temp)
            trial = env.trial_dir(reward_txt='1', details={'metric': 1.0, 'components': {'tests': 1.0}, 'passed': True}, files={'answer.md': 'x' * 20000, 'screenshot.png': b'png'})
            env.queue = [fake_results(reward=1.0, trial_dir=trial)]
            out, summary = env.go(temperature=0.7)
            self.assertEqual((out['reward'], out['stop_reason'], out['num_turns']), (1.0, 'complete', 3))
            self.assertEqual(env.configs[0]['agent']['kwargs']['temperature'], 0.7)
            self.assertFalse(summary['masked'])
            self.assertEqual((summary['domain'], summary['metric'], summary['components'], summary['tokens'], summary['turns']), ('code', 1.0, {'tests': 1.0}, 5, 3))
            self.assertEqual(len(summary['artifact']), 12000)
            self.assertTrue(summary['screenshot'].endswith('screenshot.png'))
            self.assertTrue(summary['passed'])
            self.assertEqual(len(env.sessions), 1)

    def test_retry_only_transport_errors_and_at_most_two_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            env = Env(temp)
            good = env.trial_dir(reward_txt='1', details={'metric': 1.0})
            env.queue = [RuntimeError('modal hiccup'), fake_results(reward=1.0, trial_dir=good)]
            out, summary = env.go()
            self.assertEqual((len(env.configs), out['reward'], summary['masked']), (2, 1.0, False))
            self.assertNotEqual(env.configs[0]['agent']['kwargs']['session_id'], env.configs[1]['agent']['kwargs']['session_id'])
            env = Env(temp + '/b')
            env.queue = [fake_results(exc='RuntimeError', reward=None)] * 3
            out, summary = env.go()
            self.assertEqual((len(env.configs), out['stop_reason'], out['rollout_details'], summary['mask_reason']), (2, 'error', None, 'error'))
            for exc in ('AgentTimeoutError', 'VerifierTimeoutError'):    # never retried
                env = Env(temp + f'/{exc}')
                env.queue = [fake_results(exc=exc, reward=None if exc.startswith('Verifier') else 0.0, trial_dir=env_dir(env)), fake_results(reward=1.0)]
                out, summary = env.go()
                self.assertEqual((len(env.configs), out['reward'], out['stop_reason'], summary['stop']), (1, 0.0, 'complete', {'AgentTimeoutError': 'agent_timeout', 'VerifierTimeoutError': 'verifier_timeout'}[exc]))

    def test_timeout_policy_mask_and_context_length(self):
        with tempfile.TemporaryDirectory() as temp:
            env = Env(temp, settings={'timeout_policy': 'mask'})
            env.queue = [fake_results(exc='AgentTimeoutError', reward=0.0)]
            out, summary = env.go()
            self.assertEqual((out['stop_reason'], summary['masked'], summary['mask_reason']), ('error', True, 'agent_timeout'))
            env = Env(temp + '/c')
            env.queue = [fake_results(exc='ContextLengthExceededError', reward=0.9, trial_dir=env_dir(env))]
            out, summary = env.go()
            self.assertEqual((out['reward'], out['stop_reason'], summary['stop']), (0.0, 'context_length', 'context_length'))

    def test_policy_failures_are_scored_in_eval_and_never_retried(self):
        """canary can-mimo-d: a 0.8B policy that ran out of time (webdev) or of output budget (music) was masked, so the held-out eval
        silently dropped exactly the tasks it failed. Evaluation scores those as 0; training still cannot use a trial without tokens."""
        with tempfile.TemporaryDirectory() as temp:
            for exc in ('AgentTimeoutError', 'OutputLengthExceededError'):
                env = Env(temp + f'/{exc}-eval')
                env.queue = [fake_results(exc=exc, reward=None, rollout=False), fake_results(reward=1.0)]
                out, summary = env.go(phase='eval')
                self.assertEqual((len(env.configs), summary['masked'], summary['metric'], summary['reward']), (1, False, 0.0, 0.0), exc)
                self.assertEqual(summary['stop'], {'AgentTimeoutError': 'agent_timeout', 'OutputLengthExceededError': 'output_length'}[exc])
            env = Env(temp + '/train')
            env.queue = [fake_results(exc='AgentTimeoutError', reward=0.0, rollout=False, trial_dir=env_dir(env))]
            summary = env.go(phase='train')[1]
            self.assertEqual((summary['masked'], summary['mask_reason']), (True, 'agent_timeout'))
            env = Env(temp + '/out-train')      # an overflowing reply that did leave tokens is a scored 0 for training too
            env.queue = [fake_results(exc='OutputLengthExceededError', reward=None, trial_dir=env_dir(env))]
            out, summary = env.go(phase='train')
            self.assertEqual((out['reward'], out['stop_reason'], summary['masked'], summary['stop']), (0.0, 'output_length', False, 'output_length'))
            env = Env(temp + '/mask-eval', settings={'timeout_policy': 'mask'})   # the user asked to mask timeouts: still masked in eval
            env.queue = [fake_results(exc='AgentTimeoutError', reward=None, rollout=False)]
            self.assertTrue(env.go(phase='eval')[1]['masked'])

    def test_infra_error_and_empty_rollout_are_masked(self):
        with tempfile.TemporaryDirectory() as temp:
            env = Env(temp)
            env.queue = [fake_results(trial_dir=env.trial_dir(reward_txt='1', details={'infra_error': True, 'metric': 1.0}))]
            out, summary = env.go()
            self.assertEqual((out['rollout_details'], out['stop_reason'], summary['masked'], summary['mask_reason']), (None, 'error', True, 'infra_error'))
            env = Env(temp + '/e')
            env.queue = [fake_results(rollout=False, trial_dir=env_dir(env))]
            self.assertEqual(env.go()[1]['mask_reason'], 'no_rollout')

    def test_judged_trial_uses_judge_and_masks_on_failure(self):
        request = {**RUBRIC, 'base_score': 0.2}
        with tempfile.TemporaryDirectory() as temp:
            chat = FakeChat([verdict('a', 1), verdict('b', 1)])
            env = Env(temp, domain='general', judge_obj=judge.Judge(fast(), chat=chat))
            env.queue = [fake_results(reward=0.2, trial_dir=env.trial_dir(reward_txt='0.2', details={'metric': 0.2}, request=request))]
            out, summary = env.go(phase='train')
            self.assertAlmostEqual(out['reward'], 1.0)
            self.assertAlmostEqual(summary['metric'], 1.0)
            self.assertEqual(summary['judge']['scores'], {'a': 1.0, 'b': 1.0})
            self.assertEqual(len(chat.calls), 2)                      # 1 vote per item in training
            env = Env(temp + '/eval', domain='general', judge_obj=judge.Judge(fast(eval_votes=3), chat=FakeChat([verdict('a', 1)] * 3 + [verdict('b', 0)] * 3)))
            env.queue = [fake_results(reward=0.2, trial_dir=env.trial_dir(reward_txt='0.2', request=request))]
            out, summary = env.go(phase='eval')
            self.assertAlmostEqual(out['reward'], 0.7)
            self.assertEqual(summary['judge']['meta']['calls'], 6)    # eval votes
            env = Env(temp + '/down', domain='general', judge_obj=judge.Judge(fast(retries=0), chat=FakeChat([judge.litellm.LiteLLMError('HTTP 503')] * 4)))
            env.queue = [fake_results(reward=0.2, trial_dir=env.trial_dir(reward_txt='0.2', request=request))]
            out, summary = env.go()
            self.assertEqual((out['stop_reason'], out['rollout_details'], summary['mask_reason']), ('error', None, 'judge'))   # masked, never 0
            env = Env(temp + '/nojudge', domain='general')
            env.queue = [fake_results(reward=0.2, trial_dir=env.trial_dir(reward_txt='0.2', request=request))]
            self.assertEqual(env.go()[1]['mask_reason'], 'judge')

    def test_gather_with_deadline(self):
        async def job(seconds, value):
            await asyncio.sleep(seconds)
            if value == 'boom':
                raise RuntimeError('boom')
            return value

        out = run(generator.gather_with_deadline([lambda: job(0, 'a'), lambda: job(5, 'slow'), lambda: job(0, 'boom')], 0.2))
        self.assertEqual([s for s, _ in out], ['done', 'deadline', 'error'])
        self.assertEqual(out[0][1], 'a')
        self.assertEqual([s for s, _ in run(generator.gather_with_deadline([lambda: job(0, 'x')], None))], ['done'])

    def test_rollout_metrics_per_domain_and_masked_groups(self):
        def s(domain, group_reward, **kw):
            base = {'domain': domain, 'task': 't', 'rep': 0, 'reward': group_reward, 'metric': group_reward, 'passed': group_reward >= 1, 'masked': False, 'mask_reason': None,
                    'stop': 'complete', 'turns': 4, 'tokens': 100, 'judge': None}
            return {**base, **kw}
        summaries = [s('code', 1.0), s('code', 0.0, stop='agent_timeout'), s('music', 0.5), s('music', 0.0, masked=True, mask_reason='deadline', stop='deadline', metric=None),
                     s('webdev', 0.7, judge={'meta': {'latency_s': 2.0, 'cost_usd': 0.01}}), s('webdev', 0.0, masked=True, mask_reason='judge', stop='judge', metric=None),
                     s('general', 0.2, stop='context_length')]
        groups = ['a', 'a', 'b', 'b', 'c', 'c', 'd']
        out = generator.rollout_metrics(summaries, groups)
        self.assertEqual(out['generate/code/num_trials'], 2.0)
        self.assertEqual(out['generate/code/num_timeout'], 1.0)
        self.assertEqual(out['generate/code/reward_mean'], 0.5)
        self.assertEqual(out['generate/code/pass_rate'], 0.5)
        self.assertEqual(out['generate/music/num_masked'], 1.0)
        self.assertEqual(out['generate/music/num_deadline'], 1.0)
        self.assertEqual(out['generate/music/reward_mean'], 0.5)                  # masked trial not averaged in
        self.assertEqual(out['generate/webdev/judge_masked'], 1.0)
        self.assertEqual(out['generate/webdev/judge_latency_s'], 2.0)
        self.assertEqual(out['generate/general/num_context_exceeded'], 1.0)
        self.assertEqual(out['environment/music/masked_group_fraction'], 1.0)
        self.assertEqual(out['environment/code/masked_group_fraction'], 0.0)
        self.assertAlmostEqual(out['environment/masked_group_fraction'], 2 / 4)
        self.assertEqual(out['generate/code/tokens'], 100.0)

    def test_summary_store_is_bounded_and_dump_safe(self):
        generator.SUMMARIES.clear()
        extra = {}
        summary = {'domain': 'code', 'judge': {'final': 0.5, 'scores': {'a': 1}, 'rationale': 'r' * 2000, 'meta': {'cost_usd': 0.1, 'latency_s': 1.0}}}
        generator.remember('eval', TID('1', 0), summary, extra)
        json.dumps(extra)                                                         # the SkyRL eval dump must be able to serialise it
        self.assertEqual(len(extra['mimo']['judge']['rationale']), 600)
        for i in range(generator.SUMMARY_STORE_LIMIT + 10):
            generator.remember('train', TID(str(i)), {'domain': 'code', 'judge': None})
        self.assertEqual(len(generator.SUMMARIES), generator.SUMMARY_STORE_LIMIT)
        self.assertEqual(generator.summaries_for([TID('1')], 'eval'), [None])     # oldest evicted
        generator.SUMMARIES.clear()


def env_dir(env):
    return env.trial_dir(reward_txt='0', details={})


@contextmanager
def fake_skyrl_harbor():
    """Stub modules for the SkyRL Harbor generator so make_generator_class() can be exercised without SkyRL/loguru."""
    @dataclass
    class HarborTrajectoryOutput:
        trajectory_id: object
        rollout_details: object = None
        reward: float = 0.0
        num_turns: int = 0
        stop_reason: str = 'complete'
        e2e_time: object = None

    def build(outputs, overlong_filtering):
        masked = {o.trajectory_id.instance_id for o in outputs if o.stop_reason in ('agent_timeout', 'error') or o.rollout_details is None}
        return {'rewards': [0.0 if o.trajectory_id.instance_id in masked else o.reward for o in outputs],
                'stop_reasons': ['error' if o.trajectory_id.instance_id in masked else 'complete' for o in outputs],
                'trajectory_ids': [o.trajectory_id for o in outputs], 'rollout_metrics': {'generate/num_masked_instances': len(masked)}}

    class HarborGenerator:
        def __init__(self, generator_cfg, harbor_cfg, inference_engine_client, tokenizer, max_seq_len):
            self.generator_cfg, self.inference_engine_client = generator_cfg, inference_engine_client
            self._harbor_trial_config_template = copy.deepcopy(harbor_cfg)
            self._rate_limiter = NullCtx()

        def _compute_cache_salt(self):
            return None

    module = types.ModuleType('examples.train_integrations.harbor.harbor_generator')
    module.HarborGenerator, module.HarborTrajectoryOutput, module.build_step_wise_generator_output = HarborGenerator, HarborTrajectoryOutput, build
    logger = types.SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None, debug=lambda *a, **k: None)

    class Bar:      # tqdm creates a multiprocessing lock on first use (a leaked-semaphore warning at exit): not needed here
        def __init__(self, *a, **k):
            pass

        def update(self, n=1):
            pass

        def close(self):
            pass
    stubs = {'tqdm': types.SimpleNamespace(tqdm=Bar), 'examples': types.ModuleType('examples'), 'examples.train_integrations': types.ModuleType('examples.train_integrations'),
             'examples.train_integrations.harbor': types.ModuleType('examples.train_integrations.harbor'),
             'examples.train_integrations.harbor.harbor_generator': module, 'loguru': types.SimpleNamespace(logger=logger)}
    with patch.dict(sys.modules, stubs):
        yield HarborTrajectoryOutput


class GeneratorClass(unittest.TestCase):
    def batch(self, tmp, phase, tasks):
        extras = [{'domain': generator.task_domain(t), 'rep': i, 'task_dir': t} for i, t in enumerate(tasks)]
        return {'prompts': tasks, 'trajectory_ids': [TID(str(i)) for i in range(len(tasks))], 'env_extras': extras,
                'sampling_params': {'temperature': 0.0 if phase == 'eval' else 1.0}, 'batch_metadata': types.SimpleNamespace(training_phase=phase)}

    def test_generate_deadline_masking_metrics_and_temperature(self):
        seen = {}

        async def fake_run(deps, prompt, tid, **kw):
            seen[tid.instance_id] = kw
            if 'slow' in prompt:
                await asyncio.sleep(30)
            summary = {'domain': kw['env_extra']['domain'], 'task': Path(prompt).name, 'rep': kw['env_extra']['rep'], 'reward': 1.0, 'metric': 1.0, 'passed': True,
                       'masked': False, 'mask_reason': None, 'stop': 'complete', 'turns': 2, 'tokens': 10, 'judge': None}
            return {'rollout_details': [1], 'reward': 1.0, 'num_turns': 2, 'stop_reason': 'complete', 'e2e_time': 0.1, 'trajectory_id': tid}, summary

        with fake_skyrl_harbor():
            cls = generator.make_generator_class()
            gen = cls(generator_cfg=types.SimpleNamespace(apply_overlong_filtering=True), harbor_cfg={'agent': {'kwargs': {}}}, inference_engine_client=types.SimpleNamespace(
                finish_session=lambda s: asyncio.sleep(0)), tokenizer=None, max_seq_len=1000, mimo={'step_deadline_sec': 0.3, 'caps': {}})
            batch = self.batch('x', 'eval', ['/d/code-a', '/d/music-b', '/d/webdev-slow'])
            with patch.object(generator, 'run_trajectory', fake_run):
                output = run(gen.generate(batch, disable_tqdm=True))
        self.assertEqual({k: v['temperature'] for k, v in seen.items()}, {'0': 0.0, '1': 0.0, '2': 0.0})   # eval temperature reaches the agent
        self.assertEqual({v['phase'] for v in seen.values()}, {'eval'})
        self.assertEqual(output['rewards'], [1.0, 1.0, 0.0])
        self.assertEqual(output['stop_reasons'][2], 'error')                      # deadline straggler masked
        metrics_out = output['rollout_metrics']
        self.assertEqual(metrics_out['generate/webdev/num_deadline'], 1.0)
        self.assertEqual(metrics_out['generate/webdev/num_masked'], 1.0)
        self.assertEqual(metrics_out['generate/code/pass_rate'], 1.0)
        self.assertEqual(metrics_out['environment/webdev/masked_group_fraction'], 1.0)
        self.assertEqual(metrics_out['generate/num_masked_instances'], 1)         # SkyRL's own metric is kept
        self.assertEqual(batch['env_extras'][2]['mimo']['mask_reason'], 'deadline')   # lands in the eval dump
        self.assertEqual(batch['env_extras'][0]['mimo']['domain'], 'code')

    def test_generate_validates_inputs_and_train_phase_uses_batch_temperature(self):
        async def fake_run(deps, prompt, tid, **kw):
            return {'rollout_details': None, 'reward': 0.0, 'num_turns': 0, 'stop_reason': 'error', 'e2e_time': 0.0, 'trajectory_id': tid}, \
                {'domain': 'code', 'task': 't', 'rep': 0, 'reward': 0.0, 'metric': None, 'passed': False, 'masked': True, 'mask_reason': 'error', 'stop': 'error',
                 'turns': 0, 'tokens': 0, 'judge': None}
        with fake_skyrl_harbor():
            gen = generator.make_generator_class()(generator_cfg=types.SimpleNamespace(apply_overlong_filtering=True, eval_sampling_params=types.SimpleNamespace(temperature=0.25)),
                                                   harbor_cfg={'agent': {'kwargs': {}}}, inference_engine_client=types.SimpleNamespace(finish_session=lambda s: asyncio.sleep(0)),
                                                   tokenizer=None, max_seq_len=10)
            with self.assertRaises(ValueError):
                run(gen.generate({'prompts': ['a'], 'trajectory_ids': None}))
            with self.assertRaises(ValueError):
                run(gen.generate({'prompts': ['a'], 'trajectory_ids': []}))
            batch = self.batch('x', 'train', ['/d/code-a'])
            del batch['sampling_params']['temperature']                          # falls back to the generator's eval temperature only in eval
            with patch.object(generator, 'run_trajectory', fake_run):
                output = run(gen.generate(batch, disable_tqdm=True))
            self.assertEqual(output['rollout_metrics']['environment/masked_group_fraction'], 1.0)
            self.assertEqual(output['rewards'], [0.0])


# ---------------------------------------------------------------------------------------------------------------- sampler
class Sampler(unittest.TestCase):
    def test_quotas_largest_remainder(self):
        self.assertEqual(sampler.quotas({'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1}, 40), {d: 8 for d in 'abcde'})
        q = sampler.quotas({'a': 3, 'b': 1}, 10)
        self.assertEqual((q['a'], q['b'], sum(q.values())), (8, 2, 10))           # 7.5 / 2.5 -> tie goes to the first by name
        q = sampler.quotas({'a': 10, 'b': 1, 'c': 1}, 5)
        self.assertEqual(sum(q.values()), 5)
        self.assertTrue(all(v >= 1 for v in q.values()), q)                      # every domain keeps a slot when the batch allows
        q = sampler.quotas({'a': 1, 'b': 1, 'c': 1}, 2)
        self.assertEqual(sum(q.values()), 2)
        with self.assertRaises(ValueError):
            sampler.quotas({'a': 0, 'b': 1}, 4)

    def make(self, sizes, weights=None, batch=10, seed=3):
        domains = [d for d, n in sizes.items() for _ in range(n)]
        return sampler.DomainBatchSampler(domains, weights or {d: 1.0 for d in sizes}, batch, seed), domains

    def test_proportions_coverage_and_no_duplicates(self):
        s, domains = self.make({'a': 30, 'b': 20, 'c': 10, 'd': 25, 'e': 40}, batch=10)
        self.assertEqual(len(s), 5)                                               # scarcest domain: 10 // 2
        seen = []
        for batch in s:
            self.assertEqual(len(batch), 10)
            self.assertEqual(len(set(batch)), 10)
            counts = {d: sum(domains[i] == d for i in batch) for d in 'abcde'}
            self.assertEqual(counts, {d: 2 for d in 'abcde'})
            seen += batch
        self.assertEqual(len(seen), len(set(seen)))                               # no task repeats inside an epoch
        self.assertTrue({domains[i] for i in seen} == set('abcde'))

    def test_weighted_mix_and_determinism_per_epoch(self):
        s, domains = self.make({'a': 60, 'b': 60}, {'a': 3.0, 'b': 1.0}, batch=8)
        first = list(s)
        self.assertEqual({d: sum(domains[i] == d for i in first[0]) for d in 'ab'}, {'a': 6, 'b': 2})
        second = list(s)
        self.assertNotEqual(first, second)                                       # a new epoch reshuffles
        fresh, _ = self.make({'a': 60, 'b': 60}, {'a': 3.0, 'b': 1.0}, batch=8)
        self.assertEqual(list(fresh), first)                                     # same seed -> same epoch 0
        self.assertEqual(list(fresh), second)
        fresh.set_epoch(0)
        self.assertEqual(list(fresh), first)
        other, _ = self.make({'a': 60, 'b': 60}, {'a': 3.0, 'b': 1.0}, batch=8, seed=4)
        self.assertNotEqual(list(other), first)

    def test_epochs_do_not_reuse_within_an_epoch_and_unweighted_domains_are_skipped(self):
        domains = ['a'] * 12 + ['b'] * 12 + ['probe'] * 5
        s = sampler.DomainBatchSampler(domains, {'a': 1, 'b': 1}, 4, 0)
        for epoch in range(3):
            batches = s.epoch_batches(epoch)
            flat = [i for b in batches for i in b]
            self.assertEqual(len(flat), len(set(flat)))
            self.assertTrue(all(domains[i] != 'probe' for i in flat))
        self.assertEqual(len(s), 6)

    def test_domain_smaller_than_its_quota_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'fewer training tasks'):
            self.make({'a': 3, 'b': 50}, batch=10)

    def test_trainer_hook_replaces_the_dataloader_builder(self):
        class Base:
            def __init__(self, cfg, train_dataset):
                self.cfg, self.train_dataset = cfg, train_dataset
                self.train_dataloader = self.total_training_steps = None
                self._build_train_dataloader_and_compute_training_steps()

            def _build_train_dataloader_and_compute_training_steps(self):
                self.train_dataloader, self.total_training_steps = 'stock', -1

        cfg = types.SimpleNamespace(trainer=types.SimpleNamespace(epochs=4, max_training_steps=None))
        s, _ = self.make({'a': 8, 'b': 8}, batch=4)
        built = []
        cls = sampler.make_trainer_class(Base, s, build_loader=lambda c, d, smp: built.append((c, d, smp)) or smp)
        trainer = cls(cfg, 'dataset')
        self.assertIs(trainer.train_dataloader, s)
        self.assertEqual(trainer.total_training_steps, len(s) * 4)
        self.assertEqual(built, [(cfg, 'dataset', s)])
        cfg.trainer.max_training_steps = 3
        self.assertEqual(cls(cfg, 'dataset').total_training_steps, 3)
        self.assertEqual(cls(cfg, None).train_dataloader, 'stock')               # no train dataset: stock path

    def test_override_matches_the_pinned_skyrl_trainer(self):
        """The method we override and the builder it replaces exist in the pinned SkyRL (pinned/ copy and the full checkout when present)."""
        source = (ROOT / 'pinned/pinned-trainer.py').read_text()
        self.assertIn('def _build_train_dataloader_and_compute_training_steps(self):', source)
        self.assertIn('build_dataloader(self.cfg, self.train_dataset, is_train=True)', source)
        self.assertIn('self.total_training_steps = len(self.train_dataloader) * self.cfg.trainer.epochs', source)
        entry = (ROOT / 'strl/entrypoint.py').read_text()
        self.assertIn('def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg):', entry)
        self.assertIn('_build_train_dataloader_and_compute_training_steps', (ROOT / 'strl/tasks/mimo/sampler.py').read_text())
        pin = Path('/private/tmp/claude-502/-Users-mohamed-elfeki/2543c234-2ba6-439d-a94a-907e82069f6b/scratchpad/skyrl-pin')
        if pin.is_dir():
            self.assertIn('def get_trainer(\n        self,\n        cfg,\n        tracker,\n        tokenizer,\n        train_dataset,\n        eval_dataset,\n        inference_engine_client,\n'
                          '        generator: GeneratorInterface,\n        colocate_pg,', (pin / 'skyrl/train/entrypoints/main_base.py').read_text())
            utils = (pin / 'skyrl/train/utils/trainer_utils.py').read_text()
            self.assertIn('def build_dataloader(', utils)
            self.assertIn('collate_fn=dataset.collate_fn', utils)

    def test_sampler_is_used_only_when_the_spec_returns_one(self):
        self.assertIsNone(get_environment('gsm8k').train_batch_sampler({}, None, None))
        self.assertIsNone(get_environment('coding_harbor').train_batch_sampler({}, None, None))
        self.assertIsNone(get_environment('coding_exec').train_batch_sampler({}, None, None))


# ---------------------------------------------------------------------------------------------------------------- metrics / media
def png_chunks(blob):
    assert blob[:8] == b'\x89PNG\r\n\x1a\n'
    pos, out = 8, []
    while pos < len(blob):
        size, kind = struct.unpack('>I4s', blob[pos:pos + 8])
        body = blob[pos + 8:pos + 8 + size]
        crc = struct.unpack('>I', blob[pos + 8 + size:pos + 12 + size])[0]
        assert crc == zlib.crc32(kind + body) & 0xFFFFFFFF
        out.append((kind, body))
        pos += 12 + size
    return out


def vlq(value):
    out = [value & 0x7F]
    while value >> 7:
        value >>= 7
        out.append(0x80 | (value & 0x7F))
    return bytes(reversed(out))


def tiny_midi():
    track = b''
    for delta, status, a, b in ((0, 0x90, 60, 64), (96, 0x80, 60, 0), (0, 0x90, 64, 64), (192, 0x80, 64, 0)):
        track += vlq(delta) + bytes([status, a, b])
    track += b'\x00\xff\x2f\x00'
    return b'MThd' + struct.pack('>IHHH', 6, 0, 1, 96) + b'MTrk' + struct.pack('>I', len(track)) + track


ABC = """X:1
T:Test
M:4/4
L:1/8
K:G
V:1
G2 A2 B2 c2 | d4 z4 |
V:2
[GBd]4 [Ace]4 | z8 |
"""


class Media(unittest.TestCase):
    def test_png_encoder_roundtrip(self):
        blob = metrics.encode_png([[(255, 0, 0), (0, 255, 0)], [(0, 0, 255), (10, 20, 30)]])
        chunks = png_chunks(blob)
        self.assertEqual([k for k, _ in chunks], [b'IHDR', b'IDAT', b'IEND'])
        self.assertEqual(struct.unpack('>II', chunks[0][1][:8]), (2, 2))
        raw = zlib.decompress(chunks[1][1])
        self.assertEqual(raw, b'\x00\xff\x00\x00\x00\xff\x00' + b'\x00\x00\x00\xff\x0a\x14\x1e')
        self.assertEqual(judge.png_size(blob), (2, 2))
        with self.assertRaises(ValueError):
            metrics.encode_png([[1, 2], [3, 4]])

    def test_abc_notes(self):
        notes = metrics.notes_from_abc(ABC)
        voice1 = [n for n in notes if n['voice'] == '1']
        self.assertEqual([n['pitch'] for n in voice1], [67, 69, 71, 72, 74])                       # G A B c d (G major: no accidentals on these)
        self.assertEqual([n['start'] for n in voice1], [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(voice1[-1]['dur'], 2.0)                                                   # d4 = four eighths = 2 quarters
        chord = [n for n in notes if n['voice'] == '2' and n['start'] == 0.0]
        self.assertEqual(sorted(n['pitch'] for n in chord), [67, 71, 74])
        self.assertTrue(all(n['dur'] == 2.0 for n in chord))
        self.assertEqual([n['start'] for n in notes if n['voice'] == '2' and n['pitch'] == 69], [2.0])
        sharp = metrics.notes_from_abc('K:D\nF G\n')                                                # D major: F is F#
        self.assertEqual([n['pitch'] for n in sharp], [66, 67])
        natural = metrics.notes_from_abc('K:D\n=F ^G _B,\n')
        self.assertEqual([n['pitch'] for n in natural], [65, 68, 58])
        self.assertEqual(metrics.notes_from_abc('no music here'), [])
        self.assertEqual(metrics.notes_from_abc('M:2/4\nK:C\nC\n')[0]['dur'], 0.25)                # no L: 1/16 for meters < 3/4

    def test_midi_notes(self):
        notes = metrics.notes_from_midi(tiny_midi())
        self.assertEqual([(n['pitch'], n['start'], n['dur']) for n in notes], [(60, 0.0, 1.0), (64, 1.0, 2.0)])
        with self.assertRaises(ValueError):
            metrics.notes_from_midi(b'nope')

    def test_piano_roll(self):
        import numpy as np
        roll = metrics.piano_roll(metrics.notes_from_abc(ABC), width=64, height=32)
        self.assertEqual(roll.shape, (32, 64, 3))
        self.assertEqual(roll.dtype, np.uint8)
        colours = {tuple(px) for row in roll for px in row}
        self.assertIn(metrics.PALETTE[0], colours)
        self.assertIn(metrics.PALETTE[1], colours)                                                 # two voices, two colours
        blank = metrics.piano_roll([], width=8, height=4)
        self.assertEqual(blank.shape, (4, 8, 3))
        self.assertTrue((blank == 24).all())
        png_chunks(metrics.encode_png(roll))

    def test_media_table_picks_screenshots_and_rolls_without_pil(self):
        with tempfile.TemporaryDirectory() as temp:
            shot = Path(temp) / 'screenshot.png'
            shot.write_bytes(metrics.encode_png([[(1, 1, 1)]]))
            records = [{'domain': 'webdev', 'task': 'w1', 'rep': 0, 'task_metric': 0.5, 'screenshot': str(shot)},
                       {'domain': 'webdev', 'task': 'w2', 'rep': 0, 'task_metric': 0.5, 'screenshot': None},
                       {'domain': 'music', 'task': 'm1', 'rep': 0, 'task_metric': 1.0, 'artifact': 'trunc', 'extra': {'artifact_full': ABC}},
                       {'domain': 'code', 'task': 'c1', 'rep': 0, 'task_metric': 1.0}]
            self.assertEqual(metrics.media_for(records[0]), str(shot))
            self.assertEqual(metrics.media_for({'domain': 'music', 'artifact': ABC}).shape[2], 3)
            self.assertIsNone(metrics.media_for(records[3]))
            images = []
            with patch.object(metrics, 'wandb_image', lambda source, caption='': images.append((source, caption)) or f'IMG{len(images)}'):
                columns, rows = metrics.media_table(records, limit=8)
            self.assertEqual(columns, ['domain', 'task', 'rep', 'task_metric', 'image'])
            self.assertEqual([r[1] for r in rows], ['w1', 'm1'])
            self.assertEqual(len(images), 2)
            with patch.object(metrics, 'wandb_image', lambda *a, **k: None):
                self.assertIsNone(metrics.media_table(records))
            many = [{'domain': 'music', 'task': f'm{i}', 'rep': 0, 'task_metric': 1.0, 'artifact': ABC} for i in range(20)]
            with patch.object(metrics, 'wandb_image', lambda *a, **k: 'IMG'):
                self.assertLessEqual(len(metrics.media_table(many)[1]), metrics.MEDIA_ROWS)

    def test_domain_metrics_exclude_masked_and_report_noise(self):
        records = [{'domain': 'code', 'task': 'a', 'task_metric': 1.0, 'passed': True}, {'domain': 'code', 'task': 'a', 'task_metric': 0.0, 'passed': False},
                   {'domain': 'code', 'task': 'b', 'task_metric': 0.0, 'passed': False, 'masked': True},
                   {'domain': 'webdev', 'task': 'w', 'task_metric': 0.8, 'passed': False, 'judge_variance': 0.01},
                   {'domain': 'general', 'task': 'g', 'task_metric': 0.0, 'masked': True}]
        out = metrics.domain_metrics(records)
        self.assertEqual(out['eval/code/task_metric'], 0.5)
        self.assertEqual((out['eval/code/num_trials'], out['eval/code/num_masked']), (3.0, 1.0))
        self.assertEqual(out['eval/code/pass_rate'], 0.5)
        self.assertEqual(out['eval/code/rep_std'], 0.5)
        self.assertEqual(out['eval/webdev/judge_variance'], 0.01)
        self.assertNotIn('eval/general/task_metric', out)                         # all masked: no number, only the count
        self.assertEqual(out['eval/general/num_masked'], 1.0)
        self.assertAlmostEqual(out['eval/macro_task_metric'], (0.5 + 0.8) / 2)

    def test_env_metrics_rows_and_extra_columns(self):
        ok = {'domain': 'code', 'metric': 1.0, 'passed': True, 'masked': False, 'artifact': 'A' * 5000, 'components': {'x': 1}, 'judge': {'rationale': 'why'}, 'screenshot': '/s.png'}
        row = metrics.env_metrics_row(ok)
        self.assertEqual((row['task_metric'], row['failed'], row['infra_error']), (1.0, False, 0.0))
        values = metrics.extra_values(0, [row])
        self.assertEqual(values[0], 'code')
        self.assertEqual(len(values[1]), metrics.ARTIFACT_CHARS)
        self.assertEqual(values[2:], ('why', '{"x": 1}', '/s.png'))
        self.assertEqual(len(values), len(metrics.EXTRA_COLUMNS))
        bad = metrics.env_metrics_row({'domain': 'music', 'metric': 0.1, 'passed': False, 'masked': False, 'stop': 'agent_timeout'})
        self.assertTrue(bad['failed'] and bad['failure/agent_timeout'] == 1.0)
        masked = metrics.env_metrics_row({'domain': 'general', 'masked': True, 'mask_reason': 'judge'})
        self.assertEqual((masked['task_metric'], masked['infra_error'], masked['failure/judge'], masked['failed']), (None, 1.0, 1.0, False))
        self.assertEqual(metrics.extra_values(5, None), (None,) * len(metrics.EXTRA_COLUMNS))


# ---------------------------------------------------------------------------------------------------------------- spec
def spec_cfg(tmp, raw, name=SMOKE, overrides=(), manifest=True):
    cfg = load(name, overrides=list(overrides))
    cfg['run']['data_dir'] = str(Path(tmp) / 'data')
    cfg['environment']['mimo']['raw_dir'] = str(raw)
    cfg['environment']['mimo']['revision'] = 'test'
    for d in cfg['environment']['mimo']['domains'].values():
        d.update(train=4, test=3)
    if manifest:
        get_environment('mimo').prepare_data(cfg, exporter=fake_exporter)
    return cfg


class Spec(DataCase):
    env = get_environment('mimo')

    def test_registered_and_wired(self):
        self.assertEqual((self.env.name, self.env.generator_kind, self.env.metric, self.env.needs_modal), ('mimo', 'harbor', 'macro_task_metric', True))
        self.assertTrue(self.env.needs_litellm and not self.env.uses_coding_executor)

    def test_experiments_plan_and_validate(self):
        from strl.derive import plan
        for path in (SMOKE, LORA, FULL):
            cfg = load(path)
            report = plan(cfg)
            self.assertEqual(report['axes']['environment'].split(' ')[0], 'mimo')
            self.assertIn('MIMO: per-step quota', ' '.join(report['warnings']))
        lora, full = load(LORA), load(FULL)
        self.assertEqual((lora['finetune']['mode'], lora['finetune']['lora']['rank'], lora['finetune']['lora']['alpha']), ('lora', 64, 64))
        self.assertEqual(full['finetune']['mode'], 'full')
        for cfg in (lora, full):
            self.assertEqual((cfg['model']['id'], cfg['model']['revision']), ('Qwen/Qwen3.8-27B', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'))
            self.assertEqual((cfg['compute']['nodes'], cfg['compute']['gpus_per_node'], cfg['algorithm']['prompts_per_step'], cfg['algorithm']['n_samples']), (2, 8, 40, 6))
            self.assertFalse(cfg['algorithm']['use_kl_loss'] or cfg['algorithm']['grpo_norm_by_std'])
            self.assertEqual((cfg['budget']['stall_seconds'], cfg['budget']['canary_max_seconds'], cfg['budget']['attempt_max_seconds']), (3000, 4800, 7200))
            self.assertEqual(cfg['evaluation']['final_eval'], False)
        smoke = load(SMOKE)
        self.assertEqual((smoke['compute']['gpus_per_node'], smoke['algorithm']['prompts_per_step'], smoke['algorithm']['n_samples']), (1, 10, 4))

    def test_validation_messages(self):
        bad = {
            ('reward.components=[{fn: pass_all_tests}]',): 'verifier_score',
            ('evaluation.final_eval=true',): 'final_eval',
            ('environment.mimo.domains.music.train=0', 'environment.mimo.domains.cyber.train=0', 'environment.mimo.domains.code.train=0',
             'environment.mimo.domains.general.train=0', 'environment.mimo.domains.webdev.train=0'): 'at least one domain',
            ('environment.mimo.domains.nope={train: 1}',): 'unknown domain',
            ('environment.mimo.domains.music.weight=0',): 'weight must be > 0',
            ('environment.mimo.domains.music.train=-1',): 'integer >= 0',
            ('environment.mimo.step_deadline_sec=3000',): 'exceeds budget.stall_seconds',
            ('environment.mimo.domains.code.max_agent_sec=1700', 'environment.mimo.domains.code.max_verifier_sec=300'): 'healthy trials would be cancelled and masked',
            ('algorithm.prompts_per_step=4', 'algorithm.minibatch_prompts=4'): 'below the 5 selected domains',
            ('environment.mimo.domains.code.train=1',): 'fewer train tasks than their per-step quota',
            ('environment.harbor.max_seq_len=4000',): 'max_seq_len',
            ('environment.mimo.domains.code.max_prompt_tokens=9000',): 'exceeds algorithm.max_prompt_tokens',
            ('environment.mimo.judge.votes=0',): 'judge.votes',
            ('environment.mimo.timeout_policy=drop',): 'timeout_policy',
            ('environment.mimo.bogus=1',): 'not a known key',
            ('services.litellm={}',): 'services.litellm.key_name',
            ('services.modal.sandbox_secret=""',): 'sandbox_secret',
        }
        for overrides, text in bad.items():
            with self.assertRaises(ExperimentError, msg=overrides) as caught:
                load(LORA, overrides=list(overrides))
            self.assertIn(text, str(caught.exception), overrides)
        load(LORA, overrides=['environment.mimo.domains.general.train=0', 'environment.mimo.domains.general.test=0', 'environment.mimo.domains.webdev.train=0',
                              'environment.mimo.domains.webdev.test=0', 'services.litellm={}'])    # no judged domain selected: no judge needed

    def test_skyrl_overrides_and_data_files_before_prepare(self):
        from strl.overrides import overrides
        cfg = load(LORA)
        values = dict(item.split('=', 1) for item in overrides(cfg, stage='train', run_name='t'))
        self.assertEqual(values['trainer.eval_batch_size'], '240')                  # 5 domains x 24 tasks x 2 repeats
        self.assertEqual(values['trainer.algorithm.grpo_norm_by_std'], 'false')
        self.assertEqual(values['generator.eval_n_samples_per_prompt'], '1')
        self.assertEqual(values['trainer.algorithm.max_seq_len'], '32768')
        self.assertEqual((values['generator.rate_limit.max_concurrency'], values['generator.rate_limit.trajectories_per_second']), ('256', '8.0'))
        self.assertEqual(values['generator.step_wise_trajectories'], 'true')
        self.assertEqual(values['trainer.epochs'], '34')                            # ceil(100 steps / 3 steps per epoch) >= 20
        self.assertEqual(values['generator.inference_engine.engine_init_kwargs.compilation_config.pass_config.fuse_allreduce_rms'], 'false')
        self.assertEqual(values['generator.inference_engine.engine_init_kwargs.trust_remote_code'], 'true')
        self.assertEqual(values['generator.inference_engine.tensor_parallel_size'], '2')
        self.assertIn('/tasks/train', values['data.train_data'])                    # planning before `strl prepare`: split directories
        canary = dict(item.split('=', 1) for item in overrides(cfg, stage='canary', run_name='t'))
        self.assertEqual(canary['trainer.eval_batch_size'], '10')                   # 2 per domain, once
        other = dict(item.split('=', 1) for item in overrides(load('experiments/mbpp-qwen35-binary.yaml'), stage='train', run_name='t'))
        self.assertNotIn('trainer.algorithm.grpo_norm_by_std', other)               # other environments are untouched

    def test_sizes_and_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = spec_cfg(temp, self.raw)
            self.assertEqual((self.env.validation_size(cfg), self.env.canary_size(cfg)), (15, 5))
            ids = self.env.final_eval_ids(cfg)
            self.assertEqual(len(ids), 15)
            self.assertEqual(len(set(ids)), 15)
            self.assertTrue(all('#r' in i for i in ids))
            cfg['environment']['mimo']['eval_repeats'] = 3
            cfg['environment']['mimo']['domains']['webdev']['test'] = 0
            self.assertEqual(self.env.validation_size(cfg), 36)

    def test_data_files_exact_paths_with_repeats(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = spec_cfg(temp, self.raw, overrides=['environment.mimo.eval_repeats=2'])
            train, val = self.env.data_files(cfg, 'train')
            self.assertEqual(len(train), 20)
            self.assertEqual(len(val), 30)
            self.assertEqual(val[:15], val[15:])
            self.assertTrue(all((Path(p) / 'instruction.md').is_file() for p in train + val))
            canary_train, canary_val = self.env.data_files(cfg, 'canary')
            self.assertEqual((len(canary_train), len(canary_val)), (20, 5))
            cfg['environment']['mimo']['domains']['code']['train'] = 2           # --set friendly: prefix of the prepared selection
            self.assertEqual(len(self.env.data_files(cfg, 'train')[0]), 18)
            self.env.assert_manifest(cfg)

    def test_dataset_labels_domains_reps_and_sampler(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = spec_cfg(temp, self.raw, overrides=['environment.mimo.eval_repeats=2'])
            train, val = self.env.data_files(cfg, 'train')
            fake_dataset_module = types.ModuleType('examples.train_integrations.harbor.dataset')

            class HarborTaskDataset:
                def __init__(self, data_files):
                    self.task_paths = [Path(p) for p in data_files]

                def collate_fn(self, items):
                    return items
            fake_dataset_module.HarborTaskDataset = HarborTaskDataset
            from strl.tasks.mimo import spec as spec_mod
            with patch.dict(sys.modules, {'examples': types.ModuleType('examples'), 'examples.train_integrations': types.ModuleType('examples.train_integrations'),
                                          'examples.train_integrations.harbor': types.ModuleType('examples.train_integrations.harbor'),
                                          'examples.train_integrations.harbor.dataset': fake_dataset_module}), \
                    patch.object(spec_mod.coding_harbor, 'config_class', lambda: None), patch.object(spec_mod, '_DATASET_CLASS', []):
                cls = spec_mod.make_dataset_class()
                held_out = cls(val)
                item = held_out[0]
                self.assertEqual(item['env_extras']['domain'], item['env_extras']['data_source'])
                self.assertEqual([held_out[i]['env_extras']['rep'] for i in (0, 15)], [0, 1])
                self.assertEqual(item['uid'], '0')
                self.assertEqual(held_out[15]['uid'], '15')
                self.assertEqual(len(list(held_out)), 30)
                with self.assertRaises(IndexError):
                    held_out[99]
                train_ds = cls(train)
                batch_sampler = self.env.train_batch_sampler(cfg, train_ds, types.SimpleNamespace(trainer=types.SimpleNamespace(train_batch_size=10, seed=1)))
                self.assertEqual(len(batch_sampler), 2)
                batch = next(iter(batch_sampler))
                self.assertEqual(sorted(train_ds.domains[i] for i in batch), sorted(data.DOMAINS * 2))

    def test_eval_records_collapse_and_metrics(self):
        def item(domain, task, rep, reward, mimo, stop='complete'):
            return {'env_extras': {'task_dir': f'/d/{domain}-{task}', 'domain': domain, 'rep': rep, 'mimo': mimo}, 'score': reward, 'stop_reason': stop, 'output_response': 'out'}
        mimo = lambda **kw: {'domain': 'code', 'metric': 1.0, 'passed': True, 'masked': False, 'tokens': 7, 'artifact': 'ART', 'components': {'t': 1}, **kw}  # noqa: E731
        rows = [item('code', 'a', 0, 0.0, mimo()), item('code', 'a', 0, 1.0, mimo()), item('code', 'a', 1, 0.0, mimo(metric=0.0, passed=False)),
                item('webdev', 'w', 0, 0.0, mimo(domain='webdev', masked=True, mask_reason='judge', metric=None, passed=False), stop='error'),
                item('music', 'm', 0, 0.6, mimo(domain='music', metric=0.6, passed=False, judge={'rationale': 'meh'}))]
        records = self.env.collapse_eval_records([self.env.eval_record(r, None, 5) for r in rows])
        self.assertEqual([r['sample_id'] for r in records], ['code-a#r0', 'code-a#r1', 'webdev-w#r0', 'music-m#r0'])
        self.assertEqual((records[0]['reward'], records[0]['turns'], records[0]['correct']), (1.0, 2, True))
        self.assertTrue(records[2]['masked'])
        self.assertEqual(records[3]['judge_rationale'], 'meh')
        self.assertEqual(records[0]['extra'], {'artifact_full': 'ART'})
        out = self.env.eval_metrics(records)
        self.assertEqual(out['eval/code/task_metric'], 0.5)
        self.assertEqual(out['eval/macro_task_metric'], (0.5 + 0.6) / 2)           # webdev: all masked -> not in the macro
        self.assertEqual(out['eval/task_metric'], out['eval/macro_task_metric'])
        self.assertEqual(out['eval/webdev/num_masked'], 1.0)
        # fallback without the generator's summary: reward on the last turn is the metric
        bare = self.env.eval_record({'env_extras': {'task_dir': '/d/music-x', 'rep': 0}, 'score': 0.4, 'stop_reason': 'complete', 'output_response': ''}, None, 0)
        self.assertEqual((bare['domain'], bare['task_metric'], bare['masked']), ('music', 0.4, False))

    def test_metric_by_source_failure_mode_and_critic_modes(self):
        base = [{'domain': 'code', 'task_metric': 0.0}, {'domain': 'code', 'task_metric': 0.5, 'masked': True}, {'domain': 'music', 'task_metric': 0.2}]
        trained = [{'domain': 'code', 'task_metric': 1.0}, {'domain': 'music', 'task_metric': 0.4}, {'domain': 'webdev', 'task_metric': 1.0}]
        by = self.env.metric_by_source(base, trained)
        self.assertEqual(by['code'], {'examples': 1, 'base_task_metric': 0.0, 'trained_task_metric': 1.0, 'delta': 1.0})
        self.assertAlmostEqual(by['macro']['delta'], (1.0 + 0.2) / 2)
        self.assertNotIn('webdev', by)
        self.assertEqual(self.env.failure_mode({'masked': True, 'mask_reason': 'deadline'}), 'deadline')
        self.assertEqual(self.env.failure_mode({'finish_reason': 'length'}), 'truncated')
        self.assertEqual(self.env.failure_mode({'finish_reason': 'complete', 'stop': 'agent_timeout'}), 'agent_timeout')
        self.assertEqual(self.env.failure_mode({'finish_reason': 'complete', 'stop': 'complete'}), 'wrong')
        modes = self.env.critic_modes(load(SMOKE))
        self.assertEqual(modes[-1], 'other')
        self.assertTrue({'invalid_abc', 'no_crash', 'no_patch', 'missed_rubric_item', 'blank_render'} <= set(modes))
        self.assertEqual(len(modes), len(set(modes)))
        self.assertIsNone(get_environment('gsm8k').critic_modes({}))

    def test_generator_contract_validates_real_harbor_trial_configs(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = spec_cfg(temp, self.raw)
            for task in Path(cfg['run']['data_dir']).glob('tasks/*/*'):
                (task / 'strl.json').write_text(json.dumps({**json.loads((task / 'strl.json').read_text()), 'agent': {'name': 'terminus-2'}, 'max_turns': 5}))
            receipt = self.env.generator_contract(cfg)
            self.assertEqual(set(receipt['domains']), set(data.DOMAINS))
            self.assertTrue(receipt['judge_combine_passed'] and receipt['actual_reward_passed'])
            self.assertEqual(receipt['domains']['code']['max_turns'], 5)
            self.assertEqual(receipt['domains']['code']['agent_timeout_sec'], 100)

    def test_plan_warnings_and_dry_run_requirements(self):
        warnings = self.env.plan_warnings(load('experiments/mimo-qwen38-27b-lora.yaml', overrides=['algorithm.grpo_norm_by_std=true']))
        self.assertTrue(any('grpo_norm_by_std is on' in w for w in warnings))
        self.assertTrue(any(w.startswith('JUDGE') for w in warnings))

    def test_cli_prepare_skips_the_coding_executor_for_mimo(self):
        self.assertFalse(self.env.uses_coding_executor)
        self.assertTrue(get_environment('coding_harbor').uses_coding_executor)
        self.assertIn('env.uses_coding_executor', (ROOT / 'strl/cli.py').read_text())

    def test_judge_key_reaches_the_pod_and_grpo_key_exists_in_pinned_skyrl(self):
        self.assertIn('env_spec.needs_litellm', (ROOT / 'strl/worker.py').read_text())      # the worker writes the key file for the environment's judge
        pinned = (ROOT / 'pinned/pinned-config.py').read_text()
        self.assertIn('    grpo_norm_by_std: bool = True', pinned)                        # the key mimo sets (false) is a real SkyRL algorithm option

    def test_env_metrics_from_reads_the_summary_store(self):
        generator.SUMMARIES.clear()
        generator.remember('train', TID('1', 0), {'domain': 'code', 'metric': 1.0, 'passed': True, 'masked': False, 'artifact': 'a', 'components': {}, 'judge': None})
        generator.remember('train', TID('2', 0), {'domain': 'music', 'metric': None, 'passed': False, 'masked': True, 'mask_reason': 'deadline', 'judge': None})
        rows = self.env.env_metrics_from({'trajectory_ids': [TID('1', 0), TID('2', 0), TID('3', 0)]}, 'train')
        self.assertEqual([r and r['domain'] for r in rows], ['code', 'music', None])
        self.assertIsNone(self.env.env_metrics_from({'trajectory_ids': [TID('9', 0)]}, 'train'))
        self.assertIsNone(self.env.env_metrics_from({}, 'train'))
        self.assertEqual(self.env.train_diagnostics([], [], [], 1, rows), {'train/task_metric': 1.0})
        generator.SUMMARIES.clear()


# ---------------------------------------------------------------------------------------------------------------- tracking hooks
@contextmanager
def fake_skyrl_tracking():
    class TrainingCallback:
        pass

    class TrajectoryLogger:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def log(self, **kwargs):
            self.logged = kwargs
    modules = {}
    for name in ('skyrl', 'skyrl.train', 'skyrl.train.utils'):
        modules[name] = types.ModuleType(name)
    modules['skyrl.train.utils.callbacks'] = types.SimpleNamespace(TrainingCallback=TrainingCallback)
    modules['skyrl.train.utils.trajectory_logging'] = types.SimpleNamespace(TrajectoryLogger=TrajectoryLogger)
    with patch.dict(sys.modules, modules):
        yield


class FakeTracker:
    def __init__(self):
        self.logs, self.tables, self.media = [], [], []

    def log(self, payload, step, commit=False):
        self.logs.append(payload)

    def log_samples_to_table(self, key, columns, samples, step):
        self.tables.append((key, list(columns), samples))

    def log_media_table(self, key, columns, rows, step):
        self.media.append((key, columns, rows))


class TrackingHooks(unittest.TestCase):
    env = get_environment('mimo')

    def test_on_eval_end_logs_per_domain_macro_and_media(self):
        from strl import tracking
        with tempfile.TemporaryDirectory() as temp, fake_skyrl_tracking(), patch.dict('os.environ', {'GRPO_EXPECTED_VALIDATION_SIZE': '3', 'GRPO_FIXED_QUALITATIVE': '8'}):
            export = Path(temp) / 'exports'
            dump = export / 'dumped_evals' / 'global_step_0_evals'
            dump.mkdir(parents=True)

            def row(domain, task, rep, score, mimo, stop='complete'):
                return json.dumps({'input_prompt': 'p', 'output_response': 'o', 'score': score, 'stop_reason': stop, 'env_class': None, 'data_source': domain,
                                   'env_extras': {'task_dir': f'/d/{domain}-{task}', 'domain': domain, 'rep': rep, 'mimo': mimo}})
            ok = lambda domain, metric, **kw: {'domain': domain, 'metric': metric, 'passed': metric >= 1, 'masked': False, 'artifact': 'X:1\nK:C\nCDEF|', **kw}  # noqa: E731
            (dump / 'music.jsonl').write_text(row('music', 'm', 0, 0.5, ok('music', 0.5)) + '\n')
            (dump / 'code.jsonl').write_text(row('code', 'c', 0, 1.0, ok('code', 1.0)) + '\n' + row('code', 'c', 1, 0.0, ok('code', 0.0)) + '\n')
            (dump / 'aggregated_results.jsonl').write_text('{}\n')
            callback_class = tracking.make_callback_class()
            callback = callback_class(15)
            callback.env = self.env
            tracker = FakeTracker()
            trainer = types.SimpleNamespace(cfg=types.SimpleNamespace(trainer=types.SimpleNamespace(export_path=str(export))), tracker=tracker, tokenizer=None)
            with patch.object(metrics, 'wandb_image', lambda *a, **k: 'IMG'):
                callback.on_eval_end(trainer, types.SimpleNamespace(global_step=0, metrics={}), None)
            payload = tracker.logs[0]
            self.assertEqual(payload['eval/code/task_metric'], 0.5)
            self.assertEqual(payload['eval/music/task_metric'], 0.5)
            self.assertEqual(payload['eval/macro_task_metric'], 0.5)
            self.assertEqual(payload['eval/task_metric'], 0.5)
            self.assertEqual(payload['eval/delta_from_baseline'], 0.0)
            self.assertEqual(payload['eval/macro_task_metric'], payload['eval/' + self.env.metric])
            key, columns, samples = tracker.tables[0]
            self.assertEqual(key, 'qualitative/fixed_validation')
            self.assertIn('domain', columns)
            self.assertIn('artifact', columns)
            self.assertNotIn('extra', columns)
            self.assertEqual(len(samples), 3)
            self.assertEqual(tracker.media[0][0], 'qualitative/media')
            self.assertEqual(tracker.media[0][1][-1], 'image')
            self.assertTrue((export / 'evaluation_samples.jsonl').is_file())
            # a second eval: delta from the first baseline
            (dump.parent / 'global_step_5_evals').mkdir()
            for f in dump.iterdir():
                (dump.parent / 'global_step_5_evals' / f.name).write_text(f.read_text().replace('"score": 0.5', '"score": 0.7').replace('"metric": 0.5', '"metric": 0.9'))
            callback.on_eval_end(trainer, types.SimpleNamespace(global_step=5, metrics={}), None)
            second = [p for p in tracker.logs if 'eval/macro_task_metric' in p][-1]
            self.assertAlmostEqual(second['eval/macro_task_metric'], (0.5 + 0.9) / 2)
            self.assertAlmostEqual(second['eval/delta_from_baseline'], 0.2)

    def test_default_environments_keep_their_eval_behaviour(self):
        from strl.tasks.base import EnvironmentSpec
        base = EnvironmentSpec()
        self.assertEqual((base.eval_metrics([]), base.media_table([]), base.env_metrics_from({}, 'train'), base.critic_modes({}), base.plan_warnings({})),
                         ({}, None, None, None, []))

    def test_train_logger_uses_rebuilt_env_metrics_for_columns_and_diagnostics(self):
        from strl import tracking
        generator.SUMMARIES.clear()
        for i, (domain, metric) in enumerate([('code', 1.0), ('music', 0.0)]):
            generator.remember('train', TID(str(i), 0), {'domain': domain, 'metric': metric, 'passed': metric >= 1, 'masked': False, 'artifact': f'art{i}',
                                                         'components': {}, 'judge': {'rationale': f'why{i}'}})
        with fake_skyrl_tracking():
            logger_class = tracking.make_trajectory_logger_class()
            logger = logger_class()
            logger.env = self.env
            tracker = FakeTracker()
            tokenizer = types.SimpleNamespace(decode=lambda ids, skip_special_tokens=True: ' '.join(map(str, ids)))
            output = {'response_ids': [[1], [2], [3]], 'rewards': [0.0, 1.0, 0.0], 'stop_reasons': ['complete'] * 3, 'is_last_step': [False, True, True],
                      'trajectory_ids': [TID('0', 0), TID('0', 0), TID('1', 0)]}
            with patch.dict('os.environ', {'GRPO_SAMPLES_PER_PROMPT': '1'}):
                logger.log(tracker=tracker, num_samples=3, prompts=['p0', 'p0', 'p1'], generator_output=output, tokenizer=tokenizer, global_step=1,
                           wandb_key='trajectories/train')
            self.assertEqual(tracker.logs[0]['train/task_metric'], 0.5)                # last-turn rows only: code 1.0, music 0.0
            rows = logger.build_samples(num_samples=3, prompts=['p0', 'p0', 'p1'], response_ids=output['response_ids'], rewards=output['rewards'],
                                        loss_masks=None, num_turns_list=None, tokenizer=tokenizer)
            self.assertEqual([r[-5] for r in rows], ['code', 'code', 'music'])        # the domain column
            self.assertEqual(rows[2][-4], 'art2'[:0] + 'art1')                        # artifact of trajectory 1 on its row
        generator.SUMMARIES.clear()

    def test_critic_taxonomy_and_failed_flag(self):
        from strl.monitor import critic
        critic.reset()
        try:
            self.assertTrue(critic.configure({'enabled': True, 'every_n_steps': 1, 'max_rows': 8}, modes=self.env.critic_modes(load(SMOKE))))
            rows = [metrics.env_metrics_row({'domain': 'code', 'metric': 0.9, 'passed': False, 'masked': False, 'judge': {'rationale': 'close'}}),
                    metrics.env_metrics_row({'domain': 'music', 'metric': 0.1, 'passed': False, 'masked': False, 'judge': {'rationale': 'bad abc'}}),
                    metrics.env_metrics_row({'domain': 'general', 'masked': True, 'mask_reason': 'judge'})]
            queued = critic.submit_failures(1, ['p'] * 3, ['c'] * 3, [0.0] * 3, rows, None)
            self.assertEqual(queued, 1)                                              # only the genuinely failed, unmasked trajectory
            item = critic._S['q'].get_nowait()
            self.assertEqual((item['domain'], item['judge']), ('music', 'bad abc'))
            captured = {}

            def fake_call(messages, **kw):
                captured['system'] = messages[0]['content']
                captured['user'] = messages[1]['content']
                return {'text': json.dumps({'failure_mode': 'invalid_abc', 'rationale': 'r', 'fix_hint': 'f'}), 'usage': {}, 'cost_usd': 0.0}
            row = critic.analyse(item, call=fake_call)
            self.assertEqual(row['critic_mode'], 'invalid_abc')
            self.assertIn('invalid_abc', captured['system'])
            self.assertIn('Domain: music', captured['user'])
            critic._S['modes']['invalid_abc'] += 1
            scalars, _ = critic.drain()
            self.assertEqual(scalars['critic/invalid_abc_frac'], 1.0)
            self.assertIn('critic/blank_render_frac', scalars)
        finally:
            critic.reset()
        critic.reset()
        self.assertEqual(critic.analyse({'source_step': 0, 'idx': 0, 'prompt': 'p', 'completion': 'c'},
                                        call=lambda m, **kw: {'text': '{"failure_mode": "invalid_abc"}', 'usage': {}, 'cost_usd': 0})['critic_mode'], 'other')   # default taxonomy unchanged
        self.assertEqual(critic.SYSTEM, critic.system_prompt(critic.MODES))


# ---------------------------------------------------------------------------------------------------------------- compare
class HonestPower(unittest.TestCase):
    def rows(self, values, domain='code', masked=()):
        return [{'sample_id': f'{domain}-t{i // 2}#r{i % 2}', 'task': f'{domain}-t{i // 2}', 'domain': domain, 'task_metric': v, 'masked': i in masked}
                for i, v in enumerate(values)]

    def test_drops_masked_ids_for_either_model_and_reports_counts(self):
        from strl.monitor.compare import honest_power
        base = self.rows([0, 0, 0, 0, 0, 0, 0, 0], masked={1}) + self.rows([0.2] * 4, 'webdev')
        cand = self.rows([1, 1, 1, 1, 1, 1, 1, 1], masked={6}) + self.rows([0.6] * 4, 'webdev')[:3]
        out = honest_power(base, cand, samples=500)
        self.assertEqual(out['dropped']['total'], 3)                       # code#1 masked in base, code#6 masked in cand, webdev#3 missing
        self.assertEqual(out['dropped']['by_domain'], {'code': 2, 'webdev': 1})
        self.assertEqual(out['domains']['code']['trials'], 6)
        self.assertEqual(out['domains']['code']['delta'], 1.0)
        self.assertAlmostEqual(out['domains']['webdev']['delta'], 0.4)
        self.assertAlmostEqual(out['macro']['delta'], 0.7)
        self.assertEqual(out['macro']['domains'], ['code', 'webdev'])
        self.assertTrue(out['macro']['ci_lower'] <= out['macro']['delta'] <= out['macro']['ci_upper'])
        self.assertTrue(out['domains']['code']['quality_passed'])
        self.assertEqual(out['dropped']['of'], 12)

    def test_resamples_tasks_not_trials_and_is_deterministic(self):
        from strl.monitor.compare import honest_power
        base = self.rows([0.0] * 8)
        cand = self.rows([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])           # per task: 1, 0, 1, 0
        a, b = honest_power(base, cand, samples=2000), honest_power(base, cand, samples=2000)
        self.assertEqual(a, b)
        self.assertEqual(a['domains']['code']['tasks'], 4)
        self.assertEqual(a['domains']['code']['delta'], 0.5)
        self.assertLess(a['domains']['code']['ci_lower'], 0.5)
        self.assertGreater(a['domains']['code']['ci_upper'], 0.5)
        empty = honest_power(self.rows([0.0], masked={0}), self.rows([1.0]))
        self.assertNotIn('macro', empty)
        self.assertEqual(empty['dropped']['total'], 1)


if __name__ == '__main__':
    unittest.main()


class SeamsBetweenGradersAndJudge(unittest.TestCase):
    """Track B's real grader outputs through Track A's judge.combine (found by reading both sides, not by either track's tests)."""

    def test_webdev_request_from_the_real_grader_combines_through_the_vision_formula(self):
        from strl.tasks.mimo import judge
        request = {
            'kind': 'vision', 'base_score': 0.0, 'base_weight': 0.0, 'total_weight': 1.0,
            'items': [{'id': 'webdev_rubric', 'weight': 1.0, 'question': 'full vendored rubric prompt', 'anchor': ''}],
            'images': ['/logs/verifier/screenshot.png'], 'query': 'q', 'rubric': 'webdev_eval_v1'}
        scores = {'visual': 0.6, 'query_fulfillment': 0.9, 'premium_assets': 0.3}
        self.assertAlmostEqual(judge.combine(request, scores), 0.6)   # not "no judge score for items ['webdev_rubric']"

    def test_force_zero_never_scores_positive(self):
        from strl.tasks.mimo import judge
        request = {'kind': 'rubric', 'force_zero': True, 'base_score': 0.9, 'total_weight': 1.0,
                   'items': [{'id': 'a', 'weight': 0.1, 'question': 'q', 'answer': 'x'}]}
        self.assertEqual(judge.combine(request, {'a': 1.0}), 0.0)

    # Through the generator, with the exact request shapes Track B's graders write ------------------------------------------------
    PNG = b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\rIHDR' + (64).to_bytes(4, 'big') * 2 + b'\x08\x02\x00\x00\x00' + b'\x00' * 16

    def test_general_gate_failure_scores_zero_without_a_judge_call(self):
        request = {**RUBRIC, 'force_zero': True, 'base_score': 0.0}
        with tempfile.TemporaryDirectory() as temp:
            chat = FakeChat([])
            env = Env(temp, domain='general', judge_obj=judge.Judge(fast(), chat=chat))
            env.queue = [fake_results(reward=0.0, trial_dir=env.trial_dir(reward_txt='0', details={'metric': 0.0, 'gate_failed': True}, request=request))]
            out, summary = env.go(phase='train')
            self.assertEqual((out['reward'], out['stop_reason'], summary['masked']), (0.0, 'complete', False))   # scored 0, not masked
            self.assertEqual(len(chat.calls), 0)                                                                  # and no judge cost

    def test_webdev_vision_trial_is_scored_not_masked(self):
        request = {'kind': 'vision', 'base_score': 0.0, 'base_weight': 0.0, 'total_weight': 1.0, 'images': ['/logs/verifier/screenshot.png'],
                   'items': [{'id': 'webdev_rubric', 'weight': 1.0, 'question': 'vendored rubric prompt', 'anchor': ''}], 'query': 'a cafe site'}
        with tempfile.TemporaryDirectory() as temp:
            env = Env(temp, domain='webdev', judge_obj=judge.Judge(fast(), chat=FakeChat([VISION_OK])))
            env.queue = [fake_results(reward=0.0, trial_dir=env.trial_dir(reward_txt='0', request=request, files={'screenshot.png': self.PNG}))]
            out, summary = env.go(phase='train')
            self.assertEqual((out['stop_reason'], summary['masked']), ('complete', False))
            self.assertAlmostEqual(out['reward'], (0.6 + 0.9 + 0.3) / 3)   # visual = mean of the five visual keys; MiMo eval formula


class VerifierThatNeverRan(unittest.TestCase):
    """The first graders called /usr/bin/python3, absent from two images: reward.txt stayed 0, no details.json was written, and every
    trial of those domains scored a silent 0 (the nop gate passes on a broken verifier too)."""

    def test_no_details_json_masks_the_trial_loudly(self):
        with tempfile.TemporaryDirectory() as temp:
            env = Env(temp, domain='music')
            env.queue = [fake_results(reward=0.0, trial_dir=env.trial_dir(reward_txt='0', details=None))]
            out, summary = env.go(phase='train')
            self.assertEqual((out['stop_reason'], out['rollout_details'], summary['masked'], summary['mask_reason']), ('error', None, True, 'infra_error'))
            self.assertIn('no details.json', summary['mask_detail'])

    def test_timeouts_without_details_are_still_scored_not_masked(self):
        with tempfile.TemporaryDirectory() as temp:
            env = Env(temp, domain='code')
            env.queue = [fake_results(exc='VerifierTimeoutError', reward=None, trial_dir=env.trial_dir(reward_txt='0', details=None))]
            out, summary = env.go()
            self.assertEqual((out['stop_reason'], summary['masked'], summary['reward']), ('complete', False, 0.0))


class VerifierRanCheck(unittest.TestCase):
    def test_distinguishes_a_correct_zero_from_a_verifier_that_never_ran(self):
        with tempfile.TemporaryDirectory() as temp:
            def trial(details):
                d = Path(temp) / f't{len(list(Path(temp).iterdir()))}'
                (d / 'verifier').mkdir(parents=True)
                (d / 'verifier/reward.txt').write_text('0')
                if details is not None:
                    (d / 'verifier/details.json').write_text(details)
                return d
            self.assertEqual(generator.verifier_ran(trial('{"metric": 0.0, "infra_error": false}')), (True, ''))
            self.assertEqual(generator.verifier_ran(trial(None))[1], 'no details.json')          # reward.txt = 0, nothing else: the old bug
            self.assertEqual(generator.verifier_ran(trial('{}'))[1], 'empty details.json')
            self.assertTrue(generator.verifier_ran(trial('{"infra_error": true, "error": "boom"}'))[1].startswith('infra_error'))
            self.assertIn('unreadable', generator.verifier_ran(trial('{nope'))[1])
            self.assertEqual(generator.verifier_ran(None), (False, 'no trial dir'))


class DeadlineAndEvalValidation(unittest.TestCase):
    def load(self, *overrides):
        from strl import experiment
        return experiment.load('experiments/mimo-qwen38-27b-lora.yaml', overrides=list(overrides))

    def test_uncapped_domains_must_fit_the_step_deadline_using_the_exporters_defaults(self):
        """Review finding 7: only configured caps were checked, so code's real 1200 s + 900 s verifier (> 1800) went unnoticed and healthy long
        trials would have been cancelled and masked as 'deadline'."""
        from strl import experiment
        with self.assertRaises(experiment.ExperimentError) as caught:
            self.load('environment.mimo.domains.code.max_agent_sec=null', 'environment.mimo.domains.code.max_verifier_sec=null')
        self.assertIn('domains.code: agent 1200 s + verifier 900 s', str(caught.exception))
        self.load()   # the shipped config caps code (900 + 600) and passes

    def test_eval_repeats_need_a_stochastic_eval(self):
        from strl import experiment
        with self.assertRaises(experiment.ExperimentError) as caught:
            self.load('algorithm.eval_temperature=0.0')
        self.assertIn('eval_repeats > 1', str(caught.exception))
        self.load('algorithm.eval_temperature=0.0', 'environment.mimo.eval_repeats=1')   # a single greedy eval is fine
