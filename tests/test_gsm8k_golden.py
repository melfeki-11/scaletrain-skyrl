"""The refactored framework must reproduce the GPU-VERIFIED GSM8K behaviour of commit 2916fb9 (+18.04 points).

Fixtures were generated from that commit by tests/golden/make_gsm8k_golden.py (git archive; boto3 stubbed; clock frozen).
Every allowed difference is listed explicitly in NORMALIZE; anything else fails.
"""
import json
import re
import shutil
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strl.experiment import REPO_ROOT, load  # noqa: E402
from strl.overrides import overrides  # noqa: E402

GOLDEN = json.loads((ROOT / 'tests/golden/gsm8k_2916fb9.json').read_text())
TMP = ROOT / 'tests/.tmp'


def norm_paths(values):
    # intentional difference since 2026-10-08: one model directory per id@revision (assets/models/<name>@<rev12>), not a shared assets/model
    return [re.sub(r'<ROOT>/assets/models/[^/@]+@[0-9a-f]{12}', '<ROOT>/assets/model', v.replace(str(REPO_ROOT), '<ROOT>')) for v in values]


def norm_digest(text):
    text = re.sub(r'payload-[0-9a-f]{64}\.tar\.gz', 'payload-<DIGEST>.tar.gz', text)
    return re.sub(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])', '<DIGEST>', text)


class Recorder:
    def __init__(self):
        self.calls = []

    def put_object(self, **kw):
        self.calls.append(('put_object', kw['Bucket'], kw['Key']))

    def head_object(self, **kw):
        raise KeyError('missing')

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.calls.append(('upload_file', bucket, key))


class GoldenOverrides(unittest.TestCase):
    def test_overrides_identical_for_every_stage(self):
        cfg = load('base/gsm8k.yaml')
        for name, kwargs in {'train': dict(stage='train'), 'canary': dict(stage='canary'),
                             'train_lr_steps': dict(stage='train', learning_rate=1e-6, max_steps=7)}.items():
            new = norm_paths(overrides(cfg, run_name='golden-run', **kwargs))
            self.assertEqual(sorted(new), sorted(GOLDEN['overrides'][name]), name)

    def test_resolved_values_match_legacy_config(self):
        cfg, old = load('base/gsm8k.yaml'), GOLDEN['config_resolved']
        pairs = [(cfg['model']['id'], old['model']['id']), (cfg['model']['revision'], old['model']['revision']),
                 (cfg['environment']['dataset'], old['dataset']), (cfg['compute']['image'], old['runtime']['image']),
                 (cfg['runtime']['skyrl_commit'], old['runtime']['skyrl_commit']), (cfg['tracking']['credentials_uri'], old['wandb']['credentials_uri']),
                 (cfg['budget']['final_eval_reserve_seconds'], old['budget']['final_eval_reserve_seconds']),
                 (cfg['evaluation']['eval_shards_per_model'], old['evaluation']['eval_shards_per_model'])]
        for new, legacy in pairs:
            self.assertEqual(new, legacy)


class GoldenStage(unittest.TestCase):
    # allowed, intentional differences between the old and new stager
    NORMALIZE = [
        ('/workspace/strl-run', '/workspace/qwen35-grpo'),                     # pod working directory renamed
        ('-m strl.worker', '-m q35.worker'),                                   # entry module moved
    ]

    def setUp(self):
        shutil.rmtree(TMP, ignore_errors=True)
        (TMP / 'model').mkdir(parents=True)
        (TMP / 'model/config.json').write_text(json.dumps(json.loads((ROOT / 'tests/fixtures/configs/qwen3_5.json').read_text())))
        (TMP / 'model/model.safetensors').write_bytes(b'golden-weights')
        (TMP / 'data').mkdir()
        (TMP / 'data/train.parquet').write_bytes(b'golden-data')
        (TMP / 'ledger.json').write_text(json.dumps({'total_seconds': 3600, 'nodes': 1, 'attempts': []}))
        (TMP / 'exp.yaml').write_text('extends: base/gsm8k.yaml\nmodel: {local_path: tests/.tmp/model}\nrun: {data_dir: tests/.tmp/data}\n')

    def tearDown(self):
        shutil.rmtree(TMP, ignore_errors=True)

    def normalize_command(self, text):
        lines = [line for line in text.split('\n') if not line.startswith('export STRL_MODE=')]
        text = '\n'.join(lines)
        for new, old in self.NORMALIZE:
            text = text.replace(new, old)
        text = text.replace("'tests/.tmp/model'", "'assets/model'")              # model dir is per-experiment now
        text = re.sub(r"tmp/mohamedelfeki/scaletrain/qwen35-grpo-v2/models/Qwen3\.5-0\.8B@[0-9a-f]{12}/",
                      'tmp/mohamedelfeki/scaletrain/qwen35-grpo-v2/model/', text)  # models stored per id@revision
        return norm_digest(text)

    def test_jobs_identical_up_to_documented_normalization(self):
        from strl.stage import stage
        argv = {'default': dict(mode='train'), 'dry': dict(mode='dry'),
                'team_borrow': dict(mode='train', team='gen_ai.agents_research', allow_borrowing=True)}
        frozen = time.struct_time((2026, 10, 6, 12, 0, 0, 0, 279, 0))
        for label, kwargs in argv.items():
            recorder = Recorder()
            receipt = stage(str(TMP / 'exp.yaml'), attempt='golden-a1', ledger=str(TMP / 'ledger.json'), output=str(TMP / f'{label}.json'),
                            s3=recorder, now=frozen, **kwargs)
            job = json.loads((TMP / f'{label}.json').read_text())
            golden = GOLDEN['jobs'][label]
            want = golden['job']
            # intentional difference since 2026-10-07: the ScaleTrain runbook (SKILLS.md) makes borrowing mandatory on every job
            self.assertTrue(job['allow_borrowing'], label)
            self.assertEqual({k: v for k, v in job.items() if k != 'command'}, {k: v for k, v in want.items() if k != 'command'} | {'allow_borrowing': True}, label)
            self.assertEqual(job['command'][:2], want['command'][:2])
            self.assertEqual(self.normalize_command(job['command'][2]), want['command'][2].replace('<ROOT>', str(REPO_ROOT)), label)
            runtime = json.loads((Path(receipt['staging_dir']) / 'runtime.json').read_text())
            for key in ('artifact_uri', 'budget_ledger_uri', 'remaining_seconds', 'total_seconds', 'run_name'):
                self.assertEqual(runtime[key], golden['runtime_json'][key], (label, key))
            self.assertEqual(receipt['remaining_seconds'], golden['stdout']['remaining_seconds'])
            self.assertEqual(receipt['worker_seconds'], golden['stdout']['worker_seconds'])
            self.assertEqual(recorder.calls[0], tuple(golden['s3_calls'][0]))      # write-once attempt ledger copy first
            resolved = json.loads((Path(receipt['staging_dir']) / 'resolved.json').read_text())
            self.assertNotIn(str(REPO_ROOT), json.dumps(resolved))                   # pod paths only
            self.assertTrue(resolved['model']['local_path'].startswith('/workspace/strl-run/'))


class GoldenScorer(unittest.TestCase):
    """Old vs new scorers on 266 real a1 completions (invalid, truncated, right, wrong; base and trained)."""

    def setUp(self):
        self.rows = [json.loads(line) for line in (ROOT / 'tests/golden/scorer_rows.jsonl').read_text().splitlines()]
        self.assertGreater(len(self.rows), 200)

    def test_offline_scorer_identical(self):
        from strl.tasks import get_environment
        env = get_environment('gsm8k')
        for r in self.rows:
            self.assertEqual(env.parse(r['completion']), r['_old_parsed'], r['sample_id'])
            self.assertEqual(float(env.score(r['completion'], r['ground_truth'], r['finish_reason'])), r['_old_score'], r['sample_id'])

    def test_generic_reward_path_matches_verified_reward_path(self):
        """strl_env (custom reward designs) gives the same strict reward as the patched SkyRL verifier on every row."""
        from strl.rewards import Reward, RewardContext
        from strl.tasks import get_environment
        from strl.tasks.gsm8k.reward_patch import BODY
        namespace = {}
        exec(BODY, namespace)  # the exact source written into skyrl-gym by patch_reward_verifier
        env = get_environment('gsm8k')
        reward = Reward(env, env.default_reward)
        for r in self.rows:
            if r['finish_reason'] != 'stop':
                continue  # zero_reward_on_non_stop zeroes these in both paths before the env is consulted
            verified = namespace['compute_score'](r['completion'], r['ground_truth'])
            generic, _ = reward(env.score_result(r['completion'], {'reward_spec': {'ground_truth': r['ground_truth']}}, {}),
                                RewardContext(ground_truth=r['ground_truth']))
            self.assertEqual(generic, verified, r['sample_id'])


if __name__ == '__main__':
    unittest.main()
