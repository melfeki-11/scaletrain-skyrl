"""Regenerate the GSM8K golden fixtures from the VERIFIED code at commit 2916fb9 (+18.04 points on GSM8K test).

Run from the repo root with a Python that has yaml/numpy/boto3 (the same one the tests use):

    python tests/golden/make_gsm8k_golden.py

It extracts the old q35/, config.yaml and stage.py with ``git archive`` (the working tree is never read), then records
  * the SkyRL override lists the old code produced (stages x args),
  * the ScaleTrain job the old stage.py rendered (boto3 stubbed, clock frozen; nothing touches AWS),
  * the old scorer's verdicts on a diverse sample of the real a1 completions (tests/golden/scorer_rows.jsonl).
The new framework must reproduce all of it (tests/test_gsm8k_golden.py). Absolute paths are recorded as <ROOT>.
"""
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

OLD_COMMIT = '2916fb9bf11558f9b1546c5255a868ccb9301dc6'
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARGV_MATRIX = {
    'default': ['--attempt', 'golden-a1'],
    'dry': ['--attempt', 'golden-a1', '--dry-run'],
    'team_borrow': ['--attempt', 'golden-a1', '--team', 'gen_ai.agents_research', '--allow-borrowing'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text, root):
    text = text.replace(str(root), '<ROOT>')
    text = re.sub(r'payload-[0-9a-f]{64}\.tar\.gz', 'payload-<DIGEST>.tar.gz', text)
    return re.sub(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])', '<DIGEST>', text)


def main():
    out = {'old_commit': OLD_COMMIT, 'source_sha256': {}, 'overrides': {}, 'jobs': {}}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        archive = subprocess.run(['git', 'archive', OLD_COMMIT, 'q35', 'config.yaml', 'stage.py', 'evidence/budget-ledger.json'],
                                 cwd=REPO, check=True, capture_output=True).stdout
        subprocess.run(['tar', '-x', '-C', str(root)], input=archive, check=True)
        for rel in ('q35/skyrl.py', 'q35/config.py', 'q35/answers.py', 'q35/worker.py', 'q35/tracking.py', 'stage.py', 'config.yaml'):
            out['source_sha256'][rel] = sha(root / rel)
        # fake assets so stage.py's tar/upload code paths run without the real 1.6 GB model
        (root / 'assets/model').mkdir(parents=True)
        (root / 'assets/model/config.json').write_text('{"golden": true}')
        (root / 'assets/model/model.safetensors').write_bytes(b'golden-weights')
        (root / 'assets/data').mkdir()
        (root / 'assets/data/train.parquet').write_bytes(b'golden-data')
        (root / 'assets/runtime').mkdir()
        (root / 'assets/runtime/wheel.whl').write_bytes(b'golden-wheel')
        (root / 'evidence/budget-ledger.json').write_text(json.dumps({'total_seconds': 3600, 'nodes': 1, 'attempts': []}))

        sys.path.insert(0, str(root))
        os.chdir(root)
        from q35.config import load
        from q35.skyrl import overrides
        cfg = load()
        out['config_resolved'] = json.loads(normalize(json.dumps(cfg, sort_keys=True), root))
        for name, kwargs in {'train': dict(stage='train'), 'canary': dict(stage='canary'),
                             'train_lr_steps': dict(stage='train', learning_rate=1e-6, max_steps=7)}.items():
            out['overrides'][name] = [normalize(v, root) for v in overrides(cfg, run_name='golden-run', **kwargs)]

        calls = []

        class Recorder:
            def __init__(self, region_name=None): self.region = region_name
            def put_object(self, **kw): calls.append(('put_object', kw['Bucket'], normalize(kw['Key'], root)))
            def head_object(self, **kw): raise KeyError('not found')
            def upload_file(self, path, bucket, key, ExtraArgs=None): calls.append(('upload_file', bucket, normalize(key, root)))

        fake = types.ModuleType('boto3')
        fake.Session = lambda profile_name=None: types.SimpleNamespace(client=lambda name, region_name=None: Recorder(region_name))
        fake.client = lambda name, region_name=None: Recorder(region_name)
        sys.modules['boto3'] = fake
        frozen = time.struct_time((2026, 10, 6, 12, 0, 0, 0, 279, 0))
        real_gmtime = time.gmtime
        time.gmtime = lambda *a: frozen if not a else real_gmtime(*a)
        for label, argv in ARGV_MATRIX.items():
            calls.clear()
            job_file = root / f'job-{label}.json'
            sys.argv = ['stage.py', *argv, '--output', str(job_file)]
            import io
            import contextlib
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                runpy.run_path(str(root / 'stage.py'), run_name='__main__')
            job = json.loads(job_file.read_text())
            job['command'] = [normalize(c, root) for c in job['command']]
            out['jobs'][label] = {'job': job, 's3_calls': [list(c) for c in calls],
                                  'stdout': json.loads(normalize(stdout.getvalue().strip().splitlines()[-1], root)),
                                  'runtime_json': json.loads(normalize((root / 'runtime.json').read_text(), root))}
        time.gmtime = real_gmtime

        # scorer verdicts of the OLD code on real completions from the verified run
        from q35.answers import parse_final_answer, score_answer
        rows = []
        for label in ('base', 'candidate'):
            path = REPO / 'evidence/q35v2-a1' / f'{label}.jsonl'
            if not path.is_file():
                print(f'WARNING: {path} missing; scorer golden not regenerated', file=sys.stderr)
                rows = None
                break
            data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for r in data:
                r['_model'] = label
                r['_old_parsed'] = parse_final_answer(r['completion'])
                r['_old_score'] = score_answer(r['completion'], r['ground_truth'], r['finish_reason'])
            buckets = {'invalid': [r for r in data if r['_old_parsed'] is None], 'trunc': [r for r in data if r['finish_reason'] != 'stop'],
                       'right': [r for r in data if r['_old_score'] == 1.0], 'wrong': [r for r in data if r['_old_parsed'] is not None and r['_old_score'] == 0.0]}
            for name, items in buckets.items():
                rows += items[:40]
        if rows is not None:
            seen, unique = set(), []
            for r in rows:
                key = (r['_model'], r['sample_id'])
                if key not in seen:
                    seen.add(key)
                    unique.append({k: r[k] for k in ('_model', 'sample_id', 'completion', 'ground_truth', 'finish_reason', '_old_parsed', '_old_score')})
            (HERE / 'scorer_rows.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in unique))
            out['scorer_rows'] = len(unique)
    (HERE / f'gsm8k_{OLD_COMMIT[:7]}.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'overrides': {k: len(v) for k, v in out['overrides'].items()}, 'jobs': list(out['jobs']), 'scorer_rows': out.get('scorer_rows')}))


if __name__ == '__main__':
    main()
