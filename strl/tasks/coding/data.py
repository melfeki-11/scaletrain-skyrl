"""Canonical coding tasks from open datasets -> SkyRL parquet + Harbor task dirs + a hash manifest.

    environment:
      name: coding_exec
      sources:
        train:      [{dataset: mbpp, splits: [train, validation]}]
        validation: {from: train, size: 96, seed: 17}          # train-derived; the test sets are never used for selection
        test:       [{dataset: mbpp, splits: [test]}, {dataset: humaneval, splits: [test]}]

Every task's reference solution must pass its tests in the Modal executor 3x and an AST stub (every function returns None)
must fail (non-vacuous tests); otherwise the task is dropped and listed in the manifest. Train tasks whose normalized text or
reference code duplicates a test task are dropped. MBPP prompts show ONE example test while all tests are graded (hidden tests
limit answer hard-coding).
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DATASETS = {
    'mbpp': {'hf': 'google-research-datasets/mbpp', 'config': 'full', 'revision': '4bb6404fdc6cacfda99d4ac4205087b89d32030c', 'license': 'cc-by-4.0'},
    'humaneval': {'hf': 'openai/openai_humaneval', 'config': None, 'revision': '7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544', 'license': 'mit'},
}
INSTRUCTION = 'Return only the complete Python solution in a single ```python code block.'


@dataclass
class CodingTask:
    sample_id: str
    source: str
    split: str
    task_id: str
    prompt: str
    tests: list[str]
    setup: str = ''
    entry_point: str | None = None
    reference: str = ''
    license: str = ''
    revision: str = ''
    meta: dict = field(default_factory=dict)

    def job(self, code: str) -> dict:
        return {'task_id': self.sample_id, 'code': code, 'tests': self.tests, 'setup': self.setup}


def _imports(text: str) -> str:
    return '\n'.join(line for line in text.splitlines() if re.match(r'^(from\s+\S+\s+import|import)\s', line))


def mbpp_task(row: dict, split: str) -> CodingTask:
    prompt = (f"{row['text'].strip()}\n\nYour code must pass this example test (additional hidden tests are also run):\n"
              f"{row['test_list'][0]}\n\n{INSTRUCTION}")
    return CodingTask(sample_id=f'mbpp-{split}-{int(row["task_id"]):04d}', source='mbpp', split=split, task_id=str(row['task_id']),
                      prompt=prompt, tests=list(row['test_list']), setup=row.get('test_setup_code') or '',
                      reference=row['code'], license=DATASETS['mbpp']['license'], revision=DATASETS['mbpp']['revision'])


def humaneval_task(row: dict, split: str) -> CodingTask:
    prompt = ('Complete the following Python function. Return the complete function (signature, docstring is optional, and any '
              f"imports it needs) in a single ```python code block.\n\n```python\n{row['prompt']}```")
    number = int(row['task_id'].split('/')[-1])
    return CodingTask(sample_id=f'humaneval-{split}-{number:04d}', source='humaneval', split=split, task_id=row['task_id'], prompt=prompt,
                      tests=[row['test'] + f"\ncheck({row['entry_point']})\n"], setup=_imports(row['prompt']), entry_point=row['entry_point'],
                      reference=row['prompt'] + row['canonical_solution'], license=DATASETS['humaneval']['license'],
                      revision=DATASETS['humaneval']['revision'])


def load_source(dataset: str, splits: list[str]) -> list[CodingTask]:
    from datasets import load_dataset
    spec = DATASETS[dataset]
    data = load_dataset(spec['hf'], spec['config'], revision=spec['revision']) if spec['config'] else load_dataset(spec['hf'], revision=spec['revision'])
    build = mbpp_task if dataset == 'mbpp' else humaneval_task
    return [build(dict(row), split) for split in splits for row in data[split]]


def _parse(code: str):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)   # MBPP references contain invalid escape sequences
        return ast.parse(code)


def stub(code: str) -> str:
    """Same signatures, every function body replaced by ``return None`` (tests must reject it)."""
    try:
        tree = _parse(code)
    except SyntaxError:
        return ''
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = [ast.Return(value=ast.Constant(value=None))]
    return ast.unparse(tree)


def norm_text(text: str) -> str:
    return re.sub(r'\W+', ' ', text.lower()).strip()


def code_hash(code: str) -> str:
    try:
        return hashlib.sha256(ast.dump(_parse(code)).encode()).hexdigest()
    except SyntaxError:
        return hashlib.sha256(code.encode()).hexdigest()


def validate_references(tasks: list[CodingTask], score_many) -> tuple[list[CodingTask], dict[str, str]]:
    """Keep tasks whose reference passes 3x and whose stub fails; returns (kept, dropped id -> reason)."""
    jobs = [t.job(t.reference) for t in tasks for _ in range(3)] + [t.job(stub(t.reference)) for t in tasks]
    results = score_many(jobs)
    ref, stubs = results[:3 * len(tasks)], results[3 * len(tasks):]
    kept, dropped = [], {}
    for i, task in enumerate(tasks):
        runs = ref[3 * i:3 * i + 3]
        if any(r['failure_class'] == 'infra_error' for r in runs + [stubs[i]]):
            raise RuntimeError(f'executor infrastructure error while validating {task.sample_id}: {runs}')
        if not all(r['failure_class'] == 'ok' for r in runs):
            dropped[task.sample_id] = 'reference fails: ' + ','.join(r['failure_class'] for r in runs) + f' ({runs[0].get("detail")})'[:200]
        elif stubs[i]['failure_class'] == 'ok':
            dropped[task.sample_id] = 'vacuous tests: an all-None stub passes'
        else:
            kept.append(task)
    return kept, dropped


def to_row(task: CodingTask, env_class: str = 'strl_env') -> dict:
    return {
        'data_source': task.source,
        'prompt': [{'role': 'user', 'content': task.prompt}],
        'env_class': env_class,
        'reward_spec': {'method': 'exec', 'ground_truth': json.dumps({'tests': task.tests, 'setup': task.setup, 'entry_point': task.entry_point})},
        'extra_info': {'sample_id': task.sample_id, 'task_id': task.task_id, 'source': task.source, 'split': task.split,
                       'entry_point': task.entry_point or ''},
    }


def _token_count(tokenizer, prompt: str, chat: dict) -> int:
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True, return_dict=False, **chat)
    if isinstance(ids, dict) or hasattr(ids, 'keys'):   # transformers 5 may still return a BatchEncoding
        ids = ids['input_ids']
    return len(ids)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(cfg: dict, *, score_many=None, tokenizer=None, harbor: bool = True) -> dict:
    """Build every split, validate references in Modal, check prompt lengths, write parquet + manifest (+ Harbor dirs)."""
    from datasets import Dataset
    from . import executor, harness
    env, out = cfg['environment'], Path(cfg['run']['data_dir'])
    sources = env['sources']
    score_many = score_many or executor.score_many
    train = [t for s in sources['train'] for t in load_source(s['dataset'], s['splits'])]
    test = [t for s in sources['test'] for t in load_source(s['dataset'], s['splits'])]
    test_text = {norm_text(t.prompt.split('Your code must pass')[0]) for t in test}
    test_code = {code_hash(t.reference) for t in test}
    dropped: dict[str, str] = {}
    unique = []
    for t in train:
        if norm_text(t.prompt.split('Your code must pass')[0]) in test_text or code_hash(t.reference) in test_code:
            dropped[t.sample_id] = 'duplicates a test task'
        else:
            unique.append(t)
    train, drop_train = validate_references(unique, score_many)
    test, drop_test = validate_references(test, score_many)
    dropped.update(drop_train)
    dropped.update(drop_test)
    v = sources['validation']
    rng = random.Random(v.get('seed', 17))
    order = sorted(train, key=lambda t: t.sample_id)
    rng.shuffle(order)
    validation = sorted(order[:v['size']], key=lambda t: t.sample_id)
    train = sorted(order[v['size']:], key=lambda t: t.sample_id)
    for t in validation:
        t.split = 'validation'
    canary = validation[:cfg['evaluation']['fixed_qualitative_samples']]
    splits = {'train': train, 'validation': validation, 'canary': canary, 'test': test}
    limit = cfg['algorithm']['max_prompt_tokens']
    lengths = {}
    if tokenizer is not None:
        from ... import models
        chat = models.detect(cfg['model'])[0].chat_kwargs(cfg)
        for name, rows in splits.items():
            counts = [_token_count(tokenizer, t.prompt, chat) for t in rows]
            lengths[name] = {'max': max(counts), 'p95': sorted(counts)[int(0.95 * (len(counts) - 1))], 'over_limit': sum(c > limit for c in counts)}
            if lengths[name]['over_limit'] and name != 'test':   # test prompts go to offline vLLM (sized from this max), never SkyRL
                raise RuntimeError(f'{lengths[name]["over_limit"]} {name} prompts exceed algorithm.max_prompt_tokens={limit} '
                                   f'(max {lengths[name]["max"]}); SkyRL would drop them silently. Raise max_prompt_tokens.')
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in splits.items():
        path = out / f'{name}.parquet'
        Dataset.from_list([to_row(t) for t in rows]).to_parquet(str(path))
        files[name] = {'path': path.name, 'rows': len(rows), 'sha256': _sha256(path)}
        (out / f'{name}.tasks.jsonl').write_text(''.join(json.dumps(asdict(t), sort_keys=True) + '\n' for t in rows))
    manifest = {'environment': env['name'], 'sources': sources, 'datasets': {k: DATASETS[k] for k in {s['dataset'] for g in ('train', 'test') for s in sources[g]}},
                'files': files, 'dropped': dropped, 'executor_version': harness.VERSION, 'prompt_tokens': lengths,
                'qualitative_ids': [t.sample_id for t in canary], 'test_ids': [t.sample_id for t in test],
                'counts': {k: len(v) for k, v in splits.items()}}
    if harbor:
        from .harbor import export_tasks
        manifest['harbor'] = export_tasks({'train': train, 'validation': validation, 'canary': canary, 'test': test}, out / 'harbor')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def assert_manifest(cfg: dict) -> dict:
    out = Path(cfg['run']['data_dir'])
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['environment'] != cfg['environment']['name'] and {manifest['environment'], cfg['environment']['name']} != {'coding_exec', 'coding_harbor'}:
        raise RuntimeError('coding manifest was built for a different environment')
    if manifest['sources'] != cfg['environment']['sources']:
        raise RuntimeError('coding manifest sources differ from the experiment (re-run prepare)')
    for item in manifest['files'].values():
        if _sha256(out / item['path']) != item['sha256']:
            raise RuntimeError(f'coding data hash mismatch: {item["path"]}')
    return manifest


def load_tasks(cfg: dict, split: str) -> list[CodingTask]:
    path = Path(cfg['run']['data_dir']) / f'{split}.tasks.jsonl'
    return [CodingTask(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def task_from_extras(extras: dict) -> dict[str, Any]:
    """Tests/setup for one rollout from its SkyRL env_extras (reward_spec.ground_truth is the JSON written by to_row)."""
    spec = json.loads((extras.get('reward_spec') or {})['ground_truth'])
    info = extras.get('extra_info') or {}
    return {'task_id': info.get('sample_id'), 'tests': spec['tests'], 'setup': spec.get('setup') or '', 'entry_point': spec.get('entry_point')}
