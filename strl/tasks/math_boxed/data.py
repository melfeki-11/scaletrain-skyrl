"""MATH data: download the pinned HF revision, filter (levels, subjects, problem length), cut fixed held-out sets, write parquet.

- test       = a seeded sample of ``test_size`` problems from the dataset's official TEST split (never used for selection);
- validation = the first ``validation_size`` problems of a seeded shuffle of the filtered TRAIN split (in-loop eval);
- train      = the next ``train_size`` problems of that same shuffle,
so changing ``train_size`` never changes validation or test. The reference answer is the last ``\\boxed{}`` of the worked solution
(problems without one are dropped and counted in the manifest); the solution itself is kept in ``extra_info`` only, never in the prompt.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .scoring import normalize, reference_answer

DEFAULTS = {
    'dataset_id': 'DigitalLearningGmbH/MATH-lighteval',
    'dataset_config': 'default',
    'revision': '0530c78699ea5e8eb5530600900e1f328b48acad',
    'levels': [1, 2, 3, 4, 5],
    'subjects': [],                       # [] = every subject (the dataset's `type`: Algebra, Geometry, ...)
    'max_problem_chars': 1500,            # long problems (Asymptote figures) would exceed algorithm.max_prompt_tokens
    'train_size': 3000,
    'validation_size': 256,
    'test_size': 500,
    'split_seed': 17,
    'prompt_suffix': 'Solve the problem step by step, concisely. Put only your final answer inside \\boxed{}.',
}
NAME = 'math_boxed'


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **((cfg.get('environment') or {}).get(NAME) or {})}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows_sha(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _level(value: Any) -> int | None:
    digits = ''.join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def load_source(s: dict) -> dict:
    from datasets import load_dataset
    return load_dataset(s['dataset_id'], s['dataset_config'], revision=s['revision'])


def eligible(split, s: dict) -> tuple[list[int], dict[str, int]]:
    """Source indices that pass the filters, in source order, and why the others were dropped."""
    levels, subjects = {int(x) for x in s['levels']}, set(s['subjects'] or [])
    keep, dropped = [], {'level': 0, 'subject': 0, 'too_long': 0, 'no_reference': 0}
    for i, row in enumerate(split):
        if _level(row['level']) not in levels:
            dropped['level'] += 1
        elif subjects and row['type'] not in subjects:
            dropped['subject'] += 1
        elif len(row['problem']) > int(s['max_problem_chars']):
            dropped['too_long'] += 1
        elif reference_answer(row['solution']) is None or normalize(reference_answer(row['solution'])) is None:
            dropped['no_reference'] += 1
        else:
            keep.append(i)
    return keep, dropped


def select(source: dict, s: dict) -> tuple[dict[str, list[int]], dict]:
    """split -> source indices (train/validation from the source train split, test from the source test split)."""
    train_ok, train_dropped = eligible(source['train'], s)
    test_ok, test_dropped = eligible(source['test'], s)
    rng = random.Random(int(s['split_seed']))
    shuffled = list(train_ok)
    rng.shuffle(shuffled)
    v, t = int(s['validation_size']), int(s['train_size'])
    if v + t > len(shuffled):
        raise ValueError(f'math_boxed: validation_size + train_size = {v + t} > {len(shuffled)} eligible train problems; lower them or widen '
                         'levels / subjects / max_problem_chars')
    if int(s['test_size']) > len(test_ok):
        raise ValueError(f'math_boxed: test_size {s["test_size"]} > {len(test_ok)} eligible test problems')
    test = sorted(random.Random(int(s['split_seed']) + 1).sample(test_ok, int(s['test_size'])))
    picks = {'test': test, 'validation': sorted(shuffled[:v]), 'train': sorted(shuffled[v:v + t])}
    return picks, {'train_eligible': len(train_ok), 'test_eligible': len(test_ok), 'train_dropped': train_dropped, 'test_dropped': test_dropped}


def make_record(row: dict, *, split: str, index: int, source_split: str, source_index: int, prompt_suffix: str) -> dict[str, Any]:
    truth = reference_answer(row['solution'])
    return {
        'data_source': f'strl/{NAME}',
        'prompt': [{'role': 'user', 'content': f"{row['problem']}\n\n{prompt_suffix}"}],
        'env_class': 'strl_env',
        'reward_spec': {'method': 'rule', 'ground_truth': truth},
        'extra_info': {'sample_id': f'{NAME}-{split}-{index:05d}', 'split': split, 'source_split': source_split,
                       'source_index': source_index, 'level': _level(row['level']), 'subject': row['type'],
                       'ground_truth': truth, 'solution': row['solution']},
    }


def records(cfg: dict, source: dict | None = None) -> tuple[dict[str, list[dict]], dict]:
    s = settings(cfg)
    source = source if source is not None else load_source(s)
    picks, stats = select(source, s)
    out = {}
    for split, indices in picks.items():
        src = 'test' if split == 'test' else 'train'
        out[split] = [make_record(source[src][i], split=split, index=n, source_split=src, source_index=i, prompt_suffix=s['prompt_suffix'])
                      for n, i in enumerate(indices)]
    return out, {**stats, 'source_indices': picks}


def _contract(cfg: dict) -> dict:
    s = settings(cfg)
    return {'settings': s, 'fixed_qualitative_samples': min(cfg['evaluation']['fixed_qualitative_samples'], int(s['validation_size']))}


def prepare(cfg: dict, source: dict | None = None, **_: Any) -> dict:
    from datasets import Dataset
    rows, stats = records(cfg, source)
    q = _contract(cfg)['fixed_qualitative_samples']
    rows['canary'] = rows['validation'][:q]
    root = Path(cfg['run']['data_dir'])
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, items in rows.items():
        path = root / f'{split}.parquet'
        Dataset.from_list(items).to_parquet(str(path))
        files[split] = {'path': f'{split}.parquet', 'rows': len(items), 'sha256': _sha256(path), 'rows_sha256': _rows_sha(items)}
    manifest = {'environment': NAME, **_contract(cfg), 'files': files, 'selection': stats,
                'counts': {split: len(items) for split, items in rows.items()},
                'test_ids': [r['extra_info']['sample_id'] for r in rows['test']],
                'qualitative_ids': [r['extra_info']['sample_id'] for r in rows['canary']],
                'official_test_untouched_by_selection': True}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return manifest


def assert_manifest(cfg: dict) -> dict:
    """The prepared dir matches THIS experiment's settings (dataset revision, filters, sizes, seed) and every file hash (raises otherwise)."""
    root = Path(cfg['run']['data_dir'])
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    expected = _contract(cfg)
    if manifest.get('environment') != NAME or manifest.get('settings') != json.loads(json.dumps(expected['settings'])) \
            or manifest.get('fixed_qualitative_samples') != expected['fixed_qualitative_samples']:
        raise RuntimeError('math_boxed manifest was prepared with different settings: run `strl prepare <exp>`')
    for split, item in manifest['files'].items():
        if _sha256(root / item['path']) != item['sha256']:
            raise RuntimeError(f'math_boxed data hash mismatch: {root / item["path"]}')
    picks = manifest['selection']['source_indices']
    if set(picks['train']) & set(picks['validation']):
        raise RuntimeError('math_boxed train and validation overlap')
    return manifest
