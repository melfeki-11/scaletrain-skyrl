"""Deterministic Countdown problem generation (no download): a seeded RNG builds a random expression over a subset of the numbers,
so every problem is solvable by construction; the reference solution is kept in ``extra_info`` (never in the prompt).

One unique pool (dedup key: sorted numbers + target) is generated from ``environment.countdown.seed`` and cut in a fixed order:
test first, then validation, then train, so changing ``train_size`` never changes the held-out sets.
"""
from __future__ import annotations

import hashlib
import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

from .scoring import evaluate

DEFAULTS = {'seed': 20261009, 'train_size': 512, 'validation_size': 100, 'test_size': 200, 'min_numbers': 3, 'max_numbers': 5,
            'max_number': 25, 'big_numbers': [25, 50, 75, 100], 'big_number_prob': 0.15, 'min_target': 10, 'max_target': 999,
            'prompt': None}   # None: PROMPT below; else a template with {numbers}, {target} and the <answer></answer> instruction
PROMPT = ('Using the numbers {numbers}, write an arithmetic expression that equals {target}. You may use +, -, *, / and parentheses; '
          'each number may be used at most once (you do not have to use all of them). Think briefly, then give only the expression '
          'inside <answer></answer>, for example <answer>(1 + 2) * 3</answer>.')
SPLITS = ('test', 'validation', 'train')     # cut order of the pool


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **((cfg.get('environment') or {}).get('countdown') or {})}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows_sha(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _combine(rng: random.Random, a: tuple[Fraction, str], b: tuple[Fraction, str]) -> tuple[Fraction, str] | None:
    (x, ex), (y, ey) = a, b
    op = rng.choice('+-*/')
    if op == '-' and x < y:
        (x, ex), (y, ey) = (y, ey), (x, ex)
    if op == '/' and (y == 0 or x % y or y == 1):
        return None
    value = {'+': x + y, '-': x - y, '*': x * y, '/': x / y}[op]
    if value <= 0 or value.denominator != 1:
        return None
    return value, f'({ex} {op} {ey})'


def make_problem(rng: random.Random, s: dict) -> dict | None:
    k = rng.randint(s['min_numbers'], s['max_numbers'])
    numbers = [rng.choice(s['big_numbers']) if rng.random() < s['big_number_prob'] else rng.randint(1, s['max_number']) for _ in range(k)]
    used = rng.sample(range(k), rng.randint(max(2, k - 2), k))    # a solution uses most (not always all) of the numbers
    terms = [(Fraction(numbers[i]), str(numbers[i])) for i in used]
    while len(terms) > 1:
        i, j = rng.sample(range(len(terms)), 2)
        merged = _combine(rng, terms[i], terms[j])
        if merged is None:
            return None
        terms = [t for n, t in enumerate(terms) if n not in (i, j)] + [merged]
    value, expr = terms[0]
    target = int(value)
    if not s['min_target'] <= target <= s['max_target'] or target in numbers:
        return None
    expr = expr[1:-1] if expr.startswith('(') and expr.endswith(')') else expr
    assert evaluate(expr)[0] == target
    return {'numbers': numbers, 'target': target, 'solution': expr}


def generate(cfg: dict) -> dict[str, list[dict]]:
    """split -> problems (deterministic for a given settings dict)."""
    s = settings(cfg)
    rng = random.Random(int(s['seed']))
    need = {split: int(s[f'{split}_size']) for split in SPLITS}
    pool, seen, tries = [], set(), 0
    while len(pool) < sum(need.values()):
        tries += 1
        if tries > 200 * sum(need.values()) + 10000:   # settings that cannot yield enough unique problems must fail, not hang
            raise ValueError(f'countdown settings produced only {len(pool)} unique problems of the {sum(need.values())} requested: widen '
                             'max_number / big_numbers / the target range or lower the split sizes')
        problem = make_problem(rng, s)
        if problem is None:
            continue
        key = (tuple(sorted(problem['numbers'])), problem['target'])
        if key not in seen:
            seen.add(key)
            pool.append(problem)
    out, start = {}, 0
    for split in SPLITS:
        out[split] = pool[start:start + need[split]]
        start += need[split]
    return out


def make_record(problem: dict, *, split: str, index: int, template: str | None = None) -> dict[str, Any]:
    truth = json.dumps({'numbers': problem['numbers'], 'target': problem['target']}, sort_keys=True)
    return {
        'data_source': 'strl/countdown',
        'prompt': [{'role': 'user', 'content': (template or PROMPT).format(numbers=problem['numbers'], target=problem['target'])}],
        'env_class': 'strl_env',
        'reward_spec': {'method': 'rule', 'ground_truth': truth},
        'extra_info': {'sample_id': f'countdown-{split}-{index:05d}', 'split': split, 'numbers': problem['numbers'],
                       'target': problem['target'], 'solution': problem['solution'], 'ground_truth': truth},
    }


def records(cfg: dict) -> dict[str, list[dict]]:
    template = settings(cfg)['prompt']
    return {split: [make_record(p, split=split, index=i, template=template) for i, p in enumerate(problems)]
            for split, problems in generate(cfg).items()}


def _contract(cfg: dict) -> dict:
    s = settings(cfg)
    return {'settings': s, 'fixed_qualitative_samples': min(cfg['evaluation']['fixed_qualitative_samples'], int(s['validation_size']))}


def prepare(cfg: dict, **_: Any) -> dict:
    from datasets import Dataset
    rows = records(cfg)
    q = _contract(cfg)['fixed_qualitative_samples']
    rows['canary'] = rows['validation'][:q]
    root = Path(cfg['run']['data_dir'])
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, items in rows.items():
        path = root / f'{split}.parquet'
        Dataset.from_list(items).to_parquet(str(path))
        files[split] = {'path': f'{split}.parquet', 'rows': len(items), 'sha256': _sha256(path), 'rows_sha256': _rows_sha(items)}
    manifest = {'environment': 'countdown', **_contract(cfg), 'files': files,
                'counts': {split: len(items) for split, items in rows.items()},
                'test_ids': [r['extra_info']['sample_id'] for r in rows['test']],
                'qualitative_ids': [r['extra_info']['sample_id'] for r in rows['canary']]}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return manifest


def assert_manifest(cfg: dict) -> dict:
    """The prepared dir matches THIS experiment's generation settings and every file hash (raises otherwise)."""
    root = Path(cfg['run']['data_dir'])
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    expected = _contract(cfg)
    if manifest.get('environment') != 'countdown' or manifest.get('settings') != json.loads(json.dumps(expected['settings'])) \
            or manifest.get('fixed_qualitative_samples') != expected['fixed_qualitative_samples']:
        raise RuntimeError('countdown manifest was prepared with different settings: run `strl prepare <exp>`')
    for split, item in manifest['files'].items():
        if _sha256(root / item['path']) != item['sha256']:
            raise RuntimeError(f'countdown data hash mismatch: {root / item["path"]}')
    return manifest
