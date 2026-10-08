"""Pure helpers for reassembling and comparing the sharded final test evaluation."""
import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows: Iterable[dict]):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(''.join(json.dumps(row, sort_keys=True, default=str) + '\n' for row in rows), encoding='utf-8')


def merge_shards(shards, expected_ids):
    """Reassemble per-shard rows in the canonical order; each expected ID must appear exactly once."""
    rows = [row for shard in shards for row in shard]
    by_id = {}
    for row in rows:
        if row['sample_id'] in by_id:
            raise RuntimeError('Duplicate evaluation sample: ' + row['sample_id'])
        by_id[row['sample_id']] = row
    if len(rows) != len(expected_ids) or set(by_id) != set(expected_ids):
        raise RuntimeError('Evaluation coverage mismatch')
    return [by_id[i] for i in expected_ids]


def flips(base, trained):
    if [r['sample_id'] for r in base] != [r['sample_id'] for r in trained]:
        raise RuntimeError('Paired evaluation IDs differ')
    out = {'fixed': 0, 'broken': 0, 'both_correct': 0, 'both_wrong': 0}
    for b, t in zip(base, trained):
        if b['correct'] and t['correct']: out['both_correct'] += 1
        elif b['correct']: out['broken'] += 1
        elif t['correct']: out['fixed'] += 1
        else: out['both_wrong'] += 1
    return out


def failure_mode(row):
    """GSM8K failure taxonomy (default for backward compatibility); environments pass their own via mode_fn."""
    if row.get('finish_reason') != 'stop': return 'truncated'
    if row.get('parsed_answer') is None: return 'no_final_answer_line'
    return 'wrong_answer'


def select_examples(base, trained, kind='fixed', limit=50, mode_fn=failure_mode):
    """Rows the base got wrong and the trained model got right (kind='fixed'), or the reverse ('broken')."""
    want = (False, True) if kind == 'fixed' else (True, False)
    chosen = []
    for b, t in zip(base, trained):
        if (bool(b['correct']), bool(t['correct'])) == want:
            chosen.append({'sample_id': b['sample_id'], 'prompt': b['prompt'], 'ground_truth': b['ground_truth'],
                           'base_completion': b['completion'], 'base_parsed': b.get('parsed_answer'),
                           'base_failure_mode': mode_fn(b) if kind == 'fixed' else None,
                           'trained_completion': t['completion'], 'trained_parsed': t.get('parsed_answer')})
    return chosen[:limit]
