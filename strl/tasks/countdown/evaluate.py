"""Offline greedy evaluation of one model on the Countdown test split (one shard); used only when evaluation.final_eval is true
(>= 2 GPUs). Same CLI as strl.tasks.gsm8k.evaluate: --model --output --rank --world-size [--limit]."""
import argparse
import json
from pathlib import Path

from .scoring import load_truth, score_expression
from ...experiment import current
from ...finalize import write_jsonl
from ... import models


def evaluate(model, output, limit=None, rank=0, world_size=1):
    from datasets import Dataset
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    cfg = current()
    profile = models.detect(cfg['model'])[0]
    a = cfg['algorithm']
    data = Dataset.from_parquet(str(Path(cfg['run']['data_dir']) / 'test.parquet'))
    if limit is not None:
        data = data.select(range(min(limit, len(data))))
    if not 0 <= rank < world_size:
        raise ValueError('Invalid evaluation rank')
    data = data.select(range(rank, len(data), world_size))
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['local_path'])
    prompts = [tokenizer.apply_chat_template(row['prompt'], tokenize=False, add_generation_prompt=True, **profile.chat_kwargs(cfg)) for row in data]
    from ... import offline_engine
    engine, lora = offline_engine.build(cfg, profile, model, max_model_len=a['max_prompt_tokens'] + a['max_generate_length'])
    params = SamplingParams(temperature=a['eval_temperature'], top_p=a['top_p'], max_tokens=a['max_generate_length'], seed=a['seed'])
    results = offline_engine.generate(engine, lora, prompts, params)
    rows = []
    for row, res in zip(data, results, strict=True):
        response = res.outputs[0]
        truth = row['reward_spec']['ground_truth']
        numbers, target = load_truth(truth)
        verdict = score_expression(response.text, numbers, target)
        ok = verdict['correct'] and response.finish_reason == 'stop'
        rows.append({'sample_id': row['extra_info']['sample_id'], 'prompt': row['prompt'], 'completion': response.text, 'ground_truth': truth,
                     'parsed_answer': verdict['parsed'], 'failure_class': verdict['failure_class'], 'correct': bool(ok),
                     'finish_reason': response.finish_reason, 'token_count': len(response.token_ids)})
    write_jsonl(output, rows)
    print(json.dumps({'evaluated_model': str(model), 'examples': len(rows), 'accuracy': sum(r['correct'] for r in rows) / max(1, len(rows))}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=1)
    a = p.parse_args()
    evaluate(a.model, a.output, a.limit, a.rank, a.world_size)


if __name__ == '__main__':
    main()
