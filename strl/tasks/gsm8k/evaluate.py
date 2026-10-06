"""Offline greedy evaluation of one model on the GSM8K official test split (one shard). Model-agnostic: vLLM and chat-template
kwargs come from the model profile; decoding and token limits from the experiment (identical for base and trained)."""
import argparse
import json
from pathlib import Path

from .answers import parse_final_answer, score_answer
from ...experiment import current
from ...finalize import write_jsonl
from ... import models


def evaluate(model, output, limit=None, rank=0, world_size=1):
    from datasets import Dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
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
    engine = LLM(model=str(model), tokenizer=cfg['model']['local_path'], tensor_parallel_size=1, dtype=cfg['model']['dtype'],
                 enforce_eager=cfg['inference']['enforce_eager'], gpu_memory_utilization=0.75,
                 max_model_len=a['max_prompt_tokens'] + a['max_generate_length'], max_num_seqs=128, disable_log_stats=True,
                 **profile.vllm_kwargs(cfg))
    params = SamplingParams(temperature=a['eval_temperature'], top_p=a['top_p'], max_tokens=a['max_generate_length'], seed=a['seed'])
    results = engine.generate(prompts, params)
    rows = []
    for row, res in zip(data, results, strict=True):
        response = res.outputs[0]
        truth = row['reward_spec']['ground_truth']
        rows.append({'sample_id': row['extra_info']['sample_id'], 'prompt': row['prompt'], 'completion': response.text, 'ground_truth': truth,
                     'parsed_answer': parse_final_answer(response.text), 'correct': bool(score_answer(response.text, truth, response.finish_reason)),
                     'finish_reason': response.finish_reason, 'token_count': len(response.token_ids)})
    write_jsonl(output, rows)
    print(json.dumps({'evaluated_model': str(model), 'examples': len(rows), 'accuracy': sum(r['correct'] for r in rows) / len(rows)}))


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
