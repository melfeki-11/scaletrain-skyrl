"""Offline greedy evaluation of one model on the held-out coding test set (one shard): vLLM generation, then batch execution in
the Modal executor. Identical prompts, decoding, extractor and harness for base and trained models; never masks: any remaining
infra_error after retries fails the shard (a biased metric is worse than no metric)."""
import argparse
import json
from pathlib import Path

from ... import models
from ...experiment import current
from ...finalize import write_jsonl
from . import data as coding_data
from . import executor
from .extract import extract_code


def max_model_len(cfg: dict) -> int:
    """Held-out prompts are never filtered: size the offline context from the longest tokenized test prompt (prepare records it)."""
    manifest = json.loads((Path(cfg['run']['data_dir']) / 'manifest.json').read_text())
    longest = ((manifest.get('prompt_tokens') or {}).get('test') or {}).get('max', 0)
    return max(cfg['algorithm']['max_prompt_tokens'], longest + 16) + cfg['algorithm']['max_generate_length']


def evaluate(model, output, limit=None, rank=0, world_size=1):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    cfg = current()
    profile = models.detect(cfg['model'])[0]
    a = cfg['algorithm']
    tasks = coding_data.load_tasks(cfg, 'test')
    if limit is not None:
        tasks = tasks[:limit]
    if not 0 <= rank < world_size:
        raise ValueError('Invalid evaluation rank')
    tasks = tasks[rank::world_size]
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['local_path'])
    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': t.prompt}], tokenize=False, add_generation_prompt=True,
                                             **profile.chat_kwargs(cfg)) for t in tasks]
    engine = LLM(model=str(model), tokenizer=cfg['model']['local_path'], tensor_parallel_size=1, dtype=cfg['model']['dtype'],
                 enforce_eager=cfg['inference']['enforce_eager'], gpu_memory_utilization=0.75,
                 max_model_len=max_model_len(cfg), max_num_seqs=128, disable_log_stats=True,
                 **profile.vllm_kwargs(cfg))
    params = SamplingParams(temperature=a['eval_temperature'], top_p=a['top_p'], max_tokens=a['max_generate_length'], seed=a['seed'])
    results = engine.generate(prompts, params)
    codes = [extract_code(r.outputs[0].text) for r in results]
    runnable = [i for i, c in enumerate(codes) if c is not None]
    verdicts = dict(zip(runnable, executor.score_many([tasks[i].job(codes[i]) for i in runnable])))
    rows = []
    for i, (task, res) in enumerate(zip(tasks, results, strict=True)):
        response = res.outputs[0]
        verdict = verdicts.get(i) or {'failure_class': 'no_code_block', 'passed': 0, 'total': len(task.tests)}
        if verdict['failure_class'] == 'infra_error':
            raise RuntimeError(f'executor infrastructure error on {task.sample_id} after retries: {verdict.get("detail")}')
        stopped = response.finish_reason == 'stop'
        rows.append({'sample_id': task.sample_id, 'source': task.source, 'prompt': task.prompt, 'completion': response.text,
                     'ground_truth': f'{task.source}:{task.task_id}', 'parsed_answer': codes[i], 'finish_reason': response.finish_reason,
                     'token_count': len(response.token_ids), 'failure_class': verdict['failure_class'] if stopped else 'truncated',
                     'tests_passed': verdict.get('passed', 0), 'tests_total': verdict.get('total', len(task.tests)),
                     'correct': bool(stopped and verdict['failure_class'] == 'ok')})
    write_jsonl(output, rows)
    print(json.dumps({'evaluated_model': str(model), 'examples': len(rows), 'pass_at_1': sum(r['correct'] for r in rows) / max(1, len(rows))}))


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
