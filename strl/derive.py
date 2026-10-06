"""`strl plan`: everything derived from an experiment, with estimates and decision-point warnings. Local, no spend.

Estimates come from strl/profiles.json (measured runs) scaled by first-order rules; they are labelled as estimates, and the
nearest measured profile is reported so you can judge how far the extrapolation reaches.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import models
from .tasks import get_environment

PROFILES = Path(__file__).resolve().parent / 'profiles.json'
H100_GB = 80.0


def _profiles() -> list[dict]:
    return json.loads(PROFILES.read_text())['runs']


def nearest_profile(cfg: dict) -> dict:
    runs = [r for r in _profiles() if r.get('tokens_per_second_per_gpu')]   # only full runs carry calibrated timing
    env = cfg['environment']['name']
    same = [r for r in runs if r['environment'] == env] or runs
    return min(same, key=lambda r: abs(math.log(r['params_b'] / max(1e-3, cfg.get('_params_b', r['params_b'])))))


def estimate_step_seconds(cfg: dict, params_b: float) -> dict[str, float]:
    """Ratios to the nearest measured profile (so the measured config reproduces its own numbers):
    generation ~ longest sequence x params^0.6 x sqrt(rollouts per GPU); training ~ tokens x params / GPUs."""
    p = nearest_profile(dict(cfg, _params_b=params_b))
    a, gpus = cfg['algorithm'], cfg['derived']['world_size']
    size = params_b / p['params_b']
    per_gpu = (cfg['derived']['rollouts_per_step'] / gpus) / (p['rollouts_per_step'] / p['gpus'])
    gen = p['median_generate_seconds'] * (a['max_generate_length'] / p['max_generate_length']) * size ** 0.6 * per_gpu ** 0.5
    mean_resp = min(a['max_generate_length'], p['mean_response_tokens'])
    tokens_ratio = (cfg['derived']['rollouts_per_step'] * (mean_resp + a['max_prompt_tokens'] / 2)) / \
                   (p['rollouts_per_step'] * (p['mean_response_tokens'] + 256))
    train = p['median_policy_train_seconds'] * tokens_ratio * size * (p['gpus'] / gpus)
    other = (p['median_step_seconds'] - p['median_generate_seconds'] - p['median_policy_train_seconds']) * tokens_ratio * size * (p['gpus'] / gpus)
    if cfg['environment']['name'] == 'coding_exec':
        other += 2.0                                                              # Modal scoring tail not hidden by generation
    if cfg['environment']['name'] == 'coding_harbor' and p['environment'] != 'coding_harbor':
        hb = cfg['environment'].get('harbor') or {}
        gen = max(gen * int(hb.get('max_turns', 8)) * 0.5, 60.0)                  # multi-turn agent + sandbox startup (uncalibrated)
    calibrated = p['environment'] == cfg['environment']['name'] and p['model'] == cfg['model']['id']
    return {'generate_s': round(gen, 1), 'train_s': round(train, 1), 'other_s': round(other, 1),
            'step_s': round(gen + train + other, 1), 'profile': p['name'],
            'calibrated': calibrated}   # False: extrapolated across environment/model; trust the canary's measured numbers instead


def params_from_id(model_id: str) -> float | None:
    import re
    match = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)[bB](?![a-zA-Z])', model_id.split('/')[-1])
    return float(match.group(1)) if match else None


def memory_estimate_gb(params_b: float, cfg: dict) -> dict[str, float]:
    """Per-GPU peaks for colocated FSDP + vLLM (sleep mode swaps them): training state is sharded, the TP1 engine is not."""
    gpus = cfg['derived']['world_size']
    tp = cfg['inference']['tensor_parallel_size']
    train_state = params_b * (2 + 4 + 8) / gpus + (params_b * 2 / gpus if cfg['algorithm']['use_kl_loss'] else 0.0)  # bf16+fp32 master+Adam (+ref)
    activations = 6.0 * cfg['algorithm']['micro_train_per_gpu'] * (cfg['algorithm']['max_prompt_tokens'] + cfg['algorithm']['max_generate_length']) / 4096 * max(1.0, params_b)
    engine = params_b * 2 / tp + H100_GB * cfg['inference']['gpu_memory_utilization'] * 0.5
    return {'training_phase_gb': round(train_state + activations, 1), 'rollout_phase_gb': round(engine, 1),
            'fits': train_state + activations < H100_GB * 0.9 and params_b * 2 / tp < H100_GB * cfg['inference']['gpu_memory_utilization']}


def warnings(cfg: dict, env: Any, profile: models.ModelProfile, params_b: float | None, timing: dict) -> list[str]:
    a, e, b = cfg['algorithm'], cfg['evaluation'], cfg['budget']
    out = []
    if profile.verified != 'gpu':
        out.append(f'MODEL: profile {profile.name!r} is verified at level {profile.verified!r} only: stage --mode dry, then --mode canary, before a real run. '
                   f'{profile.notes}')
    measured = [r for r in json.loads(PROFILES.read_text())['runs'] if r['environment'] == env.name and r['model'] == cfg['model']['id']]
    p0 = measured[-1]['base_task_metric'] if measured else env.expected_base_reward
    if p0 is not None and 0 < p0 < 1:
        all_same = p0 ** a['n_samples'] + (1 - p0) ** a['n_samples']
        out.append(f'SIGNAL: at base reward ~{p0:.2f}, {100 * all_same:.0f}% of groups of {a["n_samples"]} are all-equal (zero advantage); '
                   'watch train/learning_signal_fraction. More samples per prompt raise it; so does an easier curriculum.')
    elif p0 is None:
        out.append('SIGNAL: base pass rate unknown for this environment/model: the 1-GPU canary measures it (eval/task_metric at step 0); '
                   'if < 0.3, prefer prompts_per_step 32 x n_samples 16 (same rollouts, ~6x fewer all-fail groups).')
    if a['n_samples'] < 4:
        out.append('SIGNAL: n_samples < 4 gives very noisy group baselines for GRPO.')
    if a['max_generate_length'] >= 1024:
        out.append(f'SPEED: max_generate_length={a["max_generate_length"]}: generation time is set by the longest sequence (eager decode); '
                   'halving it roughly halves generate time. Truncated rollouts get reward 0 (zero_reward_on_non_stop).')
    if e['checkpoint_interval_steps'] != e['interval_steps']:
        out.append('SELECTION: only exported steps can be selected as the final checkpoint; checkpoint_interval_steps == interval_steps makes every '
                   'validated step eligible (costs one HF export per eval).')
    if a['use_kl_loss'] and a['kl_coef'] == 0:
        out.append('MEMORY: use_kl_loss with kl_coef 0 keeps a reference model for nothing; set use_kl_loss: false.')
    if a['lr'] > 1e-5:
        out.append(f'STABILITY: lr={a["lr"]} is high for full-parameter RL of small models (verified: 2e-6).')
    nondefault = [c for c in cfg['reward']['components'] if c.get('weight', 1.0) and c['fn'] not in {d['fn'] for d in env.default_reward}]
    if nondefault:
        out.append(f'REWARD: shaping terms {[c["fn"] for c in nondefault]} are trained on but NOT the yardstick: judge the run by eval/task_metric '
                   'and the held-out test, and watch environment/reward_components/* for terms that rise while the task metric does not (reward hacking).')
    seconds = b['attempt_max_seconds'] - b['bootstrap_margin_seconds'] - b['final_eval_reserve_seconds'] - 550
    steps = int(seconds / timing['step_s']) if timing['step_s'] > 0 else 0
    if steps < 20:
        out.append(f'BUDGET: only ~{steps} optimizer steps fit in the attempt after startup and final-eval reserve; consider shorter generations or a longer attempt.')
    if cfg['environment']['name'] == 'coding_harbor' and profile.name == 'qwen3_5':
        out.append('HARBOR: Qwen3.5 history templates drop <think> blocks, so step-wise merging falls back to one sequence per turn (correct, slower).')
    if params_b is None:
        out.append('MEMORY: parameter count unknown (config.json unavailable); set model.params_b for memory/step estimates.')
    return out


def plan(cfg: dict) -> dict:
    env = get_environment(cfg['environment']['name'])
    profile, hf = models.detect(cfg['model'])
    params_b = cfg['model'].get('params_b') or params_from_id(cfg['model']['id']) or (profile.param_estimate_b(hf) if hf else None)
    timing = estimate_step_seconds(cfg, params_b or 1.0)
    budget = cfg['budget']
    usable = budget['attempt_max_seconds'] - budget['bootstrap_margin_seconds'] - budget['final_eval_reserve_seconds'] - 550
    a = cfg['algorithm']
    per_gpu = cfg['derived']['rollouts_per_step'] // max(1, cfg['derived']['world_size'])
    out = {
        'experiment': cfg['name'], 'sha': cfg['_meta']['sha'], 'chain': cfg['_meta']['chain'],
        'axes': {'model': f"{cfg['model']['id']}@{cfg['model']['revision'][:12]} (profile {profile.name}, verified: {profile.verified})",
                 'environment': f"{env.name} ({env.generator_kind}; metric {env.metric}; {'Modal' if env.needs_modal else 'in-process'} scoring)",
                 'reward': ' + '.join(f"{c.get('weight', 1.0)}*{c['fn']}" for c in cfg['reward']['components']) +
                           ('  [sandbox verifier reward.txt via HarborGenerator]' if env.generator_kind == 'harbor' else
                            '  [generic reward path: strl_env]' if env.uses_generic_reward(cfg) else '  [verified built-in reward path]')},
        'derived': {**cfg['derived'], 'rollouts_per_gpu': per_gpu, 'micro_batches_per_gpu': per_gpu // a['micro_train_per_gpu'],
                    'params_b': round(params_b, 3) if params_b else None,
                    # Harbor: the trajectory holds the whole multi-turn chat (prompt + max_turns x max_generate_length)
                    'max_model_len': env.skyrl_overrides(cfg, 'train').get('trainer.algorithm.max_seq_len',
                                                                            a['max_prompt_tokens'] + a['max_generate_length']),
                    'skyrl_env_class': env.skyrl_overrides(cfg, 'train').get('environment.env_class') or f'({env.generator_kind} generator)'},
        'estimates': {**timing, 'steps_in_attempt': int(usable / timing['step_s']) if timing['step_s'] else None,
                      'memory': memory_estimate_gb(params_b or 1.0, cfg),
                      'node_hours': cfg['compute']['nodes'] * cfg['compute']['gpus_per_node'] / 8 * budget['attempt_max_seconds'] / 3600,
                      'note': 'estimates: extrapolated from the nearest measured profile; the 1-GPU canary measures the real numbers'},
        'monitoring': {'wandb': f"{cfg['tracking']['base_url']}/{cfg['tracking']['entity']}/{cfg['tracking']['project']} group={cfg['tracking']['group']}",
                       'critic': cfg['tracking'].get('critic', {}).get('enabled', False)},
        'warnings': warnings(cfg, env, profile, params_b, timing),
    }
    return out
