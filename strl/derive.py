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
from .experiment import finetune, training
from .tasks import get_environment

PROFILES = Path(__file__).resolve().parent / 'profiles.json'
H100_GB = 80.0
MEMORY_LIMIT_GB = 76.0          # usable per-GPU budget after CUDA context / NCCL buffers
FULL_SHARD_MIN_GBPS = 8.0       # cross-node all-gather bus bandwidth below which full-parameter FSDP should stay within a node


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
    if cfg['environment']['name'] == 'mimo' and p['environment'] != 'mimo':
        deadline = float((cfg['environment'].get('mimo') or {}).get('step_deadline_sec') or 1200)
        gen = max(gen, 0.6 * deadline)                                            # a step waits for its slowest agent episodes (uncalibrated)
    calibrated = p['environment'] == cfg['environment']['name'] and p['model'] == cfg['model']['id'] and cfg['compute']['nodes'] == 1
    return {'generate_s': round(gen, 1), 'train_s': round(train, 1), 'other_s': round(other, 1),
            'step_s': round(gen + train + other, 1), 'profile': p['name'],
            'calibrated': calibrated}   # False: extrapolated across environment/model; trust the canary's measured numbers instead


def params_from_id(model_id: str) -> float | None:
    import re
    match = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)[bB](?![a-zA-Z])', model_id.split('/')[-1])
    return float(match.group(1)) if match else None


def text_config(hf: dict | None) -> dict:
    return (hf or {}).get('text_config', hf or {})


def linear_shapes(hf: dict | None) -> list[tuple[int, int]] | None:
    """(in, out) features of every decoder linear layer, i.e. what LoRA ``all-linear`` wraps. Handles dense attention and the
    Qwen3.5 hybrid (Gated-DeltaNet linear-attention layers + gated full attention every ``full_attention_interval`` layers)."""
    t = text_config(hf)
    try:
        layers, h = t['num_hidden_layers'], t['hidden_size']
    except KeyError:
        return None
    inter, heads = t.get('intermediate_size', 4 * h), t.get('num_attention_heads', 1)
    kv, hd = t.get('num_key_value_heads', heads), t.get('head_dim', h // heads)
    mlp = [(h, inter), (h, inter), (inter, h)]
    full = [(h, heads * hd * (2 if t.get('attn_output_gate') else 1)), (h, kv * hd), (h, kv * hd), (heads * hd, h)]
    nk, nv, dk, dv = (t.get(k, 0) for k in ('linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim', 'linear_value_head_dim'))
    gdn = [(h, 2 * nk * dk + nv * dv), (h, nv * dv), (h, nv), (h, nv), (nv * dv, h)]   # in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
    interval = t.get('full_attention_interval')
    types = t.get('layer_types') or ['linear_attention' if nv and interval and (i + 1) % interval else 'full_attention' for i in range(layers)]
    return [shape for kind in types[:layers] for shape in (gdn if kind == 'linear_attention' and nv else full) + mlp]


def lora_params(hf: dict | None, rank: int) -> int | None:
    shapes = linear_shapes(hf)
    return None if shapes is None else rank * sum(i + o for i, o in shapes)


def memory_estimate_gb(params_b: float, cfg: dict, hf: dict | None = None, *, world: int | None = None, tp: int | None = None,
                       fsdp_size: int | None = None) -> dict[str, Any]:
    """Per-GPU peaks for colocated FSDP + vLLM (sleep mode swaps them between phases).

    training: full = 16 B/param (bf16 param + grad, fp32 master, Adam m+v) sharded over the FSDP group (cpu_offload keeps only
    bf16 param+grad on the GPU); lora = frozen bf16 base 2 B/param over the group + 16 B per LoRA param; + bf16 reference
    model when use_kl_loss; + activations (checkpointed layer inputs micro_tokens x hidden x layers x 2 B, one recomputed layer,
    bf16 logits micro_tokens x vocab x 2 B). rollout: per-engine weights 2 B x params / TP and the KV room left inside
    gpu_memory_utilization."""
    a, inf, ft, tr = cfg['algorithm'], cfg['inference'], finetune(cfg), training(cfg)
    world = world or cfg['derived']['world_size']
    tp = tp or inf['tensor_parallel_size']
    group = fsdp_size or tr['fsdp_size'] or world
    p = params_b * 1e9
    host = 0.0
    if ft['mode'] == 'lora':
        lp = lora_params(hf, ft['lora']['rank']) or 0.01 * p
        state = 2 * p / group + 16 * lp / group
    else:
        lp = 0
        state = (4 if tr['cpu_offload'] else 16) * p / group
        host = 12 * p / group if tr['cpu_offload'] else 0.0
    ref = 2 * p / group if a['use_kl_loss'] else 0.0
    tokens = tr['max_tokens_per_microbatch'] or a['micro_train_per_gpu'] * (a['max_prompt_tokens'] + a['max_generate_length'])
    t = text_config(hf)
    if 'hidden_size' in t and 'num_hidden_layers' in t:
        hidden, layers, vocab = t['hidden_size'], t['num_hidden_layers'], t.get('vocab_size', 32000)
        per_layer = tokens * hidden * 2 * (1 if tr['gradient_checkpointing'] else 17)
        activations = per_layer * layers + tokens * hidden * 34 + tokens * vocab * 2
    else:   # no config.json: the previous first-order rule (GB per 4k tokens per billion parameters)
        activations = 6.0e9 * tokens / 4096 * max(1.0, params_b)
    train = (state + ref + activations) / 1e9
    weights = 2 * p / tp / 1e9
    reserved = H100_GB * inf['gpu_memory_utilization']
    return {'training_phase_gb': round(train, 1), 'rollout_phase_gb': round(reserved, 1), 'engine_weights_gb': round(weights, 1),
            'kv_room_gb': round(reserved - weights, 1), 'host_offload_gb_per_gpu': round(host / 1e9, 1), 'fsdp_group': group,
            'lora_params_m': round(lp / 1e6, 1) if lp else 0, 'fits': train < MEMORY_LIMIT_GB and reserved - weights > 1.0}


def fits_one_gpu(cfg: dict, params_b: float | None, hf: dict | None = None) -> bool:
    """Would the experiment's model train + roll out on a single H100 (the standard 1-GPU canary)?"""
    return params_b is None or bool(memory_estimate_gb(params_b, cfg, hf, world=1, tp=1, fsdp_size=1)['fits'])


def recommend_layout(cfg: dict, bandwidth_gbps: float | None = None) -> dict[str, Any]:
    """LoRA: HSDP within a node when nodes > 1 (only the small LoRA gradients cross nodes). Full: full shard over every GPU when
    the measured cross-node bandwidth is >= FULL_SHARD_MIN_GBPS, else HSDP + cpu_offload (unknown bandwidth: full shard + NETWORK)."""
    nodes, gpn = cfg['compute']['nodes'], cfg['compute']['gpus_per_node']
    if nodes == 1:
        return {'fsdp_size': None, 'cpu_offload': False, 'reason': 'single node: full shard'}
    if finetune(cfg)['mode'] == 'lora':
        return {'fsdp_size': gpn, 'cpu_offload': False, 'reason': 'lora: shard the frozen base within each node (HSDP), replicate across nodes'}
    if bandwidth_gbps is None:
        return {'fsdp_size': None, 'cpu_offload': False, 'reason': 'full: cross-node bandwidth unknown, full shard (probe it: NETWORK)'}
    if bandwidth_gbps >= FULL_SHARD_MIN_GBPS:
        return {'fsdp_size': None, 'cpu_offload': False, 'reason': f'full: {bandwidth_gbps:.1f} GB/s cross-node, full shard'}
    return {'fsdp_size': gpn, 'cpu_offload': True, 'reason': f'full: only {bandwidth_gbps:.1f} GB/s cross-node, HSDP within a node + CPU offload'}


def layout(cfg: dict, memory: dict, network: dict | None = None) -> dict[str, Any]:
    ft, tr = finetune(cfg), training(cfg)
    lora = ft['lora']
    world = cfg['derived']['world_size']
    bandwidth = ((network or {}).get('bandwidth_gbps') or {}).get('cross_node')
    return {'mode': ft['mode'] if ft['mode'] == 'full' else f"lora r{lora['rank']} a{lora['alpha']} {lora['target_modules']}",
            'fsdp_group': memory['fsdp_group'], 'replicas': world // memory['fsdp_group'], 'cpu_offload': tr['cpu_offload'],
            'tp': cfg['inference']['tensor_parallel_size'], 'engines': cfg['derived']['num_engines'],
            'train_gb_per_gpu': memory['training_phase_gb'], 'rollout_gb_per_gpu': memory['rollout_phase_gb'],
            'recommended': recommend_layout(cfg, bandwidth)}


def warnings(cfg: dict, env: Any, profile: models.ModelProfile, params_b: float | None, timing: dict,
             memory: dict | None = None, network: dict | None = None) -> list[str]:
    a, e, b = cfg['algorithm'], cfg['evaluation'], cfg['budget']
    nodes, lora = cfg['compute']['nodes'], finetune(cfg)['mode'] == 'lora'
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
    if a['lr'] > 1e-5 and not lora:
        out.append(f'STABILITY: lr={a["lr"]} is high for full-parameter RL of small models (verified: 2e-6).')
    nondefault = [c for c in cfg['reward']['components'] if c.get('weight', 1.0) and c['fn'] not in {d['fn'] for d in env.default_reward}]
    if nondefault:
        out.append(f'REWARD: shaping terms {[c["fn"] for c in nondefault]} are trained on but NOT the yardstick: judge the run by eval/task_metric '
                   'and the held-out test, and watch environment/reward_components/* for terms that rise while the task metric does not (reward hacking).')
    seconds = b['attempt_max_seconds'] - b['bootstrap_margin_seconds'] - b['final_eval_reserve_seconds'] - 550
    steps = int(seconds / timing['step_s']) if timing['step_s'] > 0 else 0
    if steps < 20:
        out.append(f'BUDGET: only ~{steps} optimizer steps fit in the attempt after startup and final-eval reserve; consider shorter generations or a longer attempt.')
    if env.generator_kind == 'harbor' and profile.name == 'qwen3_5':
        out.append('HARBOR: Qwen3.5 history templates drop <think> blocks, so step-wise merging falls back to one sequence per turn (correct, slower).')
    out += env.plan_warnings(cfg)
    if params_b is None:
        out.append('MEMORY: parameter count unknown (config.json unavailable); set model.params_b for memory/step estimates.')
    if memory and memory['training_phase_gb'] > MEMORY_LIMIT_GB:
        out.append(f"MEMORY: estimated training peak {memory['training_phase_gb']} GB/GPU > {MEMORY_LIMIT_GB:g} GB: use finetune.mode lora, "
                   'compute.training.cpu_offload, more GPUs in the FSDP group, max_tokens_per_microbatch or shorter sequences.')
    if memory and memory['kv_room_gb'] <= 1.0:
        out.append(f"MEMORY: vLLM weights {memory['engine_weights_gb']} GB/GPU leave no KV room inside gpu_memory_utilization "
                   f"({memory['rollout_phase_gb']} GB): raise inference.tensor_parallel_size or gpu_memory_utilization.")
    if nodes > 1 and not (network or {}).get('efa_available'):
        out.append('NETWORK: EFA is unverified for this multi-node job (no probe result): the worker probe records the NCCL transport and '
                   'cross-node all-gather GB/s in network.json; on NET/Socket prefer LoRA or HSDP (compute.training.fsdp_size = gpus_per_node).')
    if nodes > 1 and lora:
        out.append('LORA_SYNC: SkyRL writes the LoRA adapter to a node-local lora_sync_path from global rank 0 only; strl patches the FSDP '
                   'worker so the first rank on every host writes its own copy before the engines reload it (contract-tested on CPU; '
                   'the 2-node canary proves it).')
    return out


def params_and_config(cfg: dict) -> tuple[models.ModelProfile, dict | None, float | None]:
    profile, hf = models.detect(cfg['model'])
    params_b = cfg['model'].get('params_b') or params_from_id(cfg['model']['id']) or (profile.param_estimate_b(hf) if hf else None)
    return profile, hf, params_b


def plan(cfg: dict, network: dict | None = None) -> dict:
    """``network``: a probe result (artifacts/network.json of an earlier attempt) the layout recommendation is based on."""
    env = get_environment(cfg['environment']['name'])
    profile, hf, params_b = params_and_config(cfg)
    timing = estimate_step_seconds(cfg, params_b or 1.0)
    memory = memory_estimate_gb(params_b or 1.0, cfg, hf)
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
                      'memory': memory,
                      'node_hours': cfg['compute']['nodes'] * cfg['compute']['gpus_per_node'] / 8 * budget['attempt_max_seconds'] / 3600,
                      'note': 'estimates: extrapolated from the nearest measured profile; the 1-GPU canary measures the real numbers'},
        'monitoring': {'wandb': f"{cfg['tracking']['base_url']}/{cfg['tracking']['entity']}/{cfg['tracking']['project']} group={cfg['tracking']['group']}",
                       'critic': cfg['tracking'].get('critic', {}).get('enabled', False)},
        'layout': layout(cfg, memory, network),
        'warnings': warnings(cfg, env, profile, params_b, timing, memory, network),
    }
    return out
