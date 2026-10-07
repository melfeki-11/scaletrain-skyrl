"""Experiment specs: one YAML is the only thing a user edits.

An experiment ``extends`` a base file and overrides only the keys it changes. ``load`` resolves the inheritance chain,
fills defaults, makes paths absolute, and validates EVERYTHING it can without GPUs or network (typos in key names,
authorization limits, batch divisibility, unknown model/environment/reward names), reporting all problems at once.
``experiment_sha`` is a content hash of the resolved config; it is stamped into W&B so runs are comparable and
reproducible.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_EXTENDS_DEPTH = 8


class ExperimentError(ValueError):
    """Raised with every validation problem found, one per line."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__('\n'.join(f'- {p}' for p in self.problems))


# Allowed keys per section. A value of None means "free-form mapping" (validated elsewhere).
SCHEMA: dict[str, Any] = {
    'name': str, 'description': str, 'extends': str,
    'model': {'id': str, 'revision': str, 'local_path': str, 'dtype': str, 'profile': str, 'params_b': (int, float)},
    'environment': None,
    'reward': {'components': list},
    'algorithm': {
        'estimator': str, 'prompts_per_step': int, 'minibatch_prompts': int, 'n_samples': int, 'micro_train_per_gpu': int,
        'micro_forward_per_gpu': int, 'lr': (int, float), 'update_epochs': int, 'use_kl_loss': bool, 'kl_coef': (int, float),
        'zero_variance_filter': bool, 'temperature': (int, float), 'top_p': (int, float), 'max_prompt_tokens': int,
        'max_generate_length': int, 'max_steps': int, 'bf16': bool, 'seed': int, 'zero_reward_on_non_stop': bool,
        'eval_temperature': (int, float),
    },
    'inference': {'tensor_parallel_size': int, 'gpu_memory_utilization': (int, float), 'max_num_seqs': int, 'enforce_eager': bool,
                  'max_env_workers': int, 'num_engines': int, 'max_num_batched_tokens': int, 'engine_init_kwargs': dict},
    'finetune': {'mode': str, 'lora': dict},
    'evaluation': {
        'interval_steps': int, 'checkpoint_interval_steps': int, 'max_checkpoints': int, 'eval_batch_size': int,
        'fixed_qualitative_samples': int, 'training_groups_to_log': int, 'bootstrap_samples': int, 'bootstrap_seed': int,
        'ci_level': (int, float), 'eval_shards_per_model': int, 'final_eval': bool,
    },
    'compute': {'nodes': int, 'gpus_per_node': int, 'instance_type': str, 'team': str, 'product': str, 'priority': str,
                'allow_borrowing': bool, 'image': str, 'training': dict},
    'runtime': {'skyrl_commit': str, 'python': str},
    'budget': {'total_seconds': int, 'attempt_max_seconds': int, 'bootstrap_margin_seconds': int,
               'final_eval_reserve_seconds': int, 'canary_max_seconds': int, 'stall_seconds': int, 'ledger': str,
               'save_optimizer_state': bool},
    'tracking': {'base_url': str, 'entity': str, 'project': str, 'group': str, 'credentials_uri': str, 'tags': list, 'critic': None,
                 'alerts': bool},
    'storage': {'bucket': str, 'prefix': str, 'profile': str, 'region': str},
    'limits': {'max_nodes': int, 'gpus_per_node': list, 'max_attempt_seconds': int, 'models': list},
    'run': {'data_dir': str, 'artifact_dir': str},
    'services': {'modal': None, 'litellm': None},
}
# Scaling knobs. Defaults live here (not in the resolved config), so experiments that do not set them keep their sha and
# render exactly the SkyRL overrides they always did.
FINETUNE_DEFAULTS = {'mode': 'full', 'lora': {'rank': 0, 'alpha': 16, 'dropout': 0.0, 'target_modules': 'all-linear', 'exclude_modules': None}}
LORA_TYPES = {'rank': int, 'alpha': int, 'dropout': (int, float), 'target_modules': (str, list), 'exclude_modules': (str, type(None))}
TRAINING_DEFAULTS = {'fsdp_size': None, 'cpu_offload': False, 'reshard_after_forward': True, 'reduce_dtype': 'fp32',
                     'max_tokens_per_microbatch': None, 'gradient_checkpointing': True}
TRAINING_TYPES = {'fsdp_size': (int, type(None)), 'cpu_offload': bool, 'reshard_after_forward': bool, 'reduce_dtype': str,
                  'max_tokens_per_microbatch': (int, type(None)), 'gradient_checkpointing': bool}


def finetune(cfg: dict) -> dict:
    """The experiment's finetune section with defaults (mode full; LoRA settings only meaningful in lora mode)."""
    raw = cfg.get('finetune') or {}
    return {'mode': raw.get('mode', 'full'), 'lora': {**FINETUNE_DEFAULTS['lora'], **(raw.get('lora') or {})}}


def training(cfg: dict) -> dict:
    """compute.training with defaults (fsdp_size None = full shard over every GPU)."""
    return {**TRAINING_DEFAULTS, **(cfg['compute'].get('training') or {})}


def save_optimizer_state(cfg: dict) -> bool:
    return bool(cfg['budget'].get('save_optimizer_state', True))


REQUIRED_SECTIONS = ('model', 'environment', 'reward', 'algorithm', 'inference', 'evaluation', 'compute', 'runtime',
                     'budget', 'tracking', 'storage', 'limits', 'run')


def parse_override(text: str) -> tuple[list[str], Any]:
    """'a.b.c=value' -> (['a', 'b', 'c'], YAML-parsed value)."""
    key, sep, value = text.partition('=')
    if not sep or not key.strip() or any(not part for part in key.strip().split('.')):
        raise ExperimentError([f'--set expects KEY=VALUE with a dotted key, got {text!r}'])
    parsed = yaml.safe_load(value) if value.strip() else ''
    if isinstance(parsed, str) and re.fullmatch(r'[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+', parsed.strip()):
        parsed = float(parsed)   # YAML 1.1 reads 3e-5 as a string
    return key.strip().split('.'), parsed


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    out = copy.deepcopy(raw)
    for text in overrides:
        path, value = parse_override(text)
        node = out
        for part in path[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[path[-1]] = value
    return out


def deep_merge(base: dict, override: dict) -> dict:
    """Mappings merge recursively; everything else (lists, scalars) is replaced by the override."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _find(path: str | Path, relative_to: Path | None) -> Path:
    candidate = Path(path)
    options = [candidate] if candidate.is_absolute() else [
        *( [relative_to / candidate] if relative_to else [] ), REPO_ROOT / candidate, Path.cwd() / candidate]
    for option in options:
        if option.is_file():
            return option.resolve()
    raise ExperimentError([f'experiment file not found: {path} (looked in {[str(o) for o in options]})'])


def resolve(path: str | Path, *, _chain: tuple[Path, ...] = ()) -> dict:
    """Load ``path`` and its ``extends`` ancestors; returns the merged raw mapping with ``_chain`` listing the files."""
    file = _find(path, _chain[-1].parent if _chain else None)
    if file in _chain:
        raise ExperimentError([f'extends cycle: {" -> ".join(str(c) for c in (*_chain, file))}'])
    if len(_chain) >= MAX_EXTENDS_DEPTH:
        raise ExperimentError([f'extends chain deeper than {MAX_EXTENDS_DEPTH}'])
    data = yaml.safe_load(file.read_text()) or {}
    if not isinstance(data, dict):
        raise ExperimentError([f'{file}: top level must be a mapping'])
    parent_ref = data.get('extends')
    if parent_ref:
        parent = resolve(parent_ref, _chain=(*_chain, file))
        chain = parent['_chain'] + [str(file)]
        merged = deep_merge({k: v for k, v in parent.items() if k != '_chain'}, {k: v for k, v in data.items() if k != 'extends'})
        merged['extends'] = parent_ref
    else:
        chain, merged = [str(file)], dict(data)
    merged['_chain'] = chain
    return merged


def _type_ok(value: Any, expected: Any) -> bool:
    expected = expected if isinstance(expected, tuple) else (expected,)
    if bool in expected:
        return isinstance(value, bool)
    return isinstance(value, expected) and not isinstance(value, bool)


def check_schema(cfg: dict) -> list[str]:
    problems: list[str] = []
    for key in cfg:
        if key.startswith('_'):
            continue
        if key not in SCHEMA:
            problems.append(f'unknown top-level key {key!r} (did you mean one of {sorted(k for k in SCHEMA if k[0] == key[0])[:4] or sorted(SCHEMA)[:6]}?)')
    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            problems.append(f'missing section {section!r}')
    for section, spec in SCHEMA.items():
        value = cfg.get(section)
        if value is None:
            continue
        if isinstance(spec, dict):
            if not isinstance(value, dict):
                problems.append(f'{section} must be a mapping')
                continue
            for key, item in value.items():
                if key not in spec:
                    problems.append(f'unknown key {section}.{key} (allowed: {sorted(spec)})')
                elif spec[key] is not None and not _type_ok(item, spec[key]):
                    problems.append(f'{section}.{key} must be {spec[key] if not isinstance(spec[key], tuple) else "/".join(t.__name__ for t in spec[key])}, got {type(item).__name__} {item!r}')
        elif spec is not None and not _type_ok(value, spec):
            problems.append(f'{section} must be {spec.__name__}')
    return problems


def _absolutize(cfg: dict, root: Path) -> None:
    for section, key in (('run', 'data_dir'), ('run', 'artifact_dir'), ('model', 'local_path')):
        value = Path(cfg[section][key])
        cfg[section][key] = str(value if value.is_absolute() else root / value)


def derive_placement(cfg: dict) -> dict:
    gpus = cfg['compute']['nodes'] * cfg['compute']['gpus_per_node']
    tp = cfg['inference']['tensor_parallel_size']
    engines = cfg['inference'].get('num_engines') or (gpus // tp if tp and gpus % tp == 0 else 0)
    return {'world_size': gpus, 'num_engines': engines,
            'rollouts_per_step': cfg['algorithm']['prompts_per_step'] * cfg['algorithm']['n_samples']}


def validate(cfg: dict, *, check_registries: bool = True) -> None:
    problems = check_schema(cfg)
    if problems:  # structural problems make the semantic checks meaningless
        raise ExperimentError(problems)
    a, i, e, c, b, lim = (cfg[k] for k in ('algorithm', 'inference', 'evaluation', 'compute', 'budget', 'limits'))
    gpn, nodes = c['gpus_per_node'], c['nodes']
    gpus = gpn * nodes
    if nodes > lim['max_nodes'] or nodes < 1:
        problems.append(f'compute.nodes={nodes} outside the authorized limits.max_nodes={lim["max_nodes"]}')
    if gpn not in lim['gpus_per_node']:
        problems.append(f'compute.gpus_per_node={gpn} not in limits.gpus_per_node={lim["gpus_per_node"]}')
    if b['attempt_max_seconds'] > lim['max_attempt_seconds']:
        problems.append(f'budget.attempt_max_seconds={b["attempt_max_seconds"]} exceeds limits.max_attempt_seconds={lim["max_attempt_seconds"]}')
    if b['attempt_max_seconds'] > b['total_seconds']:
        problems.append('budget.attempt_max_seconds cannot exceed budget.total_seconds')
    if b['final_eval_reserve_seconds'] + 900 >= b['attempt_max_seconds'] and e.get('final_eval', True):
        problems.append('budget leaves no training time: attempt_max_seconds must exceed final_eval_reserve_seconds + 900')
    if '*' not in lim['models'] and cfg['model']['id'] not in lim['models']:
        problems.append(f'model {cfg["model"]["id"]!r} is not authorized by limits.models={lim["models"]}')
    tp = i['tensor_parallel_size']
    if tp < 1 or gpus % tp:
        problems.append(f'inference.tensor_parallel_size={tp} must divide the total GPU count {gpus}')
    if a['estimator'] == 'grpo' and a['n_samples'] < 2:
        problems.append('algorithm.n_samples must be >= 2 for GRPO (group-relative advantages need a group)')
    if a['prompts_per_step'] % a['minibatch_prompts']:
        problems.append('algorithm.prompts_per_step must be a multiple of algorithm.minibatch_prompts')
    if gpus:
        expanded = a['minibatch_prompts'] * a['n_samples']
        if expanded % gpus or (expanded // gpus) % a['micro_train_per_gpu']:
            problems.append(f'minibatch_prompts*n_samples={expanded} must divide evenly into {gpus} GPUs x micro_train_per_gpu={a["micro_train_per_gpu"]} '
                            f'(per-GPU rollouts {expanded / gpus:g})')
        full = a['prompts_per_step'] * a['n_samples']
        if full % gpus or (expanded // gpus) % a['micro_forward_per_gpu']:
            problems.append(f'prompts_per_step*n_samples={full} must divide evenly into {gpus} GPUs x micro_forward_per_gpu={a["micro_forward_per_gpu"]}')
    shards = e['eval_shards_per_model']
    if e.get('final_eval', True) and (shards < 1 or shards * 2 > gpn):
        problems.append(f'evaluation.eval_shards_per_model={shards}: parallel final eval needs one GPU per shard for both models (<= {gpn // 2}); '
                        'set evaluation.final_eval: false to skip it')
    if e['checkpoint_interval_steps'] % e['interval_steps']:
        problems.append('evaluation.checkpoint_interval_steps should be a multiple of interval_steps (only exported steps can be selected)')
    if not 0 < i['gpu_memory_utilization'] < 1:
        problems.append('inference.gpu_memory_utilization must be in (0, 1)')
    if a['max_prompt_tokens'] < 1 or a['max_generate_length'] < 1:
        problems.append('algorithm.max_prompt_tokens and max_generate_length must be positive')
    if a['lr'] <= 0:
        problems.append('algorithm.lr must be positive')
    if c['priority'] not in {'high', 'normal', 'low'}:
        problems.append('compute.priority must be high, normal or low')
    problems += _scaling_problems(cfg, gpus)
    components = cfg['reward']['components']
    if not components or not all(isinstance(x, dict) and 'fn' in x for x in components):
        problems.append('reward.components must be a non-empty list of {fn, weight, ...params}')
    else:
        for x in components:
            if not _type_ok(x.get('weight', 1.0), (int, float)):
                problems.append(f'reward component {x["fn"]!r}: weight must be a number')
        if not any(isinstance(x.get('weight', 1.0), (int, float)) and x.get('weight', 1.0) != 0 for x in components):
            problems.append('reward.components: at least one component needs a non-zero weight')
    if check_registries:
        from . import rewards
        from .tasks import EnvironmentNotFound, get_environment
        try:
            env = get_environment(cfg['environment'].get('name', ''))
            problems += env.validate_experiment(cfg)
            if isinstance(components, list) and all(isinstance(x, dict) and 'fn' in x for x in components):
                problems += rewards.validate_components(env, components)
        except EnvironmentNotFound as error:
            problems.append(str(error))
    if problems:
        raise ExperimentError(problems)


def _scaling_problems(cfg: dict, gpus: int) -> list[str]:
    """finetune / compute.training / inference scaling knobs (types, ranges, divisibility)."""
    problems = []
    ft = cfg.get('finetune') or {}
    mode = ft.get('mode', 'full')
    if mode not in {'full', 'lora'}:
        problems.append(f'finetune.mode must be full or lora, got {mode!r}')
    lora = ft.get('lora') or {}
    for key, value in lora.items():
        if key not in LORA_TYPES:
            problems.append(f'unknown key finetune.lora.{key} (allowed: {sorted(LORA_TYPES)})')
        elif not _type_ok(value, LORA_TYPES[key]) and not (type(None) in LORA_TYPES[key] and value is None):
            problems.append(f'finetune.lora.{key} has the wrong type: {value!r}')
    rank = lora.get('rank', 0)
    if isinstance(rank, int) and not isinstance(rank, bool):
        if mode == 'lora' and rank <= 0:
            problems.append('finetune.mode lora needs finetune.lora.rank > 0')
        if mode == 'full' and rank > 0:
            problems.append('finetune.lora.rank > 0 is only valid with finetune.mode: lora')
    if isinstance(lora.get('dropout', 0.0), (int, float)) and not 0 <= lora.get('dropout', 0.0) < 1:
        problems.append('finetune.lora.dropout must be in [0, 1)')
    tr = cfg['compute'].get('training') or {}
    for key, value in tr.items():
        if key not in TRAINING_TYPES:
            problems.append(f'unknown key compute.training.{key} (allowed: {sorted(TRAINING_TYPES)})')
        elif not (_type_ok(value, TRAINING_TYPES[key]) or (value is None and type(None) in TRAINING_TYPES[key])):
            problems.append(f'compute.training.{key} has the wrong type: {value!r}')
    fsdp = tr.get('fsdp_size')
    if isinstance(fsdp, int) and not isinstance(fsdp, bool) and (fsdp < 1 or gpus % fsdp):
        problems.append(f'compute.training.fsdp_size={fsdp} must be a positive divisor of the {gpus} GPUs (null = full shard)')
    if tr.get('reduce_dtype', 'fp32') not in {'fp32', 'bf16'}:
        problems.append('compute.training.reduce_dtype must be fp32 or bf16')
    micro = tr.get('max_tokens_per_microbatch')
    if isinstance(micro, int) and not isinstance(micro, bool) and micro < 1:
        problems.append('compute.training.max_tokens_per_microbatch must be positive (null = sample-count micro-batches)')
    inf = cfg['inference']
    tp = inf['tensor_parallel_size']
    if 'num_engines' in inf and inf['num_engines'] * tp != gpus:
        problems.append(f'inference.num_engines={inf["num_engines"]} x tensor_parallel_size={tp} must equal the {gpus} colocated GPUs')
    if 'max_num_batched_tokens' in inf and inf['max_num_batched_tokens'] < 1:
        problems.append('inference.max_num_batched_tokens must be positive')
    return problems


def load(path: str | Path, *, check_registries: bool = True, overrides: list[str] | None = None) -> dict:
    """Resolve, apply ``--set`` overrides (dotted KEY=VALUE, YAML-parsed, recorded in _meta and the sha), fill defaults, validate."""
    raw = resolve(path)
    chain = raw.pop('_chain')
    overrides = list(overrides or [])
    if overrides:
        raw = apply_overrides(raw, overrides)
    raw.setdefault('name', Path(path).stem)
    cfg = copy.deepcopy(raw)
    problems = check_schema(cfg)
    if problems:
        raise ExperimentError(problems)
    cfg['run'].setdefault('data_dir', 'assets/data')
    cfg['run'].setdefault('artifact_dir', 'artifacts')
    cfg['model'].setdefault('local_path', 'assets/model')
    cfg['model'].setdefault('dtype', 'bfloat16')
    cfg['model'].setdefault('profile', 'auto')
    cfg['limits'].setdefault('models', ['*'])
    _absolutize(cfg, REPO_ROOT)
    cfg['algorithm'].setdefault('zero_reward_on_non_stop', True)
    cfg['algorithm'].setdefault('eval_temperature', 0.0)
    cfg['inference'].setdefault('max_env_workers', 64)
    cfg['evaluation'].setdefault('final_eval', True)
    cfg['tracking'].setdefault('alerts', True)
    cfg['budget'].setdefault('stall_seconds', 600)
    cfg['budget'].setdefault('ledger', 'evidence/budget-ledger.json')
    validate(cfg, check_registries=check_registries)
    cfg['derived'] = derive_placement(cfg)
    cfg['_meta'] = {'chain': chain, 'sha': experiment_sha(cfg, overrides), 'repo_root': str(REPO_ROOT)}
    if overrides:
        cfg['_meta']['overrides'] = overrides
    return cfg


def experiment_sha(cfg: dict, overrides: list[str] | None = None) -> str:
    """Content hash of the resolved experiment (paths made repo-relative so it is machine independent)."""
    payload = {k: v for k, v in cfg.items() if not k.startswith('_') and k != 'derived'}
    if overrides:
        payload['_overrides'] = list(overrides)
    text = json.dumps(payload, sort_keys=True, default=str).replace(str(REPO_ROOT), '<REPO>')
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def flatten(value: Any, prefix: str = '') -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            if str(k).startswith('_'):
                continue
            out.update(flatten(v, f'{prefix}{k}.' if prefix else f'{k}.'))
    else:
        out[prefix.rstrip('.')] = value
    return out


def diff(a: dict, b: dict) -> list[tuple[str, Any, Any]]:
    """Leaf-level differences between two resolved experiments: (key, a_value, b_value)."""
    fa, fb = flatten({k: v for k, v in a.items() if k != 'derived'}), flatten({k: v for k, v in b.items() if k != 'derived'})
    keys = sorted(set(fa) | set(fb))
    return [(k, fa.get(k), fb.get(k)) for k in keys if fa.get(k) != fb.get(k) and k not in {'name', 'description', 'extends'}]


# ---------------------------------------------------------------------------------------------- resolved persistence
RESOLVED_ENV = 'STRL_RESOLVED'


def save_resolved(cfg: dict, path: str | Path, *, root: str | None = None) -> None:
    """Write the resolved experiment; ``root`` re-roots repo-absolute paths (e.g. at the pod working directory)."""
    text = json.dumps(cfg, indent=2, sort_keys=True, default=str)
    if root is not None:
        text = text.replace(str(REPO_ROOT), root.rstrip('/'))
    Path(path).write_text(text + '\n')


def load_resolved(path: str | Path) -> dict:
    """Load a resolved experiment JSON written by ``save_resolved`` (no re-resolution: what ran is what was staged)."""
    return json.loads(Path(path).read_text())


def current() -> dict:
    """The experiment of the running process: $STRL_RESOLVED, else ./resolved.json."""
    import os
    return load_resolved(os.environ.get(RESOLVED_ENV) or 'resolved.json')
