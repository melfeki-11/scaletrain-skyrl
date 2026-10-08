"""LoRA exports: keep the adapter as SkyRL wrote it (PEFT adapter_model.safetensors + adapter_config.json) and record which base
it applies to; merge it into a full checkpoint when an evaluator needs plain weights (the offline final eval loads one model dir).
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

ADAPTER_FILES = ('adapter_model.safetensors', 'adapter_config.json')
LORA_KEY = re.compile(r'^(?:base_model\.model\.)?(.+)\.lora_([AB])(?:\.default)?\.weight$')


def is_adapter(export: str | Path) -> bool:
    return all((Path(export) / name).is_file() for name in ADAPTER_FILES)


def _empty_target(target: Path, *sources: Path) -> None:
    if target.resolve() in {s.resolve() for s in sources}:
        raise ValueError('Adaptation needs a separate directory')
    if target.exists() and any(target.iterdir()):
        raise ValueError('Target must be empty')
    target.mkdir(parents=True, exist_ok=True)


def keep_adapter(export, target, base_model_id=None, base_revision=None, base_path=None) -> dict:
    """Copy the adapter unchanged and write adapter.json naming the base model, so an evaluator can load base + adapter."""
    export, target = Path(export), Path(target)
    if not is_adapter(export):
        raise ValueError('Expected a PEFT adapter export (adapter_model.safetensors + adapter_config.json)')
    _empty_target(target, export)
    for name in ADAPTER_FILES:
        shutil.copy2(export / name, target / name)
    config = json.loads((export / 'adapter_config.json').read_text())
    receipt = {'passed': True, 'kind': 'lora', 'base_model_id': base_model_id, 'base_model_revision': base_revision,
               'base_path': str(base_path) if base_path else config.get('base_model_name_or_path'), 'r': config.get('r'),
               'lora_alpha': config.get('lora_alpha'), 'target_modules': config.get('target_modules'), 'source': str(export),
               'target': str(target), 'weights_preserved': True}
    (target / 'adapter.json').write_text(json.dumps(receipt, indent=2, default=str))
    return receipt


def weight_files(model_dir: Path) -> list[Path]:
    index = model_dir / 'model.safetensors.index.json'
    if index.is_file():
        return [model_dir / name for name in sorted(set(json.loads(index.read_text())['weight_map'].values()))]
    return sorted(model_dir.glob('*.safetensors'))


def merge_adapter(base, adapter, target) -> dict:
    """W' = W + scale * B @ A for every adapted linear (scale = alpha / r, or alpha / sqrt(r) with rsLoRA), shard by shard, so a
    55 GB base never has to fit in memory at once. Non-weight files (configs, tokenizer, index) are copied unchanged."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    base, adapter, target = map(Path, (base, adapter, target))
    config = json.loads((adapter / 'adapter_config.json').read_text())
    rank = int(config['r'])
    scale = float(config['lora_alpha']) / (math.sqrt(rank) if config.get('use_rslora') else rank)
    pairs: dict[str, dict] = {}
    with safe_open(adapter / 'adapter_model.safetensors', framework='pt', device='cpu') as reader:
        for key in reader.keys():
            match = LORA_KEY.match(key)
            if not match:
                raise ValueError(f'Unexpected adapter tensor {key}')
            pairs.setdefault(match.group(1), {})[match.group(2)] = reader.get_tensor(key)
    shards = weight_files(base)
    names = set()
    for path in shards:
        with safe_open(path, framework='pt', device='cpu') as reader:
            names.update(reader.keys())
    targets = {}
    for module, ab in pairs.items():
        if set(ab) != {'A', 'B'}:
            raise ValueError(f'Incomplete LoRA pair for {module}')
        # the trainer sees the text-only model (model.layers.*); a VLM wrapper checkpoint stores model.language_model.layers.*
        options = [f'{module}.weight'] + ([f'model.language_model.{module[len("model."):]}.weight'] if module.startswith('model.') else [])
        found = [name for name in options if name in names]
        if not found:
            raise ValueError(f'No base weight for adapted module {module}')
        targets[found[0]] = ab
    _empty_target(target, base, adapter)
    for item in base.iterdir():
        if item.is_file() and item.suffix != '.safetensors':
            shutil.copy2(item, target / item.name)
    merged = 0
    for path in shards:
        tensors = {}
        with safe_open(path, framework='pt', device='cpu') as reader:
            for key in reader.keys():
                tensor = reader.get_tensor(key)
                if key in targets:
                    a, b = targets[key]['A'].float(), targets[key]['B'].float()
                    tensor = (tensor.float() + scale * (b @ a)).to(tensor.dtype)
                    merged += 1
                tensors[key] = tensor
        save_file(tensors, str(target / path.name), metadata={'format': 'pt'})
    if merged != len(targets):
        raise ValueError('Not every adapted weight was merged')
    receipt = {'passed': True, 'kind': 'lora_merged', 'merged_tensors': merged, 'scale': scale, 'base': str(base), 'adapter': str(adapter),
               'target': str(target)}
    (target / 'merge.json').write_text(json.dumps(receipt, indent=2))
    return receipt
