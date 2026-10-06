from pathlib import Path
import yaml


def load(path='config.yaml'):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg['_config_path'] = str(Path(path).resolve())
    nodes = cfg['scale_train']['num_instances']
    gpus = cfg['scale_train']['gpu_count']
    t, i = cfg['training'], cfg['inference']
    per_node = cfg['scale_train']['gpus_per_node']
    if nodes != 1 or per_node not in {1, 2, 4, 8} or gpus != per_node or cfg['budget']['total_seconds'] > 3600:
        raise ValueError('This experiment is authorized for one node and at most one hour')
    if i['num_engines'] * i['tensor_parallel_size'] != gpus:
        raise ValueError('Colocated inference and training GPU counts must match')
    expanded = t['policy_mini_batch_size'] * t['grpo']['samples_per_prompt']
    if expanded % gpus or (expanded // gpus) % t['micro_train_batch_size_per_gpu']:
        raise ValueError('Expanded mini-batch must divide all GPU micro-batches exactly')
    if (t['train_batch_size'] * t['grpo']['samples_per_prompt']) % gpus or (expanded // gpus) % t['micro_forward_batch_size_per_gpu']:
        raise ValueError('Expanded train batch must divide all GPU micro-batches exactly')
    if t['train_batch_size'] % t['policy_mini_batch_size']:
        raise ValueError('Train batch must divide mini-batch')
    if cfg['evaluation']['eval_shards_per_model'] * 2 > gpus:
        raise ValueError('Parallel final evaluation needs one GPU per shard for both models')
    if cfg['model']['id'] != 'Qwen/Qwen3.5-0.8B':
        raise ValueError('Unexpected model')
    root = Path(cfg['_config_path']).parent
    for section, key in [('run', 'data_dir'), ('run', 'artifact_dir'), ('model', 'local_path')]:
        value = Path(cfg[section][key])
        cfg[section][key] = str(value if value.is_absolute() else root / value)
    return cfg
