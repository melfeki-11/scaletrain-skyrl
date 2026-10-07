from pathlib import Path
import math


def verify_export_change(base, export):
    from safetensors import safe_open
    base_files = list(Path(base).glob('*.safetensors'))
    files = list(Path(export).glob('*.safetensors'))
    if not files or not (Path(export)/'config.json').is_file():
        raise RuntimeError('No HF checkpoint export')
    def samples(paths):
        result={}
        for path in paths:
            with safe_open(path,framework='pt',device='cpu') as f:
                for key in f.keys():
                    if 'visual' in key or 'embed' in key or 'lm_head' in key:continue
                    norm=key.replace('model.language_model.','model.')
                    tensor=f.get_slice(key)
                    shape=tensor.get_shape()
                    if len(shape)==2 and shape[0]>0 and shape[1]>0:
                        result[norm]=tensor[:2,:32].float()
        return result
    before,after=samples(base_files),samples(files)
    common=set(before)&set(after)
    changes=[]
    for key in common:
        if before[key].shape!=after[key].shape:continue
        diff=(after[key]-before[key]).abs().max().item()
        if not math.isfinite(diff):raise RuntimeError('Nonfinite checkpoint parameter')
        changes.append((key,diff))
    changed=[item for item in changes if item[1]>0]
    if not changed:raise RuntimeError('Checkpoint parameters did not change')
    return {'compared_tensors':len(changes),'changed_tensors':len(changed),'max_absolute_change':max(x[1] for x in changed)}


def verify_shards(root,step,world_size=16):
    path=Path(root)/'checkpoints'/f'global_step_{step}'/'policy'
    missing=[]
    for kind in ('model','optim','extra_state'):
        for rank in range(world_size):
            p=path/f'{kind}_world_size_{world_size}_rank_{rank}.pt'
            if not p.is_file() or p.stat().st_size==0:missing.append(p.name)
    if missing:raise RuntimeError('Missing distributed checkpoint shards: '+','.join(missing))
    return {'world_size':world_size,'nonempty_rank_files':3*world_size,'path':str(path)}


SHARD_KINDS = ('model', 'optim', 'extra_state')


def local_shards(root, step, world_size):
    """The nonempty FSDP rank files THIS node holds (multi-node: each node writes the shards of the ranks it hosts)."""
    import socket
    path = Path(root)/'checkpoints'/f'global_step_{step}'/'policy'
    def present(kind, rank):
        p = path/f'{kind}_world_size_{world_size}_rank_{rank}.pt'
        return p.is_file() and p.stat().st_size > 0
    return {'hostname': socket.gethostname(), 'path': str(path),
            'ranks': {kind: [r for r in range(world_size) if present(kind, r)] for kind in SHARD_KINDS}}


def verify_cluster_shards(reports, world_size):
    """Every rank's model/optim/extra_state shard exists on SOME node; no shard has to be copied to the head."""
    missing = [f'{kind}_rank_{r}' for kind in SHARD_KINDS for r in range(world_size)
               if not any(r in report['ranks'][kind] for report in reports)]
    if missing:
        raise RuntimeError('Missing distributed checkpoint shards across nodes: ' + ','.join(missing))
    return {'world_size': world_size, 'nonempty_rank_files': 3*world_size, 'nodes': len(reports),
            'by_host': {r['hostname']: sum(len(v) for v in r['ranks'].values()) for r in reports}}


def verify_adapter_change(export):
    """A LoRA export changed the policy iff some lora_B is nonzero (PEFT initialises B to zero); every tensor must be finite."""
    import torch
    from safetensors import safe_open
    path = Path(export)/'adapter_model.safetensors'
    if not path.is_file() or not (Path(export)/'adapter_config.json').is_file():
        raise RuntimeError('No LoRA adapter export')
    count = changed = 0
    largest = 0.0
    with safe_open(path, framework='pt', device='cpu') as f:
        for key in f.keys():
            tensor = f.get_tensor(key).float()
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f'Nonfinite adapter parameter {key}')
            count += 1
            if 'lora_B' in key and tensor.abs().max().item() > 0:
                changed += 1
                largest = max(largest, tensor.abs().max().item())
    if not changed:
        raise RuntimeError('LoRA adapter did not change (every lora_B is zero)')
    return {'adapter_tensors': count, 'changed_lora_B': changed, 'max_abs_lora_B': largest}
