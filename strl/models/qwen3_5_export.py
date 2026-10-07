"""Wrap a SkyRL text-only Qwen3.5 export for the pinned vLLM loader.

Accepts a single model.safetensors or a sharded export (model.safetensors.index.json + shards; large models), and a LoRA
export (adapter_model.safetensors + adapter_config.json), which is kept as-is with its base model recorded (strl/lora.py).
"""
import json
from pathlib import Path


def rename(key):
    return 'model.language_model.'+key[len('model.'):] if key.startswith('model.') and not key.startswith('model.language_model.') else key


def adapt(base,export,target,base_model_id=None,base_revision=None):
    from safetensors import safe_open
    from safetensors.torch import save_file
    base,export,target=map(Path,[base,export,target])
    if target.resolve() in {base.resolve(),export.resolve()}:raise ValueError('Adaptation needs a separate directory')
    from ..lora import is_adapter,keep_adapter
    if is_adapter(export):return keep_adapter(export,target,base_model_id,base_revision,base)
    paths=sorted(export.glob('*.safetensors'))
    index_path=export/'model.safetensors.index.json'
    index=json.loads(index_path.read_text()) if index_path.exists() else None
    if index is not None:
        if not paths or set(index['weight_map'].values())!={p.name for p in paths}:raise ValueError('Sharded export index does not match its shard files')
    elif len(paths)!=1:
        raise ValueError('Expected one safetensors export or a model.safetensors.index.json with its shards')
    if target.exists() and any(target.iterdir()):raise ValueError('Target must be empty')
    text=json.loads((export/'config.json').read_text())
    if text.get('model_type')!='qwen3_5_text':raise ValueError('Expected Qwen3.5 text export')
    config=json.loads((base/'config.json').read_text())
    if config.get('model_type')!='qwen3_5':raise ValueError('Expected matching Qwen3.5 wrapper config')
    for k in ['hidden_size','num_hidden_layers','vocab_size']:
        if text[k]!=config['text_config'][k]:raise ValueError('Export/base architecture mismatch')
    config['text_config']=text;config['tie_word_embeddings']=text.get('tie_word_embeddings',config.get('tie_word_embeddings'))
    target.mkdir(parents=True,exist_ok=True)
    (target/'config.json').write_text(json.dumps(config,indent=2))
    count=renamed=0;seen=set()
    for path in paths:
        tensors={}
        with safe_open(path,framework='pt',device='cpu') as reader:
            for key in reader.keys():
                new=rename(key)
                if new in seen:raise ValueError('Duplicate adapted tensor name')
                seen.add(new);tensors[new]=reader.get_tensor(key);count+=1;renamed+=new!=key
        save_file(tensors,str(target/path.name),metadata={'format':'pt'})
    if count==0:raise ValueError('Empty checkpoint export')
    if index is not None:
        weight_map={rename(k):v for k,v in index['weight_map'].items()}
        if set(weight_map)!=seen:raise ValueError('Sharded export index does not list exactly the exported tensors')
        (target/index_path.name).write_text(json.dumps(dict(index,weight_map=weight_map),indent=2))
    result={'passed':True,'tensor_count':count,'tensors_renamed':renamed,'weights_preserved':True,'shards':len(paths),'source':str(export),'target':str(target)}
    (target/'adapter.json').write_text(json.dumps(result,indent=2));return result
