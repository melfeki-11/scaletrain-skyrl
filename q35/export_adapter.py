"""Wrap a SkyRL text-only Qwen3.5 export for the pinned vLLM loader."""
import json
from pathlib import Path


def adapt(base,export,target):
    from safetensors import safe_open
    from safetensors.torch import save_file
    base,export,target=map(Path,[base,export,target])
    if target.resolve() in {base.resolve(),export.resolve()}:raise ValueError('Adaptation needs a separate directory')
    paths=sorted(export.glob('*.safetensors'))
    if len(paths)!=1 or (export/'model.safetensors.index.json').exists():
        raise ValueError('Expected one unsharded safetensors export')
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
    count=renamed=0
    for path in paths:
        tensors={}
        with safe_open(path,framework='pt',device='cpu') as reader:
            for key in reader.keys():
                new=('model.language_model.'+key[len('model.'):]
                     if key.startswith('model.') and not key.startswith('model.language_model.') else key)
                if new in tensors:raise ValueError('Duplicate adapted tensor name')
                tensors[new]=reader.get_tensor(key);count+=1;renamed+=new!=key
        save_file(tensors,str(target/path.name),metadata={'format':'pt'})
    if count==0:raise ValueError('Empty checkpoint export')
    result={'passed':True,'tensor_count':count,'tensors_renamed':renamed,'weights_preserved':True,'source':str(export),'target':str(target)}
    (target/'adapter.json').write_text(json.dumps(result,indent=2));return result
