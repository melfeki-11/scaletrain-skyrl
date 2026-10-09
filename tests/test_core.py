import ast
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from strl.tasks.gsm8k.answers import parse_final_answer,score_answer
from strl.experiment import load as _load
load=lambda path='base/gsm8k.yaml':_load(path)
from strl.tasks.gsm8k.data import split_indices,prepare_dataset,assert_manifest
from strl.overrides import overrides
from strl.stats import paired_bootstrap
from strl.verify import verify_export_change,verify_shards
from strl.tracking import rollout_diagnostics
from strl.network import routable,ray_start_args,head_network
from strl.budget import remaining_seconds


class Contracts(unittest.TestCase):
    def test_checkpoint_wrapper_preserves_trained_tensors(self):
        import torch
        from safetensors.torch import save_file,load_file
        from strl.models.qwen3_5_export import adapt
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'base';export=root/'export';base.mkdir();export.mkdir()
            config=json.loads(Path('tests/fixtures/configs/qwen3_5.json').read_text());text=dict(config['text_config'])
            text.update(model_type='qwen3_5_text',architectures=['Qwen3_5ForCausalLM'])
            (base/'config.json').write_text(json.dumps(config));(export/'config.json').write_text(json.dumps(text))
            weights={'model.layers.0.linear_attn.in_proj_qkv.weight':torch.arange(160,dtype=torch.float32).reshape(4,40),'model.language_model.layers.1.mlp.up_proj.weight':torch.ones(4,40),'lm_head.weight':torch.ones(8,8)}
            save_file(weights,str(export/'model.safetensors'))
            target=root/'adapted';receipt=adapt(base,export,target);after=load_file(target/'model.safetensors')
            self.assertEqual(receipt['tensors_renamed'],1)
            for key,value in weights.items():
                mapped='model.language_model.'+key[6:] if key.startswith('model.') and not key.startswith('model.language_model.') else key
                self.assertTrue(torch.equal(value,after[mapped]))
            wrapped=json.loads((target/'config.json').read_text())
            self.assertEqual(wrapped['vision_config'],config['vision_config']);self.assertEqual(wrapped['text_config'],text)
            with self.assertRaises(ValueError):adapt(base,export,export)
            with self.assertRaises(ValueError):adapt(base,export,target)
    def test_reject_local_only_rendezvous(self):
        for ip in ['169.254.173.89','127.0.0.1','fe80::1','::','0.0.0.0']:
            with self.assertRaises(ValueError):routable(ip)
        self.assertEqual(routable('10.1.2.3'),'10.1.2.3')
    def test_disabled_and_ipv6_metrics_endpoints(self):
        from strl.ipv6_patch import PATCHES
        source=Path('pinned/pinned-vllm_metrics_scraper.py').read_text()
        for before,after in PATCHES['skyrl/train/utils/vllm_metrics_scraper.py']:
            self.assertEqual(source.count(before),1);source=source.replace(before,after,1)
        tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='discover_ray_metrics_urls')
        import types
        namespace={'List':list,'ray':types.SimpleNamespace(nodes=lambda:[{'Alive':True,'NodeManagerAddress':'2602:fb33::1','MetricsExportPort':-1},{'Alive':True,'NodeManagerAddress':'2602:fb33::2','MetricsExportPort':8080}])}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'metrics','exec'),namespace)
        self.assertEqual(namespace['discover_ray_metrics_urls'](),['http://[2602:fb33::2]:8080/metrics'])
    def test_verifier(self):
        self.assertEqual(parse_final_answer('work\n#### 1,200'),'1200')
        self.assertEqual(score_answer('work\n#### 38<|im_end|>','38','stop'),1)
        self.assertEqual(score_answer('#### 38<|im_end|> extra','38','stop'),0)
        self.assertEqual(score_answer('#### 38<|im_end|>','38','length'),0)
        for value in ['#### 42\nextra','#### 42\n#### 42','42','#### nan']:
            self.assertIsNone(parse_final_answer(value))
        self.assertEqual(score_answer('#### 42','42','length'),0)
    def test_split_no_leak(self):
        train,val=split_indices(7473,256,17)
        self.assertFalse(set(train)&set(val));self.assertEqual(len(train)+len(val),7473)
    def test_dataset_tamper(self):
        cfg=load()
        with tempfile.TemporaryDirectory() as temp:
            cfg['run']['data_dir']=temp;cfg['environment']['dataset']['validation_size']=2
            dataset={'train':[{'question':str(i),'answer':'#### 42'} for i in range(6)],'test':[{'question':'test','answer':'#### 17'}]}
            prepare_dataset(cfg,dataset=dataset);assert_manifest(cfg)
            Path(temp,'test.parquet').write_bytes(b'bad')
            with self.assertRaises(RuntimeError):assert_manifest(cfg)
    def test_pinned_schema(self):
        tree=ast.parse(Path('pinned/pinned-config.py').read_text())
        classes={c.name:{f.target.id:ast.unparse(f.annotation) for f in c.body if isinstance(f,ast.AnnAssign) and isinstance(f.target,ast.Name)} for c in tree.body if isinstance(c,ast.ClassDef)}
        for option in overrides(load(),stage='train',run_name='test'):
            current='SkyRLTrainConfig'
            for field in option.split('=',1)[0].split('.'):
                if current not in classes:
                    self.assertIn('dict',current.lower());break
                self.assertIn(field,classes[current]);current=classes[current][field].replace('Optional[','').rstrip(']')
    def test_absolute_actor_paths(self):
        cfg=load()
        for section,key in [('run','data_dir'),('run','artifact_dir'),('model','local_path')]:
            self.assertTrue(Path(cfg[section][key]).is_absolute())
    def test_single_node_head_ray(self):
        with patch('strl.network.interface_addresses',return_value=['2602:fb33::f']):
            local,address=head_network('eth0',6379)
        self.assertEqual((local,address),('2602:fb33::f','[2602:fb33::f]:6379'))
        args=ray_start_args(0,address,6379,local,192,8)
        self.assertIn('--node-ip-address=2602:fb33::f',args);self.assertIn('--head',args);self.assertIn('--num-gpus=8',args);self.assertIn('--num-cpus=192',args)
    def test_budget_one_hour(self):
        attempts=[{'job_id':'queued-canceled','status':'CANCELED'},
                  {'job_id':'failed','status':'FAILED','allocated_at':'2026-10-04T11:31:01Z','terminal_at':'2026-10-04T11:37:18Z'}]
        self.assertEqual(remaining_seconds(attempts),3600-377)
        with self.assertRaises(ValueError):remaining_seconds(attempts+[attempts[1]])
        with self.assertRaises(ValueError):remaining_seconds([{'job_id':'queued','status':'QUEUED'}])
    def test_config_authorization_envelope(self):
        import yaml
        from strl.experiment import ExperimentError
        cfg=load();self.assertEqual((cfg['compute']['nodes'],cfg['compute']['gpus_per_node'],cfg['derived']['num_engines']),(1,8,8))
        for mutate in [lambda c:c['compute'].update(nodes=2),lambda c:c['budget'].update(attempt_max_seconds=7200,total_seconds=7200),
                       lambda c:c['algorithm'].update(micro_train_per_gpu=3),lambda c:c['inference'].update(tensor_parallel_size=3),
                       lambda c:c['evaluation'].update(eval_shards_per_model=0),lambda c:c['compute'].update(gpus_per_node=3),
                       lambda c:c['algorithm'].update(n_samples=1)]:
            bad=load();mutate(bad)
            for k in [k for k in bad if k.startswith('_') or k=='derived']:bad.pop(k)
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp,'exp.yaml');path.write_text(yaml.safe_dump(bad))
                with self.assertRaises(ExperimentError):load(path)
    def test_single_node_overrides(self):
        cfg=load();values=dict(v.split('=',1) for v in overrides(cfg,stage='train',run_name='test'))
        self.assertEqual(values['trainer.placement.policy_num_nodes'],'1');self.assertEqual(values['trainer.placement.ref_num_nodes'],'1')
        self.assertEqual(values['trainer.placement.policy_num_gpus_per_node'],'8');self.assertEqual(values['generator.inference_engine.num_engines'],'8')
        self.assertEqual(values['generator.batched'],'false');self.assertEqual(values['generator.chat_template_kwargs'],"{'enable_thinking': False}")
        self.assertEqual(values['trainer.remove_microbatch_padding'],'false');self.assertNotIn('trainer.policy.model.lora.rank',values)
        # 64 prompts x 8 samples = 512 rollouts = 64 per rank, evenly divisible by the micro-batch
        per_rank=int(values['trainer.policy_mini_batch_size'])*int(values['generator.n_samples_per_prompt'])//8
        self.assertEqual(per_rank%int(values['trainer.micro_train_batch_size_per_gpu']),0)
    def test_allocated_cpus_uses_cgroup_quota(self):
        from strl.resources import allocated_cpus,cgroup_quota
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp,'cpu.max');path.write_text('1250000 100000\n');self.assertEqual(cgroup_quota(path),13)
            with patch('os.sched_getaffinity',create=True,return_value=set(range(192))):self.assertEqual(allocated_cpus(path),13)
            path.write_text('max 100000\n')
            with patch('os.sched_getaffinity',create=True,return_value=set(range(192))):self.assertEqual(allocated_cpus(path),192)
            self.assertIsNone(cgroup_quota(Path(temp,'missing')))
    def test_shard_reassembly_and_flips(self):
        from strl.finalize import merge_shards,flips,select_examples
        ids=[f'gsm8k-test-{i:05d}' for i in range(8)]
        def row(i,correct,**kw):return dict({'sample_id':ids[i],'prompt':'q','ground_truth':'1','completion':'c','correct':correct,'finish_reason':'stop','parsed_answer':'1' if correct else '2'},**kw)
        shards=[[row(i,True) for i in range(r,8,4)] for r in range(4)]
        merged=merge_shards(shards,ids);self.assertEqual([r['sample_id'] for r in merged],ids)
        with self.assertRaises(RuntimeError):merge_shards(shards[:3],ids)
        with self.assertRaises(RuntimeError):merge_shards(shards+[[row(0,True)]],ids)
        base=[row(0,False),row(1,False,parsed_answer=None),row(2,True),row(3,True),row(4,False,finish_reason='length'),row(5,False),row(6,True),row(7,False)]
        trained=[row(0,True),row(1,True),row(2,True),row(3,False),row(4,True),row(5,False),row(6,True),row(7,False)]
        self.assertEqual(flips(base,trained),{'fixed':3,'broken':1,'both_correct':2,'both_wrong':2})
        fixed=select_examples(base,trained,'fixed');self.assertEqual([r['sample_id'] for r in fixed],[ids[0],ids[1],ids[4]])
        self.assertEqual([r['base_failure_mode'] for r in fixed],['wrong_answer','no_final_answer_line','truncated'])
        self.assertEqual([r['sample_id'] for r in select_examples(base,trained,'broken')],[ids[3]])
        with self.assertRaises(RuntimeError):flips(base,list(reversed(trained)))
    def test_sync_excludes_optimizer_shards(self):
        from strl.storage import sync_local
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'checkpoints/global_step_2').mkdir(parents=True);(root/'exports').mkdir()
            (root/'checkpoints/global_step_2/optim_world_size_8_rank_0.pt').write_bytes(b'x');(root/'exports/metrics.jsonl').write_text('{}')
            with patch('strl.storage.upload_file') as up:
                sync_local(root,'s3://b/p',exclude=('checkpoints/',))
            self.assertEqual([c.args[1] for c in up.call_args_list],['s3://b/p/exports/metrics.jsonl'])
    def test_signal(self):
        stats=rollout_diagnostics(['#### 42']*4,[0,1,0,0],['stop']*4,2)
        self.assertEqual(stats['train/learning_signal_fraction'],.5)
    def test_paired_evaluation(self):
        stats=paired_bootstrap([False]*50,[True]*50,samples=100)
        self.assertEqual(stats['accuracy_delta'],1);self.assertTrue(stats['quality_passed'])
    def test_checkpoint_change(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp,'base');export=Path(temp,'export');base.mkdir();export.mkdir()
            save_file({'model.language_model.layers.0.proj.weight':torch.zeros(3,40)},str(base/'model.safetensors'))
            save_file({'model.layers.0.proj.weight':torch.ones(3,40)},str(export/'model.safetensors'))
            (export/'config.json').write_text('{}')
            self.assertEqual(verify_export_change(base,export)['changed_tensors'],1)
            save_file({'model.layers.0.proj.weight':torch.zeros(3,40)},str(export/'model.safetensors'))
            with self.assertRaises(RuntimeError):verify_export_change(base,export)
    def test_eight_rank_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp,'checkpoints/global_step_2/policy');root.mkdir(parents=True)
            for kind in ('model','optim','extra_state'):
                for rank in range(8):(root/f'{kind}_world_size_8_rank_{rank}.pt').write_bytes(b'x')
            self.assertEqual(verify_shards(temp,2,8)['nonempty_rank_files'],24)

if __name__=='__main__':unittest.main()


class AllExperiments(unittest.TestCase):
    """Every shipped experiment must plan, and every SkyRL override it renders must exist in the pinned dataclasses."""

    def test_every_experiment_overrides_exist_in_pinned_skyrl(self):
        from strl.derive import plan
        tree = ast.parse(Path('pinned/pinned-config.py').read_text())
        classes = {c.name: {f.target.id: ast.unparse(f.annotation) for f in c.body if isinstance(f, ast.AnnAssign) and isinstance(f.target, ast.Name)}
                   for c in tree.body if isinstance(c, ast.ClassDef)}
        for path in sorted(Path('experiments').glob('*.yaml')) + sorted(Path('base').glob('[!_]*.yaml')):
            if path.name in {'infra.yaml', 'canary_overlay.yaml', 'canary_overlay_large.yaml'}:
                continue
            cfg = load(str(path))
            plan(cfg)
            for stage in ('train', 'canary'):
                for option in overrides(cfg, stage=stage, run_name='t'):
                    key = option.split('=', 1)[0]
                    if key.startswith('generator.rate_limit.'):
                        continue  # HarborGeneratorConfig (examples/train_integrations/harbor/entrypoints/main_harbor.py)
                    current = 'SkyRLTrainConfig'
                    for field in key.split('.'):
                        if current not in classes:
                            self.assertIn('dict', current.lower(), (path, key))
                            break
                        self.assertIn(field, classes[current], (path, key))
                        current = classes[current][field].replace('Optional[', '').rstrip(']')

    def test_canary_overlay_keeps_axes(self):
        from strl.stage import apply_canary_overlay
        for path in ('experiments/mbpp-qwen35-partial.yaml', 'experiments/mbpp-qwen3-1p7b.yaml', 'base/gsm8k.yaml'):
            cfg = load(path)
            canary = apply_canary_overlay(cfg)
            for axis in ('model', 'environment', 'reward'):
                self.assertEqual(canary[axis], cfg[axis], (path, axis))
            self.assertEqual(canary['compute']['gpus_per_node'], 1)
            self.assertFalse(canary['evaluation']['final_eval'])


class GSM8KTerminalTokens(unittest.TestCase):
    """Generality audit: parse_final_answer returned None for every non-Qwen model (it stripped only <|endoftext|> and <|im_end|>)."""

    def test_every_common_family_terminator_is_stripped(self):
        from strl.tasks.gsm8k.answers import parse_final_answer
        for end in ('', '<|im_end|>', '<|eot_id|>', '</s>', '<end_of_turn>', '<eos>', '<|end|>', '<｜end▁of▁sentence｜>', '<|im_end|><|endoftext|>'):
            self.assertEqual(parse_final_answer('2 + 2 = 4\n#### 4' + end), '4', repr(end))
        self.assertIsNone(parse_final_answer('#### 4</s> trailing'))       # only a trailing run is removed
