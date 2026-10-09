"""Multi-node + LoRA/full scaling contracts: rendezvous, liveness, budget node accounting, --set overrides, override rendering,
memory model, sharded/LoRA exports, canary overlay choice, exact-source SkyRL patches, probe parsing. No GPUs, Ray or S3 needed."""
import ast
import io
import json
import os
import shutil
import sys
import tempfile
import time
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from botocore.exceptions import ClientError  # noqa: E402

from strl.experiment import ExperimentError, load  # noqa: E402
from strl.overrides import overrides, scaling_values  # noqa: E402

TMP = ROOT / 'tests/.tmp/scaling'
SMOKE = 'experiments/gsm8k-qwen35-2node-smoke.yaml'
SMOKE_FULL = 'experiments/gsm8k-qwen35-2node-smoke-full.yaml'
Q27 = ROOT / 'tests/fixtures/configs/qwen3_8_27b.json'


class FakeS3:
    """dict-backed S3: put_object/get_object/head_object/upload_file with NoSuchKey errors like botocore."""

    def __init__(self):
        self.objects, self.calls = {}, []

    def _missing(self, op):
        return ClientError({'Error': {'Code': 'NoSuchKey', 'Message': 'missing'}, 'ResponseMetadata': {'HTTPStatusCode': 404}}, op)

    def put_object(self, Bucket, Key, Body=b'', **kw):
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode()
        self.calls.append(('put_object', Bucket, Key))

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise self._missing('GetObject')
        return {'Body': io.BytesIO(self.objects[(Bucket, Key)]), 'LastModified': datetime.now(timezone.utc)}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise self._missing('HeadObject')
        return {'Metadata': {}}

    def upload_file(self, path, bucket, key, ExtraArgs=None, Config=None):
        self.objects[(bucket, key)] = Path(path).read_bytes()
        self.calls.append(('upload_file', bucket, key))


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


PREFIX = 's3://bucket/attempts/a1/artifacts'


class Rendezvous(unittest.TestCase):
    def test_leader_record_round_trip_ipv6(self):
        from strl.multinode import LEADER, shared_job_network
        s3 = FakeS3()
        head = shared_job_network(0, 'eth0', 6379, PREFIX, time.time() + 5, s3=s3, addresses=['2602:fb33::a', '10.0.0.1'])
        self.assertEqual(head, ('2602:fb33::a', '2602:fb33::a', '[2602:fb33::a]:6379'))
        record = json.loads(s3.objects[('bucket', f'attempts/a1/artifacts/{LEADER}')])
        self.assertEqual((record['ip'], record['port']), ('2602:fb33::a', 6379))
        peer = shared_job_network(1, 'eth0', '6379', PREFIX, time.time() + 5, s3=s3, addresses=['10.0.0.2', '2602:fb33::b'])
        self.assertEqual(peer, ('2602:fb33::b', '2602:fb33::a', '[2602:fb33::a]:6379'))

    def test_address_family_selection_and_failures(self):
        from strl.multinode import shared_job_network
        s3 = FakeS3()
        shared_job_network(0, 'eth0', 6379, PREFIX, time.time() + 5, s3=s3, addresses=['10.0.0.1'])
        self.assertEqual(shared_job_network(1, 'eth0', 6379, PREFIX, time.time() + 5, s3=s3, addresses=['2602:fb33::b', '10.0.0.2'])[2],
                         '10.0.0.1:6379')
        with self.assertRaises(RuntimeError):   # no common family
            shared_job_network(1, 'eth0', 6379, PREFIX, time.time() + 5, s3=s3, addresses=['2602:fb33::b'])
        with self.assertRaises(RuntimeError):   # port mismatch
            shared_job_network(1, 'eth0', 7000, PREFIX, time.time() + 5, s3=s3, addresses=['10.0.0.2'])
        clock = Clock()
        with self.assertRaises(TimeoutError):   # head never published
            shared_job_network(1, 'eth0', 6379, 's3://bucket/other', clock() + 30, s3=FakeS3(), addresses=['10.0.0.2'],
                               sleep=clock.sleep, clock=clock)

    def test_rank_world_and_ray_join_retries(self):
        from strl.multinode import rank_and_world, start_ray
        self.assertEqual(rank_and_world({'JOB_COMPLETION_INDEX': '1', 'NUM_INSTANCES': '2'}), (1, 2))
        self.assertEqual(rank_and_world({}), (0, 1))
        results = iter([1, 1, 0])
        calls = []

        def run(args, **kw):
            calls.append(args)
            return types.SimpleNamespace(returncode=0 if 'stop' in args else next(results))
        tries = start_ray(1, '[2602:fb33::a]:6379', 6379, '2602:fb33::b', 18, 8, time.time() + 60, run=run, sleep=lambda s: None)
        self.assertEqual(tries, 3)
        self.assertIn('--address=[2602:fb33::a]:6379', calls[-1])
        self.assertIn('--num-cpus=18', calls[-1])
        with self.assertRaises(RuntimeError):
            start_ray(1, '10.0.0.1:6379', 6379, '10.0.0.2', 8, 8, time.time() + 60, sleep=lambda s: None, attempts=2,
                      run=lambda args, **kw: types.SimpleNamespace(returncode=0 if 'stop' in args else 1))

    def test_wait_for_cluster_and_watcher(self):
        from strl.multinode import Watcher, cluster_check, wait_for_cluster
        nodes = [{'Alive': True, 'NodeManagerAddress': '10.0.0.1'}, {'Alive': True, 'NodeManagerAddress': '10.0.0.2'}]
        fake = types.SimpleNamespace(nodes=lambda: nodes, cluster_resources=lambda: {'GPU': 16})
        self.assertEqual(wait_for_cluster(2, 8, 10, ray_module=fake)['gpus'], 16)
        clock = Clock()
        short = types.SimpleNamespace(nodes=lambda: nodes[:1], cluster_resources=lambda: {'GPU': 8})
        with self.assertRaises(RuntimeError):
            wait_for_cluster(2, 8, 10, ray_module=short, sleep=clock.sleep, clock=clock)
        watcher = Watcher(cluster_check(2, short))
        self.assertIsNone(watcher())
        misses = watcher.poll(0)
        self.assertIsNone(watcher.failure)          # one missed poll is tolerated
        watcher.poll(misses)
        self.assertIn('1 of 2', watcher())
        healthy = Watcher(cluster_check(2, fake))
        self.assertEqual(healthy.poll(1), 0)
        self.assertIsNone(healthy.failure)

    def test_peer_exits_on_completion_and_on_stale_head(self):
        from strl.multinode import HEAD_HEARTBEAT, HeadStale, peer_wait, put_record
        s3, clock = FakeS3(), Clock()
        put_record(f'{PREFIX}/{HEAD_HEARTBEAT}', {'at': clock()}, s3)
        put_record(f'{PREFIX}/completion.json', {'status': 'completed'}, s3)
        self.assertEqual(peer_wait(PREFIX, 1, clock() + 100, s3=s3, sleep=clock.sleep, clock=clock)['status'], 'completed')
        self.assertIn(('bucket', 'attempts/a1/artifacts/peer-1-heartbeat.json'), s3.objects)
        stale, clock = FakeS3(), Clock()
        put_record(f'{PREFIX}/{HEAD_HEARTBEAT}', {'at': clock() - 400}, stale)
        with self.assertRaises(HeadStale):
            peer_wait(PREFIX, 1, clock() + 1000, s3=stale, sleep=clock.sleep, clock=clock)
        fresh, clock = FakeS3(), Clock()   # heartbeats keep arriving: the peer keeps waiting until the deadline
        put_record(f'{PREFIX}/heartbeat.json', {'at': clock() + 10_000}, fresh)
        with self.assertRaises(TimeoutError):
            peer_wait(PREFIX, 1, clock() + 120, s3=fresh, sleep=clock.sleep, clock=clock)


class Budget(unittest.TestCase):
    def test_node_accounting_with_legacy_entries(self):
        from strl.budget import node_fraction, remaining_seconds
        legacy = {'job_id': 'a', 'status': 'COMPLETED', 'allocated_at': '2026-10-06T10:00:00Z', 'terminal_at': '2026-10-06T10:10:00Z',
                  'gpu_fraction': 1.0}
        two = dict(legacy, job_id='b', nodes=2)
        canary = dict(legacy, job_id='c', gpu_fraction=0.125)
        self.assertEqual(remaining_seconds([legacy], 21600), 21600 - 600)
        self.assertEqual(remaining_seconds([legacy, two, canary], 21600), 21600 - 600 - 1200 - 75)
        cfg = load(SMOKE)
        self.assertEqual((node_fraction(cfg), node_fraction(cfg, 'dry')), (2.0, 0.0))
        self.assertEqual(node_fraction(load('base/gsm8k.yaml')), 1.0)


class Overrides(unittest.TestCase):
    def test_set_changes_sha_and_is_recorded(self):
        base = load(SMOKE)
        changed = load(SMOKE, overrides=['algorithm.lr=3e-5', 'compute.training.cpu_offload=true'])
        self.assertNotEqual(base['_meta']['sha'], changed['_meta']['sha'])
        self.assertEqual(changed['_meta']['overrides'], ['algorithm.lr=3e-5', 'compute.training.cpu_offload=true'])
        self.assertEqual((changed['algorithm']['lr'], changed['compute']['training']['cpu_offload']), (3e-5, True))
        self.assertNotIn('overrides', base['_meta'])
        self.assertEqual(load(SMOKE, overrides=[])['_meta']['sha'], base['_meta']['sha'])
        for bad in (['algorithm.lr'], ['algoritm.lr=1'], ['finetune.mode=qlora'], ['compute.training.fsdp_size=3'],
                    ['inference.num_engines=3'], ['finetune.lora.rank=0'], ['compute.nodes=3']):
            with self.assertRaises(ExperimentError, msg=bad):
                load(SMOKE, overrides=bad)

    def test_existing_experiments_render_nothing_new(self):
        # the GPU-verified single-node experiments that predate LoRA / multi-node (an explicit list: new experiments may use LoRA freely)
        legacy = ['gsm8k-qwen35', 'mbpp-qwen35-binary', 'mbpp-qwen35-partial', 'mbpp-qwen35-custom-reward', 'mbpp-qwen3-1p7b', 'harbor-qwen35']
        for path in [ROOT / 'experiments' / f'{name}.yaml' for name in legacy] + [ROOT / 'base/gsm8k.yaml']:
            cfg = load(str(path))
            self.assertEqual(scaling_values(cfg, 10), {}, path)
            self.assertNotIn('finetune', cfg)
            self.assertEqual(cfg['compute']['nodes'], 1)

    def test_lora_and_full_rendering(self):
        lora = dict(v.split('=', 1) for v in overrides(load(SMOKE), stage='train', run_name='t'))
        self.assertEqual((lora['trainer.policy.model.lora.rank'], lora['trainer.policy.model.lora.alpha']), ('16', '32'))
        self.assertEqual(lora['trainer.policy.model.lora.target_modules'], 'all-linear')
        self.assertEqual((lora['trainer.policy.fsdp_config.fsdp_size'], lora['trainer.ref.fsdp_config.fsdp_size']), ('8', '8'))
        self.assertEqual((lora['trainer.placement.policy_num_nodes'], lora['generator.inference_engine.num_engines']), ('2', '16'))
        full = dict(v.split('=', 1) for v in overrides(load(SMOKE_FULL), stage='train', run_name='t'))
        self.assertFalse(any('lora' in k or 'fsdp_size' in k for k in full))

    def test_every_scaling_key_exists_in_pinned_skyrl(self):
        tree = ast.parse((ROOT / 'pinned/pinned-config.py').read_text())
        classes = {c.name: {f.target.id: ast.unparse(f.annotation) for f in c.body if isinstance(f, ast.AnnAssign) and isinstance(f.target, ast.Name)}
                   for c in tree.body if isinstance(c, ast.ClassDef)}
        cfg = load(SMOKE, overrides=[
            'finetune.lora.exclude_modules=lm_head', 'finetune.lora.target_modules=[q_proj, v_proj]', 'compute.training.cpu_offload=true',
            'compute.training.reshard_after_forward=false', 'compute.training.reduce_dtype=bf16', 'compute.training.max_tokens_per_microbatch=8192',
            'compute.training.gradient_checkpointing=false', 'inference.max_num_batched_tokens=16384', 'inference.num_engines=16',
            'inference.engine_init_kwargs={gdn_prefill_backend: triton, compilation_config: {pass_config: {fuse_allreduce_rms: false}}}',
            'budget.save_optimizer_state=false'])
        rendered = scaling_values(cfg, 3)
        self.assertEqual(len(rendered), 24)
        self.assertEqual((rendered['trainer.ckpt_interval'], rendered['trainer.hf_save_interval']), (-1, 3))
        self.assertIn('generator.inference_engine.engine_init_kwargs.compilation_config.pass_config.fuse_allreduce_rms', rendered)
        values = dict(v.split('=', 1) for v in overrides(cfg, stage='train', run_name='t'))
        self.assertEqual(values['trainer.policy.model.lora.target_modules'], "['q_proj', 'v_proj']")
        for key in rendered:
            current = 'SkyRLTrainConfig'
            for field in key.split('.'):
                if current not in classes:
                    self.assertIn('dict', current.lower(), key)
                    break
                self.assertIn(field, classes[current], key)
                current = classes[current][field].replace('Optional[', '').rstrip(']')


class Memory(unittest.TestCase):
    def setUp(self):
        (TMP / 'q27').mkdir(parents=True, exist_ok=True)
        shutil.copy(Q27, TMP / 'q27/config.json')
        self.sets = ['model.id=Qwen/Qwen3.8-27B', 'model.revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0', 'model.params_b=27.8',
                     f'model.local_path={TMP / "q27"}', 'inference.tensor_parallel_size=2', 'inference.gpu_memory_utilization=0.5',
                     'algorithm.micro_train_per_gpu=1', 'algorithm.micro_forward_per_gpu=1', 'algorithm.max_prompt_tokens=4096',
                     'algorithm.max_generate_length=1024']

    def tearDown(self):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_qwen38_27b_lora_vs_full_on_16_gpus(self):
        from strl.derive import lora_params, memory_estimate_gb, plan, recommend_layout
        self.assertEqual(lora_params(json.loads(Q27.read_text()), 64), 466_911_232)
        lora = load(SMOKE, overrides=self.sets + ['finetune.lora.rank=64', 'finetune.lora.alpha=64'])
        full = load(SMOKE_FULL, overrides=self.sets)
        hf = json.loads(Q27.read_text())
        m_lora, m_full = memory_estimate_gb(27.8, lora, hf), memory_estimate_gb(27.8, full, hf)
        self.assertEqual((m_lora['fsdp_group'], m_full['fsdp_group']), (8, 16))
        self.assertAlmostEqual(m_lora['lora_params_m'], 466.9)
        self.assertLess(m_lora['training_phase_gb'], 25)            # 2 B/param over 8 + ref + LoRA states + activations
        self.assertTrue(35 < m_full['training_phase_gb'] < 42)       # 16 B/param over 16 + ref + activations
        self.assertTrue(m_lora['fits'] and m_full['fits'])
        self.assertEqual((m_full['engine_weights_gb'], m_full['kv_room_gb']), (27.8, 12.2))   # TP2 at 0.5 utilisation
        hsdp = memory_estimate_gb(27.8, full, hf, fsdp_size=8)
        self.assertGreater(hsdp['training_phase_gb'], m_full['training_phase_gb'] + 25)
        one = memory_estimate_gb(27.8, full, hf, world=1, tp=1, fsdp_size=1)
        self.assertFalse(one['fits'])
        tp1 = load(SMOKE_FULL, overrides=self.sets + ['inference.tensor_parallel_size=1'])
        report = plan(tp1)
        self.assertTrue(any(w.startswith('MEMORY: vLLM weights') for w in report['warnings']))
        self.assertTrue(any(w.startswith('NETWORK') for w in report['warnings']))
        report = plan(lora)
        self.assertTrue(any(w.startswith('LORA_SYNC') for w in report['warnings']))
        self.assertEqual(report['layout']['recommended']['fsdp_size'], 8)
        self.assertEqual(report['layout']['replicas'], 2)
        self.assertEqual(recommend_layout(full)['fsdp_size'], None)
        self.assertEqual(recommend_layout(full, 3.0), {'fsdp_size': 8, 'cpu_offload': True, 'reason': recommend_layout(full, 3.0)['reason']})
        self.assertIsNone(recommend_layout(full, 20.0)['fsdp_size'])
        offload = memory_estimate_gb(27.8, load(SMOKE_FULL, overrides=self.sets + ['compute.training.fsdp_size=8',
                                                                                   'compute.training.cpu_offload=true']), hf)
        self.assertGreater(offload['host_offload_gb_per_gpu'], 40)
        self.assertLess(offload['training_phase_gb'], hsdp['training_phase_gb'])
        big = plan(load(SMOKE_FULL, overrides=self.sets + ['compute.training.fsdp_size=8', 'algorithm.micro_train_per_gpu=2',
                                                          'algorithm.micro_forward_per_gpu=2']))
        self.assertTrue(any(w.startswith('MEMORY: estimated training peak') for w in big['warnings']))

    def test_canary_overlay_selection(self):
        from strl.stage import CANARY_OVERLAY, CANARY_OVERLAY_LARGE, apply_canary_overlay, canary_overlay_for
        self.assertEqual(canary_overlay_for(load('base/gsm8k.yaml')), CANARY_OVERLAY)
        smoke = load(SMOKE)
        self.assertEqual(canary_overlay_for(smoke), CANARY_OVERLAY_LARGE)
        canary = apply_canary_overlay(smoke)
        self.assertEqual((canary['compute']['nodes'], canary['compute']['gpus_per_node'], canary['finetune']), (2, 8, smoke['finetune']))
        self.assertEqual(canary['_meta']['canary_overlay'], 'canary_overlay_large.yaml')
        big = load('base/gsm8k.yaml', overrides=['model.id=Qwen/Qwen3.8-27B', 'model.params_b=27.8', f'model.local_path={TMP / "q27"}',
                                                 'model.revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0', 'inference.tensor_parallel_size=2'])
        self.assertEqual(canary_overlay_for(big), CANARY_OVERLAY_LARGE)
        out = apply_canary_overlay(big)
        self.assertEqual((out['compute']['gpus_per_node'], out['inference']['tensor_parallel_size']), (8, 2))
        self.assertEqual(out['_meta']['overrides'], big['_meta']['overrides'])


class Staging(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(TMP, ignore_errors=True)
        (TMP / 'model').mkdir(parents=True)
        (TMP / 'model/config.json').write_text((ROOT / 'tests/fixtures/configs/qwen3_5.json').read_text())
        (TMP / 'model/model.safetensors').write_bytes(b'weights')
        (TMP / 'data').mkdir()
        (TMP / 'data/train.parquet').write_bytes(b'data')
        (TMP / 'ledger.json').write_text(json.dumps({'total_seconds': 21600, 'nodes': 2, 'attempts': []}))
        rel = TMP.relative_to(ROOT)
        (TMP / 'exp.yaml').write_text(f'extends: {SMOKE}\nmodel: {{local_path: {rel}/model}}\nrun: {{data_dir: {rel}/data}}\n')

    def tearDown(self):
        shutil.rmtree(TMP, ignore_errors=True)

    def stage(self, mode, s3=None, **kw):
        from strl.stage import stage
        receipt = stage(str(TMP / 'exp.yaml'), attempt='mn-a1', mode=mode, ledger=str(TMP / 'ledger.json'), output=str(TMP / f'{mode}.json'),
                        s3=s3 or FakeS3(), now=time.struct_time((2026, 10, 6, 12, 0, 0, 0, 279, 0)), **kw)
        return receipt, json.loads((TMP / f'{mode}.json').read_text())

    def test_two_node_jobs(self):
        receipt, job = self.stage('train')
        self.assertEqual((job['num_instances'], job['preset_instance_type'], receipt['nodes'], receipt['gpu_fraction']), (2, 'p5.48xlarge', 2, 1.0))
        self.assertEqual(job['active_deadline_seconds'], load(SMOKE)['budget']['attempt_max_seconds'])   # follows the smoke experiment's cap
        self.assertIn('SKYRL_WORKER_NCCL_TIMEOUT_IN_S=900 SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=900', job['command'][2])
        runtime = json.loads((Path(receipt['staging_dir']) / 'runtime.json').read_text())
        self.assertEqual((runtime['nodes'], runtime['gpu_fraction']), (2, 1.0))
        receipt, job = self.stage('dry')
        self.assertEqual((job['num_instances'], job['preset_instance_type'], receipt['gpu_fraction']), (2, 'm5.2xlarge', 0.0))
        self.assertEqual(job['active_deadline_seconds'], 900 + 600)    # peers' preflight + Ray join + CPU probe
        self.assertIn('export GRPO_DRY_RUN=1', job['command'][2])
        receipt, job = self.stage('canary')
        self.assertEqual((job['num_instances'], receipt['canary_overlay']), (2, 'canary_overlay_large.yaml'))
        (TMP / 'ledger.json').write_text(json.dumps({'total_seconds': 21600, 'nodes': 2, 'attempts': [
            {'job_id': 'x', 'status': 'COMPLETED', 'allocated_at': '2026-10-06T00:00:00Z', 'terminal_at': '2026-10-06T02:45:00Z',
             'gpu_fraction': 1.0, 'nodes': 2}]}))
        with self.assertRaises(RuntimeError):      # 1800 node-s left = 900 s at 2 nodes: not enough for an attempt
            self.stage('train')

    def _stage_as(self, attempt, mode='train', s3=None, reservation=None):
        from strl.stage import stage
        return stage(str(TMP / 'exp.yaml'), attempt=attempt, mode=mode, ledger=str(TMP / 'ledger.json'), output=str(TMP / f'{attempt}.json'),
                     s3=s3 or FakeS3(), now=time.struct_time((2026, 10, 6, 12, 0, 0, 0, 279, 0)), reservation=reservation)

    def test_concurrent_launches_cannot_overspend_one_ledger(self):
        held = lambda: json.loads((TMP / 'ledger.json').read_text()).get('reservations') or {}
        (TMP / 'ledger.json').write_text(json.dumps({'total_seconds': 5000, 'nodes': 2, 'attempts': []}))   # one 2 x 8 booking = 1800 s x 2 = 3600
        first = self._stage_as('mn-a1-fb', reservation='mn-a1')
        self._stage_as('mn-a1-rb', reservation='mn-a1')                       # a race's routes share ONE reservation
        self.assertEqual(list(held()), ['mn-a1'])
        self.assertEqual(held()['mn-a1']['node_seconds'], int(first['remaining_seconds'] * 2))
        self.assertEqual(held()['mn-a1']['attempts'], ['mn-a1-fb', 'mn-a1-rb'])
        with self.assertRaises(RuntimeError):                                  # another launch sees only what is left
            self._stage_as('mn-b1')
        self._stage_as('mn-d1', mode='dry')                                    # CPU dry runs are free: never blocked, never booked
        self.assertEqual(list(held()), ['mn-a1'])
        from strl.stage import release_reservation
        release_reservation(TMP / 'ledger.json', 'mn-a1', 'mn-a1-rb')
        release_reservation(TMP / 'ledger.json', 'mn-a1', 'mn-a1-fb')
        self.assertEqual(held(), {})
        self._stage_as('mn-b1')                                                # released: the budget is available again

    def test_recording_an_attempt_releases_its_reservation(self):
        from unittest import mock
        from strl import cli
        (TMP / 'ledger.json').write_text(json.dumps({'total_seconds': 5000, 'nodes': 2, 'attempts': [],
                                                    'reservations': {'mn-a1': {'node_seconds': 3600, 'attempts': ['mn-a1-fb'], 'created': time.time()}}}))
        Path('evidence/mn-a1-fb').mkdir(parents=True, exist_ok=True)
        Path('evidence/mn-a1-fb/stage.json').write_text(json.dumps({'gpu_fraction': 1.0, 'nodes': 2}))
        try:
            with mock.patch.object(cli, 'allocation_start', lambda cfg, attempt: None), mock.patch.object(cli, 'worker_started', lambda cfg, attempt: False):
                out = cli.record_attempt(str(TMP / 'exp.yaml'), 'mn-a1-fb', 'job-1', ledger=str(TMP / 'ledger.json'), never_started=True,
                                         job={'status': 'CANCELED', 'created_at': '2026-10-06 00:00:00+00:00'})
        finally:
            shutil.rmtree('evidence/mn-a1-fb', ignore_errors=True)
        self.assertIsNone(out['recorded']['allocated_at'])
        self.assertEqual(json.loads((TMP / 'ledger.json').read_text()).get('reservations'), {})

    def test_a_failed_stage_releases_its_reservation(self):
        class Broken(FakeS3):
            def put_object(self, **kw):
                raise OSError('S3 down')
        with self.assertRaises(OSError):
            self._stage_as('mn-x1', s3=Broken())
        self.assertEqual(json.loads((TMP / 'ledger.json').read_text()).get('reservations') or {}, {})

    def test_model_already_in_s3(self):
        from strl.stage import MODEL_MANIFEST, model_prefix, stage_model
        shutil.rmtree(TMP / 'model')
        with self.assertRaises(RuntimeError) as error:
            self.stage('dry')
        self.assertIn('stage-model', str(error.exception))
        cfg = load(str(TMP / 'exp.yaml'))
        s3 = FakeS3()
        manifest = [{'name': f'model-0000{i}-of-00002.safetensors', 'bytes': 6 * 10 ** 9, 'sha256': 'ab' * 32} for i in (1, 2)]
        s3.put_object(Bucket=cfg['storage']['bucket'], Key=f"{model_prefix(cfg['storage']['prefix'], cfg)}/{MODEL_MANIFEST}",
                      Body=json.dumps(manifest).encode())
        receipt, job = self.stage('dry', s3=s3)
        self.assertFalse(any(c[0] == 'upload_file' for c in s3.calls))
        command = job['command'][2]
        self.assertIn('max_concurrency=8', command)                  # 12 GB model: multipart fetch in the pod, 4 files at a time
        self.assertIn('ThreadPoolExecutor(4)', command)              # 16 files x 16 parts filled a 30 GiB CPU pod's memory and stalled (dry-27b-a)
        self.assertIn('records=[r for r in records if not', command)           # a dry run only validates configuration: no 55 GB of weights per CPU pod
        receipt, job = self.stage('train', s3=s3)
        self.assertNotIn('records=[r for r in records if not', job['command'][2])
        out = stage_model(str(TMP / 'exp.yaml'), attempt='fetch-q', output=str(TMP / 'fetch.json'))
        job = json.loads((TMP / 'fetch.json').read_text())
        self.assertEqual((job['preset_instance_type'], job['num_instances'], out['gpu_fraction']), ('m5.2xlarge', 1, 0.0))
        for needle in ('hf_hub_download', 'download retry', 'path.unlink()', 'ThreadPoolExecutor(3)'):   # per-file, retried, bounded disk
            self.assertIn(needle, job['command'][2])
        self.assertNotIn('snapshot_download', job['command'][2])
        self.assertIn(MODEL_MANIFEST, job['command'][2])
        compile(job['command'][2].split("python -c ", 1)[1].strip().strip("'").replace("'\"'\"'", "'"), 'stage-model', 'exec')

    def test_streamed_sha(self):
        import hashlib
        from strl.stage import file_sha256
        path = TMP / 'model/model.safetensors'
        self.assertEqual(file_sha256(path, block=3), hashlib.sha256(b'weights').hexdigest())


class Exports(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def wrapper(self):
        base = self.root / 'base'
        base.mkdir()
        config = json.loads((ROOT / 'tests/fixtures/configs/qwen3_5.json').read_text())
        (base / 'config.json').write_text(json.dumps(config))
        text = dict(config['text_config'], model_type='qwen3_5_text', architectures=['Qwen3_5ForCausalLM'])
        return base, text

    def test_sharded_export_adapted_with_index(self):
        import torch
        from safetensors.torch import load_file, save_file
        from strl.models.qwen3_5_export import adapt
        base, text = self.wrapper()
        export = self.root / 'export'
        export.mkdir()
        (export / 'config.json').write_text(json.dumps(text))
        shards = {'model-00001-of-00002.safetensors': {'model.layers.0.mlp.up_proj.weight': torch.arange(12.).reshape(3, 4)},
                  'model-00002-of-00002.safetensors': {'model.layers.1.mlp.up_proj.weight': torch.ones(3, 4), 'lm_head.weight': torch.zeros(2, 4)}}
        for name, tensors in shards.items():
            save_file(tensors, str(export / name))
        index = {'metadata': {'total_size': 123}, 'weight_map': {k: n for n, t in shards.items() for k in t}}
        (export / 'model.safetensors.index.json').write_text(json.dumps(index))
        receipt = adapt(base, export, self.root / 'adapted')
        self.assertEqual((receipt['tensor_count'], receipt['tensors_renamed'], receipt['shards']), (3, 2, 2))
        new = json.loads((self.root / 'adapted/model.safetensors.index.json').read_text())
        self.assertEqual(new['metadata'], {'total_size': 123})
        self.assertEqual(new['weight_map'], {'model.language_model.layers.0.mlp.up_proj.weight': 'model-00001-of-00002.safetensors',
                                             'model.language_model.layers.1.mlp.up_proj.weight': 'model-00002-of-00002.safetensors',
                                             'lm_head.weight': 'model-00002-of-00002.safetensors'})
        out = load_file(self.root / 'adapted/model-00001-of-00002.safetensors')
        self.assertTrue(torch.equal(out['model.language_model.layers.0.mlp.up_proj.weight'], torch.arange(12.).reshape(3, 4)))
        (export / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'model-00009-of-00009.safetensors'}}))
        with self.assertRaises(ValueError):
            adapt(base, export, self.root / 'adapted2')

    def test_lora_export_kept_and_merged(self):
        import torch
        from safetensors.torch import load_file, save_file
        from strl.lora import is_adapter, merge_adapter
        from strl.models.qwen3_5_export import adapt
        from strl.verify import verify_adapter_change
        base, text = self.wrapper()
        weight = torch.ones(4, 6, dtype=torch.float32)
        save_file({'model.language_model.layers.0.self_attn.q_proj.weight': weight, 'lm_head.weight': torch.zeros(2, 6)},
                  str(base / 'model.safetensors'))
        export = self.root / 'lora'
        export.mkdir()
        a, b = torch.full((2, 6), 0.5), torch.full((4, 2), 0.25)
        save_file({'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight': a,
                   'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight': b}, str(export / 'adapter_model.safetensors'))
        (export / 'adapter_config.json').write_text(json.dumps({'r': 2, 'lora_alpha': 4, 'target_modules': ['q_proj'],
                                                                 'base_model_name_or_path': str(base)}))
        (export / 'config.json').write_text(json.dumps(text))
        self.assertTrue(is_adapter(export))
        kept = adapt(base, export, self.root / 'reload', 'Qwen/Qwen3.5-0.8B', 'abc123')
        self.assertEqual((kept['kind'], kept['base_model_id'], kept['base_model_revision'], kept['r']), ('lora', 'Qwen/Qwen3.5-0.8B', 'abc123', 2))
        self.assertEqual((self.root / 'reload/adapter_model.safetensors').read_bytes(), (export / 'adapter_model.safetensors').read_bytes())
        self.assertEqual(verify_adapter_change(export)['changed_lora_B'], 1)
        merged = merge_adapter(base, export, self.root / 'merged')
        self.assertEqual((merged['merged_tensors'], merged['scale']), (1, 2.0))
        out = load_file(self.root / 'merged/model.safetensors')
        self.assertTrue(torch.allclose(out['model.language_model.layers.0.self_attn.q_proj.weight'], weight + 2.0 * (b @ a)))
        self.assertTrue(torch.equal(out['lm_head.weight'], torch.zeros(2, 6)))
        self.assertEqual(json.loads((self.root / 'merged/config.json').read_text())['model_type'], 'qwen3_5')
        save_file({'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight': a,
                   'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight': torch.zeros(4, 2)}, str(export / 'adapter_model.safetensors'))
        with self.assertRaises(RuntimeError):
            verify_adapter_change(export)

    def test_cluster_shards(self):
        from strl.verify import local_shards, verify_cluster_shards
        reports = []
        for node in range(2):
            root = self.root / f'node{node}'
            path = root / 'checkpoints/global_step_3/policy'
            path.mkdir(parents=True)
            for kind in ('model', 'optim', 'extra_state'):
                for rank in range(node * 8, node * 8 + 8):
                    (path / f'{kind}_world_size_16_rank_{rank}.pt').write_bytes(b'x')
            reports.append(local_shards(root, 3, 16))
        self.assertEqual(reports[1]['ranks']['optim'], list(range(8, 16)))
        self.assertEqual(verify_cluster_shards(reports, 16)['nonempty_rank_files'], 48)
        with self.assertRaises(RuntimeError):
            verify_cluster_shards(reports[:1], 16)


class SkyRLPatches(unittest.TestCase):
    def apply(self, name, patches):
        from strl.ipv6_patch import patch_tree
        relative = next(iter(patches))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / relative
            path.parent.mkdir(parents=True)
            path.write_text((ROOT / 'pinned' / name).read_text())
            patch_tree(temp, patches)
            once = path.read_text()
            patch_tree(temp, patches)                      # idempotent
            self.assertEqual(path.read_text(), once)
            return once

    def test_resume_from_an_s3_path_keeps_the_double_slash(self):
        from strl.runtime import RESUME_FROM_CLOUD_PATH, scaling_patches
        from strl.experiment import load
        source = self.apply('pinned-trainer.py', RESUME_FROM_CLOUD_PATH)            # pinned SkyRL a9564572 source
        fn = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == 'load_checkpoints')
        text = ast.unparse(fn)
        self.assertNotIn('Path(self.cfg.trainer.resume_path)', text)
        path = 's3://b/p/attempts/a/artifacts/checkpoints/global_step_2'
        self.assertEqual(str(path or ''), path)                                       # what the patched line now keeps
        self.assertIn('global_step_', os.path.basename(path.rstrip('/')))
        cfg = load(str(ROOT / 'experiments/mimo-phi4-mini-lora-verify.yaml'), overrides=['run.resume_from=v1phi4-fb'])
        merged = scaling_patches(cfg)
        self.assertIn('skyrl/train/trainer.py', merged)
        self.assertIn('skyrl/backends/skyrl_train/distributed/fsdp_strategy.py', merged)

    def test_lora_adapter_save_goes_through_cloud_io(self):
        from strl.runtime import LORA_ADAPTER_CLOUD_SAVE, scaling_patches
        from strl.experiment import load
        source = self.apply('pinned-fsdp_strategy.py', LORA_ADAPTER_CLOUD_SAVE)     # pinned SkyRL a9564572 source
        fn = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == '_save_lora_adapters')
        text = ast.unparse(fn)
        self.assertNotIn('save_file(lora_params', text)
        self.assertIn("io.open_file(os.path.join(lora_save_path, 'adapter_model.safetensors'), 'wb')", text)
        import safetensors.torch, torch
        tensors = {'a.lora_A.weight': torch.ones(2, 3)}
        self.assertTrue(torch.equal(safetensors.torch.load(safetensors.torch.save(tensors))['a.lora_A.weight'], tensors['a.lora_A.weight']))
        durable_lora = load(str(ROOT / 'experiments/mimo-phi4-mini-lora-verify.yaml'))
        self.assertIn(next(iter(LORA_ADAPTER_CLOUD_SAVE)), scaling_patches(durable_lora))
        self.assertNotIn(next(iter(LORA_ADAPTER_CLOUD_SAVE)), scaling_patches(load(str(ROOT / 'experiments/mimo-qwen35-0p8b-2node-smoke.yaml'))))

    def test_lora_sync_per_host(self):
        from strl.runtime import LORA_SYNC_PER_HOST
        source = self.apply('pinned-fsdp_worker.py', LORA_SYNC_PER_HOST)
        tree = ast.parse(source)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == '_save_lora_adapters_and_sync')
        body = [ast.unparse(n) for n in fn.body]
        writer = next(i for i, s in enumerate(body) if s.startswith('if _strl_hosts.index'))
        loader = next(i for i, s in enumerate(body) if s.startswith('if _strl_rank == 0'))
        self.assertTrue(any('all_gather_object' in s for s in body[:writer]))
        self.assertIn('save_file', body[writer])
        self.assertNotIn('load_lora_adapter', body[writer])
        self.assertTrue(body[writer + 1].startswith('torch.distributed.barrier()') and loader == writer + 2)
        self.assertIn('load_lora_adapter', body[loader])
        self.assertTrue(body[-1].startswith('torch.distributed.barrier()'))

    def test_no_optimizer_checkpoint_patch_and_selection(self):
        from strl.runtime import NO_OPTIMIZER_CHECKPOINT, scaling_patches
        source = self.apply('pinned-trainer.py', NO_OPTIMIZER_CHECKPOINT)
        self.assertIn('(force_save and self.cfg.trainer.ckpt_interval > 0) or ckpt_interval_save', source)
        ast.parse(source)
        self.assertEqual(scaling_patches(load('base/gsm8k.yaml')), {})
        self.assertEqual(set(scaling_patches(load(SMOKE))), {'skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py'})
        self.assertEqual(set(scaling_patches(load(SMOKE_FULL, overrides=['budget.save_optimizer_state=false']))), {'skyrl/train/trainer.py'})


class Probe(unittest.TestCase):
    def test_transport_classification(self):
        from strl.probe import classify_transport
        self.assertEqual(classify_transport('h:1:1 [0] NCCL INFO NET/OFI Selected Provider is efa (found 32 nics)\n'
                                            'h:1:1 [0] NCCL INFO Using network Libfabric'), 'efa')
        self.assertEqual(classify_transport('NCCL INFO NET/Socket : Using [0]eth0:10.0.0.1<0>\nNCCL INFO Using network Socket'), 'socket')
        self.assertEqual(classify_transport('NCCL INFO NET/IB : Using [0]mlx5_0:1/IB\nNCCL INFO Using network IB'), 'ib')
        self.assertEqual(classify_transport('NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM'), 'nvlink')
        self.assertEqual(classify_transport(''), 'unknown')

    def test_efa_detection_and_environment(self):
        from strl.probe import configure_efa, efa_facts
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            none = efa_facts(str(root), environ={}, which=lambda name: None)
            self.assertFalse(none['efa_available'])
            env = {}
            self.assertEqual(configure_efa(env, none)['applied'], {})
            self.assertEqual(env, {})
            (root / 'dev/infiniband').mkdir(parents=True)
            (root / 'dev/infiniband/uverbs0').write_text('')
            (root / 'opt/amazon/ofi-nccl/lib').mkdir(parents=True)
            (root / 'opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so').write_text('')
            facts = efa_facts(str(root), environ={}, which=lambda name: '/usr/bin/fi_info',
                              run=lambda *a, **k: types.SimpleNamespace(returncode=0))
            self.assertTrue(facts['efa_available'])
            self.assertEqual(facts['devices'], ['uverbs0'])
            env = {'FI_EFA_USE_DEVICE_RDMA': '0'}
            self.assertEqual(configure_efa(env, facts)['applied'], {'FI_PROVIDER': 'efa'})
            self.assertEqual(env, {'FI_EFA_USE_DEVICE_RDMA': '0', 'FI_PROVIDER': 'efa'})   # explicit settings are respected
            self.assertNotIn('NCCL_PROTO', env)
            broken = efa_facts(str(root), environ={}, which=lambda name: '/usr/bin/fi_info', run=lambda *a, **k: types.SimpleNamespace(returncode=1))
            self.assertFalse(broken['efa_available'])

    def test_monitor_cluster_flags(self):
        from strl.monitor.live import cluster_flags
        socket_net = {'efa_available': False, 'transport': 'socket', 'bandwidth_gbps': {'cross_node': 2.5}}
        self.assertEqual(cluster_flags(socket_net, [{'at': 990.0}], now=1000.0), ['NETWORK(transport=socket, efa=False, cross_node=2.5GB/s)'])
        self.assertEqual(cluster_flags({'efa_available': True, 'transport': 'efa'}, [{'at': 100.0}, None], now=1000.0),
                         ['PEER_STALE(rank 1, 900s)'])
        self.assertEqual(cluster_flags(None, [None], now=1000.0), [])

    def test_summary_bandwidth_and_limits(self):
        from strl.probe import bus_bandwidth, cgroup_memory_limit, summarize
        self.assertEqual(bus_bandwidth(1 << 30, 0.1, 16)['busbw_gbps'], round((1 << 30) / 0.1 / 1e9 * 15 / 16, 2))
        facts = [{'hostname': h, 'efa': {'efa_available': False}} for h in ('a', 'b')]
        ranks = [{'transport': 'socket', 'intra_node': {'busbw_gbps': 150.0}, 'cross_node': {'busbw_gbps': g}, 'global': {'busbw_gbps': 3.0}}
                 for g in (2.0, 3.0, 4.0)]
        out = summarize([{}, {}], facts, ranks, cpu=False)
        self.assertEqual((out['transport'], out['bandwidth_gbps']['cross_node'], out['efa_available']), ('socket', 3.0, False))
        self.assertEqual(len(out['warnings']), 2)
        cpu = summarize([{}, {}], facts, None, cpu=True)
        self.assertEqual((cpu['transport'], cpu['hostnames']), ('not_measured', ['a', 'b']))
        with tempfile.TemporaryDirectory() as temp:
            v2 = Path(temp, 'memory.max')
            v2.write_text('2147483648\n')
            self.assertEqual(cgroup_memory_limit(str(v2), '/missing'), 2147483648)
            v2.write_text('max\n')
            self.assertIsNone(cgroup_memory_limit(str(v2), '/missing'))


if __name__ == '__main__':
    unittest.main()


class NeverStartedJobs(unittest.TestCase):
    """A job cancelled while QUEUED used to be charged from creation (an hour in the queue = 450 node-s at 1/8 of a node)."""

    def test_worker_started_fails_closed(self):
        from botocore.exceptions import ClientError
        from strl.cli import worker_started
        cfg = {'storage': {'bucket': 'b', 'prefix': 'p', 'profile': 'x'}}

        class S3:
            def __init__(self, code):
                self.code = code

            def head_object(self, **kw):
                if self.code is None:
                    return {}
                raise ClientError({'Error': {'Code': self.code}}, 'HeadObject')
        self.assertTrue(worker_started(cfg, 'a', S3(None)))
        self.assertFalse(worker_started(cfg, 'a', S3('404')))
        with self.assertRaises(ClientError):          # an access error is NOT evidence that nothing ran
            worker_started(cfg, 'a', S3('AccessDenied'))

    def test_an_unallocated_cancelled_attempt_costs_nothing(self):
        from strl.budget import remaining_seconds
        attempts = [{'job_id': 'j1', 'attempt': 'x', 'status': 'CANCELED', 'allocated_at': None, 'terminal_at': '2026-10-07T09:00:00Z',
                     'gpu_fraction': 0.125, 'nodes': 1}]
        self.assertEqual(remaining_seconds(attempts, 21600), 21600)


    def test_charge_starts_when_the_pods_started_not_when_the_job_was_created(self):
        """g4-mimo-a waited 3.5 h for two nodes: created_at -> terminal would have cost 7 node-hours of a 6 node-hour program."""
        from botocore.exceptions import ClientError
        from strl.cli import POD_START_SLACK_SECONDS, allocation_start
        cfg = {'storage': {'bucket': 'b', 'prefix': 'p', 'profile': 'x'}}

        class S3:
            def __init__(self, code=None):
                self.code = code

            def get_object(self, **kw):
                if self.code:
                    raise ClientError({'Error': {'Code': self.code}}, 'GetObject')
                import io
                return {'Body': io.BytesIO(b'{"started_at": 1000000.0}')}
        self.assertEqual(allocation_start(cfg, 'a', S3()), 1000000.0 - POD_START_SLACK_SECONDS)
        self.assertIsNone(allocation_start(cfg, 'a', S3('NoSuchKey')))
        with self.assertRaises(ClientError):          # fails closed: an access error is not evidence of a never-started pod
            allocation_start(cfg, 'a', S3('AccessDenied'))


    def test_sigterm_is_recorded_as_preempted_not_failed(self):
        from strl.worker import outcome_status
        self.assertEqual(outcome_status(SystemExit(15)), 'preempted')
        self.assertEqual(outcome_status(SystemExit(1)), 'failed')
        self.assertEqual(outcome_status(RuntimeError('x')), 'failed')


class ModelIdentity(unittest.TestCase):
    """Generality audit: a child that changed only model.id inherited assets/model (Qwen3.5-0.8B), planned it as Qwen3.5 and would have uploaded
    and trained those weights under the new name. One directory per id@revision, an identity marker, and a forced-profile check close it."""

    def test_default_directory_is_per_id_and_revision(self):
        from strl.experiment import default_model_dir
        self.assertEqual(default_model_dir({'id': 'meta-llama/Llama-3.1-8B-Instruct', 'revision': '0e9e39f249a16976918f6564b8830bc894c89659'}),
                         'assets/models/Llama-3.1-8B-Instruct@0e9e39f249a1')

    def test_identity_marker(self):
        import json
        import tempfile
        from strl.models import snapshot_identity, write_identity
        with tempfile.TemporaryDirectory() as temp:
            model = {'id': 'org/a', 'revision': 'r1', 'local_path': temp}
            self.assertEqual(snapshot_identity(model)[0], 'absent')
            (Path(temp) / 'config.json').write_text('{}')
            self.assertEqual(snapshot_identity(model)[0], 'unknown')
            write_identity(model)
            self.assertEqual(snapshot_identity(model)[0], 'match')
            state, message = snapshot_identity({**model, 'id': 'org/b'})
            self.assertEqual(state, 'mismatch')
            self.assertIn('org/a', message)
            self.assertEqual(json.loads((Path(temp) / 'strl-model.json').read_text())['id'], 'org/a')

    def test_forced_profile_contradicting_the_architecture_is_rejected(self):
        from strl.models import detect
        with self.assertRaises(RuntimeError) as error:
            detect({'id': 'meta-llama/x', 'profile': 'qwen3_5'}, hf_config={'architectures': ['LlamaForCausalLM']})
        self.assertIn('auto-detect', str(error.exception))
        self.assertEqual(detect({'id': 'q', 'profile': 'qwen3_5'}, hf_config={'architectures': ['Qwen3_5ForConditionalGeneration']})[0].name, 'qwen3_5')


class ModelRegistry(unittest.TestCase):
    def test_families_map_to_profiles(self):
        from strl.models import detect
        for arch, name in (('Qwen3MoeForCausalLM', 'qwen3_moe'), ('Qwen3_5MoeForConditionalGeneration', 'qwen3_5_moe'), ('MistralForCausalLM', 'dense'),
                           ('Olmo2ForCausalLM', 'dense'), ('Phi3ForCausalLM', 'dense'), ('Qwen2ForCausalLM', 'dense'), ('SomethingNewForCausalLM', 'generic_causal_lm')):
            self.assertEqual(detect({'id': 'x'}, hf_config={'architectures': [arch]})[0].name, name, arch)
        self.assertTrue(detect({'id': 'x'}, hf_config={'architectures': ['Qwen3MoeForCausalLM']})[0].thinking_flag)   # thinking off like dense Qwen3

    def test_unmapped_vision_language_wrapper_is_refused(self):
        from strl.models import detect
        with self.assertRaises(RuntimeError) as error:
            detect({'id': 'google/gemma-3-12b-it'}, hf_config={'architectures': ['Gemma3ForConditionalGeneration']})
        self.assertIn('vision-language', str(error.exception))

    def test_moe_parameter_count_includes_every_expert(self):
        from strl.derive import params_from_id
        from strl.models import ModelProfile
        qwen3_30b_a3b = {'num_hidden_layers': 48, 'hidden_size': 2048, 'vocab_size': 151936, 'intermediate_size': 6144, 'num_attention_heads': 32,
                         'num_key_value_heads': 4, 'head_dim': 128, 'num_experts': 128, 'moe_intermediate_size': 768}
        self.assertAlmostEqual(ModelProfile().param_estimate_b(qwen3_30b_a3b), 30.1, delta=0.5)
        self.assertEqual(params_from_id('mistralai/Mixtral-8x7B-v0.1'), 56.0)


class AutoLayout(unittest.TestCase):
    def test_multi_node_lora_defaults_to_hsdp_within_a_node(self):
        from strl.experiment import training
        base = {'compute': {'nodes': 2, 'gpus_per_node': 8, 'training': {}}, 'finetune': {'mode': 'lora'}}
        self.assertEqual(training(base)['fsdp_size'], 8)
        self.assertIsNone(training({**base, 'finetune': {'mode': 'full'}})['fsdp_size'])          # full: full shard unless set
        self.assertIsNone(training({**base, 'compute': {'nodes': 1, 'gpus_per_node': 8}})['fsdp_size'])
        self.assertEqual(training({**base, 'compute': {'nodes': 2, 'gpus_per_node': 8, 'training': {'fsdp_size': 16}}})['fsdp_size'], 16)


class Resume(unittest.TestCase):
    """A preempted attempt (borrowed capacity was reclaimed twice) can be continued by a new attempt from its durable checkpoint."""

    def test_overrides_point_skyrl_at_s3_and_the_resume_checkpoint(self):
        import os
        from strl.overrides import durability_values
        cfg = {'evaluation': {'durable_checkpoints': True, 'checkpoint_interval_steps': 2}, 'run': {}}
        self.assertEqual(durability_values({'evaluation': {}, 'run': {}}, 'train'), {})
        old = {k: os.environ.get(k) for k in ('GRPO_ARTIFACT_URI', 'STRL_RESUME_PATH')}
        try:
            os.environ['GRPO_ARTIFACT_URI'] = 's3://b/p/attempts/run1-p1-fb/artifacts'
            os.environ.pop('STRL_RESUME_PATH', None)
            self.assertEqual(durability_values(cfg, 'train'), {'trainer.ckpt_interval': 2, 'trainer.ckpt_path': 's3://b/p/attempts/run1-p1-fb/artifacts/checkpoints'})
            os.environ['STRL_RESUME_PATH'] = 's3://b/p/attempts/run1-fb/artifacts/checkpoints/global_step_2'
            out = durability_values({**cfg, 'run': {'resume_from': 'run1-fb'}}, 'train')
            self.assertEqual((out['trainer.resume_mode'], out['trainer.eval_before_train']), ('from_path', False))
            self.assertTrue(out['trainer.resume_path'].endswith('run1-fb/artifacts/checkpoints/global_step_2'))
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_previous_prefix_and_validation(self):
        from strl.worker import previous_prefix
        self.assertEqual(previous_prefix('s3://b/p/attempts/run1-p1-fb/artifacts', 'run1-fb'), 's3://b/p/attempts/run1-fb/artifacts')
        with self.assertRaises(ExperimentError) as error:
            load(SMOKE, overrides=['evaluation.durable_checkpoints=true', 'budget.save_optimizer_state=false'])
        self.assertIn('save_optimizer_state', str(error.exception))


class ResumePrefix(unittest.TestCase):
    def test_prepare_resume_reads_the_previous_attempt_not_this_one(self):
        # review: prepare_resume passed run_name where it meant previous, so every resume looked in its own (empty) prefix
        from unittest import mock
        from strl import worker
        asked = []

        class S3:
            def get_object(self, Bucket, Key):
                asked.append(Key)
                if '/attempts/run1-fb/' not in Key:
                    raise KeyError(Key)
                return {'Body': io.BytesIO(b'2\n')}

            def download_file(self, b, k, target):
                raise KeyError(k)
        with tempfile.TemporaryDirectory() as temp, mock.patch('boto3.client', lambda *a, **k: S3()), mock.patch.dict(os.environ, {}):
            out = worker.prepare_resume('s3://b/p/attempts/run1-p1-fb/artifacts', 'run1-p1-fb', 'run1-fb', Path(temp))
            self.assertEqual(os.environ['STRL_RESUME_PATH'], 's3://b/p/attempts/run1-fb/artifacts/checkpoints/global_step_2')
        self.assertEqual(out['step'], 2)
        self.assertTrue(asked[0].startswith('p/attempts/run1-fb/'))

    def test_lora_rank_must_be_a_vllm_rank(self):
        from strl.experiment import ExperimentError
        with self.assertRaises(ExperimentError):
            load('experiments/mimo-phi4-mini-lora-verify.yaml', overrides=['finetune.lora.rank=24'])
        load('experiments/mimo-phi4-mini-lora-verify.yaml', overrides=['finetune.lora.rank=32'])

    def test_durable_checkpoints_need_the_in_loop_result(self):
        from strl.experiment import ExperimentError
        with self.assertRaises(ExperimentError):
            load('experiments/gsm8k-qwen35.yaml', overrides=['evaluation.durable_checkpoints=true'])
        load('experiments/mimo-phi4-mini-lora-verify.yaml')     # final_eval: false: fine


class Retention(unittest.TestCase):
    """Plan P15: base vs trained on another environment's held-out set, with THIS experiment's model."""

    def test_the_pod_uses_the_staged_retention_config_not_the_yaml(self):
        # v3smk3-fb: the pod has no base/gsm8k.yaml; it must read the staged retention-resolved.json
        import os as _os
        from strl.experiment import retention_config, save_resolved
        from strl import worker
        cfg = load('experiments/mimo-qwen38-27b-lora.yaml', overrides=['evaluation.retention={experiment: base/gsm8k.yaml, limit: 300}'])
        staged = retention_config(cfg)
        with tempfile.TemporaryDirectory() as temp:
            save_resolved(staged, Path(temp) / 'retention-resolved.json', root='/workspace/strl-run')
            here = _os.getcwd()
            _os.chdir(temp)
            try:
                cfg['evaluation']['retention']['experiment'] = 'base/does-not-exist-in-the-pod.yaml'
                got = worker.retention_in_pod(cfg)
            finally:
                _os.chdir(here)
        self.assertEqual((got['environment']['name'], got['_retention_limit'], got['model']['id']), ('gsm8k', 300, cfg['model']['id']))

    def test_retention_config_uses_the_experiment_model(self):
        from strl.experiment import retention_config
        cfg = load('experiments/mimo-qwen38-27b-lora.yaml', overrides=['evaluation.retention={experiment: base/gsm8k.yaml, limit: 300}'])
        rcfg = retention_config(cfg)
        self.assertEqual(rcfg['environment']['name'], 'gsm8k')
        self.assertEqual(rcfg['model'], cfg['model'])
        self.assertEqual((rcfg['_retention_limit'], rcfg['evaluation']['eval_shards_per_model']), (300, 1))
        self.assertIsNone(retention_config(load('experiments/mimo-qwen38-27b-lora.yaml')))

    def test_retention_needs_two_gpus_and_reserves_time(self):
        from strl.worker import train_reserve_seconds
        with self.assertRaises(ExperimentError):
            load('experiments/mimo-qwen35-0p8b-smoke.yaml', overrides=['evaluation.retention={experiment: base/gsm8k.yaml}'])   # 1 GPU
        cfg = load('experiments/mimo-qwen38-27b-lora.yaml', overrides=['evaluation.retention={experiment: base/gsm8k.yaml}'])
        self.assertEqual(train_reserve_seconds(cfg, 'train') - train_reserve_seconds(load('experiments/mimo-qwen38-27b-lora.yaml'), 'train'), 900)


class RemoteCode(unittest.TestCase):
    """SkyRL loads with trust_remote_code=True: a stale bundled modeling file (Phi-4-mini under transformers 5) must yield to the native class."""

    def _dir(self, config):
        import json as _json
        import tempfile as _tempfile
        d = Path(_tempfile.mkdtemp())
        (d / 'config.json').write_text(_json.dumps(config))
        return d

    def test_native_architecture_drops_auto_map_and_keeps_the_original(self):
        from strl import models
        d = self._dir({'model_type': 'phi3', 'architectures': ['Phi3ForCausalLM'],
                       'auto_map': {'AutoConfig': 'configuration_phi3.Phi3Config', 'AutoModelForCausalLM': 'modeling_phi3.Phi3ForCausalLM'}})
        out = models.prefer_native_code(d)
        self.assertEqual(out['status'], 'remote_code_dropped')
        self.assertNotIn('auto_map', json.loads((d / 'config.json').read_text()))
        self.assertIn('auto_map', json.loads((d / 'config.json.remote').read_text()))
        self.assertEqual(models.model_code_check(d), {'class': 'Phi3ForCausalLM', 'source': models.model_code_check(d)['source']})
        self.assertEqual(models.prefer_native_code(d)['status'], 'native')       # idempotent on a second node / a retry

    def test_unknown_architecture_keeps_its_remote_code(self):
        from strl import models
        d = self._dir({'model_type': 'brand_new', 'architectures': ['BrandNewForCausalLM'], 'auto_map': {'AutoModelForCausalLM': 'm.BrandNewForCausalLM'}})
        self.assertEqual(models.prefer_native_code(d)['status'], 'remote_code')
        self.assertIn('auto_map', json.loads((d / 'config.json').read_text()))

    def test_unloadable_model_fails_the_check(self):
        from strl import models
        d = self._dir({'model_type': 'brand_new', 'architectures': ['BrandNewForCausalLM']})
        with self.assertRaises(RuntimeError):
            models.model_code_check(d)


class FullModeSwitch(unittest.TestCase):
    def test_set_full_on_a_lora_experiment_needs_no_rank_reset(self):
        from strl.experiment import load
        from strl.overrides import scaling_values
        cfg = load(str(ROOT / 'experiments/mimo-phi4-mini-lora-verify.yaml'), overrides=['finetune.mode=full', 'algorithm.lr=1e-6'])
        self.assertFalse(any(k.startswith('trainer.policy.model.lora') for k in scaling_values(cfg, 10)))
