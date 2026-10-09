"""In-image runtime preparation and the CPU preflight of the REAL training path (run before any GPU is allocated)."""
import hashlib
import importlib
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import os
import sys
import zipfile

ROUTER_WHEEL = 'vllm_router-0.1.14.post1-cp38-abi3-manylinux_2_35_x86_64.whl'
ROUTER_SHA256 = '4a1fa0bc7708e6498a12ce67f13a1c14e30e6307aeeeddd345448e8264a4718b'
SKYRL_ROOT = os.environ.get('SKYRL_ROOT', '/workspace/vendor/SkyRL')
TRAINING_IMPORTS = [
    'vllm_router.launch_router',
    'skyrl.backends.skyrl_train.inference_servers.setup',
    'skyrl.backends.skyrl_train.inference_servers.vllm_server_actor',
    'skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker',
    'skyrl.train.trainer',
]


def ensure_router():
    """The image skips vllm-router (--no-install-package); load the SkyRL-locked wheel from a writable overlay."""
    try:
        installed = version('vllm-router')
    except PackageNotFoundError:
        installed = None
    if installed == '0.1.14.post1':
        return
    wheel = Path(__file__).resolve().parent.parent / 'assets/runtime' / ROUTER_WHEEL
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != ROUTER_SHA256:
        raise RuntimeError('Pinned router wheel hash mismatch')
    target = wheel.parent / 'packages'
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise RuntimeError('Unsafe router wheel path')
        archive.extractall(target)
    sys.path.insert(0, str(target))
    os.environ['PYTHONPATH'] = str(target) + os.pathsep + os.environ.get('PYTHONPATH', '')
    importlib.invalidate_caches()
    if version('vllm-router') != '0.1.14.post1':
        raise RuntimeError('Router runtime version mismatch')


# Multi-node LoRA: pinned SkyRL writes the adapter to trainer.policy.model.lora.lora_sync_path (a node-local /tmp dir) from global
# rank 0 only, then asks EVERY vLLM server to load it from that path; servers on other hosts cannot read it. collect_lora_params
# already all-gathers the full adapter on every rank, so the first rank on each host writes that host's copy, all ranks
# barrier, and only then rank 0 sends the load request.
LORA_SYNC_PER_HOST = {
    'skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py': [
        ('        if torch.distributed.get_rank() == 0:\n'
         '            os.makedirs(lora_sync_path, exist_ok=True)\n',
         '        import socket as _strl_socket\n'
         '        _strl_hosts = [None] * torch.distributed.get_world_size()\n'
         '        torch.distributed.all_gather_object(_strl_hosts, _strl_socket.gethostname())\n'
         '        _strl_rank = torch.distributed.get_rank()\n'
         '        if _strl_hosts.index(_strl_hosts[_strl_rank]) == _strl_rank:  # strl: first rank on each host writes its copy\n'
         '            os.makedirs(lora_sync_path, exist_ok=True)\n'),
        ('\n            # Send LoRA disk loading request to inference engine.\n',
         '\n        torch.distributed.barrier()  # strl: every host holds the new adapter before any engine reloads it\n'
         '        if _strl_rank == 0:\n'
         '            # Send LoRA disk loading request to inference engine.\n'),
    ],
}
# budget.save_optimizer_state=false renders trainer.ckpt_interval=-1; the strl deadline callback still forces a save at the stop
# step, and every FSDP checkpoint carries optimizer shards, so a forced save is honoured only when checkpoints are enabled.
NO_OPTIMIZER_CHECKPOINT = {
    'skyrl/train/trainer.py': [
        ('                        will_save_ckpts = force_save or ckpt_interval_save\n',
         '                        will_save_ckpts = (force_save and self.cfg.trainer.ckpt_interval > 0) or ckpt_interval_save  # strl\n'),
    ],
}

# Durable checkpoints put trainer.ckpt_path on S3. Every FSDP checkpoint file goes through SkyRL's cloud-aware io, EXCEPT the LoRA
# adapter, written with safetensors.save_file(path): v1phi3-fb failed at its step-2 save with `No such file or directory ... at path
# "/workspace/strl-run/s3://..."`. Serialize to bytes and write through io.open_file, like the adapter_config.json next to it.
LORA_ADAPTER_CLOUD_SAVE = {
    'skyrl/backends/skyrl_train/distributed/fsdp_strategy.py': [
        ('            save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))\n',
         '            from safetensors.torch import save as _strl_safetensors_bytes  # strl: save_file cannot write to s3://\n'
         '            with io.open_file(os.path.join(lora_save_path, "adapter_model.safetensors"), "wb") as _strl_f:\n'
         '                _strl_f.write(_strl_safetensors_bytes(lora_params))\n'),
    ],
}

# resume_mode from_path wraps trainer.resume_path in pathlib.Path, which collapses s3:// to s3:/ (v2res-fb: `Checkpoint path not found:
# s3:/scale-ml/...`). Keep it a string; os.path.basename replaces Path.name. The LATEST mode already uses strings.
RESUME_FROM_CLOUD_PATH = {
    'skyrl/train/trainer.py': [
        ('            checkpoint_path = Path(self.cfg.trainer.resume_path)\n',
         '            checkpoint_path = str(self.cfg.trainer.resume_path or "")  # strl: Path() would collapse s3:// to s3:/\n'),
        ('            if GLOBAL_STEP_PREFIX not in checkpoint_path.name:\n',
         '            if GLOBAL_STEP_PREFIX not in os.path.basename(checkpoint_path.rstrip("/")):\n'),
    ],
}


def _merge(patches: dict, more: dict) -> None:
    for path, pairs in more.items():   # several patches may target one file: keep them all
        patches.setdefault(path, [])
        patches[path] = [*patches[path], *pairs]


def scaling_patches(cfg) -> dict:
    """The exact-source SkyRL patches an experiment needs (none for single-node full fine-tuning with checkpoints)."""
    from .experiment import finetune, save_optimizer_state
    patches: dict = {}
    if cfg['compute']['nodes'] > 1 and finetune(cfg)['mode'] == 'lora':
        _merge(patches, LORA_SYNC_PER_HOST)
    if not save_optimizer_state(cfg):
        _merge(patches, NO_OPTIMIZER_CHECKPOINT)
    from .overrides import durable
    if durable(cfg) and finetune(cfg)['mode'] == 'lora':
        _merge(patches, LORA_ADAPTER_CLOUD_SAVE)
    if (cfg.get('run') or {}).get('resume_from'):
        _merge(patches, RESUME_FROM_CLOUD_PATH)
    return patches


def patch_scaling(root, cfg) -> list[str]:
    """Run on EVERY node before Ray starts: Ray workers import SkyRL from each node's own tree."""
    from .ipv6_patch import patch_tree
    patches = scaling_patches(cfg)
    if patches:
        patch_tree(root, patches)
    return sorted(patches)


def preflight(env=None):
    ensure_router()
    from .ipv6_patch import patch_tree
    patch_tree(SKYRL_ROOT)
    modules = TRAINING_IMPORTS + list(getattr(env, 'preflight_imports', ()) or ())
    for name in modules:
        importlib.import_module(name)
    return {'router_version': version('vllm-router'), 'training_imports': modules, 'passed': True}


def validate_generator(cfg, experiment=None):
    """Actual generator config checks + the environment's reward contract + metrics-endpoint discovery (CPU, no engines)."""
    from .tasks import get_environment
    from .experiment import current
    experiment = experiment or current()
    env = get_environment(experiment['environment']['name'])
    from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
    generator = SkyRLGymGenerator.__new__(SkyRLGymGenerator)
    generator.generator_cfg = cfg.generator
    generator.batched = cfg.generator.batched
    generator.use_conversation_multi_turn = cfg.generator.use_conversation_multi_turn
    generator.custom_chat_template = None
    generator._validate_cfg(cfg.generator)
    env.patch_runtime(SKYRL_ROOT, experiment)
    receipt = {'generator_config_passed': True, 'batched': cfg.generator.batched, **env.generator_contract(experiment)}
    from skyrl.train.utils.vllm_metrics_scraper import discover_ray_metrics_urls
    from unittest.mock import patch
    with patch('ray.nodes', return_value=[{'Alive': True, 'NodeManagerAddress': '2602:fb33::1', 'MetricsExportPort': -1},
                                          {'Alive': True, 'NodeManagerAddress': '2602:fb33::2', 'MetricsExportPort': 8080}]):
        if discover_ray_metrics_urls() != ['http://[2602:fb33::2]:8080/metrics']:
            raise RuntimeError('Actual metrics endpoint contract failed')
    receipt['metrics_endpoints_passed'] = True
    return receipt
