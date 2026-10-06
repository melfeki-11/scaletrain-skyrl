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
