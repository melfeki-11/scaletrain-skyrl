"""Verify the pinned router and import the actual training path before allocation."""
import hashlib
import importlib
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import os
import sys
import zipfile

ROUTER_WHEEL = 'vllm_router-0.1.14.post1-cp38-abi3-manylinux_2_35_x86_64.whl'
ROUTER_SHA256 = '4a1fa0bc7708e6498a12ce67f13a1c14e30e6307aeeeddd345448e8264a4718b'


def ensure_router():
    wheel = Path(__file__).resolve().parent.parent / 'assets/runtime' / ROUTER_WHEEL
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != ROUTER_SHA256:
        raise RuntimeError('Pinned router wheel hash mismatch')
    try:
        installed = version('vllm-router')
    except PackageNotFoundError:
        installed = None
    if installed != '0.1.14.post1':
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


def preflight():
    ensure_router()
    from .ipv6_patch import patch_tree
    patch_tree('/workspace/vendor/SkyRL')
    modules = [
        'vllm_router.launch_router',
        'skyrl.backends.skyrl_train.inference_servers.setup',
        'skyrl.backends.skyrl_train.inference_servers.vllm_server_actor',
        'skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker',
        'skyrl.train.trainer',
    ]
    for name in modules:
        importlib.import_module(name)
    return {'router_version': version('vllm-router'), 'training_imports': modules, 'passed': True}


def validate_generator(cfg):
    """Run the actual generator config checks before GPU engine startup."""
    from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
    generator = SkyRLGymGenerator.__new__(SkyRLGymGenerator)
    generator.generator_cfg = cfg.generator
    generator.batched = cfg.generator.batched
    generator.use_conversation_multi_turn = cfg.generator.use_conversation_multi_turn
    generator.custom_chat_template = None
    generator._validate_cfg(cfg.generator)
    from .ipv6_patch import patch_reward_verifier
    patch_reward_verifier('/workspace/vendor/SkyRL')
    import skyrl_gym.envs.gsm8k.utils as reward_utils
    importlib.reload(reward_utils)
    from skyrl_gym.envs.gsm8k.env import GSM8kEnv
    env=GSM8kEnv(extras={'reward_spec':{'ground_truth':'38'}})
    if env.step('work\n#### 38<|im_end|>')['reward'] != 1.0:
        raise RuntimeError('Actual GSM8K EOS reward contract failed')
    if env.step('#### 38<|im_end|> extra')['reward'] != 0.0:
        raise RuntimeError('Reward accepted trailing nonterminal text')
    from skyrl.train.utils.vllm_metrics_scraper import discover_ray_metrics_urls
    from unittest.mock import patch
    with patch('ray.nodes', return_value=[{'Alive':True,'NodeManagerAddress':'2602:fb33::1','MetricsExportPort':-1}, {'Alive':True,'NodeManagerAddress':'2602:fb33::2','MetricsExportPort':8080}]):
        if discover_ray_metrics_urls() != ['http://[2602:fb33::2]:8080/metrics']:
            raise RuntimeError('Actual metrics endpoint contract failed')
    return {'generator_config_passed': True, 'batched': cfg.generator.batched, 'actual_reward_passed':True, 'metrics_endpoints_passed':True}
