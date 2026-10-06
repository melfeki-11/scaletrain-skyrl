"""Tier B: the canonical coding tasks as Harbor task dirs, solved by Terminus-2 in Modal sandboxes.

    SkyRL trainer -> HarborGenerator -> Harbor Trial (Terminus-2) -> Modal sandbox -> tests/test.sh -> reward
          ^                                                                                  |
          +---------------------- tokenized chat history + reward ---------------------------+

Pins: Harbor git 3de07a0 (0.13.1), modal 1.5.3 (SkyRL a9564572 uv.lock); shipped in the payload overlay (tools/build_overlay.py).
Every exported task shares ONE byte-identical Dockerfile (Modal builds and caches a single image), and the sandbox verifier
reuses the same hardened harness as Tier A, so both tiers grade identically.
"""
from __future__ import annotations

import copy
import hashlib
import json
from functools import wraps
from pathlib import Path
from typing import Any

from . import harness

DOCKERFILE = ('FROM python:3.12-slim\n'
              'RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*\n'
              'WORKDIR /app\n')
TASK_TOML = '''schema_version = "1.3"
artifacts = []

[metadata]
source = "strl coding export"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 600.0

[environment]
network_mode = "public"
build_timeout_sec = 600.0
os = "linux"
cpus = 1
memory_mb = 1024
storage_mb = 2048
'''
TEST_SH = '''#!/bin/bash
# reward 0 first: a crashed or killed verifier can never leave a stale pass behind
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
/usr/local/bin/python3 /tests/run_tests.py
exit 0
'''
RUN_TESTS = '''import json, os, sys
sys.path.insert(0, "/tests")
from harness import run_program
task = json.load(open("/tests/task.json"))
code = open("/app/solution.py").read() if os.path.isfile("/app/solution.py") else ""
verdict = run_program({"code": code, "tests": task["tests"], "setup": task["setup"]}, python="/usr/local/bin/python3")
json.dump(verdict, open("/logs/verifier/verdict.json", "w"))
open("/logs/verifier/reward.txt", "w").write("1" if verdict["failure_class"] == "ok" else "0")
'''
INSTRUCTION = ('{prompt}\n\n---\nYou are in a Linux terminal. Write your complete solution to /app/solution.py '
               '(it is imported as a module and graded by hidden unit tests; do not print or read input). '
               'You may run python3 to check your work. Finish as soon as /app/solution.py is written.\n')
TEMPLATE = {
    'trials_dir': 'artifacts/harbor-trials', 'timeout_multiplier': 1.0,
    'agent': {'name': 'terminus-2', 'override_timeout_sec': 600, 'kwargs': {
        'max_turns': 8, 'suppress_max_turns_warning': True, 'enable_summarize': False, 'collect_rollout_details': True, 'temperature': 1.0,
        'model_info': {'max_input_tokens': 8192, 'max_output_tokens': 1024, 'input_cost_per_token': 0.0, 'output_cost_per_token': 0.0},
        'llm_kwargs': {'timeout': 600, 'max_retries': 0, 'top_p': 1.0}}},
    'environment': {'type': 'modal', 'override_cpus': 1, 'override_memory_mb': 1024, 'override_storage_mb': 2048,
                    'suppress_override_warnings': True, 'kwargs': {}},
    'verifier': {'disable': False, 'override_timeout_sec': 120},
}


def export_tasks(splits: dict[str, list], root: Path) -> dict:
    """Write Harbor task dirs (instruction, shared Dockerfile, tests/{test.sh,run_tests.py,harness.py,task.json}, oracle solve.sh)."""
    harness_src = Path(harness.__file__).read_text()
    counts = {}
    for split, tasks in splits.items():
        for task in tasks:
            d = root / split / task.sample_id
            for sub in ('environment', 'tests', 'solution'):
                (d / sub).mkdir(parents=True, exist_ok=True)
            (d / 'instruction.md').write_text(INSTRUCTION.format(prompt=task.prompt.split('Return only the complete')[0].strip()))
            (d / 'task.toml').write_text(TASK_TOML)
            (d / 'environment/Dockerfile').write_text(DOCKERFILE)
            (d / 'tests/test.sh').write_text(TEST_SH)
            (d / 'tests/run_tests.py').write_text(RUN_TESTS)
            (d / 'tests/harness.py').write_text(harness_src)
            (d / 'tests/task.json').write_text(json.dumps({'tests': task.tests, 'setup': task.setup}))
            (d / 'solution/solve.sh').write_text("#!/bin/bash\ncat > /app/solution.py <<'STRL_EOF'\n" + task.reference + '\nSTRL_EOF\n')
            for script in ('tests/test.sh', 'solution/solve.sh'):
                (d / script).chmod(0o755)
        counts[split] = len(tasks)
    return {'root': str(root.name), 'counts': counts, 'dockerfile_sha256': hashlib.sha256(DOCKERFILE.encode()).hexdigest()}


# Terminus-2's first turn alone (system template + task + terminal) measured 801-2,511 tokens (median 826) on these tasks; the
# chat then grows by one completion + one terminal observation per turn. One number sizes the vLLM engine, the agent's
# context check and the trainer's sequence limit, so they can never disagree (can-hb-b: a 1,024 engine context failed every
# trial on turn 1 with ContextLengthExceededError).
DEFAULT_MAX_SEQ_LEN = 8192
FIRST_TURN_MAX_TOKENS = 2600


def max_seq_len(cfg: dict) -> int:
    return int((cfg['environment'].get('harbor') or {}).get('max_seq_len', DEFAULT_MAX_SEQ_LEN))


def trial_template(cfg: dict) -> dict:
    """The shared trial template for training AND evaluation (one source => the trained agent is the evaluated agent)."""
    from ... import models
    t = copy.deepcopy(TEMPLATE)
    hb = cfg['environment'].get('harbor') or {}
    kw = t['agent']['kwargs']
    kw['max_turns'] = int(hb.get('max_turns', kw['max_turns']))
    t['agent']['override_timeout_sec'] = int(hb.get('agent_timeout_sec', t['agent']['override_timeout_sec']))
    a = cfg['algorithm']
    kw['llm_kwargs']['max_tokens'] = a['max_generate_length']
    kw['model_info'].update(max_input_tokens=max_seq_len(cfg), max_output_tokens=a['max_generate_length'])
    kw['temperature'] = a['temperature']
    chat = models.detect(cfg['model'])[0].chat_kwargs(cfg)
    if chat:  # e.g. Qwen thinking off: otherwise thinking burns the per-turn budget and Terminus re-prompts past max_turns
        kw['llm_kwargs'].setdefault('extra_body', {})['chat_template_kwargs'] = chat
    modal = (cfg.get('services') or {}).get('modal') or {}
    t['environment']['kwargs']['secrets'] = [modal.get('sandbox_secret', 'modal-proxy-user')]
    t['trials_dir'] = str(Path(cfg['run']['artifact_dir']) / 'harbor-trials')
    return t


def skyrl_overrides(cfg: dict) -> dict[str, Any]:
    hb = cfg['environment'].get('harbor') or {}
    a = cfg['algorithm']
    return {
        'trainer.algorithm.max_seq_len': max_seq_len(cfg),
        'generator.inference_engine.engine_init_kwargs.max_model_len': max_seq_len(cfg),
        'generator.step_wise_trajectories': True,
        'generator.merge_stepwise_output': True,
        'generator.apply_overlong_filtering': True,
        'generator.rate_limit.enabled': True,
        'generator.rate_limit.max_concurrency': int(hb.get('max_concurrency', 64)),
        'generator.rate_limit.trajectories_per_second': float(hb.get('trajectories_per_second', 4)),
    }


def config_class():
    import sys
    from ...runtime import SKYRL_ROOT
    if SKYRL_ROOT not in sys.path:
        sys.path.insert(0, SKYRL_ROOT)
    from examples.train_integrations.harbor.entrypoints.main_harbor import HarborSkyRLConfig
    return HarborSkyRLConfig


def build_dataset(cfg: Any, split: str) -> Any:
    """SkyRL's HarborTaskDataset (one item per task directory), as examples/train_integrations/harbor/entrypoints/main_harbor.py."""
    config_class()  # puts the SkyRL root on sys.path
    from examples.train_integrations.harbor.dataset import HarborTaskDataset
    files = cfg.data.train_data if split == 'train' else cfg.data.val_data
    if split != 'train' and not (cfg.trainer.eval_interval > 0 and files):
        return None

    class SourceLabelledHarborDataset(HarborTaskDataset):
        """SkyRL groups eval metrics by env_extras.data_source, which upstream sets to each task's absolute path (one W&B key
        set per task). Label by benchmark instead (mbpp / humaneval): eval/<source>/pass_at_1 like coding_exec."""

        def __getitem__(self, index):
            return _relabel(super().__getitem__(index))

        def __iter__(self):
            return (_relabel(item) for item in super().__iter__())

    dataset = SourceLabelledHarborDataset(data_files=list(files))
    if split == 'train' and len(dataset) < cfg.trainer.train_batch_size:
        raise RuntimeError(f'Harbor train set has {len(dataset)} tasks < train_batch_size {cfg.trainer.train_batch_size}')
    return dataset


def _relabel(item: dict) -> dict:
    """Task dirs are named <source>-<split>-<id> (export_tasks); the prompt (the task path) is untouched."""
    name = Path(str(item['prompt'])).name
    extras = dict(item.get('env_extras') or {}, task_dir=str(item['prompt']),   # identity for eval records (spec.eval_record)
                  data_source=name.split('-', 1)[0] if '-' in name else 'harbor')
    return dict(item, env_extras=extras)


def build_generator(experiment: dict, cfg: Any, tokenizer: Any, inference_engine_client: Any) -> Any:
    install_compat()
    from examples.train_integrations.harbor.harbor_generator import HarborGenerator
    return HarborGenerator(generator_cfg=cfg.generator, harbor_cfg=trial_template(experiment), inference_engine_client=inference_engine_client,
                           tokenizer=tokenizer, max_seq_len=cfg.trainer.algorithm.max_seq_len)


def build_trial_config(template: dict, task_path: str, *, model_name: str | None, api_base: str | None, agent_name: str | None = None) -> dict:
    """Exactly HarborGenerator._harbor_agent_loop's per-trial construction (SkyRL a9564572)."""
    from uuid import uuid4
    config = copy.deepcopy(template)
    agent = config.setdefault('agent', {})
    if agent_name:
        agent['name'] = agent_name
    if model_name is not None:
        agent['model_name'] = model_name
    kwargs = agent.setdefault('kwargs', {})
    if api_base is not None:
        kwargs['api_base'] = api_base
    config['task'] = {'path': str(task_path)}
    kwargs['session_id'] = uuid4().hex
    if agent.get('name') in {'oracle', 'nop'}:
        agent.pop('model_name', None)
        agent['kwargs'] = {}
    return config


def contract(cfg: dict) -> dict:
    """CPU: the generator's template validates as a Harbor TrialConfig for a real exported task."""
    from harbor.models.trial.config import TrialConfig
    task = sorted((Path(cfg['run']['data_dir']) / 'harbor/canary').iterdir())[0]
    template = trial_template(cfg)
    TrialConfig.model_validate(build_trial_config(template, str(task), model_name='hosted_vllm/policy', api_base='http://127.0.0.1:8000/v1'))
    return {'trial_config_valid': True, 'agent': template['agent']['name'], 'max_turns': template['agent']['kwargs']['max_turns'],
            'sandbox': template['environment']['type'], 'actual_reward_passed': True}


def dry_run_checks(cfg: dict, n: int = 2) -> dict:
    """From the pod: oracle (reference solution) must score 1 and nop must score 0 on real exported tasks in Modal sandboxes."""
    import asyncio
    install_compat()
    tasks = sorted((Path(cfg['run']['data_dir']) / 'harbor/canary').iterdir())[:n]
    template = trial_template(cfg)
    results = asyncio.run(_run([build_trial_config(template, str(t), model_name=None, api_base=None, agent_name=a) for a in ('oracle', 'nop') for t in tasks]))
    oracle, nop = results[:n], results[n:]
    receipt = {'oracle_rewards': [r['reward'] for r in oracle], 'nop_rewards': [r['reward'] for r in nop],
               'seconds': [r['seconds'] for r in results]}
    if any(r['reward'] != 1.0 for r in oracle) or any(r['reward'] != 0.0 for r in nop):
        raise RuntimeError(f'Harbor sandbox verifier check failed: {receipt} {[r.get("exception") for r in results]}')
    return receipt


async def _run(configs: list[dict]) -> list[dict]:
    import asyncio
    return await asyncio.gather(*(run_trial(c) for c in configs))


async def run_trial(config: dict) -> dict:
    import time
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial
    started = time.monotonic()
    try:
        results = await (await Trial.create(TrialConfig.model_validate(config))).run()
    except Exception as error:
        return {'reward': None, 'exception': f'{type(error).__name__}: {error}'[:300], 'seconds': round(time.monotonic() - started, 1)}
    verifier = results.verifier_result
    return {'reward': float(verifier.rewards['reward']) if verifier and verifier.rewards else None,
            'exception': results.exception_info.exception_type if results.exception_info else None, 'seconds': round(time.monotonic() - started, 1)}


# ------------------------------------------------------------------------------------------------------------- compat patch
_PATCHED = '_strl_modal_harbor_patched'


def install_compat() -> list[str]:
    """Harbor 3de07a0 + current Modal SDK (verified in the scaletrain-modal-harbor canary): execs run in the Dockerfile WORKDIR
    (upstream fix f67fd0cc is missing), current sandbox.filesystem API for is_dir/is_file, and a verified /tests upload.
    Must run in the process that runs trials (the Ray driver task)."""
    from harbor.environments import modal as harbor_modal
    from harbor.environments.definition import parse_dockerfile_workdir
    env_cls = harbor_modal.ModalEnvironment
    if getattr(env_cls, _PATCHED, False):
        return ['already-installed']
    applied = []

    def _err(exc, *names):
        return type(exc).__name__ in names

    strategy = getattr(harbor_modal, '_ModalStrategy', None)
    if strategy is not None:
        async def is_dir(self, path, user=None):
            try:
                await self._env._sandbox.filesystem.list_files.aio(path)
                return True
            except Exception as exc:
                if _err(exc, 'SandboxFilesystemNotADirectoryError', 'NotADirectoryError', 'SandboxFilesystemNotFoundError', 'FileNotFoundError'):
                    return False
                raise

        async def is_file(self, path, user=None):
            try:
                await self._env._sandbox.filesystem.list_files.aio(path)
                return False
            except Exception as exc:
                if _err(exc, 'SandboxFilesystemNotADirectoryError', 'NotADirectoryError'):
                    return True
                if _err(exc, 'SandboxFilesystemNotFoundError', 'FileNotFoundError'):
                    return False
                raise

        strategy.is_dir, strategy.is_file = is_dir, is_file
        applied.append('modal-filesystem-api')
    original_exec = env_cls.exec

    @wraps(original_exec)
    async def exec_in_workdir(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if cwd is None and self.task_env_config.workdir is None and not self._uses_compose:
            cwd = parse_dockerfile_workdir(self._environment_definition_path)
        return await original_exec(self, command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    env_cls.exec = exec_in_workdir
    applied.append('dockerfile-workdir')
    setattr(env_cls, _PATCHED, True)
    return applied
