"""The environment axis. An ``EnvironmentSpec`` is everything the generic framework needs to know about one RL environment.

To add an environment: subclass ``EnvironmentSpec`` in ``strl/tasks/<name>/spec.py`` and register it in
``strl/tasks/__init__.py``. Nothing in the trainer, worker, stager or monitoring changes (see
.claude/skills/scaletrain-rl-experiments/references/extending.md ("A new environment")).
"""
from __future__ import annotations

import json
from typing import Any, Callable


class EnvironmentSpec:
    name: str = ''
    generator_kind: str = 'gym_single'        # gym_single | gym_multi | harbor
    verified_env_class: str | None = None     # SkyRL built-in env used when the reward is the default (GPU-verified path)
    generic_env_class: str = 'strl_env'       # registered inside the Ray task; applies reward components to the scorer result
    metric: str = 'accuracy'                  # the yardstick: independent of the training reward, comparable across reward designs
    invalid_metric: str = 'train/invalid_answer_rate'
    needs_modal: bool = False
    default_reward: list[dict] = []
    sample_id_format: str = ''                # final-eval ids, e.g. 'gsm8k-test-%05d'
    eval_module: str = ''                     # python -m <module> --model --output --rank --world-size [--split]
    extra_columns: tuple[str, ...] = ()       # extra trajectory-table columns (see extra_values)
    failure_classes: tuple[str, ...] = ()
    preflight_imports: tuple[str, ...] = ()   # extra modules the CPU preflight must import in the training image
    expected_base_reward: float | None = None # base-model pass rate used for group-signal warnings in `strl plan` (None = unknown)
    needs_litellm: bool = False               # the environment itself calls LiteLLM (judge): the worker writes the key file even without the critic
    uses_coding_executor: bool = True         # `strl prepare` deploys the coding Modal executor and loads a tokenizer for Modal environments

    # ------------------------------------------------------------------ validation / planning
    def validate_experiment(self, cfg: dict) -> list[str]:
        return []

    def reward_functions(self) -> dict[str, Callable]:
        """name -> fn(result: dict, ctx: RewardContext, **params) -> float. Used by the generic reward path."""
        raise NotImplementedError

    def tags(self, cfg: dict) -> list[str]:
        return [self.name]

    # ------------------------------------------------------------------ data
    def prepare_data(self, cfg: dict, **kwargs: Any) -> dict:
        raise NotImplementedError

    def assert_manifest(self, cfg: dict) -> dict:
        raise NotImplementedError

    def data_files(self, cfg: dict, stage: str) -> tuple[list[str], list[str]]:
        """(train parquet files, validation parquet files) for stage 'train' or 'canary'."""
        raise NotImplementedError

    def validation_size(self, cfg: dict) -> int:
        raise NotImplementedError

    def canary_size(self, cfg: dict) -> int:
        """Rows in the canary validation file (the fixed qualitative ids)."""
        return min(cfg['evaluation']['fixed_qualitative_samples'], self.validation_size(cfg))

    def canary_requires_gradient(self, cfg: dict) -> bool:
        """Whether 2 canary updates must have a nonzero gradient (False where all-equal rewards are legitimate)."""
        return True

    def dry_run_checks(self, cfg: dict) -> dict:
        """CPU dry-run: exercise this environment's external dependencies (executors, sandboxes) and return a receipt."""
        return {}

    def expected_rows(self, cfg: dict, split: str) -> int | None:
        """Rows SkyRL must keep for ``split`` ('train'|'validation') after prompt-length filtering; None = no check."""
        return None

    def final_eval_ids(self, cfg: dict) -> list[str]:
        return [self.sample_id_format % i for i in range(cfg['environment']['test_size'])]

    # ------------------------------------------------------------------ scoring: the yardstick (never the training reward)
    def parse(self, text: str) -> Any:
        raise NotImplementedError

    def score(self, text: str, truth: Any, finish_reason: str | None, **extra: Any) -> bool:
        raise NotImplementedError

    def failure_mode(self, row: dict) -> str:
        return 'wrong'

    # ------------------------------------------------------------------ SkyRL wiring
    def skyrl_overrides(self, cfg: dict, stage: str) -> dict[str, Any]:
        """Environment-specific SkyRL overrides (env_class, generator kind, prompt budget...)."""
        raise NotImplementedError

    def uses_generic_reward(self, cfg: dict) -> bool:
        """True when the experiment's reward differs from the verified default and needs the generic reward path.

        Environments without a verified built-in SkyRL env (verified_env_class None) always use the generic path."""
        if self.verified_env_class is None:
            return True
        normalize = lambda comps: sorted(json.dumps({**c, 'weight': float(c.get('weight', 1.0))}, sort_keys=True) for c in comps)
        return normalize(cfg['reward']['components']) != normalize(self.default_reward)

    def register(self, cfg: dict) -> None:
        """Called inside the Ray driver task before the trainer is built: register envs / swap the generator."""

    def patch_runtime(self, skyrl_root: str, cfg: dict) -> None:
        """Idempotent source patches for the pinned SkyRL checkout (e.g. the strict GSM8K verifier)."""

    def generator_contract(self, cfg: dict) -> dict:
        """CPU preflight: exercise the REAL reward path on edge cases and return a receipt dict (raise on failure)."""
        return {}

    def skyrl_config_class(self) -> Any:
        """The SkyRL config dataclass that parses this environment's overrides (Harbor needs HarborSkyRLConfig)."""
        from skyrl.train.config import SkyRLTrainConfig
        return SkyRLTrainConfig

    def build_dataset(self, experiment: dict, cfg: Any, split: str) -> Any:
        """Return a custom SkyRL dataset for 'train'|'validation', or None for SkyRL's PromptDataset (parquet rows)."""
        return None

    def build_generator(self, experiment: dict, cfg: Any, tokenizer: Any, inference_engine_client: Any) -> Any:
        """Return a SkyRL generator instance, or None for SkyRL's default SkyRLGymGenerator. Runs in the Ray driver task."""
        return None

    def train_batch_sampler(self, experiment: dict, dataset: Any, cfg: Any) -> Any:
        """A batch sampler (iterable of index lists with ``__len__``) for the train dataloader, or None for SkyRL's uniform shuffle.
        When set, strl/entrypoint.py swaps the trainer's ``_build_train_dataloader_and_compute_training_steps`` (see tasks/mimo/sampler.py)."""
        return None

    def plan_warnings(self, cfg: dict) -> list[str]:
        """Environment-specific lines for `strl plan` warnings."""
        return []

    # ------------------------------------------------------------------ monitoring hooks
    def train_diagnostics(self, completions: list[str], rewards: list[float], stop_reasons: list, group_size: int,
                          env_metrics: list | None = None) -> dict[str, float]:
        return {}

    def result_metrics(self, result: dict) -> dict[str, float]:
        """Extra numeric per-rollout metrics from a scorer result (aggregated by SkyRL into environment/*)."""
        return {}

    def extra_values(self, index: int, env_metrics: list | None) -> tuple:
        return ()

    def env_metrics_from(self, generator_output: dict, phase: str) -> list | None:
        """Per-row env metric dicts for a generator output that carries none (e.g. Harbor: SkyRL's step merge drops them), or None."""
        return None

    def eval_metrics(self, records: list[dict]) -> dict[str, float]:
        """Extra eval scalars from the collapsed validation records. ``eval/task_metric`` here replaces the default mean of ``correct``."""
        return {}

    def media_table(self, records: list[dict]) -> tuple[list[str], list[list]] | None:
        """(columns, rows) for the `qualitative/media` table (wandb.Image cells), or None."""
        return None

    def critic_modes(self, cfg: dict) -> tuple[str, ...] | None:
        """Failure taxonomy for the LLM critic; None keeps the default (strl.monitor.critic.MODES)."""
        return None

    def eval_record(self, item: dict, trainer: Any, step: int) -> dict:
        """Build one validation record from a SkyRL ``dumped_evals`` row."""
        raise NotImplementedError

    def collapse_eval_records(self, records: list[dict]) -> list[dict]:
        """One record per evaluated sample. Single-turn environments dump exactly one row per sample; step-wise multi-turn
        generators dump one row per turn and must merge them here (the validation check requires unique sample ids)."""
        return records

    def display_prompt(self, prompt: Any) -> str:
        return str(prompt)

    # ------------------------------------------------------------------ in-loop result (final_eval: false) and fail-fast
    def paired_result(self, base_rows: list[dict], trained_rows: list[dict], cfg: dict) -> dict:
        """Paired statistics from two in-loop eval record lists (step-0 base vs last-step trained, same sample ids).

        Default: paired bootstrap on ``correct`` over the shared ids, ``flips``, and how many ids were unpaired. Environments with
        per-domain structure (mimo) override this and add ``domains`` / ``macro`` / ``dropped`` / ``by_source``."""
        from ..finalize import flips
        from ..stats import paired_bootstrap
        b, c = {r['sample_id']: r for r in base_rows}, {r['sample_id']: r for r in trained_rows}
        ids = sorted(set(b) & set(c))
        if not ids:
            raise ValueError('base and trained evaluations share no sample ids')
        e = cfg['evaluation']
        test = paired_bootstrap([bool(b[i]['correct']) for i in ids], [bool(c[i]['correct']) for i in ids], samples=e['bootstrap_samples'],
                                seed=e['bootstrap_seed'], level=e['ci_level'])
        union = set(b) | set(c)
        return {'test': test, 'flips': flips([b[i] for i in ids], [c[i] for i in ids]),
                'dropped': {'total': len(union) - len(ids), 'of': len(union), 'by_domain': {}, 'ids': sorted(union - set(ids))[:50]}}

    def failfast_defaults(self, cfg: dict) -> dict:
        """Default ``budget.failfast`` rules this environment opts into ({} = none): keys mem_gb, domain_masked, step_slow_factor, min_steps."""
        return {}

