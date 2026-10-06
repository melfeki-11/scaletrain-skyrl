"""W&B monitoring inside the SkyRL driver task: one run per attempt, every metric on the ``optimizer_step`` axis.

Quantitative: SkyRL-native metrics + group-signal diagnostics + environment diagnostics + reward components + budget.
Qualitative: ``trajectories/train`` (full GRPO groups each step), ``qualitative/fixed_validation`` (the same fixed IDs at every
eval), ``qualitative/critic`` (LiteLLM failure analysis, off the training path), plus final tables written by the worker.
Fail-fast: non-finite metrics raise (after a W&B alert); the wall-clock deadline forces a final save + eval.

Never pass W&B's reserved ``step=``: eval/metadata/resume rows advance W&B's internal counter and silently drop metrics.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from numbers import Real
from pathlib import Path
from typing import Any, Callable

AXIS = 'optimizer_step'
SIGNAL: dict[int, float] = {}   # optimizer step -> learning-signal fraction (written by the trajectory logger, read by the canary)
AXIS_PATTERNS = (
    'train/*', 'eval/*', 'timing/*', 'budget/*', 'checkpoint/*', 'policy/*', 'reward/*', 'trainer/*', 'generate/*', 'gpu/*',
    'vllm/*', 'environment/*', 'reward_components/*', 'critic/*', 'qualitative*', 'trajectories/*', 'test/*', 'loss/*', 'ray/*',
)


def _env():
    """The environment spec of the running experiment (None in unit tests without a resolved config)."""
    try:
        from .experiment import current
        from .tasks import get_environment
        return get_environment(current()['environment']['name'])
    except Exception:
        return None


def rollout_diagnostics(completions: list[str], rewards: list[float], stop_reasons: list[str | None], group_size: int,
                        parse: Callable[[str], Any] | None = None, invalid_key: str = 'train/invalid_answer_rate') -> dict[str, float]:
    """Group-signal diagnostics. GRPO only learns from groups whose rewards differ: watch learning_signal_fraction."""
    if group_size <= 0:
        raise ValueError('group_size must be positive')
    groups = [rewards[index:index + group_size] for index in range(0, len(rewards), group_size)]
    variances = []
    for group in groups:
        mean = sum(group) / len(group)
        variances.append(sum((value - mean) ** 2 for value in group) / len(group))
    out = {
        'train/within_group_reward_variance': sum(variances) / len(variances) if variances else 0.0,
        'train/learning_signal_fraction': sum(value > 0.0 for value in variances) / len(variances) if variances else 0.0,
        'train/truncation_rate': sum(value != 'stop' for value in stop_reasons) / len(stop_reasons) if stop_reasons else 0.0,
    }
    if parse is not None:
        out[invalid_key] = sum(parse(value) is None for value in completions) / len(completions) if completions else 0.0
    return out


def per_trajectory(is_last_step: list | None, prompts: list, completions: list[str], rewards: list[float],
                   stop_reasons: list) -> tuple[list, list[str], list[float], list]:
    """Step-wise generators (Harbor) emit one row per agent turn, contiguous per trajectory, with the reward on the last turn.
    GRPO groups, the learning signal and the critic are about trajectories: sum each trajectory's turn rewards and keep its
    last turn. Single-turn output (no is_last_step) passes through unchanged."""
    if not is_last_step or len(is_last_step) != len(rewards):
        return prompts, completions, rewards, stop_reasons
    out_p, out_c, out_r, out_s, total = [], [], [], [], 0.0
    for index, last in enumerate(is_last_step):
        total += rewards[index]
        if last:
            out_p.append(prompts[index] if index < len(prompts) else None)
            out_c.append(completions[index]); out_r.append(total); out_s.append(stop_reasons[index])
            total = 0.0
    return out_p, out_c, out_r, out_s


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def alert(title: str, text: str, level: str = 'ERROR') -> None:
    """Best-effort W&B alert (needs alerts enabled in the user's W&B settings). Never raises."""
    if os.environ.get('STRL_ALERTS', '1') != '1':
        return
    try:
        import wandb
        if wandb.run is not None:
            wandb.alert(title=title[:64], text=text[:1000], level=getattr(wandb.AlertLevel, level), wait_duration=300)
    except Exception:
        pass


def make_tracking_class():
    from skyrl.train.utils.tracking import Tracking

    class AxisTracking(Tracking):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._tables: dict[str, Any] = {}
            if self.backend == 'wandb':
                self.logger.define_metric(AXIS)
                for pattern in AXIS_PATTERNS:
                    self.logger.define_metric(pattern, step_metric=AXIS)
                provenance = {k: os.environ[v] for k, v in (('strl/experiment', 'STRL_EXPERIMENT_NAME'), ('strl/experiment_sha', 'STRL_EXPERIMENT_SHA'),
                                                             ('strl/environment', 'STRL_ENVIRONMENT'), ('strl/attempt', 'GRPO_RUN_NAME'))
                              if os.environ.get(v)}
                try:
                    from .experiment import current
                    provenance['strl/resolved'] = {k: v for k, v in current().items() if not k.startswith('_')}
                except Exception:
                    pass
                if provenance:
                    self.logger.config.update(provenance, allow_val_change=True)

        def log(self, data: dict[str, Any], step: int, commit: bool = False):
            payload = dict(data)
            payload.setdefault(AXIS, int(step))
            mirror = os.environ.get('STRL_METRICS_JSONL')
            if mirror:  # complete local record of every scalar sent to W&B (S3 is the source of truth; compare/verify read it)
                scalars = {k: (float(v) if not isinstance(v, (bool, int)) else v) for k, v in payload.items() if isinstance(v, (bool, Real))}
                if len(scalars) > 1:
                    Path(mirror).parent.mkdir(parents=True, exist_ok=True)
                    with open(mirror, 'a', encoding='utf-8') as handle:
                        handle.write(json.dumps(scalars, sort_keys=True) + '\n')
            if self.backend == 'wandb':
                self.logger.log(data=payload, commit=commit)
            else:
                self.logger.log(data=payload, step=step)

        def log_samples_to_table(self, key, columns, samples, step):
            """Append rows to a per-key table. INCREMENTAL tables upload only new rows (no O(n^2) re-upload)."""
            if self.backend != 'wandb':
                return
            import wandb
            table = self._tables.get(key)
            if table is None:
                try:
                    table = wandb.Table(columns=list(columns), log_mode='INCREMENTAL')
                except TypeError:  # older SDK: fall back to a mutable table re-logged with all rows
                    table = wandb.Table(columns=list(columns))
                self._tables[key] = table
            for row in samples:
                table.add_data(*row)
            self.logger.log({key: table, AXIS: int(step)})

    return AxisTracking


def make_trajectory_logger_class():
    from skyrl.train.utils.trajectory_logging import TrajectoryLogger

    class GroupTrajectoryLogger(TrajectoryLogger):
        """Logs complete GRPO groups (the first N rows = whole groups), plus diagnostics, and feeds the critic."""
        base_columns = ('step', 'idx', 'group_id', 'rollout_index', 'prompt', 'completion', 'parsed_answer', 'reward',
                        'token_count', 'finish_reason')

        def __init__(self, *args: Any, **kwargs: Any):
            self.env = _env()
            kwargs.setdefault('columns', self.base_columns + tuple(self.env.extra_columns if self.env else ()))
            super().__init__(*args, **kwargs)

        @property
        def columns_all(self) -> tuple[str, ...]:
            return self.base_columns + tuple(self.env.extra_columns if self.env else ())

        def log(self, *, tracker, num_samples, prompts, generator_output, tokenizer, global_step, **kwargs):
            response_ids = generator_output.get('response_ids') or []
            rewards = generator_output.get('rewards') or []
            stop_reasons = generator_output.get('stop_reasons') or [None] * len(response_ids)
            env_metrics = generator_output.get('env_metrics')
            scalar_rewards = [float(sum(value)) if isinstance(value, list) else float(value) for value in rewards]
            is_training = kwargs.get('wandb_key') == 'trajectories/train'
            group_size = int(os.environ.get('GRPO_SAMPLES_PER_PROMPT', '8')) if is_training else 1
            decoded = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in response_ids]
            step = 0 if global_step is None else int(global_step)
            if is_training:
                prompts_t, decoded_t, rewards_t, stops_t = per_trajectory(generator_output.get('is_last_step'), list(prompts), decoded,
                                                                         scalar_rewards, list(stop_reasons))
                env_metrics_t = env_metrics if decoded_t is decoded else None   # step-wise: per-turn env metrics do not align
                metrics = rollout_diagnostics(decoded_t, rewards_t, stops_t, group_size)
                SIGNAL[step] = metrics['train/learning_signal_fraction']
                if self.env is not None:
                    metrics.update(self.env.train_diagnostics(decoded_t, rewards_t, stops_t, group_size, env_metrics_t))
                tracker.log(metrics, step=step, commit=False)
                try:
                    from .monitor.critic import submit_failures
                    submit_failures(step, prompts_t, decoded_t, rewards_t, env_metrics_t, self.env)
                except Exception as error:  # the critic must never affect training
                    print(f'critic submit skipped: {type(error).__name__}: {error}', flush=True)
            self._finish_reasons = list(stop_reasons)
            self._env_metrics = env_metrics
            self._group_size = group_size
            super().log(tracker=tracker, num_samples=num_samples, prompts=prompts, generator_output=generator_output,
                        tokenizer=tokenizer, global_step=global_step, **kwargs)

        def select_sample_indices(self, *, num_samples: int, rewards: list[float], total: int) -> list[int]:
            return list(range(min(num_samples, total)))

        def build_samples(self, *, num_samples, prompts, response_ids, rewards, loss_masks, num_turns_list, tokenizer, **kwargs):
            total = min(num_samples, len(response_ids), len(prompts), len(rewards))
            rows = []
            parse = self.env.parse if self.env is not None else (lambda text: None)
            for index in range(total):
                reward = rewards[index]
                scalar = float(sum(reward)) if isinstance(reward, list) else float(reward)
                completion = tokenizer.decode(response_ids[index], skip_special_tokens=True)
                finish_reason = self._finish_reasons[index] if index < len(self._finish_reasons) else None
                prompt = self.env.display_prompt(prompts[index]) if self.env is not None else str(prompts[index])
                extra = self.env.extra_values(index, self._env_metrics) if self.env is not None else ()
                parsed = parse(completion)
                rows.append((index, index // self._group_size, index % self._group_size, prompt, completion,
                             parsed if parsed is None or isinstance(parsed, (str, int, float, bool)) else str(parsed)[:200],
                             scalar, len(response_ids[index]), finish_reason, *extra))
            return rows

    return GroupTrajectoryLogger


def env_failure_fraction(metrics: dict, prompts_per_step: int) -> float | None:
    """Share of GRPO groups masked because the environment failed (not the model): coding_exec logs it directly; Harbor's
    generator reports masked instances (any failed trial masks its whole group)."""
    if isinstance(metrics.get('environment/masked_group_fraction'), Real):
        return float(metrics['environment/masked_group_fraction'])
    if isinstance(metrics.get('generate/num_masked_instances'), Real):
        return float(metrics['generate/num_masked_instances']) / max(1, int(prompts_per_step))
    return None


CANARY_MAX_ENV_FAILURE = 0.5   # a canary whose groups are mostly masked never exercised the environment: fail it


def make_callback_class():
    from skyrl.train.utils.callbacks import TrainingCallback

    class EvidenceCallback(TrainingCallback):
        eval_columns = ('step', 'sample_id', 'prompt', 'completion', 'ground_truth', 'parsed_answer', 'correct', 'reward',
                        'token_count', 'finish_reason')

        def __init__(self, checkpoint_minutes: float = 15.0):
            self.checkpoint_seconds = checkpoint_minutes * 60.0
            self.last_checkpoint_at = time.monotonic()
            self.started_at = self.last_checkpoint_at
            self.baseline_accuracy = None
            self.max_grad = 0.0
            self.env_failures: dict[int, float] = {}
            self.env = _env()

        def on_step_end(self, trainer, callback_input, control) -> None:
            step = int(callback_input.global_step)
            metrics = callback_input.metrics or {}
            for key, value in metrics.items():
                if isinstance(value, Real) and not math.isfinite(float(value)):
                    alert('Non-finite metric', f'{key}={value} at step {step} ({os.environ.get("GRPO_RUN_NAME")})')
                    raise RuntimeError(f'Nonfinite optimizer metric: {key}')
            gradients = [float(v) for k, v in metrics.items() if 'grad_norm' in k]
            self.max_grad = max([self.max_grad, *gradients]) if gradients else self.max_grad
            failure = env_failure_fraction(getattr(trainer, 'all_metrics', None) or {}, trainer.cfg.trainer.train_batch_size)
            if failure is not None:
                self.env_failures[step] = failure
                if failure > 0.05 and not any(v > 0.05 for s, v in self.env_failures.items() if s < step):
                    alert('Environment failures', f'{failure:.0%} of GRPO groups masked by environment errors at step {step} '
                          f'({os.environ.get("GRPO_RUN_NAME")}); see environment/* and generate/num_error_trajectories', 'WARN')
            if step == 2:
                worst = max([self.env_failures.get(s, 0.0) for s in (1, 2)])
                if worst >= CANARY_MAX_ENV_FAILURE:
                    alert('Canary failed', f'{worst:.0%} of GRPO groups were masked by environment failures in steps 1-2')
                    raise RuntimeError(f'Canary: {worst:.0%} of GRPO groups were masked by environment failures (the environment, not '
                                       'the model, failed: inspect harbor-trials/*/exception.txt or environment/failure/infra_error)')
                # Canary: two real finite updates. A zero gradient is a bug only if some GRPO group had reward variance;
                # all-equal groups (every sample passed or every sample failed) legitimately give exactly zero advantage.
                signal = max([SIGNAL.get(s, 0.0) for s in (1, 2)] or [0.0])
                if self.max_grad <= 0 and signal > 0 and os.environ.get('GRPO_CANARY_REQUIRE_GRADIENT', '1') == '1':
                    alert('Canary failed', f'learning signal {signal:.2f} but zero policy gradient after 2 updates')
                    raise RuntimeError('Canary: groups had reward variance but the policy gradient is zero (update path broken)')
                if self.max_grad <= 0:
                    alert('Canary: no learning signal', 'all GRPO groups had equal rewards in steps 1-2 (no gradient to test)', 'WARN')
                target = Path(trainer.cfg.trainer.export_path).parent / 'canary.json'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps({'optimizer_steps': 2, 'finite_gradients': True, 'max_grad_norm': self.max_grad,
                                              'learning_signal_steps_1_2': signal, 'gradient_path_exercised': self.max_grad > 0,
                                              'env_failure_fraction_steps_1_2': worst}, indent=2))
            try:   # watchdog heartbeat: on_log fires only after this step's eval, which can take minutes for agentic envs
                progress_path = Path(trainer.cfg.trainer.export_path).parent / 'progress.json'
                previous = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                progress_path.write_text(json.dumps(dict(previous, status='running', optimizer_step=step, updated_at=time.time(),
                                                         phase='step_end'), sort_keys=True, indent=2) + '\n', encoding='utf-8')
            except Exception as error:
                print(f'progress write skipped: {type(error).__name__}: {error}', flush=True)
            if time.monotonic() - self.last_checkpoint_at >= self.checkpoint_seconds:
                control.should_save = True
            deadline = float(os.environ['GRPO_TRAIN_DEADLINE'])
            if time.time() >= deadline:
                trainer.cfg.trainer.max_training_steps = step
                control.should_save = True
                control.should_evaluate = True

        def on_log(self, trainer, callback_input, control) -> None:
            logs = callback_input.logs or {}
            logs['optimizer_step'] = int(callback_input.global_step)
            # this attempt only (the program ledger lives in S3): nodes x GPU fraction of a node x worker wall-clock
            logs['budget/consumed_node_hours'] = float(os.environ.get('GRPO_NODES', '1')) * float(os.environ.get('STRL_GPU_FRACTION', '1')) * (
                time.time() - float(os.environ['GRPO_WORKER_BEGAN_EPOCH'])) / 3600.0
            logs['budget/final_eval_reserve_seconds'] = float(os.environ.get('GRPO_FINAL_EVAL_RESERVE_SECONDS', '0'))
            logs['checkpoint/age_seconds'] = time.monotonic() - self.last_checkpoint_at
            rewards = logs.get('reward/avg_raw_reward')
            if rewards is not None:
                logs.setdefault('train/reward', rewards)
            failure = env_failure_fraction(logs, trainer.cfg.trainer.train_batch_size)
            if failure is not None:   # one key for every environment (Harbor reports masked instances instead)
                logs.setdefault('environment/masked_group_fraction', failure)
            aliases = {
                'policy/policy_loss': 'train/policy_loss', 'policy/policy_entropy': 'train/entropy',
                'policy/grad_norm': 'train/gradient_norm', 'policy/loss_metrics/clip_ratio': 'train/clipping_fraction',
                'policy/policy_kl': 'train/kl', 'policy/policy_lr': 'train/learning_rate',
            }
            for source, target in aliases.items():
                if source in logs:
                    logs.setdefault(target, logs[source])
            try:  # drain finished critic analyses onto this row (critic runs on its own thread)
                from .monitor.critic import drain
                scalars, rows = drain()
                logs.update(scalars)
                if rows:
                    trainer.tracker.log_samples_to_table('qualitative/critic', rows[0].keys(), [list(r.values()) for r in rows],
                                                         int(callback_input.global_step))
            except Exception as error:
                print(f'critic drain skipped: {type(error).__name__}: {error}', flush=True)
            metrics_path = Path(trainer.cfg.trainer.export_path).parent / 'metrics.jsonl'
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            if not os.environ.get('STRL_METRICS_JSONL'):   # otherwise AxisTracking.log mirrors every row (incl. these)
                scalar_logs = {key: value for key, value in logs.items() if isinstance(value, (bool, Real))}
                with metrics_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps({key: float(value) if isinstance(value, Real) and not isinstance(value, (bool, int)) else value
                                             for key, value in scalar_logs.items()}, sort_keys=True) + '\n')
            progress_path = metrics_path.parent / 'progress.json'
            progress_path.write_text(json.dumps({
                'status': 'running', 'optimizer_step': int(callback_input.global_step),
                'budget_consumed_node_hours': logs['budget/consumed_node_hours'], 'updated_at': time.time(),
                'reward': logs.get('reward/avg_raw_reward'), 'step_seconds': logs.get('timing/step'),
            }, sort_keys=True, indent=2) + '\n', encoding='utf-8')

        def on_eval_end(self, trainer, callback_input, control) -> None:
            step = int(callback_input.global_step)
            dump_dir = Path(trainer.cfg.trainer.export_path) / 'dumped_evals' / f'global_step_{step}_evals'
            complete: list[dict[str, Any]] = []
            for path in sorted(dump_dir.glob('*.jsonl')):
                if path.name == 'aggregated_results.jsonl':
                    continue
                for line in path.read_text(encoding='utf-8').splitlines():
                    complete.append(self.env.eval_record(json.loads(line), trainer, step))
            complete = self.env.collapse_eval_records(complete)   # step-wise Harbor dumps one row per agent turn
            expected = int(os.environ['GRPO_EXPECTED_VALIDATION_SIZE'])
            if len(complete) != expected or len({r['sample_id'] for r in complete}) != expected:
                raise RuntimeError('Validation evidence is incomplete or sample identities are duplicated')
            complete.sort(key=lambda item: str(item.get('sample_id')))
            correct = sum(bool(item['correct']) for item in complete)
            accuracy = correct / len(complete) if complete else 0.0
            if self.baseline_accuracy is None:
                self.baseline_accuracy = accuracy
            metric = self.env.metric if self.env is not None else 'accuracy'
            payload = {
                f'eval/{metric}': accuracy,
                'eval/task_metric': accuracy,
                'eval/delta_from_baseline': accuracy - self.baseline_accuracy,
                'eval/invalid_answer_rate': sum(item['parsed_answer'] is None for item in complete) / len(complete) if complete else 0.0,
                'eval/truncation_rate': sum(item['finish_reason'] != 'stop' for item in complete) / len(complete) if complete else 0.0,
            }
            trainer.tracker.log(payload, step=step, commit=False)
            columns = self.eval_columns + tuple(k for k in (complete[0] if complete else {}) if k not in self.eval_columns and k != 'extra')
            rows = [[item.get(key) for key in columns] for item in complete[:int(os.environ.get('GRPO_FIXED_QUALITATIVE', '16'))]]
            if rows:
                trainer.tracker.log_samples_to_table(key='qualitative/fixed_validation', columns=list(columns), samples=rows, step=step)
                trainer.tracker.log({'qualitative_rows': len(rows)}, step=step, commit=False)
            output = Path(trainer.cfg.trainer.export_path) / 'evaluation_samples.jsonl'
            with output.open('a', encoding='utf-8') as handle:
                for item in complete:
                    handle.write(json.dumps(item, sort_keys=True, default=str) + '\n')
            timing = callback_input.metrics or {}
            with (Path(trainer.cfg.trainer.export_path).parent / 'evaluation_timing.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'optimizer_step': step, 'examples': len(complete),
                                         'eval_generate_seconds': float(timing.get('timing/eval_generate', 0.0))}, sort_keys=True) + '\n')

        def on_save(self, trainer, callback_input, control) -> None:
            # S3 upload is done by the worker's background sync, never inline in the training loop.
            self.last_checkpoint_at = time.monotonic()

        def on_train_end(self, trainer, callback_input, control):
            try:
                from .monitor.critic import shutdown
                shutdown(timeout=60)
            except Exception:
                pass
            root = Path(trainer.cfg.trainer.export_path).parent
            (root / 'training_completed.json').write_text(json.dumps({'optimizer_steps': int(callback_input.global_step),
                                                                       'completed_at': time.time()}, indent=2))

    return EvidenceCallback
