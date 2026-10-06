from __future__ import annotations

import hashlib
import json
import math
import os
import time
from numbers import Real
from pathlib import Path
from typing import Any

from .answers import parse_final_answer


def rollout_diagnostics(
    completions: list[str], rewards: list[float], stop_reasons: list[str | None], group_size: int
) -> dict[str, float]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    groups = [rewards[index:index + group_size] for index in range(0, len(rewards), group_size)]
    variances = []
    for group in groups:
        mean = sum(group) / len(group)
        variances.append(sum((value - mean) ** 2 for value in group) / len(group))
    return {
        "train/within_group_reward_variance": sum(variances) / len(variances) if variances else 0.0,
        "train/learning_signal_fraction": sum(value > 0.0 for value in variances) / len(variances) if variances else 0.0,
        "train/invalid_answer_rate": (
            sum(parse_final_answer(value) is None for value in completions) / len(completions) if completions else 0.0
        ),
        "train/truncation_rate": (
            sum(value != "stop" for value in stop_reasons) / len(stop_reasons) if stop_reasons else 0.0
        ),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_manifest(path: str | Path, step: int) -> dict[str, Any]:
    root = Path(path)
    files = [item for item in sorted(root.rglob("*")) if item.is_file()]
    return {
        "step": step,
        "path": str(root),
        "files": [
            {"path": str(item.relative_to(root)), "bytes": item.stat().st_size, "sha256": file_sha256(item)}
            for item in files
        ],
    }


def make_tracking_class():
    from skyrl.train.utils.tracking import Tracking

    class AxisTracking(Tracking):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            if self.backend == "wandb":
                self.logger.define_metric("optimizer_step")
                for pattern in (
                    "train/*", "eval/*", "timing/*", "budget/*", "checkpoint/*",
                    "policy/*", "reward/*", "trainer/*", "generate/*", "gpu/*", "vllm/*",
                ):
                    self.logger.define_metric(pattern, step_metric="optimizer_step")

        def log(self, data: dict[str, Any], step: int, commit: bool = False):
            payload = dict(data)
            payload.setdefault("optimizer_step", int(step))
            if self.backend == "wandb":
                self.logger.log(data=payload, commit=commit)
            else:
                self.logger.log(data=payload, step=step)

        def log_samples_to_table(self, key, columns, samples, step):
            if self.backend != "wandb":
                return
            import wandb

            if not hasattr(self, "_sample_tables"):
                self._sample_tables = {}
            prior = self._sample_tables.get(key)
            table = wandb.Table(columns=columns, data=list(prior.data) if prior is not None else [])
            for row in samples:
                table.add_data(*row)
            self.logger.log({key: table, "optimizer_step": int(step)})
            self._sample_tables[key] = table

    return AxisTracking


def make_trajectory_logger_class():
    from skyrl.train.utils.trajectory_logging import TrajectoryLogger

    class GroupTrajectoryLogger(TrajectoryLogger):
        columns = (
            "step", "idx", "group_id", "rollout_index", "prompt", "completion",
            "parsed_answer", "reward", "token_count", "finish_reason",
        )

        def log(self, *, tracker, num_samples, prompts, generator_output, tokenizer, global_step, **kwargs):
            response_ids = generator_output.get("response_ids") or []
            rewards = generator_output.get("rewards") or []
            stop_reasons = generator_output.get("stop_reasons") or [None] * len(response_ids)
            scalar_rewards = [float(sum(value)) if isinstance(value, list) else float(value) for value in rewards]
            is_training = kwargs.get("wandb_key") == "trajectories/train"
            group_size = int(__import__("os").environ.get("GRPO_SAMPLES_PER_PROMPT", "8")) if is_training else 1
            decoded = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in response_ids]
            if is_training:
                tracker.log(
                    rollout_diagnostics(decoded, scalar_rewards, list(stop_reasons), group_size),
                    step=0 if global_step is None else int(global_step),
                    commit=False,
                )
            self._finish_reasons = list(stop_reasons)
            self._group_size = group_size
            super().log(
                tracker=tracker, num_samples=num_samples, prompts=prompts,
                generator_output=generator_output, tokenizer=tokenizer,
                global_step=global_step, **kwargs,
            )

        def select_sample_indices(self, *, num_samples: int, rewards: list[float], total: int) -> list[int]:
            return list(range(min(num_samples, total)))

        def build_samples(
            self,
            *,
            num_samples: int,
            prompts: list[Any],
            response_ids: list[list[int]],
            rewards: list[float],
            loss_masks: list[list[int]],
            num_turns_list: list[int],
            tokenizer: Any,
            **kwargs: Any,
        ) -> list[tuple[Any, ...]]:
            total = min(num_samples, len(response_ids), len(prompts), len(rewards))
            rows = []
            for index in range(total):
                reward = rewards[index]
                scalar = float(sum(reward)) if isinstance(reward, list) else float(reward)
                completion = tokenizer.decode(response_ids[index], skip_special_tokens=True)
                finish_reason = self._finish_reasons[index] if index < len(self._finish_reasons) else None
                rows.append(
                    (
                        index, index // self._group_size, index % self._group_size, str(prompts[index]), completion,
                        parse_final_answer(completion), scalar, len(response_ids[index]), finish_reason,
                    )
                )
            return rows

    return GroupTrajectoryLogger


def make_callback_class():
    from skyrl.train.utils.callbacks import TrainingCallback

    class EvidenceCallback(TrainingCallback):
        def __init__(self, checkpoint_minutes: float = 15.0):
            self.checkpoint_seconds = checkpoint_minutes * 60.0
            self.last_checkpoint_at = time.monotonic()
            self.started_at = self.last_checkpoint_at
            self.baseline_accuracy = None
            self.worker_started_at = float(os.environ.get("GRPO_WORKER_BEGAN_MONOTONIC", self.started_at))

        def on_step_end(self, trainer, callback_input, control) -> None:
            step = int(callback_input.global_step)
            metrics = callback_input.metrics or {}
            for key, value in metrics.items():
                if isinstance(value, Real) and not math.isfinite(float(value)):
                    raise RuntimeError(f'Nonfinite optimizer metric: {key}')
            if step == 2:
                # Lightweight canary: two real finite updates with a nonzero gradient. Parameter change is
                # verified on the final export, so no extra checkpoint is paid for inside the time budget.
                gradients = [float(v) for k, v in metrics.items() if 'grad_norm' in k]
                if not gradients or max(gradients) <= 0:
                    raise RuntimeError('Canary requires a real nonzero finite policy gradient')
                target = Path(trainer.cfg.trainer.export_path).parent / 'canary.json'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps({'optimizer_steps': 2, 'finite_gradients': True, 'max_grad_norm': max(gradients)}, indent=2))
            if time.monotonic() - self.last_checkpoint_at >= self.checkpoint_seconds:
                control.should_save = True
            deadline = float(os.environ['GRPO_TRAIN_DEADLINE'])
            if time.time() >= deadline:
                trainer.cfg.trainer.max_training_steps = step
                control.should_save = True
                control.should_evaluate = True

        def on_log(self, trainer, callback_input, control) -> None:
            logs = callback_input.logs or {}
            logs["optimizer_step"] = int(callback_input.global_step)
            logs["budget/consumed_node_hours"] = int(os.environ.get("GRPO_NODES","1")) * (float(os.environ.get('GRPO_PREVIOUS_SECONDS','0')) + time.time() - float(os.environ["GRPO_WORKER_BEGAN_EPOCH"])) / 3600.0
            logs["budget/final_eval_reserve_seconds"] = float(
                os.environ.get("GRPO_FINAL_EVAL_RESERVE_SECONDS", "0")
            )
            logs["checkpoint/age_seconds"] = time.monotonic() - self.last_checkpoint_at
            rewards = logs.get("reward/avg_raw_reward")
            if rewards is not None:
                logs.setdefault("train/reward", rewards)
            aliases = {
                "policy/policy_loss": "train/policy_loss",
                "policy/policy_entropy": "train/entropy",
                "policy/grad_norm": "train/gradient_norm",
                "policy/loss_metrics/clip_ratio": "train/clipping_fraction",
                "policy/policy_kl": "train/kl",
                "policy/policy_lr": "train/learning_rate",
            }
            for source, target in aliases.items():
                if source in logs:
                    logs.setdefault(target, logs[source])
            metrics_path = Path(trainer.cfg.trainer.export_path).parent / "metrics.jsonl"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            scalar_logs = {
                key: value
                for key, value in logs.items()
                if isinstance(value, (bool, Real))
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {key: float(value) if isinstance(value, Real) and not isinstance(value, (bool, int)) else value
                         for key, value in scalar_logs.items()},
                        sort_keys=True,
                    )
                    + "\n"
                )
            progress_path = metrics_path.parent / "progress.json"
            progress_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "optimizer_step": int(callback_input.global_step),
                        "budget_consumed_node_hours": logs["budget/consumed_node_hours"],
                        "updated_at": time.time(),
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        def on_eval_end(self, trainer, callback_input, control) -> None:
            step = int(callback_input.global_step)
            dump_dir = Path(trainer.cfg.trainer.export_path) / "dumped_evals" / f"global_step_{step}_evals"
            rows: list[list[Any]] = []
            complete: list[dict[str, Any]] = []
            for path in sorted(dump_dir.glob("*.jsonl")):
                if path.name == "aggregated_results.jsonl":
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    item = json.loads(line)
                    extras = item.get("env_extras") or {}
                    info = extras.get("extra_info") or {}
                    spec = extras.get("reward_spec") or {}
                    completion = item.get("output_response", "")
                    completion = trainer.tokenizer.decode(trainer.tokenizer.encode(completion, add_special_tokens=False), skip_special_tokens=True)
                    reward = item.get("score", 0.0)
                    reward = float(sum(reward)) if isinstance(reward, list) else float(reward)
                    parsed = parse_final_answer(completion)
                    ground_truth = spec.get("ground_truth") or info.get("ground_truth")
                    record = {
                        "step": step,
                        "sample_id": info.get("sample_id"),
                        "prompt": item.get("input_prompt"),
                        "completion": completion,
                        "ground_truth": ground_truth,
                        "parsed_answer": parsed,
                        "correct": bool(reward > 0.0),
                        "reward": reward,
                        "token_count": len(trainer.tokenizer.encode(completion, add_special_tokens=False)),
                        "finish_reason": item.get("stop_reason"),
                    }
                    complete.append(record)
            expected = int(os.environ['GRPO_EXPECTED_VALIDATION_SIZE'])
            if len(complete) != expected or len({r['sample_id'] for r in complete}) != expected:
                raise RuntimeError('Validation evidence is incomplete or sample identities are duplicated')
            complete.sort(key=lambda item: str(item.get("sample_id")))
            correct = sum(bool(item["correct"]) for item in complete)
            accuracy = correct / len(complete) if complete else 0.0
            if self.baseline_accuracy is None:
                baseline_file = (
                    Path(trainer.cfg.trainer.export_path)
                    / "dumped_evals/global_step_0_evals/aggregated_results.jsonl"
                )
                if step > 0 and baseline_file.is_file():
                    baseline_metrics = json.loads(baseline_file.read_text(encoding="utf-8").splitlines()[0])
                    self.baseline_accuracy = float(baseline_metrics["eval/all/avg_score"])
                else:
                    self.baseline_accuracy = accuracy
            trainer.tracker.log(
                {
                    "eval/accuracy": accuracy,
                    "eval/delta_from_baseline": accuracy - self.baseline_accuracy,
                    "eval/invalid_answer_rate": (
                        sum(item["parsed_answer"] is None for item in complete) / len(complete) if complete else 0.0
                    ),
                    "eval/truncation_rate": (
                        sum(item["finish_reason"] != "stop" for item in complete) / len(complete) if complete else 0.0
                    ),
                },
                step=step,
                commit=False,
            )
            for item in complete[:16]:
                rows.append([item[key] for key in (
                    "step", "sample_id", "prompt", "completion", "ground_truth", "parsed_answer",
                    "correct", "reward", "token_count", "finish_reason",
                )])
            if rows:
                trainer.tracker.log_samples_to_table(
                    key="qualitative/fixed_validation",
                    columns=[
                        "step", "sample_id", "prompt", "completion", "ground_truth", "parsed_answer",
                        "correct", "reward", "token_count", "finish_reason",
                    ],
                    samples=rows,
                    step=step,
                )
                trainer.tracker.log({"qualitative_rows": len(rows)}, step=step, commit=False)
            output = Path(trainer.cfg.trainer.export_path) / "evaluation_samples.jsonl"
            with output.open("a", encoding="utf-8") as handle:
                for item in complete:
                    handle.write(json.dumps(item, sort_keys=True) + "\n")
            timing_path = Path(trainer.cfg.trainer.export_path).parent / "evaluation_timing.jsonl"
            timing = callback_input.metrics or {}
            with timing_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "optimizer_step": step,
                            "examples": len(complete),
                            "eval_generate_seconds": float(timing.get("timing/eval_generate", 0.0)),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

        def on_save(self, trainer, callback_input, control) -> None:
            # S3 upload is done by the worker's background sync, never inline in the training loop.
            self.last_checkpoint_at = time.monotonic()

        def on_train_end(self, trainer, callback_input, control):
            root = Path(trainer.cfg.trainer.export_path).parent
            (root/'training_completed.json').write_text(json.dumps({'optimizer_steps': int(callback_input.global_step), 'completed_at': time.time()}, indent=2))

    return EvidenceCallback
