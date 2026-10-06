---
name: scaletrain-rl
description: Train, plan, stage, launch, monitor, verify and compare RL post-training of open-weight LLMs on ScaleTrain with SkyRL GRPO, W&B (quantitative + qualitative), Modal-executed rewards, Harbor agentic sandboxes and a LiteLLM critic. Use to switch the model, the RL environment (GSM8K, coding single-turn, coding agentic) or the reward design and see the effect, or to add new models/environments/rewards. Self-contained; run from this repo.
---

# ScaleTrain RL experiment platform (strl)

Everything is driven by ONE experiment YAML and ONE CLI, run from the repo root with the local control venv:

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-local.txt   # once (~2 min)
.venv/bin/python -m unittest discover -s tests -t .                                                # unit, contract and golden tests (55)
.venv/bin/python -m strl plan experiments/mbpp-qwen35-partial.yaml --against experiments/mbpp-qwen35-binary.yaml
```

You change three axes; everything else (placement, batching, engines, credentials, sandboxes, budget, checkpoints, eval,
monitoring) is derived and validated:

| Axis | Change | Shipped |
| --- | --- | --- |
| model | `model.id` / `model.revision` (profile auto-detected) | `qwen3_5` (GPU-verified +18 pts GSM8K), `qwen3`, `llama`, `generic` |
| environment | `environment.name` | `gsm8k`, `coding_exec` (MBPP/HumanEval, tests in Modal), `coding_harbor` (Terminus-2 in Modal sandboxes) |
| reward | `reward.components` (weighted named terms or `module:function`) | coding: `pass_all_tests`, `partial_tests`, `compiles`, `no_suspicious_code`; gsm8k: `strict_answer`, `contains_number_answer`; common: `length_penalty`, `format_ok` |

The held-out evaluation always reports the environment's **task metric** (accuracy / pass@1), never the training reward, so reward
designs, models and environments are compared on the same yardstick. Read [decision-points.md](references/decision-points.md) for
every knob, its expected effect and the W&B key that shows it.

## Agent path: one experiment, end to end

```bash
cp experiments/mbpp-qwen35-binary.yaml experiments/my-exp.yaml      # edit name, tracking.group and the axis you study
.venv/bin/python -m strl plan experiments/my-exp.yaml                # free: errors, derived infra, estimates, warnings
.venv/bin/python -m strl fetch experiments/my-exp.yaml               # pinned model snapshot + data (coding: refs validated in Modal)
.venv/bin/python -m strl executor deploy                              # coding envs only: deploy + version-check the Modal executor
.venv/bin/python -m strl overlay                                     # once: Modal/Harbor packages for the training image
.venv/bin/python -m strl capacity --nodes 1 --gpus-per-node 8,4,1
.venv/bin/python -m strl stage experiments/my-exp.yaml --attempt my-exp-dry1 --mode dry && .venv/bin/python -m strl submit evidence/my-exp-dry1/job.json
.venv/bin/python -m strl stage experiments/my-exp.yaml --attempt my-exp-can1 --mode canary && .venv/bin/python -m strl submit evidence/my-exp-can1/job.json
.venv/bin/python -m strl stage experiments/my-exp.yaml --attempt my-exp-a1 --mode train && .venv/bin/python -m strl submit evidence/my-exp-a1/job.json
.venv/bin/python -m strl monitor experiments/my-exp.yaml --attempt my-exp-a1 --job-id <job_id>
.venv/bin/python -m strl ledger experiments/my-exp.yaml --attempt my-exp-a1 --job-id <job_id>
.venv/bin/python -m strl verify experiments/my-exp.yaml --attempt my-exp-a1
.venv/bin/python -m strl compare experiments/my-exp.yaml:my-exp-a1 experiments/mbpp-qwen35-binary.yaml:<other-attempt>
```

Rules: `stage` never submits; `submit` writes a receipt first and reconciles by name if anything is ambiguous (never resubmit);
always `dry` -> `canary` before a new model/environment/reward's first real run; record every terminal attempt in the ledger.

| Mode | Preset | What it proves | Cost |
| --- | --- | --- | --- |
| `dry` | m5.2xlarge | in the real image: payload, credentials (W&B, Modal, LiteLLM), real SkyRL parser + `validate_cfg` at the target GPU count, reward contract on good/bad inputs, executor/sandbox calls from the pod, W&B write+readback, a critic call | CPU only |
| `canary` | p5.48xlarge:1 | the experiment with `base/canary_overlay.yaml` (same model/env/reward, 1 GPU, 8 prompts x 4 samples, 2 updates, no final eval): engines, generator, rewards, every W&B table | `canary_max_seconds` + 600 s (1,800 s; Harbor 3,000 s) x 1/8 node |
| `train` | the experiment's preset | time-bounded GRPO, validation-only checkpoint selection, base vs trained held-out eval, paired bootstrap | per ledger |

## GSM8K vs RL environments

- **GSM8K** (`base/gsm8k.yaml`, `experiments/gsm8k-qwen35.yaml`): single-turn math, in-process strict scorer, no external services.
  Verified: Qwen3.5-0.8B, 36 updates, 50.80% -> 68.84% on the 1,319-example official test (+18.04, CI [+15.31, +20.77]).
- **coding_exec** (`base/coding_exec.yaml`): MBPP/HumanEval; the model writes one python block; tests run in the deployed Modal
  executor `strl-code-exec` (no network, limits, nonce-gated pass, candidate `__eq__` rejected); infra failures mask whole GRPO
  groups. 364 train / 96 validation / 663 held-out (499 MBPP-test + 164 HumanEval) after reference validation.
  Verified: Qwen3.5-0.8B on 4 GPUs, ~66 updates in ~48 min. Binary reward 23.38% -> 36.05% (+12.67 [+9.20, +16.29]);
  + 0.5 partial credit 33.33% (+9.95). The reward A/B and its W&B evidence are in RESULTS.md.
- **coding_harbor** (`base/coding_harbor.yaml`): the same tasks as Harbor task dirs; Terminus-2 operates a Modal sandbox and writes
  `/app/solution.py`; the sandbox verifier runs the same harness. Rewards come from the verifier (`pass_all_tests` only).
  One knob, `environment.harbor.max_seq_len` (default 8192), sizes the engine, the agent and the trainer: Terminus-2's first
  turn alone is 0.8-2.5k tokens. Canary-level evidence only (see validation.md).

## References (read what the task needs)

- [architecture.md](references/architecture.md) - data/control flow, package map, why each design choice
- [decision-points.md](references/decision-points.md) - every knob -> mechanism -> effect -> W&B key
- [extending.md](references/extending.md) - add a model, a reward, an environment (and the proof ladder)
- [launch-and-capacity.md](references/launch-and-capacity.md) - lifecycle, presets, capacity routes, budget ledger
- [monitoring.md](references/monitoring.md) - W&B quantitative/qualitative, critic, live monitor, alerts, verify/compare
- [credentials.md](references/credentials.md) - every credential locator and check (never paste secrets)
- [scaling.md](references/scaling.md) - bigger models, more nodes, fully-async agentic RL
- [troubleshooting.md](references/troubleshooting.md) - symptoms seen while building this, with fixes
- [validation.md](references/validation.md) - dated evidence for every gate and run

## Gotchas (each one cost a debugging cycle)

- SkyRL lets a dataset row's `env_class` override `environment.env_class`; GSM8K rows say `gsm8k`, so a custom reward would silently
  not apply. `strl.entrypoint.prepare_dataset` ignores the row column on the generic reward path.
- SkyRL silently drops prompts longer than `max_prompt_tokens`; `strl prepare` asserts none exceed it, and the entrypoint checks row
  counts. Held-out prompts are never filtered: the offline evaluator sizes its context from the longest test prompt.
- Executor anti-hack checks must not reject legitimate stdlib types: `collections.Counter.__eq__` is Python code; only
  candidate-defined `__eq__` (compiled as `<solution>`) is rejected. Reference validation in `strl prepare` catches regressions.
- Modal deploys roll gradually; a warm old container answers the version ping. `executor deploy` uses `strategy='recreate'` and
  polls until the new harness version answers; every result carries `harness_version`.
- The image skips `vllm-router`; `strl/runtime.py` loads the SkyRL-locked wheel (`assets/runtime`, SHA256 `4a1fa0bc...`).
- Qwen3.5 needs `language_model_only`, no sample packing, and a wrapped export for vLLM reload (profile `qwen3_5`).
- `transformers` 5 `apply_chat_template(tokenize=True)` returns a dict: count `input_ids`, not keys.
- Pinned SkyRL forwards `MODAL_TOKEN_*` into Ray if set: only `MODAL_CONFIG_PATH/PROFILE/ENVIRONMENT` are forwarded.
- `ml-admin` cannot write `s3://scale-ml`; staging uses `production-developer`. LiteLLM on a laptop needs ambient creds to assume
  `ml-worker` (`eval "$(aws configure export-credentials --profile ml-admin --format env)"`).
- A slice's wall-clock deadline costs `gpus/8` node-seconds; `stage` converts the ledger's node-seconds accordingly.
- A canary that "completes" can still be hollow. `can-hb-b` ran 2 updates while every Harbor trial died on turn 1, because the
  context was too small. The step-2 canary rule now fails when >= 50% of GRPO groups are masked by environment failures.
  Still read the canary yourself: `environment/masked_group_fraction`, `generate/num_error_trajectories`, and for Harbor
  `harbor-trials/*/exception.txt`.
- Judge reward designs only on `eval/task_metric` and the held-out test. In the A/B, partial credit raised the learning
  signal and the training reward, but not pass@1. Validation is 96 tasks (about ±5 points), so close arms need a larger
  validation split or more seeds.
- Custom `module:function` rewards must live in `user_rewards/` (or another top-level package of this repo) to ship to the pod.
