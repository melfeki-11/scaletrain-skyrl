# RL post-training of open-weight LLMs on ScaleTrain: GSM8K vs RL environments

One experiment platform (`strl`) for GRPO post-training with [SkyRL](https://github.com/NovaSky-AI/SkyRL) on ScaleTrain H100s, with:
- **W&B** for quantitative curves and qualitative tables;
- **Modal** for executed rewards and agent sandboxes;
- **Harbor** for agentic tasks;
- **LiteLLM** for a trajectory critic.

You change three axes in one YAML file; everything else is derived, validated and kept stable:

| Axis | Where | Shipped |
| --- | --- | --- |
| **model** | `model.id` + `model.revision` (profile auto-detected from `config.json`) | `qwen3_5` (GPU-verified), `qwen3` (canary-verified), `llama`, `generic` |
| **RL environment** | `environment.name` | `gsm8k`, `coding_exec` (MBPP/HumanEval, tests run in Modal), `coding_harbor` (Terminus-2 agent in Modal sandboxes) |
| **reward design** | `reward.components`: a weighted sum of named terms, or your own `module:function` | `pass_all_tests`, `partial_tests`, `compiles`, `no_suspicious_code`, `strict_answer`, `length_penalty`, `format_ok`, ... |

Held-out evaluation always reports the environment's **task metric** (accuracy or pass@1), never the training reward. That
keeps runs that change different axes comparable on one yardstick.

The agent-facing manual is the in-repo Claude Code skill [`.claude/skills/scaletrain-rl/SKILL.md`](.claude/skills/scaletrain-rl/SKILL.md).
Its [decision-points reference](.claude/skills/scaletrain-rl/references/decision-points.md) lists every knob, its mechanism, its
expected effect, and the W&B key that shows the effect.

## GSM8K vs RL environments

| | GSM8K (`experiments/gsm8k-qwen35.yaml`) | Coding RL environment (`experiments/mbpp-qwen35-*.yaml`) |
| --- | --- | --- |
| Task | single-turn math, `#### <number>` answer | single-turn python: one fenced code block per MBPP/HumanEval task |
| Reward | in-process strict scorer | the program runs against hidden tests in a deployed **Modal** executor (`strl-code-exec`): no network, rlimits, non-root, nonce-gated pass, candidate `__eq__` rejected |
| Infra failure handling | n/a | a Modal error masks the whole GRPO group (`environment/masked_group_fraction`), so it is never mislabelled as a wrong answer |
| Data | GSM8K train → seeded validation split; official 1,319-example test | MBPP train+validation (364 train / 96 validation after reference validation); held-out MBPP-test (499) + HumanEval (164) |
| Monitoring | SkyRL curves, `trajectories/*`, `qualitative/fixed_validation` | the same, plus `environment/reward_components/*`, `environment/failure/*` classes, `train/learning_signal_fraction`, `environment/exec_latency_s`, and a LiteLLM critic table `qualitative/critic` |
| External services | W&B only | W&B, Modal, LiteLLM (critic) |
| Agentic variant | - | `coding_harbor`: the same tasks as Harbor task dirs; Terminus-2 operates a Modal sandbox, and the verifier runs the same harness |

Both run through the same worker, budget ledger, watchdog, checkpoint selection (validation only), paired-bootstrap final
eval and independent re-scoring (`strl verify`).

## Results

Full tables, examples and job IDs are in [RESULTS.md](RESULTS.md). Raw evidence is under `evidence/`.

- **GSM8K, Qwen3.5-0.8B, one 8×H100 node, 36 GRPO updates:** official test accuracy went from 50.80% to **68.84%**
  (+18.04 points, paired 95% CI [+15.31, +20.77]), verified independently. The refactored code reproduces the original run's 76 SkyRL
  overrides, job command and scorer decisions exactly (golden tests). A 1-GPU canary on the new code also completed.
- **Coding RL environment, reward-design A/B. Same model, two concurrent half-node jobs, identical except the reward.**
  On 663 held-out MBPP-test + HumanEval tasks:
  - Binary `pass_all_tests` went from 23.38% to **36.05%** (+12.67, CI [+9.20, +16.29]).
  - `pass_all_tests + 0.5·partial_tests` went from 23.38% to **33.33%** (+9.95, CI [+6.33, +13.57]).
  - Partial vs binary: −2.71 [−6.18, +0.75], not separable.
  - Partial credit raised the learning signal (73% vs 53% of groups) and the training reward, but not the yardstick: the
    "reward up, metric flat" effect that `strl plan` warns about, visible in W&B.
  - 0 Modal infra errors. The LiteLLM critic cost about $0.012 per run.

## Quickstart

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-local.txt
.venv/bin/python -m unittest discover -s tests -t .                       # contract + golden tests, no network
.venv/bin/python -m strl plan experiments/gsm8k-qwen35.yaml                # validate + derived infra + estimates + warnings (free)
.venv/bin/python -m strl plan experiments/mbpp-qwen35-partial.yaml --against experiments/mbpp-qwen35-binary.yaml
```

Run one experiment end to end. `stage` never submits, and `submit` reconciles by job name instead of resubmitting.

```bash
E=experiments/mbpp-qwen35-binary.yaml
.venv/bin/python -m strl fetch $E                              # pinned model snapshot + data (coding: every reference validated in Modal)
.venv/bin/python -m strl executor deploy                       # coding only: deploy/verify the Modal executor
.venv/bin/python -m strl overlay                               # once: Modal/Harbor packages for the training image
.venv/bin/python -m strl capacity --gpus-per-node 8,4,1
.venv/bin/python -m strl stage $E --attempt my-dry1 --mode dry    && .venv/bin/python -m strl submit evidence/my-dry1/job.json
.venv/bin/python -m strl stage $E --attempt my-can1 --mode canary && .venv/bin/python -m strl submit evidence/my-can1/job.json
.venv/bin/python -m strl stage $E --attempt my-a1 --mode train    && .venv/bin/python -m strl submit evidence/my-a1/job.json
.venv/bin/python -m strl monitor $E --attempt my-a1 --job-id <job_id>
.venv/bin/python -m strl ledger $E --attempt my-a1 --job-id <job_id>
.venv/bin/python -m strl verify $E --attempt my-a1
.venv/bin/python -m strl compare $E:my-a1 experiments/mbpp-qwen35-partial.yaml:<attempt>
```

The modes:
- `dry` (CPU, `m5.2xlarge`) proves everything except the GPUs in the real image: credentials, the SkyRL parser and
  `validate_cfg` at the target GPU count, reward contracts, executor calls from the pod, W&B readback, and a critic call.
- `canary` (`p5.48xlarge:1`, at most 30 min) trains 2 real updates with the same model, environment and reward, then checks
  every W&B table.
- `train` is time-bounded by the budget ledger.

## Layout

| Path | Contents |
| --- | --- |
| `experiments/` | runnable experiments; each one `extends:` a base and changes only what it studies ([index](experiments/README.md)) |
| `base/` | `infra.yaml` (image, storage, team, budget envelope), `gsm8k.yaml`, `coding_exec.yaml`, `coding_harbor.yaml`, `canary_overlay.yaml` |
| `strl/experiment.py`, `derive.py`, `overrides.py` | schema + `extends` resolution + validation + `experiment_sha`; derived infra, estimates and warnings; SkyRL overrides |
| `strl/models/` | model profiles: wrap class, packing, chat-template flags, vLLM kwargs, export adapter |
| `strl/tasks/` | environments: `gsm8k/`, `coding/` (data, Modal executor + harness, rewards, group-masking generator, Harbor export, evaluator) |
| `strl/rewards.py`, `strl/envs.py` | the reward component registry and the generic environment that applies any reward design |
| `strl/worker.py`, `stage.py`, `entrypoint.py`, `train.py`, `runtime.py` | ScaleTrain worker (budget, credentials, watchdog, selection, final eval), staging, the SkyRL entrypoint |
| `strl/tracking.py`, `strl/monitor/` | W&B axis/tables/alerts, the LiteLLM critic thread, live monitor, cross-run compare |
| `strl/services/` | W&B / Modal / LiteLLM credential locators and smoke checks; no secret values anywhere in the repo |
| `tools/` | vendored ScaleTrain submit/capacity/lifecycle helpers, the overlay builder |
| `tests/` | unit, contract and golden-equivalence tests; `pinned/` holds the SkyRL source snapshots for static schema checks |
| `evidence/` | budget ledgers, gate results, the verified GSM8K result; per-attempt receipts are gitignored |

## Known pitfalls baked into the code

The skill's [troubleshooting reference](.claude/skills/scaletrain-rl/references/troubleshooting.md) has the full list. The ones most
likely to bite:
- **Silent reward bypass:** a dataset row's `env_class` overrides the configured environment in SkyRL, so a custom reward is silently
  skipped. The generic path ignores the row column.
- **Silent prompt drops:** SkyRL drops prompts longer than `max_prompt_tokens`. `strl prepare` asserts that none are dropped.
- **Stale Modal versions:** a Modal deploy rolls gradually, so the old version can keep answering. `strl executor deploy` recreates
  the app and polls until the new harness version answers.
- **Qwen3.5 specifics:** it needs `language_model_only`, no sample packing, and a wrapped export for the vLLM reload (the `qwen3_5`
  profile handles all three).
- **Missing router:** the image lacks `vllm-router`, so the hash-pinned wheel is loaded from a writable overlay.
