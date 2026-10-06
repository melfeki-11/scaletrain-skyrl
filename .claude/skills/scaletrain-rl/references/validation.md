# Validation evidence (dated)

What has actually been run, with job IDs, so you know which paths are proven and which are only designed.
- S3 attempt root: `s3://scale-ml/tmp/mohamedelfeki/scaletrain/qwen35-grpo-v2/attempts/<attempt>/artifacts/`.
- W&B project: `mohamed-elfeki/scaletrain-rl-envs`. GSM8K a1 is in `qwen35-grpo-gsm8k`.
- Program budget: `evidence/ledgers/rl-envs.json`, 7,200 node-seconds, user-authorized on 2026-10-06.

## Proof status by path

| Path | Status |
| --- | --- |
| gsm8k · Qwen3.5-0.8B · 8 GPUs, full train + final eval | **GPU-verified** (2026-10-06, old code `2916fb9`; the new code is proven equivalent by golden tests, a dry run and a canary) |
| coding_exec · Qwen3.5-0.8B · 4-GPU slices, train + final eval | **GPU-verified**, reward A/B (see below) |
| coding_exec · Qwen3-1.7B (model switch) | canary |
| coding_harbor · Qwen3.5-0.8B (Terminus-2 in Modal sandboxes) | canary |
| llama / generic model profiles | CPU preflight only (`verified: unverified`; `strl plan` warns) |
| multi-node, fully-async Harbor, Megatron | designed and documented in [scaling.md](scaling.md); **not GPU-verified in this repo** |

## Gates (2026-10-06)

| # | Gate | Result |
| --- | --- | --- |
| 1 | Local tests: `.venv/bin/python -m unittest discover -s tests -t .` | 47 pass. Includes the GSM8K golden equivalence: 76 overrides, the job command, scorer decisions on 2×1,319 a1 completions, and the env-var contract |
| 2 | Coding data + executor | 364 train / 96 validation / 16 canary / 663 test. Every reference passes 3× in Modal and every stub fails; 5 tasks dropped (3 duplicates, 2 broken references). Load test: 512 concurrent calls in 2.68 s, p50 0.31 s, p95 0.39 s, 0 infra errors (`evidence/gates/executor-load-test.json`) |
| 3 | Harbor + LiteLLM | oracle 10/10, nop 0/10, about 52 s per trial (`evidence/gates/harbor-oracle-nop.json`). gpt-6-luna reference eval: single-turn 18/20 ($0.0037); Harbor Terminus-2 7/10. Critic: 5 calls, $0.0004 |
| 4 | CPU dry runs in the real image (m5.2xlarge) | all 5 passed: `dry-gsm8k-a` job_db2kiics49qg07jejfag, `dry-cx-bin-a` job_db2kiikf8c4g07jrocog, `dry-cx-part-a` job_db2kij4s49qg07jejfb0, `dry-cx-q3-a` job_db2kijcf8c4g07jrocp0, `dry-hb-a` job_db2kijsf8c4g07jrocpg |
| 5 | 1-GPU canaries (2 updates each) | see the canaries table below |
| 6 | Reward A/B, 2 concurrent `p5.48xlarge:4` jobs | see "Reward A/B" below |
| 7 | Independent verification | `strl verify` re-scores in Modal; `strl compare` gives paired CIs |
| 8 | Fresh-agent forward test, using only the repo and the skill | new experiment on all three axes, `module:function` reward, `strl plan` with `--against`; the gaps it found were fixed |

What the dry runs checked:
- Modal auth from the pod via the kept profile bundle.
- The executor version and 512 calls with 0 infra errors.
- LiteLLM via the worker role.
- W&B write and readback, including an INCREMENTAL table.
- SkyRL `validate_cfg` at the target GPU count.
- The reward contract on good and bad inputs (reference, no_code, exit_hack, stub).
- For Harbor: oracle 1, nop 0.

### Canaries

| Attempt | Job | Outcome |
| --- | --- | --- |
| `can-gsm8k-a` | job_db2kkiss49qg080seceg | COMPLETED. 2 updates, grad norm 5.95 / 5.26, eval 0.50 → 0.75 on the canary validation set, env_class `gsm8k` |
| `can-cx-bin-a` | job_db2kkjcf8c4g07jrocq0 | COMPLETED. 2 updates, grad norm 4.44 / 1.94; base greedy pass@1 0.06 on 16 tasks, sampled train pass 0.12–0.19; tables `trajectories/train`, `trajectories/eval`, `qualitative/fixed_validation` |
| `can-cx-q3-a` | job_db2kkjkf8c4g07jrocqg | FAILED by design rule: all 16 rollouts passed, so there was zero variance and zero gradient. Fix: the rule now fails only when there is signal but no gradient, and the canary batch grew to 8 prompts |
| `can-hb-a` | job_db2kle4f8c4g08flbdgg | FAILED: `KeyError 'prompt'`, because the Harbor task dirs went through `PromptDataset`. Fix: `EnvironmentSpec.build_dataset` → `HarborTaskDataset`, and dry runs now build the real datasets |
| `can-cx-q3-b` | job_db2kqlcs49qg080secfg | COMPLETED in 523 s. 2 updates, grad norm 4.31 / 4.57, learning signal 0.25, sampled pass rate 0.59 / 0.53, mean response 50–80 tokens, 0 infra errors. **Model switch proven** (dense `qwen3` profile, same env and reward) |
| `can-hb-b` | job_db2kqhsf8c4g07jrocr0 | <!-- CAN-HB-B --> |

### Reward A/B (Gate 6/7)

<!-- AB-VALIDATION -->
