# Monitoring: quantitative, qualitative, alerts

One W&B run per attempt (stable id `<attempt>-<stamp>`, `resume=allow`), every metric on the `optimizer_step` axis (never W&B's
reserved `step=`, which silently drops rows after eval/resume events). Provenance in the run config: `strl/experiment`,
`strl/experiment_sha`, `strl/resolved` (the whole resolved experiment), `strl/environment`. Group = `tracking.group`, so an A/B
lands side by side.

## Quantitative (what to look at, in order)

| Question | Keys |
| --- | --- |
| Is it learning? | `eval/task_metric` (yardstick at every eval, base at step 0), `eval/delta_from_baseline`, `reward/avg_raw_reward` |
| Is there signal? | `train/learning_signal_fraction` (share of GRPO groups with reward variance), `train/within_group_reward_variance` |
| Is the reward honest? | `environment/reward_components/*` vs `eval/task_metric`; `train/suspicious_code_rate`; `critic/reward_hacking_attempt_frac` |
| Is the policy healthy? | `policy/policy_loss`, `policy/grad_norm`, `train/entropy`, `policy/policy_kl`, `train/truncation_rate` |
| Is the env healthy? | `environment/infra_error`, `environment/masked_group_fraction`, `environment/failure/*`, `environment/exec_latency_s`, `generate/num_error_trajectories` |
| Is it fast? | `timing/step`, `timing/generate`, `timing/policy_train`, `trainer/tokens_per_second_per_gpu`, `vllm/train/*`, `ray/gpu.*` |
| Budget | `budget/consumed_node_hours`, `checkpoint/age_seconds` |

## Qualitative

| Table | Rows |
| --- | --- |
| `trajectories/train` | the first 2 complete GRPO groups of every step: prompt, completion, parsed answer/code, reward, tokens, finish reason, failure class, tests passed/total |
| `qualitative/fixed_validation` | the same 16 validation ids at every eval (watch individual tasks flip) |
| `qualitative/critic` | every 5 steps, up to 8 failed rollouts classified by gpt-6-luna: `critic_mode`, rationale, fix hint, source step |
| `qualitative/fixed_by_training` | end of run: base wrong -> trained right on the held-out test |
| artifact `<attempt>-evidence` | result.json, base/candidate jsonl, fixed/broken examples |

Tables are INCREMENTAL (only new rows upload each time). The critic never blocks training: bounded queue, own thread, hard USD cap
(`critic/cost_usd`, `critic/disabled`).

## Live monitor and alerts

```bash
.venv/bin/python -m strl monitor experiments/<x>.yaml --attempt <attempt> --job-id <job_id>          # one line per minute
```

It joins ScaleTrain state, S3 `progress.json`/`heartbeat.json`/`completion.json` and W&B history, and flags `STALL`, `NAN`,
`ENV_FAILURES`, `BUDGET`. The worker independently sends W&B alerts on non-finite metrics, canary failure, stalls (watchdog
kills the child, then still evaluates the latest export) and failures. S3 is the source of truth; W&B is the evidence surface.

## After the run

```bash
.venv/bin/python -m strl verify experiments/<x>.yaml --attempt <attempt>                 # independent re-scoring
.venv/bin/python -m strl compare experiments/a.yaml:<attA> experiments/b.yaml:<attB>    # paired, same held-out ids
```
