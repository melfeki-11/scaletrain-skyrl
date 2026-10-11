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
| `qualitative/fixed_by_training`, `qualitative/broken_by_training` | end of run: base wrong -> trained right (and the reverse) on the held-out test |
| artifact `<attempt>-evidence` | result.json, base/candidate jsonl, fixed/broken examples |

Tables are INCREMENTAL (only new rows upload each time). The critic never blocks training: bounded queue, own thread, hard USD cap
(`critic/cost_usd`, `critic/disabled`).

## Live monitor and alerts

```bash
.venv/bin/python -m strl monitor experiments/<x>.yaml --attempt <attempt> --job-id <job_id>          # one line per minute
```

It joins ScaleTrain state, S3 `progress.json`/`heartbeat.json`/`completion.json` and W&B history, and flags `STALL`, `NAN`,
`ENV_FAILURES`, `BUDGET` (plus the per-domain and multi-node flags below). The worker independently sends W&B alerts on non-finite metrics, canary failure, stalls (watchdog
kills the child, then still evaluates the latest export) and failures. S3 is the source of truth; W&B is the evidence surface.

## After the run

```bash
.venv/bin/python -m strl verify experiments/<x>.yaml --attempt <attempt>                 # independent re-scoring
.venv/bin/python -m strl compare experiments/a.yaml:<attA> experiments/b.yaml:<attB>    # paired, same held-out ids
```

## Per-domain environments (mimo): keys, flags, in-loop result

Per-domain scalars (one set per selected domain `<d>` in music, cyber, code, general, webdev): `eval/<d>/{task_metric,pass_rate,num_trials,num_masked,
judge_variance,rep_std}`, `eval/macro_task_metric` (= `eval/task_metric`), `generate/<d>/{pass_rate,reward_mean,reward_native_mean,tokens,avg_turns,num_trials,num_masked,num_timeout,num_output_exceeded,num_context_exceeded,
judge_cost_usd}`, `environment/<d>/{masked,timeout,dropped}_group_fraction` (infrastructure masks / groups with a timed-out trial / groups lost to any mask), `eval/<d>/no_artifact_rate`, `test/<d>/{base,trained,delta,ci_lower,ci_upper}` (final), `test/macro/*`, `test/dropped`.
GPU memory: SkyRL's Ray monitor logs `ray/gpu.mem_used_gb.*`; when `budget.failfast.mem_gb` is on, the head also logs `gpu/mem_peak_gb` and
`gpu/mem_used_gb_max` (nvidia-smi sampled by a daemon thread, head node only, absent when not measurable).

`strl monitor` prints `music 0.42/24% cyber ...` (held-out task metric / training pass rate) and three more flags. Each alerts once per run
through `tracking.alert` (W&B alert on the attempt's run; honours `tracking.alerts`):

| Flag | Fires when | Why it matters |
| --- | --- | --- |
| `DOMAIN_MASKED(<d>=x)` | more than 15% of a domain's GRPO groups were masked (infra or judge failure; timeouts are NOT counted here) | that domain trains on fewer prompts than planned; the environment, not the model, is failing |
| `TIMEOUTS(<d>=x)` | more than 30% of a domain's groups contain a trial that ran out of time | a timed-out trial that left tokens trains on the verifier's reward for its partial work; one that left none (single-shot music, some webdev) stays in its group as a zero-loss placeholder; the policy is too slow or too long for `max_agent_sec` / `max_turns`: raise the cap or lower the turn limit. Informational: it never stops a run (a separate rule stops it when >= 90% of groups are dropped) |
| `STEP_SLOW(Ns > 1.5 x plan Ms)` | `timing/step` above 1.5 x the plan's step estimate (`plan_step_seconds` in `evidence/<attempt>/stage.json` and the pod's `runtime.json`) | the run will not reach its planned steps; silent for runs staged before the estimate existed |
| `MEM(<key>=x GB > 76)` | any logged GPU memory metric above 76 GB | OOM is close (an H100 has 80 GB) |

Multi-node: `strl monitor` adds a per-node line `nodes=head:<age>s,rank1:<age>s` (heartbeat age per node) and flags `NODE_LOST`
(a peer heartbeat older than the stale window; the head aborts the run itself), `PEER_STALE` (the head's record went stale for a peer) and
`NETWORK` (the probe found socket transport, i.e. no EFA, or cross-node all-gather below plan). These also alert once. The probe's
`network.json` is logged once to W&B as `infra/*`: `infra/nodes`, `infra/efa_available`, `infra/nccl_socket_transport`, `infra/probe_passed`,
`infra/{intra_node,cross_node,global}_busbw_GBps` (single node: absent). Measured on p5 pairs: socket NCCL, ~0.69 GB/s cross-node; fine for LoRA (adapter-only
traffic), too slow for full fine-tuning of large models without HSDP (+ offload when memory requires it).

Monitor alerts go to a separate `<run_id>-monitor` W&B run (same group) so the monitor never resumes or corrupts the worker's run.

## Reward design, judge noise and flips (mimo and every Harbor env)

| Key / table | What it tells you |
| --- | --- |
| `generate/<d>/reward_terms/<fn>` | mean of each shaping term per domain per step (training only; the held-out eval always uses the native verifier) |
| `environment/reward_components/<fn>` | the same terms pooled over domains |
| `eval/<d>/judge_variance` | mean variance of the judge's votes on the same artifact (rubric and vision items): the noise floor a domain's delta must beat |
| `qualitative/fixed_by_training` / `qualitative/broken_by_training` tables (W&B + `artifacts/final_eval/*.jsonl`) | held-out tasks the trained policy newly solves / newly fails, joined by sample id with base and trained artifacts |
| `retention/<task>/{base,trained,delta,ci_lower,ci_upper}` | general-capability retention (e.g. a GSM8K subset), base vs trained, run after training (`evaluation.retention: {experiment: base/gsm8k.yaml, limit: 200}`); `artifacts/final_eval/retention/result.json` |
| `gates.ceiling` in `attempts/<dry attempt>/dry-run.json` (a dry run with `--set run.ceiling_per_domain=2`) or `strl eval-llm --env mimo` (needs a LiteLLM key locally) | a frontier LLM on the same held-out tasks and harness: the reachable score, and a harness bug check (ceiling 0 = the task or grader is broken, not the policy) |

Reading a reward A/B: shaping terms that rise while `eval/<d>/task_metric` and `test/<d>/delta` stay flat are being gamed; a delta inside
`judge_variance` or inside the paired CI is noise.

### In-loop paired result (`evaluation.final_eval: false`)

Agents cannot run in the offline evaluator, so the held-out evaluation happens inside training and the worker pairs it after training,
before `completion.json` (`strl/inloop.py`): base = the step-0 in-loop eval, trained = the LAST in-loop eval, run by the same engines, agent
and template. There is no checkpoint selection: `selected_step` is the last step and `result.json` says so (`selection`). If the run
stopped before a second eval exists (or step 0 is missing) the result records `paired: null` and a `reason`; a pair is never fabricated.
Files in `artifacts/final_eval/` (names unchanged, so `verify` / `compare` read them): `base.jsonl`, `candidate.jsonl`, `result.json`
(`test`, `by_source`, `flips`, plus `domains`, `macro`, `dropped`, `masked_rates`, `eval_repeats`, `base_step`, `trained_step`), `README.md`. The
environment computes the statistics (`EnvironmentSpec.paired_result`; default: paired bootstrap on `correct`; mimo: `honest_power` per domain
and macro, resampling tasks, masked/missing trial ids DROPPED and counted per domain, never zeros). A one-screen table goes to the worker stdout, the
`test/*` keys to the run summary, the evidence files to a W&B artifact, and the whole result into `completion.json` as `paired_result`.

### verify, compare, report for per-domain runs

```bash
.venv/bin/python -m strl verify experiments/<x>.yaml --attempt <attempt>    # evidence/<attempt>/verified-result.json
.venv/bin/python -m strl compare experiments/a.yaml:<attA> experiments/b.yaml:<attB>   # per-domain tables + trained-vs-trained deltas
.venv/bin/python -m strl report experiments/a.yaml:<attA> experiments/b.yaml:<attB> [--no-wandb]   # evidence/results/<group>-report.md
```

`verify` recomputes the per-domain and macro statistics from the raw per-trial records with a second bootstrap implementation (point
estimates must match exactly, CIs within Monte-Carlo tolerance), checks coverage against `final_eval_ids` (task x rep: missing, unexpected and
duplicated ids), and re-grades what persisted artifacts allow offline: music (the vendored scorer on the saved ABC; skipped when
`graders.music` or `abc2midi` is missing), general (the LLM judge on up to 8 persisted rubric requests, scored twice for variance; skipped
without a LiteLLM key via `strl.services.litellm` resolution as is), webdev / cyber / code reported as skipped (screenshots and verifiers
live in the sandbox). `verified: true` only when the statistics agree, coverage is complete and no re-graded record disagrees. The judge
request is persisted in the eval records (`extra.judge_request`, eval phase, rubric kind only) for exactly this check.
`compare` prints per run: domain, base, trained, delta [CI], n tasks, dropped, plus macro; run-vs-run tables use trained-vs-trained
`honest_power`. Single-metric environments keep their tables. `report` also pushes a summary table, the markdown (artifact) and, when
`wandb_workspaces` is importable, a W&B Report to a small `report` run in the same group; W&B failures never fail the CLI.

### Fail-fast and the predictive deadline

`budget.failfast: {mem_gb: 76, domain_masked: 0.15, step_slow_factor: 1.5, min_steps: 1}` raises a `RuntimeError` (after a W&B alert, details in
`failfast.json`) at a step end when: (a) peak GPU memory at the first measured step exceeds `mem_gb` (nvidia-smi on the head; unmeasurable =
never fires), (b) a domain's masked-group fraction exceeds `domain_masked` for 2 consecutive steps, (c) the step time (generation + training)
exceeds `step_slow_factor` x the plan estimate for 2 consecutive steps; (c) only alerts (no stop) when the plan estimate was UNCALIBRATED
(another model/environment, or any multi-node shape) and never fires after the last step. Canary runs skip (b) and (c) (a different, smaller infrastructure).
Nothing is rendered unless the environment opts in (`EnvironmentSpec.failfast_defaults`; mimo does) or the experiment sets the block; `null`
disables one rule. The deadline is predictive: after each step (from the second on) the run stops, forcing ONE final eval + checkpoint, when
`now + 1.15 x mean(last 2 step times) + eval_estimate + 120 s >= GRPO_TRAIN_DEADLINE`; `eval_estimate` is the last measured eval wall time
(`evaluation_timing.jsonl` `eval_seconds`; before any training step the step-0 eval). `budget/stop_reason` (`predictive_deadline`, `deadline`) is
logged and recorded in `training_completed.json`.
