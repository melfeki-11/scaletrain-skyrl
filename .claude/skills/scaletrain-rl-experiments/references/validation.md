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
| coding_harbor · Qwen3.5-0.8B (Terminus-2 in Modal sandboxes) | canary (`can-hb-f`, after 5 failed attempts whose fixes are listed below); no full training run yet |
| llama / generic model profiles | **GPU-verified with a training effect** (SmolLM2-1.7B, Llama architecture, full FT: `r3b-2n-a1-fb`, 2026-10-11) |
| multi-node (2 x 8) | **GPU-verified with a training effect**: full fine-tuning of a 1.7B on 2 x 8 (HSDP, socket NCCL), MATH 22.4% -> 31.8% (+9.4 [+5.6, +13.2]) on 500 held-out problems (2026-10-11); LoRA at 0.8B and 8B (infrastructure proofs); 27B on 2 x 8 up to the step-0 eval only |
| math_boxed (new env, round 3) | **GPU-verified with a training effect** (above), `strl verify` 0 disagreements on 1,000 completions |
| non-Qwen models | Phi-4-mini (`dense` profile, bundled remote code) through dry run and a GPU run, 2026-10-08 section |
| fully-async Harbor, Megatron, > 2 nodes, MoE on GPU | designed and documented in [scaling.md](scaling.md); **not GPU-verified in this repo** |

## Gates (2026-10-06)

| # | Gate | Result |
| --- | --- | --- |
| 1 | Local tests: `.venv/bin/python -m unittest discover -s tests -t .` | 47 pass at the time of the gate (55 at merge). Includes the GSM8K golden equivalence: 76 overrides, the job command, scorer decisions on 2×1,319 a1 completions, and the env-var contract |
| 2 | Coding data + executor | 364 train / 96 validation / 16 canary / 663 test. Every reference passes 3× in Modal and every stub fails; 5 tasks dropped (3 duplicates, 2 broken references). Load test: 512 concurrent calls in 2.68 s, p50 0.31 s, p95 0.39 s, 0 infra errors (`evidence/gates/executor-load-test.json`) |
| 3 | Harbor + LiteLLM | oracle 10/10, nop 0/10, about 52 s per trial (`evidence/gates/harbor-oracle-nop.json`). gpt-6-luna reference eval: single-turn 18/20 ($0.0037); Harbor Terminus-2 7/10. Critic: 5 calls, $0.0004 |
| 4 | CPU dry runs in the real image (m5.2xlarge) | all 6 passed. `dry-cx-custom-a` job_db2l47cf8c4g08flbdhg is the user-code reward: `user_rewards/` was shipped in the payload, then imported and scored on all 4 contract cases in the pod. The others: `dry-gsm8k-a` job_db2kiics49qg07jejfag, `dry-cx-bin-a` job_db2kiikf8c4g07jrocog, `dry-cx-part-a` job_db2kij4s49qg07jejfb0, `dry-cx-q3-a` job_db2kijcf8c4g07jrocp0, `dry-hb-a` job_db2kijsf8c4g07jrocpg |
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
| `can-hb-b` | job_db2kqhsf8c4g07jrocr0 | **FALSE PASS**: "completed" 2 updates, but all 224 trials raised `ContextLengthExceededError` on turn 1. The engine and agent context were 1,024 tokens, while Terminus-2's first turn is 801–2,511. SkyRL logged `generate/num_error_trajectories` 32/32, but no gate read it. Fixes: one `environment.harbor.max_seq_len` (default 8,192) for engine, agent and trainer, with plan rejecting values below the first-turn size; the step-2 canary now fails when ≥ 50% of groups are masked by environment failures; Harbor masked instances feed `environment/masked_group_fraction` |
| `can-hb-c` | job_db2lfcks49qg07jejfd0 | FAILED at the step-0 validation check: the agent loop now works (13/16 trials ran without exception, median 8 turns, 1 solved; 3 hit 8,192 tokens late and train with reward 0, as SkyRL intends). The benchmark relabel had collapsed eval `sample_id`s to `mbpp`. Fix: the task path travels as `env_extras.task_dir`; a test runs relabel and `eval_record` together |
| `can-hb-d` | job_db2llics49qg07jejfdg | FAILED at the same check. The real cause (also behind `can-hb-c`): step-wise Harbor eval dumps one row per agent **turn**, 107 rows for 16 tasks, and the reward sits on the last turn. `can-hb-b` never hit this because every trial died on turn 1. Fix: an `EnvironmentSpec.collapse_eval_records` hook (Harbor merges turns per trajectory), with a regression test on the real dump (`tests/fixtures/harbor_eval_step0_can-hb-d.jsonl`) |
| `can-hb-e` | job_db2lrukf8c4g07jrocug | Step-0 validation **passed** (16/16 trials ran without exception, 1 solved). Step 1 **completed** in 477 s: 32 trajectories, 6 solved, 243 step-wise sequences, generate 229 s, logprobs 71 s, policy 137 s. Then our watchdog killed the run during step 1's eval: the 600 s stall limit was below a healthy agentic step plus eval, and `progress.json` was only written after eval. Fixes: progress is written at every optimizer step; Harbor sets `budget.stall_seconds: 1800` and `canary_max_seconds: 2400`, and plan rejects a stall limit below 2 × agent_timeout + 300; the stage canary cap follows `canary_max_seconds`; plan is calibrated on this step (477 s, ~4 steps/hour at 32 trajectories/GPU) |
| `can-hb-f` | job_db2m7s4f8c4g07jrod00 | **COMPLETED: genuine Harbor pass** (1,958 s, 1 GPU). 2 updates with grad norm 1.00 / 0.85, learning signal 0.25 per trajectory, `gradient_path_exercised: true`, environment-failure fraction 0.0 (0 error trajectories, 0 masked groups). Avg 7.1 / 7.5 agent turns; 2–3 trajectories per step hit 8,192 tokens and trained with reward 0. Sampled pass 0.125 / 0.062; validation `eval/mbpp/pass_at_1` 0.062 on 16 tasks. Steps 487 / 371 s. Critic: 8 analyses, $0.002. 112 Modal sandbox trials, 101 without exception |

### Reward A/B (Gate 6/7)

Run on 2026-10-06 as two concurrent `p5.48xlarge:4` jobs in W&B group `mbpp-reward-ab`. The arms are identical except `reward.components`.

| Attempt | Job | Updates | Held-out pass@1 (663) | Δ [95% CI] | `strl verify` |
| --- | --- | --- | --- | --- | --- |
| `ab-binary-a1` | job_db2kshss49qg07jejfbg | 66 (selected 55) | 23.38% → 36.05% | +12.67 [+9.20, +16.29] | 0 disagreements |
| `ab-partial-a1` | job_db2ksicf8c4g08flbdh0 | 69 (selected 50) | 23.38% → 33.33% | +9.95 [+6.33, +13.57] | 0 disagreements |

- **Paired B − A:** −2.71 [−6.18, +0.75] (`evidence/results/compare-reward-ab.md`).
- **Learning signal:** mean 0.53 binary vs 0.73 partial.
- **Speed:** median step 27.7 / 27.3 s; final eval 117 s for 2 × 663 tasks.
- **Environment health:** 0 infra errors and 0 masked groups.
- **W&B:** both runs finished with all five tables (`trajectories/train`, `trajectories/eval`, `qualitative/fixed_validation`,
  `qualitative/critic` with 104 rows, `qualitative/fixed_by_training`), the reward-component scalars, the critic failure-mode
  shares and the evidence artifact (`evidence/results/wandb-readback-reward-ab.json`).
- **Found while it ran:** `budget/consumed_node_hours` over-reported, adding the program's unused budget. Fixed; the deadline
  itself was always correct.
- **Profiles:** both runs were added to `strl/profiles.json`. Coding plans are now calibrated: 29.7 s predicted vs 27.7 s measured.

## Multi-node, LoRA/full and the MiMo suite (2026-10-07)

Program ledger: `evidence/ledgers/mimo-27b.json`, 21,600 node-seconds (6 node-hours) authorized for Qwen3.8-27B on 2 x 8 H100. Status of each rung:

| Rung | Result |
| --- | --- |
| 27B weights staged to S3 by a CPU job (`strl stage-model`) | COMPLETED in ~12 min: 29 files, 55.59 GB, sha256 manifest. Per-shard download with retries (HF 429s killed an earlier 27B attempt) |
| 2-node CPU dry run, LoRA (`dry2-lora-a`, 2 x m5.2xlarge) | `dry_run_passed`: rendezvous via S3 leader record, 2 distinct hosts, Ray with 16 GPUs, `validate_cfg` accepted 82 overrides |
| 2-node CPU dry run, full fine-tuning (`dry2-full-a`) | `dry_run_passed`: 76 overrides at 16 GPUs |
| `efa_available` on the CPU pods | false, `ofi_plugins: []`: the image may lack aws-ofi-nccl, so cross-node NCCL may run over TCP sockets even on p5. The GPU smoke measures it; this is why LoRA + HSDP inside a node is the recommended 27B layout |
| Chat template, step-wise prefix merging (811 real Terminus-2 turns from `can-hb-f`, every logged prompt re-rendered exactly) | Qwen3.5: 811 sequences (no merging). Qwen3.8 with `enable_thinking=false`: 209 sequences, 86% of merge points merge (~3.9x fewer training sequences). `strl check-template` + a plan warning replace a hard-coded Qwen3.5 assumption |
| MiMo smoke selection + export (`strl prepare`, 5 domains x 4 train + 2 test) | works on real tasks; found 5 bugs unit tests could not (below) |
| Strict nop gate, all 5 domains, real Harbor/Modal path | nop = 0 AND the verifier proven to have run (`details.json`); scripted positive controls: music tune scores 0.228, general's anchor-built answer reaches the judge request with 5 items |
| `dry-mimo-a` (m5.2xlarge, pod) | `dry_run_passed`: nop from the pod in all 5 domains (7-36 s), judge selftest with gpt-6-luna: rubric call ok, **vision call ok** (gpt-6-luna handles images; $0.0006), SkyRL `validate_cfg` 89 overrides, W&B incremental table readback, critic call |
| 1-GPU canary, 5 domains (`can-mimo-d`, Qwen3.5-0.8B LoRA, borrowed 1 x H100) | COMPLETED, 2 optimizer steps in 2156 s: step-0 eval + step-2 eval in all 5 domains, per-domain `generate/*` and `eval/*` metrics, judge cost logged, critic taxonomies per domain, merge 329->329 (Qwen3.5 does not merge), GPU peak 65 GB, adapter exported. Two earlier canaries (`can-mimo-a/b`) never left the queue and were cancelled unbilled; `can-mimo-c` ran out of its 2400 s canary budget after one step because evals ran at every step (canary now evaluates at step 0 and 2 only) |
| 2-node GPU smoke, Qwen3.5-0.8B LoRA, music+cyber (`g4-mimo-a`, `g4-mimo-b`; 2 x p5.48xlarge) | `g4-mimo-a` waited 3.5 h for two nodes in one network domain, then was **preempted 70 s after starting** (borrowed capacity reclaimed). `g4-mimo-b` (research team's own quota) ran 24 min: rendezvous via the S3 leader record, 16 Ray GPUs, **16-rank NCCL all-reduce on every rank** (intra-node 344 GB/s, cross-node 0.69 GB/s over **NET/Socket**: no EFA plugin in the image although the pods expose 32 EFA devices), 8 vLLM engines behind one router reaching **both nodes' engines**, step-0 eval at 16 GPUs, **2 optimizer updates** with the **per-host LoRA sync across nodes in 6.4 s**, peak GPU memory 43 GB; then it hit the platform's own fail-fast (38 music timeouts at a 120 s cap read as an environment failure). Not exercised on 2 nodes: the adapter export and the final eval (the 1-node 27B canary exercised the export) |
| 27B CPU dry runs (`dry-27b-b`, `dry-run1-a`; 2 x m5.2xlarge) | `validate_cfg` at 16 GPUs with the real 27B config and 120 + 120 tasks, nop in all 5 domains with the verifier proven to have run (incl. the 4 terminal_bench tasks), rubric + vision judge, W&B readback. `dry-27b-a` failed before any artifact: 16 files x 16 parts of the 55.6 GB weights filled a 30 GiB CPU pod (diagnosed with 2 one-pod CPU jobs that write status to S3; sequential download 390 s) |
| 1-node 27B canary (`can-27b-a`, Qwen3.8-27B LoRA r64, 8 x H100, TP2 x 4 engines) | COMPLETED, 2 optimizer steps, 2838 s. **27B loads and trains in the strl image**: ~12 min to the first generation, step 599 / 818 s (generate 416 / 571, policy_train 101 / 63, sync 15, fwd 50 / 29), peak GPU memory 62 GB (limit 76), **adapter export 122 s**, **Qwen3.8 merges step-wise turns ~6x** (223 -> 36 sequences; 5.7-6.4x in the main run), music scored 0.36 mean at step 2. ~30 s per agentic turn: a 300 s cap times out every Terminus trial after ~9 turns, which is what sized the run's caps |
| 2 x 8 main run, Qwen3.8-27B LoRA (`run1-a`; 3 routes raced for 2 h: research+borrowing, foundation+borrowing, research without borrowing) | queued ~23 h on `waiting_for_network_domain_capacity`; the research+borrowing route started first (the other two were cancelled before running, not charged). The 27B loaded over **16 GPUs on 2 nodes** (~16 min), the router saw all 8 TP2 engines, and the **step-0 base evaluation of 160 trials completed (13 min)**; then a SIGTERM (borrowed capacity reclaimed) 30 min in, before step 1. Its metrics never reached S3/W&B. Charged 3,856 node-s (pods up only) |
| 1 x 8 main run, Qwen3.8-27B LoRA (`run1-n1-a`, the fallback after the 2-hour window) | COMPLETED: 3 optimizer steps (the predictive deadline stopped it), base 4.26% -> trained 6.38%, paired macro +2.11 [-2.68, +7.08] (not significant), cyber +9.4 points [0.0, 18.8]; `strl verify` verified (160 rows each side, independent re-score identical in point estimates, 28 music artifacts re-graded with 0 disagreements; the 8 judged answers were all 0.0, so that check is vacuous). 107 of 160 held-out trials timed out in the base and in the trained evaluation (general and webdev carry no signal at these caps); W&B readback: 282 metric keys, 15 artifacts. Step 1,397-1,504 s, eval 793 s, peak memory 70.8 GB, export 121 s |

### What each rung caught

| Rung | Bug |
| --- | --- |
| reading both sides of two tracks' interface | vision `judge_request` with an `items` entry raised "no judge score" -> every webdev trial would be masked; `force_zero` from the general grader was never consumed |
| `strl prepare` on real tasks | exporter wrote `dest/<split>/<name>` while the data layer passed the final dir: every task double-nested (`tasks/test/x/test/x/`) |
| `strl prepare` on real tasks | published code images define no `CMD`: sandboxes created without a command exit at once (status 128, `ConflictError: Sandbox is shutting down`), in capture code and in Harbor trials alike. Fixed with `CMD ["sleep","infinity"]`, tested on four Dockerfile variants |
| `strl prepare` on real tasks | non-ASCII / spaced workspace file names raised `UnicodeEncodeError` (HF paths must be percent-encoded) |
| gate tooling | `assets/overlay` (Linux pod build) shadowed the Mac venv's native packages: `No module named fastuuid.fastuuid` |
| the 1-GPU canary (`can-mimo-d`, real policy, real Harbor, real judge) | the held-out eval **dropped the tasks the policy failed**: an agent timeout leaves Harbor no rollout (masked `no_rollout`), and `OutputLengthExceededError` was retried then masked. Masked ids are dropped from base and trained, so a policy that rambles or runs out of time would look better (survivorship). Fixed: policy failures (agent/verifier timeout, context or output overflow) score 0 in evaluation and are never retried; in training a timed-out trial has no tokens: it stays in its group as a one-token placeholder (reward from the verifier, loss mask 0) instead of masking every sibling as SkyRL's stock `build_step_wise_generator_output` does, and is counted in `environment/<domain>/timeout_group_fraction`. `eval/<domain>/no_artifact_rate` added because `eval/invalid_answer_rate` is the all-domain "left no artifact" rate |
| the 27B data prepare + image prebuild | 4 of 5 non-env-0 `general` images could not build: terminal_bench tasks got the MCP-sidecar Dockerfile tail (`COPY general_env/...`) but ship no `general_env/`. Then the strict nop gate showed every terminal_bench trial would be masked as `infra_error`: their tests write only `reward.txt`. Fixed: plain Dockerfile with `CMD sleep infinity`; the shared entrypoint treats them as having run only with pytest's `ctrf.json` or the anti-hack guard's REJECT verdict |
| the 2-node smoke (`g4-mimo-a`) | the ledger charged `created_at -> terminal` (queue included): a job that waited 3.5 h would have cost more than the whole program. Now charged from the worker's `budget.json` `started_at` minus 180 s. A SIGTERM (preemption) is recorded as `preempted`, not `failed` |
| the 2-node smoke (`g4-mimo-b`) | timeouts counted as environment failures, and SkyRL masks the whole prompt group when ONE trial times out: at a 50% per-trial timeout rate ~94% of 4-sample groups vanish. Fixed: a timed-out trial stays in its group as a tokenless placeholder (reward in the baseline, loss mask 0); the fail-fast rules read infrastructure masks only; `TIMEOUTS` flag and `timeout_group_fraction` report it; a separate rule still aborts when >= 90% of groups are dropped |
| independent code review (not tests, not gates) | **music and general verifiers never ran** (`/usr/bin/python3` is absent from python:3.12-slim and general-agent-env-0): every trial scored 0 silently and the nop gate passed vacuously. Also: the agent could forge `reward.json` or shadow stdlib modules via `/tests`; artifacts never reached the verifier dir; `force_zero` raised KeyError; the soft stop deadline was before the child's hard kill; eval ran greedy so `eval_repeats: 2` were duplicates; step-deadline validation ignored the exporter's default timeouts |

Fixes for the review findings: one shared hardened `test.sh` (interpreter on PATH, wipes forged verifier files, prunes unexpected `/tests` files, copies artifacts, guarantees a non-empty `details.json`); a trial whose verifier left no `details.json` is masked as an infra error; the nop gate and the pod dry run assert the verifier ran; `strl ledger --never-started` for a job cancelled in the queue.

### Honest limits of this evidence
- Full fine-tuning at 27B on 2 nodes is validated only by the config checks above (dry run at 16 GPUs, memory model), never GPU-run. The measured cross-node link (NCCL over TCP sockets, 0.69 GB/s: the image has no aws-ofi-nccl although the pods expose 32 EFA devices) makes the full-shard layout over 16 GPUs slow by design; LoRA + HSDP keeps only the ~1 GB adapter gradient on that link.
- The 2 x 8 layout at 27B was observed only up to the step-0 evaluation (`run1-a`); 2-node updates, the LoRA sync across nodes (6.4 s) and the 16-rank collectives are verified at 0.8B (`g4-mimo-b`). The 27B updates, export and final evaluation were measured on 1 x 8.
- Known gaps from the independent review, not fixed: a `--mode dry` no longer proves the weight download (the 1-node canary did); the step-deadline validation checks one trial's worst case, not queueing behind `max_concurrency`; `outcome_status` -> `preempted` is unit-tested but has not yet been seen in a pod's `completion.json`; the verifier's `python3`/`grep` could be replaced by an agent running as root in the same container (Harbor's separate-verifier mode would close it); ledger entries recorded before 2026-10-08 charge from job creation, later ones from pod start.
- The main run's result is three updates: infrastructure and measurement evidence, not a training effect (macro CI contains zero).
- Borrowed GPU capacity was reclaimed twice within 70 s and 30 min. A preempted attempt does not resume; durable mid-run checkpoints with a resume path are not implemented.
- Cyber and code positive controls (an agent actually solving a task) and webdev's render-to-judge path with a real site need a model; they are exercised by the canary and the main run, not by a scripted control. The 27B did solve some code (10%) and cyber (3-12%) held-out tasks and scored webdev 2-5% through the vision judge, so those paths scored non-zero end to end.
- One code image was checked for future-commit leakage (`git rev-list --all` equals `HEAD`: 202 commits, no refs ahead); the other images, the cyber `submit.sh` surface, and judge prompt injection were not audited.


## Hardening round: capacity race, a non-Qwen model, multi-node re-verification (2026-10-08/09)

Program ledger: `evidence/ledgers/platform-hardening.json`, 14,400 node-seconds (4 node-hours), authorized by the user for 1-2 GPU canaries
(a non-Qwen LoRA model on a MiMo subset with a shaped reward, and a resume test) plus one 2 x 8 0.8B smoke.

### The capacity race (`strl launch`, every launch below)

Every launch raced fb / rb / ro / fo. All 13 races were won by `fb` (foundation + borrowing) within 4-19 s; CPU dry runs never left the queue on
the research team's quota (`rb`, `ro`). In every GPU race before the "stop submitting once a route started" change, `rb` was ALSO allocated
within the same poll and had to be cancelled and charged (46-484 node-s each); one such loser's pod wrote `budget.json` 89 s
after its job was CANCELED (`v1phi-rb`), which the first launcher version recorded as never started. Fixed: a loser ever seen allocated is charged
through a 180 s tail (`CANCEL_TAIL_SECONDS`). The `v1phi-rb` ledger entry predates the fix and is under-charged by ~64 node-s (left unedited: the
ledger is append-only; correcting it is the user's call).

### What each GPU attempt found

| Attempt | Shape | Outcome | Bug found -> fix |
| --- | --- | --- | --- |
| `v1dry-fb` | CPU | dry_run_passed; ceiling cyber 1.0 / music 0.0 (gpt-6-luna, 97.6 s) | - |
| `v1phi-fb` (Phi-4-mini LoRA, music + cyber) | 1 x 2 GPU | FAILED at policy init: `ImportError: cannot import name 'LossKwargs'` | SkyRL loads every model with `trust_remote_code=True`; Phi's `auto_map` runs a bundled modeling file written for transformers 4. The worker now uses the native class when transformers has it (`model_code: remote_code_dropped`), and `model_code_check` imports the class in every pod, so a dry run catches it for free |
| `v3smk-fb` (0.8B, music + cyber) | 2 x 8 | rendezvous, probe, step-0 eval, step 1 (350 s); then FAILED: `Fail-fast (domain_masked): 88% of music groups masked` | single-shot replies overflowing `max_generate_length` (`OutputLengthExceededError`, no tokens) were counted as infrastructure masks. Now a scored placeholder (reward 0, zero loss), `generate/<d>/num_output_exceeded`, never `masked` |
| `v1phid2-fb` | CPU | dry_run_passed with `model_code: remote_code_dropped, class Phi3ForCausalLM` | - |
| `v1phi2-fb` | 1 x 2 GPU | past policy init (the fix works); FAILED at step 2 on the same music overflow fail-fast (staged before that fix) | Phi-4-mini's ABC pieces overflow 2,048 tokens: the experiment now uses 4,096 |
| `v3smk2-fb` | 2 x 8 | both updates done (352 / 250 s; generate 222 / 221 s, policy_train 81 / 21 s; LoRA sync across nodes), no overflow masks; then FAILED after the LAST step: `Fail-fast (step_slow): 250s > 1.5 x the plan estimate 39s` | an UNCALIBRATED estimate (a 27B profile scaled down by size) stopped a healthy run at its last step. Only calibrated estimates stop a run now, no rule stops at the last step (advisory alert instead), and the estimator floors agentic generation at 0.35 x step deadline (212 s planned for this smoke) |
| `v1phid3-fb` | CPU | dry_run_passed; the new shaping contract ran the shipped term in the pod (`turn_efficiency_bonus`: pass 1.035, fail 0.0 in both domains) | - |
| `v1phi3-fb` | 1 x 2 GPU | step-0 eval (513 s, 4,096-token music), 2 updates (395 / 394 s; policy_train 14 s), no fail-fast; then FAILED saving the step-2 durable checkpoint: `No such file or directory ... "/workspace/strl-run/s3://..."` | pinned SkyRL writes every FSDP checkpoint file through its cloud io EXCEPT the LoRA adapter (`safetensors.save_file(path)`). Exact-source patch `LORA_ADAPTER_CLOUD_SAVE` (tested on the pinned source); the dry run now round-trips a safetensors file through SkyRL io at the real S3 `ckpt_path` (`durable_checkpoint_io` gate) |
| `v3smk3-fb` | 2 x 8 | **COMPLETED**: probe (socket NCCL, cross-node 0.68 GB/s, intra 343 GB/s), 2 updates, LoRA sync across hosts, export `global_step_2/policy` in S3, in-loop paired result (0.8B scores 0 on these tasks). `strl verify` verified (coverage 8/8, statistics agree, 6 music artifacts re-graded, 0 disagreements); W&B readback has `infra/*`, per-domain keys, flip and trajectory tables; `strl monitor --once` prints `nodes=head:311s,rank1:235s` | retention failed: the pod re-resolved `base/gsm8k.yaml`, which the payload does not ship. Fixed: the pod reads the staged `retention-resolved.json` |
| `v1phid4-fb` | CPU | dry_run_passed; `durable_checkpoint_io` round trip through SkyRL io at the real S3 `ckpt_path`; the LoRA save patch applied to the image's SkyRL | - |
| `v1phi4-fb` | 1 x 2 GPU | **COMPLETED**: 2 updates, in-loop paired result, durable checkpoint `global_step_2` in S3 (model + optimizer shards per rank, `data.pt`, `trainer_state.pt`, `lora_adapter/adapter_model.safetensors`); retention failed (staged before the fix above) | - |
| `v2res-fb` (`run.resume_from=v1phi4-fb`, `max_steps` 4) | 1 x 2 GPU | the worker found the checkpoint (`resumed_from: v1phi4-fb step 2`, carried its step-0 eval records and canary), SkyRL rendered `resume_mode: from_path`; then FAILED in `load_checkpoints`: `Checkpoint path not found: s3:/scale-ml/...` | pinned SkyRL wraps `resume_path` in `pathlib.Path`, collapsing `s3://`. Exact-source patch `RESUME_FROM_CLOUD_PATH` keeps it a string (the rest of the load path uses SkyRL's cloud io). The independent review's resume-prefix fix was already in this attempt |
| `v2res2-fb` (same) | 1 x 2 GPU | **COMPLETED: resume proven.** SkyRL loaded `v1phi4-fb`'s complete `global_step_2` from S3 (weights, optimizer, data position), trained steps 3-4 (~360 s each), saved its own durable `global_step_4`; the paired result keeps V1's step-0 eval as the base (`base_step` 0, `trained_step` 4), `strl verify` verified. Retention through the LoRA adapter served by vLLM: GSM8K 200 test problems 89.5% -> 90.0%, +0.5 [-2.0, +3.0] (no regression). W&B: `generate/<d>/reward_terms/user_rewards.examples:turn_efficiency_bonus`, `retention/gsm8k/*`. Won `fb` after 1 s with no other route submitted | - |

Spend: 12,215 of 14,400 node-seconds (3.39 node-hours), 87% of it before the resume run on the three 2 x 8 attempts and their charged losing routes.
Not exercised on GPU this round: `judge_variance` and judged domains (general, webdev), full fine-tuning, the 27B on 2 x 8 past step 0, > 2 nodes, MoE.

### Forward test (a fresh agent with only the repo and the skill, CPU only)

Prepared four experiments (Qwen2.5-7B LoRA and full on music + code, a custom token-cost shaping term with 9 unit tests, Qwen3.8-27B 2 x 8 on
cyber + general with the full launch command). It found 12 gaps; the code ones are fixed (`--set finetune.mode=full` refused an inherited LoRA
rank; `plan` silent on a missing ledger or unprepared data; BUDGET warning noise; the dry run never called mimo shaping terms; terms saw
`max_generate_length` 1024; the fallback name `-fb1` collided with route codes and could truncate to the original name; a relaunch name could
exceed 26 characters hours into a launch) and the documentation ones are in SKILL.md (lr by size, `result`/`ctx` fields, data dirs, large-model
staging, start-up times, A/B groups). Its experiment files were removed after the gaps were fixed.

### Independent code review of the branch (2026-10-09)

A separate reviewer read the whole `platform-hardening` diff and simulated the launcher with fakes. Fixed, each with a regression test:
- **resume never worked:** `prepare_resume` looked for the checkpoint under the NEW attempt's prefix (`run_name` passed for `previous`); every relaunch
  would have failed in the pod after its GPUs were allocated.
- durable checkpoints with the offline final eval (`final_eval: true`) failed every finished run (local checkpoint dir moved to S3): now rejected by
  `plan` until supported.
- launcher accounting: losers were charged their whole queue wait when their pod left no trace yet (now from the first allocated observation minus
  one poll); a loser seen running only while settling was recorded uncharged; a queued job that FAILED without a pod was charged; an S3 error could
  crown a route that never ran as the winner; ledger write failures vanished into `launch.json` (now retried, exit 3); a stage/submit error mid-race
  left submitted routes running unmanaged (now cancelled first); a preempted run with no relaunch left exited 0 (now 4); recording ignored `--set`
  overrides; concurrent launchers could lose ledger entries (file lock + atomic replace).
- a LoRA rank vLLM does not accept (e.g. 24) would crash the engines after the run had started: `plan` rejects it.

A second review of those fixes found six more gaps, all fixed with tests: an unsettled preempted job did not stop the relaunch nor fail the
launch; the `--fallback-set` race ran even when a loser could not be settled; unknown pod evidence on routes that all ended could poll forever;
"never started" was decided at the CANCELED observation although a pod's trace can arrive ~90 s later (now confirmed after the cancel tail, and a
"pod ran" refusal re-records the job as charged); an interrupt while waiting orphaned the queued routes; the manual `strl ledger` hint ignored
`--set` (now `strl ledger --set`). Also: a ledger entry spelled `CANCELLED` broke every later budget check.

Known limits, not fixed:
- coding_harbor hardening: the program under test runs as `python -I` (without `-S`, so a planted `.pth` in system site-packages still loads), and an
  agent running as root could replace `/usr/local/bin/python3` itself. Single-turn `coding_exec` runs in the separate Modal executor and is not affected.

## Verification round 2 (2026-10-09)

- Ledger: `v1phi-rb` corrected with the user's permission (+86 node-s: its pod ran 89 s after CANCELED); four losers whose "recorded" entries were
  lost to concurrent unlocked ledger writes before the lock existed (`v1phi-ro`, `v1phi-fo`, `v1phid2-ro`, `v1phid3-rb`, none with a pod trace)
  re-recorded uncharged through the locked path.
- Budget reservations: `stage` now books each launch's worst case under the ledger lock (a race's routes share one booking), so parallel launches
  cannot overspend a program; released on record or on a failed stage, expired after 72 h.

### Fresh-agent forward tests (no context but the repo and this skill), 2026-10-09

| Agent | Task | Real jobs | Outcome | What it changed in the repo/skill |
| --- | --- | --- | --- | --- |
| A | Qwen3-8B LoRA, 2 x 8, MiMo cyber + code; then a real 2 x 8 GPU proof inside a 9,000 node-s ledger | stage-model (16 GB, ~90 s), two 2-node CPU dry runs (one with the gpt-6-luna ceiling: cyber 1.0, code 0.5), the 2 x 8 GPU proof (below) | dry runs passed every gate (2 hosts, 16 Ray GPUs, 82 overrides, nop 0 with the verifier proven) | attempt-sizing arithmetic, BUDGET/CANARY plan warnings, reservation slack, plan step count with the in-loop reserve |
| B | SmolLM2-1.7B (Llama arch, ChatML) GSM8K reward A/B, one 1-GPU canary | 2 dry runs, canary `fb-smolb-can-fb` | canary COMPLETED: 2 updates, grad norm 0.056 / 0.066, learning signal 0.625, brevity term only paid on correct answers; 25-37% of eval completions fail on the `#### N` format | `fetch` top-level files only (19 GB of ONNX skipped), verify refuses canaries, derived env_class diff, canary semantics and gate documented |
| C | add a new environment (`countdown`) from extending.md | 2 dry runs | dry run passed (80 overrides, reward contract 7 cases, 512/100 rows); the env is now shipped | `score_result` declared, generic verify re-scoring, launch prints dry-run gates, extending.md single-turn walkthrough, `assets/` ignored |
| D | Qwen3-30B-A3B MoE full FT, 2 x 8 feasibility | stage-model (61 GB, ~5 min), 2-node dry run | dry run passed (HSDP 8 + offload, TP2); found the offline final eval hard-coded to TP1 (a 61 GB model would train then fail its eval) | TP-aware offline eval, measured probe by default, collective-cost estimate, FSDP buffers in memory, MoE section |
| E (second loop, after the fixes) | countdown + OLMo-2 1B LoRA 4 GPUs, own reward A/B; 2 x 8 full plan; launch + preemption plan | stage-model (~40 s), 2 dry runs | both dry runs passed; correct, budget-aware recommendations (advised against 2 x 8 full FT for a 1.5B model) | offload advice size-aware, eval shards a maximum, single-turn W&B keys, dry-run limits |

### 2 x 8 GPU proof launched from a fresh agent's preparation (2026-10-09)

`mimo-qwen3-8b-lora-2node-proof.yaml` (then `fa-qwen3-8b-cc-proof.yaml`), prepared, sized and dry-run by fresh agent A; launched by the
coordinator after the user confirmed (the agent's own GPU launch was blocked by its harness permission policy: a sub-agent cannot see the
user's approval). Attempt `fa-q3pf-a1-fb`, job_db4g8armjdjg07lcrs20, ledger `evidence/ledgers/fresh-agent-multinode.json`:
- won route `fb` after 1 s; no other route submitted (the stop-submitting change); reservation 6,240 node-s booked at stage, released at record.
- COMPLETED in 1,715 s: 2 nodes, socket NCCL (cross-node 0.63 GB/s, intra 344 GB/s), step-0 eval 229 s, steps ~300 s, LoRA sync across hosts,
  exports gathered from BOTH nodes (`exports_gathered.nodes[*].synced`), in-loop paired result (base step 0, trained step 2; 8B scores 0 on the
  4+4 capped held-out tasks: infrastructure evidence, not a training effect).
- `strl verify`: verified (coverage complete). W&B readback: `infra/*`, per-domain `generate/*` / `eval/*`, `qualitative/{fixed,broken}_by_training`,
  `qualitative/critic`, `trajectories/*`, `test/*`, `gpu/mem_peak_gb`, `timing/step`.
- Charged 3,634 node-s (agent's estimate: ~4,800; worst case 5,880); 5,366 left.

### 1-GPU proof of the new environment, the TP-aware sequential final eval and generic verify (2026-10-09)

`countdown-qwen35.yaml` with `--set budget.ledger=evidence/ledgers/platform-hardening.json --set algorithm.max_steps=8 --set
budget.attempt_max_seconds=2400 --set evaluation.interval_steps=4 --set evaluation.checkpoint_interval_steps=4`; attempt `cd-v1-fb`
(won `fb` immediately; 352 node-s booked, 197 charged): COMPLETED, 2 updates (the predictive deadline stopped it: steps ~240 s on 1 GPU),
LoRA adapter changed (186 `lora_B` tensors, served by vLLM in the eval), offline final eval on the SAME GPU, base then trained, 200
held-out problems in 351 s: 25.0% -> 25.0%, zero flips (two tiny updates). `strl verify` re-scored all 400 completions with countdown's
own scorer: verified.

### Independent review of this round's diff (2026-10-09)

Fixed with tests: a route rejected by ScaleTrain (or a submit error) kept its budget booking for 72 h; no manual release (`strl ledger
--release`); bookings expired by age while a launch still queued (now refreshed hourly while waiting or running); the retention eval placed
GPUs by the retention YAML's node size (a 2-GPU experiment at TP2 got `CUDA_VISIBLE_DEVICES=2,3`); a failed re-stage dropped an earlier valid
booking; countdown `prepare` could loop forever on narrow settings; countdown in-loop scoring ignored truncation; the default network probe is
labelled as assumed; plan warns `EVAL` when the final eval runs base then trained. Accepted: losers being settled are not booked (~one poll).

## Verification round 3: learning effects, shape racing, fresh agents (2026-10-11)

Ledger `evidence/ledgers/learning-proof.json` (user-authorized 2026-10-11: 10,800 node-s); 5,828 spent, 4,972 left. Every launch won route
`fb` within 1 s (2 x 8 included); no route or shape alternative was needed, but `--race-shape` was used on both train runs.

### Training effects (held-out, paired, `strl verify` 0 disagreements)

| Attempt | What | Held-out | Base -> trained | Δ [95% CI] | fixed / broken | Charged |
| --- | --- | --- | --- | --- | --- | --- |
| `r3b-2n-a1-fb` | SmolLM2-1.7B (Llama arch) **full FT on 2 x 8** (HSDP 8 x 2, socket NCCL 0.7 GB/s), new env `math_boxed` (MATH L1-3), exact match + 0.1 format, 30 steps x 512 rollouts, 28 s/step | 500 official-test problems | 22.4% -> 31.8% | **+9.4 [+5.6, +13.2]** | 71 / 24 | 3,148 node-s |
| `r3a-a2-fb` | Phi-4-mini **LoRA r32 on a 1 x 4 slice**, countdown (3-4 numbers, step-by-step prompt), correct + brevity-if-correct, `grpo_norm_by_std: false`, lr 2e-5, 34 steps x 256 rollouts (16 x 16), 89 s/step | 500 problems | 36.8% -> 71.6% | **+34.8 [+29.6, +40.0]** | 205 / 31 | 2,186 node-s |

What moved (failure classes, base -> trained): math_boxed `ok` 112 -> 159, `no_answer` 81 -> 58, `wrong_answer` 307 -> 283, mean length
242 -> 322 tokens (more reasoning, not only format). countdown `ok` 184 -> 358, truncated at the 1,024 cap 220 -> 105, `illegal_numbers`
91 -> 24, median length 596 -> 191: a large part of the gain is learning to search efficiently and finish inside the cap; the brevity term
stayed small (0.02 -> 0.06 vs correctness 0.25 -> 0.78) and is paid only on correct answers. Stability: grad norm flat (~0.02 countdown),
entropy 1.9 -> 0.7 (no collapse), 0 infrastructure failures. Validation curves: math 0.336 / 0.375 / 0.402 / 0.418 (steps 0/10/20/30);
countdown 0.43 / 0.57 / 0.71 / 0.71 / 0.74 / 0.70 / 0.68 / 0.71 (every 5; step 20 selected). W&B: finished, `test/*`, `infra/*`, tables.

### What the GPU rungs caught

- `r3a-can1-fb` (canary) + `r3a-a1-fb` (cancelled at step 4, 304 node-s): under countdown's default prompt ("Think briefly, then give only
  the expression") Phi-4-mini answered in 15-18 tokens and solved ~2% with learning signal 0.125, also on 3-4-number problems. The scorer
  was right; the prompt starved the model. Fix: `environment.countdown.prompt` (validated, part of the data manifest); base became ~43%.
- `r3b-can1-fb` (canary on the 1-node variant): base 0.56 on the 16 canary problems, signal 0.75: launched as is.

### Fresh agents (no context but the repo and the skill)

| Agent | Task | Outcome | What it changed |
| --- | --- | --- | --- |
| A | Phi-4-mini LoRA countdown + own brevity reward, ~1 node-h | built experiment, reward + tests; blocked by an expired SSO session | the AWS pre-check, `grpo_norm_by_std` warning + docs, profile rows |
| B | new env from a downloaded HF math dataset + full FT of a non-Qwen 1.5-4B on 2 x 8, 1-node fallback | built `math_boxed` (21 tests); found the 1-GPU canary breaking on an inherited `fsdp_size` | canary overlay fix, dataset-split + prompt-length docs, BUDGET wording |
| C | Qwen3-14B LoRA MiMo code + music, 2 x 8, reward A/B | prepared, not run (~38k node-s); CPU submit denied by its harness | BUDGET/clamp lines, eval-aware steps/attempt, fail-fast docs, permission rule |
| D (second loop) | Granite-3.3-2B LoRA math_boxed reward A/B with its own term | both dry runs passed; harness allowed CPU submits | A/B budget check in `plan --against`, `compare` fairness flags, ledger lifecycle, A/B rules |

### Independent review of this round's diff

Fixed with tests: a second resumed relaunch dropped the alternative shape (2 nodes from a 1-node checkpoint); a failing alternative shape
cancelled the healthy primary race; shapes validated up front; name check per launch family; exact BUDGET clamp; offline `verify`;
math_boxed over-credit cases (`-12 \text{ or } -30`, `1, 125`, `10 = 3`); the norm warning for small default terms.

## Modal moved to the team environment `genai-research` (2026-10-12)

`main` (the shared workspace default) reached 99% of its budget; at the cap Modal stops everything still running there. Everything strl
deploys or runs in Modal now targets `genai-research` (`services.modal.environment` in `base/coding_exec.yaml` and `base/mimo.yaml`; code
default `strl/services/modal.py` `DEFAULT_ENVIRONMENT`; `tools/prebuild_mimo_images.py` and `tools/gate_mimo.py` default `MODAL_ENVIRONMENT`
to it). Harbor creates its `__harbor__` sandbox app and resolves the sandbox secret in `MODAL_ENVIRONMENT`, so sandboxes move with it.
- `strl executor deploy`: `strl-code-exec` deployed in `genai-research`, version ping `fab5f1b39071ab5c` (same harness as before).
- CPU dry run `mgr-exec-dry1-fb` (job_db6jq10c78j0089ltok0, `mbpp-qwen35-binary.yaml`): passed; pod gate `modal_auth ok,
  modal_environment genai-research`; 512 executor calls from the pod, 104 calls/s, p95 1.32 s, 0 infrastructure errors; reward contract,
  W&B and critic gates passed.
- Not yet possible: Harbor/MiMo sandboxes. `genai-research` has no `modal-proxy-user` secret (secrets are per environment; values are not
  readable, so it cannot be copied by an agent). Until its owner creates it there, `coding_harbor` / `mimo` dry runs fail the `modal_secret`
  gate. `coding_exec` no longer declares a sandbox secret (it never mounted one).
- Left in `main` (idle, 0 tasks): `strl-code-exec` and `strl-mimo-inspect`.
