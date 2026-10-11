---
name: scaletrain-rl-experiments
description: Launch, run, monitor, verify and compare RL post-training experiments (SkyRL GRPO) of open-weight LLMs on ScaleTrain GPUs as fast as capacity allows. Use to train any OSS model (Qwen, Phi, Mistral, OLMo, Llama...; LoRA or full; 1 GPU to multi-node) on any RL environment (GSM8K, coding, Harbor agentic sandboxes in Modal, the 5-domain MiMo suite or a new MiMo-style suite) with any reward design, to get GPUs ASAP by racing every team/borrowing route, to watch W&B metrics and tables, and to judge whether a change helped. Self-contained: needs only this repo.
---

# ScaleTrain RL experiments (`strl`)

One experiment = one YAML in `experiments/`. One CLI, `.venv/bin/python -m strl`, run from the repo root (every path below is relative
to it). You change the model, the environment, the reward, the scale; everything else (placement, engines, credentials, sandboxes, budget,
checkpoints, evaluation, monitoring) is derived and validated by `strl plan` before anything costs money.

## Golden rules

1. **Never spend GPUs before `strl plan` is clean and a `--mode dry` passed.** Dry runs are free (CPU pod, the real image) and catch
   most bugs. A new model, environment or reward gets a `--mode canary` (2 updates, small) before its first real run.
2. **Always launch with `strl launch`.** It races every capacity route (team x borrowing) at once and keeps the first that starts.
   Never submit one job and wait. Never lower the priority.
3. **Judge on the task metric, never the training reward.** `eval/task_metric` (per domain: `eval/<d>/task_metric`) and the paired
   held-out result `test/*` with its CI. A shaping term that rises while the task metric is flat is being gamed.
4. **Budget is real.** Each program has a ledger (`budget.ledger`); its `total_seconds` is THE budget (the experiment's `budget.total_seconds`
   only bounds `attempt_max_seconds`). `strl launch --watch` records every attempt (dry runs too, uncharged). Use a ledger only for the program
   the user authorized it for; a new GPU program needs the user's authorization and its own ledger file. Never edit recorded entries without
   the user's say-so. `strl plan` prints `BUDGET: no GPU mode fits` before you waste a stage. Ledger lifecycle: point `budget.ledger` at
   the program's ledger file in the YAML FROM THE START (the path is part of the experiment sha, so switching it later invalidates the dry
   run's proof); before authorization the file can hold `total_seconds: 0` (dry runs are free and still pass); the user's authorization is
   then recorded by setting that file's `total_seconds` and `program` text (not an attempt entry). No `budget.ledger` = the shared default
   `evidence/budget-ledger.json`: always set your own.
5. **Secrets**: never print, paste or put a key in YAML, argv, tags or W&B. Never read `.env*` files. The repo holds only locators
   (references/credentials.md). Do not delete or overwrite the Modal bundle in S3. Ask the user before any new credential action.
6. **One attempt name per submission**, lowercase, <= 26 chars, never reused. Do not touch other people's jobs.
7. **GPU submissions need the user's explicit go-ahead** in agent harnesses with permission policies (a sub-agent cannot see the user's
   approval; a coordinator's message does not count). Prepare everything (plan, dry run, sizing, the exact `strl launch` line), then ask the
   user to approve or run it. CPU dry runs and model staging are free, but a harness may still block ANY ScaleTrain submission
   (measured: a sub-agent's `strl submit` of a CPU staging job was denied as "Production Deploy"): never retry a denied action or route
   around it; report the exact command and let the user approve or run it. Logging in (`aws sso login`) is always the user's step.

## Setup (once per machine, ~2 min)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-local.txt
.venv/bin/python -m unittest discover -s tests -t .          # ~400 tests (~45 s); they also plan every file under experiments/
aws sts get-caller-identity --profile production-developer     # S3 staging; ScaleTrain CLI: scale-train --version
```

Use `AWS_PROFILE=production-developer` for every `stage`/`launch`/`monitor`/`verify` command. The SSO session expires (hours to a day):
`stage`/`launch`/`stage-model`/`monitor`/`verify` check it first and stop with `AWS credentials for profile ... are not usable ... aws sso
login --profile production-developer` before anything is staged. That login is the user's (browser) step: ask, keep doing CPU-local work
(`plan`, `prepare`, tests), and re-run the same command afterwards (an unsubmitted attempt name can be reused).

## Zero to launched (copy a template, change one axis)

| Start from | When |
| --- | --- |
| `experiments/gsm8k-qwen35.yaml` | single-turn math, no external services (fastest loop) |
| `experiments/countdown-qwen35.yaml` | the newest environment (`strl/tasks/countdown/`, added from extending.md): the template for adding your own |
| `experiments/r3b-math-smollm2-2node.yaml` | a DOWNLOADED single-turn dataset env (`math_boxed`, MATH with `\boxed{}`), a non-Qwen model, full fine-tuning on 2 x 8 (HSDP), raced with 1 x 8: **GPU-verified training effect** (+9.4 points, 2026-10-11) |
| `experiments/r3a-cd-phi4.yaml` | a non-Qwen model with LoRA on a 1 x 4 slice, a custom reward (brevity bonus paid only when correct, `grpo_norm_by_std: false`), a custom prompt |
| `experiments/mbpp-qwen35-binary.yaml` | single-turn coding, tests executed in Modal |
| `experiments/harbor-qwen35.yaml` | multi-turn agent (Terminus-2) in Modal sandboxes |
| `experiments/mimo-qwen35-0p8b-smoke.yaml` | MiMo agentic suite, 0.8B, 1 GPU (cheap end-to-end) |
| `experiments/mimo-phi4-mini-lora-verify.yaml` | MiMo subset, a non-Qwen model, LoRA, shaped reward, durable checkpoints, retention eval |
| `experiments/mimo-qwen3-8b-lora-2node.yaml` | a mid-size dense model on 2 x 8 H100, LoRA, a MiMo subset (cyber + code) |
| `experiments/mimo-qwen38-27b-lora.yaml` / `-full.yaml` | 27B on 2 x 8 H100, LoRA / full |
| `experiments/mimo-example-music-code-full.yaml` | template for a NEW MiMo program (own data dir, ledger, full fine-tuning, custom shaping term) |

```bash
cp experiments/mimo-phi4-mini-lora-verify.yaml experiments/my-exp.yaml       # set name, description, tracking.group, then the axis you study
.venv/bin/python -m strl plan experiments/my-exp.yaml                         # errors, layout, memory, step time, budget, warnings: read them all
.venv/bin/python -m strl plan experiments/my-exp.yaml --against experiments/mimo-phi4-mini-lora-verify.yaml   # the exact resolved diff
.venv/bin/python -m strl fetch experiments/my-exp.yaml                        # small models: pinned snapshot + data (mimo/coding: runs prepare)
.venv/bin/python -m strl check-template experiments/my-exp.yaml               # agentic envs: do this model's turns merge into one sequence?
AWS_PROFILE=production-developer .venv/bin/python -m strl launch experiments/my-exp.yaml --attempt myexp-dry --mode dry --watch
AWS_PROFILE=production-developer .venv/bin/python -m strl launch experiments/my-exp.yaml --attempt myexp-can --mode canary --watch
AWS_PROFILE=production-developer .venv/bin/python -m strl launch experiments/my-exp.yaml --attempt myexp-a1 --mode train --watch --relaunch-on-preempt 2
```

`launch` prints the WINNING attempt name (`myexp-a1-fb`, `-rb`, `-ro` or `-fo`); use it in every later command. Run it in the background
(`... > evidence/launch-myexp-a1.log 2>&1 &`) and follow the log for `WINNER`, `preempt`, `Traceback`, `next:`.

Extra preparation by case:
- **Model revision**: `curl -s https://huggingface.co/api/models/<id> | python3 -c 'import sys,json;print(json.load(sys.stdin)["sha"])'`.
- **Model weights**: up to ~10 GB, `strl fetch` downloads locally and `stage` uploads them (`fetch` also runs `prepare` when the
  experiment's data is missing; `stage-model` works for any size and touches nothing local). Above ~10 GB (e.g. a 7B in bf16 is 15 GB) skip the
  local download: `strl stage-model experiments/my-exp.yaml --attempt myexp-sm --team gen_ai.agent_training_foundation`, then
  `strl submit evidence/myexp-sm/job.json` (a CPU job copies HF -> S3 with a sha256 manifest: 16 GB ~2 min, 61 GB ~5 min; free, not
  ledgered); follow it with `strl status <job_id>` until COMPLETED. `stage` (dry runs included: the pod skips the weights but the manifest must
  exist) refuses until the model is local or staged. Already staged? Read-only:
  `aws s3 ls s3://scale-ml/tmp/mohamedelfeki/scaletrain/qwen35-grpo-v2/models/<name>@<rev12>/ --profile production-developer`.
- **Coding envs**: `strl executor deploy` once. **Harbor/MiMo**: `strl overlay` once (Modal/Harbor packages for the image).
- **MiMo data**: `run.data_dir` holds the prepared tasks. Shipped: `assets/mimo/data` (all 5 domains, 24 train / 24 test), `data-smoke`
  (5 x 4/2), `data-2node` (music + cyber, 16/4). `plan` prints `DATA` when the dir does not cover the selection. Lowering counts reuses a
  dir; more tasks, other filters or another seed need a new `run.data_dir` and `strl prepare experiments/my-exp.yaml` (downloads the pinned
  parquet from HF and exports task dirs locally; no Modal, no GPUs), then
  `.venv/bin/python tools/prebuild_mimo_images.py <run.data_dir>/tasks --concurrency 8` and the nop gate
  `.venv/bin/python tools/gate_mimo.py <domain> <task dir>...` (nop must score 0 with the verifier proven to have run).

## What to change (YAML keys; or try without editing: `--set key=value` on plan/stage/launch, recorded in the experiment sha)

| Axis | Keys | Example |
| --- | --- | --- |
| **Model** | `model.id`, `model.revision` (a commit sha). Leave `model.local_path` and `model.profile` unset (auto). | `--set model.id=microsoft/Phi-4-mini-instruct --set model.revision=cfbefacb99257ffa30c83adab238a50856ac3083` |
| **Fine-tuning** | `finetune.mode: lora\|full` (full ignores an inherited `finetune.lora`), `finetune.lora.{rank,alpha,target_modules}` (rank must be a vLLM rank: 8/16/32/64/128/256). lr: LoRA 1e-5; full 2e-6 at <= 2B (verified on 0.8B), 1e-6 at 7B+ | `--set finetune.mode=full --set algorithm.lr=1e-6` |
| **Environment** | `environment.name` (`gsm8k`, `coding_exec`, `coding_harbor`, `mimo`, `countdown`, or your own: extending.md); for mimo `environment.mimo.domains.<d>.{train,test,weight,filters,max_turns,max_agent_sec}` | `--set environment.mimo.domains.code.train=0 --set environment.mimo.domains.code.test=0` (off) |
| **Reward** | `reward.components`: weighted named terms or `module:function` from `user_rewards/` | see below |
| **Scale** | `compute.nodes` (+ `limits.max_nodes` >= it), `compute.gpus_per_node` (1/2/4/8), `inference.tensor_parallel_size` (raise it until `plan` shows KV room > ~10 GB: bf16 weights / TP must fit in `gpu_memory_utilization` x 80 GB; the offline final eval uses the same TP) | `--set compute.nodes=2 --set limits.max_nodes=2 --set inference.tensor_parallel_size=2` |
| **GRPO** | `algorithm.{prompts_per_step,n_samples,lr,temperature,max_generate_length,max_steps,use_kl_loss}` | `--set algorithm.n_samples=8` |
| **Evaluation** | `evaluation.{interval_steps,retention,durable_checkpoints}`, `environment.mimo.eval_repeats`, `run.ceiling_per_domain` (dry) | `--set evaluation.retention.limit=200` |

Reward designs:

```yaml
reward:
  components:
    - {fn: verifier_score, weight: 1.0}                               # mimo: the native grader, exactly once; coding: pass_all_tests; gsm8k: strict_answer
    - {fn: partial_tests, weight: 0.5}                                # built-in shaping (coding_exec)
    - {fn: 'user_rewards.examples:turn_efficiency_bonus', weight: 1.0, bonus: 0.05, max_turns: 10}   # your function
```

A term is `fn(result: dict, ctx, **params) -> float` in `user_rewards/<module>.py` (shipped to the pod automatically). For mimo, `result` =
`{domain, task, reward (native 0-1), metric, passed (grader's verdict, else metric >= 0.999), components (grader sub-scores), turns,
tokens (generated tokens summed over turns, prompts excluded; 0 for a placeholder), seconds, stop ('complete', 'agent_timeout',
'output_length', ...), artifact (the deliverable, truncated)}` and `ctx` has only `extras['domain']` and `max_generate_length` (no completion
string). Shaping changes the TRAINING reward only. The dry run calls every term on a synthetic pass/fail per domain (`shaping` in the receipt).
For single-turn envs (gsm8k, coding_exec, countdown) `result` is the scorer's dict (`completion`, `parsed`, `correct`, `failure_class`;
coding adds `passed`, `total`, `compiled`, `suspicious`) and `ctx` has `ground_truth`, `extras`, `max_generate_length` and `count_tokens`
(the model's real tokenizer in the pod). countdown adds `target`, `value`, `valid`, `legal_numbers`, `numbers_used`, `distance`
(`strl/tasks/countdown/scoring.py`); math_boxed adds `normalized`, `truth_normalized`, `single_answer`, `boxed_count`
(`strl/tasks/math_boxed/scoring.py`). The dry run's reward-contract cases (per-case reward, components) are printed by `launch --watch` (from
`skyrl_validate.generator.cases` / `generic_reward_samples` / `shaping` in `dry-run.json`). Any custom term moves gsm8k from SkyRL's
built-in env to `strl_env` (proven equivalent; `plan B --against A` shows it as a derived diff).
Built-in names: `strl/tasks/<env>/rewards.py`, common `length_penalty`, `format_ok`. `strl plan` rejects unknown names/params.
**Small shaping terms need `algorithm.grpo_norm_by_std: false`.** GRPO divides each group's advantages by the group's reward std (SkyRL
default on): in a group where every answer is equally correct, only the shaping term varies, and its tiny std turns a 0.1 brevity bonus into a
full-size advantage, so the run optimises length instead of correctness. `plan` warns `REWARD: grpo_norm_by_std is on`; mimo sets it off.

Models: any dense decoder works through the `dense`/`generic` profiles (Qwen2/3/3.5/3.8, Phi-3/4, Mistral, OLMo, Gemma text, Granite, GLM4...);
`plan` refuses unmapped vision-language wrappers and contradictory profiles, warns on MoE (memory counts all experts) and unknown memory.
Gated models (Llama, Gemma) need the HF license accepted. Checkpoints that bundle remote modeling code (`auto_map`, e.g. Phi) are switched
to transformers' native class in the pod (`model_code` in `worker-started`/`dry-run.json`). Details: references/extending.md.

## Getting capacity ASAP (every launch, every time)

`strl launch` submits one job per route (in order, stopping as soon as a submitted one starts) and keeps the first that starts (losers are
cancelled and recorded; queue time is never charged):
`fb` foundation+borrowing, `rb` research+borrowing, `ro` research own quota, `fo` foundation own quota. Measured: the same job waited ~23 h on
one route and started first on another; dry runs start in seconds on `fb`.
- Prefer the smallest shape that answers the question: `gpus_per_node` 1/2/4 slices admit far faster than 8, and 1 node far faster than 2.
- **Race shapes, not only routes**: `--race-shape "compute.nodes=1"` (repeatable; each value is one alternative shape, space-separated
  `K=V`s, e.g. `"compute.gpus_per_node=4"` for a 1 x 8 run) keeps the experiment's own shape on every route and, once it has queued
  `--race-shape-after-minutes` (default 10) without starting, submits the alternative on every route too; the first job to start wins
  (`<attempt>-s1-<route>`; the launch receipt's `shape` names its overrides) and the rest are cancelled. The global batch is unchanged (only
  speed and per-GPU memory change), but check each alternative first: `strl plan <exp> --set <its K=Vs>` must fit, and a dry run with the same
  `--set`s proves it (`--set` changes the experiment sha). A relaunch that resumes a checkpoint stays on the checkpoint's shape. Use it
  whenever a result on the alternative shape is acceptable (ask the user when the shape is part of the question, e.g. a 2-node study).
- Multi-node needs nodes in one network domain (`waiting_for_network_domain_capacity` can last hours on every route). Add
  `--max-wait-hours 2 --fallback-set compute.nodes=1` (repeatable) when a 1-node result is acceptable (ask the user: it changes the
  experiment). The fallback races as `<attempt>-alt` and starts fresh (a checkpoint resumes only at the same world size). `launch`
  checks up front that every name it may create (`-alt`, `-p<k>` relaunches, route suffix) fits; ~19 characters is safe with all flags.
- Borrowed capacity can be reclaimed (`preemption_reported`): use `--watch --relaunch-on-preempt 2` with `evaluation.durable_checkpoints: true`
  (+ `budget.save_optimizer_state: true`; in-loop-result envs such as mimo) so the relaunch resumes from S3 (verified: `v2res2-fb`).
  Single-turn envs with the offline final eval cannot resume yet: a relaunch restarts (keep their attempts short);
  or `--routes ro,fo` to avoid borrowing. Launch exit codes: 0 ok, 2 no capacity, 3 a job unsettled/unrecorded (fix the ledger by hand
  as printed), 4 preempted with no relaunch left, 5 the watched job ended FAILED or CANCELED.
- Keep CPU work (prepare, image prebuild, dry runs, docs) going while a GPU job queues. `strl capacity` explains waits but needs the
  `ml-admin` profile, which only the user may authorize: agents skip it. There is no other read-only capacity view an agent can use (the
  `scale-train` queue screen is interactive); the race itself is the capacity test: it submits every route (and `--race-shape` shape) and
  the log line `<attempt>-<route>[STATUS detail]` per poll shows which ones are queued and why (`waiting_for_network_domain_capacity`, ...).
- Canary jobs are capped at 3600 s whatever `budget.canary_max_seconds` says (plan warns `CANARY`); a 2 x 8 canary therefore books up to
  (3600 + 420) x 2 = 8040 node-s (stage clamps any attempt to what the ledger has left; plan's `BUDGET` lines say when and by how much). Size per-domain caps so step-0 eval + 2 steps fit, or go straight to a short `train` attempt (`max_steps` 2-4, small
  `attempt_max_seconds`): train mode applies the same step-2 canary rule.
Details: references/launch-and-capacity.md.

## What to monitor

```bash
AWS_PROFILE=production-developer .venv/bin/python -m strl monitor experiments/my-exp.yaml --attempt <winner> --job-id <job>   # 1 line/min + flags
```

W&B: the URL is on `strl plan`'s `monitoring` line (`tracking.entity`/`project`, e.g. `https://scaleai.wandb.io/mohamed-elfeki/scaletrain-rl-envs`),
group = `tracking.group`, run id = `<attempt>-<stamp>`, x-axis `optimizer_step`. S3 `attempts/<attempt>/artifacts/` is the source of truth (`completion.json`, `training.log`, `failure.log`).

| Question | Look at | Healthy |
| --- | --- | --- |
| Did it start right? | monitor line, `artifacts/worker-started.json` (`model_code`), `training.log` | first generation ~5-10 min (0.8-4B), ~12-16 min (27B); the step-0 eval then takes as long as its slowest trial (agentic: up to `max_agent_sec`) |
| Is it learning? | `eval/task_metric`, `eval/<d>/task_metric`, `eval/delta_from_baseline` | rises above step 0 (base) |
| Is there signal? | `train/learning_signal_fraction`, `generate/<d>/pass_rate` | > 0.3; ~0 or ~1 pass rate = no gradient |
| Is the reward honest? | `generate/<d>/reward_terms/<fn>`, `environment/reward_components/*` vs task metric, `qualitative/critic` | terms do not rise alone |
| Is the env healthy? | `environment/<d>/{masked,timeout,dropped}_group_fraction`, `eval/<d>/no_artifact_rate`, flags `DOMAIN_MASKED`, `TIMEOUTS`, `ENV_FAILURES` | masked < 5%; timeouts mean raise caps or lower `max_turns` |
| Is the policy stable? | `policy/grad_norm`, `train/entropy`, `policy/policy_kl`, `train/truncation_rate`, flag `NAN` | entropy not -> 0 |
| Is it fast / fitting? | `timing/step` vs plan (`STEP_SLOW`), `gpu/mem_peak_gb` (`MEM` > 76), `infra/*` and `NODE_LOST`/`NETWORK` (multi-node) | |
| Qualitative | tables `trajectories/train`, `trajectories/eval`, `qualitative/fixed_validation`, `qualitative/critic`, `qualitative/{fixed,broken}_by_training` | read 5 rows per domain |

After the run (in order):

```bash
.venv/bin/python -m strl verify experiments/my-exp.yaml --attempt <winner>          # independent recomputation/re-scoring -> verified: true
.venv/bin/python -m strl compare experiments/a.yaml:<attA> experiments/b.yaml:<attB>   # paired, same held-out ids, per domain + macro
.venv/bin/python -m strl report experiments/my-exp.yaml:<winner>                      # evidence/results/<group>-report.md + W&B report run
```

Single-turn envs (gsm8k, coding_exec, countdown) use the same panels without the domain level: `eval/task_metric`,
`environment/reward_components/<fn>`, `environment/failure/<class>`, `environment/task_metric` (training hit rate), and the final
`test/{baseline_accuracy,candidate_accuracy,accuracy_delta,ci_lower,ci_upper}`; compare two arms with `strl compare a.yaml:<A> b.yaml:<B>`.

How to decide: the change helped only if `test/<d>/delta` (or `test/macro`, or single-turn `test/accuracy_delta`) has a CI excluding 0 and exceeds `eval/<d>/judge_variance`
for judged domains; check `retention/*` did not drop (general ability) and compare against the ceiling (`dry-run.json` `gates.ceiling` from a
dry run with `--set run.ceiling_per_domain=2`). A few updates are infrastructure evidence, not a training effect: say so.
Details: references/monitoring.md, references/decision-points.md.

**Reading `plan`.** "Clean" means no errors. Every `!` line is advisory and must be read; expected ones: `MODEL` (new family), `SIGNAL`,
`SPEED`, `REWARD` (you added shaping), `MIMO`, `LORA_SYNC` / `NETWORK` (multi-node). Act on: `DATA` (run prepare), `LEDGER` / `BUDGET`
(no GPU mode fits, an attempt would be clamped, or a canary + run do not both fit), `MEMORY` (fits=False), `CANARY`, `REWARD:
grpo_norm_by_std`, `HARBOR` (turns will not merge: correct but several times more training tokens; no YAML fix exists: a served chat
template that keeps an empty think block would restore merging, but it is not exposed as a key; budget the extra train time instead).
`steps/attempt` counts the in-loop evals (step 0 + every `interval_steps`, ~one generate phase each for agentic envs). Step-time estimates marked UNCALIBRATED come from another model/env: size `max_steps` / `attempt_max_seconds` as start-up
(5-16 min) + step-0 eval + steps x step time + final eval, then let the canary or a short run measure it.

**What a dry run does NOT prove**: weights are skipped, so vLLM never loads the model or its LoRA, and the reward-contract cases use fixed
strings, not the model's chat template; the canary (1 GPU) is the first real load. Point `budget.ledger` at the ledger in the YAML (not via
`--set`) so the dry run proves the same experiment sha the GPU run will use.

**After a dry run** `strl launch --watch` prints the gates it passed (`model_code`, validated layout, environment checks, W&B, critic, durable
checkpoint I/O, ceiling, shaping). An environment with no external services shows `environment {}`: that is normal.

## Checklists by experiment type

- **Canary = the experiment on 1 GPU** with a small batch (8 prompts x 4 samples, 2 updates, no final eval) whenever the model fits one
  GPU (`base/canary_overlay.yaml`), else on the experiment's own placement (`canary_overlay_large.yaml`). Its gate is
  `attempts/<a>/artifacts/<a>/canary.json` (finite non-zero gradient, environment-failure fraction); `strl verify` is for train attempts.
  Not every key exists in every env: `generate/num_error_trajectories`, `environment/masked_group_fraction` and `gpu/mem_peak_gb` are
  Harbor/mimo (fail-fast) keys, and the critic table only appears when `tracking.critic.enabled` (off in `base/gsm8k.yaml`).
- **New OSS model**: revision sha -> `plan` (profile, memory, warnings) -> `fetch` or `stage-model` -> `check-template` (agentic) -> dry
  (proves `model_code`, validate_cfg) -> canary on a 1-2 GPU slice with LoRA -> run. Add its measured row to `strl/profiles.json`.
- **New reward design**: function in `user_rewards/`, unit-test it (`tests/`), `plan` -> dry (calls the terms in the pod; read the
  per-case rewards in `generic_reward_samples`) -> an A/B:
  experiments identical except `reward.components` (everything else equal, including `algorithm.grpo_norm_by_std`: set it on BOTH arms even
  though only the shaped arm triggers the warning; with it off, LoRA has used lr 2e-5, full FT keeps 2e-6), their own `tracking.group` (one
  group per study; arms told apart by `tracking.tags`), launched together -> `compare a:attA b:attB [c:attC]`. Check `plan B --against A`
  shows only the reward diff (it also warns `BUDGET: the two arms book ...` when both do not fit the ledger together).
  Fairness: both arms must finish the SAME number of steps on the SAME shape: set `max_steps` well below plan's steps/attempt (the
  predictive deadline otherwise stops the slower arm earlier), launch both arms with the same flags and no `--race-shape` (or check both
  winners' `shape`). The decision number is `compare`'s trained-vs-trained line `**B vs A** ... points [CI]`; it prints
  `! NOT A CLEAN A/B: optimizer steps differ / placement differs / base evaluations disagree` when the arms are not comparable (measured:
  the 2026-10-06 MBPP reward A/B ran 66 vs 69 steps).
- **Different env subset / mix (mimo)**: change `domains.<d>.{train,weight}` only (held-out `test` sets are fixed and comparable); a new
  `run.data_dir` + `strl prepare` when the selection changes. Turn a domain off with `train: 0` AND `test: 0`.
- **Scale up (more GPUs / 2+ nodes)**: `plan` shows the layout (multi-node LoRA = HSDP automatically); dry with the target `compute.nodes`
  (rendezvous + validate_cfg at that GPU count) -> a 0.8B smoke at the same shape (`experiments/mimo-qwen35-0p8b-2node-smoke.yaml`) -> the run.
  Cross-node NCCL runs over sockets (~0.69 GB/s): LoRA is fine, full fine-tuning of large models is slow (references/scaling.md).
- **New environment (single-turn, verifiable)**: references/extending.md "A new single-turn environment" (worked example: countdown):
  package under `strl/tasks/<name>/`, one registry line, `base/<name>.yaml` with its own `run.data_dir`, `strl prepare`, unit tests, plan, dry run.
- **New MiMo-style suite**: references/extending.md ("A new MiMo-style suite"): graders, nop/tamper gates, judge outside the sandbox.
- **Resume / continue**: `--set run.resume_from=<attempt> --set algorithm.max_steps=<more>` on an attempt run with durable checkpoints
  (same GPU count); its paired result keeps the earlier attempt's step-0 base.
- **Retention (did RL break general ability?)**: `evaluation.retention: {experiment: base/gsm8k.yaml, limit: 200}` (TP <= GPUs per node); read
  `retention/gsm8k/{base,trained,delta,ci_*}`.

## Gotchas that cost real cycles

- A canary that "completes" can be hollow: read `environment/*masked*`, `generate/num_error_trajectories`, `harbor-trials/*/exception.txt`.
- **Read the canary's completions before the run** (W&B `trajectories/train`, or `failure_class` counts in its `metrics.jsonl`). Measured
  2026-10-11: under countdown's default prompt ("Think briefly, then give only the expression") Phi-4-mini answered in 15-18 tokens with no
  search and solved ~2% (learning signal 0.125); with a "work it out step by step" prompt (`environment.countdown.prompt`) its base rose to
  ~43% with signal 0.88. A base pass rate near 0 is usually the prompt or the format, not the model: fix it before buying steps.
- Agentic trials: turns x seconds-per-turn must fit `max_agent_sec`; timeouts score 0 in eval. Size `environment.harbor.max_seq_len` above
  the first turn (Terminus-2: 0.8-2.5k tokens).
- Templates that drop `<think>` history (Qwen3.5) do not merge turns: several times more training tokens (`check-template`).
- A CPU dry run on the research team's quota may never start: the race wins on `fb` anyway.
- Small models on single-shot music often overflow `max_generate_length`: those trials are scored placeholders (`generate/<d>/num_output_exceeded`,
  `environment/<d>/dropped_group_fraction`), not infrastructure masks; raise `algorithm.max_generate_length` if a domain has no trainable groups.
- `plan` also runs the chat-template merge check when the tokenizer is local (`HARBOR` warning); `check-template` forces it from the hub.
- `assets/overlay` must come after the venv on `sys.path` locally (Linux `.so` files). Custom reward modules must live in this repo.
- Some chat templates inject today's date (Granite) or a default system prompt (SmolLM2): prompts then differ by day / grow by ~30 tokens.
  Pin a date-free template behaviour or accept it; measure the longest prompt WITH the template against `algorithm.max_prompt_tokens`.
- LiteLLM (judge, critic, ceiling) runs in the pod via its role; do not export admin credentials locally (references/credentials.md).

## References

[launch-and-capacity.md](references/launch-and-capacity.md) routes, preemption, resume, budget ·
[decision-points.md](references/decision-points.md) every knob -> effect -> W&B key ·
[monitoring.md](references/monitoring.md) metrics, tables, flags, verify/compare/report ·
[extending.md](references/extending.md) new model / reward / environment / suite ·
[scaling.md](references/scaling.md) big models, multi-node, network ·
[troubleshooting.md](references/troubleshooting.md) symptom -> fix ·
[credentials.md](references/credentials.md) locators and checks ·
[architecture.md](references/architecture.md) data/control flow, package map ·
[validation.md](references/validation.md) what is proven, with job ids
