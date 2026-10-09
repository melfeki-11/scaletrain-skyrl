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
4. **Budget is real.** Each program has a ledger (`budget.ledger`); `strl launch --watch` records every attempt. A new GPU program needs the
   user's authorization and its own ledger file. Never edit recorded ledger entries.
5. **Secrets**: never print, paste or put a key in YAML, argv, tags or W&B. Never read `.env*` files. The repo holds only locators
   (references/credentials.md). Do not delete or overwrite the Modal bundle in S3. Ask the user before any new credential action.
6. **One attempt name per submission**, lowercase, <= 26 chars, never reused. Do not touch other people's jobs.

## Setup (once per machine, ~2 min)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-local.txt
.venv/bin/python -m unittest discover -s tests -t .          # ~300 tests; they also plan every file under experiments/
aws sts get-caller-identity --profile production-developer     # S3 staging; ScaleTrain CLI: scale-train --version
```

Use `AWS_PROFILE=production-developer` for every `stage`/`launch`/`monitor`/`verify` command.

## Zero to launched (copy a template, change one axis)

| Start from | When |
| --- | --- |
| `experiments/gsm8k-qwen35.yaml` | single-turn math, no external services (fastest loop) |
| `experiments/mbpp-qwen35-binary.yaml` | single-turn coding, tests executed in Modal |
| `experiments/harbor-qwen35.yaml` | multi-turn agent (Terminus-2) in Modal sandboxes |
| `experiments/mimo-qwen35-0p8b-smoke.yaml` | MiMo agentic suite, 0.8B, 1 GPU (cheap end-to-end) |
| `experiments/mimo-phi4-mini-lora-verify.yaml` | MiMo subset, a non-Qwen model, LoRA, shaped reward, durable checkpoints, retention eval |
| `experiments/mimo-qwen38-27b-lora.yaml` / `-full.yaml` | 27B on 2 x 8 H100, LoRA / full |

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
- **Model weights**: up to ~10 GB, `strl fetch` downloads locally and `stage` uploads them. Above ~10 GB (e.g. a 7B in bf16 is 15 GB) skip the
  local download: `strl stage-model experiments/my-exp.yaml --attempt myexp-sm --team gen_ai.agent_training_foundation`, then
  `strl submit evidence/myexp-sm/job.json` (a CPU job copies HF -> S3 with a sha256 manifest, ~12 min for 55 GB). Already staged? Read-only:
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
| **Environment** | `environment.name` (`gsm8k`, `coding_exec`, `coding_harbor`, `mimo`); for mimo `environment.mimo.domains.<d>.{train,test,weight,filters,max_turns,max_agent_sec}` | `--set environment.mimo.domains.code.train=0 --set environment.mimo.domains.code.test=0` (off) |
| **Reward** | `reward.components`: weighted named terms or `module:function` from `user_rewards/` | see below |
| **Scale** | `compute.nodes`, `compute.gpus_per_node` (1/2/4/8), `inference.tensor_parallel_size`, `limits.max_nodes` | `--set compute.gpus_per_node=8 --set inference.tensor_parallel_size=2` |
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
Built-in names: `strl/tasks/<env>/rewards.py`, common `length_penalty`, `format_ok`. `strl plan` rejects unknown names/params.

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
- Multi-node needs nodes in one network domain (`waiting_for_network_domain_capacity` can last hours on every route). Add
  `--max-wait-hours 2 --fallback-set compute.nodes=1` (repeatable) when a 1-node result is acceptable (ask the user: it changes the
  experiment). The fallback races as `<attempt>-alt` and starts fresh (a checkpoint resumes only at the same world size). `launch`
  checks up front that every name it may create (`-alt`, `-p<k>` relaunches, route suffix) fits; ~19 characters is safe with all flags.
- Borrowed capacity can be reclaimed (`preemption_reported`): use `--watch --relaunch-on-preempt 2` with `evaluation.durable_checkpoints: true`
  (+ `budget.save_optimizer_state: true`; in-loop-result envs such as mimo) so the relaunch resumes from S3 (verified: `v2res2-fb`);
  or `--routes ro,fo` to avoid borrowing. Launch exit codes: 0 ok, 2 no capacity, 3 a job unsettled/unrecorded (fix the ledger by hand
  as printed), 4 preempted with no relaunch left, 5 the watched job FAILED.
- Keep CPU work (prepare, image prebuild, dry runs, docs) going while a GPU job queues. `strl capacity` (read-only, `ml-admin`) explains waits.
Details: references/launch-and-capacity.md.

## What to monitor

```bash
AWS_PROFILE=production-developer .venv/bin/python -m strl monitor experiments/my-exp.yaml --attempt <winner> --job-id <job>   # 1 line/min + flags
```

W&B: the URL is on `strl plan`'s `monitoring` line (`tracking.entity`/`project`, e.g. `https://scaleai.wandb.io/mohamed-elfeki/scaletrain-rl-envs`),
group = `tracking.group`, run id = the attempt, x-axis `optimizer_step`. S3 `attempts/<attempt>/artifacts/` is the source of truth (`completion.json`, `training.log`, `failure.log`).

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

How to decide: the change helped only if `test/<d>/delta` (or `test/macro`) has a CI excluding 0 and exceeds `eval/<d>/judge_variance`
for judged domains; check `retention/*` did not drop (general ability) and compare against the ceiling (`dry-run.json` `gates.ceiling` from a
dry run with `--set run.ceiling_per_domain=2`). A few updates are infrastructure evidence, not a training effect: say so.
Details: references/monitoring.md, references/decision-points.md.

## Checklists by experiment type

- **New OSS model**: revision sha -> `plan` (profile, memory, warnings) -> `fetch` or `stage-model` -> `check-template` (agentic) -> dry
  (proves `model_code`, validate_cfg) -> canary on a 1-2 GPU slice with LoRA -> run. Add its measured row to `strl/profiles.json`.
- **New reward design**: function in `user_rewards/`, unit-test it (`tests/`), `plan` -> dry (calls the terms in the pod) -> an A/B:
  experiments identical except `reward.components`, their own `tracking.group` (one group per study; arms told apart by `tracking.tags`),
  launched together -> `compare a:attA b:attB [c:attC]`. Check `plan B --against A` shows only the reward diff.
- **Different env subset / mix (mimo)**: change `domains.<d>.{train,weight}` only (held-out `test` sets are fixed and comparable); a new
  `run.data_dir` + `strl prepare` when the selection changes. Turn a domain off with `train: 0` AND `test: 0`.
- **Scale up (more GPUs / 2+ nodes)**: `plan` shows the layout (multi-node LoRA = HSDP automatically); dry with the target `compute.nodes`
  (rendezvous + validate_cfg at that GPU count) -> a 0.8B smoke at the same shape (`experiments/mimo-qwen35-0p8b-2node-smoke.yaml`) -> the run.
  Cross-node NCCL runs over sockets (~0.69 GB/s): LoRA is fine, full fine-tuning of large models is slow (references/scaling.md).
- **New MiMo-style suite**: references/extending.md ("A new MiMo-style suite"): graders, nop/tamper gates, judge outside the sandbox.
- **Resume / continue**: `--set run.resume_from=<attempt> --set algorithm.max_steps=<more>` on an attempt run with durable checkpoints
  (same GPU count); its paired result keeps the earlier attempt's step-0 base.
- **Retention (did RL break general ability?)**: `evaluation.retention: {experiment: base/gsm8k.yaml, limit: 200}` (>= 2 GPUs); read
  `retention/gsm8k/{base,trained,delta,ci_*}`.

## Gotchas that cost real cycles

- A canary that "completes" can be hollow: read `environment/*masked*`, `generate/num_error_trajectories`, `harbor-trials/*/exception.txt`.
- Agentic trials: turns x seconds-per-turn must fit `max_agent_sec`; timeouts score 0 in eval. Size `environment.harbor.max_seq_len` above
  the first turn (Terminus-2: 0.8-2.5k tokens).
- Templates that drop `<think>` history (Qwen3.5) do not merge turns: several times more training tokens (`check-template`).
- A CPU dry run on the research team's quota may never start: the race wins on `fb` anyway.
- Small models on single-shot music often overflow `max_generate_length`: those trials are scored placeholders (`generate/<d>/num_output_exceeded`,
  `environment/<d>/dropped_group_fraction`), not infrastructure masks; raise `algorithm.max_generate_length` if a domain has no trainable groups.
- `plan` also runs the chat-template merge check when the tokenizer is local (`HARBOR` warning); `check-template` forces it from the hub.
- `assets/overlay` must come after the venv on `sys.path` locally (Linux `.so` files). Custom reward modules must live in this repo.
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
