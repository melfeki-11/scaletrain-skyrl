# Decision points: the knobs you change, what they do, and where to see the effect

All knobs live in an experiment YAML (`experiments/*.yaml`). `python -m strl plan <exp>` prints the derived consequences and warnings
before you spend anything. "Watch" names the W&B key (project `scaletrain-rl-envs`, x-axis `optimizer_step`).

## The three free axes

| Knob | Mechanism | Expected effect | Watch | Evidence |
| --- | --- | --- | --- | --- |
| `model.id` / `model.revision` | profile auto-detected from `config.json` architectures (`strl/models/`) | bigger models: higher base pass rate (more learning signal), slower steps, more memory | `eval/task_metric` at step 0 (base), `timing/generate`, `ray/gpu.mem_used_gb.avg` | Qwen3.5-0.8B GSM8K: 51.2 s/step on 8 H100 |
| `environment.name` | `gsm8k`, `coding_exec` (single-turn, Modal-executed tests), `coding_harbor` (Terminus-2 agent in Modal sandboxes) | agentic tiers are harder and slower per step but exercise tool use | `eval/task_metric`, `generate/avg_num_turns`, `timing/generate` | gpt-6-luna: 0.90 single-turn vs 0.70 agentic on the same tasks |
| `reward.components` | weighted sum of named terms; each logged as `environment/reward_components/<fn>` | dense terms (partial credit, format) raise the learning signal on hard tasks but can be gamed | `train/learning_signal_fraction`, `environment/reward_components/*` vs `eval/task_metric` | measured (RESULTS.md, Qwen3.5-0.8B MBPP): +0.5·partial_tests raised the signal from 0.53 to 0.73 and the train reward from 0.58 to 0.80, but held-out pass@1 came out at 33.3% vs 36.1% for binary (−2.7 [−6.2, +0.8]) |

Reward rule of thumb: **judge every reward design by `eval/task_metric` and the held-out test**, never by its own reward. A
component that rises while the task metric stays flat is being gamed; the qualitative critic's `critic/reward_hacking_attempt_frac`
and `train/suspicious_code_rate` are early warnings.

## GRPO signal

| Knob | Mechanism | Effect | Watch |
| --- | --- | --- | --- |
| `algorithm.n_samples` (group size) | advantage = reward - group mean (/std) | all-equal groups give zero gradient; P(all-equal) = p^n + (1-p)^n at pass rate p | `train/learning_signal_fraction` (target > 0.3), `reward/num_zero_variance_filtered` |
| `algorithm.prompts_per_step` x `n_samples` | rollouts per step | keep the product fixed and trade breadth for group size: 32x16 has ~6x fewer all-fail groups than 64x8 at p=0.25 | same + `timing/generate` |
| `algorithm.temperature` | sampling diversity | higher -> more variance in groups (signal) and more garbage | `train/learning_signal_fraction`, `train/truncation_rate` |
| `algorithm.use_kl_loss`, `kl_coef` | KL to the reference policy | stability vs speed of drift; `use_kl_loss: false` also frees the reference model's memory | `policy/policy_kl`, `train/entropy` |
| `algorithm.lr` | step size | verified 2e-6 for 0.8B full-parameter; > 1e-5 risks collapse | `policy/grad_norm`, `train/entropy` (collapse = entropy -> 0) |
| `algorithm.zero_variance_filter` | drop all-equal groups from the batch | saves compute on no-signal groups; smaller effective batch | `reward/num_zero_variance_filtered` |

## Throughput

| Knob | Mechanism | Effect | Watch |
| --- | --- | --- | --- |
| `algorithm.max_generate_length` | eager decode: step time is set by the LONGEST sequence | the main lever; GSM8K hit its 1,024 cap every step; truncated rollouts score 0 (`zero_reward_on_non_stop`) | `timing/generate`, `generate/max_num_tokens`, `train/truncation_rate` |
| `compute.gpus_per_node` | data parallel width (1 TP1 engine per GPU) | near-linear for small models; slices (`:1/:2/:4`) admit faster | `trainer/tokens_per_second_per_gpu` |
| `inference.gpu_memory_utilization` | vLLM KV cache share (colocated with FSDP via sleep mode) | more concurrent sequences vs OOM risk in the train phase | `vllm/train/kv_cache_usage_perc` |
| `inference.max_env_workers` | concurrent env steps (Modal calls) | < 32 serialises scoring; 64 measured: 512 calls in 2.7 s, p95 0.39 s | `environment/exec_latency_s`, `timing/generate` |
| `environment.harbor.max_turns`, `max_concurrency` | agent episodes; concurrent sandboxes | more turns: harder tasks solvable, longer contexts, slower | `generate/avg_num_turns`, `generate/num_timeout_trajectories` |
| `budget.stall_seconds`, `budget.canary_max_seconds` | watchdog (no optimizer step for N s kills the run) and the canary's time to 2 updates | too low kills healthy agentic runs (a Harbor step waits for its slowest trajectory, and the canary evals after every step); too high wastes budget on a hung run | `progress.json` / `strl monitor` STALL, `timing/step` | coding/gsm8k 600 / 1200; Harbor 1800 / 2400 (`can-hb-e`: step 477 s + eval ~230 s on 1 GPU) |
| `environment.harbor.max_seq_len` (default 8192) | ONE context size for the vLLM engine, the agent's context check and the trainer (the plan's `max_model_len` for Harbor) | too small: every trial fails on turn 1 (Terminus-2's first turn is 0.8-2.5k tokens; plan rejects < 2600 + 2 x max_generate_length); larger: more turns fit, more activation memory and slower policy training (can-hb-e: 243 step-wise sequences padded to 8,019 tokens took 137 s); `strl plan`'s memory estimate covers weights, optimizer state and KV cache only, so the canary is the memory check | `generate/trajectories_context_length_exceeded`, `generate/num_error_trajectories`, `environment/masked_group_fraction` |

## Selection, evaluation, budget

| Knob | Effect | Watch |
| --- | --- | --- |
| `evaluation.interval_steps` == `checkpoint_interval_steps` | every validated step is selectable (only exported steps can be the final checkpoint) | `eval/task_metric` per eval |
| `environment.sources.validation.size` | selection noise: SE ~ sqrt(p(1-p)/n) (~5 points at 96) | `eval/task_metric` |
| `budget.attempt_max_seconds`, `final_eval_reserve_seconds` | training gets attempt - startup (~550 s) - reserve | `budget/consumed_node_hours`, `strl plan` steps_in_attempt |
| `budget.stall_seconds` | watchdog: kill + alert + evaluate the latest export | heartbeat in `tools`/`strl monitor` |
| `budget.failfast: {mem_gb, domain_masked, step_slow_factor, min_steps}` | in-run RuntimeError + alert: step-1 peak GPU memory above `mem_gb` (head nvidia-smi), a domain's masked-group fraction above `domain_masked` or step time above `step_slow_factor` x the plan estimate for 2 consecutive steps; renders nothing unless the environment opts in (mimo: 76 / 0.15 / 1.5 / 1) or you set it, `null` disables a rule; canary skips the last two | stops a doomed run in minutes instead of hours: `failfast.json`, W&B alert, `gpu/mem_peak_gb`, `environment/<d>/masked_group_fraction`, `timing/step` |
| predictive deadline (automatic, no knob) | stop after a step when `now + 1.15 x mean(last 2 steps) + last eval time + 120 s export >= deadline`; forces one final eval + export; the first step keeps the plain deadline | the last eval and export always fit inside the allocation instead of being cut off: `budget/stop_reason`, `evaluation_timing.jsonl` |
| `tracking.critic.*` | LLM failure analysis every N steps, K rows, USD cap | `qualitative/critic`, `critic/*_frac`, `critic/cost_usd` |

## Fine-tuning mode, scale, and the multi-domain mix

| Knob | Mechanism | Effect / when | Watch |
| --- | --- | --- | --- |
| `finetune.mode: lora \| full` | LoRA trains adapters on a frozen bf16 base (adapter-only gradient traffic, adapter synced to vLLM); full trains all weights | LoRA: far less memory and cross-node traffic, needs a larger lr (smoke 1e-5 vs full 1e-6..2e-6), lower capacity. Full: 16 B/param sharded; needs fast cross-node links or HSDP + offload | plan `layout` line, `train/gradient_norm`, `network.json` |
| `finetune.lora.{rank,alpha,target_modules}` | capacity of the adapter (all-linear r64 = 467M params on 27B) | higher rank: more capacity/memory | plan LoRA params, `eval/task_metric` |
| `compute.nodes` x `gpus_per_node` | placement; `limits.max_nodes` gates it; budget charges `nodes x gpus/8` per wall-second | more rollouts/step; cross-node NCCL becomes a risk (EFA vs sockets) | `network.json` (transport, GB/s), `infra` flags NETWORK / PEER_STALE |
| `compute.training.{fsdp_size,cpu_offload,max_tokens_per_microbatch}` | FSDP shard group, offload, micro-batching by tokens | LoRA: `fsdp_size = gpus_per_node` keeps the frozen base gather on NVLink; full: full shard if cross-node >= ~8 GB/s else HSDP + offload | step phases `timing/*`, GPU memory |
| `inference.{tensor_parallel_size,gpu_memory_utilization,engine_init_kwargs}` | vLLM engines = GPUs / TP; KV room = utilization x 80 GB - weights | 27B: TP2 x 8 engines at 0.5 (0.75 failed after TP init); GDN prefill backend `triton` (flashinfer segfaults) | engine start logs, `generate/*` |
| `environment.mimo.domains.<d>.{train,test,weight,filters}` | which domains/how many tasks/share of each step (largest-remainder quotas) | weight sets per-step prompts per domain; `test` is a fixed grouped set (changing `train` never changes it) | `generate/<d>/*`, `eval/<d>/task_metric`, `environment/<d>/masked_group_fraction` |
| `environment.mimo.eval_repeats` | each test task evaluated N times (distinct `rep`) | tighter paired CIs; N x eval cost | `eval/<d>/rep_std` |
| `environment.mimo.judge.{model,vision_model,votes,eval_votes}` | outside-sandbox LLM judge for rubric/vision items; outage masks, never scores 0 | judge model changes the reward scale: compare only within a run; votes cut noise | `generate/<d>/judge_*`, `eval/<d>/judge_variance` |
| `environment.mimo.step_deadline_sec`, per-domain `max_turns` / `max_agent_sec` / `max_verifier_sec` | `max_turns` bounds a trial by turns (a run-level cap on the task's own limit), the `max_*_sec` caps bound it by the clock, the step deadline cancels stragglers (masked) | choose turns x ~35 s (27B) < `max_agent_sec` so trials end by their turn limit: a timed-out trial is scored (0 in eval; in training it trains on the verifier's reward if it left tokens, else it is a zero-loss placeholder). Measured: at ~30 s per turn, 8-14 turns and 600 s left 107 of 160 held-out trials timed out and general/webdev scored ~0: give general and webdev far more turns | `generate/<d>/num_timeout`, `environment/<d>/timeout_group_fraction`, `generate/<d>/num_deadline` |
