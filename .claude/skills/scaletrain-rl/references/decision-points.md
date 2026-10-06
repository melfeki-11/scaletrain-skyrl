# Decision points: the knobs you change, what they do, and where to see the effect

All knobs live in an experiment YAML (`experiments/*.yaml`). `python -m strl plan <exp>` prints the derived consequences and warnings
before you spend anything. "Watch" names the W&B key (project `scaletrain-rl-envs`, x-axis `optimizer_step`).

## The three free axes

| Knob | Mechanism | Expected effect | Watch | Evidence |
| --- | --- | --- | --- | --- |
| `model.id` / `model.revision` | profile auto-detected from `config.json` architectures (`strl/models/`) | bigger models: higher base pass rate (more learning signal), slower steps, more memory | `eval/task_metric` at step 0 (base), `timing/generate`, `ray/gpu.mem_used_gb.avg` | Qwen3.5-0.8B GSM8K: 51.2 s/step on 8 H100 |
| `environment.name` | `gsm8k`, `coding_exec` (single-turn, Modal-executed tests), `coding_harbor` (Terminus-2 agent in Modal sandboxes) | agentic tiers are harder and slower per step but exercise tool use | `eval/task_metric`, `generate/avg_num_turns`, `timing/generate` | gpt-6-luna: 0.90 single-turn vs 0.70 agentic on the same tasks |
| `reward.components` | weighted sum of named terms; each logged as `environment/reward_components/<fn>` | dense terms (partial credit, format) raise the learning signal on hard tasks but can be gamed | `train/learning_signal_fraction`, `environment/reward_components/*` vs `eval/task_metric` | A/B in RESULTS.md |

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
| `environment.harbor.max_seq_len` (default 8192) | ONE context size for the vLLM engine, the agent's context check and the trainer (the plan's `max_model_len` for Harbor) | too small: every trial fails on turn 1 (Terminus-2's first turn is 0.8-2.5k tokens; plan rejects < 2600 + 2 x max_generate_length); larger: more turns fit, more memory | `generate/trajectories_context_length_exceeded`, `generate/num_error_trajectories`, `environment/masked_group_fraction` |

## Selection, evaluation, budget

| Knob | Effect | Watch |
| --- | --- | --- |
| `evaluation.interval_steps` == `checkpoint_interval_steps` | every validated step is selectable (only exported steps can be the final checkpoint) | `eval/task_metric` per eval |
| `environment.sources.validation.size` | selection noise: SE ~ sqrt(p(1-p)/n) (~5 points at 96) | `eval/task_metric` |
| `budget.attempt_max_seconds`, `final_eval_reserve_seconds` | training gets attempt - startup (~550 s) - reserve | `budget/consumed_node_hours`, `strl plan` steps_in_attempt |
| `budget.stall_seconds` | watchdog: kill + alert + evaluate the latest export | heartbeat in `tools`/`strl monitor` |
| `tracking.critic.*` | LLM failure analysis every N steps, K rows, USD cap | `qualitative/critic`, `critic/*_frac`, `critic/cost_usd` |
