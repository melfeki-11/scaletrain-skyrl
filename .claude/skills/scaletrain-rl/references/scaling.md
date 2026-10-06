# Scaling up

Verified in this repo: one node (1-8 H100, slices included), colocated sync GRPO, FSDP, TP1 vLLM engines. Everything below is
the path to more scale; each step needs the same proof ladder (dry -> canary -> bounded run) before a long run.

## Bigger models on one node

- Change `model.*`; `strl plan` shows the memory estimate per phase. Above ~4B full-parameter on 8 H100, raise
  `inference.tensor_parallel_size` (engines become gpus/TP; plan re-derives) and lower `micro_train_per_gpu`.
- Long contexts: `max_prompt_tokens + max_generate_length` is the vLLM `max_model_len`; memory grows with it.
- LoRA / Megatron / MoE are SkyRL-supported but not wired here: add a model profile that emits the overrides
  (`trainer.policy.model.lora.*`, `trainer.strategy=megatron` + `megatron_config.*`) and prove an update reaches the engines.

## More nodes

- `limits.max_nodes` gates it. The single-node worker refuses `NUM_INSTANCES != 1`; multi-node needs the rendezvous from the
  verified two-node qwen35-grpo run (`ScaleTrain/qwen35-grpo/q35/network.py`, `worker.py`): routable job-interface IPs (never
  link-local), a private head-IP record in S3, `ray start` on peers with allocated CPUs, a cross-host NCCL collective before
  training, per-node checkpoint shard sync, distinct physical hosts (plain slices co-locate replicas). Port it as
  `strl/multinode.py` and gate it with a 2-node canary.

## Agentic environments at scale (Harbor)

- Use SkyRL's fully-async trainer for long multi-turn rollouts: `trainer.fully_async.enabled=true`, `max_staleness_steps=4`,
  `num_parallel_generation_workers in [mini_batch, mini_batch*(staleness+1)]`, `trainer.algorithm.policy_loss_type=rollout_is`
  (+ geometric sequence mask), `trainer.placement.colocate_all=false` (dedicated inference GPUs). Callbacks are unavailable in
  the fully-async trainer, so the deadline/canary logic must move into a trainer subclass.
- Sandbox throughput: `environment.harbor.max_concurrency` and `trajectories_per_second` (HarborGenerator rate limiter); Modal
  images are cached per Dockerfile (all exported tasks share one).
- Qwen3.5/Qwen3 history templates drop `<think>` blocks, so step-wise merging falls back to one sequence per turn (correct, O(T^2)
  tokens); a custom served chat template that keeps an empty think block restores merging.

## Faster single-turn environments

- `max_generate_length` first (eager decode: step time = longest sequence). Then `enforce_eager: false` (CUDA graphs) after a
  canary (risky with GDN + sleep mode), and `zero_variance_filter: true` to skip no-signal groups.
- Executor: `max_containers`/`@modal.concurrent` in `strl/tasks/coding/modal_app.py`; measured 191-196 calls/s at p95 0.39-0.50 s.
