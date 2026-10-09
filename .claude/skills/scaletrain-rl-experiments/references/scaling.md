# Scaling up

Verified in this repo: one node (1-8 H100, slices included), colocated sync GRPO, FSDP, TP1 vLLM engines. Everything below is
the path to more scale; each step needs the same proof ladder (dry -> canary -> bounded run) before a long run.

## Bigger models on one node

- Change `model.*`; `strl plan` shows the memory estimate per phase. Above ~4B full-parameter on 8 H100, raise
  `inference.tensor_parallel_size` (engines become gpus/TP; plan re-derives) and lower `micro_train_per_gpu`.
- Long contexts: `max_prompt_tokens + max_generate_length` is the vLLM `max_model_len`; memory grows with it.
- Megatron / MoE are SkyRL-supported but not wired here (`trainer.strategy=megatron` + `megatron_config.*`).

## LoRA vs full fine-tuning, FSDP layout (experiment keys; defaults render nothing)

```yaml
finetune: {mode: lora, lora: {rank: 64, alpha: 64, dropout: 0.0, target_modules: all-linear, exclude_modules: null}}  # default mode: full
compute:
  training: {fsdp_size: 8, cpu_offload: false, reshard_after_forward: true, reduce_dtype: fp32,      # fsdp_size null = full shard
             max_tokens_per_microbatch: null, gradient_checkpointing: true}
inference: {tensor_parallel_size: 2, num_engines: 8, gpu_memory_utilization: 0.5, max_num_batched_tokens: 8192,
            engine_init_kwargs: {gdn_prefill_backend: triton}}
budget: {save_optimizer_state: true}   # false: no FSDP checkpoints, exactly one HF export at the stop step
```

- `strl plan` prints the layout (mode, FSDP group x replicas, offload, TP x engines, per-GPU train/rollout GB, LoRA params) and
  the recommendation: LoRA -> HSDP within a node (`fsdp_size = gpus_per_node`); full -> full shard when the probe measured
  >= 8 GB/s cross-node, else HSDP + `cpu_offload`. `--network <attempt>/network.json` feeds a measured probe in.
- Any key can be tried without editing YAML: `strl plan|stage <exp> --set compute.training.fsdp_size=8 --set finetune.mode=lora`
  (recorded in `_meta.overrides`, part of the experiment sha).
- LoRA is enabled on the vLLM side by pinned SkyRL itself (`enable_lora`, `max_lora_rank` from `trainer.policy.model.lora.rank`).
  LoRA exports are PEFT adapters: the worker keeps them (`reload/`, `adapter.json` names the base id/revision), checks a
  nonzero `lora_B`, and evaluates a merged copy (`merged/`, W + alpha/r B A, shard by shard).
- Large models: `strl stage-model <exp> --attempt A` renders a CPU job that snapshot-downloads in the pod (16 workers) and uploads
  to the model prefix with a sha256 manifest; `stage` uses it when the model is not local; pods fetch > 10 GB models with 16
  threads + multipart. Sharded HF exports (`model.safetensors.index.json`) are adapted for vLLM shard by shard.

## More nodes (`strl/multinode.py`, `strl/probe.py`; ported from the verified two-node qwen35-grpo run)

- `limits.max_nodes` gates it (default 1); `compute.nodes: N` renders `num_instances: N`, charges `N x gpus/8` node-seconds per
  wall-second (ledger entries carry `nodes`; old entries count as 1), and uses 900 s NCCL / engine-health timeouts (single node
  keeps 300/600). Multi-node programs use their own ledger (e.g. `evidence/ledgers/mimo-27b.json`).
- Rendezvous: the head (JOB_COMPLETION_INDEX 0) writes `leader-network.json` (routable job-interface IP, never link-local) to the
  attempt prefix; peers poll it, match the address family, run the same preflight / SkyRL patches, then `ray start --address`
  (30 tries). The head waits for N nodes x G GPUs on distinct hosts and runs the probe: TCP both ways, NCCL all-reduce, 1 GiB
  all-gather GB/s intra- and cross-node, NCCL transport (NET/OFI efa, IB, Socket), /dev/shm, cgroup memory, GPUs per node.
  EFA (verbs device + aws-ofi-nccl + fi_info) sets `FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1`; its absence only warns.
- Liveness: head `head-heartbeat.json` every 60 s and a Ray watcher that aborts training when a node is lost; peers write
  `peer-<rank>-heartbeat.json`, exit 0 on `completion.json`, nonzero when the head is silent > 300 s (`monitor`: PEER_STALE,
  NETWORK). Each node lists its own checkpoint shards over Ray (nothing is copied to the head).
- LoRA across nodes: SkyRL writes the adapter to a node-local `lora_sync_path` from global rank 0 and then tells every vLLM
  server to load that path, which other hosts cannot read. `strl.runtime.LORA_SYNC_PER_HOST` patches the FSDP worker so the
  first rank on every host writes its copy (the adapter is already all-gathered on every rank), all ranks barrier, then rank 0
  sends the load request.
- Proof ladder: `--mode dry` with nodes 2 (2 x m5.2xlarge: rendezvous, 2-node Ray with placeholder GPUs, CPU probe,
  validate_cfg at 16 GPUs) -> canary (`base/canary_overlay_large.yaml`, keeps the 2 x 8 placement) ->
  `experiments/gsm8k-qwen35-2node-smoke(-full).yaml` -> the large model.

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
