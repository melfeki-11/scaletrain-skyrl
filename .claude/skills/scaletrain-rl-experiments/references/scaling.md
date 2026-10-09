# Scaling up

Verified status (2026-10-08): one node at 1-8 H100 with slices, for Qwen3.5-0.8B, Qwen3-1.7B, Phi-4-mini and Qwen3.8-27B (LoRA, TP2 x 4 engines,
62 GB peak). Two nodes x 8 H100 for Qwen3.5-0.8B LoRA (16-rank NCCL, LoRA sync across hosts, export gathered from every node, 2 updates).
Two nodes x 8 for Qwen3.8-27B up to the step-0 eval only (load over 16 GPUs, 8 TP2 engines). Never GPU-run: full fine-tuning above 0.8B
on 2 nodes, more than 2 nodes, MoE models, Megatron, the fully-async trainer. Each new rung needs the proof ladder
(dry -> canary -> bounded run) before a long run.

## Bigger models on one node

- Change `model.*`; `strl plan` shows the memory estimate per phase. Above ~4B full-parameter on 8 H100, raise
  `inference.tensor_parallel_size` (engines become gpus/TP; plan re-derives) and lower `micro_train_per_gpu`.
- Long contexts: `max_prompt_tokens + max_generate_length` is the vLLM `max_model_len`; memory grows with it.
- MoE (`Qwen3MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration`) has profiles (`qwen3_moe`, `qwen3_5_moe`) and the plan counts
  TOTAL parameters for memory (all experts are resident) with a `MOE` warning; FSDP over experts works but is slow and unverified.
  Expert parallelism needs Megatron (`trainer.strategy=megatron` + `megatron_config.*`), SkyRL-supported but not wired here.
- Memory: `strl plan` estimates weights + optimizer + KV cache, and prints the MEASURED peak from `strl/profiles.json` when a run of the
  same model exists (`memory: ... measured peak 62 GB`). With no measurement it warns `MEMORY unknown`: the 1-node canary is the check.

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
  nonzero `lora_B`, and the offline evaluator SERVES the adapter on the frozen base through vLLM (`enable_lora` + `LoRARequest`,
  `strl/offline_engine.py`; `served: vllm-lora` in the result), exactly as in training: no bf16 merge, no rounding drift.
- Large models: `strl stage-model <exp> --attempt A` renders a CPU job that snapshot-downloads in the pod (16 workers) and uploads
  to the model prefix with a sha256 manifest; `stage` uses it when the model is not local; pods fetch > 10 GB models with 16
  threads + multipart. Sharded HF exports (`model.safetensors.index.json`) are adapted for vLLM shard by shard.

## Mixture-of-experts (MoE) models (planned and dry-run proven, not GPU-run)

Measured by a fresh agent on Qwen3-30B-A3B (61 GB bf16, `qwen3_moe` profile), full fine-tuning on coding_exec, 2 x 8 (`fd-dry-q3moe-a-fb`
dry run passed: 2-node Ray, native `Qwen3MoeForCausalLM`, validate_cfg with fsdp_size 8 + cpu_offload + TP2):
- Memory counts every expert (all are resident). TP1 does not fit the vLLM weights (KV room negative): use TP2+ (the offline final eval uses
  the same TP). Full FT fits only with HSDP 8 + `cpu_offload` (~27-31 GB/GPU, ~45 GB host RAM per GPU); without offload the FSDP layer
  buffers push it over the limit.
- HSDP 8 x 2 has the SAME per-GPU memory as 1 node with full shard 8: the second node buys only data-parallel/rollout throughput and costs
  ~22 s per optimizer step of cross-node gradient all-reduce over sockets (plan prints the estimate). Start on 1 x 8; go to 2 nodes when a
  canary shows the step is generation-bound.
- No expert parallelism (that needs Megatron); HF MoE layers loop over experts in Python, so training steps are slow. LoRA on expert linears
  through the image's vLLM is unverified: canary it before relying on it.

## More nodes (`strl/multinode.py`, `strl/probe.py`; ported from the verified two-node qwen35-grpo run)

- `limits.max_nodes` gates it (default 1); `compute.nodes: N` renders `num_instances: N`, charges `N x gpus/8` node-seconds per
  wall-second (ledger entries carry `nodes`; old entries count as 1), and uses 900 s NCCL / engine-health timeouts (single node
  keeps 300/600). Multi-node programs use their own ledger (e.g. `evidence/ledgers/mimo-27b.json`).
- Rendezvous: the head (JOB_COMPLETION_INDEX 0) writes `leader-network.json` (routable job-interface IP, never link-local) to the
  attempt prefix; peers poll it, match the address family, run the same preflight / SkyRL patches, then `ray start --address`
  (30 tries). The head waits for N nodes x G GPUs on distinct hosts and runs the probe: TCP both ways, NCCL all-reduce, 1 GiB
  all-gather GB/s intra- and cross-node, NCCL transport (NET/OFI efa, IB, Socket), /dev/shm, cgroup memory, GPUs per node.
  EFA (verbs device + aws-ofi-nccl + fi_info) sets `FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1`; its absence only warns.
- Exports and artifacts on multi-node: SkyRL writes checkpoint shards and the HF export on whichever node holds the rank; before Ray stops
  the head runs `gather_exports` (every node uploads its `exports/` to S3, the head downloads them; `identity.exports_gathered`). A 60 s
  background sync on every node never raises into training.
- Network: the probe's `network.json` (transport, GB/s) feeds `strl plan --network` and W&B `infra/*`. Measured on p5 pairs:
  NCCL over TCP sockets at ~0.69 GB/s cross-node (the image lacks aws-ofi-nccl although the pods expose EFA devices). That is why
  multi-node LoRA defaults to HSDP (`fsdp_size = gpus_per_node`, automatic): only the adapter gradient crosses nodes. Full fine-tuning
  over sockets needs HSDP + `cpu_offload` and will be slow; installing aws-ofi-nccl in the image is the real fix (not done).
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
