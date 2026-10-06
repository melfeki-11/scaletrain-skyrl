# Qwen3.5-0.8B · SkyRL GRPO · GSM8K · one 8×H100 node on ScaleTrain

Full-parameter GRPO of `Qwen/Qwen3.5-0.8B` on GSM8K with [SkyRL](https://github.com/NovaSky-AI/SkyRL), run as a single
ScaleTrain job on one `p5.48xlarge` (8×H100) within a hard one-hour allocation budget. The job trains, selects a checkpoint
on a train-derived validation split, then compares base vs trained on the untouched official GSM8K test split (1,319
paired examples) and logs examples the base model got wrong and the trained model got right.

This repo is the code only. Launching, capacity checks and submission use the (not included) `scaletrain-services` Claude
Code skill's helpers (`scripts/run.py capacity|scaletrain|lifecycle`); any equivalent `scale-train` CLI flow works.

## Result (job `job_db2hb8sf8c4g07jroc60`, COMPLETED; details and examples in [RESULTS.md](RESULTS.md))

| Official GSM8K test, 1,319 paired examples | base | trained (36 GRPO updates) |
| --- | --- | --- |
| accuracy | 50.80% (670) | **68.84% (908)** |

+18.04 points, paired 95% bootstrap CI [+15.31, +20.77]; 312 problems fixed, 74 broken. Validation (train-derived) rose 51.6% → 72.7%.
One run, 8×H100, ~46 min of worker time (≤ ~3,010 s of the 3,600 s allocation), median 51 s/step, ~3.7k tokens/s/GPU.
Efficiency is provisional: only this configuration was measured. Rollout generation (35 s of a 51 s step, eager vLLM) is the bottleneck.

## Design (what makes it efficient)

| Choice | Why |
| --- | --- |
| 1 node × 8 GPUs, 8 colocated TP1 vLLM engines + 8-rank FSDP policy/ref | A 0.8B model needs no cross-node traffic; removes Ray/NCCL rendezvous and custom placement |
| 64 prompts × 8 samples per step = 512 rollouts (64/GPU), 1 on-policy update per step | Even divisibility, no off-policy staleness |
| Time-bounded training (`GRPO_TRAIN_DEADLINE`), not step-bounded | Uses all of the budget except a reserved final-eval window |
| Checkpoint/HF export every 20 steps + at the deadline; S3 sync runs in a background loop and skips optimizer shards | Nothing blocks the training loop on uploads |
| Lightweight 2-step canary inside the same job (finite, nonzero gradient) | No separate GPU canary job |
| Final test eval: base on GPUs 0-3 and trained on GPUs 4-7 concurrently, 4 shards each | ~1 model-load + 1 generation round instead of 4 sequential ones |
| CPU-only preflight job (`m5.2xlarge`) against the exact 8-GPU overrides | Parser, `validate_cfg`, generator checks, reward/metrics contracts and imports are verified before any GPU is requested |

## Layout

- `config.yaml` – every pin and hyper-parameter (model/dataset revisions, SkyRL commit, image, batch, budget, W&B, S3).
- `q35/` – worker and library: `worker.py` (single-node supervisor), `skyrl.py` (SkyRL override builder), `entrypoint.py` /
  `tracking.py` (W&B forwarding, deadline-aware callback, diagnostics), `runtime.py` + `ipv6_patch.py` (router overlay,
  EOS-aware reward and metrics-port fixes), `evaluate.py` (sharded vLLM test eval), `export_adapter.py` (Qwen wrapper config
  for vLLM reload), `finalize.py`, `stats.py`, `verify.py`, `data.py`, `answers.py`.
- `stage.py` – uploads hash-pinned payload/model to S3 and renders `job.json`. `runtime_probe_stage.py` – renders the CPU preflight job.
- `scripts/fetch_assets.py` – downloads the pinned model, builds the pinned dataset splits, fetches the hash-checked router wheel.
- `tests.py` – local contract tests (`python tests.py`). `pinned/` – upstream SkyRL source snapshots used by static schema tests.

## Run order

```bash
python -m pip install -r requirements-local.txt
python scripts/fetch_assets.py                    # model + GSM8K splits + router wheel (untracked)
python tests.py                                   # local gates
SKILL=<path to scaletrain-services skill>
# 1) CPU preflight in the training image (no GPU budget)
python runtime_probe_stage.py --name q35v2-preflight-1
python $SKILL/scripts/run.py scaletrain validate runtime-check-job.json
python $SKILL/scripts/run.py scaletrain submit runtime-check-job.json --user-id <you> --receipt evidence/preflight-submission.json
# 2) capacity (foundation→research without borrowing, then with borrowing)
python $SKILL/scripts/run.py capacity --nodes 1 --gpus-per-node 8 --output evidence/capacity.json
# 2b) optional CPU rehearsal of bootstrap/budget/services/Ray (no training)
python stage.py --attempt q35v2-dry-1 --dry-run --output dry-job.json   # then validate + submit dry-job.json as above
# 3) stage + submit the training job; budget comes from evidence/budget-ledger.json
python stage.py --attempt q35v2-a1 [--team gen_ai.agents_research] [--allow-borrowing]
python $SKILL/scripts/run.py scaletrain validate job.json
python $SKILL/scripts/run.py scaletrain submit job.json --user-id <you> --receipt evidence/submission-a1.json
python $SKILL/scripts/run.py scaletrain get <job_id>
```

After the job completes: `python scripts/verify_results.py --attempt q35v2-a1 --job-id <id> --wandb-run <entity/project/run_id>`
re-downloads the outputs, re-scores them against the local test parquet, recomputes the paired bootstrap and writes `RESULTS.md`.
Note: `scale-train submit` via the helper may raise a `TypeError` after the job was created; reconcile with `scaletrain list` by exact job name and never resubmit blindly.

After a terminal attempt, append its allocated/terminal timestamps to `evidence/budget-ledger.json` before staging another one.

## Known pitfalls baked into the code

- The strict reward must strip only the terminal `<|im_end|>`/`<|endoftext|>` marker; otherwise every reward is 0.
- Qwen3.5 hybrid (GDN) layers: `remove_microbatch_padding=false`, `language_model_only=true`, wrap `Qwen3_5DecoderLayer`.
- `generator.batched=true` is rejected with `chat_template_kwargs` (needed for `enable_thinking=false`) only at engine start; use `batched=false`.
- The image lacks `vllm-router==0.1.14.post1`; `q35.runtime` loads the hash-pinned wheel from a writable overlay.
- Ray metrics discovery must skip port `-1` and bracket IPv6; both are patched.
- SkyRL's HF export is a text-only config; vLLM needs the original wrapper config around it (`export_adapter.py`). Existing `model.language_model.*` keys must not be re-prefixed.
- Advertise only the allocated CPUs to Ray (`q35/resources.py`), not the host's.
- Every secret is a locator (S3 object / secrets manager). Nothing in this repo contains credential values.
