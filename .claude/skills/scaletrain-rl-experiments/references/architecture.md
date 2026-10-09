# Architecture

```
Mac (control venv .venv)                                  ScaleTrain EKS pod (existing SkyRL image, no build)
------------------------                                  ------------------------------------------------
experiments/x.yaml --strl plan--> resolved + warnings
            --strl stage--> S3: payload-<sha>.tar.gz  ---->  bootstrap: verify sha, extract, fetch model files by sha
                                overlays/overlay-<sha>         strl.worker (STRL_MODE dry|canary|train)
                                models/<name>@<rev12>/*          budget.json (write-once) -> credentials (W&B, Modal, LiteLLM)
                     job.json (k8s-preset, never submitted)      -> Ray head (allocated CPUs) -> preflight (real imports,
            --strl launch--> stage + submit ONE job per capacity route (fb/rb/ro/fo), keep the first that starts,
                             cancel + ledger the rest, watch, relaunch (and resume) on preemption                  SkyRL commit, data hashes, reward contract)
                                                                 -> strl.train: overrides -> real parser + validate_cfg
                                                                    -> strl.entrypoint (Ray driver task):
                                                                       SkyRL RayPPOTrainer (GRPO, colocated FSDP + vLLM)
                                                                         generator (env spec chooses):
                                                                           gsm8k        SkyRLGymGenerator -> gsm8k env
                                                                           coding_exec  GroupMaskingGymGenerator -> strl_env
                                                                                        -> Modal executor strl-code-exec
                                                                           coding_harbor HarborGenerator -> Harbor Trial
                                                                                        -> Modal sandbox -> verifier
                                                                         AxisTracking + TrajectoryLogger + EvidenceCallback
                                                                         critic thread (LiteLLM) -> W&B tables/scalars
                                                                 -> watchdog (stall) + heartbeat + 60 s S3 sync
                                                                 -> select on validation -> profile export adapter
                                                                 -> [multi-node] gather every node's exports to S3 + head
                                                                 -> final eval: base + trained concurrently (offline vLLM;
                                                                    LoRA adapter served by vLLM, strl/offline_engine.py)
                                                                    or in-loop paired result (agentic: strl/inloop.py)
                                                                 -> [retention eval: e.g. GSM8K subset, base vs adapter]
                                                                 -> paired bootstrap -> W&B evidence -> completion.json
            --strl monitor--> status line from W&B + S3 + ScaleTrain (alerts: STALL, NAN, ENV_FAILURES, BUDGET)
            --strl verify --> re-score from S3 independently;  --strl compare --> paired deltas between attempts
```

## Package map

| Path | Role |
| --- | --- |
| `strl/experiment.py` | YAML schema, `extends` chains, strict validation (typos, limits, divisibility), content sha |
| `strl/models/` | model axis: `qwen3_5` (GPU-verified), `qwen3` (canary-verified), `llama`/`generic` (preflight) |
| `strl/tasks/` | environment axis: `EnvironmentSpec` + `gsm8k/`, `coding/` |
| `strl/rewards.py` + `strl/tasks/*/rewards.py` | reward axis: named components, weighted sum, `module:function` plugins |
| `strl/envs.py` | `strl_env`: generic SkyRL env = environment scorer + reward components (registered in the Ray task) |
| `strl/overrides.py` | experiment -> SkyRL overrides (common + model profile + environment) |
| `strl/derive.py`, `strl/profiles.json` | `strl plan`: placement, estimates from measured runs, decision-point warnings |
| `strl/stage.py`, `strl/worker.py`, `strl/train.py`, `strl/entrypoint.py` | staging, pod lifecycle, validated SkyRL launch |
| `strl/launch.py` | the capacity race (`strl launch`): routes, winner, losers' ledger entries, preemption relaunch with resume |
| `strl/multinode.py`, `strl/probe.py` | S3 rendezvous, Ray join, liveness watcher; network probe (`network.json`, W&B `infra/*`) |
| `strl/offline_engine.py` | one vLLM engine builder for offline eval (LoRA served directly) |
| `strl/inloop.py`, `strl/evals.py` | paired base-vs-trained from in-loop evals (agentic); frontier-LLM ceiling / reference evals |
| `strl/tasks/mimo/` | the multi-domain agentic suite: data, export, graders, generator, judge, results |
| `strl/tracking.py`, `strl/monitor/` | W&B axis/tables/alerts, critic, live monitor, comparison |
| `strl/services/` | W&B, Modal (credentials, profile), LiteLLM (key, chat) |
| `strl/tasks/coding/harness.py` | hardened unit-test execution (runs inside Modal and inside Harbor sandboxes) |
| `tools/` | vendored ScaleTrain helpers (`scaletrain.py`, `capacity.py`, `lifecycle.py`), `build_overlay.py` (run it as `strl overlay`) |
| `tests/` | contracts incl. golden equivalence to the verified GSM8K commit |

## Why each design choice

- **Golden equivalence**: the GSM8K path is the GPU-verified one (+18.04 points). `tests/test_gsm8k_golden.py` proves the framework
  renders the identical 76 SkyRL overrides, the identical staged job (up to listed normalizations) and identical scorer verdicts on
  266 real completions.
- **Rewards never run in the trainer pod** for coding: untrusted model code executes in Modal (no network, no Modal API, CPU and
  memory limits), and the pass condition requires a per-call nonce printed after the tests (exit tricks never pass).
- **Group masking**: an executor outage must not bias GRPO advantages, so a group with any `infra_error` rollout is masked whole.
- **Generic vs verified reward path**: the default GSM8K reward stays on SkyRL's built-in env (verified); any custom reward design
  switches to `strl_env`, and the dataset's per-row `env_class` column is ignored so the switch actually takes effect.
- **Task metric != reward**: evaluation always reports the environment's yardstick, so reward designs are comparable.
- **Observability never blocks training**: critic on a thread with a bounded queue, INCREMENTAL W&B tables, background S3 sync.
- **Budget**: one program ledger in node-seconds; an attempt is charged `nodes x gpus/8` per wall-second from its pod start (the worker's
  write-once `budget.json` minus 420 s of image pull and bootstrap) to the first terminal observation. Queue time is never charged, so racing
  four routes costs nothing extra; a job is recorded `never_started` only on positive evidence that no pod ran.
- **Capacity race**: no team/borrowing route is reliably fastest (measured), so every launch submits all of them at once.
- **Durable checkpoints + resume** (`evaluation.durable_checkpoints`, `run.resume_from`): SkyRL checkpoints go to S3 so a preempted run
  continues with the same weights, optimizer and data position; the paired result keeps the first attempt's step-0 base.
