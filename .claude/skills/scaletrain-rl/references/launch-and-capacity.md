# Launch, capacity and budget

## The lifecycle (every attempt, in order)

```bash
.venv/bin/python -m strl plan experiments/<x>.yaml                          # free: validate + derive + warnings
.venv/bin/python -m strl capacity --nodes 1 --gpus-per-node 8,4,1             # read-only routes (foundation -> research, no borrowing first)
.venv/bin/python -m strl stage experiments/<x>.yaml --attempt <x>-dry1 --mode dry      # CPU m5.2xlarge, ~3-5 min
.venv/bin/python -m strl submit evidence/<x>-dry1/job.json
.venv/bin/python -m strl stage experiments/<x>.yaml --attempt <x>-can1 --mode canary   # 1 GPU, 2 updates, <= 30 min
.venv/bin/python -m strl submit evidence/<x>-can1/job.json
.venv/bin/python -m strl stage experiments/<x>.yaml --attempt <x>-a1 --mode train      # the real run
.venv/bin/python -m strl submit evidence/<x>-a1/job.json
.venv/bin/python -m strl monitor experiments/<x>.yaml --attempt <x>-a1 --job-id <job_id>
.venv/bin/python -m strl ledger experiments/<x>.yaml --attempt <x>-a1 --job-id <job_id>   # after the terminal state
.venv/bin/python -m strl verify experiments/<x>.yaml --attempt <x>-a1
```

- Attempt names: unique, lowercase, <= 32 chars. Re-staging an unsubmitted attempt is allowed; a submitted name is never reused.
- `stage` never submits. `submit` writes a receipt before the POST and refuses an existing name; if it reports an error, it
  reconciles by name (`reconciled_by_name`) - never resubmit blindly.
- Job shape (rendered for you): `k8s-preset`, team `gen_ai.agent_training_foundation`, product `agent_training_foundation`,
  priority `high`, `allow_borrowing: false` unless `--allow-borrowing`. Presets: `m5.2xlarge` (dry), `p5.48xlarge` (8 H100),
  `p5.48xlarge:4|:2|:1` (slices; one node).
- `--team gen_ai.agents_research` when capacity says foundation has no room; borrowing only after both non-borrowing routes.

## Capacity reading

`strl capacity` reports quota headroom AND whether any ready node in one network domain can fit the shape. "not enough distinct
ready nodes" with large headroom means the cluster is busy right now: submit anyway (high priority queues first) or try smaller
slices. A queued job is not a failure; never cancel/resubmit to chase small fluctuations.

## Budget

- A program ledger (`budget.ledger`, e.g. `evidence/ledgers/rl-envs.json`) holds total node-seconds and every terminal attempt.
- An attempt is charged `(terminal - created) x gpus/8` (conservative: includes queue + pod start). CPU dry runs are free.
- `stage` refuses when the remaining budget cannot cover bootstrap + training + final-eval reserve; canaries are capped at 1,800 s.
- After every terminal attempt: `strl ledger <exp> --attempt A --job-id J` (it refuses non-terminal jobs and duplicates).

## Inside one attempt (worker phases)

budget.json (write-once) -> credentials -> Ray (allocated CPUs, not the host's) -> preflight (real imports, SkyRL commit, data
hashes, reward contract) -> training child (deadline-bounded; canary file must appear in time; stall watchdog) -> validation-only
checkpoint selection -> export adaptation (model profile) -> FSDP shard + parameter-change checks -> final eval (base and trained
concurrently, half the GPUs each) -> paired bootstrap -> W&B evidence + artifact -> completion.json.
