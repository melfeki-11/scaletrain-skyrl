# Launch, capacity and budget

## The lifecycle (every attempt, in order)

```bash
.venv/bin/python -m strl plan experiments/<x>.yaml                          # free: validate + derive + warnings
.venv/bin/python -m strl capacity --nodes 1 --gpus-per-node 8,4,1             # read-only routes (foundation -> research, no borrowing first)
.venv/bin/python -m strl stage experiments/<x>.yaml --attempt <x>-dry1 --mode dry      # CPU m5.2xlarge, ~3-5 min
.venv/bin/python -m strl submit evidence/<x>-dry1/job.json
.venv/bin/python -m strl stage experiments/<x>.yaml --attempt <x>-can1 --mode canary   # 1 GPU, 2 updates, job cap canary_max_seconds + 600 s
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
- `stage` refuses when the remaining budget cannot cover bootstrap + training + final-eval reserve; a canary job is capped at `budget.canary_max_seconds` + 600 s (1,800 s by default; 3,000 s for Harbor, whose base sets 2,400), and must write `canary.json` within `canary_max_seconds` of the child start.
- After every terminal attempt: `strl ledger <exp> --attempt A --job-id J` (it refuses non-terminal jobs and duplicates).
- `strl plan <exp>` prints the remaining node-seconds of the experiment's ledger and what one train attempt or canary would reserve.
- **New program, new budget.** Only start one when the user authorizes a new GPU budget. Create a ledger file and point
  `budget.ledger` at it, in the experiment or a base:
  ```json
  {"program": "<what and who authorized it, with the date>", "total_seconds": 7200, "nodes": 1, "attempts": []}
  ```
  `strl ledger` appends attempts of the form `{job_id, attempt, status, allocated_at, terminal_at, gpu_fraction, evidence}`.
  Never edit or delete recorded attempts.

## Inside one attempt (worker phases)

budget.json (write-once) -> credentials -> Ray (allocated CPUs, not the host's) -> preflight (real imports, SkyRL commit, data
hashes, reward contract) -> training child (deadline-bounded; canary file must appear in time; stall watchdog) -> validation-only
checkpoint selection -> export adaptation (model profile) -> FSDP shard + parameter-change checks -> final eval (base and trained
concurrently, half the GPUs each) -> paired bootstrap -> W&B evidence + artifact -> completion.json.
