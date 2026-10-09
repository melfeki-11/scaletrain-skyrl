# Launch, capacity and budget

## The lifecycle (every attempt, in order)

```bash
.venv/bin/python -m strl plan experiments/<x>.yaml                                       # free: validate + derive + warnings
.venv/bin/python -m strl launch experiments/<x>.yaml --attempt <x>-dry --mode dry --watch      # CPU m5.2xlarge, ~3-6 min, free
.venv/bin/python -m strl launch experiments/<x>.yaml --attempt <x>-can --mode canary --watch   # 2 real updates, small batch
.venv/bin/python -m strl launch experiments/<x>.yaml --attempt <x>-a1 --mode train --watch     # the real run
.venv/bin/python -m strl monitor experiments/<x>.yaml --attempt <winner> --job-id <job>       # live line + alerts (another shell)
.venv/bin/python -m strl verify experiments/<x>.yaml --attempt <winner>
```

`strl launch` stages, submits, races, cancels the losers, follows the winner and records every attempt in the ledger. It prints the
winning attempt name (`<x>-a1-fb`, `-rb`, `-ro` or `-fo`); every later command uses that name. Use `stage` + `submit` only for one-off
jobs (`strl stage-model`, diagnostics).

## Getting GPUs as fast as possible: ALWAYS race the routes

Capacity for one shape is granted per route = (team quota, borrowing). Measured 2026-10-07/08: the same 2 x 8 job waited ~23 h on one
route and started first on another; a CPU job on the research team's quota never started while the same job on the foundation team
started at once; borrowed capacity was reclaimed after 70 s and after 30 min. No route is reliably fastest, so `strl launch` submits
one job per route and keeps the first that starts:

| code | team | borrowing | note |
| --- | --- | --- | --- |
| `fb` | gen_ai.agent_training_foundation | on | the ScaleTrain runbook's default; tried first, wins ties |
| `rb` | gen_ai.agents_research | on | |
| `ro` | gen_ai.agents_research | off | own quota only: cannot be reclaimed by a quota owner |
| `fo` | gen_ai.agent_training_foundation | off | own quota only (often 0 headroom) |

- `--routes fb,ro` restricts the race. Every route uses high priority (never lower it).
- The losers are cancelled the moment the winner leaves the queue and recorded uncharged, on positive evidence that their pods never
  started (no `budget.json` / `worker-started.json`). Two routes starting inside one poll interval: the extra one is charged for the
  seconds its pod was up.
- `--max-wait-hours H --fallback-set compute.nodes=1` (repeatable) gives up on the shape after H hours and races again with the overrides
  as `<attempt>-alt`, e.g. a 2 x 8 run that falls back to 1 x 8 (what the user asked for when 2 nodes were scarce). The fallback starts fresh:
  a checkpoint resumes only at the same world size.
- Names: the launcher creates `<attempt>[-alt][-p<k>]-<route>`; it refuses up front an attempt name that would overflow 26 characters
  before the route suffix (~19 characters is safe with every flag).
- Shapes: only `p5.48xlarge` (8 x H100) and its slices `:4 :2 :1` (one node). A multi-node job needs every node in ONE network domain:
  `status_detail: waiting_for_network_domain_capacity` is topology, not quota, and every route can be blocked at once (`strl capacity
  --nodes 2 --gpus-per-node 8` shows `compatible_nodes` per route). Keep CPU work (prepare, image prebuild, CPU dry runs) going while
  you wait; budget wall-clock, not node-seconds.
- `strl capacity` is read-only and needs the `ml-admin` profile; it explains a wait, the race does not depend on it.

## Preemption (borrowed capacity can be taken back)

ScaleTrain reports it as `QUEUED` + `status_detail: preemption_reported`; the worker records `completion.json` status `preempted`.
A preempted attempt cannot continue in place (its budget window is anchored at its first start). Two defences:
- `strl launch ... --watch --relaunch-on-preempt N` cancels the requeued job, records it, and races again as `<x>-a1-p1`.
- With `evaluation.durable_checkpoints: true` (and `budget.save_optimizer_state: true`) every checkpoint (model, optimizer, data
  position) goes to the attempt's S3 prefix, and the relaunch RESUMES (`run.resume_from=<preempted attempt>` is added automatically):
  same weights, optimizer and step, and the earlier attempt's step-0 evaluation stays the base of the paired result. You can also
  resume by hand: `strl launch <exp> --attempt <x>-r1 --mode train --set run.resume_from=<attempt>`. A resume needs the same world size.
  Cost: a 27B LoRA checkpoint holds the full model state (~7 GB per rank), so use it for long runs on borrowed capacity.

## Budget

- A program ledger (`budget.ledger`, e.g. `evidence/ledgers/platform-hardening.json`) holds total node-seconds and every terminal attempt.
- An attempt is charged `(terminal - pod start) x nodes x gpus/8`, pod start = the worker's write-once `budget.json` `started_at` minus
  420 s (image pull + bootstrap). Queue time is never charged. CPU dry runs are free. (Entries recorded before 2026-10-08 charged from job
  creation.)
- `stage` refuses when the remaining budget cannot cover bootstrap + training + the final-eval/retention reserve; a canary job is capped
  at `budget.canary_max_seconds` + 600 s.
- `strl launch --watch` records the winner and the losers itself. Otherwise, after every terminal attempt:
  `strl ledger <exp> --attempt A --job-id J` (`--never-started` for a job cancelled in the queue).
- `strl plan <exp>` prints the remaining node-seconds of the experiment's ledger and what one train attempt or canary would reserve.
- **New program, new budget.** Only start one when the user authorizes a new GPU budget. Create a ledger file and point
  `budget.ledger` at it, in the experiment or a base:
  ```json
  {"program": "<what and who authorized it, with the date>", "total_seconds": 7200, "nodes": 1, "attempts": []}
  ```
  Never edit or delete recorded attempts.

## Job shape and rules

- Rendered for you: `k8s-preset`, product `agent_training_foundation`, priority `high`, team and borrowing from the route.
- Attempt names: lowercase, <= 26 characters for `launch` minus the `-alt` / `-p<k>` suffixes you enable (checked up front), never reused once submitted.
- `stage` never submits; `submit` writes a receipt before the POST and reconciles by name on any error (never resubmit blindly).
- One stage directory per attempt (`evidence/<attempt>/`): two jobs sharing a directory share a receipt and the second submit is refused.

## Inside one attempt (worker phases)

budget.json (write-once) -> credentials -> [multi-node: S3 rendezvous, Ray join, network probe] -> preflight (real imports, SkyRL commit,
data hashes, reward contract) -> [resume: point SkyRL at the previous checkpoint] -> training child (deadline-bounded; 60 s background
artifact sync; canary rule at step 2; stall watchdog; liveness) -> [multi-node: gather every node's exports to S3 + the head] ->
offline final eval (gsm8k/coding: validation-selected checkpoint, LoRA served by vLLM) or in-loop paired result (mimo) -> [retention eval]
-> W&B evidence + artifact -> completion.json.
