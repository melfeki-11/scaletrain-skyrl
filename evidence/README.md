Per-attempt receipts land in `evidence/<attempt>/`: `stage.json`, `job.json`, `submission.json` and downloaded results. They are
gitignored. The committed files are:

- `ledgers/rl-envs.json`: the program budget ledger for the rl-envs framework runs (node-seconds, with `gpu_fraction` per
  attempt). `strl stage` reads it and `strl ledger` appends to it.
- `budget-ledger.json`: the ledger of the original GSM8K program (attempt `q35v2-a1`).
- `verified-result.json`, `a1-submission.json`, `a1-wandb-readback.json`: the independently verified GSM8K result.
- `gates/`: local gate outputs (executor load test, Harbor oracle/nop, LiteLLM reference evals).
- `results/`: copies of `strl verify` outputs (`<attempt>.json`, re-scored in Modal) and of the `strl compare` report for the coding A/B.
