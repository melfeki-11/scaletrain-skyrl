# Troubleshooting (symptoms actually hit while building and verifying this repo)

| Symptom | Cause | Fix |
| --- | --- | --- |
| `experiment ... is invalid: - unknown key algorithm.lrr` | typo in the YAML | `strl plan` lists every problem at once; fix and re-plan |
| `reward component 'pass_all_test' ...` / `does not accept parameter(s)` | wrong component name / param | names come from `strl/tasks/<env>/rewards.py`, `strl/rewards.py` COMMON, or `module:function` |
| a custom GSM8K reward has no effect | row `env_class` overrode the env (SkyRL dataset.py:89) | fixed in `strl.entrypoint.prepare_dataset`; check `environment.env_class=strl_env` in `artifacts/schema.json` |
| `N <split> prompts exceed algorithm.max_prompt_tokens` | SkyRL would silently drop them | raise `max_prompt_tokens` (train/validation only; test prompts are never filtered) |
| `train dataset has N rows after SkyRL prompt-length filtering, expected M` | prompts dropped by SkyRL | same as above; re-run `strl prepare` |
| `reference fails: wrong_answer` for tasks returning `Counter` | harness rejected a stdlib `__eq__` | fixed: only candidate-defined `__eq__` is rejected; `strl prepare` would flag a regression |
| `deployed executor version X != local harness Y` | rolling deploy, old warm container answered | `strl executor deploy` (recreate strategy + polling); results carry `harness_version` |
| `environment/infra_error` > 0 / `masked_group_fraction` > 0 | Modal transport/timeouts | groups are masked (no bias); > 5% means the executor is overloaded: raise `max_containers` or lower `max_env_workers` |
| `executor infrastructure error during validation` | in-loop eval refuses biased metrics | retry the attempt; check Modal status / `strl executor check` |
| `RuntimeError: Canary requires a real nonzero finite policy gradient` | all groups had equal rewards in 2 steps | expected only on tiny/hard setups; `coding_harbor` disables the check (all-zero is legitimate for small agents) |
| `stage`: `FileNotFoundError ... evidence/<attempt>/job.json` | old CLI | fixed (directories are created) |
| `attempt X was already staged against a different budget ledger` | write-once attempt ledger copy | choose a new attempt name |
| `Insufficient remaining budget` | program ledger exhausted for this preset | record terminal attempts (`strl ledger`), use a smaller slice, or ask for budget |
| `capacity`: `not enough distinct ready nodes in one network domain` with large headroom | cluster busy right now | submit anyway (high priority queues first) or a smaller slice; don't chase fluctuations |
| `scaletrain.py submit` TypeError after POST | old helper (datetime in receipt) | fixed (`default=str`); `strl submit` reconciles by name if anything goes wrong |
| `NoCredentialsError` from LiteLLM key resolution on the laptop | no ambient credentials to assume `ml-worker` | `eval "$(aws configure export-credentials --profile ml-admin --format env)"` |
| `aws s3 cp ... AccessDenied` on `scale-ml` | wrong profile | `production-developer` |
| `No module named 'vllm_router'` in the trainer | the image skips it | `strl/runtime.ensure_router` loads `assets/runtime/vllm_router-0.1.14.post1-...whl` (`strl fetch` downloads it) |
| `No module named 'dockerfile_parse'` / `harbor` in the pod | overlay missing | `strl overlay`; `stage` refuses Modal experiments without it |
| Harbor `Unknown parameter: 'session_id'` / `max_tokens is not supported` (external models) | OpenAI-family proxy routes | `strl eval-llm` drops `session_id` and uses `max_completion_tokens` |
| `SyntaxWarning: invalid escape sequence` during `prepare` | MBPP references | silenced; harmless |
| zsh: `set -- $spec` does not split / `==` not found | zsh word-splitting | use bash or Python loops for batch staging |
| `strl monitor` shows `BUDGET` and `node_h` > 1 at step 1 of a 4-GPU attempt | metric bug in attempts staged before 2026-10-06 13:00 (`ab-*-a1`): the program's unused budget was added | fixed: `budget/consumed_node_hours` = nodes x GPU fraction x worker wall-clock. The deadline was always correct; the ledger (`strl ledger`) is the budget source of truth |
| custom `module:function` reward passes locally but `No module named ...` in the pod | module outside the shipped tree | put it in `user_rewards/`; `strl plan` now rejects modules outside the repo and `stage` ships the package |
| Harbor canary/run: reward 0 everywhere, `generate/num_error_trajectories` = batch size, `harbor-trials/*/exception.txt` says `ContextLengthExceededError` | context smaller than Terminus-2's first turn (0.8–2.5k tokens) | `environment.harbor.max_seq_len` (default 8192) sizes the engine, the agent and the trainer together; plan rejects values that cannot fit turn 1; the canary now fails on >= 50% masked groups |
| Harbor W&B eval keys like `eval/_workspace_strl-run_..._mbpp-train-0601/pass_at_1` | upstream `HarborTaskDataset` labels data_source with the task path | fixed: labelled by benchmark (`eval/mbpp/*`, `eval/humaneval/*`) |
