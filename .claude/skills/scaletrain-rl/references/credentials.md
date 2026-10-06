# Credentials and access

Nothing in the repo, the job record, the payload or W&B contains a secret value: only locators. Never paste a key into chat, YAML,
argv, job tags or W&B config. If a credential is missing or expired, stop and ask the user; never switch to another key/project.

| What | Where it lives | Who reads it | Check |
| --- | --- | --- | --- |
| AWS SSO (S3, staging) | profile `production-developer` (account 307185671274; bucket `scale-ml`) | `strl stage`, `strl verify/compare/monitor` | `aws sts get-caller-identity --profile production-developer` |
| AWS SSO (EKS capacity) | profile `ml-admin` (account 692474966980) | `strl capacity` | same with `--profile ml-admin` |
| ScaleTrain CLI | `scale-train` (0.10.9), `~/.scaletrain/config`, user id `mohamed.elfeki` | `tools/scaletrain.py` (headless; resolves the CLI's own Python) | `scale-train --version`; `python -m strl status --list` |
| W&B key | encrypted S3 locator `tracking.credentials_uri` (an existing approved object) | the worker (pod role), scoped to `WANDB_BASE_URL=https://scaleai.wandb.io` | dry run gate `wandb` (write + readback) |
| Modal tokens (pod) | encrypted S3 bundle `services.modal.credentials_uri` = `s3://scale-ml/tmp/mohamedelfeki/scaletrain/modal-harbor/private/mh-cpu-261006a-modal.env` (user asked to KEEP and reuse it; do not delete or overwrite without asking) | worker -> mode-600 profile TOML; only `MODAL_CONFIG_PATH/PROFILE/ENVIRONMENT` reach Ray | dry run gate `modal` |
| Modal tokens (Mac) | mode-600 `ScaleTrain/.env.modal` (`MODAL_TOKEN_ID/SECRET`), or `$STRL_MODAL_ENV_FILE` | `strl.services.modal.local_profile` (temp profile, deleted after use) | `python -m strl executor check` |
| Modal sandbox secret | Modal Secret `modal-proxy-user` in environment `main` | Harbor sandboxes (`coding_harbor`) | dry run gate `modal_secret` |
| LiteLLM key | `team/GENAIML/secret-store-key` field `HIL_BENCH_EVAL_1`, reached by assuming `arn:aws:iam::307185671274:role/ml-worker` | critic (worker writes a mode-600 key file), `strl eval-llm`, `strl critic` | dry run gate `critic` |
| LiteLLM billing | project ObjectId `696011502a26dc4a64b24f55` (`services.litellm.project_id`), sent as `user` + `metadata.tags` | every LiteLLM call | budget caps: `tracking.critic.budget_usd` |
| GitHub | `gh` authenticated as the repo owner | PRs | `gh auth status` |

Laptop LiteLLM calls need ambient credentials to assume `ml-worker` (there is no `ml-worker` profile locally):

```bash
eval "$(aws configure export-credentials --profile ml-admin --format env)" && .venv/bin/python -m strl eval-llm base/coding_exec.yaml --limit 20
```

Gotchas that cost real time:
- Pinned SkyRL copies `MODAL_TOKEN_ID/SECRET` into Ray `runtime_env` if they are set: the worker unsets them; `strl.entrypoint`
  forwards only an allowlist and strips them again.
- `ml-admin` gets AccessDenied on `s3://scale-ml/...`; staging uses `production-developer`.
- W&B alerts (`wandb.alert`) only notify if alerts are enabled in the user's W&B settings (Settings -> Alerts).
- Uploading a new credential bundle to S3 is a credential action: ask the user first.
