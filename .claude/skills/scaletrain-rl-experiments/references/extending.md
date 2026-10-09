# Extending the three axes

Each recipe ends with the same proof ladder: `python -m strl plan` -> `python -m unittest discover -s tests -t .` ->
`strl stage --mode dry` (CPU, real parser + reward contract + services in the pod) -> `strl stage --mode canary` (1 GPU, 2 real
updates, every W&B table) -> a real run. Do not skip rungs for a new axis value.

## A new model (most models need only YAML)

1. In an experiment: set `model.id` and `model.revision` (a commit sha, never a branch:
   `curl -s https://huggingface.co/api/models/<id> | python3 -c 'import sys,json;print(json.load(sys.stdin)["sha"])'`). Leave
   `model.local_path` and `model.profile` UNSET: the snapshot goes to `assets/models/<name>@<rev12>` with a `strl-model.json` identity
   marker (stage refuses to upload a snapshot whose marker names another model), and the profile is detected from `config.json`.
2. `python -m strl plan <exp>` (reads `config.json` from the hub at the pinned revision when nothing is local; needs network) shows the
   profile, its verification level, LoRA/full layout, memory and warnings (`MODEL never-run`, `MOE`, `MEMORY unknown`, `HARBOR unchecked template`).
3. `python -m strl fetch <exp>` (small models: pinned snapshot to the local dir) or `python -m strl stage-model <exp> --attempt <x>-sm`
   (large models: a CPU job copies HF -> S3 with a sha256 manifest; then `stage` never uploads from the laptop).
4. Profiles shipped (`strl/models/__init__.py` `_ARCH_TO_PROFILE`):

   | Architecture | Profile | Verified |
   | --- | --- | --- |
   | `Qwen3_5ForConditionalGeneration` / `ForCausalLM` (Qwen3.5, Qwen3.6, Qwen3.8) | `qwen3_5` (language_model_only, wrapped export) | gpu |
   | `Qwen3_5Moe*` | `qwen3_5_moe` | none |
   | `Qwen3ForCausalLM` | `qwen3` | canary |
   | `Qwen3MoeForCausalLM` | `qwen3_moe` | none |
   | `LlamaForCausalLM` | `llama` | preflight |
   | Qwen2, Mistral, OLMo/OLMo2, Phi3 (Phi-4-mini), Gemma2/3 text, Granite, Cohere2, GLM4 | `dense` | none (Phi-4-mini: see validation.md) |
   | any other `*ForCausalLM` | `generic_causal_lm` | none |
   | an unmapped `*ForConditionalGeneration` (a VLM wrapper, e.g. Gemma3 VLM) | REFUSED by plan | - |

   A forced `model.profile` that contradicts the architecture is refused. Gated models (Llama, Gemma) need the license accepted on HF.
5. If the architecture needs special handling, add `strl/models/<name>.py` with a `ModelProfile` subclass and register it in
   `strl/models/__init__.py` (`_PROFILES` + `_ARCH_TO_PROFILE`, or `register_profile(name, 'module:Class', architectures)`). Hooks:
   - `trainer_overrides` (FSDP wrap class, packing, language_model_only),
   - `generator_overrides` (vLLM engine kwargs),
   - `chat_kwargs` (e.g. `enable_thinking=False`), `vllm_kwargs` (offline eval),
   - `prepare_export_for_vllm` (if SkyRL's HF export does not reload in vLLM as is; see `qwen3_5_export.py`),
   - nothing for bundled remote code: when `config.json` has `auto_map` and transformers knows the architecture, the worker uses the
     native class (`strl.models.prefer_native_code`; SkyRL itself always passes `trust_remote_code=True`),
   - set `verified` honestly (`preflight` until a canary passes, `canary`, then `gpu` after a measured held-out improvement).
6. Agentic/Harbor environments: run `strl check-template <exp>`. SkyRL merges a trajectory's turns into one training sequence only if
   `prompt_i + response_i` is a token prefix of `prompt_i+1`; templates that render history differently from the generation prompt
   (Qwen3.5 drops `<think>\n\n</think>\n\n` from history) get one sequence per turn: several times more training tokens. Measure
   on real transcripts before trusting it (Qwen3.5: 811 turns -> 811 sequences; Qwen3.8, thinking off: -> 209, 86% of merge points).
   Fix with a template that keeps history verbatim, not by changing the trainer. Measured merge on the probe: Phi-4, OLMo2, Qwen2.5 merge;
   Mistral and DeepSeek-R1 distills do not.
7. Terminal EOS markers differ by family (`<|im_end|>` Qwen, `<|eot_id|>` Llama, `<|end|>` Phi, `</s>` Mistral): the GSM8K scorer strips
   the known list (`strl/tasks/gsm8k/answers.py` `TERMINAL_TOKENS`); add a new family's marker there.
8. Ladder for a new model: plan -> dry -> canary on the smallest GPU slice that fits (LoRA) -> run. Add a row to `strl/profiles.json` from the
   canary (step seconds, tokens/s, memory peak) so later plans are calibrated.
9. Add the model to `limits.models` only if your config uses an allowlist (the default is `["*"]`).

## A new reward design (YAML, or one Python function)

```yaml
reward:
  components:
    - {fn: pass_all_tests, weight: 1.0}        # the yardstick term
    - {fn: partial_tests, weight: 0.5}         # shaping
    - {fn: length_penalty, weight: 1.0, max_tokens: 300}
    - {fn: 'user_rewards.examples:docstring_bonus', weight: 1.0, bonus: 0.1}   # your own code (quote: YAML and ':')
```

- A component is `fn(result: dict, ctx: RewardContext, **params) -> float`. `result` is the environment scorer output
  (coding: `parsed` (the extracted code, up to 2,000 chars), `failure_class, passed, total, compiled, has_code, suspicious,
  correct, completion`; gsm8k: `parsed` (the extracted answer), `correct, completion`).
- Your own code goes in `user_rewards/<module>.py` (worked example: `user_rewards/examples.py` +
  `experiments/mbpp-qwen35-custom-reward.yaml`). `strl stage` ships the top-level package of every `module:function` component
  in the payload; `strl plan` rejects modules outside the repo tree (stdlib, the local venv, `tests/`), which could not run in
  the pod. A `module:function` term is logged under its full name, e.g. `environment/reward_components/user_rewards.examples:docstring_bonus`.
- Names resolve: environment components (`strl/tasks/<env>/rewards.py`), common (`length_penalty`, `format_ok`), `module:function`.
- `strl plan` rejects unknown names and unknown parameters. Each term is logged as `environment/reward_components/<fn>`.
- `coding_harbor` rewards come from the sandbox verifier: only `pass_all_tests` there; explore designs with `coding_exec`.
- `mimo`: the reward starts from the native domain grader. `reward.components` must contain `{fn: verifier_score}` exactly once (its weight scales
  the native score) and may add custom `module:function` terms (not the common `length_penalty`/`format_ok`, which need a completion string). Terms
  shape the TRAINING reward only: evaluation, `eval/<d>/task_metric` and `test/*` always use the native score, and `generate/<d>/reward_native_mean`
  next to `generate/<d>/reward_mean` shows whether the shaping is being gamed. A term receives `result = {domain, task, reward (native), metric,
  passed, components (the grader's sub-scores), turns, tokens, seconds, stop, artifact (the agent's deliverable, truncated)}`; the worked example is
  `user_rewards/examples.py:turn_efficiency_bonus` in `experiments/mimo-example-music-code-full.yaml`.
- A custom GSM8K reward automatically switches from SkyRL's built-in `gsm8k` env to `strl_env` (same scorer, proven equivalent on
  266 real completions in `tests/test_gsm8k_golden.py`).

## A new environment (one Python package)

1. `strl/tasks/<name>/spec.py`: subclass `EnvironmentSpec` (`strl/tasks/base.py`). Required:
   - data: `prepare_data`, `assert_manifest`, `data_files`, `validation_size`, `final_eval_ids` (stable ids, deterministic order);
   - scoring: `score_result(action, extras, cfg) -> dict` (the reward components consume it), `parse`, `failure_mode`;
   - wiring: `skyrl_overrides` (at least `environment.env_class`), `register` (call `strl.envs.register_generic_env()`),
     optionally `build_generator` (multi-turn / agentic) and `skyrl_config_class`;
   - evidence: `generator_contract` (exercise the REAL reward path on known-good and known-bad inputs; it runs in the CPU dry run),
     `dry_run_checks` (external services from the pod), `eval_record` (validation rows), `eval_module` (offline final eval).
   - `metric`: the yardstick name; `failure_classes`: logged as `environment/failure/<class>`.
2. Register it in `strl/tasks/__init__.py` (`_REGISTRY`), add `base/<name>.yaml` (extends `base/infra.yaml`) and an experiment.
3. If scoring executes untrusted output, do it remotely (pattern: `strl/tasks/coding/executor.py` + `harness.py`): never in the
   trainer pod, nonce-gated passes, resource limits, group masking for infrastructure failures (`strl/tasks/coding/generator.py`).
4. Planned next domains reuse the same pieces: reasoning (gsm8k-style scorer), deep research / browser (Harbor tasks or a judge via
   `strl/services/litellm.py`), CUA (Harbor + a GUI sandbox image).


## A new suite of verifiable agentic tasks (the `mimo` pattern; any verl-format dataset)

The MiMo environment (`strl/tasks/mimo/`) is the template for "many domains, one trainer". Contract in `strl/tasks/mimo/__init__.py`.

1. **Per domain, a grader** (`graders/<domain>.py`): `build(row, task_dir)` writes `tests/` + Dockerfile pieces; `grade(...)` is a pure function
   unit-tested on temp dirs (no Modal). The in-sandbox `tests/test.sh` writes `reward.txt = 0` FIRST, then grades, and writes
   `details.json` ({metric, components, infra_error}). Judge-scored parts go to `judge_request.json` (kind rubric | vision;
   `force_zero` when a gate failed) and are scored OUTSIDE the sandbox: the key must never enter one.
2. **Anti-tamper by construction**: the agent may be root. Re-verify from material it cannot touch: tests uploaded at verify time,
   pristine binaries captured at export time (cyber), the base commit captured at export time and test files reset before applying the
   patch (code), MCP servers run as another user (general), reference solutions never in the task dir. Nop must score 0; tampering must score 0.
3. **Export** (`export.export_task(row, dest, split=)`): `dest` IS the task dir (`tasks/<split>/<domain>-<id>`); deterministic; Dockerfile
   `FROM` the published image + static tmux + an explicit `CMD` where the image has none; `strl.json` (agent, max_turns, max_tokens, artifacts,
   judge); `task.toml` per-domain timeouts/CPU/memory (no global `override_*`).
4. **Data** (`data.py`): runtime selection per domain (`train`, `test`, `weight`, `filters`); fixed held-out test sets grouped by a leak-free key
   (project, repo, scenario, tag), hashed in the manifest, independent of the training selection.
5. **Generator**: subclass of SkyRL's HarborGenerator; per-task trial config (strl.json + task.toml), no retries on agent/verifier timeouts,
   step deadline for stragglers (masked, counted), infra/judge failures MASK (never score 0), per-domain metrics.
6. **Prove it on real tasks before GPU**: `strl prepare <exp>` on the smoke experiment, `tools/gate_mimo.py <domain> <task dirs>` (nop = 0, no
   infra error), `--mode dry` (judge selftest, nop from the pod, validate_cfg), then the 1-GPU canary. Unit tests alone missed: double-nested export
   dirs, sandboxes that exit at once, non-ASCII paths, a vision request shape that masked every trial, an ignored `force_zero`.


## Starting a new MiMo program (what must not be shared with another experiment)

`mimo-example-music-code-full.yaml` is the template. Three things are inherited from `base/mimo.yaml` and silently shared unless you change them:
- `budget.ledger` (`evidence/ledgers/mimo-27b.json` is the authorized 27B program's node-second budget: a new program gets its own file, created with
  the user's authorization);
- `run.data_dir` (the exported tasks and manifest: a different selection needs a different directory, then `strl prepare <exp>`);
- `services.modal.credentials_uri` (a locator the user approved for this project; confirm it before a new project reuses it).
Switching a domain off takes both `train: 0` and `test: 0` (`train: 0` alone keeps its held-out set as an evaluation-only transfer probe). For a
small model set `inference.tensor_parallel_size: 1`, `environment.harbor.max_seq_len` (16k is enough for a 2B), and size `compute.training.fsdp_size`
to the GPUs of one node; read the plan's UNCALIBRATED step estimate as a placeholder and let a canary measure it.
Every file under `experiments/` must plan: the test suite loads them all.


## A new MiMo-style suite: what is generic and what is MiMo-specific (honest list)

Generic (reuse as is): the Harbor generator subclass (`strl/tasks/mimo/generator.py`: placeholders for timed-out trials, deadline
masking, per-domain metrics, shaping terms, judge hook), the judge (`judge.py`: rubric + vision, votes, variance, outside the sandbox), the
in-loop paired result (`strl/inloop.py`), `honest_power` statistics, the hardened shared `test.sh`, the agents (`SingleShotAgent`,
Terminus-2), monitoring flags, retention, the ceiling (`strl/evals.py` `mimo()`), fail-fast.

MiMo-specific (copy and adapt, do not import): `data.py` (the verl parquet columns, the pinned dataset revision, per-domain grouping keys),
`export.py` (Dockerfile tails per domain), `graders/<domain>.py`, the `DOMAINS` tuple, and `base/mimo.yaml`. For a new suite:
1. `cp -r strl/tasks/mimo strl/tasks/<suite>`; rename the env (`name = '<suite>'`), register it in `strl/tasks/__init__.py`.
2. Rewrite `data.py` for the suite's columns and its leak-free grouping key per domain; keep the manifest hashes.
3. Write one grader per domain (contract above); unit-test `grade()` on temp dirs; nop and tamper must score 0.
4. `base/<suite>.yaml` extending `base/infra.yaml` with your `domains`, caps and judge; a smoke experiment with 2-4 tasks per domain.
5. Ladder: unit tests -> `strl prepare` -> `tools/gate_mimo.py` style nop gate per domain -> dry (with `run.ceiling_per_domain=2`)
   -> 1-GPU canary (0.8B LoRA) -> the run.
