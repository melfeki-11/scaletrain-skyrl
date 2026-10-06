# Extending the three axes

Each recipe ends with the same proof ladder: `python -m strl plan` -> `python -m unittest discover -s tests -t .` ->
`strl stage --mode dry` (CPU, real parser + reward contract + services in the pod) -> `strl stage --mode canary` (1 GPU, 2 real
updates, every W&B table) -> a real run. Do not skip rungs for a new axis value.

## A new model (most models need only YAML)

1. In an experiment: set `model.id`, `model.revision` (a commit sha, never a branch) and `model.local_path: assets/models/<name>`.
2. `python -m strl fetch <exp>` (downloads the pinned snapshot; data is reused if its manifest verifies).
3. `python -m strl plan <exp>` shows the detected profile and its verification level. Detection reads `config.json` from
   `model.local_path`; when that snapshot is missing, plan fetches only `config.json` from the HF hub at the pinned revision
   (needs network). Offline and unstaged, set `model.profile` explicitly. Profiles:
   - `Qwen3_5ForConditionalGeneration` -> `qwen3_5` (GPU-verified), `Qwen3ForCausalLM` -> `qwen3`, `LlamaForCausalLM` -> `llama`,
     anything else -> `generic_causal_lm` (no special overrides).
4. If the architecture needs special handling, add `strl/models/<name>.py` with a `ModelProfile` subclass and register it in
   `strl/models/__init__.py` (`_PROFILES` + `_ARCH_TO_PROFILE`). Hooks:
   - `trainer_overrides` (FSDP wrap class, packing, language_model_only),
   - `generator_overrides` (vLLM engine kwargs),
   - `chat_kwargs` (e.g. `enable_thinking=False`), `vllm_kwargs` (offline eval),
   - `prepare_export_for_vllm` (if SkyRL's HF export does not reload in vLLM as is; see `qwen3_5_export.py`),
   - set `verified` honestly (`preflight` until a canary passes, `canary`, then `gpu` after a measured held-out improvement).
5. Terminal EOS markers differ by family (`<|im_end|>` Qwen, `<|eot_id|>` Llama): scorers strip only known terminal markers.
6. Add the model to `limits.models` only if your config uses an allowlist (the default is `["*"]`).

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
