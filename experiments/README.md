# Experiments

One YAML per experiment. Each `extends:` a base in `base/` and changes only what it studies. Run `python -m strl plan <file>`
before anything else: it validates, derives the infrastructure, estimates time/cost and prints the decision-point warnings.

| File | Axis changed vs its base | Purpose |
| --- | --- | --- |
| `gsm8k-qwen35.yaml` | none | the GPU-verified GSM8K run (+18.04 points) |
| `mbpp-qwen35-binary.yaml` | compute: 4 GPUs | reward A/B, arm A: pass_all_tests |
| `mbpp-qwen35-partial.yaml` | reward | reward A/B, arm B: pass_all_tests + 0.5 partial_tests |
| `mbpp-qwen3-1p7b.yaml` | model | model switch to Qwen3-1.7B (dense Qwen3 profile) |
| `harbor-qwen35.yaml` | environment | same tasks as multi-turn Harbor/Terminus-2 tasks in Modal sandboxes |
| `mbpp-qwen35-custom-reward.yaml` | reward (your own code) | arm A + `user_rewards.examples:docstring_bonus`, the `module:function` pattern |
| `mimo-qwen38-27b-lora.yaml` | environment `mimo` (five MiMo domains), model, LoRA, 2 nodes | multi-domain agentic RL, Qwen3.8-27B, LoRA r64; `strl prepare` first |
| `mimo-qwen38-27b-full.yaml` | finetune mode | the same, full fine-tuning (full shard over 16 GPUs) |
| `mimo-qwen35-0p8b-smoke.yaml` | model, compute: 1 GPU | cheap end-to-end check of the mimo path (also the 1-GPU canary arm) |

Merge rules for `extends:`:
- Mappings merge key by key.
- **Lists and scalars replace** the parent's value. `reward.components` and `tracking.tags` are always the complete list.
- Everything you do not set is inherited, **including `tracking.group`**. Children of `mbpp-qwen35-binary.yaml` land in the
  `mbpp-reward-ab` W&B group. Set your own `tracking.group` unless you want your run plotted alongside that A/B.
- `strl plan <exp> --against <other>` prints the exact resolved diff. Check it before staging.
- An experiment file can live anywhere. `extends:` takes an absolute path, a path relative to the file's own folder, or a
  repo-relative path (for example `experiments/mbpp-qwen35-binary.yaml`). Always run the CLI from the repo root.
