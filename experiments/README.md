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
| `mimo-qwen35-0p8b-2node-smoke.yaml` | model, compute: 2 x 8 | multi-node smoke at 0.8B (rendezvous, 16-rank NCCL, LoRA sync, export gather, retention) |
| `mimo-qwen38-27b-lora-run1.yaml` / `-run1-1node.yaml` | selection, budget | the 27B main run on 2 x 8 and its 1 x 8 fallback |
| `mimo-example-music-code-full.yaml` | domains, finetune mode, reward (shaping) | template for a new MiMo program: subset, full fine-tuning, custom shaping term |
| `mimo-qwen3-8b-lora-2node.yaml` / `-proof.yaml` | model (8B), domains (cyber + code), 2 x 8 | mid-size dense LoRA on 2 nodes; the bounded `-proof` variant ran and verified (`fa-q3pf-a1-fb`) |
| `countdown-qwen35.yaml` | environment (`countdown`, new) | single-turn verifiable arithmetic; the template for adding your own environment; 1-GPU offline final eval |
| `mimo-phi4-mini-lora-verify.yaml` | model (non-Qwen), LoRA, reward, eval | Phi-4-mini on music+cyber: shaped reward, durable checkpoints (resume), GSM8K retention |
| `gsm8k-qwen35-2node-smoke.yaml` / `-full.yaml` | compute: 2 nodes | multi-node smoke on GSM8K, LoRA / full |
| `r3b-math-smollm2-2node.yaml` / `-1node.yaml` | environment (`math_boxed`, new, downloaded MATH), model (SmolLM2-1.7B, Llama arch), full FT, 2 x 8 | **ran and verified** (`r3b-2n-a1-fb`): 22.4% -> 31.8% on 500 held-out problems, +9.4 [+5.6, +13.2]; launched with `--race-shape` for the 1-node variant |
| `r3a-cd-phi4.yaml` | model (Phi-4-mini), LoRA, 1 x 4, reward (brevity bonus only when correct), prompt (step by step) | a fresh agent's countdown learning run; the canary showed the default prompt starved a non-reasoning model of signal |
| `r3c-verifier.yaml` / `r3c-shaped.yaml` | model (Qwen3-14B), MiMo music + code, LoRA, 2 x 8, reward A/B | prepared + planned by a fresh agent; NOT run (needs ~38,000 node-s of authorized budget) |

Merge rules for `extends:`:
- Mappings merge key by key.
- **Lists and scalars replace** the parent's value. `reward.components` and `tracking.tags` are always the complete list.
- Everything you do not set is inherited, **including `tracking.group`**. Children of `mbpp-qwen35-binary.yaml` land in the
  `mbpp-reward-ab` W&B group. Set your own `tracking.group` unless you want your run plotted alongside that A/B.
- `strl plan <exp> --against <other>` prints the exact resolved diff. Check it before staging.
- An experiment file can live anywhere. `extends:` takes an absolute path, a path relative to the file's own folder, or a
  repo-relative path (for example `experiments/mbpp-qwen35-binary.yaml`). Always run the CLI from the repo root.
