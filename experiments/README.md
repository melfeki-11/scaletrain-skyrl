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
