# mimo-smoke: results report

Evidence: base = the step-0 in-loop evaluation, trained = the last in-loop evaluation (same engines, agent and template); held-out trials only; masked / missing trial ids are dropped on both sides and counted, never scored as zero.

| attempt | model | reward | steps | base | trained | delta [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| can-mimo-d | Qwen/Qwen3.5-0.8B | 1.0*verifier_score | None | - | - | - |

| attempt | learning signal first -> last | train reward last (max) | val task metric last (max) | reward components last | step s (last) | top critic modes |
| --- | --- | --- | --- | --- | --- | --- |
| can-mimo-d | 0.000 -> 0.000 | 0.000 (0.000) | 0.000 (0.000) | - | 546.732 | blank_render 38%, timeout_or_inefficient 25%, invalid_abc 12% |

## can-mimo-d: training

Training pass rate by step (generate/<domain>/pass_rate)

| step | music | cyber | code | general | webdev |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Training reward by step (generate/<domain>/reward_mean)

| step | music | cyber | code | general | webdev |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Held-out task metric at every eval (eval/<domain>/task_metric)

| step | music | cyber | code | general | webdev | macro |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

| learning signal (groups with reward variance) | first | mean | last |
| --- | --- | --- | --- |
| train/learning_signal_fraction | 0.000 | 0.000 | 0.000 |

| domain | masked groups mean | masked groups max | tokens share | judge cost (USD) |
| --- | --- | --- | --- | --- |
| music | 0.000 | 0.000 | 5.0% | 0.000 |
| cyber | 0.000 | 0.000 | 27.5% | 0.000 |
| code | 0.000 | 0.000 | 29.0% | 0.000 |
| general | 0.000 | 0.000 | 29.1% | 0.004 |
| webdev | 1.000 | 1.000 | 9.4% | 0.001 |

| step-time phase | mean s | max s | steps |
| --- | --- | --- | --- |
| step | 579.4 | 612.1 | 2 |
| generate | 306.8 | 314.6 | 2 |
| fwd_logprobs_values_reward | 67.3 | 83.5 | 2 |
| policy_train | 200.3 | 226.3 | 2 |
| train_critic_and_policy | 200.8 | 226.7 | 2 |
| sync_weights | 2.3 | 2.3 | 2 |
| save_hf_model | 3.5 | 3.5 | 1 |
| eval | 310.1 | 324.5 | 2 |

Infrastructure: peak GPU memory (head node) 65.3 GB; critic cost 0.001 USD
Critic failure-mode shares (last analysed step): blank_render 38%, timeout_or_inefficient 25%, invalid_abc 12%, no_submission 12%, truncated_or_incomplete 12%, breaks_existing_tests 0%, build_error 0%, did_not_run_tests 0%
