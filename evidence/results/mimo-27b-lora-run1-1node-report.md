# mimo-27b-lora-run1-1node: results report

Evidence: base = the step-0 in-loop evaluation, trained = the last in-loop evaluation (same engines, agent and template); held-out trials only; masked / missing trial ids are dropped on both sides and counted, never scored as zero.

| attempt | model | reward | steps | base | trained | delta [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| run1-n1-a | Qwen/Qwen3.8-27B | 1.0*verifier_score | 3 | 4.26% | 6.38% | +2.11 [-2.68, +7.08] |

| attempt | learning signal first -> last | train reward last (max) | val task metric last (max) | reward components last | step s (last) | top critic modes |
| --- | --- | --- | --- | --- | --- | --- |
| run1-n1-a | 0.225 -> 0.150 | 0.044 (0.044) | 0.064 (0.064) | - | 1396.939 | timeout_or_inefficient 88%, no_submission 8%, incomplete_deliverable 4% |

**run1-n1-a** per domain (trained = step 3: last_step: no checkpoint selection, the final step is the trained model)

| domain | base | trained | delta [95% CI] | n tasks | dropped |
| --- | --- | --- | --- | --- | --- |
| code | 0.1000 | 0.1000 | +0.0000 [-0.2000, +0.2000] | 15 | 2 |
| cyber | 0.0312 | 0.1250 | +0.0938 [+0.0000, +0.1875] | 16 | 0 |
| general | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 16 | 0 |
| music | 0.0573 | 0.0433 | -0.0140 [-0.0813, +0.0517] | 16 | 0 |
| webdev | 0.0246 | 0.0506 | +0.0260 [-0.0193, +0.0806] | 16 | 0 |
| **macro** | 0.0426 | 0.0638 | +0.0211 [-0.0268, +0.0708] | 79 | 2 |

## run1-n1-a: training

Training pass rate by step (generate/<domain>/pass_rate)

| step | music | cyber | code | general | webdev |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.00 | 0.04 | 0.10 | 0.00 | 0.00 |
| 2 | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 |
| 3 | 0.00 | 0.19 | 0.00 | 0.00 | 0.00 |

Training reward by step (generate/<domain>/reward_mean)

| step | music | cyber | code | general | webdev |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.05 | 0.04 | 0.10 | 0.00 | 0.02 |
| 2 | 0.07 | 0.04 | 0.00 | 0.00 | 0.01 |
| 3 | 0.03 | 0.19 | 0.00 | 0.00 | 0.00 |

Held-out task metric at every eval (eval/<domain>/task_metric)

| step | music | cyber | code | general | webdev | macro |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.057 | 0.031 | 0.100 | 0.000 | 0.025 | 0.043 |
| 3 | 0.043 | 0.125 | 0.100 | 0.000 | 0.051 | 0.064 |

| learning signal (groups with reward variance) | first | mean | last |
| --- | --- | --- | --- |
| train/learning_signal_fraction | 0.225 | 0.175 | 0.150 |

| domain | masked groups mean | masked groups max | tokens share | judge cost (USD) |
| --- | --- | --- | --- | --- |
| music | 0.000 | 0.000 | 7.3% | 0.000 |
| cyber | 0.000 | 0.000 | 29.6% | 0.000 |
| code | 0.042 | 0.125 | 27.3% | 0.000 |
| general | 0.000 | 0.000 | 27.0% | 0.856 |
| webdev | 0.000 | 0.000 | 8.8% | 1.042 |

| step-time phase | mean s | max s | steps |
| --- | --- | --- | --- |
| step | 1413.7 | 1504.1 | 3 |
| generate | 821.1 | 864.0 | 3 |
| fwd_logprobs_values_reward | 130.7 | 155.8 | 3 |
| policy_train | 387.8 | 449.3 | 3 |
| train_critic_and_policy | 390.9 | 452.8 | 3 |
| sync_weights | 14.1 | 14.6 | 3 |
| save_hf_model | 120.9 | 120.9 | 1 |
| eval | 792.9 | 856.5 | 2 |

Infrastructure: peak GPU memory (head node) 70.8 GB; critic cost 0.095 USD
Critic failure-mode shares (last analysed step): timeout_or_inefficient 88%, no_submission 8%, incomplete_deliverable 4%, blank_render 0%, breaks_existing_tests 0%, build_error 0%, did_not_run_tests 0%, edge_case 0%
