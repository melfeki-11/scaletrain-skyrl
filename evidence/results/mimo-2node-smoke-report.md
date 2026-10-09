# mimo-2node-smoke: results report

Evidence: base = the step-0 in-loop evaluation, trained = the last in-loop evaluation (same engines, agent and template); held-out trials only; masked / missing trial ids are dropped on both sides and counted, never scored as zero.

| attempt | model | reward | steps | base | trained | delta [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| v3smk3-fb | Qwen/Qwen3.5-0.8B | 1.0*verifier_score | 2 | 0.00% | 0.00% | +0.00 [+0.00, +0.00] |

| attempt | learning signal first -> last | train reward last (max) | val task metric last (max) | reward components last | step s (last) | top critic modes |
| --- | --- | --- | --- | --- | --- | --- |
| v3smk3-fb | 0.000 -> 0.000 | 0.000 (0.000) | 0.000 (0.000) | - | 267.740 | invalid_abc 62%, wrong_length 31%, truncated_or_incomplete 6% |

**v3smk3-fb** per domain (trained = step 2: last_step: no checkpoint selection, the final step is the trained model)

| domain | base | trained | delta [95% CI] | n tasks | dropped |
| --- | --- | --- | --- | --- | --- |
| cyber | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 4 | 0 |
| music | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 4 | 0 |
| **macro** | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 8 | 0 |

## v3smk3-fb: training

Training pass rate by step (generate/<domain>/pass_rate)

| step | music | cyber |
| --- | --- | --- |
| 1 | 0.00 | 0.00 |
| 2 | 0.00 | 0.00 |

Training reward by step (generate/<domain>/reward_mean)

| step | music | cyber |
| --- | --- | --- |
| 1 | 0.00 | 0.00 |
| 2 | 0.00 | 0.00 |

Held-out task metric at every eval (eval/<domain>/task_metric)

| step | music | cyber | macro |
| --- | --- | --- | --- |
| 0 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |

| learning signal (groups with reward variance) | first | mean | last |
| --- | --- | --- | --- |
| train/learning_signal_fraction | 0.000 | 0.000 | 0.000 |

| domain | masked groups mean | masked groups max | tokens share | judge cost (USD) |
| --- | --- | --- | --- | --- |
| music | 0.000 | 0.000 | 11.1% | 0.000 |
| cyber | 0.000 | 0.000 | 88.9% | 0.000 |

| step-time phase | mean s | max s | steps |
| --- | --- | --- | --- |
| step | 308.5 | 349.2 | 2 |
| generate | 223.2 | 224.6 | 2 |
| fwd_logprobs_values_reward | 22.9 | 39.8 | 2 |
| policy_train | 50.3 | 78.8 | 2 |
| train_critic_and_policy | 51.3 | 79.8 | 2 |
| sync_weights | 6.1 | 6.2 | 2 |
| save_hf_model | 6.8 | 6.8 | 1 |
| eval | 206.4 | 209.5 | 2 |

Infrastructure: transport=socket efa=False nodes=2 bandwidth_gbps={"intra_node": 343.23, "cross_node": 0.68, "global": 2.32}; peak GPU memory (head node) 43.3 GB; critic cost 0.052 USD
Critic failure-mode shares (last analysed step): invalid_abc 62%, wrong_length 31%, truncated_or_incomplete 6%, edge_case 0%, format_violation 0%, invalid_poc 0%, misread_task 0%, no_crash 0%
