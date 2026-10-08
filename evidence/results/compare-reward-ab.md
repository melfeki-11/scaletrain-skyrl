| attempt | model | reward | steps | base | trained | delta [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| ab-binary-a1 | Qwen/Qwen3.5-0.8B | 1.0*pass_all_tests | 66 | 23.38% | 36.05% | +12.67 [+9.20, +16.29] |
| ab-partial-a1 | Qwen/Qwen3.5-0.8B | 1.0*pass_all_tests + 0.5*partial_tests | 69 | 23.38% | 33.33% | +9.95 [+6.33, +13.57] |

| attempt | learning signal first -> last | train reward last (max) | val task metric last (max) | reward components last | step s (last) | top critic modes |
| --- | --- | --- | --- | --- | --- | --- |
| ab-binary-a1 | 0.375 -> 0.750 | 0.582 (0.648) | 0.417 (0.427) | pass_all_tests 0.58 | 42.554 | syntax_or_runtime_error 39%, wrong_algorithm 38%, misread_task 12% |
| ab-partial-a1 | 0.625 -> 0.688 | 0.797 (1.070) | 0.302 (0.448) | partial_tests 0.58, pass_all_tests 0.51 | 32.536 | wrong_algorithm 38%, syntax_or_runtime_error 33%, misread_task 15% |

**ab-partial-a1 vs ab-binary-a1** (trained vs trained, 663 paired tasks): -2.71 points [-6.18, +0.75]
