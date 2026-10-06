"""Your own reward components, referenced from an experiment as ``user_rewards.<module>:<function>``.

``strl stage`` ships this package in the pod payload; ``strl plan`` validates names and parameters. Signature:
``fn(result: dict, ctx: strl.rewards.RewardContext, **params) -> float``. For coding environments ``result`` carries
``parsed`` (the extracted code), ``passed``/``total`` tests, ``failure_class``, ``has_code`` and ``completion``.
"""
