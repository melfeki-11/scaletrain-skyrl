"""Patch the pinned SkyRL GSM8K reward with the strict terminal-answer verifier (the GPU-verified default reward path).

SkyRL's stock verifier accepts the first ``#### n`` anywhere and sees the real terminal <|im_end|> in the text, which made
every strict reward 0 on Qwen. The patch strips only a terminal EOS marker, still rejects trailing ordinary text and
truncation, and is exercised by ``generator_contract`` in the CPU preflight.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "# grpo-mini strict terminal answer verifier\n"
BODY = (
    MARKER
    + "from strl.tasks.gsm8k.answers import parse_final_answer\n\n"
    + "def extract_solution(solution_str, method='strict'):\n"
    + "    return parse_final_answer(solution_str)\n\n"
    + "def compute_score(solution_str, ground_truth, method='strict', format_score=0.0, score=1.0):\n"
    + "    answer = parse_final_answer(solution_str)\n"
    + "    return score if answer is not None and answer == str(ground_truth) else 0.0\n"
)


def patch_reward_verifier(root: str | Path) -> None:
    path = Path(root) / "skyrl-gym/skyrl_gym/envs/gsm8k/utils.py"
    path.write_text(BODY, encoding="utf-8")
