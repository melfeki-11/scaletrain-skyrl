"""`mimo` environment: any subset of XiaomiMiMo/MiMo-V2.6-RL-oss (rev 639865fd) as Harbor tasks, one trainer, five domains.

Domains: music (single-shot), cyber (ARVO crash reproduction), code (SWE), general (MCP + rubric), webdev (render + vision).
Every selected task is exported to a Harbor task dir; SkyRL's HarborGenerator (subclassed in generator.py) runs the trials.

CONTRACT between the exporter/graders (export.py, agents.py, graders/) and the generator/spec/data (generator.py, spec.py,
data.py, judge.py). Change it only in this docstring + both sides.

Task dir  <data_dir>/tasks/<split>/<domain>-<instance_id>/     (split: train | test | canary)
  instruction.md            prompt + where to write the output (agent-facing)
  task.toml                 per-domain [agent].timeout_sec, [verifier].timeout_sec, [environment] cpus / memory_mb / storage_mb
  environment/Dockerfile    FROM the task image (mirror digest or docker.io/xiaomimimo/mimo-v2.6-rl-oss:<tag>) + static tmux
  tests/test.sh             writes 0 to /logs/verifier/reward.txt FIRST, then grades; never leaves a stale pass
  strl.json                 {"domain", "instance_id", "group" (split-grouping key), "image", "agent": {"name": "terminus-2"} or
                             {"import_path": "strl.tasks.mimo.agents:SingleShotAgent"}, "max_turns", "max_tokens",
                             "artifacts": ["/app/answer.md", ...]  (files copied to the host verifier dir for tables),
                             "judge": null | {"kind": "rubric" | "vision"}}
  solution/solve.sh         oracle where one exists (music: none; cyber/code: none -> use gpt-6-luna positive control)

Verifier outputs (inside the sandbox, collected by Harbor into <trial_dir>/verifier/):
  reward.txt         float in [0,1]. For judge == null this IS the reward. For judged domains it is only the in-sandbox
                     (rule) part, already divided by the TOTAL weight, and judge_request.json carries the rest.
  details.json       {"domain", "metric" (the domain task metric in [0,1]), "components": {...}, "infra_error": bool, ...}
  judge_request.json (judged domains only) {"kind", "base_score" (sum w*s of rule items), "base_weight", "total_weight",
                      "items": [{"id", "weight", "question", "anchor", "answer"}], "images": ["/logs/verifier/screenshot.png"],
                      "query": "<original user request, vision>"}
  artifacts          files named in strl.json["artifacts"] (+ screenshot.png, midi/abc, poc, patch.diff) for W&B tables

Judge hand-off (generator.py + judge.py, OUTSIDE the sandbox; the judge key never enters a sandbox):
  final = (base_score + sum_i w_i * s_i) / total_weight, s_i from the LiteLLM judge in [0,1]; judge failure => the trial is
  masked (infra error), never scored 0. infra_error in details.json also masks the trial (reward 0 + group masked by SkyRL).

Trial template per task = base template (strl/tasks/coding/harbor.py trial_template) overridden by strl.json (agent, max_turns,
max_tokens) and task.toml; NO global override_* keys. Terminus-2 tasks use the default agent; SingleShotAgent makes exactly one
policy call and writes the reply to the first artifact path, with metadata["n_episodes"] = 1.

Python API (export.py, track B)
  DOMAINS = ("music", "cyber", "code", "general", "webdev")
  image_ref(domain, instance_json: dict) -> str          docker.io/xiaomimimo/mimo-v2.6-rl-oss:<tag> via image-mapping.jsonl
  export_task(row: dict, dest: Path, *, split: str) -> dict
        row = {"domain", "instance_id", "group", "raw": <parquet row as dict; extra_info.instance_json is a JSON string>}
        returns the manifest entry {"task_dir", "domain", "instance_id", "group", "image", "tree_sha256"}; deterministic (same row =>
        byte-identical dir). Tasks are written under dest = <data_dir>/tasks/<split>/<domain>-<instance_id>.
  graders/<domain>.py   build(row, task_dir) writes tests/ files; each grader also exposes grade(workdir, ...) used by unit tests
Data API (data.py, track A)
  load_pool(domain) -> list[row]; build_selection(cfg) -> {"train": [row], "test": [row], "canary": [row]} with fixed grouped test
  sets per domain (hash-pinned in <data_dir>/manifest.json); prepare(cfg) calls export_task for every selected row.
Eval repetitions: the exported dataset lists a test task `eval_repeats` times with env_extras.rep = 0..n-1 (SkyRL's step-wise eval
dump has no per-sample index), so eval_n_samples_per_prompt stays 1 and identity = (task_dir name, rep).
"""
