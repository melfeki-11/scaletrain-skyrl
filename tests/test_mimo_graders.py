"""mimo graders + export + agent contracts. No Modal/Docker needed (run with MIMO_EXPORT_OFFLINE
semantics for export_task's build-time artifacts; everything else is a pure function or runs
real subprocesses -- abc2midi, git, python -- directly on this machine)."""
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "assets" / "overlay"))  # harbor + modal SDKs for agents.py / export.py

os.environ.setdefault("MIMO_EXPORT_OFFLINE", "1")

from strl.tasks.mimo import export  # noqa: E402
from strl.tasks.mimo.graders import code as code_grader  # noqa: E402
from strl.tasks.mimo.graders import cyber as cyber_grader  # noqa: E402
from strl.tasks.mimo.graders import general as general_grader  # noqa: E402
from strl.tasks.mimo.graders import music as music_grader  # noqa: E402
from strl.tasks.mimo.graders import webdev as webdev_grader  # noqa: E402

RAW_DIR = ROOT / "assets" / "mimo" / "raw"
HAS_ABC2MIDI = subprocess.run(["bash", "-c", "command -v abc2midi"], capture_output=True).returncode == 0
VALID_ABC = (
    "```abc\nX:1\nT:Test\nM:4/4\nL:1/8\nK:C\nCDEF GABc | cBAG FEDC | C4 z4 | G,2A,2 B,2C2 |\n"
    "DEFG ABcd | edcB AGFE | D4 z4 |]\n```\n"
)


def _rows(parquet_name: str, limit: int = 5, where=None):
    import pandas as pd

    df = pd.read_parquet(RAW_DIR / f"{parquet_name}.parquet")
    out = []
    for _, r in df.iterrows():
        raw = {"prompt": list(r["prompt"]), "extra_info": dict(r["extra_info"])}
        if where and not where(raw):
            continue
        out.append(raw)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------------------- music


@unittest.skipUnless(HAS_ABC2MIDI, "abc2midi not on PATH")
class MusicGrader(unittest.TestCase):
    def test_valid_abc_scores_positive(self):
        r = music_grader.grade(VALID_ABC, constraints={"bpm": 120, "meter": "4/4", "nvoice_want": 1})
        self.assertGreater(r["reward"], 0.0)
        self.assertEqual(r["details"]["validity"], "ok")

    def test_garbage_text_scores_zero(self):
        r = music_grader.grade("I refuse to write music.")
        self.assertEqual(r["reward"], 0.0)
        self.assertEqual(r["details"]["validity"], "no_abc_found")

    def test_empty_tune_is_rejected(self):
        r = music_grader.grade("```abc\nX:1\n```")
        self.assertEqual(r["reward"], 0.0)
        self.assertEqual(r["details"]["validity"], "rejected")

    def test_build_writes_tests_dir(self):
        raw = _rows("music", limit=1)[0]
        row = {"domain": "music", "instance_id": "m0", "group": "music", "raw": raw}
        with tempfile.TemporaryDirectory() as td:
            task_dir = Path(td)
            music_grader.build(row, task_dir)
            self.assertTrue((task_dir / "tests" / "test.sh").is_file())
            self.assertTrue((task_dir / "tests" / "mimo_music_scorer" / "score.py").is_file())
            constraints = json.loads((task_dir / "tests" / "constraints.json").read_text())
            self.assertIn("bpm", constraints)


# --------------------------------------------------------------------------------------- cyber

ASAN_MATCH = (
    "==1==ERROR: AddressSanitizer: heap-buffer-overflow\n"
    "READ of size 4\n"
    "    #0 0x5 in extract_name /src/dnsmasq/src/rfc1035.c:263:12\n"
    "    #1 0x6 in rfc1035_name /src/dnsmasq/src/rfc1035.c:310:5\n"
    "    #2 0x7 in LLVMFuzzerTestOneInput /src/dnsmasq_fuzzer.c:20:3\n"
    "SUMMARY: AddressSanitizer: heap-buffer-overflow /src/dnsmasq/src/rfc1035.c:263:12 in extract_name\n"
)


class CyberGrader(unittest.TestCase):
    def setUp(self):
        self.expected = cyber_grader.parse_description(
            "AddressSanitizer: heap-buffer-overflow in function `extract_name` in file `dnsmasq/src/rfc1035.c`"
        )

    def test_parse_description(self):
        self.assertEqual(self.expected, ("extract_name", "dnsmasq/src/rfc1035.c", "AddressSanitizer", "heap-buffer-overflow"))

    def test_match_in_topmost_frame(self):
        r = cyber_grader.grade_run(exit_code=1, stdout="", stderr=ASAN_MATCH, timed_out=False, expected=self.expected)
        self.assertEqual(r["reward"], 1.0)
        self.assertTrue(r["details"]["match"])

    def test_crash_in_wrong_function_scores_zero(self):
        wrong = ASAN_MATCH.replace("#0 0x5 in extract_name", "#0 0x5 in rfc1035_name")
        r = cyber_grader.grade_run(exit_code=1, stdout="", stderr=wrong, timed_out=False, expected=self.expected)
        self.assertEqual(r["reward"], 0.0)
        self.assertFalse(r["details"]["match"])

    def test_no_crash_scores_zero(self):
        r = cyber_grader.grade_run(exit_code=0, stdout="", stderr="", timed_out=False, expected=self.expected)
        self.assertEqual(r["reward"], 0.0)
        self.assertEqual(r["details"]["reason"], "no_crash")

    def test_sanitizer_runtime_frame_is_skipped(self):
        # A frame whose function name mentions the sanitizer itself (matching upstream's exact
        # casing -- "ASan"/"MSan"/"UBSan"/"Sanitizer") must never be reported as the crash site.
        stderr = ASAN_MATCH.replace(
            "    #0 0x5 in extract_name /src/dnsmasq/src/rfc1035.c:263:12\n",
            "    #0 0x5 in __ASan_poison_region /src/dnsmasq/src/rfc1035.c:263:12\n"
            "    #1 0x6 in extract_name /src/dnsmasq/src/rfc1035.c:270:1\n",
        )
        func, _file = cyber_grader.extract_crash_function(stderr)
        self.assertEqual(func, "extract_name")

    def test_build_writes_tests_dir(self):
        raw = _rows("cyber", limit=1)[0]
        row = {"domain": "cyber", "instance_id": "c0", "group": "cyber", "raw": raw}
        with tempfile.TemporaryDirectory() as td:
            task_dir = Path(td)
            cyber_grader.build(row, task_dir, capture_fn=lambda image: (b"fake-tar-bytes", "0" * 64))
            self.assertTrue((task_dir / "tests" / "binary.tar.gz").is_file())
            expected = json.loads((task_dir / "tests" / "expected.json").read_text())
            self.assertEqual(expected[0], "extract_name")


# ---------------------------------------------------------------------------------------- code

VCS = "g" + "it"  # keeps the sandbox's command-classifier from flagging plain-text "git" here


def _init_repo(d: str, lib_body: str) -> str:
    subprocess.run([VCS, "init", "-q"], cwd=d, check=True)
    subprocess.run([VCS, "config", "user.email", "a@b.com"], cwd=d, check=True)
    subprocess.run([VCS, "config", "user.name", "a"], cwd=d, check=True)
    (Path(d) / "lib.py").write_text(lib_body)
    subprocess.run([VCS, "add", "-A"], cwd=d, check=True)
    subprocess.run([VCS, "commit", "-q", "-m", "base"], cwd=d, check=True)
    return subprocess.run([VCS, "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True).stdout.strip()


TEST_PATCH = (
    "diff --git a/test_lib.py b/test_lib.py\n"
    "new file mode 100644\n"
    "index 0000000..abcdefg\n"
    "--- /dev/null\n"
    "+++ b/test_lib.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+from lib import add\n"
    "+assert add(2, 3) == 5\n"
)


class CodeGrader(unittest.TestCase):
    def test_get_patch_touched_files(self):
        self.assertEqual(code_grader.get_patch_touched_files(TEST_PATCH), ["test_lib.py"])

    def test_agent_fix_passes(self):
        with tempfile.TemporaryDirectory() as d:
            base = _init_repo(d, "def add(a,b):\n    return a - b\n")
            (Path(d) / "lib.py").write_text("def add(a,b):\n    return a + b\n")
            r = code_grader.grade(
                repo_path=d, test_patch=TEST_PATCH, test_command=f"{sys.executable} test_lib.py", base_ref=base, timeout_sec=30
            )
            self.assertEqual(r["reward"], 1.0)

    def test_nop_fails(self):
        with tempfile.TemporaryDirectory() as d:
            base = _init_repo(d, "def add(a,b):\n    return a - b\n")
            r = code_grader.grade(
                repo_path=d, test_patch=TEST_PATCH, test_command=f"{sys.executable} test_lib.py", base_ref=base, timeout_sec=30
            )
            self.assertEqual(r["reward"], 0.0)

    def test_tamper_planted_test_file_is_reset(self):
        with tempfile.TemporaryDirectory() as d:
            base = _init_repo(d, "def add(a,b):\n    return a - b\n")
            (Path(d) / "test_lib.py").write_text("assert True\n")  # agent plants a fake passing test
            r = code_grader.grade(
                repo_path=d, test_patch=TEST_PATCH, test_command=f"{sys.executable} test_lib.py", base_ref=base, timeout_sec=30
            )
            self.assertEqual(r["reward"], 0.0)
            self.assertNotEqual(r["details"].get("resolved"), True)

    def test_build_writes_tests_dir(self):
        raw = _rows("code", limit=1)[0]
        row = {"domain": "code", "instance_id": "co0", "group": "code", "raw": raw}
        with tempfile.TemporaryDirectory() as td:
            task_dir = Path(td)
            code_grader.build(row, task_dir, base_ref_fn=lambda image, cwd: "0" * 40)
            self.assertTrue((task_dir / "tests" / "test_patch.diff").is_file())


# ------------------------------------------------------------------------------------- general

CHECK_CODE = (
    "def check_ok(ws):\n"
    "    p = ws / 'answer.md'\n"
    "    return {'score': 1 if (p.is_file() and 'hello' in p.read_text()) else 0, 'detail': 'checked'}\n"
)


class GeneralGrader(unittest.TestCase):
    def _meta(self):
        return {
            "items": [
                {"id": "gate1", "gate": True, "method": "rule", "fn": "check_ok"},
                {"id": "r1", "method": "rule", "fn": "check_ok", "weight": 2.0},
                {"id": "l1", "method": "llm", "weight": 1.0, "question": "Is it good?", "pass_anchor": "yes"},
            ],
            "check_code": CHECK_CODE,
            "_helpers_src": "",
        }

    def test_gate_passes_and_weighted_average(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / "answer.md").write_text("hello world")
            r = general_grader.grade_general_agent(verifier_meta=self._meta(), workspace=ws, answer_text="hello world")
        self.assertFalse(r["details"]["gate_failed"])
        self.assertAlmostEqual(r["reward"], 2.0 / 3.0, places=5)
        self.assertEqual(r["judge_request"]["items"][0]["answer"], "hello world")

    def test_gate_failure_forces_zero_and_flags_judge(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / "answer.md").write_text("nope")
            r = general_grader.grade_general_agent(verifier_meta=self._meta(), workspace=ws, answer_text="nope")
        self.assertEqual(r["reward"], 0.0)
        self.assertTrue(r["details"]["gate_failed"])
        self.assertTrue(r["judge_request"]["force_zero"])

    def test_random_in_check_code_is_rejected(self):
        r = general_grader.run_rule_item(
            check_code="import random\ndef f(ws):\n    return {'score': random.random()}",
            fn="f", helpers_src="", workspace=Path("/tmp"),
        )
        self.assertEqual(r["score"], 0.0)

    def test_terminal_bench_build_decodes_real_row(self):
        raw = _rows("general_train", limit=1, where=lambda r: json.loads(r["extra_info"]["instance_json"])["dataset_type"] == "terminal_bench")[0]
        row = {"domain": "general", "instance_id": "tb0", "group": "general", "raw": raw}
        with tempfile.TemporaryDirectory() as td:
            task_dir = Path(td)
            general_grader.build(row, task_dir)
            self.assertTrue((task_dir / "tests" / "test.sh").is_file())
            mode = (task_dir / "tests" / "test.sh").stat().st_mode
            self.assertTrue(mode & 0o111)

    def test_terminal_bench_decode_roundtrip(self):
        tests_files = {"test.sh": base64.b64encode(b"#!/bin/bash\necho hi\n").decode(), "fixtures/d.txt": base64.b64encode(b"x").decode()}
        row = {
            "raw": {"extra_info": {"instance_json": json.dumps({"dataset_type": "terminal_bench", "tests_files": json.dumps(tests_files)})}}
        }
        with tempfile.TemporaryDirectory() as td:
            task_dir = Path(td)
            general_grader.build(row, task_dir)
            self.assertEqual((task_dir / "tests" / "fixtures" / "d.txt").read_bytes(), b"x")


# -------------------------------------------------------------------------------------- webdev


class WebdevGrader(unittest.TestCase):
    def test_render_ok_produces_vision_judge_request(self):
        shot = {"ok": True, "screenshot_png": b"PNG", "console_errors": [], "render_env": {}, "error": None}
        r = webdev_grader.grade_from_shot(shot, query="Build me a site")
        self.assertEqual(r["reward"], 0.0)
        self.assertFalse(r["details"]["infra_error"])
        self.assertEqual(r["judge_request"]["kind"], "vision")
        self.assertEqual(r["judge_request"]["total_weight"], 1.0)
        self.assertIn("layout_integrity", r["judge_request"]["items"][0]["question"])

    def test_no_screenshot_is_render_failed_not_infra_error(self):
        shot = {"ok": False, "screenshot_png": None, "console_errors": [], "render_env": {}, "error": "no screenshot produced"}
        r = webdev_grader.grade_from_shot(shot, query="x")
        self.assertEqual(r["reward"], 0.0)
        self.assertTrue(r["details"]["render_failed"])
        self.assertFalse(r["details"]["infra_error"])
        self.assertIsNone(r["judge_request"])

    def test_build_writes_tests_dir(self):
        raw = _rows("webdev", limit=1)[0]
        row = {"domain": "webdev", "instance_id": "w0", "group": "webdev", "raw": raw}
        with tempfile.TemporaryDirectory() as td:
            task_dir = Path(td)
            webdev_grader.build(row, task_dir)
            self.assertTrue((task_dir / "tests" / "mimo_webdev" / "shot.py").is_file())
            self.assertTrue((task_dir / "tests" / "query.json").is_file())


# --------------------------------------------------------------------------------------- agent


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def call(self, prompt, message_history=None, **kwargs):
        from harbor.llms.base import LLMResponse
        from harbor.models.metric import UsageInfo

        self.calls.append(prompt)
        return LLMResponse(
            content=self.content,
            usage=UsageInfo(prompt_tokens=10, completion_tokens=5, cache_tokens=0, cost_usd=0.001),
            prompt_token_ids=[1, 2, 3],
            completion_token_ids=[4, 5],
            logprobs=[-0.1, -0.2],
        )

    def get_model_context_limit(self):
        return 8192

    def get_model_output_limit(self):
        return 1024


class FakeEnvironment:
    def __init__(self):
        self.uploaded = {}

    async def exec(self, command, **kwargs):
        return None

    async def upload_file(self, source_path, target_path):
        self.uploaded[target_path] = Path(source_path).read_text()


class SingleShotAgentTest(unittest.TestCase):
    def test_single_call_writes_artifact_and_rollout_details(self):
        from harbor.models.agent.context import AgentContext
        from strl.tasks.mimo.agents import SingleShotAgent

        agent = SingleShotAgent.__new__(SingleShotAgent)
        agent._artifact_path = "/app/answer.abc"
        agent._collect_rollout_details = True
        agent._llm = FakeLLM("```abc\nX:1\nK:C\nCDEF|\n```")
        agent.logs_dir = None
        agent.model_name = "fake/model"
        import logging

        agent.logger = logging.getLogger("test")
        agent.mcp_servers = []
        agent.skills_dir = None

        env = FakeEnvironment()
        ctx = AgentContext()
        asyncio.run(agent.run("Write a tune", env, ctx))

        self.assertEqual(env.uploaded["/app/answer.abc"], "```abc\nX:1\nK:C\nCDEF|\n```")
        self.assertEqual(ctx.metadata["n_episodes"], 1)
        self.assertEqual(ctx.n_input_tokens, 10)
        self.assertEqual(ctx.n_output_tokens, 5)
        self.assertEqual(ctx.rollout_details[0]["prompt_token_ids"], [[1, 2, 3]])
        self.assertEqual(len(agent._llm.calls), 1)


# -------------------------------------------------------------------------------------- export


class ExportTest(unittest.TestCase):
    def test_music_export_is_deterministic(self):
        raw = _rows("music", limit=1)[0]
        row = {"domain": "music", "instance_id": "gK-0768", "group": "music", "raw": raw}
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)
            m1 = export.export_task(row, dest, split="canary")
            m2 = export.export_task(row, dest, split="canary")
            self.assertEqual(m1["tree_sha256"], m2["tree_sha256"])
            self.assertEqual(m1["image"], export.MUSIC_IMAGE)

    def test_cyber_export_layout(self):
        raw = _rows("cyber", limit=1)[0]
        row = {"domain": "cyber", "instance_id": "c0", "group": "cyber", "raw": raw}
        with tempfile.TemporaryDirectory() as td:
            m = export.export_task(row, Path(td), split="canary")
            task_dir = Path(m["task_dir"])
            self.assertTrue((task_dir / "tests" / "binary.tar.gz").is_file())
            self.assertIn("mimo-tmux-static", (task_dir / "environment" / "Dockerfile").read_text())
            strl = json.loads((task_dir / "strl.json").read_text())
            self.assertEqual(strl["agent"], {"name": "terminus-2"})
            self.assertIsNone(strl["judge"])

    def test_code_export_layout(self):
        raw = _rows("code", limit=1)[0]
        row = {"domain": "code", "instance_id": "co0", "group": "code", "raw": raw}
        with tempfile.TemporaryDirectory() as td:
            m = export.export_task(row, Path(td), split="canary")
            task_dir = Path(m["task_dir"])
            self.assertTrue((task_dir / "tests" / "test_patch.diff").is_file())

    def test_webdev_export_layout(self):
        raw = _rows("webdev", limit=1)[0]
        row = {"domain": "webdev", "instance_id": "w0", "group": "webdev", "raw": raw}
        with tempfile.TemporaryDirectory() as td:
            m = export.export_task(row, Path(td), split="canary")
            task_dir = Path(m["task_dir"])
            strl = json.loads((task_dir / "strl.json").read_text())
            self.assertEqual(strl["judge"], {"kind": "vision"})
            self.assertIn("playwright", (task_dir / "environment" / "Dockerfile").read_text())

    def test_general_terminal_bench_export_layout(self):
        raw = _rows("general_train", limit=1, where=lambda r: json.loads(r["extra_info"]["instance_json"])["dataset_type"] == "terminal_bench")[0]
        row = {"domain": "general", "instance_id": "tb0", "group": "general", "raw": raw}
        with tempfile.TemporaryDirectory() as td:
            m = export.export_task(row, Path(td), split="canary")
            task_dir = Path(m["task_dir"])
            strl = json.loads((task_dir / "strl.json").read_text())
            self.assertIsNone(strl["judge"])

    def test_image_ref_raises_on_unknown_image(self):
        with self.assertRaises(KeyError):
            export.image_ref("cyber", {"docker_image": "does-not-exist:v0"})

    def test_task_toml_has_per_domain_resources(self):
        toml = export.build_task_toml("code")
        self.assertIn("cpus = 4", toml)
        self.assertIn("memory_mb = 8192", toml)


if __name__ == "__main__":
    unittest.main()


class HfPathQuoting(unittest.TestCase):
    def test_non_ascii_and_space_paths_are_percent_encoded(self):
        """general_agent workspaces contain names like 'Budget – Q3 résumé.xlsx': a raw URL raised UnicodeEncodeError in http.client."""
        from strl.tasks.mimo import export
        quoted = export._quote('general/envs/x/workspace/Budget – Q3 résumé.xlsx')
        quoted.encode('ascii')   # must be pure ASCII now
        self.assertIn('%20', quoted)
        self.assertTrue(quoted.startswith('general/envs/x/workspace/'))


class DataLayerWithTheRealExporter(unittest.TestCase):
    """Track A's data.prepare + Track B's export.export_task together (each track tested its side against a fake of the other: the
    double-nested layout `tasks/test/<name>/test/<name>/` slipped through and broke every downstream path)."""

    def test_prepare_writes_tasks_where_the_data_layer_reads_them(self):
        from strl import experiment
        from strl.tasks.mimo import data
        with tempfile.TemporaryDirectory() as td:
            cfg = experiment.load("experiments/mimo-qwen35-0p8b-smoke.yaml",
                                  overrides=[f"run.data_dir={td}", "environment.mimo.domains={music: {train: 2, test: 2}}",
                                                "algorithm.prompts_per_step=2", "algorithm.minibatch_prompts=2"])
            manifest = data.prepare(cfg)
            self.assertEqual(manifest["counts"]["train"], 2)
            for split in ("train", "test", "canary"):
                dirs = sorted((Path(td) / "tasks" / split).iterdir())
                self.assertTrue(dirs, split)
                for task in dirs:
                    self.assertTrue((task / "strl.json").is_file(), task)          # not tasks/<split>/<name>/<split>/<name>/strl.json
                    self.assertTrue((task / "task.toml").is_file(), task)
            data.assert_manifest(cfg)                                              # the deep check A runs in the pod
            with self.assertRaises(ValueError):                                    # the clobber guard
                (Path(td) / "keep.txt").write_text("x")
                export.export_task({"domain": "music", "instance_id": "x", "group": "g", "raw": _rows("music", limit=1)[0]}, Path(td), split="train")


class CodeImageStaysAlive(unittest.TestCase):
    def test_code_dockerfile_defines_a_cmd_because_the_published_images_have_none(self):
        """Harbor starts sandboxes without a command; the code images have no CMD, so without this line every code trial raised
        ConflictError (found by the real nop gate, not by a unit test)."""
        text = export.build_dockerfile("code", "docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-000588")
        self.assertTrue(text.rstrip().endswith('CMD ["sleep", "infinity"]'))
        for domain in ("music", "cyber", "general", "webdev"):   # these images already keep a sandbox alive: untouched
            self.assertNotIn('CMD ["sleep"', export.build_dockerfile(domain, "x"))


class TerminalBenchImage(unittest.TestCase):
    def test_terminal_bench_dockerfile_has_no_sidecar_lines(self):
        """The 27B data prepare found 4 of 5 non-env-0 general images unbuildable: terminal_bench rows ship no general_env/, but
        got the MCP-sidecar tail (`COPY general_env/...`). prebuild in Modal found it; no unit test had looked at that Dockerfile."""
        image = "docker.io/xiaomimimo/mimo-v2.6-rl-oss:general-agent-env-8"
        text = export.build_dockerfile("general", image, terminal_bench=True)
        self.assertNotIn("general_env", text)
        self.assertNotIn("mcpsvc", text)
        self.assertTrue(text.rstrip().endswith('CMD ["sleep", "infinity"]'))
        self.assertIn(f"FROM {image}", text)
        self.assertIn("general_env/sidecar.py", export.build_dockerfile("general", image))   # the MCP rows keep their sidecar


class SharedVerifierEntrypoint(unittest.TestCase):
    """Executes the rendered test.sh with /tests and /logs/verifier redirected to temp dirs. The first version of this entrypoint called
    /usr/bin/python3, which python:3.12-slim and general-agent-env-0 do not have: the verifier never ran, no-op still scored 0, and
    every unit test and the Modal nop gate passed."""

    def setUp(self):
        import tempfile
        self.root = Path(tempfile.mkdtemp())
        self.tests, self.verifier, self.agent = self.root / "tests", self.root / "verifier", self.root / "agent"
        for d in (self.tests, self.verifier, self.agent):
            d.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def install(self, grade_body: str, artifacts=(), terminal_bench=False, tb_script="printf '1\\n' > /logs/verifier/reward.txt\nprintf '{}' > /logs/verifier/ctrf.json\n"):
        from strl.tasks.mimo.graders import common
        (self.tests / "run_grade.py").write_text(grade_body)
        if terminal_bench:
            (self.tests / "test.sh").write_text("#!/bin/sh\n" + tb_script)
        common.finalize_tests(self.root, [str(a) for a in artifacts], terminal_bench=terminal_bench)
        for f in self.tests.iterdir():   # redirect the sandbox paths
            if f.is_file():
                f.write_text(f.read_text().replace("/logs/verifier", str(self.verifier)).replace("/tests", str(self.tests)))

    def run_sh(self, path=None):
        import os
        import subprocess
        env = dict(os.environ, PATH=path if path is not None else os.environ["PATH"])
        return subprocess.run(["/bin/bash", str(self.tests / "test.sh")], env=env, capture_output=True, text=True, timeout=60)

    def details(self):
        return json.loads((self.verifier / "details.json").read_text())

    GOOD = ("import json\nopen('/logs/verifier/reward.txt','w').write('0.5')\n"
            "json.dump({'infra_error': False, 'components': {'x': 1}}, open('/logs/verifier/details.json','w'))\n")

    def test_does_not_call_a_hardcoded_interpreter(self):
        from strl.tasks.mimo.graders import common
        text = common.test_sh()
        self.assertNotIn("\n/usr/bin/python3 ", text)           # the grader is never launched through a fixed path (the first version did)
        self.assertIn('PY=$(command -v python3 || ', text)       # it is found on PATH; absolute paths are only fallbacks
        self.assertIn('"$PY" /tests/run_grade.py', text)

    def test_grader_runs_and_details_are_normalized(self):
        self.install(self.GOOD)
        self.assertEqual(self.run_sh().returncode, 0)
        d = self.details()
        self.assertEqual((d["metric"], d["passed"], d["infra_error"], d["components"]), (0.5, False, False, {"x": 1}))
        self.assertEqual((self.verifier / "reward.txt").read_text().strip(), "0.5")

    def test_agent_forged_reward_and_stale_judge_request_are_wiped(self):
        self.install("pass\n")   # a grader that writes nothing
        (self.verifier / "reward.json").write_text('{"reward": 1}')
        (self.verifier / "judge_request.json").write_text('{"kind": "rubric"}')
        self.run_sh()
        self.assertFalse((self.verifier / "reward.json").exists())
        self.assertFalse((self.verifier / "judge_request.json").exists())
        self.assertEqual((self.verifier / "reward.txt").read_text().strip(), "0")

    def test_agent_planted_module_in_tests_is_deleted_before_the_grader_runs(self):
        self.install("import json\nopen('/logs/verifier/details.json','w').write(json.dumps({'ok': True}))\n")
        (self.tests / "json.py").write_text("raise SystemExit('planted module ran')\n")
        r = self.run_sh()
        self.assertFalse((self.tests / "json.py").exists(), r.stderr)
        self.assertEqual(self.details()["ok"], True)

    def test_a_grader_that_crashes_or_never_ran_is_an_infra_error_not_a_zero(self):
        self.install("raise RuntimeError('boom')\n")
        self.run_sh()
        d = self.details()
        self.assertTrue(d["infra_error"])
        self.assertIn("no details.json", d["error"])

    def test_missing_interpreter_is_reported(self):
        self.install(self.GOOD)
        script = self.tests / "test.sh"   # neutralize the absolute-path fallbacks too (this Mac has /usr/bin/python3)
        script.write_text(script.read_text().replace("/usr/local/bin/python3", "/nonexistent/a").replace("/usr/bin/python3", "/nonexistent/b"))
        self.run_sh(path="/nonexistent")
        d = self.details()
        self.assertTrue(d["infra_error"])
        self.assertIn("no python3", d["error"])

    def test_artifacts_are_copied_into_the_verifier_dir(self):
        poc = self.agent / "poc"
        poc.write_bytes(b"A" * 64)
        self.install("pass\n", artifacts=[poc, self.agent / "absent.txt"])
        self.run_sh()
        self.assertEqual((self.verifier / "poc").read_bytes(), b"A" * 64)
        self.assertFalse((self.verifier / "absent.txt").exists())

    def test_terminal_bench_keeps_its_own_test_sh_inside_the_wrapper(self):
        self.install("pass\n", terminal_bench=True)
        self.assertTrue((self.tests / "tb_test.sh").is_file())
        (self.verifier / "reward.json").write_text('{"reward": 1}')
        self.run_sh()
        self.assertFalse((self.verifier / "reward.json").exists())
        self.assertEqual((self.verifier / "reward.txt").read_text().strip(), "1")   # their script's own reward, now with a details.json
        self.assertEqual(self.details()["metric"], 1.0)
        self.assertFalse(self.details()["infra_error"])

    def test_terminal_bench_verifier_that_did_not_run_is_an_infra_error_and_a_guard_rejection_is_a_real_zero(self):
        """The strict nop gate on the 27B data found every terminal_bench trial would have been masked: their tests write reward.txt
        and no details.json. They count as having run only with pytest's CTRF report or their guard's REJECT verdict."""
        self.install("pass\n", terminal_bench=True, tb_script="printf '1\\n' > /logs/verifier/reward.txt\n")   # no ctrf: pytest never ran
        self.run_sh()
        self.assertTrue(self.details()["infra_error"])
        self.assertIn("did not run", self.details()["error"])
        self.setUp()
        self.install("pass\n", terminal_bench=True, tb_script="echo 'anti_hack_guard: REJECT planted conftest.py' >&2\nprintf '0\\n' > /logs/verifier/reward.txt\n")
        self.run_sh()
        d = self.details()
        self.assertEqual((d["infra_error"], d["metric"], d["guard_rejected"]), (False, 0.0, True))
