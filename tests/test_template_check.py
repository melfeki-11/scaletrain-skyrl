"""Step-wise prefix merging depends on the chat template: the real Qwen3.5 and Qwen3.8 templates, rendered without any tokenizer."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jinja2  # noqa: E402
import jinja2.ext  # noqa: E402
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402

from strl.models.template_check import prefix_property  # noqa: E402

TEMPLATES = ROOT / 'tests/fixtures/templates'


class TemplateOnlyTokenizer:
    """apply_chat_template(tokenize=False) the way transformers renders a chat template (sandboxed jinja, raise_exception, tojson)."""
    eos_token = '<|im_end|>'

    def __init__(self, path: Path):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
        env.globals['raise_exception'] = lambda message: (_ for _ in ()).throw(jinja2.exceptions.TemplateError(message))
        self.template = env.from_string(path.read_text())

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert not tokenize
        return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)


class Probe(unittest.TestCase):
    def test_qwen35_history_drops_think_so_nothing_merges(self):
        """Measured on 811 real Terminus-2 turns (can-hb-f): 811 sequences, i.e. zero merges."""
        result = prefix_property(TemplateOnlyTokenizer(TEMPLATES / 'qwen3_5_chat_template.jinja'), {'enable_thinking': False})
        self.assertFalse(result['ok'], result)
        self.assertIn('think', result['detail'])

    def test_qwen38_keeps_history_verbatim_so_turns_merge(self):
        """Same transcripts re-rendered with Qwen3.8's template: 209 sequences for 811 turns (86% of merge points merge)."""
        result = prefix_property(TemplateOnlyTokenizer(TEMPLATES / 'qwen3_8_chat_template.jinja'), {'enable_thinking': False})
        self.assertTrue(result['ok'], result)

    def test_qwen38_with_thinking_on_does_not_merge(self):
        """Thinking on ends the generation prompt with an open <think>: the model's reasoning is not in history -> no merge."""
        result = prefix_property(TemplateOnlyTokenizer(TEMPLATES / 'qwen3_8_chat_template.jinja'), {'enable_thinking': True})
        self.assertFalse(result['ok'], result)

    def test_unrenderable_probe_is_reported_not_raised(self):
        class Broken:
            eos_token = ''

            def apply_chat_template(self, *a, **k):
                raise ValueError('system message required')
        self.assertIsNone(prefix_property(Broken())['ok'])

    def test_plan_warns_for_harbor_when_merging_is_broken(self):
        from unittest import mock

        from strl import experiment
        from strl.derive import plan
        cfg = experiment.load('experiments/harbor-qwen35.yaml')
        broken = {'ok': False, 'detail': 'diverges', 'model': 'x', 'profile': 'p'}
        with mock.patch('strl.models.template_check.check', return_value=broken):
            warnings = plan(cfg)['warnings']
        self.assertTrue(any(w.startswith("HARBOR: x's chat template breaks step-wise prefix merging") for w in warnings), warnings)
        with mock.patch('strl.models.template_check.check', return_value=dict(broken, ok=True)):
            self.assertFalse(any(w.startswith('HARBOR') for w in plan(cfg)['warnings']))
        with mock.patch('strl.models.template_check.check', return_value=None):
            self.assertTrue(any('merge check skipped' in w for w in plan(cfg)['warnings']))


if __name__ == '__main__':
    unittest.main()
