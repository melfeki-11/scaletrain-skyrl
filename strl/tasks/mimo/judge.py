"""LLM judge for the judged MiMo domains (general: rubric, webdev: vision), run OUTSIDE the sandbox with the LiteLLM key.

    environment:
      mimo:
        judge: {model: openai/gpt-6-luna, vision_model: null, votes: 1, eval_votes: 3, concurrency: 16, timeout_s: 120, retries: 2}

``Judge.score(request)`` takes the verifier's ``judge_request.json`` (see strl/tasks/mimo/__init__.py) and returns
{'scores', 'final', 'rationale', 'dims', 'meta'}; ``combine(request, scores)`` is the contract formula
    final = (base_score + sum_i w_i * s_i) / total_weight .
Every failure (transport after retries, unparsable verdict, missing image, no usable vote) raises ``JudgeUnavailable``: the
generator masks that trial; a judge outage is never a zero.

rubric: MiMo's text judge. One call per item (isolated verdicts), temperature 0, the item's pass anchor, evidence = the item's
        ``answer`` text; median of ``votes`` binary verdicts.
vision: MiMo's webdev evaluation rubric (rubric_prompts.py, verbatim) on the verifier's screenshot; one call returns seven
        dimensions; scores = {visual, query_fulfillment, premium_assets}, median over votes per key.
"""
from __future__ import annotations

import asyncio
import base64
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from ...services import litellm
from . import rubric_prompts

DEFAULTS = {'model': 'openai/gpt-6-luna', 'vision_model': None, 'votes': 1, 'eval_votes': 3, 'concurrency': 16, 'timeout_s': 120,
            'retries': 2, 'project_id': litellm.PROJECT_ID, 'max_completion_tokens': 1024, 'vision_max_completion_tokens': 2048,
            'max_evidence_chars': 20000, 'query_cap': 1500, 'max_megapixels': 20.0, 'usd_per_mtok_in': 2.0, 'usd_per_mtok_out': 8.0,
            'backoff_s': 2.0, 'margin_s': 10.0}      # margin_s: the asyncio deadline is timeout_s + margin_s (the HTTP client enforces timeout_s)
VISION_KEYS = ('visual', 'query_fulfillment', 'premium_assets')
RUBRIC_PROMPT = """你在核验一份工作交付物是否满足一个**客观检查点**(有唯一对错,不是审美)。
下面给出交付物相关文件的抽取文本,以及检查点。严格判断:满足=1,不满足=0。
**只判 0 或 1,不给部分分**(部分满足=不满足)。**只依据给出的文本证据**,看不到证据就判 0;不要臆测。
证据文本可能含【已截断…】标记或「⚠ 未抽取内容」清单:若检查点的目标内容恰好落在被截断/
未抽取的部分,判 0,并在 why 里写明「证据截断」或「证据未抽取」(供审计区分,仍不要臆测给分)。

【交付物文本】
{evidence}

【检查点】
{items}

只输出一个 json 对象(可包在 json 代码块):
{"results": {"检查点id": {"score": 0, "why": "≤25字依据"}}}"""


class JudgeUnavailable(RuntimeError):
    """The judge gave no usable verdict: the trial is masked (infrastructure failure), never scored 0."""


def judge_settings(cfg: dict | None) -> dict:
    """DEFAULTS overridden by ``environment.mimo.judge`` of a resolved experiment (or by a plain settings dict)."""
    raw = cfg or {}
    if 'environment' in raw:
        raw = ((raw['environment'].get('mimo') or {}).get('judge')) or {}
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown judge settings {sorted(unknown)} (allowed: {sorted(DEFAULTS)})')
    return {**DEFAULTS, **raw}


def settings_problems(cfg: dict) -> list[str]:
    raw = ((cfg.get('environment') or {}).get('mimo') or {}).get('judge') or {}
    problems = [f'environment.mimo.judge.{k} is not a judge setting (allowed: {sorted(DEFAULTS)})' for k in raw if k not in DEFAULTS]
    for key in ('votes', 'eval_votes', 'concurrency', 'retries'):
        if key in raw and (not isinstance(raw[key], int) or raw[key] < (0 if key == 'retries' else 1)):
            problems.append(f'environment.mimo.judge.{key} must be a positive integer')
    if 'timeout_s' in raw and not (isinstance(raw['timeout_s'], (int, float)) and raw['timeout_s'] > 0):
        problems.append('environment.mimo.judge.timeout_s must be > 0')
    return problems


# ------------------------------------------------------------------------------------------------------------ parsing
def extract_json(text: str) -> dict | None:
    """First JSON object in ``text`` (judges wrap verdicts in fences or prose)."""
    if not text or '{' not in text:
        return None
    return litellm.parse_json_object(text)


def parse_rubric_verdict(text: str, item_id: str) -> tuple[float, str] | None:
    """MiMo's text-judge verdict {"results": {id: {"score": 0|1, "why": ...}}} -> (score in [0,1], why); None when unusable."""
    data = extract_json(text)
    entry = ((data or {}).get('results') or {}).get(item_id)
    if not isinstance(entry, dict) or not isinstance(entry.get('score'), (int, float)) or isinstance(entry.get('score'), bool):
        return None
    return max(0.0, min(1.0, float(entry['score']))), str(entry.get('why', ''))[:200]


def parse_vision_verdict(text: str) -> dict | None:
    """MiMo's seven-dimension verdict -> {'dims', 'visual', 'query_fulfillment', 'premium_assets', 'reason'}. A missing or
    non-numeric dimension makes the verdict unusable (never defaulted to 0)."""
    data = extract_json(text)
    if data is None:
        return None
    dims: dict[str, float] = {}
    for key in rubric_prompts.ALL_KEYS:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        dims[key] = max(0.0, min(1.0, float(value)))
    visual = sum(dims[k] for k in rubric_prompts.VISUAL_KEYS) / len(rubric_prompts.VISUAL_KEYS)
    return {'dims': dims, 'visual': round(visual, 4), 'query_fulfillment': dims['query_fulfillment'], 'premium_assets': dims['premium_assets'],
            'reason': str(data.get('reason', ''))[:500]}


def vision_final(scores: dict[str, float]) -> float:
    """MiMo's evaluation score: the equal-weight mean of visual, query_fulfillment, premium_assets."""
    return sum(scores[k] for k in VISION_KEYS) / len(VISION_KEYS)


# ------------------------------------------------------------------------------------------------------------ the contract formula
def combine(request: dict, scores: dict[str, float]) -> float:
    """final = (base_score + sum_i w_i * s_i) / total_weight over the request's items. Vision requests may omit ``items``: the
    three vision keys then weigh 1/3 each with total_weight 1 and base 0 (the MiMo evaluation score)."""
    if request.get('force_zero'):   # an in-sandbox gate failed (general: anti-tamper / required rule): never judged, never positive
        return 0.0
    items = request.get('items')
    if request.get('kind') == 'vision':   # the vision score is MiMo's fixed formula; any `items` only carry the rubric text
        return max(0.0, min(1.0, float(request.get('base_score', 0.0)) + vision_final(scores)))
    total = float(request.get('total_weight') or 0.0)
    if total <= 0:
        raise ValueError('judge_request.total_weight must be > 0')
    missing = [it['id'] for it in items or [] if it['id'] not in scores]
    if missing:
        raise ValueError(f'no judge score for items {missing}')
    judged = sum(float(it.get('weight', 1.0)) * max(0.0, min(1.0, float(scores[it['id']]))) for it in items or [])
    return max(0.0, min(1.0, (float(request.get('base_score', 0.0)) + judged) / total))


# ------------------------------------------------------------------------------------------------------------ images
def png_size(data: bytes) -> tuple[int, int] | None:
    if data[:8] == b'\x89PNG\r\n\x1a\n' and len(data) >= 24:
        return int.from_bytes(data[16:20], 'big'), int.from_bytes(data[20:24], 'big')
    return None


def data_url(data: bytes) -> str:
    mime = 'image/png' if data[:8] == b'\x89PNG\r\n\x1a\n' else 'image/jpeg' if data[:2] == b'\xff\xd8' else 'application/octet-stream'
    if mime == 'application/octet-stream':
        raise JudgeUnavailable('screenshot is neither PNG nor JPEG')
    return f'data:{mime};base64,{base64.b64encode(data).decode()}'


def shrink_png(data: bytes, max_megapixels: float) -> bytes:
    """Proportional downscale above ``max_megapixels`` (the judge refuses ~46 MP images permanently). Needs PIL; without it (or for
    JPEG) the image is sent unchanged: a missing optional dependency must not become a masked trial."""
    size = png_size(data)
    if size is None or size[0] * size[1] <= max_megapixels * 1e6:
        return data
    try:
        import io
        from PIL import Image
    except ImportError:
        return data
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(io.BytesIO(data)).convert('RGB')
    k = (max_megapixels * 1e6 / (image.width * image.height)) ** 0.5
    out = io.BytesIO()
    image.resize((max(1, int(image.width * k)), max(1, int(image.height * k)))).save(out, 'PNG')
    return out.getvalue()


# ------------------------------------------------------------------------------------------------------------ the judge
class Judge:
    """Async judge. ``chat(messages, **kw) -> {'text','usage','cost_usd'}`` is injectable (tests); default: Scale LiteLLM."""

    def __init__(self, settings: dict | None = None, chat: Callable[..., dict] | None = None):
        self.cfg = judge_settings(settings)
        self._chat = chat
        self._key: str | None = None
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: Any = None
        self._no_temperature: set[str] = set()

    # -- plumbing
    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem, self._sem_loop = asyncio.Semaphore(int(self.cfg['concurrency'])), loop
        return self._sem

    def _call(self, messages: list[dict], model: str, max_tokens: int) -> dict:
        kwargs = dict(model=model, max_completion_tokens=max_tokens, project_id=self.cfg['project_id'], timeout=float(self.cfg['timeout_s']),
                      tag='strl-mimo-judge')
        if self._chat is not None:
            return self._chat(messages, temperature=0, **kwargs)
        if self._key is None:
            self._key = litellm.resolve_key()
        try:
            if model not in self._no_temperature:
                return litellm.chat(messages, key=self._key, temperature=0, **kwargs)
        except litellm.LiteLLMError as error:
            if 'HTTP 400' not in str(error):
                raise
            self._no_temperature.add(model)      # reasoning models reject temperature: remember, retry without
        return litellm.chat(messages, key=self._key, **kwargs)

    async def _ask(self, messages: list[dict], model: str, max_tokens: int, parse: Callable[[str], Any], meta: dict) -> Any:
        """One verdict: up to retries+1 attempts over transport errors and unusable output; raises JudgeUnavailable."""
        attempts, last = int(self.cfg['retries']) + 1, 'no attempt'
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(float(self.cfg['backoff_s']) * attempt)
            started = time.monotonic()
            try:
                async with self._semaphore():
                    response = await asyncio.wait_for(asyncio.to_thread(self._call, messages, model, max_tokens),
                                                      float(self.cfg['timeout_s']) + float(self.cfg['margin_s']))
            except (litellm.LiteLLMError, asyncio.TimeoutError, OSError) as error:
                last = f'{type(error).__name__}: {error}'[:160]
                continue
            usage = response.get('usage') or {}
            cost = response.get('cost_usd')
            meta['calls'] += 1
            meta['latency_s'] += time.monotonic() - started
            meta['prompt_tokens'] += int(usage.get('prompt_tokens', 0))
            meta['completion_tokens'] += int(usage.get('completion_tokens', 0))
            meta['cost_usd'] += float(cost) if cost is not None else (
                usage.get('prompt_tokens', 0) * self.cfg['usd_per_mtok_in'] + usage.get('completion_tokens', 0) * self.cfg['usd_per_mtok_out']) / 1e6
            parsed = parse(response.get('text', ''))
            if parsed is not None:
                return parsed
            last = 'unparsable verdict'
        raise JudgeUnavailable(f'judge gave no usable verdict after {attempts} attempts ({last})')

    def _meta(self, model: str, votes: int) -> dict:
        return {'model': model, 'votes': votes, 'calls': 0, 'latency_s': 0.0, 'prompt_tokens': 0, 'completion_tokens': 0, 'cost_usd': 0.0}

    # -- scoring
    async def score(self, request: dict, *, workdir: str | Path | None = None, votes: int | None = None, eval: bool = False) -> dict:
        votes = int(votes or (self.cfg['eval_votes'] if eval else self.cfg['votes']))
        kind = request.get('kind')
        if self._chat is None and self._key is None:      # a missing key is deterministic: fail fast, do not burn retries
            try:
                self._key = await asyncio.to_thread(litellm.resolve_key)
            except litellm.LiteLLMError as error:
                raise JudgeUnavailable(str(error)) from None
        if kind == 'rubric':
            result = await self._score_rubric(request, votes)
        elif kind == 'vision':
            result = await self._score_vision(request, votes, Path(workdir) if workdir else None)
        else:
            raise JudgeUnavailable(f'unknown judge kind {kind!r}')
        try:
            result['final'] = combine(request, result['scores'])
        except ValueError as error:
            raise JudgeUnavailable(str(error)) from None
        return result

    async def _score_rubric(self, request: dict, votes: int) -> dict:
        meta = self._meta(self.cfg['model'], votes)
        items = [it for it in request.get('items') or [] if it.get('question') is not None]
        cap = int(self.cfg['max_evidence_chars'])

        async def one(item: dict) -> tuple[str, float, str]:
            evidence = str(item.get('answer', request.get('answer', '')))[:cap]
            line = f'- [{item["id"]}] {item["question"]}(满足=1:{item.get("anchor", item.get("pass_anchor", ""))})'
            messages = [{'role': 'user', 'content': RUBRIC_PROMPT.replace('{evidence}', evidence).replace('{items}', line)}]
            verdicts = await asyncio.gather(*[self._ask(messages, self.cfg['model'], int(self.cfg['max_completion_tokens']),
                                                        lambda text, i=item['id']: parse_rubric_verdict(text, i), meta) for _ in range(votes)])
            return item['id'], round(statistics.median(v[0] for v in verdicts), 4), verdicts[0][1]

        rows = await asyncio.gather(*[one(it) for it in items])
        return {'scores': {i: s for i, s, _ in rows}, 'rationale': '; '.join(f'{i}: {why}' for i, _, why in rows)[:1500], 'dims': None, 'meta': meta}

    async def _score_vision(self, request: dict, votes: int, workdir: Path | None) -> dict:
        model = self.cfg['vision_model'] or self.cfg['model']
        meta = self._meta(model, votes)
        images = request.get('images') or []
        if not images:
            raise JudgeUnavailable('vision judge_request has no images')
        content: list[dict] = []
        for ref in images:
            path = Path(ref)
            if workdir is not None and not path.is_file():
                path = workdir / path.name                    # '/logs/verifier/screenshot.png' -> <trial>/verifier/screenshot.png
            if not path.is_file():
                raise JudgeUnavailable(f'screenshot {ref} not found')
            content.append({'type': 'image_url', 'image_url': {'url': data_url(shrink_png(path.read_bytes(), float(self.cfg['max_megapixels'])))}})
        prompt = rubric_prompts.build_prompt().format(query=str(request.get('query', ''))[:int(self.cfg['query_cap'])])
        messages = [{'role': 'user', 'content': [*content, {'type': 'text', 'text': prompt}]}]
        verdicts = await asyncio.gather(*[self._ask(messages, model, int(self.cfg['vision_max_completion_tokens']), parse_vision_verdict, meta)
                                          for _ in range(votes)])
        scores = {k: round(statistics.median(v[k] for v in verdicts), 4) for k in VISION_KEYS}
        dims = {k: round(statistics.median(v['dims'][k] for v in verdicts), 4) for k in rubric_prompts.ALL_KEYS}
        return {'scores': scores, 'rationale': verdicts[0]['reason'], 'dims': dims, 'meta': meta}

    async def rescore_variance(self, request: dict, *, workdir: str | Path | None = None, n: int = 2) -> dict:
        """Judge the same payload ``n`` times independently (1 vote each): per-key values and population variance, the judge-noise
        floor the eval reports (``environment/<domain>/judge_variance``)."""
        runs = await asyncio.gather(*[self.score(request, workdir=workdir, votes=1) for _ in range(n)])
        finals = [r['final'] for r in runs]
        keys = sorted(runs[0]['scores'])
        return {'finals': finals, 'variance': statistics.pvariance(finals) if n > 1 else 0.0, 'max_abs_diff': max(finals) - min(finals),
                'per_key': {k: [r['scores'][k] for r in runs] for k in keys}, 'cost_usd': sum(r['meta']['cost_usd'] for r in runs)}


# ------------------------------------------------------------------------------------------------------------ selftest
def selftest(settings: dict | None = None, chat: Callable[..., dict] | None = None) -> dict:
    """ONE tiny real call per kind: a rubric item with a trivial anchor, and a vision call on an in-memory 64x64 PNG. Returns a
    receipt {kind: {ok, model, latency_s, tokens, cost_usd, final|error}}; raises JudgeUnavailable only via the receipt's ok flag."""
    from .metrics import encode_png
    judge = Judge(settings, chat=chat)
    rubric = {'kind': 'rubric', 'base_score': 0.0, 'base_weight': 0.0, 'total_weight': 1.0,
              'items': [{'id': 'sum', 'weight': 1.0, 'question': 'Does the deliverable state that 2 + 2 equals 4?',
                         'anchor': 'The deliverable says 2 + 2 = 4.', 'answer': 'Result: 2 + 2 = 4.'}]}
    pixels = [[(200, 30, 30) if y < 24 else (240, 240, 240) for x in range(64)] for y in range(64)]
    vision = {'kind': 'vision', 'base_score': 0.0, 'base_weight': 0.0, 'total_weight': 1.0, 'images': ['selftest.png'],
              'query': 'A tiny landing page with a red header banner.'}
    import tempfile
    receipt: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as temp:
        (Path(temp) / 'selftest.png').write_bytes(encode_png(pixels))
        for kind, request in (('rubric', rubric), ('vision', vision)):
            try:
                result = asyncio.run(judge.score(request, workdir=temp, votes=1))
                meta = result['meta']
                receipt[kind] = {'ok': True, 'model': meta['model'], 'final': round(result['final'], 3), 'scores': result['scores'], 'latency_s': round(meta['latency_s'], 2),
                                 'prompt_tokens': meta['prompt_tokens'], 'completion_tokens': meta['completion_tokens'], 'cost_usd': round(meta['cost_usd'], 5)}
            except Exception as error:
                receipt[kind] = {'ok': False, 'model': judge.cfg['vision_model'] or judge.cfg['model'] if kind == 'vision' else judge.cfg['model'],
                                 'error': f'{type(error).__name__}: {error}'[:300]}
    return receipt
