"""MiMo metrics and media: per-domain/macro task metrics from eval records, trial summaries -> tracking rows, and the images
(music piano rolls from ABC/MIDI, webdev screenshots) for the `qualitative/media` W&B table. numpy + zlib only; PIL/wandb are
touched only inside ``wandb_image`` (lazily, in the training pod)."""
from __future__ import annotations

import json
import re
import struct
import zlib
from pathlib import Path
from typing import Any

from .data import DOMAINS

ARTIFACT_CHARS = 4000
MEDIA_ROWS = 8


# ------------------------------------------------------------------------------------------------------------ task metrics
def passed(summary: dict) -> bool:
    return bool(summary.get('passed', (summary.get('metric') or 0.0) >= 0.999))


def domain_metrics(records: list[dict], prefix: str = 'eval') -> dict[str, float]:
    """Per-domain and macro task metrics from eval records {domain, task_metric, passed, masked, judge_variance?}.

    Masked trials (infrastructure or judge failure) are excluded from the metric and counted in ``<prefix>/<d>/num_masked``: a
    failed judge is never a zero. ``<prefix>/macro_task_metric`` averages the domains that have at least one valid trial."""
    out: dict[str, float] = {}
    means = []
    for domain in DOMAINS:
        rows = [r for r in records if r.get('domain') == domain]
        if not rows:
            continue
        valid = [r for r in rows if not r.get('masked')]
        out[f'{prefix}/{domain}/num_trials'] = float(len(rows))
        out[f'{prefix}/{domain}/num_masked'] = float(len(rows) - len(valid))
        if valid:
            mean = sum(float(r.get('task_metric') or 0.0) for r in valid) / len(valid)
            out[f'{prefix}/{domain}/task_metric'] = mean
            out[f'{prefix}/{domain}/pass_rate'] = sum(bool(r.get('passed')) for r in valid) / len(valid)
            means.append(mean)
        variances = [float(r['judge_variance']) for r in valid if r.get('judge_variance') is not None]
        if variances:
            out[f'{prefix}/{domain}/judge_variance'] = sum(variances) / len(variances)
        reps = _rep_spread(valid)
        if reps is not None:
            out[f'{prefix}/{domain}/rep_std'] = reps
    if means:
        out[f'{prefix}/macro_task_metric'] = sum(means) / len(means)
    return out


def _rep_spread(rows: list[dict]) -> float | None:
    """Mean within-task std of the metric across repeats (sampling noise of the eval itself)."""
    by_task: dict[str, list[float]] = {}
    for row in rows:
        by_task.setdefault(str(row.get('task')), []).append(float(row.get('task_metric') or 0.0))
    spreads = []
    for values in by_task.values():
        if len(values) > 1:
            mean = sum(values) / len(values)
            spreads.append((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5)
    return sum(spreads) / len(spreads) if spreads else None


def env_metrics_row(summary: dict | None, domain: str | None = None) -> dict[str, Any]:
    """One trajectory summary -> the per-row env_metrics dict the tracking/critic code reads (keys: task_metric, failure/<mode>,
    infra_error plus the table columns)."""
    s = summary or {}
    row: dict[str, Any] = {'domain': s.get('domain') or domain, 'task_metric': None if s.get('masked') else s.get('metric'),
                           'failed': bool(s) and not s.get('masked') and not passed(s) and (s.get('metric') or 0.0) < 0.5,
                           'infra_error': 1.0 if s.get('masked') else 0.0, 'artifact': s.get('artifact'),
                           'judge_rationale': (s.get('judge') or {}).get('rationale'), 'screenshot': s.get('screenshot'),
                           'components': json.dumps(s.get('components') or {}, sort_keys=True)[:1000]}
    if s.get('masked'):
        row['failure/' + str(s.get('mask_reason') or 'masked')] = 1.0
    elif s and not passed(s):
        row['failure/' + str(s.get('stop') if s.get('stop') not in (None, 'complete') else 'wrong')] = 1.0
    return row


EXTRA_COLUMNS = ('domain', 'artifact', 'judge_rationale', 'verifier_components', 'screenshot')


def extra_values(index: int, env_metrics: list | None) -> tuple:
    m = (env_metrics[index] if env_metrics and index < len(env_metrics) else None) or {}
    return m.get('domain'), (m.get('artifact') or '')[:ARTIFACT_CHARS] or None, m.get('judge_rationale'), m.get('components'), m.get('screenshot')


# ------------------------------------------------------------------------------------------------------------ PNG
def encode_png(pixels: Any) -> bytes:
    """RGB image (HxWx3 uint8 numpy array or nested lists) -> PNG bytes, no PIL."""
    import numpy as np
    array = np.asarray(pixels, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError('encode_png expects an HxWx3 RGB image')
    height, width = array.shape[:2]
    raw = b''.join(b'\x00' + array[y].tobytes() for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xFFFFFFFF)

    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b'')


# ------------------------------------------------------------------------------------------------------------ ABC / MIDI -> notes
STEPS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
SHARP_ORDER = 'FCGDAEB'
KEY_SHARPS = {'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7, 'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5, 'Gb': -6, 'Cb': -7}
MODE_OFFSET = {'': 0, 'maj': 0, 'ion': 0, 'm': -3, 'min': -3, 'aeo': -3, 'mix': -1, 'dor': -2, 'phr': -4, 'lyd': 1, 'loc': -5}
NOTE_RE = re.compile(r'(?P<acc>[\^_=]*)(?P<pitch>[A-Ga-g])(?P<oct>[,\']*)(?P<len>\d*(?:/\d*)?)')


def _fraction(text: str, default: float = 1.0) -> float:
    if not text:
        return default
    if '/' not in text:
        return float(text)
    num, _, den = text.partition('/')
    return (float(num) if num else 1.0) / (float(den) if den else 2.0)


def _key_signature(spec: str) -> dict[str, int]:
    match = re.match(r'\s*([A-G][#b]?)\s*([A-Za-z]*)', spec)
    if not match:
        return {}
    tonic, mode = match.group(1), match.group(2).lower()[:3]
    sharps = KEY_SHARPS.get(tonic, 0) + MODE_OFFSET.get(mode, 0) if mode in MODE_OFFSET else KEY_SHARPS.get(tonic, 0)
    if sharps >= 0:
        return {n: 1 for n in SHARP_ORDER[:sharps]}
    return {n: -1 for n in SHARP_ORDER[::-1][:-sharps]}


def notes_from_abc(text: str) -> list[dict]:
    """Minimal ABC reader -> [{voice, start, dur, pitch}] (start/dur in quarter notes). Handles header fields (M, L, K, V), multiple
    voices, accidentals/octaves/lengths, rests, chords; ignores decorations, grace notes, tuplet scaling and repeats."""
    meter, explicit, unit, signature, voice = 1.0, None, 0.5, {}, '1'
    clock: dict[str, float] = {}
    notes: list[dict] = []
    body = False
    length = re.compile(r'\d*(?:/\d*)?')
    for line in text.splitlines():
        line = line.split('%', 1)[0].rstrip()
        field = re.match(r'^([A-Za-z]):\s*(.*)$', line)
        if field:
            key, value = field.group(1), field.group(2)
            if key == 'M':
                num, _, den = value.partition('/')
                meter = int(num) / int(den) if num.isdigit() and den.isdigit() and int(den) else 1.0
            elif key == 'L' and '/' in value:
                explicit = _fraction(value.strip()) * 4         # ABC L:1/8 -> an eighth = 0.5 quarter notes
                unit = explicit
            elif key == 'K':
                signature, body = _key_signature(value), True
                unit = explicit if explicit is not None else (0.25 if meter < 0.75 else 0.5)
            elif key == 'V':
                voice = value.split()[0] if value.split() else '1'
            continue
        if not body or not line.strip():
            continue
        line = re.sub(r'"[^"]*"|![^!]*!|\{[^}]*\}|\[[A-Za-z]:[^\]]*\]', '', line)
        pos = 0
        while pos < len(line):
            if line[pos] == '[' and line.find(']', pos) > pos:
                end = line.find(']', pos)
                tail = length.match(line, end + 1).group(0)
                start = clock.get(voice, 0.0)
                for match in NOTE_RE.finditer(line[pos + 1:end]):
                    notes.append({'voice': voice, 'start': start, 'dur': unit * _fraction(tail) * _fraction(match.group('len')),
                                  'pitch': _pitch(match, signature)})
                clock[voice] = start + unit * _fraction(tail)
                pos = end + 1 + len(tail)
                continue
            rest = re.match(r'[zZxX](\d*(?:/\d*)?)', line[pos:])
            if rest:
                clock[voice] = clock.get(voice, 0.0) + unit * _fraction(rest.group(1))
                pos += rest.end()
                continue
            match = NOTE_RE.match(line, pos)
            if match:
                dur = unit * _fraction(match.group('len'))
                notes.append({'voice': voice, 'start': clock.get(voice, 0.0), 'dur': dur, 'pitch': _pitch(match, signature)})
                clock[voice] = clock.get(voice, 0.0) + dur
                pos = match.end()
                continue
            pos += 1
    return notes


def _pitch(match: re.Match, signature: dict[str, int]) -> int:
    letter = match.group('pitch')
    base = 60 + STEPS[letter.upper()] + (12 if letter.islower() else 0)
    base += 12 * match.group('oct').count("'") - 12 * match.group('oct').count(',')
    acc = match.group('acc')
    if '=' in acc:
        return base
    if acc:
        return base + acc.count('^') - acc.count('_')
    return base + signature.get(letter.upper(), 0)


def notes_from_midi(data: bytes) -> list[dict]:
    """Minimal Standard MIDI File reader -> [{voice, start, dur, pitch}] in quarter notes (voice = track index)."""
    if data[:4] != b'MThd':
        raise ValueError('not a MIDI file')
    division = struct.unpack('>H', data[12:14])[0]
    if division & 0x8000:
        raise ValueError('SMPTE time division is not supported')
    pos, track, notes = 14, 0, []
    while pos + 8 <= len(data):
        kind, size = data[pos:pos + 4], struct.unpack('>I', data[pos + 4:pos + 8])[0]
        chunk, pos = data[pos + 8:pos + 8 + size], pos + 8 + size
        if kind != b'MTrk':
            continue
        tick, i, status, active = 0, 0, 0, {}
        while i < len(chunk):
            delta = 0
            while True:
                byte = chunk[i]; i += 1
                delta = (delta << 7) | (byte & 0x7F)
                if not byte & 0x80:
                    break
            tick += delta
            if chunk[i] & 0x80:
                status = chunk[i]; i += 1
            if status == 0xFF:
                i += 1
                length = chunk[i]; i += 1 + length
            elif status in (0xF0, 0xF7):
                length = chunk[i]; i += 1 + length
            else:
                kind_ = status & 0xF0
                size_ = 1 if kind_ in (0xC0, 0xD0) else 2
                a, b = chunk[i], (chunk[i + 1] if size_ == 2 else 0)
                i += size_
                if kind_ == 0x90 and b > 0:
                    active[(status & 0x0F, a)] = tick
                elif kind_ in (0x80, 0x90) and (status & 0x0F, a) in active:
                    start = active.pop((status & 0x0F, a))
                    notes.append({'voice': str(track), 'start': start / division, 'dur': max(tick - start, 1) / division, 'pitch': a})
        track += 1
    return notes


PALETTE = [(86, 180, 233), (230, 159, 0), (0, 158, 115), (240, 228, 66), (204, 121, 167), (213, 94, 0), (0, 114, 178), (150, 150, 150)]


def piano_roll(notes: list[dict], width: int = 640, height: int = 320) -> Any:
    """Notes -> HxWx3 uint8 numpy piano roll (time right, pitch up, one colour per voice, bar-free). Empty input -> blank canvas."""
    import numpy as np
    image = np.full((height, width, 3), 24, dtype=np.uint8)
    if not notes:
        return image
    low, high = min(n['pitch'] for n in notes) - 1, max(n['pitch'] for n in notes) + 2
    total = max(n['start'] + n['dur'] for n in notes) or 1.0
    voices = sorted({n['voice'] for n in notes})
    row_h = height / (high - low)
    for note in notes:
        x0 = int(note['start'] / total * (width - 1))
        x1 = max(x0 + 1, int((note['start'] + note['dur']) / total * (width - 1)))
        y1 = height - int((note['pitch'] - low) * row_h)
        y0 = max(0, y1 - max(1, int(row_h)))
        image[y0:y1, x0:x1] = PALETTE[voices.index(note['voice']) % len(PALETTE)]
    return image


def wandb_image(source: Any, caption: str = '') -> Any:
    """wandb.Image from a path or an RGB array; None when wandb/PIL are unavailable (media must never break logging)."""
    try:
        import wandb
        return wandb.Image(str(source) if isinstance(source, (str, Path)) else source, caption=caption[:200])
    except Exception:
        return None


def media_for(summary: dict, trial_dir: str | Path | None = None) -> Any:
    """The image of one trial: webdev screenshot (path) or a music piano roll from the ABC/MIDI artifact; None otherwise."""
    domain = summary.get('domain')
    if domain == 'webdev' and summary.get('screenshot') and Path(summary['screenshot']).is_file():
        return summary['screenshot']
    if domain == 'music' and summary.get('artifact'):
        try:
            return piano_roll(notes_from_abc(summary['artifact']))
        except Exception:
            return None
    return None


def media_table(records: list[dict], limit: int = MEDIA_ROWS) -> tuple[list[str], list[list]] | None:
    """Rows for `qualitative/media`: <= ``limit`` webdev screenshots / music piano rolls, evenly split between the two domains."""
    columns = ['domain', 'task', 'rep', 'task_metric', 'image']
    picks: list[tuple[dict, Any]] = []
    for domain in ('webdev', 'music'):
        mine = [r for r in records if r.get('domain') == domain and not r.get('masked')]
        for record in mine[:limit // 2]:
            full = (record.get('extra') or {}).get('artifact_full') or record.get('artifact')
            source = media_for({'domain': domain, 'screenshot': record.get('screenshot'), 'artifact': full})
            if source is not None:
                picks.append((record, source))
    rows = []
    for record, source in picks[:limit]:
        image = wandb_image(source, f'{record.get("domain")} {record.get("task")} metric={record.get("task_metric")}')
        if image is not None:
            rows.append([record.get('domain'), record.get('task'), record.get('rep'), record.get('task_metric'), image])
    return (columns, rows) if rows else None
