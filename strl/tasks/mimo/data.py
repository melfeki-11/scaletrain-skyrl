"""MiMo data: pool loading, grouped held-out test sets, deterministic selection, manifest, prompt-length report.

Config (``environment.mimo``; every key optional except the domains you want):
    revision: 639865fd                       # dataset revision; parquet sha256 are pinned per revision (PINNED_SHA256)
    seed: 17                                 # drives every ordering below
    eval_repeats: 2                          # each held-out task is evaluated this many times (env_extras.rep)
    canary: 2                                # held-out tasks per domain in the canary validation set
    raw_dir: assets/mimo/raw                 # parquet + image-mapping.jsonl (repo-relative or absolute)
    domains:
      code: {train: 24, weight: 1.0, test: 24, filters: {test_kind: single}, test_filters: {}, max_prompt_tokens: 4096}

Selection rules (all deterministic functions of the pinned data + seed):
  * every row has a GROUP key (cyber: project; code: test-file path family; general: scenario / category+image; webdev: hash
    bucket; music: tag+meter). Oversized groups are split by hash so no group dominates a split.
  * TEST per domain: groups are visited in hash order, each contributes at most TEST_GROUP_CAP rows, until ``test`` rows are
    taken. The test set depends only on (revision, seed, domain, test count, test_filters, screening): never on ``train`` or
    ``filters``. Every group that contributed a test row is excluded from training entirely (no group leakage).
  * TRAIN per domain: rows of non-test groups passing ``filters`` in hash order, first ``train``. Counts are prefix-stable:
    train=12 is the first half of train=24, so ``--set environment.mimo.domains.code.train=12`` needs no re-prepare.
  * CANARY: the first ``canary`` test rows per domain (exported once more under split ``canary``).
  * SCREENING (Track B): ``<data_dir>/screening.json`` ({instance_id: reason} or {domain: {...}}) and/or a ``screen(row)``
    callback drop invalid rows (e.g. code tasks whose unmodified repo already passes) from the pool BEFORE selection; the
    reasons are stored in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

DOMAINS = ('music', 'cyber', 'code', 'general', 'webdev')
REVISION = '639865fd'
PARQUET = {'code': 'code.parquet', 'cyber': 'cyber.parquet', 'general': 'general_train.parquet', 'music': 'music.parquet',
           'webdev': 'webdev.parquet'}
PINNED_SHA256 = {REVISION: {
    'code': 'e15733cf2451cfbc5492a4120f7f8cfddbad818aa9f0b324c79888dd1fece161',
    'cyber': '9e6e69034f4f3e6cfb5c901de642dddc7f4bacf35ab886b0e8eb53eb03084ba1',
    'general': '9bfdc8b05d2bf2cb9e89815443d35507e4be30368aebf70928421795c10259b9',
    'music': '8fa2c95fb008df032eae184f3304dd3a73e3d25727e3aac0a715484859fa4856',
    'webdev': '56ca6c96642666a27b756e20694ba60a4386d1761baadae86fd6768553d2cbac'}}
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / 'assets/mimo/raw'
TEST_GROUP_CAP = 6               # held-out rows one group may contribute (the rest of the group is simply unused)
DEFAULT_TEST = 24
SPLITS = ('train', 'test', 'canary')
SELECTION_KEYS = {'revision', 'seed', 'eval_repeats', 'canary', 'raw_dir', 'domains'}
DOMAIN_KEYS = {'train', 'weight', 'test', 'filters', 'test_filters', 'max_prompt_tokens', 'max_agent_sec', 'max_verifier_sec'}
RUNTIME_KEYS = {'judge', 'step_deadline_sec', 'timeout_policy', 'max_trial_attempts', 'summary_chars'}
FILTER_KEYS = {
    'general': {'source', 'lang', 'category'}, 'code': {'test_kind', 'cwd'}, 'cyber': {'sanitizer', 'project'},
    'music': {'lang', 'length', 'meter', 'tag'}, 'webdev': set()}
LENGTH_FILTERS = {'max_prompt_chars', 'min_prompt_chars'}
GENERIC_DIRS = {'tests', 'test', 'src', 'lib', '__tests__', 'spec', 'specs', 'testing', 'e2e', 'integration', 'unit', 'pkg', 'internal',
                'cmd', 'packages', 'apps', 'python', 'go', 'core', 'api', 'server', 'client', 'common', 'x', 'cli', 'static', 'test-data',
                'source', 'app', 'fixtures'}


class DataError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------------ config
def mimo_config(cfg: dict) -> dict:
    """``environment.mimo`` with defaults. Domains with train=0 are kept (they may still have a test set)."""
    raw = (cfg.get('environment') or {}).get('mimo') or {}
    out = {'revision': raw.get('revision', REVISION), 'seed': int(raw.get('seed', 17)), 'eval_repeats': int(raw.get('eval_repeats', 2)),
           'canary': int(raw.get('canary', 2)), 'raw_dir': raw.get('raw_dir'), 'domains': {}}
    for name, d in (raw.get('domains') or {}).items():
        d = d or {}
        out['domains'][name] = {'train': int(d.get('train', 0)), 'weight': float(d.get('weight', 1.0)), 'test': int(d.get('test', DEFAULT_TEST)),
                                'filters': dict(d.get('filters') or {}), 'test_filters': dict(d.get('test_filters') or {}),
                                'max_prompt_tokens': d.get('max_prompt_tokens'), 'max_agent_sec': d.get('max_agent_sec'),
                                'max_verifier_sec': d.get('max_verifier_sec')}
    return out


def config_problems(cfg: dict) -> list[str]:
    raw = (cfg.get('environment') or {}).get('mimo') or {}
    problems = [f'environment.mimo.{k} is not a known key (allowed: {sorted(SELECTION_KEYS | RUNTIME_KEYS)})'
                for k in raw if k not in SELECTION_KEYS | RUNTIME_KEYS]
    domains = raw.get('domains')
    if not isinstance(domains, dict) or not domains:
        return problems + ['environment.mimo.domains must list at least one domain']
    for name, d in domains.items():
        where = f'environment.mimo.domains.{name}'
        if name not in DOMAINS:
            problems.append(f'{where}: unknown domain (known: {list(DOMAINS)})')
            continue
        d = d or {}
        problems += [f'{where}.{k} is not a known key (allowed: {sorted(DOMAIN_KEYS)})' for k in d if k not in DOMAIN_KEYS]
        for key in ('train', 'test'):
            if key in d and (not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] < 0):
                problems.append(f'{where}.{key} must be an integer >= 0')
        for key in ('max_agent_sec', 'max_verifier_sec', 'max_prompt_tokens'):
            if d.get(key) is not None and (not isinstance(d[key], (int, float)) or isinstance(d[key], bool) or d[key] <= 0):
                problems.append(f'{where}.{key} must be a positive number')
        if not isinstance(d.get('weight', 1.0), (int, float)) or d.get('weight', 1.0) <= 0:
            problems.append(f'{where}.weight must be > 0')
        for key in ('filters', 'test_filters'):
            bad = [k for k in (d.get(key) or {}) if k not in FILTER_KEYS[name] | LENGTH_FILTERS]
            problems += [f'{where}.{key}.{k} is not a filter of {name} (allowed: {sorted(FILTER_KEYS[name] | LENGTH_FILTERS)})' for k in bad]
    if int(raw.get('eval_repeats', 2)) < 1:
        problems.append('environment.mimo.eval_repeats must be >= 1')
    return problems


def selected_domains(mcfg: dict) -> list[str]:
    """Domains with training weight (train > 0), in canonical order."""
    return [d for d in DOMAINS if d in mcfg['domains'] and mcfg['domains'][d]['train'] > 0]


def eval_domains(mcfg: dict) -> list[str]:
    """Domains with a held-out set (a domain with train 0 and test > 0 is an eval-only transfer probe)."""
    return [d for d in DOMAINS if d in mcfg['domains'] and mcfg['domains'][d]['test'] > 0]


def raw_dir(mcfg: dict | None = None, override: str | Path | None = None) -> Path:
    path = Path(override or (mcfg or {}).get('raw_dir') or DEFAULT_RAW_DIR)
    return path if path.is_absolute() else REPO_ROOT / path


# ------------------------------------------------------------------------------------------------------------------ hashing
def sha(*parts: Any) -> str:
    return hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()


def bucket(text: str, n: int) -> int:
    return int(sha(text)[:8], 16) % n


def canonical_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------------------------------ rows
def instance_json(raw: dict) -> dict:
    value = (raw.get('extra_info') or {}).get('instance_json')
    if isinstance(value, str):
        return json.loads(value) if value else {}
    return value or {}


def prompt_text(raw: dict) -> str:
    return '\n'.join(str(m.get('content', '')) for m in (raw.get('prompt') or []))


def task_text(raw: dict) -> str:
    """What the policy has to read: the longer of the dataset prompt and the full problem statement (cyber prompts are one line)."""
    return max(prompt_text(raw), str(instance_json(raw).get('problem_statement') or ''), key=len)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def instance_id(domain: str, raw: dict) -> str:
    info = raw.get('extra_info') or {}
    value = info.get('instance_id') or info.get('src_id') or instance_json(raw).get('instance_id')
    if not value:
        raise DataError(f'{domain} row without an instance id: {sorted(info)}')
    return str(value)


def raw_group(domain: str, raw: dict, iid: str) -> str:
    info, ij = raw.get('extra_info') or {}, instance_json(raw)
    if domain == 'cyber':
        match = re.search(r'in file `([^`]+)`', str(ij.get('problem_statement') or prompt_text(raw)))
        return f'cyber:{match.group(1).split("/")[0].lower()}' if match else f'cyber:hash{bucket(iid, 64)}'
    if domain == 'code':
        files = [f for f in re.findall(r'^diff --git a/(\S+)', str(ij.get('test_patch') or ''), re.M)
                 if not f.startswith('mimo_') and f != 'test_commands.json']
        for path in files:
            dirs = [p for p in path.split('/')[:-1] if p not in GENERIC_DIRS]
            if dirs:
                return 'code:' + '/'.join(dirs[:2])
        return f'code:hash{bucket(iid, 64)}'
    if domain == 'general':
        if str(raw.get('data_source', '')).endswith('terminal_bench'):
            return f'tb:{ij.get("category", "")}:{ij.get("docker_image", iid)}'
        match = re.match(r'(s3k_\d+)_', iid) or re.match(r'(.+?)_t\d+_rl_', iid)
        return f'ga:{match.group(1) if match else iid}'
    if domain == 'webdev':
        return f'webdev:b{bucket(iid, 512)}'
    return f'music:{info.get("tag", "")}|{info.get("meter", "")}'


def refine_groups(rows: list[dict]) -> list[dict]:
    """Split groups holding more than max(8, 2.5% of the pool) rows into hash sub-groups (so one coarse key, e.g. a shared
    directory name, cannot swallow a split)."""
    cap = max(8, len(rows) // 40)
    size: dict[str, int] = defaultdict(int)
    for row in rows:
        size[row['group']] += 1
    for row in rows:
        n = size[row['group']]
        if n > cap:
            row['group'] = f'{row["group"]}#{bucket(row["instance_id"], -(-n // (cap // 2)))}'
    return rows


def make_row(domain: str, raw: dict) -> dict:
    iid = instance_id(domain, raw)
    return {'domain': domain, 'instance_id': iid, 'group': raw_group(domain, raw, iid), 'raw': raw}


def row_attrs(row: dict) -> dict[str, Any]:
    raw, ij, info = row['raw'], instance_json(row['raw']), row['raw'].get('extra_info') or {}
    source = str(raw.get('data_source', '')).split('/')[-1] if row['domain'] == 'general' else None
    lang = info.get('lang') or (re.search(r'_(en|zh)_t\d', row['instance_id']) or [None, None])[1]
    description = str(ij.get('description') or prompt_text(raw))
    return {'source': source, 'lang': lang, 'length': info.get('length'), 'meter': info.get('meter'), 'tag': info.get('tag'),
            'category': ij.get('category'), 'cwd': ij.get('cwd'), 'project': row['group'].split(':', 1)[-1].split('#')[0],
            'sanitizer': re.sub(r'sanitizer:?$', '', description.split(' ')[0].lower()) if row['domain'] == 'cyber' else None,
            'test_kind': ('dual' if '&&' in str(ij.get('test_command', '')) else 'single') if row['domain'] == 'code' else None,
            'prompt_chars': len(task_text(raw))}


def passes(row: dict, filters: dict, max_prompt_tokens: int | None = None) -> bool:
    attrs = row_attrs(row)
    for key, want in (filters or {}).items():
        if key == 'max_prompt_chars':
            if attrs['prompt_chars'] > want:
                return False
        elif key == 'min_prompt_chars':
            if attrs['prompt_chars'] < want:
                return False
        elif str(attrs.get(key)) not in {str(w) for w in (want if isinstance(want, list) else [want])}:
            return False
    return max_prompt_tokens is None or estimate_tokens(task_text(row['raw'])) <= max_prompt_tokens


# ------------------------------------------------------------------------------------------------------------------ pool
@lru_cache(maxsize=16)
def _read(path: str, mtime: float, size: int) -> tuple:
    import pyarrow.parquet as pq
    return tuple(pq.read_table(path).to_pylist())


@lru_cache(maxsize=16)
def _digest(path: str, mtime: float, size: int) -> str:
    return file_sha256(Path(path))


def parquet_path(domain: str, raw: str | Path | None = None) -> Path:
    return raw_dir(None, raw) / PARQUET[domain]


def parquet_sha256(domain: str, raw: str | Path | None = None) -> str:
    path = parquet_path(domain, raw)
    stat = path.stat()
    return _digest(str(path), stat.st_mtime, stat.st_size)


def load_pool(domain: str, raw: str | Path | None = None, revision: str = REVISION) -> list[dict]:
    """Every row of one domain as {domain, instance_id, group, raw} in parquet order. The parquet's sha256 must match the pin of
    ``revision`` (an unpinned revision is accepted and recorded in the manifest). Fresh row dicts on every call."""
    if domain not in PARQUET:
        raise DataError(f'unknown domain {domain!r}')
    path = parquet_path(domain, raw)
    if not path.is_file():
        raise DataError(f'{path} missing: download MiMo-V2.6-RL-oss@{revision} into {path.parent}')
    expected = PINNED_SHA256.get(revision, {}).get(domain)
    if expected and parquet_sha256(domain, raw) != expected:
        raise DataError(f'{path.name} does not match the pinned sha256 of revision {revision}')
    stat = path.stat()
    rows = refine_groups([make_row(domain, dict(r)) for r in _read(str(path), stat.st_mtime, stat.st_size)])
    ids = [r['instance_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise DataError(f'{domain}: duplicate instance ids')
    return rows


def load_screening(data_dir: str | Path | None, screen: Callable[[dict], str | None] | None = None, pool: dict[str, list] | None = None,
                   extra: dict | None = None) -> dict[str, str]:
    """instance_id -> reason for every row Track B's screening rejected (screening.json + callback + ``extra``)."""
    out: dict[str, str] = {}
    path = Path(data_dir) / 'screening.json' if data_dir else None
    if path is not None and path.is_file():
        data = json.loads(path.read_text())
        for key, value in data.items():
            if isinstance(value, dict):
                out.update({str(k): str(v) for k, v in value.items()})
            else:
                out[str(key)] = str(value)
    out.update({str(k): str(v) for k, v in (extra or {}).items()})
    if screen is not None:
        for rows in (pool or {}).values():
            for row in rows:
                reason = screen(row)
                if reason:
                    out[row['instance_id']] = str(reason)
    return out


# ------------------------------------------------------------------------------------------------------------------ selection
def select_test(pool: list[dict], n: int, seed: int, domain: str, filters: dict | None = None) -> tuple[list[dict], list[str]]:
    """Fixed grouped held-out set: (rows, groups that contributed). Independent of any training setting."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        if passes(row, filters or {}):
            by_group[row['group']].append(row)
    rows: list[dict] = []
    groups: list[str] = []
    for group in sorted(by_group, key=lambda g: sha(seed, domain, 'test-group', g)):
        if len(rows) >= n:
            break
        members = sorted(by_group[group], key=lambda r: sha(seed, domain, 'test-row', r['instance_id']))
        rows.extend(members[:min(TEST_GROUP_CAP, n - len(rows))])
        groups.append(group)
    return rows, groups


def select_train(pool: list[dict], test_groups: set[str], n: int, seed: int, domain: str, filters: dict | None = None,
                 max_prompt_tokens: int | None = None) -> list[dict]:
    candidates = [r for r in pool if r['group'] not in test_groups and passes(r, filters or {}, max_prompt_tokens)]
    return sorted(candidates, key=lambda r: sha(seed, domain, 'train', r['instance_id']))[:n]


def select(cfg: dict, *, screen: Callable[[dict], str | None] | None = None, raw: str | Path | None = None, screened: dict | None = None,
           data_dir: str | Path | None = None) -> dict:
    """Per-domain selection with everything the manifest records. ``build_selection`` is the flat contract view."""
    mcfg = mimo_config(cfg)
    data_dir = data_dir or (cfg.get('run') or {}).get('data_dir')
    pools = {d: load_pool(d, raw_dir(mcfg, raw), mcfg['revision']) for d in DOMAINS if d in mcfg['domains']}
    drop = load_screening(data_dir, screen, pools, screened)
    out: dict[str, Any] = {'config': mcfg, 'screened': {}, 'domains': {}}
    for domain, pool in pools.items():
        d = mcfg['domains'][domain]
        kept = [r for r in pool if r['instance_id'] not in drop]
        out['screened'].update({r['instance_id']: drop[r['instance_id']] for r in pool if r['instance_id'] in drop})
        test, groups = select_test(kept, d['test'], mcfg['seed'], domain, d['test_filters'])
        train = select_train(kept, set(groups), d['train'], mcfg['seed'], domain, d['filters'], d['max_prompt_tokens'])
        if len(train) < d['train'] or len(test) < d['test']:
            raise DataError(f'{domain}: pool too small after filters/screening: train {len(train)}/{d["train"]}, test {len(test)}/{d["test"]}')
        out['domains'][domain] = {'pool': len(pool), 'screened': sum(r['instance_id'] in drop for r in pool), 'train': train, 'test': test,
                                  'canary': test[:mcfg['canary']], 'test_groups': groups}
    return out


def build_selection(cfg: dict, **kwargs: Any) -> dict[str, list[dict]]:
    """The contract view: {"train": [row], "test": [row], "canary": [row]} over every configured domain (canonical order)."""
    chosen = select(cfg, **kwargs)
    return {split: [r for d in DOMAINS if d in chosen['domains'] for r in chosen['domains'][d][split]] for split in SPLITS}


def selection_ids(chosen: dict) -> dict[str, dict[str, list[str]]]:
    return {d: {s: [r['instance_id'] for r in info[s]] for s in SPLITS} for d, info in chosen['domains'].items()}


def selection_shas(mcfg: dict, ids: dict[str, dict[str, list[str]]]) -> dict[str, str]:
    """``test_sha``: only the held-out ids (+ seed/revision); ``selection_sha``: everything (+ the selection filters)."""
    test = {d: v['test'] for d, v in sorted(ids.items())}
    full = {d: {'ids': v, 'filters': mcfg['domains'][d]['filters'], 'test_filters': mcfg['domains'][d]['test_filters']} for d, v in sorted(ids.items())}
    return {'test_sha': canonical_sha({'revision': mcfg['revision'], 'seed': mcfg['seed'], 'test': test}),
            'selection_sha': canonical_sha({'revision': mcfg['revision'], 'seed': mcfg['seed'], 'domains': full})}


# ------------------------------------------------------------------------------------------------------------------ prompt report
def prompt_report(selection: dict[str, list[dict]], tokenizer: Any = None) -> dict:
    """Per domain length percentiles of the task text, in tokens (``tokenizer.encode``) or chars/4 (``method`` says which)."""
    count = (lambda t: len(tokenizer.encode(t, add_special_tokens=False))) if tokenizer is not None else estimate_tokens
    report: dict[str, Any] = {'method': 'tokenizer' if tokenizer is not None else 'chars/4'}
    by_domain: dict[str, list[int]] = defaultdict(list)
    for rows in selection.values():
        for row in rows:
            by_domain[row['domain']].append(count(task_text(row['raw'])))
    for domain, values in sorted(by_domain.items()):
        values.sort()
        report[domain] = {'n': len(values), 'min': values[0], 'p50': values[len(values) // 2], 'p90': values[int(0.9 * (len(values) - 1))], 'max': values[-1]}
    return report


def load_tokenizer(cfg: dict) -> Any:
    """The model tokenizer when a local snapshot exists, else None (callers fall back to chars/4)."""
    path = Path((cfg.get('model') or {}).get('local_path', ''))
    if not (path / 'config.json').is_file():
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(path))
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------------------------ layout / manifest
def task_name(domain: str, iid: str) -> str:
    return f'{domain}-{iid}'


def task_domain(path_or_name: str | Path) -> str:
    """Domain of a task directory (its name is ``<domain>-<instance_id>``); 'unknown' for anything else."""
    head = Path(str(path_or_name)).name.split('-', 1)[0]
    return head if head in DOMAINS else 'unknown'


def task_dir(data_dir: str | Path, split: str, domain: str, iid: str) -> Path:
    return Path(data_dir) / 'tasks' / split / task_name(domain, iid)


def manifest_path(cfg: dict) -> Path:
    return Path(cfg['run']['data_dir']) / 'manifest.json'


def load_manifest(cfg: dict) -> dict:
    path = manifest_path(cfg)
    if not path.is_file():
        raise DataError(f'{path} missing: run `python -m strl prepare <experiment>`')
    return json.loads(path.read_text())


def prepare(cfg: dict, *, exporter: Callable[..., dict] | None = None, screen: Callable[[dict], str | None] | None = None,
            tokenizer: Any = None, raw: str | Path | None = None, screened: dict | None = None) -> dict:
    """Select, export every selected task through ``exporter(row, dest, split=...)`` (Track B's ``export.export_task``) and write
    <data_dir>/manifest.json. Stale task dirs of earlier selections are removed so a directory scan can never pick them up."""
    if exporter is None:
        from .export import export_task as exporter
    data_dir = Path(cfg['run']['data_dir'])
    chosen = select(cfg, screen=screen, raw=raw, screened=screened, data_dir=data_dir)
    mcfg = chosen['config']
    wanted = {split: {task_name(d, r['instance_id']) for d, info in chosen['domains'].items() for r in info[split]} for split in SPLITS}
    import shutil
    for split in SPLITS:
        root = data_dir / 'tasks' / split
        if root.is_dir():
            for stale in (p for p in root.iterdir() if p.is_dir() and p.name not in wanted[split]):
                shutil.rmtree(stale)
    tasks: dict[str, dict] = {split: {} for split in SPLITS}
    for domain, info in chosen['domains'].items():
        for split in SPLITS:
            for row in info[split]:
                dest = task_dir(data_dir, split, domain, row['instance_id'])
                entry = dict(exporter(row, dest, split=split))
                entry.pop('task_dir', None)   # absolute paths do not survive the move into the pod; the layout is the contract
                tasks[split][task_name(domain, row['instance_id'])] = {**entry, 'group': row['group']}
    ids = selection_ids(chosen)
    selection = {s: [r for d in DOMAINS if d in chosen['domains'] for r in chosen['domains'][d][s]] for s in SPLITS}
    manifest = {
        'kind': 'mimo', 'revision': mcfg['revision'], 'seed': mcfg['seed'], 'eval_repeats': mcfg['eval_repeats'], 'canary_per_domain': mcfg['canary'],
        'domains': {d: {'train': ids[d]['train'], 'test': ids[d]['test'], 'canary': ids[d]['canary'], 'test_groups': info['test_groups'],
                        'pool': info['pool'], 'screened': info['screened'], 'filters': mcfg['domains'][d]['filters'],
                        'train_max_est_tokens': max([estimate_tokens(task_text(r['raw'])) for r in info['train']] or [0]),
                        'test_filters': mcfg['domains'][d]['test_filters'], 'max_prompt_tokens': mcfg['domains'][d]['max_prompt_tokens'],
                        'parquet_sha256': parquet_sha256(d, raw_dir(mcfg, raw))} for d, info in chosen['domains'].items()},
        **selection_shas(mcfg, ids), 'screened': chosen['screened'],
        'counts': {s: len(selection[s]) for s in SPLITS}, 'prompt_tokens': prompt_report(selection, tokenizer if tokenizer is not None else load_tokenizer(cfg)),
        'tasks': tasks}
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


# ------------------------------------------------------------------------------------------------------------------ runtime view
def runtime_selection(cfg: dict, manifest: dict | None = None) -> dict[str, dict[str, list[str]]]:
    """{split: {domain: [ids]}} for THIS run: the manifest's ids truncated to the configured counts (prefix-stable), so a
    ``--set`` of train/test counts works without re-preparing as long as it asks for no more than was prepared. Domains
    with train=0 contribute nothing to training/validation."""
    manifest = manifest or load_manifest(cfg)
    mcfg = mimo_config(cfg)
    out: dict[str, dict[str, list[str]]] = {s: {} for s in SPLITS}
    for domain in [d for d in DOMAINS if d in mcfg['domains']]:
        d, m = mcfg['domains'][domain], (manifest['domains'].get(domain) or {})
        for split, want in (('train', d['train']), ('test', d['test'])):
            have = m.get(split) or []
            if want > len(have):
                raise DataError(f'{domain}: {want} {split} tasks requested but only {len(have)} were prepared: re-run `strl prepare`')
            if want:
                out[split][domain] = have[:want]
        if d['test']:
            out['canary'][domain] = (m.get('canary') or [])[:min(mcfg['canary'], d['test'])]
    return out


def split_paths(cfg: dict, split: str, manifest: dict | None = None, *, repeats: int = 1) -> list[str]:
    """Task dirs of ``split`` ('train' | 'test' | 'canary') in domain order; each repeated ``repeats`` times (held-out eval reps)."""
    selection = runtime_selection(cfg, manifest)[split]
    data_dir = cfg['run']['data_dir']
    return [str(task_dir(data_dir, split, d, i)) for rep in range(repeats) for d in DOMAINS if d in selection for i in selection[d]]


def assert_manifest(cfg: dict, *, deep: bool | None = None, raw: str | Path | None = None) -> dict:
    """Verify the prepared data against the experiment. Shallow (works in the pod, no parquet): manifest shas recompute, counts
    cover the configuration, filters are the prepared ones, test/train are disjoint by id AND group, every task dir exists with
    instruction.md/task.toml/strl.json naming the right domain and instance. Deep (parquet present locally): additionally the
    parquet sha256 pins and a full re-selection that must reproduce the manifest."""
    manifest = load_manifest(cfg)
    mcfg = mimo_config(cfg)
    problems: list[str] = []
    if manifest.get('kind') != 'mimo' or manifest.get('revision') != mcfg['revision']:
        problems.append(f'manifest revision {manifest.get("revision")} != configured {mcfg["revision"]}')
    ids = {d: {s: m.get(s, []) for s in SPLITS} for d, m in manifest['domains'].items()}
    shas = selection_shas({**mcfg, 'seed': manifest['seed'], 'domains': {d: {**mcfg['domains'].get(d, {}), 'filters': m['filters'], 'test_filters': m['test_filters']}
                                                                       for d, m in manifest['domains'].items()}}, ids)
    for key in ('test_sha', 'selection_sha'):
        if shas[key] != manifest.get(key):
            problems.append(f'manifest {key} does not recompute: the manifest was edited or is corrupt')
    if manifest['seed'] != mcfg['seed']:
        problems.append(f'manifest seed {manifest["seed"]} != configured seed {mcfg["seed"]}: re-run `strl prepare`')
    if manifest['eval_repeats'] < mcfg['eval_repeats']:
        problems.append('manifest eval_repeats is lower than configured')
    for domain in [d for d in DOMAINS if d in mcfg['domains']]:
        d, m = mcfg['domains'][domain], manifest['domains'].get(domain)
        if m is None:
            problems.append(f'{domain}: not in the prepared manifest')
            continue
        if d['train'] > len(m['train']) or d['test'] > len(m['test']):
            problems.append(f'{domain}: configured train/test {d["train"]}/{d["test"]} exceed prepared {len(m["train"])}/{len(m["test"])}')
        if d['filters'] != m['filters'] or d['test_filters'] != m['test_filters']:
            problems.append(f'{domain}: filters differ from the prepared ones: re-run `strl prepare`')
        if set(m['train']) & set(m['test']):
            problems.append(f'{domain}: train and test share instances')
        groups = {manifest['tasks']['train'].get(task_name(domain, i), {}).get('group') for i in m['train']}
        if groups & set(m['test_groups']):
            problems.append(f'{domain}: a test group leaks into train')
        cap = d['max_prompt_tokens']
        if cap and m.get('train_max_est_tokens', 0) > cap:
            problems.append(f'{domain}: a prepared train prompt has ~{m["train_max_est_tokens"]} tokens > max_prompt_tokens {cap}')
    selection = runtime_selection(cfg, manifest) if not problems else {}
    data_dir = Path(cfg['run']['data_dir'])
    for split, per_domain in selection.items():
        for domain, id_list in per_domain.items():
            for iid in id_list:
                path = task_dir(data_dir, split, domain, iid)
                for name in ('instruction.md', 'task.toml', 'strl.json'):
                    if not (path / name).is_file():
                        problems.append(f'{path.relative_to(data_dir)}/{name} missing')
                        break
                else:
                    meta = json.loads((path / 'strl.json').read_text())
                    if meta.get('domain') != domain or str(meta.get('instance_id')) != iid:
                        problems.append(f'{path.name}/strl.json names {meta.get("domain")}/{meta.get("instance_id")}')
    if deep is None:
        deep = (raw_dir(mcfg, raw) / PARQUET['code']).is_file()
    if deep and not problems:
        problems += _deep_problems(cfg, manifest, mcfg, raw)
    if problems:
        raise DataError('mimo data manifest invalid:\n' + '\n'.join(f'- {p}' for p in problems[:20]))
    return manifest


def _deep_problems(cfg: dict, manifest: dict, mcfg: dict, raw: str | Path | None) -> list[str]:
    problems = []
    for domain, m in manifest['domains'].items():
        if m['parquet_sha256'] != parquet_sha256(domain, raw_dir(mcfg, raw)):
            problems.append(f'{domain}: parquet sha256 changed since prepare')
    if problems:
        return problems
    prepared = {**cfg, 'environment': {**cfg['environment'], 'mimo': {
        **(cfg['environment'].get('mimo') or {}), 'seed': manifest['seed'], 'canary': manifest['canary_per_domain'], 'domains': {
            d: {'train': len(m['train']), 'test': len(m['test']), 'filters': m['filters'], 'test_filters': m['test_filters'],
                'max_prompt_tokens': m.get('max_prompt_tokens'), 'weight': 1.0} for d, m in manifest['domains'].items()}}}}
    again = selection_ids(select(prepared, raw=raw, screened=manifest.get('screened')))
    for domain, m in manifest['domains'].items():
        for split in SPLITS:
            if again[domain][split] != m[split]:
                problems.append(f'{domain}: re-selection of {split} does not reproduce the manifest')
    return problems
