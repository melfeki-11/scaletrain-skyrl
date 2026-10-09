"""`strl launch`: get GPUs as fast as ScaleTrain allows, every time.

    python -m strl launch experiments/<x>.yaml --attempt <name> --mode dry|canary|train [--set K=V ...]
        [--routes fb,rb,ro,fo] [--max-wait-hours H] [--watch] [--relaunch-on-preempt N]

Why: capacity for the same shape is granted per (team quota, borrowing) route, and the routes behave differently over time. Measured on
2026-10-07/08: a 2 x 8 job sat ~23 h on `waiting_for_network_domain_capacity` while the same job on another route started first; borrowed
capacity was reclaimed after 70 s and after 30 min; a CPU job on another team's quota never started while the same job on the foundation
team started at once. No single route is reliably fastest, so the launcher RACES all of them:

  1. stage one attempt per route (`<name>-fb`, `-rb`, `-ro`, `-fo`: team foundation/research x borrowing on/off; same experiment, same budget
     reservation; the ownership defaults of the ScaleTrain runbook - foundation team, high priority, borrowing - are route `fb`, tried first);
  2. submit all of them (each with its own write-once receipt);
  3. poll; the FIRST job that leaves the queue wins, every other job is cancelled at once;
  4. record every loser in the program ledger: not charged when its pod never started (no budget.json / worker-started.json), charged from
     pod start otherwise (two routes can start inside one poll interval);
  5. with --watch: follow the winner to its terminal state and record it; when ScaleTrain reports a preemption (the job falls back to QUEUED with
     `preemption_reported`), cancel it (a preempted attempt cannot resume: budget window anchored at the first start, no optimizer-state
     checkpoints), record it, and with --relaunch-on-preempt N race again as `<name>-p<k>`.

Only one route can ever train: the loser cancellation is the guarantee, not luck. Budget: each route reserves the attempt's budget at stage
time, but only the winner (plus at most one poll interval of a loser that started concurrently) is ever charged.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

FOUNDATION, RESEARCH = 'gen_ai.agent_training_foundation', 'gen_ai.agents_research'
# (code, team, allow_borrowing) in the order the ScaleTrain runbook prefers them
ROUTES = (('fb', FOUNDATION, True), ('rb', RESEARCH, True), ('ro', RESEARCH, False), ('fo', FOUNDATION, False))
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELED', 'CANCELLED'}
MAX_BASE_NAME = 26          # room for '-fb' and '-p9'


def select_routes(spec: str | None) -> list[tuple[str, str, bool]]:
    if not spec or spec == 'all':
        return list(ROUTES)
    wanted = [c.strip() for c in spec.split(',') if c.strip()]
    unknown = [c for c in wanted if c not in {r[0] for r in ROUTES}]
    if unknown:
        raise ValueError(f'unknown route(s) {unknown}; choose from {[r[0] for r in ROUTES]} (fb = foundation+borrowing, rb = research+borrowing, '
                         'ro = research own quota, fo = foundation own quota)')
    return [r for c in wanted for r in ROUTES if r[0] == c]


@dataclass
class Api:
    """Everything the launcher does to the outside world (the CLI wires the real ScaleTrain helpers; tests pass fakes)."""
    stage: Callable[..., dict]                     # (experiment, attempt=, mode=, team=, allow_borrowing=, overrides=) -> stage receipt
    submit: Callable[[str], str | None]            # job file -> job id (None: rejected)
    get: Callable[[str], dict]                     # job id -> {'status', 'status_detail', ...}
    cancel: Callable[[str], Any]
    record: Callable[..., Any]                     # (experiment, attempt, job_id, never_started=, terminal_at=) -> ledger receipt
    pods_started: Callable[[str, str], bool]       # (experiment, attempt) -> did a pod ever run (budget.json / worker-started.json; fail closed)
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], Any] = time.sleep
    log: Callable[[str], Any] = field(default=lambda m: print(m, flush=True))


def _state(api: Api, job_id: str) -> tuple[str, str]:
    try:
        job = api.get(job_id)
        return str(job.get('status')), str(job.get('status_detail') or '')
    except Exception as error:   # a transient API failure is not a state change
        return 'UNKNOWN', type(error).__name__


CANCEL_TAIL_SECONDS = 180   # a job cancelled after allocation can keep its pod for minutes (measured: budget.json written 89 s after CANCELED)


def _settle_losers(api: Api, experiment: str, losers: dict[str, str], poll: float, timeout: float = 600.0,
                   allocated: set[str] | frozenset = frozenset()) -> list[dict]:
    """Cancel every loser, wait until each is terminal, record it in the ledger. Uncharged only on positive evidence that no pod ran: a
    job ever seen allocated (IN_PROGRESS) is charged even before its pod leaves a trace, through CANCEL_TAIL_SECONDS after the terminal
    observation, because the pod outlives the API's CANCELED."""
    for attempt, job_id in losers.items():
        try:
            api.cancel(job_id)
        except Exception as error:
            api.log(f'cancel {attempt} ({job_id}) failed: {error}; retrying while settling')
    out, pending, until = [], dict(losers), api.clock() + timeout
    while pending and api.clock() < until:
        for attempt, job_id in list(pending.items()):
            status, _ = _state(api, job_id)
            if status in TERMINAL:
                never = attempt not in allocated and not api.pods_started(experiment, attempt)
                tail = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(api.clock() + CANCEL_TAIL_SECONDS)) if attempt in allocated else None
                try:
                    receipt = api.record(experiment, attempt, job_id, never_started=never and status in {'CANCELED', 'CANCELLED'}, terminal_at=tail)
                except Exception as error:
                    receipt = {'error': str(error)}
                out.append({'attempt': attempt, 'job_id': job_id, 'status': status, 'charged': not never, 'ledger': receipt})
                pending.pop(attempt)
            elif status != 'UNKNOWN':
                try:
                    api.cancel(job_id)
                except Exception:
                    pass
        if pending:
            api.sleep(poll)
    out += [{'attempt': a, 'job_id': j, 'status': 'NOT_TERMINAL', 'charged': None, 'ledger': {'error': 'record it with strl ledger once terminal'}}
            for a, j in pending.items()]
    return out


def race(experiment: str, attempt: str, mode: str, api: Api, *, overrides: list[str] | None = None, routes: str | None = None,
         poll: float = 15.0, max_wait_seconds: float | None = None) -> dict:
    """Stage + submit one job per route, keep the first that starts, cancel the rest. Returns the launch receipt."""
    if len(attempt) > MAX_BASE_NAME:
        raise ValueError(f'attempt name {attempt!r} is longer than {MAX_BASE_NAME} characters (route suffixes need room)')
    chosen, jobs, receipts = select_routes(routes), {}, {}
    for code, team, borrow in chosen:
        name = f'{attempt}-{code}'
        # Staging takes ~30 s per route; when capacity is free the first route is often running before the last is submitted. Measured: in
        # all 5 GPU races of 2026-10-08 a second route was allocated in the same poll and had to be cancelled and charged. Stop submitting
        # once any submitted route started (no added latency when capacity is scarce: every route is still submitted).
        if jobs and any(_state(api, j)[0] not in TERMINAL | {'QUEUED', 'UNKNOWN'} for j in jobs.values()):
            api.log(f'a submitted route already started: not submitting {name} or later routes')
            break
        extra = [] if borrow else ['compute.allow_borrowing=false']
        receipts[name] = api.stage(experiment, attempt=name, mode=mode, team=team, allow_borrowing=borrow, overrides=[*(overrides or []), *extra])
        job_id = api.submit(receipts[name]['job_file'])
        if job_id:
            jobs[name] = job_id
            api.log(f'submitted {name}: team={team} borrowing={borrow} job={job_id}')
        else:
            api.log(f'{name} was not accepted by ScaleTrain; continuing with the other routes')
    if not jobs:
        raise RuntimeError('no route was accepted by ScaleTrain')
    started_at, last, ran, allocated = api.clock(), {}, {}, set()
    while True:
        states = {name: _state(api, job_id) for name, job_id in jobs.items()}
        allocated |= {n for n, (s, _) in states.items() if s not in TERMINAL | {'QUEUED', 'UNKNOWN'}}
        if states != last:
            api.log(time.strftime('%H:%M:%S ') + ' '.join(f'{n}[{s} {d}]' for n, (s, d) in states.items()))
            last = states
        running = [n for n, (s, _) in states.items() if s not in TERMINAL | {'QUEUED', 'UNKNOWN'}]
        if not running:   # a short job (a CPU dry run) can start AND finish between two polls: it ran if its pod left a trace
            for n, (s, _) in states.items():
                if s in TERMINAL and n not in ran:
                    ran[n] = api.pods_started(experiment, n)
            running = [n for n in jobs if ran.get(n)]
        if running:
            winner = running[0]
            losers = {n: j for n, j in jobs.items() if n != winner}
            settled = _settle_losers(api, experiment, losers, poll, allocated=allocated)
            api.log(f'WINNER {winner} ({jobs[winner]}) after {api.clock() - started_at:.0f} s; cancelled {sorted(losers)}')
            return {'winner': winner, 'job_id': jobs[winner], 'waited_seconds': round(api.clock() - started_at), 'routes': [c for c, _, _ in chosen],
                    'jobs': jobs, 'losers': settled, 'receipt': receipts[winner]}
        dead = [n for n, (s, _) in states.items() if s in TERMINAL]   # ended without a pod
        if dead and len(dead) == len(jobs):
            settled = _settle_losers(api, experiment, dict(jobs), poll, allocated=allocated)
            return {'winner': None, 'reason': 'every route ended before starting', 'jobs': jobs, 'losers': settled}
        if max_wait_seconds is not None and api.clock() - started_at > max_wait_seconds:
            settled = _settle_losers(api, experiment, dict(jobs), poll, allocated=allocated)
            api.log(f'no route started within {max_wait_seconds / 3600:.1f} h: all cancelled')
            return {'winner': None, 'reason': f'no capacity within {max_wait_seconds:.0f} s', 'jobs': jobs, 'losers': settled}
        api.sleep(poll)


def watch(experiment: str, attempt: str, job_id: str, api: Api, *, poll: float = 60.0) -> dict:
    """Follow a started job to its end. Returns {'outcome': 'terminal'|'preempted', 'status', ...} after recording it in the ledger."""
    while True:
        status, detail = _state(api, job_id)
        if status in TERMINAL:
            try:
                receipt = api.record(experiment, attempt, job_id)
            except Exception as error:
                receipt = {'error': str(error)}
            return {'outcome': 'terminal', 'status': status, 'attempt': attempt, 'job_id': job_id, 'ledger': receipt}
        if status == 'QUEUED' and 'preempt' in detail:
            api.log(f'{attempt} was preempted ({detail}): cancelling the requeued job (a preempted attempt cannot resume)')
            terminal_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(api.clock()))
            settled = _settle_losers(api, experiment, {attempt: job_id}, poll=min(poll, 15.0), allocated={attempt})
            return {'outcome': 'preempted', 'status': status, 'attempt': attempt, 'job_id': job_id, 'terminal_at': terminal_at, 'settled': settled}
        api.sleep(poll)


def launch(experiment: str, attempt: str, mode: str, api: Api, *, overrides: list[str] | None = None, routes: str | None = None,
           poll: float = 15.0, max_wait_seconds: float | None = None, follow: bool = False, relaunch_on_preempt: int = 0,
           resume_on_preempt: bool = False) -> list[dict]:
    """race (+ watch, + relaunch after preemption; with resume_on_preempt, which needs evaluation.durable_checkpoints, the relaunch continues
    the preempted attempt from its latest checkpoint instead of starting over). Returns one receipt per race."""
    history, name, extra = [], attempt, []
    for k in range(relaunch_on_preempt + 1):
        result = race(experiment, name, mode, api, overrides=[*(overrides or []), *extra], routes=routes, poll=poll, max_wait_seconds=max_wait_seconds)
        history.append(result)
        if not result.get('winner') or not follow:
            break
        outcome = watch(experiment, result['winner'], result['job_id'], api)
        result['outcome'] = outcome
        if outcome['outcome'] != 'preempted':
            break
        name = f'{attempt}-p{k + 1}'
        extra = [f"run.resume_from={result['winner']}"] if resume_on_preempt else []
        if k < relaunch_on_preempt:
            api.log(f'relaunching as {name} (preemption {k + 1} of {relaunch_on_preempt} allowed)')
    receipt_dir = Path('evidence') / attempt
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / 'launch.json').write_text(json.dumps(history, indent=2, default=str) + '\n')
    return history


def real_api(user_id: str) -> Api:
    """The real ScaleTrain wiring (tools/scaletrain.py via the CLI helpers, strl.stage, the ledger)."""
    from . import cli
    from .stage import stage as stage_fn

    def stage(experiment, *, attempt, mode, team, allow_borrowing, overrides):
        receipt = stage_fn(experiment, attempt=attempt, mode=mode, team=team, allow_borrowing=allow_borrowing,
                           output=f'evidence/{attempt}/job.json', overrides=overrides)
        Path(f'evidence/{attempt}').mkdir(parents=True, exist_ok=True)
        Path(f'evidence/{attempt}/stage.json').write_text(json.dumps(receipt, indent=2))
        return receipt

    def submit(job_file: str) -> str | None:
        job = json.loads(Path(job_file).read_text())
        if cli._scaletrain('validate', job_file).returncode:
            return None
        receipt = str(Path(job_file).parent / 'submission.json')
        out = cli._scaletrain('submit', job_file, '--user-id', user_id, '--receipt', receipt)
        if out.returncode == 0:
            return json.loads(out.stdout.strip().splitlines()[-1])['id']
        listing = json.loads(cli._scaletrain('list', '--user-id', user_id, '--limit', '30').stdout or '[]')   # reconcile, never resubmit
        match = [j for j in listing if j['name'] == job['job_name']]
        return match[0]['id'] if match else None

    def get(job_id: str) -> dict:
        return json.loads(cli._scaletrain('get', job_id).stdout)

    def cancel(job_id: str):
        return cli._scaletrain('cancel', '--user-id', user_id, job_id)

    def record(experiment, attempt, job_id, never_started=False, terminal_at=None):
        return cli.record_attempt(experiment, attempt, job_id, never_started=never_started, terminal_at=terminal_at)

    def pods_started(experiment: str, attempt: str) -> bool:
        cfg = cli._load(experiment)   # fails closed (any S3 error other than not-found raises -> treated as started below)
        try:
            return cli.worker_started(cfg, attempt) or cli.allocation_start(cfg, attempt) is not None
        except Exception:
            return True

    return Api(stage=stage, submit=submit, get=get, cancel=cancel, record=record, pods_started=pods_started)


FALLBACK_SUFFIX = '-alt'   # not '-fb...': route codes already end attempt names (fb/rb/ro/fo)


def check_names(attempt: str, *, relaunch_on_preempt: int = 0, fallback: bool = False) -> None:
    """Every attempt name a launch may create must fit MAX_BASE_NAME BEFORE anything is submitted (a relaunch hours later must not fail on
    its name, and truncation must never reproduce an earlier name)."""
    longest = attempt + (FALLBACK_SUFFIX if fallback else '') + (f'-p{relaunch_on_preempt}' if relaunch_on_preempt else '')
    if len(longest) > MAX_BASE_NAME:
        raise ValueError(f'attempt name {attempt!r} is too long: with these flags the launcher may create {longest!r} ({len(longest)} > '
                         f'{MAX_BASE_NAME} characters before the route suffix); use at most {MAX_BASE_NAME - len(longest) + len(attempt)} characters')


def main_launch(a) -> int:
    try:
        check_names(a.attempt, relaunch_on_preempt=a.relaunch_on_preempt or 0, fallback=bool(a.max_wait_hours and a.fallback_set))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    api = real_api(a.user_id)
    from . import cli
    from .overrides import durable
    resumable = durable(cli._load(a.experiment, a.set))
    history = launch(a.experiment, a.attempt, a.mode, api, overrides=a.set, routes=a.routes, poll=a.poll,
                     max_wait_seconds=a.max_wait_hours * 3600 if a.max_wait_hours else None, follow=a.watch,
                     relaunch_on_preempt=a.relaunch_on_preempt, resume_on_preempt=resumable)
    if a.max_wait_hours and not history[-1].get('winner') and a.fallback_set:   # e.g. 2 nodes never came: fall back to a smaller shape
        api.log(f'no capacity within {a.max_wait_hours} h: falling back with {a.fallback_set}')
        history += launch(a.experiment, a.attempt + FALLBACK_SUFFIX, a.mode, api, overrides=[*a.set, *a.fallback_set], routes=a.routes,
                          poll=a.poll, follow=a.watch, relaunch_on_preempt=a.relaunch_on_preempt, resume_on_preempt=resumable)
    print(json.dumps([{k: v for k, v in h.items() if k != 'receipt'} for h in history], indent=1, default=str))
    last = history[-1]
    if not last.get('winner'):
        print(f"\nno route started ({last.get('reason')}); fall back to a smaller shape (e.g. --set compute.nodes=1) or wait longer", file=sys.stderr)
        return 2
    print(f"\nnext: python -m strl monitor {a.experiment} --attempt {last['winner']} --job-id {last['job_id']}")
    return 0
