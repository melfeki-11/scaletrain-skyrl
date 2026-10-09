"""`strl launch`: get GPUs as fast as ScaleTrain allows, every time.

    python -m strl launch experiments/<x>.yaml --attempt <name> --mode dry|canary|train [--set K=V ...]
        [--routes fb,rb,ro,fo] [--max-wait-hours H] [--watch] [--relaunch-on-preempt N]

Why: capacity for the same shape is granted per (team quota, borrowing) route, and the routes behave differently over time. Measured on
2026-10-07/08: a 2 x 8 job sat ~23 h on `waiting_for_network_domain_capacity` while the same job on another route started first; borrowed
capacity was reclaimed after 70 s and after 30 min; a CPU job on another team's quota never started while the same job on the foundation
team started at once. No single route is reliably fastest, so the launcher RACES all of them:

  1. stage one attempt per route (`<name>-fb`, `-rb`, `-ro`, `-fo`: team foundation/research x borrowing on/off; same experiment; the
     ownership defaults of the ScaleTrain runbook - foundation team, high priority, borrowing - are route `fb`, tried first);
  2. submit them in order (each with its own write-once receipt), stopping once a submitted one has started;
  3. poll; the FIRST job that leaves the queue wins, every other job is cancelled at once;
  4. record every loser in the program ledger: uncharged only on positive evidence that no pod ran; a job ever seen allocated is charged from
     that observation (minus one poll) through a cancel tail, because its pod outlives CANCELED;
  5. with --watch: follow the winner to its terminal state and record it; when ScaleTrain reports a preemption (the job falls back to QUEUED with
     `preemption_reported`), cancel and record it, and with --relaunch-on-preempt N race again as `<name>-p<k>`, resuming from the latest
     attempt that left a durable checkpoint (evaluation.durable_checkpoints) or starting fresh.

Only one route can ever train: the loser cancellation is the guarantee, not luck; a loser that cannot be settled fails the launch loudly.
Budget: stage only CHECKS the remaining budget (no reservation), so concurrent launches against one ledger can overspend it together.
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
    record: Callable[..., Any]                     # (experiment, attempt, job_id, never_started=, terminal_at=, allocated_floor=) -> ledger receipt
    pods_started: Callable[[str, str], bool | None]   # (experiment, attempt) -> did a pod ever run (budget.json / worker-started.json); None = unknown
    has_checkpoint: Callable[[str, str], bool] = lambda experiment, attempt: True   # durable checkpoint in S3 (resume candidates)
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
ACTIVE = lambda status: status not in TERMINAL | {'QUEUED', 'UNKNOWN'}   # noqa: E731  allocated / running


def _iso(epoch: float) -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(epoch))


def _record(api: Api, experiment: str, attempt: str, job_id: str, tries: int = 3, **kwargs) -> dict:
    """Write one ledger entry, retrying; a failure is returned as {'error', 'unrecorded': True} so the launch exits non-zero (never silent)."""
    error = None
    for i in range(tries):
        try:
            return api.record(experiment, attempt, job_id, **kwargs)
        except (Exception, SystemExit) as exc:   # cli._load raises SystemExit on an invalid YAML
            if 'already recorded' in str(exc):
                return {'already_recorded': True}
            if kwargs.get('never_started') and 'a pod ran' in str(exc):   # its trace appeared meanwhile: charge it (with the cancel tail)
                kwargs = {**kwargs, 'never_started': False, 'terminal_at': kwargs.get('terminal_at') or _iso(api.clock() + CANCEL_TAIL_SECONDS)}
                continue
            error = exc
            if i + 1 < tries:
                api.sleep(10)
    api.log(f'LEDGER: could not record {attempt} ({job_id}): {error}; record it by hand: strl ledger {experiment} --attempt {attempt} '
            f'--job-id {job_id} (add the launch\'s --set values)')
    return {'error': str(error), 'unrecorded': True}


def _settle_losers(api: Api, experiment: str, losers: dict[str, str], poll: float, timeout: float = 600.0,
                   seen: dict[str, float] | None = None) -> list[dict]:
    """Cancel every loser, wait until each is terminal, record it in the ledger. ``seen`` maps attempt -> first time it was observed
    allocated. Uncharged only on positive evidence that no pod ran; a job ever seen allocated (before or while settling) is charged from that
    observation minus one poll, through CANCEL_TAIL_SECONDS after the terminal observation (its pod outlives CANCELED)."""
    seen = dict(seen or {})
    for attempt, job_id in losers.items():
        try:
            api.cancel(job_id)
        except Exception as error:
            api.log(f'cancel {attempt} ({job_id}) failed: {error}; retrying while settling')
    out, pending, until, confirm = [], dict(losers), api.clock() + timeout + CANCEL_TAIL_SECONDS, {}
    while pending and api.clock() < until:
        for attempt, job_id in list(pending.items()):
            status, _ = _state(api, job_id)
            if ACTIVE(status):
                seen.setdefault(attempt, api.clock())
            if status in TERMINAL:
                ran = api.pods_started(experiment, attempt)
                if attempt not in seen and ran is False:
                    # a pod cancelled during its start-up leaves its trace up to ~90 s after CANCELED: claim "never started" only if the
                    # trace is still absent CANCEL_TAIL_SECONDS after the terminal observation
                    confirm.setdefault(attempt, api.clock() + CANCEL_TAIL_SECONDS)
                    if api.clock() < confirm[attempt]:
                        continue
                never = attempt not in seen and ran is False    # unknown (None) is charged: fail closed
                kwargs = {'never_started': never}
                if not never:   # every charged loser: its pod may outlive the terminal observation
                    kwargs['terminal_at'] = _iso(api.clock() + CANCEL_TAIL_SECONDS)
                if attempt in seen:
                    kwargs['allocated_floor'] = seen[attempt] - poll
                receipt = _record(api, experiment, attempt, job_id, **kwargs)
                out.append({'attempt': attempt, 'job_id': job_id, 'status': status, 'charged': not never, 'ledger': receipt})
                pending.pop(attempt)
            elif status != 'UNKNOWN':
                try:
                    api.cancel(job_id)
                except Exception as error:
                    api.log(f'cancel {attempt} ({job_id}) failed again: {error}')
        if pending:
            api.sleep(poll if not confirm else min(poll, max(1.0, min(confirm.values()) - api.clock())))
    for a, j in pending.items():
        api.log(f'UNSETTLED: {a} ({j}) is still not terminal after {timeout:.0f} s: cancel it by hand and record it with strl ledger')
    out += [{'attempt': a, 'job_id': j, 'status': 'NOT_TERMINAL', 'charged': None, 'unsettled': True,
             'ledger': {'error': 'record it with strl ledger once terminal', 'unrecorded': True}} for a, j in pending.items()]
    return out


def race(experiment: str, attempt: str, mode: str, api: Api, *, overrides: list[str] | None = None, routes: str | None = None,
         poll: float = 15.0, max_wait_seconds: float | None = None) -> dict:
    """Stage + submit one job per route, keep the first that starts, cancel the rest. Returns the launch receipt."""
    if len(attempt) > MAX_BASE_NAME:
        raise ValueError(f'attempt name {attempt!r} is longer than {MAX_BASE_NAME} characters (route suffixes need room)')
    chosen, jobs, receipts, seen = select_routes(routes), {}, {}, {}
    try:
        _submit_routes(api, experiment, attempt, mode, chosen, overrides, jobs, receipts, seen)
    except BaseException:   # never leave submitted routes running unmanaged (a transient stage/submit error mid-race)
        if jobs:
            api.log(f'launch error while submitting: cancelling the {len(jobs)} submitted route(s) first')
            _settle_losers(api, experiment, dict(jobs), poll, seen=seen)
        raise
    if not jobs:
        raise RuntimeError('no route was accepted by ScaleTrain')
    try:
        return _poll_routes(api, experiment, poll, max_wait_seconds, chosen, jobs, receipts, seen)
    except BaseException:   # Ctrl-C / an error while waiting: cancel everything still unsettled before leaving
        api.log('launch interrupted while waiting for capacity: cancelling every submitted route')
        _settle_losers(api, experiment, dict(jobs), poll, seen=seen)
        raise


def _poll_routes(api, experiment, poll, max_wait_seconds, chosen, jobs, receipts, seen) -> dict:
    started_at, last, ran, unknown_polls = api.clock(), {}, {}, 0
    while True:
        states = {name: _state(api, job_id) for name, job_id in jobs.items()}
        for n, (st, _) in states.items():
            if ACTIVE(st):
                seen.setdefault(n, api.clock())
        if states != last:
            api.log(time.strftime('%H:%M:%S ') + ' '.join(f'{n}[{s} {d}]' for n, (s, d) in states.items()))
            last = states
        running = [n for n, (s, _) in states.items() if ACTIVE(s)]
        if not running:   # a short job (a CPU dry run) can start AND finish between two polls: it ran if its pod left a trace
            for n, (s, _) in states.items():
                if s in TERMINAL and ran.get(n) is None:
                    ran[n] = api.pods_started(experiment, n)   # None (S3 unknown) is retried next poll, never a winner
            running = [n for n in jobs if ran.get(n) is True]
            for n in running:
                seen.setdefault(n, api.clock())
        if running:
            winner = running[0]
            losers = {n: j for n, j in jobs.items() if n != winner}
            settled = _settle_losers(api, experiment, losers, poll, seen=seen)
            api.log(f'WINNER {winner} ({jobs[winner]}) after {api.clock() - started_at:.0f} s; cancelled {sorted(losers)}')
            return {'winner': winner, 'job_id': jobs[winner], 'waited_seconds': round(api.clock() - started_at), 'routes': [c for c, _, _ in chosen],
                    'jobs': jobs, 'losers': settled, 'receipt': receipts[winner], 'winner_seen_at': seen.get(winner)}
        terminal = all(s in TERMINAL for s, _ in states.values())
        unknown_polls = unknown_polls + 1 if terminal and any(ran.get(n) is None for n in jobs) else 0
        dead = [n for n, (s, _) in states.items() if s in TERMINAL and (ran.get(n) is False or unknown_polls >= 10)]   # ended without a pod
        if dead and len(dead) == len(jobs):   # (unknown evidence for 10 polls: give up; the settle step charges unknowns)
            settled = _settle_losers(api, experiment, dict(jobs), poll, seen=seen)
            return {'winner': None, 'reason': 'every route ended before starting', 'jobs': jobs, 'losers': settled}
        if max_wait_seconds is not None and api.clock() - started_at > max_wait_seconds:
            settled = _settle_losers(api, experiment, dict(jobs), poll, seen=seen)
            api.log(f'no route started within {max_wait_seconds / 3600:.1f} h: all cancelled')
            return {'winner': None, 'reason': f'no capacity within {max_wait_seconds:.0f} s', 'jobs': jobs, 'losers': settled}
        api.sleep(poll)


def _submit_routes(api, experiment, attempt, mode, chosen, overrides, jobs, receipts, seen):
    for code, team, borrow in chosen:
        name = f'{attempt}-{code}'
        # Staging takes ~30 s per route; when capacity is free the first route is often running before the last is submitted. Measured: in
        # all 5 GPU races of 2026-10-08 a second route was allocated in the same poll and had to be cancelled and charged. Stop submitting
        # once any submitted route started (no added latency when capacity is scarce: every route is still submitted).
        states = {n: _state(api, j)[0] for n, j in jobs.items()}
        if any(ACTIVE(st) for st in states.values()):
            for n, st in states.items():
                if ACTIVE(st):
                    seen.setdefault(n, api.clock())
            api.log(f'a submitted route already started: not submitting {name} or later routes')
            break
        extra = [] if borrow else ['compute.allow_borrowing=false']
        receipts[name] = api.stage(experiment, attempt=name, mode=mode, team=team, allow_borrowing=borrow, overrides=[*(overrides or []), *extra],
                                   reservation=attempt)   # the routes of one race share one budget reservation
        job_id = api.submit(receipts[name]['job_file'])
        if job_id:
            jobs[name] = job_id
            api.log(f'submitted {name}: team={team} borrowing={borrow} job={job_id}')
        else:
            api.log(f'{name} was not accepted by ScaleTrain; continuing with the other routes')


def watch(experiment: str, attempt: str, job_id: str, api: Api, *, poll: float = 60.0, seen_at: float | None = None,
          floor_poll: float | None = None) -> dict:
    """Follow a started job to its end. Returns {'outcome': 'terminal'|'preempted', 'status', ...} after recording it in the ledger.
    ``seen_at``: when the race first saw it allocated (the charge floor when the pod left no budget.json, e.g. a bootstrap failure)."""
    seen_at = seen_at if seen_at is not None else api.clock()
    while True:
        status, detail = _state(api, job_id)
        if status in TERMINAL:
            receipt = _record(api, experiment, attempt, job_id, allocated_floor=seen_at - (floor_poll if floor_poll is not None else poll))
            return {'outcome': 'terminal', 'status': status, 'attempt': attempt, 'job_id': job_id, 'ledger': receipt}
        if status == 'QUEUED' and 'preempt' in detail:
            api.log(f'{attempt} was preempted ({detail}): cancelling the requeued job (the budget window is anchored at its first start)')
            terminal_at = _iso(api.clock())
            settled = _settle_losers(api, experiment, {attempt: job_id}, poll=min(poll, 15.0), seen={attempt: seen_at})
            return {'outcome': 'preempted', 'status': status, 'attempt': attempt, 'job_id': job_id, 'terminal_at': terminal_at, 'settled': settled}
        api.sleep(poll)


def launch(experiment: str, attempt: str, mode: str, api: Api, *, overrides: list[str] | None = None, routes: str | None = None,
           poll: float = 15.0, max_wait_seconds: float | None = None, follow: bool = False, relaunch_on_preempt: int = 0,
           resume_on_preempt: bool = False) -> list[dict]:
    """race (+ watch, + relaunch after preemption; with resume_on_preempt, which needs evaluation.durable_checkpoints, the relaunch continues
    the preempted attempt from its latest checkpoint instead of starting over). Returns one receipt per race."""
    history, name, extra, winners = [], attempt, [], []
    for k in range(relaunch_on_preempt + 1):
        result = race(experiment, name, mode, api, overrides=[*(overrides or []), *extra], routes=routes, poll=poll, max_wait_seconds=max_wait_seconds)
        history.append(result)
        if not result.get('winner') or not follow:
            break
        outcome = watch(experiment, result['winner'], result['job_id'], api, seen_at=result.get('winner_seen_at'), floor_poll=poll)
        if any(x.get('unsettled') for x in outcome.get('settled') or []):
            result['outcome'] = outcome
            api.log('the preempted job could not be settled: NOT relaunching (two attempts could train at once)')
            break
        result['outcome'] = outcome
        if outcome['outcome'] != 'preempted':
            break
        name = f'{attempt}-p{k + 1}'
        winners.append(result['winner'])
        # continue from the latest attempt that left a durable checkpoint (a relaunch preempted before its first save has none)
        source = next((w for w in reversed(winners) if api.has_checkpoint(experiment, w)), None) if resume_on_preempt else None
        extra = [f'run.resume_from={source}'] if source else []
        if resume_on_preempt and not source:
            api.log('no durable checkpoint yet in any preempted attempt: the relaunch starts fresh')
        if k < relaunch_on_preempt:
            api.log(f'relaunching as {name} (preemption {k + 1} of {relaunch_on_preempt} allowed)')
    receipt_dir = Path('evidence') / attempt
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / 'launch.json').write_text(json.dumps(history, indent=2, default=str) + '\n')
    return history


def real_api(user_id: str, overrides: list[str] | None = None) -> Api:
    """The real ScaleTrain wiring (tools/scaletrain.py via the CLI helpers, strl.stage, the ledger)."""
    from . import cli
    from .stage import stage as stage_fn

    def stage(experiment, *, attempt, mode, team, allow_borrowing, overrides, reservation=None):
        receipt = stage_fn(experiment, attempt=attempt, mode=mode, team=team, allow_borrowing=allow_borrowing,
                           output=f'evidence/{attempt}/job.json', overrides=overrides, reservation=reservation)
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
        out = cli._scaletrain('cancel', '--user-id', user_id, job_id)
        if out.returncode:
            raise RuntimeError(f'cancel {job_id} failed: {(out.stderr or out.stdout)[-300:]}')
        return out

    def record(experiment, attempt, job_id, never_started=False, terminal_at=None, allocated_floor=None):
        return cli.record_attempt(experiment, attempt, job_id, never_started=never_started, terminal_at=terminal_at,
                                  overrides=overrides, allocated_floor=allocated_floor)

    def pods_started(experiment: str, attempt: str) -> bool | None:
        """True / False on positive S3 evidence; None when S3 cannot tell (expired credentials...): charged, but never a winner."""
        try:
            cfg = cli._load(experiment, overrides)
            return bool(cli.worker_started(cfg, attempt) or cli.allocation_start(cfg, attempt) is not None)
        except (Exception, SystemExit):
            return None

    def has_checkpoint(experiment: str, attempt: str) -> bool:
        try:
            import boto3
            st = cli._load(experiment, overrides)['storage']
            boto3.Session(profile_name=st['profile']).client('s3').head_object(
                Bucket=st['bucket'], Key=f"{st['prefix']}/attempts/{attempt}/artifacts/checkpoints/latest_ckpt_global_step.txt")
            return True
        except (Exception, SystemExit):
            return False

    return Api(stage=stage, submit=submit, get=get, cancel=cancel, record=record, pods_started=pods_started, has_checkpoint=has_checkpoint)


def _problems(history: list[dict]) -> list[dict]:
    """Jobs the launcher could not settle (still able to start) or could not record in the ledger, across every race and preemption."""
    rows = [l for h in history for l in h.get('losers') or []]
    rows += [l for h in history for l in (h.get('outcome') or {}).get('settled') or []]
    out = [l for l in rows if l.get('unsettled') or (l.get('ledger') or {}).get('unrecorded')]
    out += [h['outcome'] for h in history if ((h.get('outcome') or {}).get('ledger') or {}).get('unrecorded')]
    return out


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
    api = real_api(a.user_id, a.set)
    from . import cli
    from .overrides import durable
    resumable = durable(cli._load(a.experiment, a.set))
    history = launch(a.experiment, a.attempt, a.mode, api, overrides=a.set, routes=a.routes, poll=a.poll,
                     max_wait_seconds=a.max_wait_hours * 3600 if a.max_wait_hours else None, follow=a.watch,
                     relaunch_on_preempt=a.relaunch_on_preempt, resume_on_preempt=resumable)
    if a.max_wait_hours and not history[-1].get('winner') and a.fallback_set and not _problems(history):   # e.g. 2 nodes never came
        api.log(f'no capacity within {a.max_wait_hours} h: falling back with {a.fallback_set}')
        history += launch(a.experiment, a.attempt + FALLBACK_SUFFIX, a.mode, api, overrides=[*a.set, *a.fallback_set], routes=a.routes,
                          poll=a.poll, follow=a.watch, relaunch_on_preempt=a.relaunch_on_preempt, resume_on_preempt=resumable)
    print(json.dumps([{k: v for k, v in h.items() if k != 'receipt'} for h in history], indent=1, default=str))
    last = history[-1]
    problems = _problems(history)
    if problems:
        print(f'\nLAUNCH NEEDS ATTENTION: {len(problems)} job(s) unsettled or not recorded in the ledger (see UNSETTLED / LEDGER lines above)', file=sys.stderr)
        return 3
    if not last.get('winner'):
        print(f"\nno route started ({last.get('reason')}); fall back to a smaller shape (e.g. --set compute.nodes=1) or wait longer", file=sys.stderr)
        return 2
    if (last.get('outcome') or {}).get('outcome') == 'terminal' and last['outcome'].get('status') != 'COMPLETED':   # never report success
        print(f"\n{last['winner']} ended {last['outcome'].get('status')}: read artifacts/failure.log and training.log of that attempt", file=sys.stderr)
        return 5
    if (last.get('outcome') or {}).get('outcome') == 'preempted':
        print(f"\n{last['winner']} was preempted and no relaunch was left (--relaunch-on-preempt N to continue)", file=sys.stderr)
        return 4
    if a.mode == 'dry' and (last.get('outcome') or {}).get('status') == 'COMPLETED':
        print(dry_run_summary(a.experiment, last['winner'], a.set))
    print(f"\nnext: python -m strl monitor {a.experiment} --attempt {last['winner']} --job-id {last['job_id']}")
    return 0


def dry_run_summary(experiment: str, attempt: str, overrides: list[str] | None = None) -> str:
    """One screen of what a finished dry run proved (artifacts/dry-run.json), so nobody has to dig through S3 for it."""
    try:
        import boto3
        from . import cli
        st = cli._load(experiment, overrides)['storage']
        key = f"{st['prefix']}/attempts/{attempt}/artifacts/dry-run.json"
        d = json.loads(boto3.Session(profile_name=st['profile']).client('s3').get_object(Bucket=st['bucket'], Key=key)['Body'].read())
    except (Exception, SystemExit) as error:
        return f'\ndry run {attempt}: could not read dry-run.json ({type(error).__name__}: {error})'
    g = d.get('gates') or {}
    v = g.get('skyrl_validate') or {}
    lines = [f"\ndry run {attempt}: {d.get('status')}  (s3://{st['bucket']}/{key})",
             f"  model_code   {(d.get('model_code') or {}).get('status')} {(d.get('model_code') or {}).get('class')}",
             f"  skyrl        validated={v.get('validated')} overrides={v.get('overrides')} generator_contract="
             f"{'passed' if (v.get('generator') or {}).get('generator_config_passed') else (v.get('generator') or {})}",
             f"  layout       {json.dumps(v.get('layout') or {}, default=str)[:400]}",
             f"  environment  {json.dumps(g.get('environment'), default=str)[:300]}",
             f"  wandb        {json.dumps(g.get('wandb'), default=str)[:160]}"]
    for name in ('critic', 'durable_checkpoint_io', 'ceiling'):
        if name in g:
            lines.append(f"  {name:<12} {json.dumps(g[name], default=str)[:300]}")
    if (v.get('generator') or {}).get('shaping'):
        lines.append(f"  shaping      {json.dumps(v['generator']['shaping'])[:300]}")
    if d.get('network'):
        lines.append(f"  network      {json.dumps(d['network'], default=str)[:200]}")
    return '\n'.join(lines)
