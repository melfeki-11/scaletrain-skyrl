"""Retry accounting uses allocated intervals, never time spent queued."""
import math
from datetime import datetime


def remaining_seconds(attempts, total=3600):
    """Node-seconds left in a program ledger. An attempt is charged ``nodes x gpu_fraction`` of its allocated interval
    (gpu_fraction is per node: a 1-GPU canary on an 8-GPU preset is 1/8, a 2 x 8 H100 job is 2 x 1); CPU-only attempts carry
    gpu_fraction 0; entries recorded before multi-node support have no ``nodes`` and count as 1. When ScaleTrain reports no
    allocation time, the job's creation time is recorded instead (conservative: queue and pod start are then charged)."""
    seen = set()
    consumed = 0
    for attempt in attempts:
        if attempt['job_id'] in seen:
            raise ValueError('Duplicate budget attempt')
        seen.add(attempt['job_id'])
        if not attempt.get('allocated_at'):
            if attempt['status'] not in {'CANCELED', 'CANCELLED', 'FAILED', 'COMPLETED'}:
                raise ValueError('Cannot retry an active or ambiguous job')
            continue
        if not attempt.get('terminal_at'):
            raise ValueError('Allocated job has no terminal evidence')
        start = datetime.fromisoformat(attempt['allocated_at'].replace('Z', '+00:00'))
        end = datetime.fromisoformat(attempt['terminal_at'].replace('Z', '+00:00'))
        if end < start: raise ValueError('Invalid allocation interval')
        consumed += math.ceil((end - start).total_seconds() * float(attempt.get('gpu_fraction', 1.0)) * int(attempt.get('nodes', 1)))
    return max(0, total - consumed)


def node_fraction(cfg, mode='train'):
    """Node-seconds charged per wall-second: nodes x gpus_per_node / 8 (CPU dry runs are free)."""
    return 0.0 if mode == 'dry' else cfg['compute']['nodes'] * cfg['compute']['gpus_per_node'] / 8.0


RESERVATION_TTL_SECONDS = 72 * 3600   # a 2 x 8 job can queue ~23 h; a reservation nobody released expires after 3 days


def reserved_seconds(data: dict, *, exclude: str | None = None, now: float | None = None) -> int:
    """Node-seconds held by launches that are staged but not yet recorded (``data['reservations']``: group -> {node_seconds, attempts,
    created}). ``exclude``: the caller's own group (a race's routes share one reservation)."""
    import time
    now = time.time() if now is None else now
    return int(sum(r['node_seconds'] for g, r in (data.get('reservations') or {}).items()
                   if g != exclude and now - float(r.get('touched', r.get('created', now))) < RESERVATION_TTL_SECONDS))


def release_attempt(data: dict, attempt: str, group: str | None = None) -> bool:
    """Drop ``attempt`` from its reservation group(s) inside ``data`` (a locked ledger); a group goes when no attempt is left."""
    found = False
    for g, held in list((data.get('reservations') or {}).items()):
        if (group is None or g == group) and attempt in held.get('attempts', []):
            held['attempts'] = [x for x in held['attempts'] if x != attempt]
            found = True
            if not held['attempts']:
                del data['reservations'][g]
    return found


def locked_ledger(path):
    """Context manager: exclusive file lock on a program ledger, yields its data; the data is written back atomically on a clean exit.
    Every writer (stage reservations, record_attempt) goes through it: concurrent launchers share one ledger."""
    import contextlib
    import fcntl
    import json
    import os
    from pathlib import Path

    @contextlib.contextmanager
    def _cm():
        p = Path(path)
        with open(str(p) + '.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(p.read_text())
            yield data
            tmp = Path(str(p) + '.tmp')
            tmp.write_text(json.dumps(data, indent=2) + '\n')
            os.replace(tmp, p)
    return _cm()
