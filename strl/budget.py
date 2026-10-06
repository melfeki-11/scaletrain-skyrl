"""Retry accounting uses allocated intervals, never time spent queued."""
import math
from datetime import datetime


def remaining_seconds(attempts, total=3600):
    """Node-seconds left in a program ledger. A GPU-slice attempt is charged ``gpu_fraction`` of its allocated interval
    (a 1-GPU canary on an 8-GPU preset is 1/8); CPU-only attempts carry gpu_fraction 0. Queue time is never charged."""
    seen = set()
    consumed = 0
    for attempt in attempts:
        if attempt['job_id'] in seen:
            raise ValueError('Duplicate budget attempt')
        seen.add(attempt['job_id'])
        if not attempt.get('allocated_at'):
            if attempt['status'] not in {'CANCELED', 'FAILED', 'COMPLETED'}:
                raise ValueError('Cannot retry an active or ambiguous job')
            continue
        if not attempt.get('terminal_at'):
            raise ValueError('Allocated job has no terminal evidence')
        start = datetime.fromisoformat(attempt['allocated_at'].replace('Z', '+00:00'))
        end = datetime.fromisoformat(attempt['terminal_at'].replace('Z', '+00:00'))
        if end < start: raise ValueError('Invalid allocation interval')
        consumed += math.ceil((end - start).total_seconds() * float(attempt.get('gpu_fraction', 1.0)))
    return max(0, total - consumed)
