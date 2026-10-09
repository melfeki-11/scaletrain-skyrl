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
