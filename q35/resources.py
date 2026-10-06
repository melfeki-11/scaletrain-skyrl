"""CPU resources actually granted to this container (a slice can expose every host CPU)."""
import math
import os
from pathlib import Path


def cgroup_quota(path='/sys/fs/cgroup/cpu.max'):
    try:
        quota, period = Path(path).read_text().split()[:2]
        return None if quota == 'max' else math.ceil(int(quota) / int(period))
    except (OSError, ValueError):
        return None


def allocated_cpus(quota_path='/sys/fs/cgroup/cpu.max'):
    visible = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    quota = cgroup_quota(quota_path)
    return max(1, min(visible, quota) if quota else visible)
