"""Group masking for infrastructure failures (training only).

An env can only return a reward; if an executor outage scored a rollout 0, that 0 would enter its GRPO group's mean/std and bias
the advantages of the other samples. SkyRL's HarborGenerator already masks a whole prompt group when any rollout fails; this does
the same for single-turn envs: every group containing an ``infra_error`` rollout gets zero loss masks and zero rewards (an all-zero
group has zero advantage). Eval batches are never masked: evaluation retries harder and fails loudly instead.
"""
from __future__ import annotations

from typing import Any


def mask_infra_groups(output: dict, trajectory_ids: list | None) -> dict[str, float]:
    env_metrics = output.get('env_metrics') or []
    ids = [getattr(t, 'instance_id', None) for t in (trajectory_ids or output.get('trajectory_ids') or [])]
    if not env_metrics or len(ids) != len(env_metrics):
        return {'environment/masked_group_fraction': 0.0, 'environment/masked_rollouts': 0.0}
    bad = {ids[i] for i, m in enumerate(env_metrics) if (m or {}).get('infra_error')}
    masked = 0
    for i, instance in enumerate(ids):
        if instance in bad:
            output['loss_masks'][i] = [0] * len(output['loss_masks'][i])
            reward = output['rewards'][i]
            output['rewards'][i] = [0.0] * len(reward) if isinstance(reward, list) else 0.0
            masked += 1
    groups = len(set(ids))
    return {'environment/masked_group_fraction': len(bad) / groups if groups else 0.0, 'environment/masked_rollouts': float(masked)}


def make_generator_class():
    from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

    class GroupMaskingGymGenerator(SkyRLGymGenerator):
        async def generate(self, input_batch: Any, **kwargs: Any):
            output = await super().generate(input_batch, **kwargs)
            metadata = input_batch.get('batch_metadata')
            if getattr(metadata, 'training_phase', 'train') == 'train':
                stats = mask_infra_groups(output, input_batch.get('trajectory_ids'))
                output['rollout_metrics'] = {**(output.get('rollout_metrics') or {}), **stats}
            return output

    return GroupMaskingGymGenerator
