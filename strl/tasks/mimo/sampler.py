"""Domain-stratified batches: every optimizer step draws ``batch_size`` prompts split across the training domains in proportion to
their configured weights (largest-remainder rounding, every weighted domain gets >= 1 when the batch allows), shuffled within
each domain, deterministic per (seed, epoch), no duplicate inside a batch and none repeated within an epoch.

An epoch ends when the scarcest domain runs out (``len = min_d(n_d // quota_d)``), so every step of an epoch has the same domain
mix. SkyRL's stock dataloader would draw uniformly from the concatenated dataset and the mix would follow the dataset sizes.

Wiring (strl/entrypoint.py): ``EnvironmentSpec.train_batch_sampler`` returns a ``DomainBatchSampler``; ``make_trainer_class``
overrides ``RayPPOTrainer._build_train_dataloader_and_compute_training_steps`` (pin a9564572) to build the StatefulDataLoader
with ``batch_sampler=`` instead of ``batch_size/shuffle``. Resuming mid-epoch restarts the epoch counter (training here runs
without resume; ``set_epoch`` exists for callers that resume).
"""
from __future__ import annotations

import random
from typing import Any, Callable, Iterator, Sequence


def quotas(weights: dict[str, float], batch_size: int) -> dict[str, int]:
    """Largest-remainder apportionment of ``batch_size`` slots by weight; each domain gets at least 1 slot when batch_size >= the
    number of domains (donated by the domain with the largest quota above 1)."""
    if not weights or any(w <= 0 for w in weights.values()):
        raise ValueError('every domain weight must be > 0')
    total = sum(weights.values())
    exact = {d: batch_size * w / total for d, w in weights.items()}
    out = {d: int(v) for d, v in exact.items()}
    for domain in sorted(exact, key=lambda d: (-(exact[d] - out[d]), d))[:batch_size - sum(out.values())]:
        out[domain] += 1
    if batch_size >= len(weights):
        for domain in [d for d in out if out[d] == 0]:
            donor = max((d for d in out if out[d] > 1), key=lambda d: (out[d], d))
            out[donor] -= 1
            out[domain] = 1
    return out


class DomainBatchSampler:
    """Batch sampler over dataset indices. ``domains[i]`` is the domain of dataset item ``i``; ``weights`` maps domain -> weight
    (domains without weight, e.g. eval-only ones, are never sampled)."""

    def __init__(self, domains: Sequence[str], weights: dict[str, float], batch_size: int, seed: int = 0):
        self.members: dict[str, list[int]] = {d: [] for d in weights}
        for index, domain in enumerate(domains):
            if domain in self.members:
                self.members[domain].append(index)
        self.quota = quotas(weights, batch_size)
        short = {d: (len(self.members[d]), q) for d, q in self.quota.items() if len(self.members[d]) < q}
        if short:
            raise ValueError('domains with fewer training tasks than their per-step quota (tasks, quota): ' + str(short))
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0

    def __len__(self) -> int:
        return min(len(self.members[d]) // q for d, q in self.quota.items())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def epoch_batches(self, epoch: int) -> list[list[int]]:
        order = {d: random.Random(f'{self.seed}|{epoch}|{d}').sample(m, len(m)) for d, m in self.members.items()}
        batches = []
        for step in range(len(self)):
            batch = [i for d, q in self.quota.items() for i in order[d][step * q:(step + 1) * q]]
            random.Random(f'{self.seed}|{epoch}|step{step}').shuffle(batch)   # mix domains across the GPUs that split the batch
            batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        epoch, self.epoch = self.epoch, self.epoch + 1
        yield from self.epoch_batches(epoch)


def build_sampled_dataloader(cfg: Any, dataset: Any, sampler: DomainBatchSampler) -> Any:
    """SkyRL's ``build_dataloader`` (trainer_utils.py, pin a9564572) with ``batch_sampler`` in place of batch_size/shuffle/drop_last."""
    import torch
    from torchdata.stateful_dataloader import StatefulDataLoader
    generator = torch.Generator()
    generator.manual_seed(cfg.trainer.seed)
    workers = cfg.data.dataloader.num_workers
    return StatefulDataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collate_fn, num_workers=workers,
                              persistent_workers=cfg.data.dataloader.persistent_workers, generator=generator,
                              multiprocessing_context='spawn' if workers > 0 else None)


def make_trainer_class(base: type, sampler: DomainBatchSampler, build_loader: Callable[..., Any] | None = None) -> type:
    """Subclass of SkyRL's RayPPOTrainer whose train dataloader draws domain-stratified batches. ``build_loader(cfg, dataset, sampler)``
    is injectable (tests)."""
    loader = build_loader or build_sampled_dataloader

    class DomainSampledTrainer(base):
        def _build_train_dataloader_and_compute_training_steps(self):
            if self.train_dataset is None:
                return super()._build_train_dataloader_and_compute_training_steps()
            self.train_dataloader = loader(self.cfg, self.train_dataset, sampler)
            self.total_training_steps = len(self.train_dataloader) * self.cfg.trainer.epochs
            if self.cfg.trainer.max_training_steps is not None:
                self.total_training_steps = min(self.total_training_steps, self.cfg.trainer.max_training_steps)

    return DomainSampledTrainer
