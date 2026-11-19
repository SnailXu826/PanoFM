"""Shared runtime, data loaders, and portable checkpoint I/O."""

import os
import random
import tempfile
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int
    world_size: int
    workers: int
    seed: int
    amp: bool

    @property
    def primary(self):
        return self.rank == 0

    def autocast(self):
        if self.amp:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def model(self, module, unused=False):
        module = module.to(self.device)
        if self.world_size == 1:
            return module
        device_ids = [self.device.index] if self.device.type == "cuda" else None
        return DistributedDataParallel(module, device_ids=device_ids, find_unused_parameters=unused)

    def local_batch(self, global_batch: int) -> int:
        if global_batch < self.world_size or global_batch % self.world_size:
            raise ValueError("The batch size must be positive and divisible by the number of ranks")
        return global_batch // self.world_size


@contextmanager
def runtime(device: str, workers: int, seed: int, distributed: bool = False):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not distributed and world_size > 1:
        raise ValueError("Run this command with one process")
    rank = int(os.environ.get("RANK", "0")) if world_size > 1 else 0
    selected = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                            "cpu" if device == "auto" else device)
    if selected.type == "cuda":
        index = int(os.environ["LOCAL_RANK"]) if world_size > 1 else (selected.index or 0)
        selected = torch.device("cuda", index)
        torch.cuda.set_device(selected)
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    amp = selected.type == "cuda" and torch.cuda.is_bf16_supported()
    if world_size > 1:
        torch.distributed.init_process_group("nccl" if selected.type == "cuda" else "gloo")
    try:
        yield Runtime(selected, rank, world_size, workers, seed, amp)
    finally:
        if world_size > 1:
            torch.distributed.destroy_process_group()


class ArchBatches:
    """Pack whole radiographs into batches so contralateral teeth stay together."""

    def __init__(self, records, size: int | None = None, runtime: Runtime | None = None):
        groups = defaultdict(list)
        for index, record in enumerate(records):
            groups[record.image_id].append(index)
        self.groups = list(groups.values())
        self.size = size
        self.runtime = runtime
        self.epoch = 0
        if size is not None and max(map(len, self.groups)) > size:
            raise ValueError("The per-rank batch size must fit the largest radiograph")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _batches(self):
        if self.size is None:
            return self.groups
        generator = torch.Generator().manual_seed(self.runtime.seed + self.epoch)
        order = torch.randperm(len(self.groups), generator=generator).tolist()
        batches, batch = [], []
        for index in order:
            group = self.groups[index]
            if len(group) + len(batch) > self.size:
                batches.append(batch)
                batch = []
            batch.extend(group)
        batches.append(batch)
        steps = len(batches) // self.runtime.world_size
        if steps == 0:
            raise ValueError("Instance pretraining needs at least one batch for every rank")
        return batches[self.runtime.rank:steps * self.runtime.world_size:self.runtime.world_size]

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())


def loader(dataset, rt: Runtime, batch_size: int | None = None, *, training=False, batch_sampler=None):
    options = dict(num_workers=rt.workers, pin_memory=rt.device.type == "cuda",
                   persistent_workers=rt.workers > 0)
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **options)
    sampler = None
    if training and rt.world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True, seed=rt.seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training and sampler is None,
                      sampler=sampler, **options)


def set_epoch(batches, epoch: int):
    for sampler in (batches.sampler, batches.batch_sampler):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)


def loss_mean(total: torch.Tensor, count: int | torch.Tensor) -> torch.Tensor:
    """Normalize a local loss sum for DDP's averaged parameter gradients."""
    denominator = torch.as_tensor(count, device=total.device, dtype=torch.float32).clone()
    dist = torch.distributed
    if dist.is_initialized():
        dist.all_reduce(denominator)
        total = total * dist.get_world_size()
    return total / denominator.clamp_min(1)


def save_checkpoint(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A checkpoint reader sees a complete file even if the writer is interrupted.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        destination = Path(temporary.name)
    try:
        torch.save(values, destination)
        destination.replace(path)
    finally:
        destination.unlink(missing_ok=True)


def load_checkpoint(path: Path, *kinds: str) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("kind") not in kinds:
        raise ValueError(f"Expected a checkpoint of type {', '.join(kinds)}")
    return state
