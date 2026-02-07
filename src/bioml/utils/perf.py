from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ThroughputReport:
    num_batches: int
    batch_size: int
    seconds: float
    batches_per_s: float
    samples_per_s: float


@torch.inference_mode()
def measure_dataloader_throughput(
    loader: DataLoader,
    num_batches: int = 200,
    warmup: int = 20,
    device: torch.device | None = None,
) -> ThroughputReport:
    """Measure dataloader throughput. Optionally includes host->device transfer cost."""
    it = iter(loader)
    for _ in range(warmup):
        batch = next(it)
        if device is not None:
            _move_to_device(batch, device)

    start = time.perf_counter()
    bs = loader.batch_size or 1
    for _ in range(num_batches):
        batch = next(it)
        if device is not None:
            _move_to_device(batch, device)
    end = time.perf_counter()

    seconds = end - start
    bps = num_batches / seconds
    sps = (num_batches * bs) / seconds
    return ThroughputReport(
        num_batches=num_batches,
        batch_size=bs,
        seconds=seconds,
        batches_per_s=bps,
        samples_per_s=sps,
    )


def _move_to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        return [_move_to_device(x, device) for x in batch]
    return batch
