from __future__ import annotations

import pyarrow as pa
from src.bioml.data.dataset import MultiModalBiometricDataset
from torch.utils.data import DataLoader


def make_dataloader(
    manifest: pa.Table,
    target_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    ds = MultiModalBiometricDataset(manifest, target_size=target_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
