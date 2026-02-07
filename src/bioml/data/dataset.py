from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset


def _load_npy(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    return np.load(path, mmap_mode="r")


class MultiModalBiometricDataset(Dataset):
    """
    Returns:
      fingerprint: FloatTensor [1,H,W]
      iris_left:  FloatTensor [1,H,W]
      iris_right: FloatTensor [1,H,W]
      label: LongTensor []       (dense class_id: 0..C-1)
      person_id: LongTensor []   (original id from dataset)
      sample_id: LongTensor []
    """

    def __init__(self, manifest: pa.Table, target_size: tuple[int, int], normalize: bool = True):
        self.manifest = manifest
        self.target_size = target_size
        self.normalize = normalize
        self._rows = manifest.to_pylist()

        # Dense label encoding (stable, deterministic)
        person_ids = [int(x) for x in manifest["person_id"].to_pylist()]
        unique = sorted(set(person_ids))
        self.label_to_class = {pid: i for i, pid in enumerate(unique)}
        self.class_to_label = {i: pid for pid, i in self.label_to_class.items()}

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self._rows[idx]

        fp = self._to_tensor(_load_npy(r.get("fingerprint_npy")))
        il = self._to_tensor(_load_npy(r.get("iris_left_npy")))
        ir = self._to_tensor(_load_npy(r.get("iris_right_npy")))

        fp = fp if fp is not None else torch.zeros((1, *self.target_size), dtype=torch.float32)
        il = il if il is not None else torch.zeros((1, *self.target_size), dtype=torch.float32)
        ir = ir if ir is not None else torch.zeros((1, *self.target_size), dtype=torch.float32)

        person_id = int(r["person_id"])
        label = self.label_to_class[person_id]
        sample_id = int(r["sample_id"])

        return {
            "fingerprint": fp,
            "iris_left": il,
            "iris_right": ir,
            "label": torch.tensor(label, dtype=torch.long),  # ✅ dense 0..C-1
            "person_id": torch.tensor(person_id, dtype=torch.long),  # original
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
        }

    def _to_tensor(self, arr: np.ndarray | None) -> torch.Tensor | None:
        if arr is None:
            return None
        if arr.ndim == 3:
            arr = arr.mean(axis=2)

        # If you want to remove the "not writable" warning, use .copy() here:
        # x = torch.from_numpy(np.asarray(arr).copy()).to(torch.float32)
        x = torch.from_numpy(np.asarray(arr)).to(torch.float32)

        if self.normalize:
            x = x / 255.0
        return x.unsqueeze(0)
