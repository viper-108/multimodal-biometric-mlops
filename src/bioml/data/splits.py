from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa


@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float
    test: float
    seed: int


def add_split_column(table: pa.Table, cfg: SplitConfig) -> pa.Table:
    """
    Adds 'split' column with values in {"train","val","test"}.

    Fixes small-n behavior:
      - ensures at least 1 train sample per label when cfg.train > 0 and n>0
      - uses rounded allocation (then clamps) instead of pure floor
    """
    assert abs(cfg.train + cfg.val + cfg.test - 1.0) < 1e-6

    rows = table.to_pylist()
    by_label: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_label.setdefault(int(r["person_id"]), []).append(i)

    rng = np.random.default_rng(cfg.seed)
    split = ["train"] * len(rows)

    for _, idxs in by_label.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)

        if n == 0:
            continue

        # Rounded allocations
        n_train = round(n * cfg.train)
        n_val = round(n * cfg.val)

        # Ensure at least 1 train sample when possible
        if cfg.train > 0 and n_train == 0:
            n_train = 1

        # Clamp so we don't exceed n
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        train_idxs = idxs[:n_train]
        val_idxs = idxs[n_train : n_train + n_val]
        test_idxs = idxs[n_train + n_val :]

        for j in val_idxs:
            split[j] = "val"
        for j in test_idxs:
            split[j] = "test"
        for j in train_idxs:
            split[j] = "train"

    return table.append_column("split", pa.array(split))
