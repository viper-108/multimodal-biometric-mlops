from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
