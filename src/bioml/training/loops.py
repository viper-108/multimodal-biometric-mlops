from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    acc: float
    seconds: float


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    grad_clip_norm: float,
) -> EpochMetrics:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    t0 = time.perf_counter()

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for batch in loader:
        x_fp = batch["fingerprint"].to(device, non_blocking=True)
        x_il = batch["iris_left"].to(device, non_blocking=True)
        x_ir = batch["iris_right"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(x_fp, x_il, x_ir)
            loss = F.cross_entropy(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_acc += _accuracy(logits.detach(), y) * bs
        n += bs

    seconds = time.perf_counter() - t0
    return EpochMetrics(loss=total_loss / n, acc=total_acc / n, seconds=seconds)


@torch.inference_mode()
def eval_one_epoch(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> EpochMetrics:
    model.eval()
    t0 = time.perf_counter()

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for batch in loader:
        x_fp = batch["fingerprint"].to(device, non_blocking=True)
        x_il = batch["iris_left"].to(device, non_blocking=True)
        x_ir = batch["iris_right"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        logits = model(x_fp, x_il, x_ir)
        loss = F.cross_entropy(logits, y)

        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_acc += _accuracy(logits, y) * bs
        n += bs

    seconds = time.perf_counter() - t0
    return EpochMetrics(loss=total_loss / n, acc=total_acc / n, seconds=seconds)
