from __future__ import annotations

import torch
from torch import nn


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
