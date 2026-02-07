from __future__ import annotations

from typing import Any

import torch.nn as nn
from src.bioml.models.fusion import FusionCNN


def build_model(cfg: Any) -> nn.Module:
    name = cfg.model.name
    if name == "fusion_cnn":
        return FusionCNN(
            in_channels=int(cfg.model.backbone.in_channels),
            channels=list(cfg.model.backbone.channels),
            kernel_size=int(cfg.model.backbone.kernel_size),
            embedding_dim=int(cfg.model.embedding_dim),
            num_classes=int(cfg.model.num_classes),
            dropout=float(cfg.model.dropout),
        )
    raise ValueError(f"Unknown model: {name}")
