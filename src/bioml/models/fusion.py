from __future__ import annotations

import torch
import torch.nn as nn
from src.bioml.models.backbones import TinyCNN


class FusionCNN(nn.Module):
    """Three-branch CNN + fusion head."""

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        kernel_size: int,
        embedding_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.fp = TinyCNN(in_channels, channels, kernel_size, embedding_dim)
        self.il = TinyCNN(in_channels, channels, kernel_size, embedding_dim)
        self.ir = TinyCNN(in_channels, channels, kernel_size, embedding_dim)

        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(
        self, fingerprint: torch.Tensor, iris_left: torch.Tensor, iris_right: torch.Tensor
    ) -> torch.Tensor:
        z_fp = self.fp(fingerprint)
        z_il = self.il(iris_left)
        z_ir = self.ir(iris_right)
        z = torch.cat([z_fp, z_il, z_ir], dim=1)
        return self.head(z)
