from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """Lightweight CNN backbone producing an embedding via global average pooling."""

    def __init__(self, in_channels: int, channels: list[int], kernel_size: int, embedding_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                nn.Conv2d(
                    c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
                ),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ]
            c_in = c_out

        self.features = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_in, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.proj(x)
