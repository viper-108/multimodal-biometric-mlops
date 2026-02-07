from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ReproducibilityReport:
    seed: int
    deterministic: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    pythonhashseed: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int, deterministic: bool = True) -> ReproducibilityReport:
    """
    Seed Python, NumPy, and PyTorch. Optionally enable deterministic algorithms.

    Notes:
      - Full determinism may reduce throughput; this is a trade-off explicitly documented.
      - Some CUDA ops remain non-deterministic depending on driver / hardware.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return ReproducibilityReport(
        seed=seed,
        deterministic=deterministic,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
    )
