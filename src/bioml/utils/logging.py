from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("bioml")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class JsonlLogger:
    """Lightweight JSONL metrics logger (append-only)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe: dict[str, Any] = {k: to_jsonable(v) for k, v in payload.items()}
    path.write_text(json.dumps(safe, indent=2, ensure_ascii=False), encoding="utf-8")
