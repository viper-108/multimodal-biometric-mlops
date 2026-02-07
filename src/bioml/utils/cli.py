from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """
    Resolve repository root even when invoked from subdirectories.
    """
    cur = Path(os.getcwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return cur
