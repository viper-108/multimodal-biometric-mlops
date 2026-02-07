from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path


def get_git_sha(repo_root: Path) -> str | None:
    if not (repo_root / ".git").exists():
        return None
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def pip_freeze() -> str:
    try:
        return subprocess.check_output(["python", "-m", "pip", "freeze"]).decode("utf-8")
    except Exception:
        return ""


def system_info() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
