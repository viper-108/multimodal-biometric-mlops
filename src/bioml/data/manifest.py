from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ManifestPaths:
    processed_dir: Path
    manifest_path: Path
    arrays_dir: Path


def resolve_manifest(processed_dir: Path, manifest_name: str, arrays_subdir: str) -> ManifestPaths:
    processed_dir = processed_dir.resolve()
    return ManifestPaths(
        processed_dir=processed_dir,
        manifest_path=processed_dir / manifest_name,
        arrays_dir=processed_dir / arrays_subdir,
    )


def load_manifest(manifest_path: Path) -> pa.Table:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run preprocessing to create it."
        )
    return pq.read_table(manifest_path)
