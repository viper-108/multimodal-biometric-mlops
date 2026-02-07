from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ModalityPaths:
    fingerprint: Path | None
    iris_left: Path | None
    iris_right: Path | None


def build_raw_index(
    raw_dir: Path,
    image_extensions: list[str],
    modality_aliases: dict[str, list[str]],
) -> pa.Table:
    """
    Scan `raw_dir` and create a canonical index table.

    Soft assumptions:
      - dataset contains per-person directories (person_id inferred from folder name)
      - within each person folder there are subfolders for modalities, but we also support
        layouts where modality is encoded in the filename.

    Output columns:
      person_id: int
      person_key: string
      session_id: int
      fingerprint_path: string?
      iris_left_path: string?
      iris_right_path: string?
    """
    raw_dir = raw_dir.resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    exts = {e.lower() for e in image_extensions}
    files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    grouped: dict[str, list[Path]] = {}
    for p in files:
        person_key = _infer_person_key(p, raw_dir)
        grouped.setdefault(person_key, []).append(p)

    rows = []
    for person_key, paths in grouped.items():
        person_id = _person_key_to_int(person_key)
        session_groups = _group_by_session(paths)

        for session_id, session_paths in session_groups.items():
            mpaths = _select_modalities(session_paths, modality_aliases)
            if not (mpaths.fingerprint or mpaths.iris_left or mpaths.iris_right):
                continue
            rows.append(
                {
                    "person_id": person_id,
                    "person_key": person_key,
                    "session_id": session_id,
                    "fingerprint_path": str(mpaths.fingerprint) if mpaths.fingerprint else None,
                    "iris_left_path": str(mpaths.iris_left) if mpaths.iris_left else None,
                    "iris_right_path": str(mpaths.iris_right) if mpaths.iris_right else None,
                }
            )

    if not rows:
        raise RuntimeError(
            "No samples found. Ensure raw_dir contains images and image_extensions are correct."
        )
    return pa.Table.from_pylist(rows)


def write_manifest(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# def _infer_person_key(p: Path, raw_dir: Path) -> str:
#     rel = p.relative_to(raw_dir)
#     return rel.parts[0] if rel.parts else "unknown"


def _infer_person_key(p: Path, raw_dir: Path) -> str:
    rel = p.relative_to(raw_dir)

    # Pick the first directory component that looks like an ID (digits).
    for part in rel.parts:
        if part.isdigit():
            return part

    # fallback
    return rel.parts[0] if rel.parts else "unknown"


def _person_key_to_int(person_key: str) -> int:
    digits = re.findall(r"\d+", person_key)
    if digits:
        return int(digits[0])
    return abs(hash(person_key)) % 10_000_000


def _group_by_session(paths: list[Path]) -> dict[int, list[Path]]:
    groups: dict[int, list[Path]] = {}
    for p in paths:
        digits = re.findall(r"\d+", p.stem)
        session_id = int(digits[-1]) if digits else 0
        groups.setdefault(session_id, []).append(p)
    return groups


def _select_modalities(paths: list[Path], modality_aliases: dict[str, list[str]]) -> ModalityPaths:
    def score(path: Path, modality: str) -> int:
        toks = [path.name.lower(), str(path.parent).lower()]
        aliases = [a.lower() for a in modality_aliases.get(modality, [])]
        return sum(1 for a in aliases for t in toks if a in t)

    best = {"fingerprint": None, "iris_left": None, "iris_right": None}
    best_score = {"fingerprint": -1, "iris_left": -1, "iris_right": -1}

    for p in paths:
        for m in best:
            s = score(p, m)
            if s > best_score[m]:
                best_score[m] = s
                best[m] = p

    return ModalityPaths(
        fingerprint=best["fingerprint"] if best_score["fingerprint"] > 0 else None,
        iris_left=best["iris_left"] if best_score["iris_left"] > 0 else None,
        iris_right=best["iris_right"] if best_score["iris_right"] > 0 else None,
    )
