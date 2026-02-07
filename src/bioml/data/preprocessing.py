from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class PreprocessResult:
    sample_id: int
    fingerprint_npy: str | None
    iris_left_npy: str | None
    iris_right_npy: str | None


def preprocess_manifest(
    manifest: pa.Table,
    processed_dir: Path,
    arrays_subdir: str,
    target_size: tuple[int, int],
    grayscale: bool,
    num_workers: int,
    backend: str = "multiprocessing",
    profile: bool = True,
) -> tuple[pa.Table, dict[str, Any]]:
    """
    Convert raw image paths -> resized numpy arrays stored on disk, and produce an updated manifest.

    Metadata stays in Parquet (PyArrow); heavy tensors stored as .npy for mmap-friendly reads.
    """
    processed_dir = processed_dir.resolve()
    arrays_dir = processed_dir / arrays_subdir
    arrays_dir.mkdir(parents=True, exist_ok=True)

    manifest = _with_sample_id(manifest)
    rows = manifest.to_pylist()

    worker_args = [
        (
            r["sample_id"],
            r.get("fingerprint_path"),
            r.get("iris_left_path"),
            r.get("iris_right_path"),
            str(arrays_dir),
            target_size,
            grayscale,
        )
        for r in rows
    ]

    t0 = time.perf_counter()
    if backend == "ray":
        out = _run_ray(worker_args, num_workers)
    else:
        out = _run_multiprocessing(worker_args, num_workers)
    elapsed = time.perf_counter() - t0

    updated_rows = []
    for r, pr in zip(rows, out, strict=True):
        updated_rows.append(
            {
                **r,
                "fingerprint_npy": pr.fingerprint_npy,
                "iris_left_npy": pr.iris_left_npy,
                "iris_right_npy": pr.iris_right_npy,
            }
        )

    table = pa.Table.from_pylist(updated_rows)
    stats: dict[str, Any] = {
        "num_samples": table.num_rows,
        "elapsed_s": elapsed,
        "samples_per_s": float(table.num_rows / elapsed) if elapsed > 0 else None,
        "backend": backend,
        "num_workers": num_workers,
        "target_size": list(target_size),
        "grayscale": grayscale,
    }

    if profile:
        counts = {
            "has_fingerprint": int(
                pc.sum(pc.cast(pc.is_valid(table["fingerprint_npy"]), pa.int32())).as_py()
            ),
            "has_iris_left": int(
                pc.sum(pc.cast(pc.is_valid(table["iris_left_npy"]), pa.int32())).as_py()
            ),
            "has_iris_right": int(
                pc.sum(pc.cast(pc.is_valid(table["iris_right_npy"]), pa.int32())).as_py()
            ),
        }
        stats.update(counts)

    return table, stats


def save_preprocess_artifacts(
    processed_dir: Path, manifest_name: str, table: pa.Table, stats: dict[str, Any]
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, processed_dir / manifest_name)
    (processed_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _with_sample_id(table: pa.Table) -> pa.Table:
    if "sample_id" in table.column_names:
        return table
    sample_id = pa.array(list(range(table.num_rows)), type=pa.int64())
    return table.append_column("sample_id", sample_id)


def _load_and_resize(path: str, target_size: tuple[int, int], grayscale: bool) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L" if grayscale else "RGB")
        img = img.resize((target_size[1], target_size[0]))  # PIL uses (W,H)
        return np.asarray(img)


def _save_npy(arr: np.ndarray, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    return str(out_path)


def _process_one(args) -> PreprocessResult:
    sample_id, fp, il, ir, arrays_dir, target_size, grayscale = args
    arrays_dir = Path(arrays_dir)

    def handle(path: str | None, suffix: str) -> str | None:
        if not path:
            return None
        try:
            arr = _load_and_resize(path, target_size, grayscale)
            out_path = arrays_dir / f"{int(sample_id)}_{suffix}.npy"
            return _save_npy(arr, out_path)
        except Exception:
            return None

    return PreprocessResult(
        sample_id=int(sample_id),
        fingerprint_npy=handle(fp, "fp"),
        iris_left_npy=handle(il, "il"),
        iris_right_npy=handle(ir, "ir"),
    )


def _run_multiprocessing(worker_args, num_workers: int) -> list[PreprocessResult]:
    from concurrent.futures import ProcessPoolExecutor

    num_workers = max(1, int(num_workers))
    out: list[PreprocessResult] = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for r in tqdm(ex.map(_process_one, worker_args), total=len(worker_args), desc="preprocess"):
            out.append(r)
    return out


def _run_ray(worker_args, num_workers: int) -> list[PreprocessResult]:
    try:
        import ray  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Ray backend requested but ray is not installed. Install with: pip install -e '.[ray]'"
        ) from e

    if not ray.is_initialized():
        ray.init(num_cpus=max(1, int(num_workers)), ignore_reinit_error=True, log_to_driver=False)

    @ray.remote
    def _remote_process_one(a):
        return _process_one(a)

    futures = [_remote_process_one.remote(a) for a in worker_args]
    return ray.get(futures)
