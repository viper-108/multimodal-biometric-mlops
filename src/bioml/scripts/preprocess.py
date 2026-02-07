from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from src.bioml.data.indexer import build_raw_index
from src.bioml.data.preprocessing import preprocess_manifest, save_preprocess_artifacts
from src.bioml.data.splits import SplitConfig, add_split_column
from src.bioml.utils.cli import repo_root
from src.bioml.utils.logging import setup_logging


def _maybe_download_from_kaggle(cfg: Any, raw_dir: Path) -> None:
    if not bool(cfg.dataset.download_from_kaggle):
        return
    # Requires kaggle API token at ~/.kaggle/kaggle.json
    import subprocess

    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        str(cfg.dataset.kaggle_dataset),
        "-p",
        str(raw_dir),
        "--unzip",
    ]
    subprocess.check_call(cmd)


@hydra.main(config_path="../../../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = setup_logging()
    root = repo_root()
    raw_dir = (root / Path(cfg.dataset.raw_dir)).resolve()
    processed_dir = (root / Path(cfg.dataset.processed_dir)).resolve()

    _maybe_download_from_kaggle(cfg, raw_dir)

    logger.info("building raw index", extra={"raw_dir": str(raw_dir)})
    table = build_raw_index(
        raw_dir=raw_dir,
        image_extensions=list(cfg.dataset.image_extensions),
        modality_aliases={k: list(v) for k, v in cfg.dataset.modalities.items()},
    )

    split_cfg = SplitConfig(
        train=float(cfg.dataset.split.train),
        val=float(cfg.dataset.split.val),
        test=float(cfg.dataset.split.test),
        seed=int(cfg.dataset.split.seed),
    )
    table = add_split_column(table, split_cfg)

    logger.info(
        "preprocessing",
        extra={"backend": cfg.preprocess.backend, "workers": cfg.preprocess.num_workers},
    )
    table2, stats = preprocess_manifest(
        manifest=table,
        processed_dir=processed_dir,
        arrays_subdir=str(cfg.preprocess.arrays_subdir),
        target_size=tuple(cfg.preprocess.target_size),
        grayscale=bool(cfg.preprocess.grayscale),
        num_workers=int(cfg.preprocess.num_workers),
        backend=str(cfg.preprocess.backend),
        profile=bool(cfg.preprocess.profile),
    )

    save_preprocess_artifacts(processed_dir, str(cfg.preprocess.manifest_name), table2, stats)
    (processed_dir / "config_preprocess.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    logger.info("done", extra={"processed_dir": str(processed_dir), **stats})


if __name__ == "__main__":
    main()
