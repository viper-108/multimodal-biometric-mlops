from __future__ import annotations

from pathlib import Path

import hydra
import pyarrow.compute as pc
from omegaconf import DictConfig
from src.bioml.data.dataloading import make_dataloader
from src.bioml.data.manifest import load_manifest, resolve_manifest
from src.bioml.utils.cli import repo_root
from src.bioml.utils.logging import save_json, setup_logging
from src.bioml.utils.perf import measure_dataloader_throughput


@hydra.main(config_path="../../../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = setup_logging()
    root = repo_root()
    processed_dir = (root / Path(cfg.dataset.processed_dir)).resolve()

    mp = resolve_manifest(processed_dir, cfg.preprocess.manifest_name, cfg.preprocess.arrays_subdir)
    table = load_manifest(mp.manifest_path)
    train_tbl = table.filter(pc.equal(table["split"], "train"))

    if train_tbl.num_rows == 0:
        logger.warning("train split is empty; falling back to full table for benchmarking")
        train_tbl = table

    loader = make_dataloader(
        train_tbl,
        target_size=tuple(cfg.preprocess.target_size),
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=False,
        num_workers=int(cfg.dataloader.num_workers),
        pin_memory=bool(cfg.dataloader.pin_memory),
        persistent_workers=bool(cfg.dataloader.persistent_workers),
        prefetch_factor=int(cfg.dataloader.prefetch_factor),
    )

    report = measure_dataloader_throughput(loader, num_batches=200, warmup=20, device=None)
    payload = report.__dict__
    logger.info("throughput", extra=payload)

    out_path = processed_dir / "dataloader_benchmark.json"
    save_json(out_path, payload)
    logger.info("saved", extra={"path": str(out_path)})


if __name__ == "__main__":
    main()
