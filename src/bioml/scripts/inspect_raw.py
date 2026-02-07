from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig
from src.bioml.data.indexer import build_raw_index
from src.bioml.utils.cli import repo_root
from src.bioml.utils.logging import setup_logging


@hydra.main(config_path="../../../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = setup_logging()
    root = repo_root()
    raw_dir = (root / Path(cfg.dataset.raw_dir)).resolve()

    table = build_raw_index(
        raw_dir=raw_dir,
        image_extensions=list(cfg.dataset.image_extensions),
        modality_aliases={k: list(v) for k, v in cfg.dataset.modalities.items()},
    )

    rows = table.to_pylist()
    logger.info("raw scan complete", extra={"raw_dir": str(raw_dir), "rows": len(rows)})

    for r in rows[:10]:
        logger.info("sample", extra=r)


if __name__ == "__main__":
    main()
