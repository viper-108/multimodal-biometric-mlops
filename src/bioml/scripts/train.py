from __future__ import annotations

import hydra
from omegaconf import DictConfig
from src.bioml.training.train import train
from src.bioml.utils.logging import setup_logging


@hydra.main(config_path="../../../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = setup_logging()
    run_dir = train(cfg)
    logger.info("training complete", extra={"run_dir": str(run_dir)})


if __name__ == "__main__":
    main()
