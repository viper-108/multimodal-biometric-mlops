from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig
from src.bioml.inference.predict import predict_one
from src.bioml.utils.logging import setup_logging


@hydra.main(config_path="../../../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = setup_logging()
    out = predict_one(cfg)
    logger.info("prediction", extra=out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
