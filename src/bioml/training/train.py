from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import torch
from omegaconf import OmegaConf
from src.bioml.data.dataloading import make_dataloader
from src.bioml.data.manifest import load_manifest, resolve_manifest
from src.bioml.models.registry import build_model
from src.bioml.training.checkpointing import save_checkpoint
from src.bioml.training.loops import eval_one_epoch, train_one_epoch
from src.bioml.training.optim import make_optimizer
from src.bioml.utils.cli import repo_root
from src.bioml.utils.io import get_git_sha, pip_freeze, system_info
from src.bioml.utils.logging import JsonlLogger, save_json, setup_logging
from src.bioml.utils.reproducibility import seed_everything
from torch.utils.tensorboard import SummaryWriter


def pick_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg: Any) -> Path:
    logger = setup_logging()
    root = repo_root()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (root / cfg.run.output_dir / f"{run_id}_{cfg.run.experiment_name}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    rep = seed_everything(int(cfg.train.seed), deterministic=bool(cfg.train.deterministic))

    save_json(run_dir / "reproducibility.json", rep.to_dict())
    save_json(run_dir / "system.json", system_info())
    save_json(run_dir / "git.json", {"sha": get_git_sha(root)})
    (run_dir / "pip_freeze.txt").write_text(pip_freeze(), encoding="utf-8")
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    mp = resolve_manifest(
        Path(cfg.dataset.processed_dir), cfg.preprocess.manifest_name, cfg.preprocess.arrays_subdir
    )
    table = load_manifest(mp.manifest_path)

    # Deterministic num_classes based on sorted unique person_ids
    person_ids = [int(x) for x in table["person_id"].to_pylist()]
    unique_labels = sorted(set(person_ids))
    cfg.model.num_classes = len(unique_labels)

    person_ids = sorted(set(int(x) for x in table["person_id"].to_pylist()))
    label_map = {i: pid for i, pid in enumerate(person_ids)}
    save_json(run_dir / "label_map.json", {"class_id_to_person_id": label_map})

    # unique_labels = pc.unique(table["person_id"]).to_pylist()
    # cfg.model.num_classes = int(len(unique_labels))
    logger.info("num_classes inferred", extra={"num_classes": cfg.model.num_classes})

    train_tbl = table.filter(pc.equal(table["split"], "train"))
    val_tbl = table.filter(pc.equal(table["split"], "val"))

    target_size = tuple(cfg.preprocess.target_size)
    train_loader = make_dataloader(
        train_tbl,
        target_size=target_size,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=True,
        num_workers=int(cfg.dataloader.num_workers),
        pin_memory=bool(cfg.dataloader.pin_memory),
        persistent_workers=bool(cfg.dataloader.persistent_workers),
        prefetch_factor=int(cfg.dataloader.prefetch_factor),
    )
    val_loader = make_dataloader(
        val_tbl,
        target_size=target_size,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=False,
        num_workers=int(cfg.dataloader.num_workers),
        pin_memory=bool(cfg.dataloader.pin_memory),
        persistent_workers=bool(cfg.dataloader.persistent_workers),
        prefetch_factor=int(cfg.dataloader.prefetch_factor),
    )

    device = pick_device(str(cfg.train.device))
    model = build_model(cfg).to(device)
    optimizer = make_optimizer(
        model, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay)
    )

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    metrics = JsonlLogger(run_dir / "metrics.jsonl")

    ckpt_dir = run_dir / cfg.train.checkpoint.save_dirname
    best_val = float("inf")

    for epoch in range(int(cfg.train.epochs)):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp=bool(cfg.train.amp),
            grad_clip_norm=float(cfg.train.grad_clip_norm),
        )
        va = eval_one_epoch(model=model, loader=val_loader, device=device)

        payload = {
            "epoch": epoch,
            "train_loss": tr.loss,
            "train_acc": tr.acc,
            "train_seconds": tr.seconds,
            "val_loss": va.loss,
            "val_acc": va.acc,
            "val_seconds": va.seconds,
        }
        metrics.log(payload)
        writer.add_scalar("loss/train", tr.loss, epoch)
        writer.add_scalar("loss/val", va.loss, epoch)
        writer.add_scalar("acc/train", tr.acc, epoch)
        writer.add_scalar("acc/val", va.acc, epoch)

        logger.info("epoch", extra=payload)

        if bool(cfg.train.checkpoint.save_last):
            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch)

        if va.loss < best_val:
            best_val = va.loss
            save_checkpoint(
                ckpt_dir / "best.pt", model, optimizer, epoch, extra={"best_val_loss": best_val}
            )

    writer.close()
    return run_dir
