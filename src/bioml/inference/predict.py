from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import torch
from src.bioml.data.dataset import MultiModalBiometricDataset
from src.bioml.data.manifest import load_manifest, resolve_manifest
from src.bioml.models.registry import build_model
from src.bioml.training.checkpointing import load_checkpoint


def _load_label_map(path: str | None) -> dict[int, int] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"label_map_path not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    # stored as {"class_id_to_person_id": {"0": 1, "1": 2, ...}}
    raw = obj.get("class_id_to_person_id", obj)
    return {int(k): int(v) for k, v in raw.items()}


@torch.inference_mode()
def predict_one(cfg: Any) -> dict[str, Any]:
    mp = resolve_manifest(
        Path(cfg.dataset.processed_dir),
        cfg.preprocess.manifest_name,
        cfg.preprocess.arrays_subdir,
    )
    table = load_manifest(mp.manifest_path)

    # num_classes must match training
    unique_person_ids = pc.unique(table["person_id"]).to_pylist()
    cfg.model.num_classes = len(unique_person_ids)

    ds = MultiModalBiometricDataset(table, target_size=tuple(cfg.preprocess.target_size))
    sample = ds[int(cfg.infer.sample_id)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    load_checkpoint(Path(cfg.infer.checkpoint_path), model)
    model.eval()

    x_fp = sample["fingerprint"].unsqueeze(0).to(device)
    x_il = sample["iris_left"].unsqueeze(0).to(device)
    x_ir = sample["iris_right"].unsqueeze(0).to(device)

    logits = model(x_fp, x_il, x_ir).cpu().squeeze(0)
    probs = torch.softmax(logits, dim=0)

    topk = int(cfg.infer.topk)
    v, i = torch.topk(probs, k=min(topk, probs.numel()))

    # Load mapping produced by training (optional)
    label_map = _load_label_map(getattr(cfg.infer, "label_map_path", ""))

    true_class_id = int(sample["label"].item())
    true_person_id = int(sample.get("person_id", sample["label"]).item())

    preds = []
    for rank, (cid, prob) in enumerate(zip(i.tolist(), v.tolist(), strict=True), start=1):
        cid = int(cid)
        preds.append(
            {
                "rank": rank,
                "class_id": cid,
                "person_id": int(label_map[cid]) if label_map is not None else None,
                "prob": float(prob),
            }
        )

    out = {
        "sample_id": int(sample["sample_id"].item()),
        "true": {
            "class_id": true_class_id,
            "person_id": int(label_map[true_class_id]) if label_map is not None else true_person_id,
        },
        "predictions": preds,
        "meta": {
            "num_classes": int(cfg.model.num_classes),
            "checkpoint_path": str(cfg.infer.checkpoint_path),
            "label_map_path": str(getattr(cfg.infer, "label_map_path", "")),
        },
    }
    return out
