import numpy as np
import pyarrow as pa
from omegaconf import OmegaConf
from src.bioml.data.preprocessing import save_preprocess_artifacts
from src.bioml.data.splits import SplitConfig, add_split_column
from src.bioml.training.train import train


def test_smoke_train_cpu(tmp_path):
    processed_dir = tmp_path / "processed"
    arrays = processed_dir / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)

    h, w = 32, 32
    rows = []
    for i in range(6):
        arr = (np.random.rand(h, w) * 255).astype("uint8")
        fp = arrays / f"{i}_fp.npy"
        il = arrays / f"{i}_il.npy"
        ir = arrays / f"{i}_ir.npy"
        np.save(fp, arr)
        np.save(il, arr)
        np.save(ir, arr)
        rows.append(
            {
                "person_id": i % 3,
                "session_id": 0,
                "fingerprint_npy": str(fp),
                "iris_left_npy": str(il),
                "iris_right_npy": str(ir),
                "sample_id": i,
            }
        )

    table = pa.Table.from_pylist(rows)
    table = add_split_column(table, SplitConfig(train=0.7, val=0.3, test=0.0, seed=0))
    save_preprocess_artifacts(processed_dir, "manifest.parquet", table, {"ok": True})

    cfg = OmegaConf.create(
        {
            "run": {"output_dir": str(tmp_path / "runs"), "experiment_name": "smoke"},
            "dataset": {"processed_dir": str(processed_dir)},
            "preprocess": {
                "manifest_name": "manifest.parquet",
                "arrays_subdir": "arrays",
                "target_size": [h, w],
            },
            "dataloader": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": 2,
            },
            "model": {
                "name": "fusion_cnn",
                "embedding_dim": 16,
                "dropout": 0.1,
                "backbone": {"in_channels": 1, "channels": [8, 16], "kernel_size": 3},
                "num_classes": 3,
            },
            "train": {
                "device": "cpu",
                "seed": 1,
                "deterministic": True,
                "epochs": 1,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "grad_clip_norm": 1.0,
                "amp": False,
                "checkpoint": {"save_dirname": "checkpoints", "save_last": True},
            },
        }
    )

    run_dir = train(cfg)
    assert (run_dir / "checkpoints" / "last.pt").exists()
