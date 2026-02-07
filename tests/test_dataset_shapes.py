import numpy as np
import pyarrow as pa
import torch
from src.bioml.data.dataset import MultiModalBiometricDataset


def test_dataset_shapes(tmp_path):
    H, W = 64, 64
    arr = (np.random.rand(H, W) * 255).astype("uint8")
    fp = tmp_path / "0_fp.npy"
    il = tmp_path / "0_il.npy"
    ir = tmp_path / "0_ir.npy"
    np.save(fp, arr)
    np.save(il, arr)
    np.save(ir, arr)

    table = pa.Table.from_pylist(
        [
            {
                "sample_id": 0,
                "person_id": 3,
                "fingerprint_npy": str(fp),
                "iris_left_npy": str(il),
                "iris_right_npy": str(ir),
            }
        ]
    )

    ds = MultiModalBiometricDataset(table, target_size=(H, W))
    item = ds[0]
    assert item["fingerprint"].shape == (1, H, W)
    assert item["iris_left"].shape == (1, H, W)
    assert item["iris_right"].shape == (1, H, W)
    assert item["label"].dtype == torch.long
