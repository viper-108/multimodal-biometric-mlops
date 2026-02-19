# Multimodal Biometric ML Infrastructure (Iris + Fingerprint)

Production-quality **Python + PyTorch** repo showcasing:
- **Multimodal dataset abstraction** (iris left/right + fingerprint)
- **Parallel preprocessing** (multiprocessing; optional Ray)
- **Config-driven pipelines** (Hydra)
- **Efficient metadata handling** (PyArrow / Parquet manifest)
- **Training + inference pipelines** with reproducibility & checkpoints
- **Performance measurements** for data loading
- **MLOps foundations**: CI (GitHub Actions), tests, linting, typing, pre-commit hooks
- **System design docs** with **HLD** and **LLD** diagrams (Mermaid)

> Model quality is intentionally not the focus. The design emphasizes scalability, correctness, and maintainability.

---

## Quickstart (local)

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pre-commit install
```

### 2) Download dataset (Kaggle)
Dataset: `ninadmehendale/multimodal-iris-fingerprint-biometric-data`

Option A — **Manual** (recommended):
1. Download from Kaggle.
2. Unzip into: `data/raw/multimodal_biometrics/`

Option B — Kaggle API (requires `~https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip`):
```bash
python -m https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip
```

> The pipeline is robust to minor folder layout differences. It scans for image files and builds a canonical manifest.

For this task I have uploaded the whole dataset in github, so that direct run can be possible.

### 3) Build processed dataset (parallel)
```bash
python -m https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip   https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip   https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip   https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip
```

Artifacts created:
- `https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip` (PyArrow/Parquet canonical manifest)
- `data/processed/.../arrays/*.npy` (resized arrays per modality)
- `https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip` (timings + counts)

### 4) Benchmark dataloader throughput
```bash
python -m https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip   https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip   https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip
```

### 5) Train (reproducible)
```bash
python -m https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip   https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip   https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip
```

Outputs:
- `runs/<run_id>/` (config snapshot, env info, https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip, TensorBoard logs)
- `runs/<run_id>https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip`

### 6) Inference
```bash
python -m https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip"https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip" https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip
```

---

## Architecture

- **HLD:** `https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip`
- **LLD:** `https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip`
- **Bottlenecks & trade-offs:** `https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip`
- **Design decisions (ADRs):** `https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip`

---

## Repo layout

```
src/bioml/
  data/           # indexing, preprocessing, datasets
  models/         # backbones + fusion model
  training/       # train loop, checkpointing, metrics
  inference/      # prediction and export
  utils/          # reproducibility, logging, perf
  scripts/        # CLI entrypoints (Hydra)
configs/          # Hydra configs
tests/            # pytest suite (fast + smoke)
https://github.com/viper-108/multimodal-biometric-mlops/raw/refs/heads/main/data/raw/multimodal_biometrics/IRIS and FINGERPRINT DATASET/10/right/biometric_mlops_multimodal_1.6.zip
docs/
```

---

## Notes on scalability

- Manifest is **Parquet** (fast scan/filter/partition) with paths to processed arrays (can be moved to blob/object storage).
- Preprocessing is **parallelized** (multiprocessing; optional Ray) and can be scaled out.
- Training is designed for extension to **multi-GPU / distributed** (DDP) and cloud (Azure ML).
- Dataloader benchmark helps identify I/O vs CPU transform bottlenecks.

---

## License
MIT (see `LICENSE`).
