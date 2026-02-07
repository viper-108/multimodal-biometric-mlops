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

Option B — Kaggle API (requires `~/.kaggle/kaggle.json`):
```bash
python -m src.bioml.scripts.preprocess dataset.download_from_kaggle=true
```

> The pipeline is robust to minor folder layout differences. It scans for image files and builds a canonical manifest.

### 3) Build processed dataset (parallel)
```bash
python -m src.bioml.scripts.preprocess   dataset.raw_dir=data/raw/multimodal_biometrics   dataset.processed_dir=data/processed/multimodal_biometrics   preprocess.num_workers=8
```

Artifacts created:
- `data/processed/.../manifest.parquet` (PyArrow/Parquet canonical manifest)
- `data/processed/.../arrays/*.npy` (resized arrays per modality)
- `data/processed/.../stats.json` (timings + counts)

### 4) Benchmark dataloader throughput
```bash
python -m src.bioml.scripts.benchmark_dataloader   dataset.processed_dir=data/processed/multimodal_biometrics   dataloader.num_workers=8 dataloader.batch_size=64
```

### 5) Train (reproducible)
```bash
python -m src.bioml.scripts.train   dataset.processed_dir=data/processed/multimodal_biometrics   train.epochs=2 train.seed=42
```

Outputs:
- `runs/<run_id>/` (config snapshot, env info, metrics.jsonl, TensorBoard logs)
- `runs/<run_id>/checkpoints/last.pt`

### 6) Inference
```bash
python -m src.bioml.scripts.infer   infer.checkpoint_path=runs/<run_id>/checkpoints/last.pt   infer.sample_id=0   dataset.processed_dir=data/processed/multimodal_biometrics
```

---

## Architecture

- **HLD:** `docs/architecture/HLD.md`
- **LLD:** `docs/architecture/LLD.md`
- **Bottlenecks & trade-offs:** `docs/architecture/bottlenecks.md`
- **Design decisions (ADRs):** `docs/architecture/decisions.md`

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
.github/workflows/ci.yml
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
