# Local runbook

```bash
python -m src.bioml.scripts.preprocess preprocess.num_workers=8
python -m src.bioml.scripts.benchmark_dataloader dataloader.num_workers=8 dataloader.batch_size=64
python -m src.bioml.scripts.train train.epochs=5 train.seed=42
python -m src.bioml.scripts.infer infer.checkpoint_path=runs/<id>/checkpoints/best.pt infer.sample_id=0
```
