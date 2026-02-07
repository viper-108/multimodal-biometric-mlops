# Azure adaptation (high-level)

- Data: Azure Blob Storage
- Training: Azure ML job using container + mounted blob
- Images: Azure Container Registry
- Artifacts: versioned checkpoints + configs in blob

Production improvements:
- shard `.npy` outputs (WebDataset / tar shards)
- add registry step + model promotion workflow
