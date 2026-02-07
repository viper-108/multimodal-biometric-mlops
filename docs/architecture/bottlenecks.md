# Bottlenecks, scalability, trade-offs

## Likely bottlenecks
1. Image decoding + resizing (preprocessing)
2. Dataloader CPU transforms
3. Disk / object-store I/O throughput
4. Determinism vs throughput

## Scalability knobs
- preprocess.num_workers
- dataloader.num_workers / prefetch_factor / pin_memory
- sharding strategy for arrays (for object storage)

## Cloud notes (Azure)
- Blob Storage for manifest + arrays (or shards)
- Azure ML for training jobs
- ACR for container images
