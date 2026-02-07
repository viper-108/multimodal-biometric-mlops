# Design decisions (mini ADRs)

## ADR-001: Parquet manifest + npy arrays
Metadata in Parquet for fast scans and stable schema. Arrays stored as `.npy` for mmap reads.

## ADR-002: Multiprocessing-first preprocessing
Default is ProcessPoolExecutor; Ray is optional for distributed runs.

## ADR-003: Pure PyTorch loops
Demonstrates fundamentals (AMP, grad clipping, checkpointing) without a high-level framework.
