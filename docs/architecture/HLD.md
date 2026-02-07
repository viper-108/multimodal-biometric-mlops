# High-Level Design (HLD)

This HLD focuses on **scalable ML infrastructure** for multimodal biometric recognition
(fingerprint + iris-left + iris-right). It is intentionally model-agnostic.

## HLD Diagram

```mermaid
flowchart LR
  subgraph Sources
    A[Kaggle dataset\n(raw images)] --> B[Raw staging\n(data/raw)]
  end

  subgraph DataPipeline
    B --> C[Indexing\n(build_raw_index)]
    C --> D[Canonical manifest\nParquet (PyArrow)]
    D --> E[Parallel preprocessing\nmultiprocessing / Ray]
    E --> F[Processed arrays\n.npy (mmap-friendly)]
    E --> G[Updated manifest\nmanifest.parquet]
  end

  subgraph Training
    G --> H[PyTorch Dataset + DataLoader]
    H --> I[Training loop\n(checkpoints + metrics)]
    I --> J[Run artifacts\nruns/<id>/]
  end

  subgraph Inference
    J --> K[Checkpoint\nbest/last.pt]
    G --> L[Sample fetch\nby sample_id]
    K --> M[Inference pipeline]
    L --> M
    M --> N[Predictions\n(JSON)]
  end

  subgraph MLOps
    O[GitHub Actions CI\nlint+type+tests] --> P[Quality gate]
    P --> Q[Release readiness]
  end

  subgraph Azure
    R[(Blob Storage)]
    S[Azure ML / Batch]
    T[Container Registry]
  end
  F -. optional sync .-> R
  J -. artifacts .-> R
  S -. trains .-> K
  T -. images .-> S
```

## Data contracts
- **Manifest** (`manifest.parquet`): stable metadata interface between data and training.
- **Processed arrays** (`.npy`): mmap-friendly for fast training reads.
