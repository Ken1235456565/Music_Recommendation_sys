# HPC Music Recommendation System

A three-layer distributed music recommendation system built on HPC technologies, demonstrating the performance advantages of parallelism in large-scale recommendation scenarios. The system is deployed on a Slurm cluster and uses HDF5 as the unified data exchange format between layers.

---

## Architecture Overview

```
MSD Metadata / User-Song Interaction Matrix
              ↓
Layer 1: Feature Extraction     →  Dask distributed (embarrassingly parallel)
Layer 2: Collaborative Filtering →  scipy.sparse CSR/CSC + mpi4py ALS
              ↓  HDF5
Layer 3: Sequential Recommendation → PyTorch Transformer + DDP (multi-GPU)
              ↓
        Recommendation Results
```

**Job dependency structure (Slurm DAG):**

```
Job A (Layer 1: Feature Extraction) ──┐
                                       ├──→ Job C (Layer 3: Transformer Training)
Job B (Layer 2: ALS Training)    ──────┘
```

Layer 1 and Layer 2 are independent and run in parallel. Layer 3 depends on both completing successfully (`--dependency=afterok`).

---

## Data Source

- **Million Song Dataset (MSD):** http://millionsongdataset.com/
  - Full dataset: ~280 GB, native HDF5 format
  - ~1M users × 380K songs, interaction matrix density ~0.001%

---

## Layer 1 — Feature Extraction

### Goal
Extract content-based metadata features for cold-start songs (no interaction history), solving the new-song onboarding problem.

### Parallel Model
The feature extraction task is **embarrassingly parallel** — each song is fully independent with zero cross-task communication. **Process-level parallelism** is chosen over thread-level because CPU-bound workloads cannot achieve true parallelism under Python's GIL.

### Task Scheduling

```
[Master Process]
    ↓ Scan filesystem, build task queue (~1M file paths)
    ↓ Sort by file size (large files first, avoids long-tail straggler effect)
[Worker Pool]
    Each worker: read metadata → extract features → write to local temporary HDF5 shard
[Reducer]
    Merge all temporary shards → output unified features.h5
```

**Key design decision:** Workers never write to a shared HDF5 file simultaneously to avoid I/O contention. Each worker writes its own shard; a single reducer process merges them sequentially.

### Slurm Resource Strategy
Layer 1 requests **1 high-core-count CPU node (32–64 cores)**. Multi-node is deliberately avoided because:
- Tasks have zero inter-process communication — cross-node network latency provides no benefit
- Concurrent reads from a shared filesystem become the bottleneck; more nodes worsen I/O contention rather than alleviating it

### Technology
- **Dask distributed** (default process scheduler)
- Reference: https://examples.dask.org/applications/embarrassingly-parallel.html

---

## Layer 2 — Sparse Matrix Storage & Collaborative Filtering

### Goal
Train user and item latent vectors via ALS (Alternating Least Squares) on the extremely sparse MSD user-song interaction matrix.

### Why Sparse Representation
| Format | Memory (1M × 380K matrix) |
|--------|--------------------------|
| Dense  | ~3 TB                    |
| CSR    | ~3 GB                    |

Sparse representation is not an optimization — it is a prerequisite for the problem to be tractable at all.

### Sparse Format Selection

| Format | Row Slicing | Column Slicing | Random Access | MPI Communication |
|--------|-------------|----------------|---------------|-------------------|
| CSR    |  Native   |  Requires conversion |  Slow | High (row blocks map directly) |
| CSC    |  Requires conversion |  Native |  Slow | High (column blocks) |
| COO    |  Needs sort |  Needs sort |  Slow | Low (coordinate pairs, high redundancy) |

ALS iterates by fixing item vectors and scanning all songs rated by a given user — a **row-access pattern** that makes CSR the natural choice.

### Distributed ALS Communication Pattern

ALS produces **asymmetric communication requirements** across its two alternating steps:

**Step A — Fix item vectors, update user vectors:**
- Each MPI rank holds a row-partition of the user matrix
- Requires reading item vectors for all songs rated by its local users
- Item vectors are globally shared → **All-Gather or broadcast**

**Step B — Fix user vectors, update item vectors:**
- Access pattern reverses to column-access
- CSR performance degrades; matrix must be transposed to CSC
- This global transpose is equivalent to an **All-to-All** operation and is the primary communication bottleneck — it should be measured independently in benchmarking

### HPC Value of This Design
The format switch (CSR ↔ CSC) is not a data structure detail — it is a communication event. The cost of switching formats is itself part of the distributed overhead and warrants separate instrumentation in performance analysis.

### Technology
- **scipy.sparse** — CSR/CSC storage: https://docs.scipy.org/doc/scipy/reference/sparse.html
- **implicit** — ALS implementation: https://benfred.github.io/implicit/
- **mpi4py** — distributed coordination: https://mpi4py.readthedocs.io/en/stable/

> **Note:** The distributed ALS loop requires either modifying `implicit`'s internals or reimplementing the ALS update loop directly with `mpi4py`. These are significantly different implementation paths and the chosen approach should be documented explicitly.

---

## Layer 3 — Sequential Recommendation (Transformer + DDP)

### Core Idea
Rather than traditional collaborative filtering scoring, recommendation is framed as a **sequence generation problem**: given a user's listening history as a token sequence, predict the next song. This transforms recommendation into a language modeling task.

### DDP Communication Model
PyTorch DDP is **synchronous data parallelism**: each GPU holds a complete model copy, forward passes run independently, and gradients are synchronized via **all-reduce** during backpropagation.

**Key efficiency mechanism — gradient communication overlapped with computation:**

```
[GPU 0]  Forward → Backward (layer N grad) → all-reduce (layer N) → Backward (layer N-1) → ...
[GPU 1]  Forward → Backward (layer N grad) → all-reduce (layer N) → Backward (layer N-1) → ...
```

DDP pipelines gradient all-reduce with subsequent layer backward computation, partially hiding communication latency inside compute time. This is the fundamental reason DDP outperforms a naive parameter server approach.

### Batch Construction & Sequence Padding

User histories vary in length. Padding strategy directly affects GPU utilization:

| Strategy | Description | GPU Utilization |
|----------|-------------|-----------------|
| Static padding | Pad all sequences to global max length | Low — short sequences waste compute |
| Dynamic padding | Pad to the longest sequence within each batch | Better |
| **Bucket sampling** | Group sequences of similar length into the same batch | **Optimal for HPC** |

Under DDP, each GPU constructs its own batches independently. Bucket sampling must ensure similar sequence length distributions across GPUs, otherwise faster GPUs stall waiting for slower ones to complete backpropagation.

### Cold-Start Fusion

Layer 3 fuses outputs from both upstream layers to handle cold-start scenarios:

- **Song content features** (from Layer 1): loaded from `features.h5` by song ID
- **User latent vectors** (from Layer 2): loaded from ALS output HDF5 by user ID

**Dynamic weighting:** the shorter a user's interaction history, the higher the weight assigned to content features. This weighting must be computed on-GPU to avoid CPU–GPU transfer overhead.

**I/O bottleneck mitigation:** concurrent random HDF5 reads by multiple GPU processes create file lock contention. Solution: preload all feature vectors into **shared memory (`/dev/shm`)** before training begins. Each GPU process reads directly from shared memory, eliminating repeated disk I/O.

> **Pre-deployment check:** Verify that the combined size of user latent vectors + song content features fits within the available `/dev/shm` capacity on the target cluster nodes.

### Technology
- **PyTorch DDP:** https://docs.pytorch.org/tutorials/beginner/dist_overview.html
- Requires CUDA

---

## Data Flow

```
Layer 1 output:  features.h5        (song_id → content feature vector)
Layer 2 output:  als_vectors.h5     (user_id → latent vector, song_id → latent vector)
                        ↓
               Preloaded into /dev/shm
                        ↓
Layer 3 input:   fused embeddings per (user, song) pair
```

All inter-layer data exchange uses **HDF5** as the unified format, enabling random access by ID without loading entire files into memory.

---

## Benchmarking

The primary focus is **weak scaling**, as the realistic operational concern is: *as the user base and song catalog grow, can the system keep up?*

### Scaling Definitions

- **Strong scaling:** Fixed problem size, increasing resources. Ideal: N processes → N× speedup. In practice, communication overhead and synchronization barriers cause the curve to bend downward. The inflection point identifies the optimal configuration.
- **Weak scaling:** Fixed work per process, increasing total resources. If total time grows as nodes are added, cross-node communication overhead is scaling with problem size.

### Per-Layer Performance Expectations

| Layer | Bottleneck Type | Expected Strong Scaling Efficiency |
|-------|----------------|-------------------------------------|
| Layer 1 | I/O bound (metadata reads) + Memory bound (feature loading) | 70–80% |
| Layer 2 | Communication bound (global ALS synchronization, CSR↔CSC transpose) | 60–75% |
| Layer 3 | Compute bound (matrix multiply dominates) | 80–90% (communication hidden by compute) |

### Benchmarking Procedure

1. Define problem size increments for weak scaling (e.g., fix users-per-rank; scale total users proportionally with rank count)
2. Instrument each layer independently — measure compute time, communication time, and I/O time separately
3. For Layer 2: measure the CSR→CSC transpose step as an isolated event
4. Record actual throughput vs. theoretical predictions
5. Analyze deviation sources: load imbalance, synchronization overhead, filesystem contention

---

## Deployment

The system is deployed on a **Slurm cluster**. Jobs are submitted as a DAG using `--dependency=afterok`, allowing Slurm to manage execution order automatically without manual intervention between stages.

```bash
jid1=$(sbatch --parsable layer1_feature_extraction.sh)
jid2=$(sbatch --parsable layer2_als_training.sh)
jid3=$(sbatch --parsable --dependency=afterok:$jid1:$jid2 layer3_transformer_training.sh)
```

---

## Repository Structure (Proposed)

```
.
├── layer1/
│   ├── extract_features.py       # Dask feature extraction pipeline
│   └── merge_shards.py           # HDF5 shard reducer
├── layer2/
│   ├── build_sparse_matrix.py    # CSR matrix construction from MSD
│   └── distributed_als.py        # mpi4py ALS training loop
├── layer3/
│   ├── dataset.py                # Sequence dataset + bucket sampler
│   ├── model.py                  # Transformer recommendation model
│   └── train_ddp.py              # DDP training entry point
├── slurm/
│   ├── layer1.sh
│   ├── layer2.sh
│   └── layer3.sh
├── utils/
│   └── hdf5_io.py                # Shared HDF5 read/write utilities
└── README.md
```
