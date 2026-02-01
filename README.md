# LLM Distributed Training Pipeline — Architecture & Module Overview

This repository implements a **production-inspired, research-grade distributed training pipeline** for large language models.  
Its design mirrors how **real labs organize training infrastructure**, while remaining readable, explicit, and extensible.

This document explains the repository **from a module and pipeline perspective**:
- What each module represents as a **component**
- Its **role in the training pipeline**
- Its **inputs and outputs**
- How modules interact

The goal is that a **research engineer can understand the system end-to-end by reading this file**.

---

## High-Level Pipeline Overview

At a conceptual level, the system is a **deterministic, replayable training pipeline**:

Config → RunContext → DistributedContext
↓
Data Pipeline → Training Loop → Checkpointing
↓ ↓
Metrics & Logs Runtime Monitors

This project follows a Pipes-and-Filters architecture as its primary design pattern, adapted for large-scale, distributed ML training. It is supported by the following complementary patterns:

1. **Pipeline pattern** — data flows through a well-defined, ordered sequence of processing stages (streaming source → sharding → batching → training), each stage transforming the data incrementally.
2. **Iterator / Pull-based streaming pattern** — all data components are lazy and iterable, enabling streaming over datasets larger than memory and precise control over consumption.
3. **Explicit State & Checkpoint pattern** — all long-lived progress (dataset offsets, run metadata, RNG state) is modeled as serializable state, allowing deterministic resumption after failure.
4. **Context Object pattern** — global concerns (run identity, filesystem layout, distributed rank/world info) are encapsulated in explicit context objects rather than implicit globals.

Each layer is explicit:
- **Configuration & context** define *what* is being run
- **Distributed & data layers** define *how* it runs
- **Training loop** defines *what computation happens*
- **Runtime & experiments** ensure observability and reproducibility

---

## Execution flow (high level)

1. `train.py` calls `init_distributed()` → creates `DistributedContext`
2. Config, paths, and logging are initialized
3. `RunContext` is constructed from these components
4. The rest of the training system operates solely on `RunContext`

This separation ensures clarity, reproducibility, and correctness across single-GPU, multi-GPU, and multi-node runs.

## Entrypoint

### `train.py`

**Role**
- The **orchestrator** of the entire training run
- Wires together all major components
- Defines the lifecycle: init → train → checkpoint → shutdown

**Inputs**
- CLI arguments
- YAML configuration files
- Environment variables (`RANK`, `WORLD_SIZE`, etc.)

**Outputs**
- A fully initialized training run
- Exit status with rich diagnostics on failure


## Core runtime (`/core`)

The `/core` directory contains the **foundational runtime primitives** of the training system.  
Everything here is **model-agnostic**, **algorithm-agnostic**, and designed to be reused across all pipelines.  
Higher-level components (pipelines, training loops, models) depend on `/core`, never the other way around.

This repository enforces **strong reproducibility guarantees** inspired by large-scale training systems used in research labs.

### Configuration system
All training runs are driven by a **single resolved YAML configuration**:
- Base configs are loaded from YAML
- Overrides are applied via deep merge
- The final configuration is fully explicit and immutable
- Every run saves its resolved config to disk

This ensures runs are **comparable, debuggable, and replayable**.

Relevant files:
- `core/config.py`  
  Loads YAML configs, applies overrides, validates required fields, and produces an immutable `FrozenConfig`.

### Deterministic seeding
Randomness is controlled centrally and deterministically:
- Python, NumPy, and PyTorch (CPU + CUDA) are seeded
- Seeds are rank-aware (`base_seed + global_rank`)
- cuDNN deterministic flags are enforced

Relevant files:
- `core/seeding.py`  
  Implements rank-aware, deterministic seeding for distributed runs.

### Run context & reproducibility
Each run has a unique, deterministic identity:
- Run ID includes timestamp + hash of resolved config
- All artifacts (logs, checkpoints, configs) live under a single run directory
- The resolved config is persisted verbatim

Relevant files:
- `core/run_context.py`  
  Creates the run directory structure and binds run identity to configuration state.

---

## Distributed runtime (`/core/distributed`)

The `distributed` submodule encapsulates **all distributed execution concerns**.  
No other part of the codebase reads environment variables or touches `torch.distributed` directly.

This submodule is responsible for **process coordination, rank semantics, and collective correctness**. All higher-level training logic depends on this layer behaving deterministically and fail-fast.

### Process launch & rank semantics
The runtime supports multiple launch styles (e.g. `torchrun`, single-GPU execution) while presenting a **uniform rank model** to the rest of the system.

**Invariants:**
- Each process has a globally unique `(global_rank, world_size)`
- Each process has a `(local_rank, local_world_size)` within its node
- Rank information is derived exclusively from the launcher environment
- Training code never reads launcher-specific environment variables directly

Relevant files:
- `core/distributed/env.py`  
  Discovers rank, world size, and node topology from the environment.
- `core/distributed/context.py`  
  Defines the `DistributedContext` object used throughout the system.

---

### Process group initialization
All collective communication is mediated through explicitly initialized process groups.

**Invariants:**
- Process groups are initialized exactly once
- NCCL is used for GPU collectives; Gloo is used as a fallback where required
- The backend choice is explicit and centrally controlled
- Training code never calls `torch.distributed.init_process_group` directly

Relevant files:
- `core/distributed/init.py`  
  Initializes and tears down distributed process groups.
- `core/distributed/context.py`  
  Stores initialized group handles and backend metadata.

---

### Topology abstraction
The system assumes a **logically flat topology** while retaining enough structure to support node-local and global distinctions.

**Invariants:**
- Intra-node and inter-node communication are abstracted behind the same API
- Local rank information is preserved for device placement and sharding
- No training logic encodes assumptions about physical interconnects

Relevant files:
- `core/distributed/topology.py`  
  Encodes node-local vs global rank relationships.
- `core/distributed/context.py`  
  Exposes topology metadata to downstream systems.

---

### Failure behavior & diagnostics
The runtime is designed to **fail fast** and surface actionable diagnostics rather than masking distributed errors.

**Invariants:**
- Initialization failures abort the job immediately
- Rank and topology metadata are always logged on startup
- Partial initialization is not allowed

Relevant files:
- `core/distributed/init.py`
- `core/distributed/utils.py`

---

### Integration contract
All distributed-aware components receive a `DistributedContext` instance.  
No component may:
- Inspect environment variables directly
- Initialize process groups
- Infer rank semantics implicitly

This enforces a **single distributed authority** across the codebase.

Primary interface:
- `DistributedContext`

## Data Pipeline

This repository implements a **streaming, rank-aware data pipeline** designed for large-scale language model training.  
The data layer is treated as a **stateful distributed system** with explicit correctness and resume guarantees.

### Streaming-first design
All datasets are consumed as **streams**, not as finite indexed collections.

**Invariants:**
- Data is never fully materialized in memory
- Iteration proceeds monotonically through a global sample stream
- Epoch-based semantics are not used
- Dataset size may exceed available memory or local disk

Relevant files:
- `core/data/sources/streaming_source.py`  
  Defines the streaming source interface.
- `core/data/dataset.py`  
  Adapts streaming sources into PyTorch `IterableDataset`s.

---

### Rank-aware deterministic sharding
Each process consumes a **deterministic shard** of the global data stream.

**Invariants:**
- Sample ownership is determined solely by `(global_rank, world_size)`
- Sharding is stable and reproducible
- No coordination or communication is required between ranks
- Data assignment is independent of worker count or launch style

Relevant files:
- `core/data/sharding.py`  
  Implements deterministic modulo-based sharding.

---

### Dataset state & resumption
Dataset progress is explicitly tracked and checkpointed.

**Invariants:**
- Dataset state is monotonic and append-only
- Training can resume from an exact global offset
- Dataset state is owned by the data layer, not the training loop
- Partial progress is never inferred implicitly

Relevant files:
- `core/data/state.py`  
  Defines the serializable dataset cursor.
- `core/data/factory.py`  
  Wires dataset state into dataset construction.

---

### World-size changes & replayability
The data pipeline supports **best-effort replay** when world size changes.

**Contract:**
- Resuming from a checkpoint with a different world size is allowed
- Sample-to-rank assignment may change
- Global ordering and determinism are preserved

This mirrors the behavior of production-scale LLM training systems.

---

### Pipeline abstraction (forward-compatible)
The data layer exposes a **pipeline abstraction** for future extension.

**Planned capabilities:**
- Asynchronous prefetch
- Multi-stage decoding and preprocessing
- Bounded queues with backpressure
- Overlap of data loading and compute

Relevant files:
- `core/data/pipeline/pipeline.py`
- `core/data/pipeline/stages.py`
- `core/data/pipeline/queues.py`
- `core/data/pipeline/backpressure.py`

These components are intentionally minimal in the current implementation and will be activated incrementally.

---

### Integration contract
The training loop consumes data exclusively through a PyTorch `DataLoader`.  
All data semantics—streaming, sharding, resumption—are fully encapsulated within the data layer.

Training code must not:
- Index datasets
- Implement sharding logic
- Track dataset offsets
- Assume epoch boundaries


## Models Layer (Algorithms)

Pure **model definitions**, minimal systems logic.

---

### `models/qwen/model.py`
- Core transformer architecture

### `models/qwen/attention.py`
- Attention implementations
- FlashAttention integration point

### `models/qwen/multimodal.py`
- Vision / multimodal adapters

---

## Training Layer (Algorithms + Systems Interface)

---

### `training/loop.py`

**Role**
- Own the training lifecycle
- Step scheduling, evaluation hooks

---

### `training/step.py`

**Role**
- Single training step:
  forward → loss → backward

---

### `training/precision.py`

**Role**
- Mixed precision (AMP) handling
- Gradient scaling

---

### `training/checkpointing.py`

**Role**
- Save model, optimizer, and pipeline state

---

### `training/resume.py`

**Role**
- Restore full training state
- Validate compatibility

---

## Runtime Monitoring

Ensures **training health and debuggability**.

---

### `runtime/profiler.py`
- Step-level profiling
- Compute vs communication timing

### `runtime/memory.py`
- GPU/CPU memory monitoring

### `runtime/health.py`
- Liveness checks
- Stall detection

### `runtime/watchdog.py`
- Fail-fast termination with diagnostics

---

## Experiment Tracking

Minimal, explicit experiment management.

---

### `experiments/schema.py`
- Metric and event schemas

### `experiments/tracking.py`
- Collect and aggregate metrics across ranks

### `experiments/artifacts.py`
- Manage saved artifacts (configs, checkpoints, logs)

---

## How to Read This Repo (RE Perspective)

A research engineer typically:
1. Opens `train.py`
2. Inspects `DistributedContext`
3. Reviews the data pipeline
4. Examines the training loop
5. Verifies checkpointing & resume logic
6. Reads `docs/architecture.md`

Everything is designed to **line up conceptually**.

---

## Design Philosophy Summary

- **Hybrid scope**: educational clarity + production realism
- **Hard edges, soft internals**
- **Explicit over magical**
- **Replayable execution**
- **Fail-fast with rich diagnostics**
- **Flat topology abstraction**

---

## Intended Use

- Single-GPU laptop (development)
- Multi-GPU single node
- Multi-node cloud clusters

The same codepath applies to all.

---

**This repository is meant to demonstrate deep understanding of modern LLM training systems, not just model code.**
```
