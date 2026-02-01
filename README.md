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

```

Config → RunContext → DistributedContext
↓
Data Pipeline → Training Loop → Checkpointing
↓                ↓
Metrics & Logs    Runtime Monitors

```

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

---

### `core/config.py`
**Role:** Load and freeze experiment configuration.  
**Inputs:** Path to a YAML config file.  
**Outputs:** An immutable configuration object (read-only mapping).  
**Interactions:**  
- Consumed by `train.py` and `RunContext`
- Used indirectly by all pipeline components via `RunContext`

---

### `core/paths.py`
**Role:** Define the canonical filesystem layout for a run.  
**Inputs:** Base output directory, run ID.  
**Outputs:** A dictionary of resolved paths (`root`, `logs`, `checkpoints`, `artifacts`).  
**Interactions:**  
- Used during run bootstrap
- Paths are stored in `RunContext` and reused by logging, checkpointing, and artifacts

---

### `core/logging.py`
**Role:** Configure rank-aware logging.  
**Inputs:** Logger name, optional log file path, `is_master` flag.  
**Outputs:** A configured Python `logging.Logger`.  
**Interactions:**  
- Initialized during run bootstrap
- Logger is stored in `RunContext`
- Used by all downstream components for structured logging

---

### `core/run_context.py`
**Role:** Represent the execution context of a single run.  
**Inputs:**  
- `DistributedContext`  
- Frozen config  
- Run ID  
- Paths  
- Logger  
**Outputs:** A `RunContext` object passed through the pipeline.  
**Interactions:**  
- Depends on `core.distributed`
- Acts as the glue between distributed state, config, logging, and filesystem
- Passed explicitly to pipelines, trainers, and utilities

---

## Distributed runtime (`/core/distributed`)

The `distributed` submodule encapsulates **all distributed execution concerns**.  
No other part of the codebase reads environment variables or touches `torch.distributed` directly.

---

### `core/distributed/env.py`
**Role:** Discover and normalize distributed environment variables.  
**Inputs:** Process environment (e.g. `RANK`, `WORLD_SIZE`).  
**Outputs:** A normalized dictionary describing the distributed setup.  
**Interactions:**  
- Used only by `init.py`
- Abstracts over different launch styles (single-GPU, `torchrun`)

---

### `core/distributed/utils.py`
**Role:** Centralize small distributed policies.  
**Inputs:** Runtime state (e.g. CUDA availability).  
**Outputs:** Backend choice (`nccl` / `gloo`), default timeouts.  
**Interactions:**  
- Used by process group initialization
- Keeps policy decisions isolated and changeable

---

### `core/distributed/process_group.py`
**Role:** Initialize PyTorch process groups.  
**Inputs:** Backend, world size, global rank.  
**Outputs:** A fully initialized `torch.distributed` runtime.  
**Interactions:**  
- Called only from `init.py`
- No other module initializes or touches process groups

---

### `core/distributed/context.py`
**Role:** Hold immutable facts about distributed execution.  
**Inputs:** Ranks, world size, topology assumptions, backend.  
**Outputs:** A frozen `DistributedContext` object.  
**Interactions:**  
- Created during distributed initialization
- Embedded inside `RunContext`
- Used throughout the system to reason about rank, master status, and topology

---

### `core/distributed/init.py`
**Role:** Bootstrap distributed execution (Step 1).  
**Inputs:** Environment variables, CUDA availability.  
**Outputs:** A fully constructed `DistributedContext`.  
**Interactions:**  
- Orchestrates `env.py`, `utils.py`, and `process_group.py`
- Called exactly once at the start of `train.py`
- Must run before any model, data, or pipeline code

---

### `distributed/ddp.py`

**Role**
- Apply DDP wrapping
- Encapsulate DDP-specific behavior

**Inputs**
- Model
- Process group

**Outputs**
- DDP-wrapped model

---

### `distributed/fsdp.py`

**Role**
- Apply FSDP / ZeRO-style sharding
- Configure sharding, offload, and checkpoint format

**Outputs**
- Sharded model
- FSDP state handles

---

### `distributed/topology.py`

**Role**
- Abstract hardware topology assumptions
- Currently flat, future extensible

---

## Data Layer (Pipeline-First)

This layer mirrors **production LLM training pipelines**.

---

### `data/sources/streaming_source.py`

**Role**
- Stateless raw data access
- Stream individual samples

**Outputs**
- Unbatched samples

---

### `data/pipeline/stages.py`

**Role**
- Define pipeline stages:
  - decode
  - tokenize
  - augment
  - pack

**Inputs**
- Samples

**Outputs**
- Transformed samples

---

### `data/pipeline/queues.py`

**Role**
- Thread/process-safe queues
- Buffer between stages

---

### `data/pipeline/backpressure.py`

**Role**
- Flow control
- Prevent unbounded memory growth

**Mechanism**
- Semaphore-based backpressure

---

### `data/pipeline/pipeline.py`

**Role**
- Orchestrate async minibatch pipeline
- Spawn workers and manage lifecycle

**Outputs**
- Minibatches for training

---

### `data/sharding.py`

**Role**
- Assign non-overlapping data shards per rank

---

### `data/state.py`

**Role**
- Track dataset progress and RNG state
- Enable replayable execution

---

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
