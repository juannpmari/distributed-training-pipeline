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

**Design principle**
> `train.py` is intentionally thin. It should read like a story, not an implementation.

---

## Core Layer (Hard Edges)

The `core/` package defines **global invariants**.  
Nothing here depends on higher-level modules.

---

### `core/config.py`

**Role**
- Load and validate YAML configurations
- Resolve defaults and derived values
- Freeze configuration for reproducibility

**Inputs**
- YAML config files
- Optional CLI overrides

**Outputs**
- Immutable configuration object

**Used by**
- All other modules

---

### `core/run_context.py`

**Role**
- Define *what a training run is*
- Generate semantic + hash-based run IDs
- Create run directory structure

**Inputs**
- Configuration
- Git metadata (commit hash, dirty state)
- Environment metadata

**Outputs**
- Run ID
- Paths for:
  - logs
  - checkpoints
  - artifacts
  - metrics

---

### `core/distributed_context.py`

**Role**
- Single source of truth for distributed state
- Abstracts `torch.distributed` details

**Inputs**
- Configuration
- Environment variables

**Outputs**
- Rank / world size
- Process group handles
- Utilities (`is_main_rank`, `barrier`, `all_reduce`, etc.)

---

### `core/logging.py`

**Role**
- Unified logging interface
- Dual-channel logging:
  - Human-readable logs
  - Structured metrics/events

**Inputs**
- RunContext
- DistributedContext
- Log events

**Outputs**
- Log files
- Structured metric records

---

### `core/exceptions.py`

**Role**
- Define semantic error types
- Attach contextual diagnostics to failures

**Outputs**
- Structured, debuggable exceptions

---

## Distributed Layer (Systems)

Owns **how processes communicate and synchronize**.

---

### `distributed/init.py`

**Role**
- Initialize `torch.distributed`
- Validate environment consistency

**Outputs**
- Initialized process group

---

### `distributed/process_groups.py`

**Role**
- Define logical communication groups:
  - Data parallel groups
  - FSDP groups
  - (Future) tensor parallel groups

**Outputs**
- Named process groups

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
