#!/usr/bin/env python3

import sys
import traceback

import torch

from core.distributed import init_distributed


def main():
    """
    Top-level training entrypoint.

    Responsibilities:
    - initialize distributed runtime
    - establish process identity
    - act as lifecycle root (everything hangs off this)
    """

    # ------------------------------------------------------------------
    # Step 1: Distributed initialization (BOOTSTRAP PHASE)
    # ------------------------------------------------------------------
    ctx = init_distributed()

    # ------------------------------------------------------------------
    # Sanity logging (temporary; later replaced by logging module)
    # ------------------------------------------------------------------
    if ctx.is_master():
        print("=" * 80)
        print("Training job started")
        print(f"World size      : {ctx.world_size}")
        print(f"Num nodes       : {ctx.num_nodes}")
        print(f"GPUs per node   : {ctx.gpus_per_node}")
        print(f"Backend         : {ctx.backend}")
        print("=" * 80)

    print(
        f"[rank={ctx.global_rank} | local_rank={ctx.local_rank}] "
        f"CUDA device = {torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}"
    )

    # ------------------------------------------------------------------
    # Placeholder for Step 2+
    # ------------------------------------------------------------------
    # build_pipeline(ctx)
    # run_training_loop(ctx)
    # handle_shutdown(ctx)

    if ctx.is_master():
        print("Initialization complete. Ready for next steps.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail fast, fail loudly (production-style)
        print("FATAL ERROR during training startup", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)