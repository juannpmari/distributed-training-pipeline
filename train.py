# train.py
import argparse
import logging

from core.config import load_config
from core.run_context import create_run_context
from core.distributed_context import init_distributed
from core.logging import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-name", default="experiment")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Init distributed
    dist_ctx = init_distributed()

    # Create run context (main rank only)
    if dist_ctx.is_main_rank():
        run_ctx = create_run_context(
            base_dir=args.runs_dir,
            experiment_name=args.experiment_name,
            config=config.raw,
        )
    else:
        run_ctx = None

    # Barrier so dirs exist
    dist_ctx.barrier()

    # Setup logging
    if run_ctx is not None:
        log_dir = run_ctx.logs_dir
    else:
        # non-main ranks infer run dir later; temporary fallback
        log_dir = None

    setup_logging(log_dir or ".", dist_ctx.rank)

    logging.info("Run started")
    logging.info(f"Distributed: {dist_ctx.is_distributed}")
    logging.info(f"Rank: {dist_ctx.rank} / {dist_ctx.world_size}")

    logging.info("Step 0 skeleton initialized successfully")


if __name__ == "__main__":
    main()
