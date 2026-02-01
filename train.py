from core.distributed import init_distributed
from core.config import load_config
from core.paths import build_run_paths
from core.logging import setup_logger
from core.run_context import RunContext
import uuid
import os

def main():
    dist = init_distributed()

    cfg = load_config("config.yaml")

    run_id = f"{cfg['name']}-{uuid.uuid4().hex[:8]}"
    paths = build_run_paths(cfg["output_dir"], run_id)

    if dist.is_master():
        for p in paths.values():
            os.makedirs(p, exist_ok=True)

    logger = setup_logger(
        name="train",
        log_file=str(paths["logs"] / f"rank{dist.global_rank}.log"),
        is_master=dist.is_master(),
    )

    ctx = RunContext(
        distributed=dist,
        config=cfg,
        run_id=run_id,
        paths=paths,
        logger=logger,
    )

    logger.info("RunContext initialized")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail fast, fail loudly (production-style)
        print("FATAL ERROR during training startup", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)