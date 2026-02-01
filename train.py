# train.py
from pathlib import Path

from core.config import resolve_config
from core.seeding import seed_everything
from core.run_context import create_run_context
from core.distributed.env import get_distributed_context

def main():
    dist = get_distributed_context()

    cfg = resolve_config(
        base_config_path=Path("configs/train.yaml"),
        overrides={},  # CLI/env overrides later
    )

    run_ctx = create_run_context(
        base_dir=Path("runs"),
        experiment_name="llm_pretrain",
        resolved_config=cfg.to_dict(),
    )

    cfg.save_yaml(run_ctx.root_dir / "resolved_config.yaml")

    effective_seed = seed_everything(
        base_seed=cfg.data["seed"],
        rank=dist.rank,
        deterministic=True,
    )

    if dist.is_rank_zero:
        print(f"Run {run_ctx.run_id} | seed={effective_seed}")

    # continue to Step 3 (model, optimizer, data)


if __name__ == "__main__":
    main()
