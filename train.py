# train.py
from pathlib import Path

from torch.utils.data import DataLoader

from core.config import resolve_config
from core.seeding import seed_everything
from core.run_context import create_run_context
from core.distributed.env import get_distributed_context

from core.training.loop import train_one_epoch
from core.training.loss import build_loss
from core.training.optimizer import build_optimizer
from core.training.model import build_model

from core.data.factory import build_dataset
from core.data.state import DatasetState

from core.distributed.ddp import wrap_ddp

from core.tracking.logger import EventLogger
from core.training.checkpoint import save_checkpoint



def main():
    # ---- distributed context ----
    dist = get_distributed_context()

    # ---- config resolution ----
    cfg = resolve_config(
        base_config_path=Path("configs/train.yaml"),
        overrides={},
    )

    if dist.is_rank_zero:
        repro = run_ctx.root_dir / "reproduce.sh"
        repro.write_text(
            "#!/bin/bash\n"
            f"python train.py --config {run_ctx.root_dir / 'resolved_config.yaml'}\n"
        )
        repro.chmod(0o755)


    # ---- run context ----
    run_ctx = create_run_context(
        base_dir=Path("runs"),
        experiment_name="llm_pretrain",
        resolved_config=cfg.to_dict(),
    )

    logger = EventLogger(run_ctx.logs_dir / "events.jsonl")

    if dist.is_rank_zero:
        logger.log("run_start", {"run_id": run_ctx.run_id})

    cfg.save_yaml(run_ctx.root_dir / "resolved_config.yaml")

    # ---- seeding ----
    effective_seed = seed_everything(
        base_seed=cfg.data["seed"],
        rank=dist.rank,
        deterministic=True,
    )

    if dist.is_rank_zero:
        print(f"Run {run_ctx.run_id} | seed={effective_seed}")

    # ============================
    # Step 3: model + optimizer
    # ============================
    model = build_model(cfg).to(dist.device)
    if dist.world_size > 1:
        model = wrap_ddp(model, device_id=dist.local_rank)
    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)

    # ============================
    # Step 4: streaming dataset
    # ============================
    dataset_state = DatasetState(global_offset=0)

    dataset = build_dataset(
        config=cfg,
        distributed_ctx=dist,
        dataset_state=dataset_state,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=0,      # important for resumable streaming
        pin_memory=True,
    )

    # ============================
    # Training loop
    # ============================
    for epoch in range(cfg.training.epochs):
        metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=dist.device,
            epoch=epoch,
            dist_ctx=dist,
        )

        if dist.is_rank_zero:
            print(f"[epoch {epoch}] metrics={metrics}")
            path = save_checkpoint(run_ctx, model, optimizer, step, epoch)
            logger.log("checkpoint_saved", {"path": str(path)})


if __name__ == "__main__":
    main()