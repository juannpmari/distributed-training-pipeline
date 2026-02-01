# core/training/checkpoint.py
import torch


def save_checkpoint(
    run_ctx,
    model,
    optimizer,
    step,
    epoch,
    extra_state=None,
):
    path = run_ctx.checkpoints_dir / f"ckpt_step_{step}.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "extra_state": extra_state or {},
    }, path)
    return path


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
