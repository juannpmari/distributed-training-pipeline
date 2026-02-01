# core/distributed/sync.py
import torch
import torch.distributed as dist


def sync_and_validate_loss(loss: torch.Tensor):
    """
    Ensures all ranks have the same loss value.
    """
    with torch.no_grad():
        loss_clone = loss.detach().clone()
        dist.all_reduce(loss_clone, op=dist.ReduceOp.SUM)
        loss_clone /= dist.get_world_size()

        if not torch.allclose(loss, loss_clone, atol=1e-6):
            raise RuntimeError(
                f"Loss mismatch across ranks: local={loss.item()} global={loss_clone.item()}"
            )

        return loss_clone
