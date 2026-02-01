# core/distributed/ddp.py
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def wrap_ddp(model: torch.nn.Module, device_id: int):
    """
    Wraps a model in DistributedDataParallel.
    """
    return DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
