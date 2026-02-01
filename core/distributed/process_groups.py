# core/distributed/process_group.py
import torch.distributed as dist
from .utils import default_timeout

def init_process_group(backend: str, world_size: int, rank: int):
    if dist.is_initialized():
        return

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        timeout=default_timeout(),
    )
