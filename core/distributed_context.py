from dataclasses import dataclass
import os
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool

    def is_main_rank(self) -> bool:
        return self.rank == 0

    def barrier(self):
        if self.is_distributed:
            dist.barrier()


def init_distributed(backend: str = "nccl") -> DistributedContext:
    if "RANK" not in os.environ:
        # Single-process execution
        return DistributedContext(
            rank=0,
            world_size=1,
            local_rank=0,
            is_distributed=False,
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_distributed=True,
    )
