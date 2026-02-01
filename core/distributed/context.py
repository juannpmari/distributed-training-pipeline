from dataclasses import dataclass

@dataclass(frozen=True)
class DistributedContext:
    # Global
    world_size: int
    global_rank: int

    # Local
    local_rank: int
    node_rank: int

    # Topology
    num_nodes: int
    gpus_per_node: int

    # Backend
    backend: str
    is_distributed: bool

    def is_master(self) -> bool:
        return self.global_rank == 0

    def is_local_master(self) -> bool:
        return self.local_rank == 0
