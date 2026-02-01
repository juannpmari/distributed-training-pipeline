# core/distributed/init.py
import torch
from .env import read_distributed_env
from .utils import choose_backend
from .process_group import init_process_group
from .context import DistributedContext

def init_distributed() -> DistributedContext:
    env = read_distributed_env()
    backend = choose_backend()

    if env["is_distributed"]:
        init_process_group(
            backend=backend,
            world_size=env["world_size"],
            rank=env["global_rank"],
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(env["local_rank"])

    # Infer topology (flat assumption)
    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_nodes = env["world_size"] // gpus_per_node

    ctx = DistributedContext(
        world_size=env["world_size"],
        global_rank=env["global_rank"],
        local_rank=env["local_rank"],
        node_rank=env["node_rank"],
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        backend=backend,
        is_distributed=env["is_distributed"],
    )

    return ctx
