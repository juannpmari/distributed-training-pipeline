# core/distributed/env.py
import os

def read_distributed_env():
    """
    Supports:
    - torchrun
    - single-process fallback
    """
    if "RANK" not in os.environ:
        # Single-process mode
        return {
            "is_distributed": False,
            "global_rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "node_rank": 0,
        }

    return {
        "is_distributed": True,
        "global_rank": int(os.environ["RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "node_rank": int(os.environ.get("NODE_RANK", 0)),
    }
