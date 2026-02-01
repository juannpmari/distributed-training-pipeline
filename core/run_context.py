# core/run_context.py
from dataclasses import dataclass
from core.distributed.context import DistributedContext

@dataclass
class RunContext:
    distributed: DistributedContext
    config: dict
    run_id: str
    paths: dict
    logger: object

    def is_master(self) -> bool:
        return self.distributed.is_master()
