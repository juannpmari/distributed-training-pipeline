import torch
from torch.utils.data import IterableDataset

from data.sharding import shard_iterator
from data.state import DatasetState


class StreamingIterableDataset(IterableDataset):
    def __init__(
        self,
        source,
        *,
        distributed_ctx,
        state: DatasetState,
    ):
        self.source = source
        self.ctx = distributed_ctx
        self.state = state

    def __iter__(self):
        base_iter = iter(self.source)

        sharded = shard_iterator(
            base_iter,
            global_rank=self.ctx.global_rank,
            world_size=self.ctx.world_size,
            start_offset=self.state.global_offset,
        )

        for idx, sample in sharded:
            self.state.global_offset = idx + 1
            yield sample
