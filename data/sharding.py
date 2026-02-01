def is_sample_for_rank(sample_idx: int, global_rank: int, world_size: int) -> bool:
    """
    Deterministically assigns samples to ranks.
    """
    return sample_idx % world_size == global_rank


def shard_iterator(iterator, *, global_rank, world_size, start_offset=0):
    """
    Filters a global stream into a per-rank stream.
    """
    for idx, sample in enumerate(iterator):
        if idx < start_offset:
            continue
        if is_sample_for_rank(idx, global_rank, world_size):
            yield idx, sample
