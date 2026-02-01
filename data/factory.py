from data.sources.streaming_source import RangeSource
from data.state import DatasetState
from data.dataset import StreamingIterableDataset


def build_dataset(config, distributed_ctx, dataset_state):
    # toy example; later this becomes S3 / files / etc
    source = RangeSource(config.data.num_samples)

    return StreamingIterableDataset(
        source=source,
        distributed_ctx=distributed_ctx,
        state=dataset_state,
    )
