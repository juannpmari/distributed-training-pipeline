class StreamingSource:
    """
    Abstract streaming data source.
    Must yield samples indefinitely or until exhaustion.
    """

    def __iter__(self):
        raise NotImplementedError

class RangeSource(StreamingSource):
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            yield i