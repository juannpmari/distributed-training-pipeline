class PipelineStage:
    """
    Base class for a pipeline stage.
    """

    def run(self, input_iter):
        raise NotImplementedError
