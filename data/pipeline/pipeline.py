class Pipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self, source_iter):
        it = source_iter
        for stage in self.stages:
            it = stage.run(it)
        return it
