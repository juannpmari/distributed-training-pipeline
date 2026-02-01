from dataclasses import dataclass


@dataclass
class DatasetState:
    """
    Serializable cursor for streaming datasets.
    Represents how many samples this rank has consumed.
    """
    global_offset: int

    def advance(self, n: int):
        self.global_offset += n
