from typing import Sequence


class BaseEpisodicReplayHandler:

    def __init__(self):
        raise NotImplementedError

    def record_protocol(self, protocol: Sequence[int], reward: float):
        raise NotImplementedError

    def generator(self):
        raise NotImplementedError


