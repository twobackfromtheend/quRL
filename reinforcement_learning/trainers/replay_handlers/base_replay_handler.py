import numpy as np


class BaseReplayHandler:

    def __init__(self):
        raise NotImplementedError

    def record_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        raise NotImplementedError

    def generator(self):
        raise NotImplementedError


