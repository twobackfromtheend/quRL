import random
from collections import deque

import numpy as np


class ExperienceReplayHandler:

    def __init__(self, size: int = 1e3):
        self.size = size
        self.memory = deque(maxlen=size)

    def record_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def generator(self, count: int):
        batch = random.sample(self.memory, count)

        for experience in batch:
            yield experience


