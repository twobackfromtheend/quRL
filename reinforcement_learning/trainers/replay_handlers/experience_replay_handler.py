import random
from collections import deque
from typing import Deque, List

import numpy as np


class Experience:
    def __init__(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def as_tuple(self):
        """
        Returns in tuple as (state, action, reward, next_state, done)
        :return:
        """
        return self.state, self.action, self.reward, self.next_state, self.done


class ExperienceReplayHandler:
    """
    N.B. keras-rl uses their own memory class (as of writing), claiming deque is non-performant.
    But https://github.com/keras-rl/keras-rl/issues/165 - benchmarks show deque is faster.
    """
    def __init__(self, size: int = 100000, batch_size: int = 128):
        self.size = size
        self.batch_size = batch_size
        self.memory: Deque[Experience] = deque(maxlen=size)

    def record_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self):
        if self.batch_size > len(self.memory):
            raise InsufficientExperiencesError
        return random.sample(self.memory, self.batch_size)

    def generator(self) -> List[Experience]:
        batch = self.sample()

        for experience in batch:
            yield experience


class InsufficientExperiencesError(Exception):
    pass
