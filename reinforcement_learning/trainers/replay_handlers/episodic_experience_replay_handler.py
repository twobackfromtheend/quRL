import random
from collections import deque
from typing import Deque, List, Sequence

import numpy as np

from reinforcement_learning.trainers.replay_handlers.experience import Experience, InsufficientExperiencesError


class EpisodicExperienceReplayHandler:
    """
    N.B. keras-rl uses their own memory class (as of writing), claiming deque is non-performant.
    But https://github.com/keras-rl/keras-rl/issues/165 - benchmarks show deque is faster.
    """

    def __init__(self, size: int = 100000, batch_size: int = 24):
        self.size = size
        self.batch_size = batch_size
        self.memory: Deque[Sequence[Experience]] = deque(maxlen=size)
        self.episode_buffer = []

    def record_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        # Save step data for episode in episode_buffer
        self.episode_buffer.append(Experience(state, action, reward, next_state, done))

        if done:
            # Append full episode to memoery
            self.memory.append(self.episode_buffer)
            self.episode_buffer = []

    def sample(self):
        if self.batch_size > len(self.memory):
            raise InsufficientExperiencesError
        return random.sample(self.memory, self.batch_size)

    def generator(self) -> List[List[Experience]]:
        batch = self.sample()

        for episode in batch:
            yield episode
