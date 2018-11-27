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


class InsufficientExperiencesError(Exception):
    pass
