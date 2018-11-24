from typing import Callable, Union

import numpy as np

from reinforcement_learning.trainers.policies.base_policy import BasePolicy


class SoftmaxPolicy(BasePolicy):
    @classmethod
    def get_action(cls, q_values: np.ndarray, B_RL: float,
                   log_func: Union[None, Callable] = None):
        # exploration: 1, B_RL: 0. exploration: 0, B_RL: infinity
        cls.log_if(log_func, f"q_values: {q_values}")
        e_x = np.exp(B_RL * (q_values - np.max(q_values)))
        probabilities = e_x / e_x.sum(axis=0)
        action = int(np.random.choice([_i for _i in range(len(probabilities))], p=probabilities))
        cls.log_if(log_func, f"action: {action} (softmaxed from {probabilities} with B_RL: {B_RL})")
        return action
