from typing import Callable, Union

import numpy as np

from reinforcement_learning.trainers.policies.base_policy import BasePolicy


class EpsilonGreedyPolicy(BasePolicy):
    @classmethod
    def get_action(cls, epsilon: float, q_values: np.ndarray, get_random_action: Callable,
                   log_func: Union[None, Callable] = None):
        if np.random.random() < epsilon:
            action = get_random_action()
            cls.log_if(log_func, f"action: {action} (randomly generated)")
        else:
            cls.log_if(log_func, f"q_values: {q_values}")
            action = int(np.argmax(q_values))
            cls.log_if(log_func, f"action: {action} (argmaxed)")
        return action
