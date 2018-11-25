import logging
from enum import Enum, auto
from typing import Callable, Union

logger = logging.getLogger(__name__)


class ExplorationMethod(Enum):
    SOFTMAX = auto()
    EPSILON = auto()


class ExplorationOptions:
    def __init__(self,
                 method: ExplorationMethod = ExplorationMethod.EPSILON,
                 starting_value: float = None,
                 limiting_value: float = None,
                 epsilon_decay: float = None,
                 softmax_total_episodes: int = None):
        """
        Defines the exploration behaviour of the agent.

        If method is EPSILON, the agent has a chance of taking a random action instead of being greedy (epsilon-greedy).
        If method is SOFTMAX, the agent chooses the action probabilistically based on the q values.
        :param method: Exploration method - EPSILON or SOFTMAX.
        :param starting_value: initial value of epsilon/B_RL
            for EPSILON, defaults to 0.8
            for SOFTMAX, defaults to 0
        :param limiting_value:
            for EPSILON, this is the minimum epsilon beyond which decay is halted. (default 0.04)
            for SOFTMAX, this is the maximum B_RL beyond which increase is halted (default 100)
        :param epsilon_decay: the rate at which epsilon is decayed per episode, default: 0.998
        :param softmax_total_episodes: total number of episodes, required for SOFTMAX
        """
        """
        :param starting_value: initial exploration rate, default: 0.8
        :param decay: the rate at which exploration is decayed per episode, default: 0.99
        """
        self.method = method
        self.starting_value = starting_value

        if self.method == ExplorationMethod.EPSILON:
            if starting_value is None:
                starting_value = 0.8
            else:
                assert 0 <= starting_value <= 1, f"Initial exploration rate has to be between 0 and 1, not {starting_value}"

            if limiting_value is None:
                limiting_value = 0.04
            else:
                assert 0 <= limiting_value <= 1, f"Limiting epsilon has to be between 0 and 1, not {limiting_value}"

            if epsilon_decay is None:
                epsilon_decay = 0.998
            else:
                assert 0 <= epsilon_decay <= 1, f"Exploration rate decay has to be between 0 and 1, not {epsilon_decay}"

            self.starting_value = starting_value
            self.limiting_value = limiting_value
            self.decay = epsilon_decay

        elif self.method == ExplorationMethod.SOFTMAX:
            assert softmax_total_episodes is not None, "total_episodes is a required parameter for SOFTMAX"
            self.limiting_value = limiting_value if limiting_value is not None else 100
            self.softmax_total_episodes = softmax_total_episodes

    def get_value(self, i: int):
        """
        :param i: Episode number
        :return:
        """
        if self.method == ExplorationMethod.EPSILON:
            return self.get_epsilon(i)
        elif self.method == ExplorationMethod.SOFTMAX:
            return self.get_B_RL(i)
        else:
            raise ValueError

    def get_epsilon(self, i: int):
        """
        :param i: Episode number
        :return:
        """
        assert self.method == ExplorationMethod.EPSILON, "Calling get_epsilon when method is not set to EPSILON."
        epsilon = self.starting_value * self.decay ** i
        return max(self.limiting_value, epsilon)

    def get_B_RL(self, i: int):
        """
        :param i: Episode number
        :return:
        """
        assert self.method == ExplorationMethod.SOFTMAX, "Calling get_B_RL when method is not set to SOFTMAX."
        B_RL = self.softmax_total_episodes / (self.softmax_total_episodes - i)
        return min(self.limiting_value, B_RL)


DiscountRate = Union[float, Callable[[int], float]]


class QLearningHyperparameters:
    def __init__(self, discount_rate: DiscountRate, exploration_options: ExplorationOptions = ExplorationOptions()):
        """
        Defines hyperparameters required for Q-learning.
        :param discount_rate: (aka discount rate) - used to calculate future discounted reward, suggested: 0.95
            Can be a function that takes in the episode number as argument and returns a float
        :param exploration_options: See ExplorationOptions
        """
        self.discount_rate = discount_rate
        if isinstance(discount_rate, float):
            assert 0 < discount_rate <= 1, "Decay rate (discount rate) has to be between 0 and 1"
            self.discount_rate = lambda _: discount_rate
        self.exploration_options = exploration_options
