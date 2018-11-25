from enum import Enum

from reinforcement_learning.trainers.hyperparameters import ExplorationOptions, ExplorationMethod


class ExplorationPreset(Enum):
    EPSILON_08 = ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=0.8, epsilon_decay=0.999,
                                    limiting_value=0.1)
    EPSILON_05 = ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=0.5, epsilon_decay=0.999,
                                    limiting_value=0.1)
    SOFTMAX_10000 = ExplorationOptions(method=ExplorationMethod.SOFTMAX, starting_value=0.5,
                                       softmax_total_episodes=10000)
