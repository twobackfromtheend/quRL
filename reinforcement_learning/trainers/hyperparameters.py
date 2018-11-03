import logging

logger = logging.getLogger(__name__)


class ExplorationOptions:
    def __init__(self, starting_value: float = 0.5, decay: float = 0.998, min_value: float = 0.1):
        """
        Defines exploration rate, the rate at which an agent randomly decides its action instead of being greedy.
        :param starting_value: initial exploration rate, default: 0.8
        :param decay: the rate at which exploration is decayed per episode, default: 0.99
        """
        assert 0 <= starting_value <= 1, "Initial exploration rate has to be between 0 and 1"
        assert 0 <= decay <= 1, "Exploration rate decay has to be between 0 and 1"

        self.starting_value = starting_value
        self.current_value = starting_value
        self.min_value = min_value
        self.decay = decay

    def decay_current_value(self):
        self.current_value = max(self.min_value, self.current_value * self.decay)
        logger.debug(f'decayed epsilon: {self.current_value}')


class QLearningHyperparameters:
    def __init__(self, decay_rate: float, exploration_options: ExplorationOptions = ExplorationOptions()):
        """
        Defines hyperparameters required for Q-learning.
        :param decay_rate: (aka discount rate) - used to calculate future discounted reward, suggested: 0.95
        :param exploration_options: See ExplorationOptions
        """
        self.decay_rate = decay_rate
        assert 0 < decay_rate <= 1, "Decay rate (discount rate) has to be between 0 and 1"
        self.exploration_options = exploration_options
