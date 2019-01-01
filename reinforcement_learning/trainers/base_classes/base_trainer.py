import logging
from typing import List, Union

from logger_utils.logger_utils import log_process
from quantum_evolution.envs.base_q_env import BaseQEnv
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.time_sensitive_envs.base_time_sensitive_env import BaseTimeSensitiveEnv
from reinforcement_learning.trainers.base_classes.base_options import BaseTrainerOptions
from reinforcement_learning.trainers.base_classes.hyperparameters import QLearningHyperparameters, DDPGHyperparameters

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self,
                 model: BaseModel,
                 env: Union[BaseQEnv, BaseTimeSensitiveEnv],
                 hyperparameters: Union[QLearningHyperparameters, DDPGHyperparameters],
                 options: BaseTrainerOptions):
        self.model = model
        self.env = env
        self.hyperparameters = hyperparameters
        self.options = options
        self.reward_totals: List[float] = None
        self.tensorboard = options.with_tensorboard

    def train(self, episodes: int):
        raise NotImplementedError

    def get_q_values(self, state):
        raise NotImplementedError

    def update_learning_rate(self, learning_rate: float):
        self.model.set_learning_rate(learning_rate)

    @log_process(logger, "saving model")
    def save_model(self):
        self.model.save_model(self.__class__.__name__)

    def evaluate_model(self, render: bool, *args):
        raise NotImplementedError
