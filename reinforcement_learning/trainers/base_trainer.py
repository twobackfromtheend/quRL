from typing import List

from quantum_evolution.envs.base_q_env import BaseQEnv
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.trainers.hyperparameters import QLearningHyperparameters


class BaseTrainer:

    def __init__(self,
                 model: BaseModel,
                 env: BaseQEnv,
                 hyperparameters: QLearningHyperparameters,
                 with_tensorboard: bool):
        self.model = model
        self.env = env
        self.hyperparameters = hyperparameters
        self.reward_totals: List[float] = None
        self.tensorboard = with_tensorboard

    def train(self):
        raise NotImplementedError

    def get_q_values(self, state):
        raise NotImplementedError
