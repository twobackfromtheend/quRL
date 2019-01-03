import logging
from typing import Union

from quantum_evolution.envs.base_q_env import BaseQEnv
from reinforcement_learning.models.base_critic_model import BaseCriticModel
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.time_sensitive_envs.base_time_sensitive_env import BaseTimeSensitiveEnv
from reinforcement_learning.trainers.base_classes.base_options import BaseTrainerOptions
from reinforcement_learning.trainers.base_classes.base_trainer import BaseTrainer
from reinforcement_learning.trainers.base_classes.hyperparameters import QLearningHyperparameters, DDPGHyperparameters

logger = logging.getLogger(__name__)


class BaseActorCriticTrainer(BaseTrainer):
    def __init__(self, model: BaseModel, env: Union[BaseQEnv, BaseTimeSensitiveEnv],
                 hyperparameters: Union[QLearningHyperparameters, DDPGHyperparameters], options: BaseTrainerOptions,
                 critic_model: BaseCriticModel):
        super().__init__(model, env, hyperparameters, options)
        self.critic_model = critic_model
