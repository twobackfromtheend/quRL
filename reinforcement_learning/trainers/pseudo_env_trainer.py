import logging
from typing import List

import numpy as np

from logger_utils.logger_utils import log_process
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.tensorboard_logger import create_callback, tf_log
from reinforcement_learning.trainers.base_trainer import BaseTrainer
from reinforcement_learning.trainers.hyperparameters import QLearningHyperparameters

logger = logging.getLogger(__name__)


class PseudoEnvTrainer(BaseTrainer):
    def __init__(self, model: BaseModel, env, hyperparameters: QLearningHyperparameters, with_tensorboard: bool):
        super().__init__(model, env)
        self.hyperparameters = hyperparameters
        self.env = env
        self.reward_totals: List[float] = None
        self.tensorboard = create_callback(self.model.model) if with_tensorboard else None

    @log_process(logger, 'training')
    def train(self, episodes: int = 1000, render: bool = False):
        exploration = self.hyperparameters.exploration_options
        gamma = self.hyperparameters.decay_rate

        reward_totals = []
        for i in range(episodes):
            logger.info(f"Episode {i}/{episodes}")
            observation = self.env.reset()
            exploration.decay_current_value()
            logger.info(f"exploration: {exploration.current_value}")

            action = int(np.argmax(self.get_q_values(observation)))
            logger.debug(f"original action  : {self.env.convert_int_to_bit_list(action, self.env.N)}")
            action = self.env.randomise_action(action, exploration.current_value)
            logger.debug(f"randomised action: {self.env.convert_int_to_bit_list(action, self.env.N)}")

            new_observation, reward, done, info = self.env.step(action)
            logger.debug(f"new_observation: {new_observation}")

            # https://keon.io/deep-q-learning/
            target = reward + gamma * np.max(self.get_q_values(new_observation))

            logger.debug(f"target: {target}")
            target_vec = self.get_q_values(observation)
            logger.debug(f"target_vec: {target_vec}")
            target_vec[action] = target

            loss = self.model.model.train_on_batch(observation.reshape((1, -1)), target_vec.reshape((1, -1)))
            logger.info(f"loss: {loss}")
            if self.tensorboard:
                tf_log(self.tensorboard, ['train_loss', 'train_mae', 'reward'], [loss[0], loss[1], reward], i)

            logger.info(f"Episode: {i}, reward: {reward}")
            reward_totals.append(reward)

            if render and (i == 50 or i % 400 == 0):
                self.env.render()

        self.reward_totals = reward_totals

    def get_q_values(self, state) -> np.ndarray:
        logger.debug("Get Q values")
        q_values = self.model.model.predict(state.reshape((1, -1)))
        logger.debug(f"Q values {q_values}")
        return q_values[0]
