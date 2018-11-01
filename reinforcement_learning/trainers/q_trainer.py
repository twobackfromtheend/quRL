import logging
from typing import List

import numpy as np

from logger_utils.logger_utils import log_process
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.tensorboard_logger import create_callback, tf_log
from reinforcement_learning.trainers.base_trainer import BaseTrainer

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
        assert 0.5 < decay_rate <= 1, "Decay rate (discount rate) has to be between 0 and 1"
        self.exploration_options = exploration_options


class QTrainer(BaseTrainer):
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
            if self.tensorboard:
                self.tensorboard = create_callback(self.model.model)

            logger.info(f"Episode {i}/{episodes}")
            observation = self.env.reset()
            exploration.decay_current_value()

            reward_total = 0
            for time in range(500):
                if render:
                    self.env.render()
                logger.debug(f"time: {time}")
                if np.random.random() < exploration.current_value:
                    action = self.env.action_space.sample()
                    logger.debug(f"action: {action} (randomly generated)")
                else:
                    action = np.argmax(self.get_q_values(observation)[0])
                    logger.debug(f"action: {action} (argmaxed)")
                new_observation, reward, done, info = self.env.step(action)
                logger.debug(f"new_observation: {new_observation}")

                # https://keon.io/deep-q-learning/
                reward = reward if not done else -100  # Prevents reward from ballooning
                target = reward + gamma * np.max(self.get_q_values(new_observation)[0])

                logger.debug(f"target: {target}")
                target_vec = self.get_q_values(observation)[0]
                logger.debug(f"target_vec: {target_vec}")
                target_vec[action] = target

                loss = self.model.model.train_on_batch(observation.reshape((1, -1)), target_vec.reshape((1, -1)))
                logger.debug(f"loss: {loss}")
                if self.tensorboard:
                    tf_log(self.tensorboard, ['train_loss', 'train_mae'], loss, time)
                observation = new_observation
                reward_total += reward
                if done:
                    logger.info(f"Episode: {i}, time: {time}")
                    break
            logger.debug(f"reward total: {reward_total}")

            reward_totals.append(reward_total)

        self.reward_totals = reward_totals

    def get_q_values(self, state) -> np.ndarray:
        logger.debug("Get Q values")
        q_values = self.model.model.predict(state.reshape((1, -1)))
        logger.debug(f"Q values {q_values}")
        return q_values
