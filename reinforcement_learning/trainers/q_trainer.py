import logging
from typing import Union

import numpy as np

from logger_utils.logger_utils import log_process
from quantum_evolution.envs.base_q_env import BaseQEnv
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.tensorboard_logger import tf_log, create_callback
from reinforcement_learning.time_sensitive_envs.base_time_sensitive_env import BaseTimeSensitiveEnv
from reinforcement_learning.trainers.base_classes.base_options import BaseTrainerOptions
from reinforcement_learning.trainers.base_classes.base_trainer import BaseTrainer
from reinforcement_learning.trainers.base_classes.hyperparameters import QLearningHyperparameters, ExplorationOptions, \
    ExplorationMethod
from reinforcement_learning.trainers.policies.epsilon_greedy import EpsilonGreedyPolicy
from reinforcement_learning.trainers.policies.softmax import SoftmaxPolicy

logger = logging.getLogger(__name__)


class QTrainer(BaseTrainer):
    """
    Performs Watkins's Q-learning

    """

    def __init__(self, model: BaseModel, env: Union[BaseQEnv, BaseTimeSensitiveEnv],
                 hyperparameters: QLearningHyperparameters, options: BaseTrainerOptions):
        super().__init__(model, env, hyperparameters, options)
        self.evaluation_tensorboard = None
        self.evaluation_rewards = []
        self.episode_number: int = 0
        self.step_number: int = 0

    @log_process(logger, 'training')
    def train(self, episodes: int = 1000):
        exploration = self.hyperparameters.exploration_options
        render = self.options.render
        evaluate_every = self.options.evaluate_every
        save_every = self.options.save_every

        if self.tensorboard:
            self.tensorboard = create_callback(self.model.model)
        reward_totals = []
        for i in range(episodes):
            logger.info(f"\nEpisode {i}/{episodes}")
            observation = self.env.reset()
            logger.info(f"exploration method: {exploration.method}, value: {exploration.get_value(i)}")

            reward_total = 0
            losses = []
            actions = []
            done = False
            print(self.model.model)
            while not done:
                action = self.get_action(observation)

                actions.append(action)
                new_observation, reward, done, info = self.env.step(action)
                logger.debug(f"new_observation: {new_observation}")

                loss = self.train_on_step(observation, action, reward, new_observation, done)
                losses.append(loss)
                observation = new_observation
                reward_total += reward
                if render:
                    self.env.render()
                self.step_number += 1

            if self.tensorboard and losses:
                losses = list(zip(*losses))
                tf_log(
                    self.tensorboard,
                    ['train_loss', 'train_mae', 'reward'],
                    [np.mean(losses[0]), np.mean(losses[1]), reward_total], i
                )
            logger.info(f"actions: {actions}")
            logger.info(f"Episode: {i}, reward_total: {reward_total}")

            reward_totals.append(reward_total)

            if i % evaluate_every == evaluate_every - 1:
                self.evaluate_model(render, i // evaluate_every)
            if i % save_every == save_every - 1:
                self.save_model()

            self.episode_number += 1
        self.reward_totals = reward_totals
        self.save_model()

    def train_on_step(self, state, action, reward, next_state, done) -> float:
        # Get targets (refactored to handle different calculation based on done)
        target = self.get_target(reward, next_state, done)

        # Create target_vecs
        target_vec = self.model.predict(state.reshape((1, -1)))[0]
        target_vec[action] = target
        loss = self.model.train_on_batch(state.reshape((1, -1)), target_vec.reshape((1, -1)))
        return loss

    def get_target(self, reward: float, next_state: np.ndarray, done: bool) -> float:
        """
        Calculates targets based on done.
        If done,
            target = reward
        If not done,
            target = r + gamma * max(q_values(next_state))
        :param reward:
        :param next_state:
        :param done:
        :return: Target - 1D np.array
        """
        if done:
            return reward

        gamma = self.hyperparameters.discount_rate(self.episode_number)
        target_q_values = self.get_q_values(next_state)[0]
        target = reward + gamma * np.max(target_q_values)
        return target

    def get_q_values(self, state) -> np.ndarray:
        q_values = self.model.predict(state.reshape((1, -1)))
        logger.debug(f"Q values {q_values}")
        return q_values[0]

    def get_action(self, observation, log_func: Union[logging.debug, logging.info] = logging.debug):
        exploration = self.hyperparameters.exploration_options
        q_values = self.get_q_values(observation)

        if exploration.method == ExplorationMethod.EPSILON:
            return EpsilonGreedyPolicy.get_action(
                exploration.get_epsilon(self.episode_number), q_values, self.env.get_random_action, log_func)
        elif exploration.method == ExplorationMethod.SOFTMAX:
            return SoftmaxPolicy.get_action(q_values, exploration.get_B_RL(self.episode_number), log_func)
        else:
            raise ValueError(f"Unknown exploration method: {exploration.method}")

    @log_process(logger, "evaluating model")
    def evaluate_model(self, render, tensorboard_batch_number: int = None):
        if self.evaluation_tensorboard is None and tensorboard_batch_number is not None:
            self.evaluation_tensorboard = create_callback(self.model.model)
        done = False
        reward_total = 0
        observation = self.env.reset()
        actions = []
        while not done:
            action = int(np.argmax(self.get_q_values(observation)))
            actions.append(action)
            new_observation, reward, done, info = self.env.step(action)

            observation = new_observation
            reward_total += reward
            if render:
                self.env.render()
        if tensorboard_batch_number is not None:
            tf_log(self.evaluation_tensorboard, ['reward'], [reward_total], tensorboard_batch_number)
        logger.info(f"actions: {actions}")
        logger.info(f"Evaluation reward: {reward_total}")
        self.evaluation_rewards.append(reward_total)


if __name__ == '__main__':
    from reinforcement_learning.models.dense_model import DenseModel
    from reinforcement_learning.models.q_table import QTable

    logging.basicConfig(level=logging.INFO)

    # model = DenseModel(inputs=inputs, outputs=2, layer_nodes=(12, 12), learning_rate=3e-3,
    #                    inner_activation='relu', output_activation='linear')

    from qutip import *

    t = 0.5
    # t = 2.4
    # t = 3
    from quantum_evolution.envs.q_env_2 import QEnv2
    from reinforcement_learning.runners.utils.quantum_variables import get_quantum_variables
    initial_state, target_state, hamiltonian_datas, N = get_quantum_variables(t)

    env = QEnv2(hamiltonian_datas, t, N=N,
                initial_state=initial_state, target_state=target_state)
    model = QTable(inputs=2, outputs=2, learning_rate=0.2)
    EPISODES = 20000


    trainer = QTrainer(
        model, env,
        hyperparameters=QLearningHyperparameters(
            0.99,
            # ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=1, epsilon_decay=0.9994,
            #                    limiting_value=0.1)
            ExplorationOptions(method=ExplorationMethod.SOFTMAX, starting_value=0.01, softmax_total_episodes=EPISODES)
        ),
        options=BaseTrainerOptions(render=False)
    )
    trainer.train(episodes=EPISODES)
    logger.info(f"max reward total: {max(trainer.reward_totals)}")
    logger.info(f"last evaluation reward: {trainer.evaluation_rewards[-1]}")
