import logging
from typing import Union, List

import numpy as np

from logger_utils.logger_utils import log_process
from quantum_evolution.envs.base_q_env import BaseQEnv
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.tensorboard_logger import tf_log, create_callback
from reinforcement_learning.time_sensitive_envs.base_time_sensitive_env import BaseTimeSensitiveEnv
from reinforcement_learning.trainers.base_classes.base_trainer import BaseTrainer
from reinforcement_learning.trainers.base_classes.hyperparameters import QLearningHyperparameters, ExplorationOptions, \
    ExplorationMethod
from reinforcement_learning.trainers.policies.epsilon_greedy import EpsilonGreedyPolicy
from reinforcement_learning.trainers.policies.softmax import SoftmaxPolicy
from reinforcement_learning.trainers.rq_options import RQTrainerOptions

logger = logging.getLogger(__name__)


class RQTrainer(BaseTrainer):
    """
    Performs Watkins's Q-learning

    """

    def __init__(self, model: BaseModel, env: Union[BaseQEnv, BaseTimeSensitiveEnv],
                 hyperparameters: QLearningHyperparameters, options: RQTrainerOptions):
        super().__init__(model, env, hyperparameters, options)
        self.evaluation_tensorboard = None
        self.evaluation_rewards = []
        self.episode_number: int = 0
        self.step_number: int = 0
        self.step_buffer: List[np.ndarray] = []  # list of states

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
            self.step_buffer = []
            observation = self.env.reset()
            logger.info(f"exploration method: {exploration.method}, value: {exploration.get_value(i)}")

            reward_total = 0
            losses = []
            actions = []
            done = False
            print(f"QTable length: {len(self.model.model)}")
            while not done:
                self.step_buffer.append(observation)
                action = self.get_action(self.get_observation())

                actions.append(action)
                new_observation, reward, done, info = self.env.step(action)
                logger.debug(f"new_observation: {new_observation}")

                loss = self.train_on_step(action, reward, new_observation, done)
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

    def get_observation(self, step_buffer=None):
        rnn_steps = self.options.rnn_steps

        if step_buffer is None:
            step_buffer = self.step_buffer

        step_buffer_length = len(step_buffer)
        if step_buffer_length >= rnn_steps:
            return np.array(step_buffer[-rnn_steps:])

        observation = [step_buffer[0] for _ in range(rnn_steps)]
        for i, j in enumerate(range(rnn_steps - step_buffer_length, rnn_steps)):
            # i is a counter from 0
            # j is a counter to fill up the last values with the existing values
            observation[j] = step_buffer[i]
        return np.array(observation)

    def train_on_step(self, action, reward, next_state, done) -> float:
        # Get targets (refactored to handle different calculation based on done)

        states = self.get_observation()
        next_states = self.get_observation(self.step_buffer + [next_state])

        target = self.get_target(reward, next_states, done)
        # Create target_vecs
        target_vec = self.model.predict(states.reshape((1, -1)))[0]
        target_vec[action] = target
        loss = self.model.train_on_batch(states.reshape((1, -1)), target_vec.reshape((1, -1)))
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
        step_buffer = []
        actions = []
        while not done:
            step_buffer.append(observation)
            action = int(np.argmax(self.get_q_values(self.get_observation(step_buffer=step_buffer))))
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
    model = QTable(inputs=2, outputs=2, learning_rate=0.05)
    EPISODES = 20000


    trainer = RQTrainer(
        model, env,
        hyperparameters=QLearningHyperparameters(
            0.99,
            # ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=1, epsilon_decay=0.9994,
            #                    limiting_value=0.1)
            ExplorationOptions(method=ExplorationMethod.SOFTMAX, starting_value=100, limiting_value=1000,
                               softmax_total_episodes=EPISODES)
        ),
        options=RQTrainerOptions(rnn_steps=10, render=False)
    )
    trainer.train(episodes=EPISODES)
    logger.info(f"max reward total: {max(trainer.reward_totals)}")
    logger.info(f"last evaluation reward: {trainer.evaluation_rewards[-1]}")
