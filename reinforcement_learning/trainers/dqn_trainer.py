import logging
from typing import Union

import numpy as np
from tensorflow.python.keras import backend as K

from logger_utils.logger_utils import log_process
from quantum_evolution.envs.base_q_env import BaseQEnv
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.tensorboard_logger import tf_log, create_callback
from reinforcement_learning.trainers.base_trainer import BaseTrainer
from reinforcement_learning.trainers.hyperparameters import QLearningHyperparameters, ExplorationOptions, \
    ExplorationMethod
from reinforcement_learning.trainers.policies.epsilon_greedy import EpsilonGreedyPolicy
from reinforcement_learning.trainers.policies.softmax import SoftmaxPolicy
from reinforcement_learning.trainers.replay_handlers.step.experience_replay_handler import ExperienceReplayHandler, \
    InsufficientExperiencesError

logger = logging.getLogger(__name__)


class DQNTrainer(BaseTrainer):
    """
    Performs a gradient update on each episode.
    """

    def __init__(self, model: BaseModel, env: BaseQEnv, hyperparameters: QLearningHyperparameters,
                 with_tensorboard: bool):
        super().__init__(model, env, hyperparameters, with_tensorboard)
        self.target_model = model.create_copy()
        self.evaluation_tensorboard = None
        self.evaluation_rewards = []
        self.replay_handler = ExperienceReplayHandler()
        self.episode_number: int = 0
        self.step_number: int = 0

    @log_process(logger, 'training')
    def train(self, episodes: int = 1000, render: bool = False,
              save_every: int = 1000,
              evaluate_every: int = 50,
              update_target_soft: bool = True,
              update_target_tau: float = 0.001):
        exploration = self.hyperparameters.exploration_options

        if self.tensorboard:
            self.tensorboard = create_callback(self.model.model)
        reward_totals = []
        for i in range(episodes):
            logger.info(f"\nEpisode {i}/{episodes}")
            observation = self.env.reset()

            self.update_learning_rate(i)
            logger.info(f"exploration method: {exploration.method}, value: {exploration.get_value(i)}")

            reward_total = 0
            losses = []
            actions = []
            done = False

            while not done:
                action = self.get_action(observation)

                actions.append(action)
                new_observation, reward, done, info = self.env.step(action)
                logger.debug(f"new_observation: {new_observation}")

                self.replay_handler.record_experience(observation, action, reward, new_observation, done)
                try:
                    loss = self.batch_experience_replay()
                    losses.append(loss)
                except InsufficientExperiencesError:
                    pass
                observation = new_observation
                reward_total += reward
                if render:
                    self.env.render()
                self.step_number += 1

            if self.tensorboard and losses:
                losses = list(zip(*losses))
                tf_log(self.tensorboard,
                       ['train_loss', 'train_mae', 'reward'],
                       [np.mean(losses[0]), np.mean(losses[1]), reward_total], i)
            logger.info(f"actions: {actions}")
            logger.info(f"Episode: {i}, reward_total: {reward_total}")

            reward_totals.append(reward_total)

            if i % evaluate_every == evaluate_every - 1:
                self.evaluate_model(render, i // evaluate_every)
            if i % save_every == save_every - 1:
                self.save_model()
            self.update_target_model(update_target_soft, update_target_tau)
            # if i % update_target_every == update_target_every - 1:
            #     self.target_model.model.set_weights(self.model.model.get_weights())

            self.episode_number += 1
        self.reward_totals = reward_totals
        self.save_model()

    def batch_experience_replay(self):
        # General machinery
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for experience in self.replay_handler.generator():
            states.append(experience.state)
            actions.append(experience.action)
            rewards.append(experience.reward)
            next_states.append(experience.next_state)
            dones.append(experience.done)

        # Converting to np.arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Get targets (refactored to handle different calculation based on done)
        targets = self.get_targets(rewards, next_states, dones)

        # Create target_vecs
        target_vecs = self.target_model.model.predict(states)
        for i, action in enumerate(actions):
            # Set target values in target_vecs
            target_vecs[i, action] = targets[i]

        loss = self.model.model.train_on_batch(states, target_vecs)
        return loss

    def get_targets(self, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        Calculates targets based on done.
        If done,
            target = reward
        If not done,
            target = r + gamma * max(q_values(next_state))
        :param states:
        :param actions:
        :param rewards:
        :param next_states:
        :param dones:
        :return: Targets - 1D np.array
        """
        # Targets initialised w/ done == True steps
        targets = rewards.copy()

        # Targets for done == False steps calculated with target network
        done_false_indices = dones == False
        gamma = self.hyperparameters.discount_rate(self.episode_number)
        target_q_values = self.target_model.model.predict(next_states[done_false_indices])
        targets[done_false_indices] = rewards[done_false_indices] + gamma * np.max(target_q_values, axis=1)
        return targets

    def get_policy_q_values(self, state) -> np.ndarray:
        logger.debug("Get policy Q values")
        q_values = self.model.model.predict(state.reshape((1, -1)))
        logger.debug(f"Q values {q_values}")
        return q_values[0]

    def get_target_q_values(self, state) -> np.ndarray:
        logger.debug("Get target Q values")
        q_values = self.target_model.model.predict(state.reshape((1, -1)))
        logger.debug(f"Q values {q_values}")
        return q_values[0]

    def get_action(self, observation,
                   log_func: Union[logging.debug, logging.info] = logging.debug):
        exploration = self.hyperparameters.exploration_options
        q_values = self.get_policy_q_values(observation)

        if exploration.method == ExplorationMethod.EPSILON:
            return EpsilonGreedyPolicy.get_action(
                exploration.get_epsilon(self.episode_number), q_values, self.env.get_random_action, log_func)
        elif exploration.method == ExplorationMethod.SOFTMAX:
            return SoftmaxPolicy.get_action(q_values, exploration.get_B_RL(self.episode_number), log_func)
        else:
            raise ValueError(f"Unknown exploration method: {exploration.method}")

    def update_target_model(self, soft: bool, tau: float):
        """
        Update target model's weights with policy model's.
        If soft,
            theta_target <- tau * theta_policy + (1 - tau) * theta_target
            tau << 1.
        :param soft:
        :param tau: tau << 1 (recommended: 0.001)
        :return:
        """
        if soft:
            self.target_model.model.set_weights(
                tau * np.array(self.model.model.get_weights())
                + (1 - tau) * np.array(self.target_model.model.get_weights())
            )
        else:
            self.target_model.model.set_weights(self.model.model.get_weights())

    @log_process(logger, "saving model")
    def save_model(self):
        self.model.save_model(self.__class__.__name__)

    @log_process(logger, "evaluating model")
    def evaluate_model(self, render, tensorboard_batch_number: int = None):
        if self.evaluation_tensorboard is None and tensorboard_batch_number is not None:
            self.evaluation_tensorboard = create_callback(self.model.model)
        done = False
        reward_total = 0
        observation = self.env.reset()
        actions = []
        while not done:
            action = int(np.argmax(self.get_policy_q_values(observation)))
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

    def update_learning_rate(self, i: int):
        current_learning_rate = float(K.get_value(self.model.model.optimizer.lr))
        logger.info(f"learning rate: {current_learning_rate}")
        new_learning_rate = self.get_learning_rate(i)
        K.set_value(self.model.model.optimizer.lr, new_learning_rate)

    @staticmethod
    def get_learning_rate(i: int) -> float:
        return 1e-5
        # https://www.wolframalpha.com/input/?i=y+%3D+((cos(x+%2F+100)+%2B+1.000)+%2F+2+*+6+*+10%5E-3)+*+exp(-(x+%2F+1000))+%2B+3+*+10%5E-5+for+x+from+0+to+3000
        # return ((math.cos(i / 100) + 1.000) / 2 * 6 * 10 ** -3) * math.e ** -(i / 1000) + 3 * 10 ** -5


if __name__ == '__main__':
    from qutip import *
    from quantum_evolution.simulations.base_simulation import HamiltonianData
    from reinforcement_learning.models.dense_model import DenseModel

    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    initial_state = (-sigmaz() + 2 * sigmax()).groundstate()[1]
    target_state = (-sigmaz() - 2 * sigmax()).groundstate()[1]


    def placeholder_callback(t, args):
        raise RuntimeError


    hamiltonian_datas = [
        HamiltonianData(-sigmaz()),
        HamiltonianData(-sigmax(), placeholder_callback)
    ]

    # RUN FOR QEnv2
    # N = 10
    # t = 0.5
    # # N = 60
    # # t = 3
    # from quantum_evolution.time_sensitive_envs.q_env_2 import QEnv2
    # env = QEnv2(hamiltonian_datas, t, N=N,
    #             initial_state=initial_state, target_state=target_state)
    # model = DenseModel(inputs=2, outputs=2, layer_nodes=(24, 24), learning_rate=3e-3,
    #                    inner_activation='relu', output_activation='linear')

    # RUN FOR QEnv3
    # N = 10
    # t = 0.5
    N = 48
    t = 2.4
    # N = 60
    # t = 3
    from quantum_evolution.envs.q_env_3 import QEnv3
    env = QEnv3(hamiltonian_datas, t, N=N,
                initial_state=initial_state, target_state=target_state)
    model = DenseModel(inputs=3, outputs=2, layer_nodes=(24, 24), learning_rate=3e-3,
                       inner_activation='relu', output_activation='linear')

    # RUN FOR CARTPOLE
    # from reinforcement_learning.time_sensitive_envs.cartpole_env import CartPoleTSEnv
    # time_sensitive = False
    # env = CartPoleTSEnv(time_sensitive=time_sensitive)
    # inputs = 5 if time_sensitive else 4
    # model = DenseModel(inputs=inputs, outputs=2, layer_nodes=(48, 48), learning_rate=3e-3,
    #                    inner_activation='relu', output_activation='linear')

    # RUN FOR ACROBOT
    # from reinforcement_learning.time_sensitive_envs.acrobot_env import AcrobotTSEnv
    # env = AcrobotTSEnv(sparse=True)
    # model = DenseModel(inputs=7, outputs=3, layer_nodes=(48, 48, 24), learning_rate=3e-3,
    #                    inner_activation='relu', output_activation='linear')

    EPISODES = 20000


    def discount_rate(i: int) -> float:
        return min(0.9999, 0.97 + (1 - 0.97) * EPISODES / 2)


    trainer = DQNTrainer(
        model, env,
        hyperparameters=QLearningHyperparameters(
            0.98,
            ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=0.8, epsilon_decay=0.999)
            # ExplorationOptions(method=ExplorationMethod.SOFTMAX, starting_value=0.5, softmax_total_episodes=EPISODES)
        ),
        with_tensorboard=True
    )
    trainer.train(render=False, episodes=EPISODES)
    logger.info(f"max reward total: {max(trainer.reward_totals)}")
    logger.info(f"last evaluation reward: {trainer.evaluation_rewards[-1]}")