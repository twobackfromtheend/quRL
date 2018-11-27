import logging
from typing import Union

import numpy as np

from logger_utils.logger_utils import log_process
from quantum_evolution.envs.base_q_env import BaseQEnv
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.tensorboard_logger import tf_log, create_callback
from reinforcement_learning.time_sensitive_envs.base_time_sensitive_env import BaseTimeSensitiveEnv
from reinforcement_learning.trainers.base_classes.base_trainer import BaseTrainer
from reinforcement_learning.trainers.base_classes.hyperparameters import QLearningHyperparameters, ExplorationOptions, \
    ExplorationMethod
from reinforcement_learning.trainers.dqn_options import DQNTrainerOptions
from reinforcement_learning.trainers.policies.epsilon_greedy import EpsilonGreedyPolicy
from reinforcement_learning.trainers.policies.softmax import SoftmaxPolicy
from reinforcement_learning.trainers.replay_handlers.experience_replay_handler import ExperienceReplayHandler, \
    InsufficientExperiencesError

logger = logging.getLogger(__name__)


class DQNTrainer(BaseTrainer):
    """
    Performs a gradient update on each episode.
    """

    def __init__(self, model: BaseModel, env: Union[BaseQEnv, BaseTimeSensitiveEnv],
                 hyperparameters: QLearningHyperparameters, options: DQNTrainerOptions):
        super().__init__(model, env, hyperparameters, options)
        self.target_model = model.create_copy()
        self.evaluation_tensorboard = None
        self.evaluation_rewards = []
        self.replay_handler = ExperienceReplayHandler()
        self.episode_number: int = 0
        self.step_number: int = 0

    @log_process(logger, 'training')
    def train(self, episodes: int = 1000):
        exploration = self.hyperparameters.exploration_options
        learning_rate_override = self.options.learning_rate_override
        update_target_every = self.options.update_target_every
        update_target_soft = self.options.update_target_soft
        update_target_tau = self.options.update_target_tau
        render = self.options.render
        evaluate_every = self.options.evaluate_every
        save_every = self.options.save_every

        if self.tensorboard:
            self.tensorboard = create_callback(self.model.model)
        reward_totals = []
        for i in range(episodes):
            logger.info(f"\nEpisode {i}/{episodes}")
            observation = self.env.reset()
            if learning_rate_override:
                self.update_learning_rate(learning_rate_override(i))
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

                if i % update_target_every == 0:
                    self.update_target_model(update_target_soft, update_target_tau)

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

    def get_action(self, observation, log_func: Union[logging.debug, logging.info] = logging.debug):
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

    def get_q_values(self, state):
        """
        Not needed - see get_target_q_values and get_policy_q_values
        :param state:
        :return:
        """
        raise RuntimeError


if __name__ == '__main__':
    from reinforcement_learning.models.dense_model import DenseModel

    logging.basicConfig(level=logging.INFO)

    from reinforcement_learning.time_sensitive_envs.cartpole_env import CartPoleTSEnv

    time_sensitive = False
    env = CartPoleTSEnv(time_sensitive=time_sensitive)
    inputs = 5 if time_sensitive else 4
    model = DenseModel(inputs=inputs, outputs=2, layer_nodes=(48, 48), learning_rate=3e-3,
                       inner_activation='relu', output_activation='linear')

    EPISODES = 20000

    trainer = DQNTrainer(
        model, env,
        hyperparameters=QLearningHyperparameters(
            0.95,
            ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=0.5, epsilon_decay=0.999,
                               limiting_value=0.1)
            # ExplorationOptions(method=ExplorationMethod.SOFTMAX, starting_value=0.5, softmax_total_episodes=EPISODES)
        ),
        options=DQNTrainerOptions()
    )
    trainer.train(episodes=EPISODES)
    logger.info(f"max reward total: {max(trainer.reward_totals)}")
    logger.info(f"last evaluation reward: {trainer.evaluation_rewards[-1]}")
