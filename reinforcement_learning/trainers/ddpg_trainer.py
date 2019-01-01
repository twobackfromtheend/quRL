import logging
from typing import Union

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam

from logger_utils.logger_utils import log_process
from quantum_evolution.envs.base_q_env import BaseQEnv
from reinforcement_learning.models.base_critic_model import BaseCriticModel
from reinforcement_learning.models.base_nn_model import BaseNNModel
from reinforcement_learning.tensorboard_logger import tf_log, create_callback
from reinforcement_learning.time_sensitive_envs.base_time_sensitive_env import BaseTimeSensitiveEnv
from reinforcement_learning.trainers.base_classes.base_trainer import BaseTrainer
from reinforcement_learning.trainers.base_classes.hyperparameters import QLearningHyperparameters, ExplorationOptions, \
    ExplorationMethod, DDPGHyperparameters
from reinforcement_learning.trainers.dqn_options import DQNTrainerOptions
from reinforcement_learning.trainers.policies.ornstein_uhlenbeck import OrnsteinUhlenbeck
from reinforcement_learning.trainers.replay_handlers.experience_replay_handler import ExperienceReplayHandler, \
    InsufficientExperiencesError

logger = logging.getLogger(__name__)


class DDPGTrainer(BaseTrainer):
    """
    Performs DDPG
    """

    def __init__(self, model: BaseNNModel, env: Union[BaseQEnv, BaseTimeSensitiveEnv],
                 hyperparameters: DDPGHyperparameters, options: DQNTrainerOptions,
                 critic_model: BaseCriticModel):
        super().__init__(model, env, hyperparameters, options)
        # self.model is the actor model
        self.target_actor_model = model.create_copy()

        self.critic_model = critic_model
        self.target_critic_model = critic_model.create_copy()

        self.evaluation_tensorboard = None
        self.evaluation_rewards = []
        self.replay_handler = ExperienceReplayHandler()
        self.episode_number: int = 0
        self.step_number: int = 0

        self.critic_action_input = self.critic_model.action_input
        combined_inputs = []
        state_inputs = []
        for _input in self.critic_model.model.input:
            if _input == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(_input)
                state_inputs.append(_input)
        self.critic_action_input_idx = self.critic_model.model.input.index(self.critic_action_input)

        combined_inputs[self.critic_action_input_idx] = self.model.model(state_inputs)

        combined_output = self.critic_model.model(combined_inputs)

        updates = Adam(lr=.001, clipnorm=1.).get_updates(
            params=self.model.model.trainable_weights, loss=-K.mean(combined_output))

        self.actor_train_fn = K.function(state_inputs + [K.learning_phase()],
                                         [self.model.model(state_inputs)], updates=updates)

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
            if isinstance(exploration, OrnsteinUhlenbeck):
                exploration.reset_states()
                logger.info(f"exploration method: OrnsteinUhlenbeck, theta: {exploration.theta}")
            else:
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

                if self.step_number % update_target_every == 0:
                    self.update_target_models(update_target_soft, update_target_tau)

                observation = new_observation
                reward_total += reward
                if render:
                    self.env.render()
                self.step_number += 1

            if self.tensorboard and losses:
                tf_log(self.tensorboard,
                       ['train_loss', 'reward'],
                       [np.mean(losses), reward_total], i)
            logger.info(f"actions: {np.concatenate(actions)}")
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

        # Train critic
        target_next_actions = self.target_actor_model.predict_on_batch(next_states)

        next_states_with_next_actions = [next_states]
        next_states_with_next_actions.insert(self.critic_action_input_idx, target_next_actions)
        target_q_values = self.target_critic_model.predict_on_batch(next_states_with_next_actions).flatten()
        gamma = self.hyperparameters.discount_rate(self.episode_number)

        discounted_next_state_rewards = gamma * target_q_values
        next_state_rewards_coeff = dones == 0  # Only add the next state reward if not done.
        targets = rewards + discounted_next_state_rewards * next_state_rewards_coeff

        states_with_actions = [states]
        states_with_actions.insert(self.critic_action_input_idx, actions)

        critic_loss = self.critic_model.train_on_batch(states_with_actions, targets)

        # Train actor
        actor_inputs = [states, True]  # True tells model that it's in training mode.
        action_values = self.actor_train_fn(actor_inputs)[0]

        return critic_loss

    def ___get_targets(self, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray) -> np.ndarray:
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
        target_q_values = self.target_model.predict(next_states[done_false_indices])
        targets[done_false_indices] = rewards[done_false_indices] + gamma * np.max(target_q_values, axis=1)
        return targets

    def get_action(self, observation, log_func: Union[logging.debug, logging.info] = logging.debug):
        exploration = self.hyperparameters.exploration_options
        action = self.model.predict(observation.reshape((1, -1))).flatten()

        if isinstance(exploration, OrnsteinUhlenbeck):
            return exploration.get_action(action)
        elif exploration.method == ExplorationMethod.EPSILON:
            if np.random.random() < exploration.get_epsilon(self.episode_number):
                action = self.env.get_random_action()
                log_func(f"action: {action} (randomly generated)")
                return self.env.get_random_action()
            else:
                log_func(f"action: {action} (generated by actor)")
                return action
        else:
            raise ValueError(f"Unknown exploration method: {exploration.method}")

    def update_target_models(self, soft: bool, tau: float):
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
            self.target_actor_model.model.set_weights(
                tau * np.array(self.model.model.get_weights())
                + (1 - tau) * np.array(self.target_actor_model.model.get_weights())
            )
            self.target_critic_model.model.set_weights(
                tau * np.array(self.critic_model.model.get_weights())
                + (1 - tau) * np.array(self.target_critic_model.model.get_weights())
            )
        else:
            self.target_actor_model.model.set_weights(self.model.model.get_weights())
            self.target_critic_model.model.set_weights(self.critic_model.model.get_weights())

    @log_process(logger, "evaluating model")
    def evaluate_model(self, render, tensorboard_batch_number: int = None):
        if self.evaluation_tensorboard is None and tensorboard_batch_number is not None:
            self.evaluation_tensorboard = create_callback(self.model.model)
        done = False
        reward_total = 0
        observation = self.env.reset()
        actions = []
        while not done:
            action = self.model.predict(observation.reshape((1, -1))).flatten()
            actions.append(action)
            new_observation, reward, done, info = self.env.step(action)

            observation = new_observation
            reward_total += reward
            if render:
                self.env.render()
        if tensorboard_batch_number is not None:
            tf_log(self.evaluation_tensorboard, ['reward'], [reward_total], tensorboard_batch_number)
        logger.info(f"actions: {np.concatenate(actions)}")
        logger.info(f"Evaluation reward: {reward_total}")
        self.evaluation_rewards.append(reward_total)


if __name__ == '__main__':
    from reinforcement_learning.models.dense_model import DenseModel

    logging.basicConfig(level=logging.INFO)

    from reinforcement_learning.time_sensitive_envs.pendulum_env import PendulumTSEnv

    time_sensitive = False
    # env = PendulumTSEnv(time_sensitive=time_sensitive, discrete=True)
    env = PendulumTSEnv(time_sensitive=time_sensitive)
    inputs = 4 if time_sensitive else 3
    outputs = 1
    actor_model = DenseModel(inputs=inputs, outputs=outputs, layer_nodes=(48, 48), learning_rate=3e-3,
                             inner_activation='relu', output_activation='tanh')
    critic_model = BaseCriticModel(inputs=inputs, outputs=outputs)

    EPISODES = 20000

    trainer = DDPGTrainer(
        actor_model, env, critic_model=critic_model,
        hyperparameters=DDPGHyperparameters(
            0.95,
            # ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=0.9, epsilon_decay=0.99,
            #                    limiting_value=0.03)
            OrnsteinUhlenbeck(theta=0.15)
        ),
        options=DQNTrainerOptions(render=True)
    )
    trainer.train(episodes=EPISODES)
    logger.info(f"max reward total: {max(trainer.reward_totals)}")
    logger.info(f"last evaluation reward: {trainer.evaluation_rewards[-1]}")
