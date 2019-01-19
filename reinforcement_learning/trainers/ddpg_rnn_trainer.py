import logging
from typing import Union, List

import numpy as np

from logger_utils.logger_utils import log_process
from quantum_evolution.envs.base_q_env import BaseQEnv
from reinforcement_learning.models.base_critic_model import BaseCriticModel
from reinforcement_learning.models.base_nn_model import BaseNNModel
from reinforcement_learning.models.rnn__critic_model import RNNCriticModel
from reinforcement_learning.tensorboard_logger import tf_log, create_callback
from reinforcement_learning.time_sensitive_envs.base_time_sensitive_env import BaseTimeSensitiveEnv
from reinforcement_learning.trainers.base_classes.base_actor_critic_trainer import BaseActorCriticTrainer
from reinforcement_learning.trainers.base_classes.hyperparameters import ExplorationMethod, DDPGHyperparameters
from reinforcement_learning.trainers.ddpg_rnn_options import DDPGRNNTrainerOptions
from reinforcement_learning.trainers.policies.ornstein_uhlenbeck import OrnsteinUhlenbeck
from reinforcement_learning.trainers.replay_handlers.episodic_experience_replay_handler import \
    EpisodicExperienceReplayHandler
from reinforcement_learning.trainers.replay_handlers.experience_replay_handler import InsufficientExperiencesError, \
    ExperienceReplayHandler

logger = logging.getLogger(__name__)


class DDPGRNNTrainer(BaseActorCriticTrainer):
    """
    Performs DDPG
    """

    def __init__(self, model: BaseNNModel, env: Union[BaseQEnv, BaseTimeSensitiveEnv],
                 hyperparameters: DDPGHyperparameters, options: DDPGRNNTrainerOptions,
                 critic_model: BaseCriticModel):
        super().__init__(model, env, hyperparameters, options, critic_model)
        # self.model is the actor model
        self.target_actor_model = model.create_copy()
        self.target_critic_model = critic_model.create_copy()

        self.evaluation_tensorboard = None
        self.evaluation_rewards = []
        # self.replay_handler = EpisodicExperienceReplayHandler()
        self.replay_handler = ExperienceReplayHandler()

        self.episode_number: int = 0
        self.step_number: int = 0

        self.actor_train_fn = self.critic_model.get_actor_train_fn(self.model, self.options.actor_optimizer)

        self.step_buffer: List[np.ndarray] = []  # list of states

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
                self.step_buffer.append(observation)
                stacked_observation = self.get_stacked_observation()
                action = self.get_action(stacked_observation)

                actions.append(action)
                new_observation, reward, done, info = self.env.step(action)
                new_stacked_observation = self.get_stacked_observation(step_buffer=self.step_buffer + [new_observation])
                logger.debug(f"new_observation: {new_observation}")

                self.replay_handler.record_experience(stacked_observation, action, reward, new_stacked_observation, done)
                try:
                    loss = self.batch_experience_replay()
                    losses.append(loss)
                except InsufficientExperiencesError:
                    logger.warning(f"Insufficient experiences (step number: {self.step_number}), continuing")
                    pass

                if self.step_number % update_target_every == 0:
                    self.update_target_models(update_target_soft, update_target_tau)

                observation = new_observation
                reward_total += reward
                if render:
                    self.env.render()
                self.step_number += 1

            # try:
            #     losses = self.batch_episode_experience_replay()
            # except InsufficientExperiencesError:
            #     losses = []
            #     pass

            if self.tensorboard and losses:
                tf_log(self.tensorboard,
                       ['train_loss', 'reward'],
                       [np.mean(losses), reward_total], i)
            logger.info(f"actions: {np.array2string(np.concatenate(actions), max_line_width=np.inf)}")
            logger.info(f"Episode: {i}, reward_total: {reward_total}")

            reward_totals.append(reward_total)

            if i % evaluate_every == evaluate_every - 1:
                self.evaluate_model(render, i // evaluate_every)
            if i % save_every == save_every - 1:
                self.save_model()

            self.episode_number += 1
        self.reward_totals = reward_totals
        self.save_model()

    def get_stacked_observation(self, step_buffer=None):
        """
        :param step_buffer: Defaults to self.step_buffer
        :return:
        """
        rnn_steps = self.options.rnn_steps
        if step_buffer is None:
            step_buffer = self.step_buffer
        step_buffer_length = len(step_buffer)
        if step_buffer_length >= rnn_steps:
            return np.array(step_buffer[-rnn_steps:]).reshape((1, self.options.rnn_steps, -1))
        else:
            # Fill list with all initial_states ie. [0, 0, 0, 1, 2] where there are only 3 steps in buffer
            observation = [np.zeros_like(step_buffer[0]) for _ in range(rnn_steps)]
            for i, j in enumerate(range(rnn_steps - step_buffer_length, rnn_steps)):
                # i is a counter from 0
                # j is a counter to fill up the last values with the existing values
                observation[j] = step_buffer[i]
        return np.array(observation).reshape((1, self.options.rnn_steps, -1))

    def __________batch_episode_experience_replay(self):
        losses = []
        for episode in self.replay_handler.generator():
            # Pick random step to train on
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_next_states = []
            episode_dones = []
            for step_number in range(len(episode)):
                # General machinery
                states = []
                next_states = []
                for i in reversed(range(self.options.rnn_steps)):
                    experience_index = 0 if step_number < i else step_number - i
                    experience = episode[experience_index]
                    states.append(experience.state)
                    # This gives next state of [0, 0, 1] for state of [0, 0, 0] (and so on)
                    next_state = experience.state if step_number < i else experience.next_state
                    next_states.append(next_state)

                # Converting to np.arrays
                states = np.array(states).reshape(self.get_rnn_shape())
                action = episode[step_number].action
                reward = episode[step_number].reward
                done = episode[step_number].done
                next_states = np.array(next_states).reshape(self.get_rnn_shape())
                # Get targets (refactored to handle different calculation based on done)
                episode_next_states.append(next_states)
                episode_rewards.append(reward)
                episode_dones.append(done)
                episode_states.append(states)
                episode_actions.append(action)

            episode_states = np.squeeze(np.array(episode_states), axis=1)
            episode_next_states = np.squeeze(np.array(episode_next_states), axis=1)
            episode_rewards = np.array(episode_rewards)
            episode_actions = np.array(episode_actions)
            episode_dones = np.array(episode_dones)

            targets = self.get_critic_targets(episode_rewards, episode_next_states, episode_dones)

            states_with_actions = self.critic_model.create_input(episode_states, episode_actions)
            critic_loss = self.critic_model.train_on_batch(states_with_actions, targets)

            # Train actor
            actor_inputs = [episode_states, True]  # True tells model that it's in training mode.
            action_values = self.actor_train_fn(actor_inputs)[0]  # actions not needed for anything.

            losses.append(critic_loss)

        return losses

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
        states = np.array(states).squeeze(axis=1)
        actions = np.array(actions).squeeze(axis=1)
        rewards = np.array(rewards)
        next_states = np.array(next_states).squeeze(axis=1)
        dones = np.array(dones)

        # Train critic
        # target_next_actions = self.target_actor_model.predict_on_batch(next_states)
        #
        # next_states_with_next_actions = self.critic_model.create_input(next_states, target_next_actions)
        # target_q_values = self.target_critic_model.predict_on_batch(next_states_with_next_actions).flatten()
        # gamma = self.hyperparameters.discount_rate(self.episode_number)
        #
        # discounted_next_state_rewards = gamma * target_q_values
        # next_state_rewards_coeff = dones == 0  # Only add the next state reward if not done.
        # targets = rewards + discounted_next_state_rewards * next_state_rewards_coeff

        targets = self.get_critic_targets(rewards, next_states, dones)

        states_with_actions = self.critic_model.create_input(states, actions)
        critic_loss = self.critic_model.train_on_batch(states_with_actions, targets)

        # Train actor
        actor_inputs = [states, True]  # True tells model that it's in training mode.
        action_values = self.actor_train_fn(actor_inputs)[0]  # actions not needed for anything.
        return critic_loss

    def get_critic_targets(self, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray) -> np.ndarray:
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
        targets = rewards.copy().astype(np.float64)

        # Targets for done == False steps calculated with target network
        done_false_indices = dones == 0
        gamma = self.hyperparameters.discount_rate(self.episode_number)

        # Below calculations only concern those where done is false
        _next_states = next_states[done_false_indices]
        target_next_actions = self.target_actor_model.predict_on_batch(_next_states)

        next_states_with_next_actions = self.target_critic_model.create_input(_next_states, target_next_actions)

        target_q_values = self.target_critic_model.predict_on_batch(next_states_with_next_actions).flatten()
        targets[done_false_indices] += gamma * target_q_values
        return targets

    def get_action(self, observation, log_func: Union[logging.debug, logging.info] = logging.debug):
        exploration = self.hyperparameters.exploration_options
        action = self.model.predict(observation).flatten()

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
        step_buffer = []

        while not done:
            step_buffer.append(observation)
            action = self.model.predict(self.get_stacked_observation(step_buffer)).flatten()
            actions.append(action)
            new_observation, reward, done, info = self.env.step(action)

            observation = new_observation
            reward_total += reward
            if render:
                self.env.render()
        if tensorboard_batch_number is not None:
            tf_log(self.evaluation_tensorboard, ['reward'], [reward_total], tensorboard_batch_number)
        logger.info(f"actions: {np.array2string(np.concatenate(actions))}")
        logger.info(f"Evaluation reward: {reward_total}")
        self.evaluation_rewards.append(reward_total)

    def get_rnn_shape(self):
        return 1, self.options.rnn_steps, -1


if __name__ == '__main__':
    from reinforcement_learning.models.lstm_non_stateful_model import LSTMNonStatefulModel
    from reinforcement_learning.models.simple_rnn_model import SimpleRNNModel

    logging.basicConfig(level=logging.INFO)

    from reinforcement_learning.time_sensitive_envs.pendulum_env import PendulumTSEnv

    time_sensitive = False
    exclude_velocity = False
    # env = PendulumTSEnv(time_sensitive=time_sensitive, discrete=True)
    env = PendulumTSEnv(time_sensitive=time_sensitive, exclude_velocity=exclude_velocity)
    inputs = 4 if time_sensitive else 3
    if exclude_velocity:
        inputs -= 1
    outputs = 1
    rnn_steps = 1
    actor_model = SimpleRNNModel(inputs=inputs, outputs=outputs, rnn_steps=rnn_steps, learning_rate=3e-3,
                                 inner_activation='relu', output_activation='linear')
    critic_model = RNNCriticModel(inputs=inputs, outputs=outputs, rnn_steps=rnn_steps)

    EPISODES = 20000

    trainer = DDPGRNNTrainer(
        actor_model, env, critic_model=critic_model,
        hyperparameters=DDPGHyperparameters(
            0.95,
            # ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=0.9, epsilon_decay=0.99,
            #                    limiting_value=0.03)
            OrnsteinUhlenbeck(theta=0.15, sigma=0.3)
        ),
        options=DDPGRNNTrainerOptions(rnn_steps=rnn_steps, render=True)
    )
    trainer.train(episodes=EPISODES)
    logger.info(f"max reward total: {max(trainer.reward_totals)}")
    logger.info(f"last evaluation reward: {trainer.evaluation_rewards[-1]}")
