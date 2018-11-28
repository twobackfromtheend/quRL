import logging
from typing import Union, List

import numpy as np

from logger_utils.logger_utils import log_process
from quantum_evolution.envs.base_q_env import BaseQEnv
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.tensorboard_logger import tf_log, create_callback
from reinforcement_learning.time_sensitive_envs.base_time_sensitive_env import BaseTimeSensitiveEnv
from reinforcement_learning.trainers.base_classes.hyperparameters import QLearningHyperparameters, ExplorationOptions, \
    ExplorationMethod
from reinforcement_learning.trainers.dqn_trainer import DQNTrainer
from reinforcement_learning.trainers.drqn_options import DRQNTrainerOptions
from reinforcement_learning.trainers.replay_handlers.episodic_experience_replay_handler import \
    EpisodicExperienceReplayHandler
from reinforcement_learning.trainers.replay_handlers.experience_replay_handler import InsufficientExperiencesError

logger = logging.getLogger(__name__)


class DRQNBatchedTrainer(DQNTrainer):
    """
    Required as different model is needed due to Keras inflexibility (need to specify batch size in model creation)
    """

    def __init__(self, model: BaseModel, env: Union[BaseQEnv, BaseTimeSensitiveEnv],
                 hyperparameters: QLearningHyperparameters, options: DRQNTrainerOptions):
        super().__init__(model, env, hyperparameters, options)
        self.replay_handler = EpisodicExperienceReplayHandler()
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
            self.step_buffer = []
            state = self.env.reset()
            if learning_rate_override:
                self.update_learning_rate(learning_rate_override(i))
            logger.info(f"exploration method: {exploration.method}, value: {exploration.get_value(i)}")

            reward_total = 0
            actions = []
            done = False

            while not done:
                self.step_buffer.append(state)
                action = self.get_action(self.get_observation())

                actions.append(action)
                new_state, reward, done, info = self.env.step(action)
                logger.debug(f"new_state: {new_state}")

                self.replay_handler.record_experience(state, action, reward, new_state, done)

                if self.step_number % update_target_every == 0:
                    self.update_target_model(update_target_soft, update_target_tau)

                state = new_state
                reward_total += reward
                if render:
                    self.env.render()
                self.step_number += 1

            self.reset_model_state()

            try:
                losses = self.batch_episode_experience_replay()
            except InsufficientExperiencesError:
                losses = []
                pass

            if self.tensorboard:
                if losses:
                    losses = list(zip(*losses))
                    tf_log(self.tensorboard,
                           ['train_loss', 'train_mae', 'reward'],
                           [np.mean(losses[0]), np.mean(losses[1]), reward_total], i)
                else:
                    tf_log(self.tensorboard, ['reward'], [reward_total], i)
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
        """
        :param step_buffer: Defaults to self.step_buffer
        :return:
        """
        rnn_steps = self.options.rnn_steps
        if step_buffer is None:
            step_buffer = self.step_buffer
        step_buffer_length = len(step_buffer)
        if step_buffer_length >= rnn_steps:
            return np.array(step_buffer[-rnn_steps:])
        else:
            # Fill list with all initial_states ie. [0, 0, 0, 1, 2] where there are only 3 steps in buffer
            observation = [step_buffer[0] for _ in range(rnn_steps)]
            for i, j in enumerate(range(rnn_steps - step_buffer_length, rnn_steps)):
                # i is a counter from 0
                # j is a counter to fill up the last values with the existing values
                observation[j] = step_buffer[i]
            return np.array(observation)

    def batch_episode_experience_replay(self):
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
            episode_dones = np.array(episode_dones)

            targets = self.get_targets(episode_rewards, episode_next_states, episode_dones)
            target_vecs = self.target_model.model.predict(episode_states)
            for i, action in enumerate(episode_actions):
                # Set target values in target_vecs
                target_vecs[i, action] = targets[i]

            loss = self.model.model.train_on_batch(episode_states, target_vecs)
            losses.append(loss)

            self.reset_model_state()

        return losses

    def get_action(self, observation, log_func: Union[logging.debug, logging.info] = logging.debug):
        if len(self.step_buffer) < self.options.rnn_steps:
            return self.env.get_random_action()

        return super().get_action(observation, log_func)

    def get_policy_q_values(self, state) -> np.ndarray:
        logger.debug("Get policy Q values")
        q_values = self.model.model.predict(state.reshape(self.get_rnn_shape()))
        logger.debug(f"Q values {q_values}")
        return q_values[0]

    @log_process(logger, "evaluating model")
    def evaluate_model(self, render, tensorboard_batch_number: int = None):
        if self.evaluation_tensorboard is None and tensorboard_batch_number is not None:
            self.evaluation_tensorboard = create_callback(self.model.model)
        done = False
        reward_total = 0
        state = self.env.reset()
        actions = []
        step_buffer = []
        while not done:
            step_buffer.append(state)
            observation = self.get_observation(step_buffer)
            action = int(np.argmax(self.get_policy_q_values(observation)))
            actions.append(action)
            new_state, reward, done, info = self.env.step(action)

            state = new_state
            reward_total += reward
            if render:
                self.env.render()
        if tensorboard_batch_number is not None:
            tf_log(self.evaluation_tensorboard, ['reward'], [reward_total], tensorboard_batch_number)
        logger.info(f"actions: {actions}")
        logger.info(f"Evaluation reward: {reward_total}")
        self.evaluation_rewards.append(reward_total)

    def get_rnn_shape(self):
        return 1, self.options.rnn_steps, -1

    def reset_model_state(self):
        self.model.model.reset_states()


if __name__ == '__main__':
    from reinforcement_learning.models.lstm_non_stateful_model import LSTMNonStatefulModel

    logging.basicConfig(level=logging.INFO)

    from reinforcement_learning.time_sensitive_envs.cartpole_env import CartPoleTSEnv

    time_sensitive = False
    env = CartPoleTSEnv(time_sensitive=time_sensitive)
    inputs = 5 if time_sensitive else 4
    rnn_steps = 3
    model = LSTMNonStatefulModel(inputs=inputs, outputs=2, rnn_steps=rnn_steps, learning_rate=3e-3,
                                 inner_activation='relu', output_activation='linear')

    EPISODES = 20000

    trainer = DRQNBatchedTrainer(
        model, env,
        hyperparameters=QLearningHyperparameters(
            0.95,
            ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=0.5, epsilon_decay=0.999,
                               limiting_value=0.1)
            # ExplorationOptions(method=ExplorationMethod.SOFTMAX, starting_value=0.5, softmax_total_episodes=EPISODES)
        ),
        options=DRQNTrainerOptions(rnn_steps=rnn_steps, update_target_soft=False, render=True)
    )
    trainer.train(episodes=EPISODES)
    logger.info(f"max reward total: {max(trainer.reward_totals)}")
    logger.info(f"last evaluation reward: {trainer.evaluation_rewards[-1]}")
