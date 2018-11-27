import logging
from typing import Union

import numpy as np

from logger_utils.logger_utils import log_process
from quantum_evolution.envs.base_q_env import BaseQEnv
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.tensorboard_logger import tf_log, create_callback
from reinforcement_learning.time_sensitive_envs.base_time_sensitive_env import BaseTimeSensitiveEnv
from reinforcement_learning.trainers.base_classes.hyperparameters import QLearningHyperparameters, ExplorationOptions, \
    ExplorationMethod
from reinforcement_learning.trainers.dqn_options import DQNTrainerOptions
from reinforcement_learning.trainers.dqn_trainer import DQNTrainer
from reinforcement_learning.trainers.replay_handlers.episodic_experience_replay_handler import \
    EpisodicExperienceReplayHandler
from reinforcement_learning.trainers.replay_handlers.experience_replay_handler import InsufficientExperiencesError

logger = logging.getLogger(__name__)


class DRQNTrainer(DQNTrainer):
    """
    Performs a gradient update on each episode.
    """

    def __init__(self, model: BaseModel, env: Union[BaseQEnv, BaseTimeSensitiveEnv],
                 hyperparameters: QLearningHyperparameters, options: DQNTrainerOptions):
        super().__init__(model, env, hyperparameters, options)
        self.replay_handler = EpisodicExperienceReplayHandler()

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
            actions = []
            done = False

            while not done:
                action = self.get_action(observation)

                actions.append(action)
                new_observation, reward, done, info = self.env.step(action)
                logger.debug(f"new_observation: {new_observation}")

                self.replay_handler.record_experience(observation, action, reward, new_observation, done)

                if i % update_target_every == 0:
                    self.update_target_model(update_target_soft, update_target_tau)

                observation = new_observation
                reward_total += reward
                if render:
                    self.env.render()
                self.step_number += 1

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

    def batch_episode_experience_replay(self):
        losses = []
        for episode in self.replay_handler.generator():
            # General machinery
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            for experience in episode:
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
            losses.append(loss)
            # TODO: Currently only has last loss.
        return losses


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

    trainer = DRQNTrainer(
        model, env,
        hyperparameters=QLearningHyperparameters(
            0.95,
            ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=1.0, epsilon_decay=0.999,
                               limiting_value=0.1)
            # ExplorationOptions(method=ExplorationMethod.SOFTMAX, starting_value=0.5, softmax_total_episodes=EPISODES)
        ),
        options=DQNTrainerOptions(render=True)
    )
    trainer.train(episodes=EPISODES)
    logger.info(f"max reward total: {max(trainer.reward_totals)}")
    logger.info(f"last evaluation reward: {trainer.evaluation_rewards[-1]}")
