import logging
import math
from typing import List

import numpy as np
from qutip.solver import Result
from tensorflow.python.keras import backend as K

from logger_utils.logger_utils import log_process
from quantum_evolution.envs.base_pseudo_env import BasePseudoEnv
from quantum_evolution.plotter.bloch_animator import BlochAnimator
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.tensorboard_logger import tf_log, create_callback
from reinforcement_learning.trainers.base_trainer import BaseTrainer
from reinforcement_learning.trainers.hyperparameters import QLearningHyperparameters, ExplorationOptions, \
    ExplorationMethod
from reinforcement_learning.trainers.replay_handler import ReplayHandler

logger = logging.getLogger(__name__)


class QEnv2Trainer(BaseTrainer):
    def __init__(self, model: BaseModel, env: BasePseudoEnv, hyperparameters: QLearningHyperparameters,
                 with_tensorboard: bool):
        super().__init__(model, env, hyperparameters, with_tensorboard)
        self.evaluation_tensorboard = None
        self.evaluation_rewards = []

        self.replay_handler = ReplayHandler()

    @log_process(logger, 'training')
    def train(self, episodes: int = 1000, render: bool = False,
              save_every: int = 500,
              evaluate_every: int = 50,
              replay_every: int = 40):
        exploration = self.hyperparameters.exploration_options

        if self.tensorboard:
            self.tensorboard = create_callback(self.model.model)
        reward_totals = []
        for i in range(episodes):
            logger.info(f"\nEpisode {i}/{episodes}")
            observation = self.env.reset()
            exploration.decay_current_value()
            logger.info(f"exploration: {exploration.current_value}")
            self.update_learning_rate(i)

            reward_total = 0
            losses = []
            actions = []
            done = False
            while not done:
                action = self.get_action(observation)

                actions.append(action)
                new_observation, reward, done, info = self.env.step(action)
                logger.debug(f"new_observation: {new_observation}")

                loss = self.train_on_step(observation, action, new_observation, reward, done)
                logger.debug(f"loss: {loss}")
                losses.append(loss)

                observation = new_observation
                reward_total += reward
                if render:
                    self.env.render()
                # time.sleep(0.05)
            logger.info(f"actions: {actions}")
            logger.info(f"Episode: {i}, reward_total: {reward_total}")
            if self.tensorboard:
                train_loss = 0
                train_mae = 0
                for loss in losses:
                    train_loss += loss[0]
                    train_mae += loss[1]
                tf_log(self.tensorboard,
                       ['train_loss', 'train_mae', 'reward'],
                       [train_loss, train_mae, reward_total], i)

            reward_totals.append(reward_total)

            self.replay_handler.record_protocol(actions, reward_total)

            if i % evaluate_every == evaluate_every - 1:
                self.evaluate_model(render, i // evaluate_every)
            if i % save_every == save_every - 1:
                self.save_model()

            if i % replay_every == replay_every - 1:
                self.replay_trainer(render)

        self.reward_totals = reward_totals

    def train_on_step(self, observation, action: int, new_observation, reward: float, done: bool) -> List[float]:
        """
        Trains model on a single step.
        :param observation:
        :param action:
        :param new_observation:
        :param reward:
        :return: train_loss, train_mae (List[float])
        """
        if done:
            target = reward
            # target = -100  # ONLY FOR CARTPOLE
        else:
            gamma = self.hyperparameters.decay_rate
            target = reward + gamma * np.max(self.get_q_values(new_observation))
        logger.debug(f"target: {target}")
        target_vec = self.get_q_values(observation)
        logger.debug(f"original_target_vec: {target_vec}")
        # print(target_vec, target, action)
        target_vec[action] = target
        logger.debug(f"target_vec: {target_vec}")
        loss = self.model.model.train_on_batch(observation.reshape((1, -1)), target_vec.reshape((1, -1)))
        return loss

    def get_q_values(self, state) -> np.ndarray:
        logger.debug("Get Q values")
        q_values = self.model.model.predict(state.reshape((1, -1)))
        logger.debug(f"Q values {q_values}")
        return q_values[0]

    def get_action(self, observation):
        exploration = self.hyperparameters.exploration_options
        if exploration.method == ExplorationMethod.EPSILON:
            if np.random.random() < exploration.current_value:
                try:
                    action = self.env.get_random_action()
                except AttributeError:
                    # Gym env.
                    action = self.env.action_space.sample()
                logger.debug(f"action: {action} (randomly generated)")
            else:
                action = int(np.argmax(self.get_q_values(observation)))
                logger.debug(f"action: {action} (argmaxed)")
            return action
        elif exploration.method == ExplorationMethod.SOFTMAX:
            q_values = self.get_q_values(observation)
            # exploration: 1, B_RL: 0. exploration: 0, B_RL: infinity
            B_RL = (1 - exploration.current_value) / exploration.current_value
            logger.info(f"q_values: {q_values}")
            e_x = np.exp(B_RL * (q_values - np.max(q_values)))
            probabilities = e_x / e_x.sum(axis=0)
            action = int(np.random.choice([0, 1], p=probabilities))
            logger.info(f"action: {action} (softmaxed from {probabilities} with B_RL: {B_RL})")
            return action
        else:
            raise ValueError(f"Unknown exploration method: {exploration.method}")

    def replay_trainer(self, render: bool):
        for protocol in self.replay_handler.generator():
            logger.info(f"replay with protocol: {protocol}")
            observation = self.env.reset()
            reward_total = 0
            for action in protocol:
                new_observation, reward, done, info = self.env.step(action)
                loss = self.train_on_step(observation, action, new_observation, reward, done)
                if render:
                    self.env.render()
                observation = new_observation

                reward_total += reward

                if done:
                    break

            logger.info(f"replay reward: {reward_total}")

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
        # states = []
        # result: Result = None  # Instantiate here to avoid linting bringing up that it might not exist below.
        while not done:
            action = int(np.argmax(self.get_q_values(observation)))
            actions.append(action)
            new_observation, reward, done, info = self.env.step(action)

            observation = new_observation
            # result = self.env.simulation.result
            # states += result.states
            reward_total += reward
            if render:
                self.env.render()
        if tensorboard_batch_number is not None:
            tf_log(self.evaluation_tensorboard, ['reward'], [reward_total], tensorboard_batch_number)
        logger.info(f"actions: {actions}")
        logger.info(f"Evaluation reward: {reward_total}")
        self.evaluation_rewards.append(reward_total)

        # result.states = states
        # self.save_animation(result, tensorboard_batch_number)

    @log_process(logger, "saving evaluation animation")
    def save_animation(self, result: Result, i: int):
        bloch_animation = BlochAnimator([result], static_states=[self.env.target_state])
        bloch_animation.generate_animation()
        # TODO: Make work
        # bloch_animation.show()
        # bloch_animation.save(filename=f"evaluation_{i}.mp4")

    def update_learning_rate(self, i: int):
        current_learning_rate = float(K.get_value(self.model.model.optimizer.lr))
        logger.info(f"learning rate: {current_learning_rate}")
        new_learning_rate = self.get_learning_rate(i)
        K.set_value(self.model.model.optimizer.lr, new_learning_rate)

    @staticmethod
    def get_learning_rate(i: int) -> float:
        # return 8e-3
        return ((math.cos(i / 100) + 1.000) / 2 * 3 * 10 ** -3) * math.e ** -(i / 1000) + 3 * 10 ** -5


if __name__ == '__main__':
    from qutip import *
    from quantum_evolution.envs.q_env_2 import QEnv2
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
    N = 10
    t = 0.5
    env = QEnv2(hamiltonian_datas, t, N=N,
                initial_state=initial_state, target_state=target_state)
    model = DenseModel(inputs=2, outputs=2, layer_nodes=(48, 48), learning_rate=3e-3,
                       inner_activation='relu', output_activation='linear')

    # RUN FOR CARTPOLE
    # Uncomment the target above in train_on_step()
    # import gym
    # env = gym.make('CartPole-v0')
    # model = DenseModel(inputs=4, outputs=2, layer_nodes=(48, 48), learning_rate=3e-3,
    #                    inner_activation='relu', output_activation='linear')

    trainer = QEnv2Trainer(
        model, env,
        hyperparameters=QLearningHyperparameters(
            0.95,
            # ExplorationOptions(0.8, 0.992, min_value=0.04)  # Prob. of at least 1 randomised is 56% for N = 40?.
            # ExplorationOptions(0.8, 0.995, min_value=0.04, method=ExplorationMethod.EPSILON)
            ExplorationOptions(0.5, 0.992, min_value=0.04, method=ExplorationMethod.SOFTMAX)  # B_RL = 999
            # https://www.wolframalpha.com/input/?i=EXP(-0.2+*+(1+-+0.8*0.992%5Ex)+%2F+(0.8*0.992%5Ex))+%2F+(1+%2B+EXP(-0.2+*+(1+-+0.8*0.992%5Ex)+%2F+(0.8*0.992%5Ex)))++for+x+from+0+to+1000
            # For Q-values of 0.4 and 0.6, 0.04 will give 0.8%
        ),
        with_tensorboard=True
    )
    trainer.train(render=True, episodes=1000)
    logger.info(f"max reward total: {max(trainer.reward_totals)}")
    logger.info(f"last evaluation reward: {trainer.evaluation_rewards[-1]}")
