import logging
from typing import Union

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
from reinforcement_learning.trainers.replay_handlers.exclusive_best_replay_handler import ExclusiveBestReplayHandler

logger = logging.getLogger(__name__)


class EpisodicData:
    def __init__(self):
        self.states: list = None
        self.targets: list = None

    def reset(self):
        """
        Initialises the states and targets attributes
        """
        self.states = []
        self.targets = []

    def record_step_data(self, state, target):
        self.states.append(state)
        self.targets.append(target)

    def get_data_to_train(self):
        """
        Gets the saved data as NumPy arrays to be trained on.
        :return:
        """
        return np.array(self.states), np.array(self.targets)


class QEnv2TrainerEpisodic(BaseTrainer):
    def __init__(self, model: BaseModel, env: BasePseudoEnv, hyperparameters: QLearningHyperparameters,
                 with_tensorboard: bool):
        super().__init__(model, env, hyperparameters, with_tensorboard)
        self.evaluation_tensorboard = None
        self.evaluation_rewards = []
        self.replay_handler = ExclusiveBestReplayHandler()
        self.episodic_data = EpisodicData()

    @log_process(logger, 'training')
    def train(self, episodes: int = 1000, render: bool = False,
              save_every: int = 500000,
              evaluate_every: int = 50,
              replay_every: int = 40):
        exploration = self.hyperparameters.exploration_options

        if self.tensorboard:
            self.tensorboard = create_callback(self.model.model)
        reward_totals = []
        for i in range(episodes):
            logger.info(f"\nEpisode {i}/{episodes}")
            observation = self.env.reset()

            self.episodic_data.reset()
            self.update_learning_rate(i)
            logger.info(f"exploration method: {exploration.method}, value: {exploration.get_value(i)}")

            reward_total = 0
            actions = []
            done = False

            while not done:
                action = self.get_action(observation, i)

                actions.append(action)
                new_observation, reward, done, info = self.step(action)
                logger.debug(f"new_observation: {new_observation}")

                self.record_step_data(observation, action, new_observation, reward, done)

                observation = new_observation
                reward_total += reward
                if render:
                    self.env.render()
            logger.info(f"actions: {actions}")
            logger.info(f"Episode: {i}, reward_total: {reward_total}")

            loss = self.train_on_episode()
            if self.tensorboard:
                tf_log(self.tensorboard,
                       ['train_loss', 'train_mae', 'reward'],
                       [loss[0], loss[1], reward_total], i)

            reward_totals.append(reward_total)

            self.replay_handler.record_protocol(actions, reward_total)

            if i % evaluate_every == evaluate_every - 1:
                self.evaluate_model(render, i // evaluate_every)
            if i % save_every == save_every - 1:
                self.save_model()

            if i % replay_every == replay_every - 1:
                self.replay_trainer(render)

        self.reward_totals = reward_totals

    def step(self, action):
        new_state, reward, done, info = self.env.step(action)
        return new_state, reward, done, info

    def record_step_data(self, state, action: int, new_state, reward: float, done: bool):
        target_vec = self.get_q_values(state)
        if done:
            target = reward
        else:
            gamma = self.hyperparameters.decay_rate
            target = reward + gamma * np.max(self.get_q_values(new_state))
        target_vec[action] = target

        self.episodic_data.record_step_data(state, target_vec)

    def train_on_episode(self):
        states, targets = self.episodic_data.get_data_to_train()
        # print(states, targets)
        loss = self.model.model.train_on_batch(states, targets)
        return loss

    def get_q_values(self, state) -> np.ndarray:
        logger.debug("Get Q values")
        q_values = self.model.model.predict(state.reshape((1, -1)))
        logger.debug(f"Q values {q_values}")
        return q_values[0]

    def get_action(self, observation, i: int,
                   logging_level: Union[logging.debug, logging.info] = logging.info):
        exploration = self.hyperparameters.exploration_options
        if exploration.method == ExplorationMethod.EPSILON:
            if np.random.random() < exploration.get_epsilon(i):
                action = self.env.get_random_action()
                logging_level(f"action: {action} (randomly generated)")
            else:
                q_values = self.get_q_values(observation)
                logging_level(f"q_values: {q_values}")
                action = int(np.argmax(q_values))
                logging_level(f"action: {action} (argmaxed)")
            return action
        elif exploration.method == ExplorationMethod.SOFTMAX:
            q_values = self.get_q_values(observation)
            # exploration: 1, B_RL: 0. exploration: 0, B_RL: infinity
            B_RL = exploration.get_B_RL(i)
            logging_level(f"q_values: {q_values}")
            e_x = np.exp(B_RL * (q_values - np.max(q_values)))
            probabilities = e_x / e_x.sum(axis=0)
            action = int(np.random.choice([_i for _i in range(len(probabilities))], p=probabilities))
            logging_level(f"action: {action} (softmaxed from {probabilities} with B_RL: {B_RL})")
            return action
        else:
            raise ValueError(f"Unknown exploration method: {exploration.method}")

    def replay_trainer(self, render: bool):
        for protocol in self.replay_handler.generator():
            logger.info(f"replay with protocol: {protocol}")
            observation = self.env.reset()
            self.episodic_data.reset()

            reward_total = 0
            for action in protocol:
                new_observation, reward, done, info = self.step(action)
                self.record_step_data(observation, action, new_observation, reward, done)
                if render:
                    self.env.render()
                observation = new_observation

                reward_total += reward

                if done:
                    break
            self.train_on_episode()
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
            new_observation, reward, done, info = self.step(action)

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
        return 3e-3
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
    # from quantum_evolution.envs.q_env_2 import QEnv2
    # env = QEnv2(hamiltonian_datas, t, N=N,
    #             initial_state=initial_state, target_state=target_state)
    # model = DenseModel(inputs=2, outputs=2, layer_nodes=(24, 24), learning_rate=3e-3,
    #                    inner_activation='relu', output_activation='linear')

    # RUN FOR QEnv3
    N = 10
    t = 0.5
    # N = 60
    # t = 3
    from quantum_evolution.envs.q_env_3 import QEnv3
    env = QEnv3(hamiltonian_datas, t, N=N,
                initial_state=initial_state, target_state=target_state)
    model = DenseModel(inputs=3, outputs=2, layer_nodes=(24, 24), learning_rate=3e-3,
                       inner_activation='relu', output_activation='linear')

    # RUN FOR CARTPOLE
    # from reinforcement_learning.envs.cartpole_env import CartPoleTSEnv
    # env = CartPoleTSEnv()
    # model = DenseModel(inputs=5, outputs=2, layer_nodes=(48, 48), learning_rate=3e-3,
    #                    inner_activation='relu', output_activation='linear')

    # RUN FOR ACROBOT
    # from reinforcement_learning.envs.acrobot_env import AcrobotTSEnv
    # env = AcrobotTSEnv(sparse=True)
    # model = DenseModel(inputs=7, outputs=3, layer_nodes=(48, 48, 24), learning_rate=3e-3,
    #                    inner_activation='relu', output_activation='linear')

    EPISODES = 5000
    trainer = QEnv2TrainerEpisodic(
        model, env,
        hyperparameters=QLearningHyperparameters(
            1,
            ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=0.8, epsilon_decay=0.998)
            # ExplorationOptions(method=ExplorationMethod.SOFTMAX, starting_value=0.5, softmax_total_episodes=EPISODES)
        ),
        with_tensorboard=True
    )
    trainer.train(render=False, episodes=EPISODES, replay_every=40)
    logger.info(f"max reward total: {max(trainer.reward_totals)}")
    logger.info(f"last evaluation reward: {trainer.evaluation_rewards[-1]}")
