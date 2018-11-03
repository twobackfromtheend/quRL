import logging

import numpy as np

from logger_utils.logger_utils import log_process
from quantum_evolution.envs.base_pseudo_env import BasePseudoEnv
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.tensorboard_logger import tf_log, create_callback
from reinforcement_learning.trainers.base_trainer import BaseTrainer
from reinforcement_learning.trainers.hyperparameters import QLearningHyperparameters, ExplorationOptions

logger = logging.getLogger(__name__)


class QEnv2Trainer(BaseTrainer):
    def __init__(self, model: BaseModel, env: BasePseudoEnv, hyperparameters: QLearningHyperparameters,
                 with_tensorboard: bool):
        super().__init__(model, env, hyperparameters, with_tensorboard)
        self.evaluation_tensorboard = None

    @log_process(logger, 'training')
    def train(self, episodes: int = 1000, render: bool = False, save_every: int = 200, evaluate_every: int = 50):
        exploration = self.hyperparameters.exploration_options
        gamma = self.hyperparameters.decay_rate

        if self.tensorboard:
            self.tensorboard = create_callback(self.model.model)
        reward_totals = []
        for i in range(episodes):
            logger.info(f"Episode {i}/{episodes}")
            observation = self.env.reset()
            exploration.decay_current_value()
            logger.info(f"exploration: {exploration.current_value}")
            reward_total = 0
            losses = []
            actions = []
            done = False
            while not done:
                if np.random.random() < exploration.current_value:
                    action = self.env.get_random_action()
                    logger.debug(f"action: {action} (randomly generated)")
                else:
                    action = int(np.argmax(self.get_q_values(observation)))
                    logger.debug(f"action: {action} (argmaxed)")
                actions.append(action)
                new_observation, reward, done, info = self.env.step(action)
                logger.debug(f"new_observation: {new_observation}")

                target = reward + gamma * np.max(self.get_q_values(new_observation))
                logger.debug(f"target: {target}")
                target_vec = self.get_q_values(observation)
                logger.debug(f"target_vec: {target_vec}")
                target_vec[action] = target

                loss = self.model.model.train_on_batch(observation.reshape((1, -1)), target_vec.reshape((1, -1)))
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

            if i % evaluate_every == 1:
                self.evaluate_model(render, i // evaluate_every)
            if i >= save_every and i % save_every == 0:
                self.save_model()

        self.reward_totals = reward_totals

    def get_q_values(self, state) -> np.ndarray:
        logger.debug("Get Q values")
        q_values = self.model.model.predict(state.reshape((1, -1)))
        logger.debug(f"Q values {q_values}")
        return q_values[0]

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
        HamiltonianData(sigmax(), placeholder_callback)
    ]
    N = 40
    t = 0.4
    env = QEnv2(hamiltonian_datas, t, N=N,
                initial_state=initial_state, target_state=target_state)
    model = DenseModel(inputs=2, outputs=2, learning_rate=3e-3)

    trainer = QEnv2Trainer(
        model, env,
        hyperparameters=QLearningHyperparameters(
            0.95,
            ExplorationOptions(0.8, 0.992, min_value=0.04)  # Prob. of at least 1 randomised is 56%.
        ),
        with_tensorboard=True
    )
    trainer.train(render=True)
