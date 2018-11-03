import logging

import numpy as np

from logger_utils.logger_utils import log_process
from reinforcement_learning.tensorboard_logger import tf_log, create_callback
from reinforcement_learning.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class QEnv2Trainer(BaseTrainer):
    @log_process(logger, 'training')
    def train(self, episodes: int = 1000, render: bool = False):
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
            done = False
            while not done:
                if np.random.random() < exploration.current_value:
                    action = self.env.get_random_action()
                    logger.info(f"action: {action} (randomly generated)")
                else:
                    action = int(np.argmax(self.get_q_values(observation)))
                    logger.info(f"action: {action} (argmaxed)")

                new_observation, reward, done, info = self.env.step(action)
                logger.debug(f"new_observation: {new_observation}")

                target = reward + gamma * np.max(self.get_q_values(new_observation))
                logger.debug(f"target: {target}")
                target_vec = self.get_q_values(observation)
                logger.debug(f"target_vec: {target_vec}")
                target_vec[action] = target

                loss = self.model.model.train_on_batch(observation.reshape((1, -1)), target_vec.reshape((1, -1)))
                logger.info(f"loss: {loss}")
                losses.append(loss)

                reward_total += reward
                if render:
                    self.env.render()
                # time.sleep(0.05)
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

        self.reward_totals = reward_totals

    def get_q_values(self, state) -> np.ndarray:
        logger.debug("Get Q values")
        q_values = self.model.model.predict(state.reshape((1, -1)))
        logger.debug(f"Q values {q_values}")
        return q_values[0]


if __name__ == '__main__':
    from qutip import *
    from quantum_evolution.envs.q_env_2 import QEnv2
    from quantum_evolution.simulations.base_simulation import HamiltonianData
    from reinforcement_learning.models.dense_model import DenseModel
    from reinforcement_learning.trainers.hyperparameters import QLearningHyperparameters

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
    t = 2
    env = QEnv2(hamiltonian_datas, t, N=N,
                initial_state=initial_state, target_state=target_state)
    model = DenseModel(inputs=2, outputs=2, learning_rate=3e-3)

    trainer = QEnv2Trainer(model, env, hyperparameters=QLearningHyperparameters(0.95), with_tensorboard=True)
    trainer.train(render=True)
