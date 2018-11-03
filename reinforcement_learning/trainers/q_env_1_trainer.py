import logging

import numpy as np

from logger_utils.logger_utils import log_process
from reinforcement_learning.tensorboard_logger import tf_log
from reinforcement_learning.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class QEnv1Trainer(BaseTrainer):
    @log_process(logger, 'training')
    def train(self, episodes: int = 1000, render: bool = False):
        exploration = self.hyperparameters.exploration_options
        gamma = self.hyperparameters.decay_rate

        reward_totals = []
        for i in range(episodes):
            logger.info(f"Episode {i}/{episodes}")
            observation = self.env.reset()
            exploration.decay_current_value()
            logger.info(f"exploration: {exploration.current_value}")

            action = int(np.argmax(self.get_q_values(observation)))
            logger.debug(f"original action  : {self.env.convert_int_to_bit_list(action, self.env.N)}")
            action = self.env.randomise_action(action, exploration.current_value)
            logger.debug(f"randomised action: {self.env.convert_int_to_bit_list(action, self.env.N)}")

            new_observation, reward, done, info = self.env.step(action)
            logger.debug(f"new_observation: {new_observation}")

            # https://keon.io/deep-q-learning/
            target = reward + gamma * np.max(self.get_q_values(new_observation))

            logger.debug(f"target: {target}")
            target_vec = self.get_q_values(observation)
            logger.debug(f"target_vec: {target_vec}")
            target_vec[action] = target

            loss = self.model.model.train_on_batch(observation.reshape((1, -1)), target_vec.reshape((1, -1)))
            logger.info(f"loss: {loss}")
            if self.tensorboard:
                tf_log(self.tensorboard, ['train_loss', 'train_mae', 'reward'], [loss[0], loss[1], reward], i)

            logger.info(f"Episode: {i}, reward: {reward}")
            reward_totals.append(reward)

            if render and (i == 50 or i % 400 == 0):
                self.env.render()

        self.reward_totals = reward_totals

    def get_q_values(self, state) -> np.ndarray:
        logger.debug("Get Q values")
        q_values = self.model.model.predict(state.reshape((1, -1)))
        logger.debug(f"Q values {q_values}")
        return q_values[0]


if __name__ == '__main__':
    from qutip import *
    from quantum_evolution.envs.q_env_1 import QEnv1
    from quantum_evolution.simulations.base_simulation import HamiltonianData
    from reinforcement_learning.models.dense_model import DenseModel
    from reinforcement_learning.trainers.hyperparameters import QLearningHyperparameters

    logger = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG)

    initial_state = (-sigmaz() + 2 * sigmax()).groundstate()[1]
    target_state = (-sigmaz() - 2 * sigmax()).groundstate()[1]


    def placeholder_callback(t, args):
        raise RuntimeError


    hamiltonian_datas = [
        HamiltonianData(-sigmaz()),
        HamiltonianData(sigmax(), placeholder_callback)
    ]
    N = 20
    t_list = np.linspace(0, 3, 200)
    env = QEnv1(hamiltonian_datas, t_list=t_list, N=N,
                initial_state=initial_state, target_state=target_state)
    model = DenseModel(inputs=3, outputs=2 ** N, learning_rate=3e-2)

    trainer = QEnv1Trainer(model, env, hyperparameters=QLearningHyperparameters(0.01), with_tensorboard=True)
    trainer.train(render=True)
