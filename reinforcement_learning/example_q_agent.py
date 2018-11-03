import logging

import numpy as np
from qutip import sigmax, sigmaz

from quantum_evolution.envs.q_env_1 import QEnv1
from quantum_evolution.simulations.base_simulation import HamiltonianData
from reinforcement_learning.base_agent import BaseAgent
from reinforcement_learning.models.dense_model import DenseModel
from reinforcement_learning.trainers.hyperparameters import QLearningHyperparameters
from reinforcement_learning.trainers.pseudo_env_trainer import PseudoEnvTrainer

AGENT_TO_TEST = BaseAgent
MODEL = DenseModel
TRAINER = PseudoEnvTrainer

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
ENV = QEnv1(hamiltonian_datas, t_list=t_list, N=N,
            initial_state=initial_state, target_state=target_state)


class ExampleQAgent:
    def __init__(self):
        model = MODEL(
            inputs=3,
            outputs=2 ** N,
            learning_rate=3e-2
        )
        trainer = TRAINER(model, ENV, hyperparameters=QLearningHyperparameters(0.01), with_tensorboard=True)
        self.agent = AGENT_TO_TEST(model, trainer)

    def train_agent(self):
        self.agent.trainer.train(episodes=10000, render=True)


if __name__ == '__main__':
    example_agent = ExampleQAgent()
    example_agent.train_agent()
