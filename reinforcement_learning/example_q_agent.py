import logging

import numpy as np
from qutip import sigmax

from quantum_evolution.envs.q_env_1 import QEnv1
from quantum_evolution.simulations.base_simulation import HamiltonianData
from reinforcement_learning.base_agent import BaseAgent
from reinforcement_learning.models.dense_model import DenseModel
from reinforcement_learning.trainers.pseudo_env_trainer import PseudoEnvTrainer, QLearningHyperparameters

AGENT_TO_TEST = BaseAgent
MODEL = DenseModel
TRAINER = PseudoEnvTrainer

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


hamiltonian_data = HamiltonianData(sigmax(), lambda t, args: 0)
N = 20
t_list = np.linspace(0, 2 * np.pi, 200)
ENV = QEnv1([hamiltonian_data], t_list=t_list, N=N)


class ExampleQAgent:
    def __init__(self):
        model = MODEL(
            inputs=3,
            outputs=2 ** N
        )
        trainer = TRAINER(model, ENV, hyperparameters=QLearningHyperparameters(0.95), with_tensorboard=False)
        self.agent = AGENT_TO_TEST(model, trainer)

    def train_agent(self):
        self.agent.trainer.train(render=True)


if __name__ == '__main__':
    example_agent = ExampleQAgent()
    example_agent.train_agent()
