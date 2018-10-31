import logging
import unittest

import gym

from reinforcement_learning.base_agent import BaseAgent
from reinforcement_learning.models.dense_model import DenseModel
from reinforcement_learning.trainers.q_trainer import QTrainer, QLearningHyperparameters

AGENT_TO_TEST = BaseAgent
MODEL = DenseModel
TRAINER = QTrainer
ENV = gym.make('CartPole-v0')

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class TestAgent:
    def __init__(self):
        model = MODEL(
            inputs=ENV.observation_space.shape[0],
            outputs=ENV.action_space.n
        )
        trainer = TRAINER(model, ENV, hyperparameters=QLearningHyperparameters(0.95), with_tensorboard=True)
        self.agent = AGENT_TO_TEST(model, trainer)

    def train_agent(self):
        self.agent.trainer.train()


if __name__ == '__main__':
    test_agent = TestAgent()
    test_agent.train_agent()
