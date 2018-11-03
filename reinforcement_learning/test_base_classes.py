import unittest

from qutip import rand_ket

from quantum_evolution.envs.base_pseudo_env import BasePseudoEnv
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.models.dense_model import DenseModel
from reinforcement_learning.trainers.base_trainer import BaseTrainer
from reinforcement_learning.trainers.hyperparameters import QLearningHyperparameters


class TestClasses(unittest.TestCase):
    def test_dense_model(self):
        with self.assertRaises(NotImplementedError):
            BaseModel()

    def test_base_pseudo_env(self):
        with self.assertRaises(NotImplementedError):
            pseudo_env = BasePseudoEnv(rand_ket(2), rand_ket(2))
            pseudo_env.reset()

    def test_base_trainer(self):
        model = DenseModel(5, 5)
        trainer = BaseTrainer(model, None, QLearningHyperparameters(0.8), False)

        with self.assertRaises(NotImplementedError):
            trainer.train()


if __name__ == '__main__':
    unittest.main()
