import unittest

from qutip import rand_ket

from quantum_evolution.envs.base_q_env import BaseQEnv
from reinforcement_learning.models.base_nn_model import BaseNNModel
from reinforcement_learning.models.dense_model import DenseModel
from reinforcement_learning.trainers.base_classes.base_options import BaseTrainerOptions
from reinforcement_learning.trainers.base_classes.base_trainer import BaseTrainer
from reinforcement_learning.trainers.base_classes.hyperparameters import QLearningHyperparameters


class TestClasses(unittest.TestCase):
    def test_dense_model(self):
        with self.assertRaises(NotImplementedError):
            BaseNNModel()

    def test_base_pseudo_env(self):
        with self.assertRaises(NotImplementedError):
            pseudo_env = BaseQEnv(rand_ket(2), rand_ket(2))
            pseudo_env.reset()

    def test_base_trainer(self):
        model = DenseModel(5, 5)
        trainer = BaseTrainer(model, None, QLearningHyperparameters(0.8), BaseTrainerOptions())

        with self.assertRaises(NotImplementedError):
            trainer.train(1)


if __name__ == '__main__':
    unittest.main()
