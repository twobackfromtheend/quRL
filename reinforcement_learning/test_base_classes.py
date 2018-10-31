import unittest

from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.models.dense_model import DenseModel
from reinforcement_learning.trainers.base_trainer import BaseTrainer


class TestClasses(unittest.TestCase):
    def test_dense_model(self):
        with self.assertRaises(NotImplementedError):
            BaseModel()

    def test_base_trainer(self):
        model = DenseModel(5, 5)
        trainer = BaseTrainer(model, None)

        with self.assertRaises(NotImplementedError):
            trainer.train()


if __name__ == '__main__':
    unittest.main()
