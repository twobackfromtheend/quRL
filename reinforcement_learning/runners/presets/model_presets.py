from enum import Enum
from functools import partial

from reinforcement_learning.models.dense_model import DenseModel


class ModelPreset(Enum):
    DEFAULT = DenseModel
    DENSE_MODEL_MSE = partial(DenseModel, loss_fn='mse')
    DENSE_MODEL_LR_1_4 = partial(DenseModel, learning_rate=1e-4)
    DENSE_MODEL_HUBER_200 = partial(DenseModel, loss_fn='huber_200')




