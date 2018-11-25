from enum import Enum

from reinforcement_learning.models.dense_model import DenseModel


class ModelPreset(Enum):
    DEFAULT = DenseModel
    DENSE_MODEL_MSE = lambda *args, **kwargs: DenseModel(*args, **kwargs, loss_fn='mse')
    DENSE_MODEL_HUBER_200 = lambda *args, **kwargs: DenseModel(*args, **kwargs, loss_fn='huber_200')




