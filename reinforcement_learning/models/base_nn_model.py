import time
from typing import Union, Callable

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential, Model, backend as K
from tensorflow.python.keras.models import save_model, load_model

from reinforcement_learning.models.base_model import BaseModel


class BaseNNModel(BaseModel):

    def __init__(self, inputs: int, outputs: int, load_from_filepath: str = None, **kwargs):
        super().__init__(inputs, outputs)
        self.load_from_filepath = load_from_filepath  # Needed for copying with __dict__.
        if load_from_filepath is not None:
            self.model = load_model(load_from_filepath)
        else:
            self.model = self.build_model()

    def build_model(self) -> Union[Sequential, Model]:
        raise NotImplementedError

    def save_model(self, name: str, use_timestamp: bool = True):
        filename = f"model_{name}"
        if use_timestamp:
            filename += f"{time.strftime('%Y%m%d-%H%M%S')}"
        save_model(self.model, filename)

    def get_loss_fn(self, loss_fn_input):
        """
        Handles huber loss input.
        Huber loss with clip delta (e.g. 200) can be specified with huber_CLIPDELTA (e.g. "huber_200")
        :param loss_fn_input:
        :return:
        """
        if isinstance(loss_fn_input, str) and loss_fn_input.startswith('huber_'):
            return self.get_huber_loss(float(loss_fn_input[6:]))
        else:
            return loss_fn_input

    @staticmethod
    def get_huber_loss(clip_delta: float = 200) -> Callable:
        def loss_fn(y_true, y_pred, clip_delta=clip_delta):
            error = y_true - y_pred
            cond = tf.keras.backend.abs(error) < clip_delta

            squared_loss = 0.5 * tf.keras.backend.square(error)
            linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

            return tf.where(cond, squared_loss, linear_loss)

        return loss_fn

    def set_learning_rate(self, learning_rate: float):
        K.set_value(self.model.optimizer.lr, learning_rate)

    def predict(self, x: np.ndarray):
        return self.model.predict(x)

    def train_on_batch(self, x: np.ndarray, y: np.ndarray):
        return self.model.train_on_batch(x, y)
