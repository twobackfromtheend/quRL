import logging
from typing import Sequence, Callable

import tensorflow as tf
from tensorflow import keras

from logger_utils.logger_utils import log_process
from reinforcement_learning.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class DenseModel(BaseModel):

    def __init__(self, inputs: int, outputs: int,
                 layer_nodes: Sequence[int] = (24, 24),
                 inner_activation=tf.nn.relu, output_activation='linear',
                 regularizer=keras.regularizers.l2(1e-4),
                 learning_rate=0.003,
                 loss_fn='mse', **kwargs):
        logger.info(f'Creating DenseModel with {inputs} inputs and {outputs} outputs.')
        self.inputs = inputs
        self.outputs = outputs
        self.layer_nodes = layer_nodes
        self.inner_activation = inner_activation
        self.output_activation = output_activation
        self.regularizer = regularizer
        self.learning_rate = learning_rate
        self.loss_fn = self.get_loss_fn(loss_fn)

        super().__init__()

    @log_process(logger, 'building model')
    def build_model(self) -> keras.Sequential:
        model = keras.Sequential()

        # Verbose version needed because https://github.com/tensorflow/tensorflow/issues/22837#issuecomment-428327601
        # model.add(keras.layers.Dense(self.inputs))
        model.add(keras.layers.Dense(input_shape=(self.inputs,), units=self.inputs))

        for _layer_nodes in self.layer_nodes:
            model.add(
                keras.layers.Dense(_layer_nodes, activation=self.inner_activation, kernel_regularizer=self.regularizer)
            )

        model.add(keras.layers.Dense(self.outputs, activation=self.output_activation))
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(loss=self.loss_fn, optimizer=optimizer, metrics=['mae'])
        return model

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
