import logging
from typing import Sequence

import tensorflow as tf
from tensorflow import keras

from logger_utils.logger_utils import log_process
from reinforcement_learning.models.base_nn_model import BaseNNModel

logger = logging.getLogger(__name__)


class DenseModel(BaseNNModel):

    def __init__(self, inputs: int, outputs: int,
                 layer_nodes: Sequence[int] = (24, 24),
                 inner_activation=tf.nn.relu, output_activation='linear',
                 regularizer=keras.regularizers.l2(1e-4),
                 learning_rate=0.003,
                 loss_fn='mse', **kwargs):
        logger.info(f'Creating DenseModel with {inputs} inputs and {outputs} outputs.')
        self.layer_nodes = layer_nodes
        self.inner_activation = inner_activation
        self.output_activation = output_activation
        self.regularizer = regularizer
        self.learning_rate = learning_rate
        self.loss_fn = self.get_loss_fn(loss_fn)

        super().__init__(inputs, outputs)

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
