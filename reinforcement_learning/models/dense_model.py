import logging
from typing import Sequence

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
                 learning_rate=0.003, **kwargs):
        logger.info(f'Creating DenseModel with {inputs} inputs and {outputs} outputs.')
        self.inputs = inputs
        self.outputs = outputs
        self.layer_nodes = layer_nodes
        self.inner_activation = inner_activation
        self.output_activation = output_activation
        self.regularizer = regularizer
        self.learning_rate = learning_rate

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
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        return model
