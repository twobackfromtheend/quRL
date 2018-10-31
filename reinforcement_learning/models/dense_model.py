import logging
from typing import Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Sequential

from logger_utils.logger_utils import log_process
from reinforcement_learning.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class DenseModel(BaseModel):

    def __init__(self, inputs: int, outputs: int,
                 layer_nodes: Sequence[int] = (64, 64, 64),
                 inner_activation=tf.nn.relu, output_activation=tf.nn.sigmoid):
        logger.info(f'Creating DenseModel with {inputs} inputs and {outputs} outputs.')
        self.inputs = inputs
        self.outputs = outputs
        self.layer_nodes = layer_nodes
        self.inner_activation = inner_activation
        self.output_activation = output_activation
        super().__init__()

    @log_process(logger, 'building model')
    def build_model(self) -> Sequential:
        model = keras.Sequential()

        model.add(keras.layers.Dense(self.inputs))

        for _layer_nodes in self.layer_nodes:
            model.add(keras.layers.Dense(_layer_nodes, activation=self.inner_activation))

        model.add(keras.layers.Dense(self.outputs, activation=self.output_activation))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        return model
