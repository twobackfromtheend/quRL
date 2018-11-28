import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, TimeDistributed, LSTM, CuDNNLSTM

from logger_utils.logger_utils import log_process
from reinforcement_learning.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):

    def __init__(self, inputs: int, outputs: int, rnn_steps: int,
                 inner_activation=tf.nn.relu, output_activation='linear',
                 learning_rate=0.003,
                 loss_fn='mse', **kwargs):
        logger.info(f'Creating LSTMModel with {inputs} inputs and {outputs} outputs.')
        self.inputs = inputs
        self.rnn_steps = rnn_steps
        self.outputs = outputs
        self.inner_activation = inner_activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.loss_fn = self.get_loss_fn(loss_fn)

        super().__init__()

    @log_process(logger, 'building model')
    def build_model(self) -> keras.Sequential:
        batch_size = 1

        model = keras.Sequential()
        batch_input_shape = (batch_size, self.rnn_steps, self.inputs)
        model.add(CuDNNLSTM(24, batch_input_shape=batch_input_shape, stateful=True, return_sequences=True))
        model.add(CuDNNLSTM(24, batch_input_shape=batch_input_shape, stateful=True))

        model.add(Dense(self.outputs))
        #
        # model.add(LSTM(48))

        # model.add(Dense(self.outputs, activation=self.output_activation))
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(loss=self.loss_fn, optimizer=optimizer, metrics=['mae'])
        return model
