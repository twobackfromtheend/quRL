import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import SimpleRNN, TimeDistributed

from logger_utils.logger_utils import log_process
from reinforcement_learning.models.base_nn_model import BaseNNModel
from reinforcement_learning.models.utils import get_LSTM_layer

logger = logging.getLogger(__name__)


class SimpleRNNModel(BaseNNModel):
    """
    Is not limited to a fixed batch_size.
    Downside being that steps have to be repeated in memory
    (ie [[0, 1, 2], [1, 2, 3], [2, 3, 4]])

    May not lead to increased memory usage if properly implemented as pointers perhaps?
    """
    def __init__(self, inputs: int, outputs: int, rnn_steps: int,
                 inner_activation=tf.nn.relu, output_activation='linear',
                 learning_rate=0.001,
                 loss_fn='mse', **kwargs):
        logger.info(f'Creating SimpleRNNModel with {inputs} inputs and {outputs} outputs.')
        self.rnn_steps = rnn_steps
        self.inner_activation = inner_activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.loss_fn = self.get_loss_fn(loss_fn)

        super().__init__(inputs, outputs)

    @log_process(logger, 'building model')
    def build_model(self) -> keras.Sequential:
        model = keras.Sequential()
        input_shape = (self.rnn_steps, self.inputs)
        model.add(SimpleRNN(128, input_shape=input_shape))

        # model.add(Dense(24, input_shape=input_shape, activation='relu'))
        # # model.add(Input(shape=input_shape))
        model.add(Flatten())

        # model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.outputs, activation=self.output_activation))
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(loss=self.loss_fn, optimizer=optimizer, metrics=['mae'])
        print(model.summary())
        return model
