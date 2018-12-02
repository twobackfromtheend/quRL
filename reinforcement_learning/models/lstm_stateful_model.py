import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

from logger_utils.logger_utils import log_process
from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.models.utils import get_LSTM_layer

logger = logging.getLogger(__name__)

LSTM_LAYER = get_LSTM_layer()


class LSTMStatefulModel(BaseModel):
    """
    Limited to a fixed batch size due to Keras, as batch_size is required param when creating network.

    Currently fixed to batch_size = 1 to allow use as policy network (where you pass in 1 state and expect 1 action).
    Possible future workaround might involve creating a separate (policy) network that copies weights,
    with batch_size = 1.

    Current implementation might be wrong as subsequent batches (maybe) should not overlap
    (ie should be [state0], then [state1], then [state2] as opposed to [0, 1, 2], [1, 2, 3], [2, 3, 4]
    as the latter involves repeating states)
    """

    def __init__(self, inputs: int, outputs: int,
                 inner_activation=tf.nn.relu, output_activation='linear',
                 learning_rate=0.003,
                 loss_fn='mse', **kwargs):
        logger.info(f'Creating LSTMStatefulModel with {inputs} inputs and {outputs} outputs.')
        self.inputs = inputs
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
        batch_input_shape = (batch_size, 1, self.inputs)
        model.add(LSTM_LAYER(24, batch_input_shape=batch_input_shape, stateful=True, return_sequences=True))
        model.add(LSTM_LAYER(24, batch_input_shape=batch_input_shape, stateful=True))

        model.add(Dense(self.outputs, activation=self.output_activation))
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(loss=self.loss_fn, optimizer=optimizer, metrics=['mae'])
        return model
