import logging

from tensorflow.python.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

from logger_utils.logger_utils import log_process
from reinforcement_learning.models.base_nn_model import BaseNNModel

logger = logging.getLogger(__name__)


class BaseCriticModel(BaseNNModel):
    def __init__(self, inputs: int, outputs: int, load_from_filepath: str = None, **kwargs):
        self.action_input: Input = None

        super().__init__(inputs, outputs, load_from_filepath, **kwargs)

    @log_process(logger, 'building model')
    def build_model(self) -> Model:
        self.action_input = Input(shape=(self.outputs,), name='action_input')
        observation_input = Input(shape=(self.inputs,), name='observation_input')
        x = Concatenate()([self.action_input, observation_input])

        x = Flatten()(x)

        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)

        x = Dense(1, activation='linear')(x)

        model = Model(inputs=[self.action_input, observation_input], outputs=x)

        model.compile(optimizer=Adam(lr=1e-3), loss='mse')
        return model
