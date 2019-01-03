import logging

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import Function
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam, Optimizer

from logger_utils.logger_utils import log_process
from reinforcement_learning.models.base_nn_model import BaseNNModel

logger = logging.getLogger(__name__)


class BaseCriticModel(BaseNNModel):
    def __init__(self, inputs: int, outputs: int, load_from_filepath: str = None, **kwargs):
        self.action_input: Input = None

        super().__init__(inputs, outputs, load_from_filepath, **kwargs)

        # After build_model is called and self.model is set.
        self.action_input_index = self.model.input.index(self.action_input)

    @log_process(logger, 'building BaseCriticModel')
    def build_model(self) -> Model:
        inputs, x = self._get_input_layers()

        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)

        x = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=x)

        model.compile(optimizer=Adam(lr=1e-3), loss='mse')
        print(model.summary())
        return model

    def _get_input_layers(self):
        self.action_input = Input(shape=(self.outputs,), name='action_input')
        self.observation_input = Input(shape=(self.inputs,), name='observation_input')
        inputs = [self.action_input, self.observation_input]
        x = Concatenate()(inputs)
        x = Flatten()(x)
        return inputs, x

    def create_input(self, states: np.ndarray, actions: np.ndarray):
        input_ = [states]
        input_.insert(self.action_input_index, actions)
        return input_

    def get_actor_train_fn(self, actor_model: BaseNNModel, actor_optimizer: Optimizer) -> Function:
        combined_inputs = []
        state_inputs = []
        for _input in self.model.input:
            if _input == self.action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(_input)
                state_inputs.append(_input)

        combined_inputs[self.action_input_index] = actor_model.model(state_inputs)

        combined_output = self.model(combined_inputs)

        updates = actor_optimizer.get_updates(
            params=actor_model.model.trainable_weights, loss=-K.mean(combined_output))

        actor_train_fn = K.function(
            state_inputs + [K.learning_phase()],
            [actor_model.model(state_inputs)],
            updates=updates
        )

        return actor_train_fn
