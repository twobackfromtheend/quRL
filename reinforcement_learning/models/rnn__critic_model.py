import logging

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import Function
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Input, Concatenate, SimpleRNN
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam, Optimizer

from logger_utils.logger_utils import log_process
from reinforcement_learning.models.base_critic_model import BaseCriticModel
from reinforcement_learning.models.base_nn_model import BaseNNModel

logger = logging.getLogger(__name__)


class RNNCriticModel(BaseCriticModel):
    def __init__(self, inputs: int, outputs: int, rnn_steps: int, load_from_filepath: str = None, **kwargs):
        self.rnn_steps = rnn_steps
        super().__init__(inputs, outputs, load_from_filepath, **kwargs)

    def _get_input_layers(self):
        self.action_input = Input(shape=(self.outputs,), name='action_input')
        self.observation_input = Input(shape=(self.rnn_steps, self.inputs), name='observation_input')
        inputs = [self.action_input, self.observation_input]

        observation = SimpleRNN(128)(self.observation_input)
        # observation = Flatten()(self.observation_input)

        x = Concatenate(name='concatenated_inputs')([self.action_input, observation])
        x = Flatten(name='flattened_concatenation')(x)
        return inputs, x
