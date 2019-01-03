from functools import partial

from reinforcement_learning.models.lstm_non_stateful_model import LSTMNonStatefulModel
from reinforcement_learning.runners.presets.discount_rate_presets import DiscountRatePreset
from reinforcement_learning.runners.presets.env_presets import EnvPreset
from reinforcement_learning.runners.trainer_runner import run
from reinforcement_learning.runners.utils.quantum_variables import get_quantum_variables
from reinforcement_learning.trainers.base_classes.hyperparameters import ExplorationOptions, ExplorationMethod
from reinforcement_learning.trainers.drqn_batched_trainer import DRQNBatchedTrainer
from reinforcement_learning.trainers.drqn_options import DRQNTrainerOptions

TRAINER = DRQNBatchedTrainer
rnn_steps = 10
TRAINER_OPTIONS = DRQNTrainerOptions(rnn_steps=rnn_steps)

ENV = EnvPreset.SINGLEQENV
MODEL = partial(LSTMNonStatefulModel, rnn_steps=rnn_steps, learning_rate=3e-3,
                inner_activation='relu', output_activation='linear')

EPISODES = 20000
DISCOUNT_RATE = DiscountRatePreset.INCREASING_20000
EXPLORATION = ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=1.0, epsilon_decay=0.9995,
                                    limiting_value=0.8)
T = 0.5

initial_state, target_state, hamiltonian_datas, N = get_quantum_variables(T)

run(
    trainer=TRAINER, model=MODEL, env=ENV, episodes=EPISODES, discount_rate=DISCOUNT_RATE, exploration=EXPLORATION,
    env_kwargs={
        'hamiltonian': hamiltonian_datas,
        't': T,
        'N': N,
        'initial_state': initial_state,
        'target_state': target_state,
    },
    trainer_options=TRAINER_OPTIONS
)
