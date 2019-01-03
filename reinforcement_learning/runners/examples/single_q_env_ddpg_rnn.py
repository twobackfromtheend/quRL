from functools import partial

from reinforcement_learning.models.rnn__critic_model import RNNCriticModel
from reinforcement_learning.models.simple_rnn_model import SimpleRNNModel
from reinforcement_learning.runners.presets.discount_rate_presets import DiscountRatePreset
from reinforcement_learning.runners.presets.env_presets import EnvPreset
from reinforcement_learning.runners.trainer_runner import run
from reinforcement_learning.runners.utils.quantum_variables import get_quantum_variables
from reinforcement_learning.trainers.base_classes.hyperparameters import ExplorationMethod, ExplorationOptions
from reinforcement_learning.trainers.ddpg_rnn_options import DDPGRNNTrainerOptions
from reinforcement_learning.trainers.ddpg_rnn_trainer import DDPGRNNTrainer
from reinforcement_learning.trainers.policies.ornstein_uhlenbeck import OrnsteinUhlenbeck

TRAINER = DDPGRNNTrainer
rnn_steps = 10
TRAINER_OPTIONS = DDPGRNNTrainerOptions(rnn_steps=rnn_steps)

ENV = EnvPreset.SINGLEQENV_CONTINUOUS
MODEL = partial(SimpleRNNModel, rnn_steps=rnn_steps, learning_rate=1e-3,
                inner_activation='relu', output_activation='tanh')
CRITIC_MODEL = partial(RNNCriticModel, rnn_steps=rnn_steps)

EPISODES = 20000
DISCOUNT_RATE = DiscountRatePreset.INCREASING_20000
# EXPLORATION = ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=1.0, epsilon_decay=0.9995,
#                                     limiting_value=0.8)
EXPLORATION = OrnsteinUhlenbeck(theta=0.15, sigma=0.2, dt=1 / rnn_steps)

T = 0.5

initial_state, target_state, hamiltonian_datas, N = get_quantum_variables(T)

run(
    trainer=TRAINER, model=MODEL, critic_model=CRITIC_MODEL,
    env=ENV, episodes=EPISODES, discount_rate=DISCOUNT_RATE, exploration=EXPLORATION,
    env_kwargs={
        'hamiltonian': hamiltonian_datas,
        't': T,
        'N': N,
        'initial_state': initial_state,
        'target_state': target_state,
        'action_coefficient': 4  # For tanh activation
    },
    trainer_options=TRAINER_OPTIONS
)
