from reinforcement_learning.runners.presets.discount_rate_presets import DiscountRatePreset
from reinforcement_learning.runners.presets.env_presets import EnvPreset
from reinforcement_learning.runners.presets.model_presets import ModelPreset
from reinforcement_learning.runners.trainer_runner import run
from reinforcement_learning.runners.utils.quantum_variables import get_quantum_variables
from reinforcement_learning.trainers.base_classes.hyperparameters import ExplorationOptions, ExplorationMethod
from reinforcement_learning.trainers.double_dqn_trainer import DoubleDQNTrainer
from reinforcement_learning.trainers.dqn_options import DQNTrainerOptions

TRAINER = DoubleDQNTrainer
TRAINER_OPTIONS = DQNTrainerOptions()

ENV = EnvPreset.QENV3
MODEL = ModelPreset.DENSE_MODEL

EPISODES = 20000
DISCOUNT_RATE = DiscountRatePreset.INCREASING_20000
EXPLORATION = ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=0.7, epsilon_decay=0.9995,
                                 limiting_value=0.03)
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
