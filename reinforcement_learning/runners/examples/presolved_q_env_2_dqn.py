from reinforcement_learning.runners.presets.discount_rate_presets import DiscountRatePreset
from reinforcement_learning.runners.presets.env_presets import EnvPreset
from reinforcement_learning.runners.presets.exploration_presets import ExplorationPreset
from reinforcement_learning.runners.presets.model_presets import ModelPreset
from reinforcement_learning.runners.trainer_runner import run
from reinforcement_learning.runners.utils.quantum_variables import get_quantum_variables
from reinforcement_learning.trainers.dqn_trainer import DQNTrainer

TRAINER = DQNTrainer
ENV = EnvPreset.PRESOLVED_QENV2
MODEL = ModelPreset.DEFAULT

EPISODES = 20000
DISCOUNT_RATE = DiscountRatePreset.INCREASING_20000
EXPLORATION = ExplorationPreset.EPSILON_08
T = 0.5

RENDER = False

initial_state, target_state, hamiltonian_datas, N = get_quantum_variables(T)

run(
    trainer=TRAINER, model_preset=MODEL, env_preset=ENV, episodes=EPISODES, discount_rate=DISCOUNT_RATE,
    exploration_preset=EXPLORATION, render=RENDER,
    env_kwargs={
        'hamiltonian': hamiltonian_datas,
        't': T,
        'N': N,
        'initial_state': initial_state,
        'target_state': target_state,
    }
)
