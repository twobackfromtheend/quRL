from reinforcement_learning.runners.presets.discount_rate_presets import DiscountRatePreset
from reinforcement_learning.runners.presets.env_presets import EnvPreset
from reinforcement_learning.runners.presets.exploration_presets import ExplorationPreset
from reinforcement_learning.runners.presets.model_presets import ModelPreset
from reinforcement_learning.runners.trainer_runner import run
from reinforcement_learning.trainers.dqn_options import DQNTrainerOptions
from reinforcement_learning.trainers.dqn_trainer import DQNTrainer

TRAINER = DQNTrainer
TRAINER_OPTIONS = DQNTrainerOptions(render=True)

ENV = EnvPreset.CARTPOLE
MODEL = ModelPreset.DENSE_MODEL

EPISODES = 20000
DISCOUNT_RATE = DiscountRatePreset.CONST_97
EXPLORATION = ExplorationPreset.EPSILON_05


run(trainer=TRAINER, model=MODEL, env=ENV, episodes=EPISODES, discount_rate=DISCOUNT_RATE,
    exploration=EXPLORATION, trainer_options=TRAINER_OPTIONS)
