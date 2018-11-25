from reinforcement_learning.runners.presets.discount_rate_presets import DiscountRatePreset
from reinforcement_learning.runners.presets.env_presets import EnvPreset
from reinforcement_learning.runners.presets.exploration_presets import ExplorationPreset
from reinforcement_learning.runners.presets.model_presets import ModelPreset
from reinforcement_learning.runners.trainer_runner import run
from reinforcement_learning.trainers.dqn_trainer import DQNTrainer

TRAINER = DQNTrainer
ENV = EnvPreset.CARTPOLE
MODEL = ModelPreset.DEFAULT

EPISODES = 20000
DISCOUNT_RATE = DiscountRatePreset.CONST_97
EXPLORATION = ExplorationPreset.EPSILON_05

RENDER = True

run(trainer=TRAINER, model_preset=MODEL, env_preset=ENV, episodes=EPISODES, discount_rate=DISCOUNT_RATE,
    exploration_preset=EXPLORATION, render=RENDER)
