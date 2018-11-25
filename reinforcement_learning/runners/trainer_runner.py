from typing import Type

from reinforcement_learning.runners.presets.discount_rate_presets import DiscountRatePreset
from reinforcement_learning.runners.presets.env_presets import EnvPreset
from reinforcement_learning.runners.presets.exploration_presets import ExplorationPreset
from reinforcement_learning.runners.presets.model_presets import ModelPreset
from reinforcement_learning.runners.utility_classes.env_data import EnvData
from reinforcement_learning.trainers.base_trainer import BaseTrainer
from reinforcement_learning.trainers.hyperparameters import QLearningHyperparameters


def run(trainer: Type[BaseTrainer], model_preset: ModelPreset, env_preset: EnvPreset, episodes: int,
        discount_rate: DiscountRatePreset,
        exploration_preset: ExplorationPreset, with_tensorboard: bool = True, render: bool = False):
    hyperparameters = QLearningHyperparameters(discount_rate.value, exploration_preset.value)
    env_data: EnvData = env_preset.value
    model = model_preset.value(env_data.inputs, env_data.outputs)
    _trainer = trainer(model, env_data.env_class(), hyperparameters, with_tensorboard)
    _trainer.train(episodes=episodes, render=render)
    print(f"max reward total: {max(_trainer.reward_totals)}")
    print(f"last evaluation reward: {_trainer.evaluation_rewards[-1]}")
