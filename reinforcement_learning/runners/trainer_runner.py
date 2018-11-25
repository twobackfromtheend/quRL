import logging
from typing import Type

from reinforcement_learning.runners.presets.discount_rate_presets import DiscountRatePreset
from reinforcement_learning.runners.presets.env_presets import EnvPreset
from reinforcement_learning.runners.presets.exploration_presets import ExplorationPreset
from reinforcement_learning.runners.presets.model_presets import ModelPreset
from reinforcement_learning.runners.utils.env_data import EnvData
from reinforcement_learning.trainers.base_trainer import BaseTrainer
from reinforcement_learning.trainers.hyperparameters import QLearningHyperparameters


def run(trainer: Type[BaseTrainer], model_preset: ModelPreset, env_preset: EnvPreset, episodes: int,
        discount_rate: DiscountRatePreset,
        exploration_preset: ExplorationPreset, with_tensorboard: bool = True, render: bool = False,
        env_kwargs: dict=None, train_kwargs: dict=None,
        logging_level=logging.INFO):

    env_kwargs = env_kwargs if env_kwargs is not None else {}
    train_kwargs = train_kwargs if train_kwargs is not None else {}
    logging.basicConfig(level=logging_level)

    hyperparameters = QLearningHyperparameters(discount_rate.value, exploration_preset.value)
    env_data: EnvData = env_preset.value

    model = model_preset.value(env_data.inputs, env_data.outputs)
    env = env_data.env_class(**env_kwargs)
    _trainer = trainer(model, env, hyperparameters, with_tensorboard)

    _trainer.train(episodes=episodes, render=render, **train_kwargs)
    print(f"max reward total: {max(_trainer.reward_totals)}")
    print(f"last evaluation reward: {_trainer.evaluation_rewards[-1]}")


if __name__ == '__main__':
    from reinforcement_learning.trainers.dqn_trainer import DQNTrainer

    TRAINER = DQNTrainer
    ENV = EnvPreset.CARTPOLE_TS
    MODEL = ModelPreset.DEFAULT

    EPISODES = 20000
    DISCOUNT_RATE = DiscountRatePreset.CONST_97
    EXPLORATION = ExplorationPreset.EPSILON_05

    RENDER = True

    run(trainer=TRAINER, model_preset=MODEL, env_preset=ENV, episodes=EPISODES, discount_rate=DISCOUNT_RATE,
        exploration_preset=EXPLORATION, render=RENDER)
