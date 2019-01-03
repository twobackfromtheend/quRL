import logging
from functools import partial
from typing import Type, Union, Callable

from reinforcement_learning.models.base_critic_model import BaseCriticModel
from reinforcement_learning.models.base_nn_model import BaseNNModel
from reinforcement_learning.runners.presets.discount_rate_presets import DiscountRatePreset
from reinforcement_learning.runners.presets.env_presets import EnvPreset
from reinforcement_learning.runners.presets.exploration_presets import ExplorationPreset
from reinforcement_learning.runners.presets.model_presets import ModelPreset
from reinforcement_learning.runners.utils.env_data import EnvData
from reinforcement_learning.trainers.base_classes.base_actor_critic_trainer import BaseActorCriticTrainer
from reinforcement_learning.trainers.base_classes.base_options import BaseTrainerOptions
from reinforcement_learning.trainers.base_classes.base_trainer import BaseTrainer
from reinforcement_learning.trainers.base_classes.hyperparameters import QLearningHyperparameters, DiscountRate, \
    ExplorationOptions, DDPGHyperparameters
from reinforcement_learning.trainers.policies.ornstein_uhlenbeck import OrnsteinUhlenbeck


def run(trainer: Type[BaseTrainer],
        model: Union[ModelPreset, Callable[[int, int], BaseNNModel], partial],
        env: Union[EnvPreset, EnvData],
        episodes: int,
        discount_rate: Union[DiscountRatePreset, DiscountRate],
        exploration: Union[ExplorationPreset, ExplorationOptions, OrnsteinUhlenbeck],
        trainer_options: BaseTrainerOptions,
        critic_model: Union[ModelPreset, Callable[[int, int], BaseCriticModel], partial] = None,
        env_kwargs: dict = None,
        logging_level=logging.INFO):
    if issubclass(trainer, BaseActorCriticTrainer):
        run_actor_critic(trainer=trainer, model=model, env=env, episodes=episodes, discount_rate=discount_rate,
                         exploration=exploration, trainer_options=trainer_options, critic_model=critic_model,
                         env_kwargs=env_kwargs, logging_level=logging_level)
    else:
        Model = model.value if isinstance(model, ModelPreset) else model
        env_data: EnvData = env.value if isinstance(env, EnvPreset) else env
        exploration_options = exploration.value if isinstance(exploration, ExplorationPreset) else exploration
        discount_rate = discount_rate.value if isinstance(discount_rate, DiscountRatePreset) else discount_rate

        logging.basicConfig(level=logging_level)

        hyperparameters = QLearningHyperparameters(discount_rate, exploration_options)
        _model = Model(env_data.inputs, env_data.outputs)

        env_kwargs = {} if env_kwargs is None else env_kwargs
        _env = env_data.env_class(**env_kwargs)
        if critic_model is None:
            _trainer = trainer(_model, _env, hyperparameters, options=trainer_options)
        else:
            assert issubclass(trainer, BaseActorCriticTrainer), \
                'trainer must be subclass of BaseActorCriticTrainer if critic_model param is given.'
            _trainer = trainer(_model, _env, hyperparameters, options=trainer_options, critic_model=_critic_model)
        _trainer.train(episodes=episodes)
        print_reward_stats(_trainer)
        return _trainer, _model


def run_actor_critic(trainer: Type[BaseTrainer],
                     model: Union[ModelPreset, Callable[[int, int], BaseNNModel], partial],
                     env: Union[EnvPreset, EnvData],
                     episodes: int,
                     discount_rate: Union[DiscountRatePreset, DiscountRate],
                     exploration: Union[ExplorationPreset, ExplorationOptions, OrnsteinUhlenbeck],
                     trainer_options: BaseTrainerOptions,
                     critic_model: Union[ModelPreset, Callable[[int, int], BaseCriticModel], partial] = None,
                     env_kwargs: dict = None,
                     logging_level=logging.INFO):
    Model = model.value if isinstance(model, ModelPreset) else model
    CriticModel = critic_model.value if isinstance(critic_model, ModelPreset) else critic_model
    env_data: EnvData = env.value if isinstance(env, EnvPreset) else env
    exploration_options = exploration.value if isinstance(exploration, ExplorationPreset) else exploration
    discount_rate = discount_rate.value if isinstance(discount_rate, DiscountRatePreset) else discount_rate

    logging.basicConfig(level=logging_level)

    hyperparameters = DDPGHyperparameters(discount_rate, exploration_options)
    _model = Model(env_data.inputs, env_data.outputs)
    if critic_model is not None:
        _critic_model = CriticModel(env_data.inputs, env_data.outputs)

    env_kwargs = {} if env_kwargs is None else env_kwargs
    _env = env_data.env_class(**env_kwargs)
    if critic_model is None:
        _trainer = trainer(_model, _env, hyperparameters, options=trainer_options)
    else:
        assert issubclass(trainer, BaseActorCriticTrainer), \
            'trainer must be subclass of BaseActorCriticTrainer if critic_model param is given.'
        _trainer = trainer(_model, _env, hyperparameters, options=trainer_options, critic_model=_critic_model)
    _trainer.train(episodes=episodes)
    print_reward_stats(_trainer)
    return _trainer, _model


def print_reward_stats(_trainer):
    print(f"max reward total: {max(_trainer.reward_totals)}")
    print(f"max evaluation reward: {max(_trainer.evaluation_rewards)}")
    print(f"last evaluation reward: {_trainer.evaluation_rewards[-1]}")


if __name__ == '__main__':
    from reinforcement_learning.trainers.dqn_trainer import DQNTrainer
    from reinforcement_learning.trainers.dqn_options import DQNTrainerOptions

    TRAINER = DQNTrainer
    TRAINER_OPTIONS = DQNTrainerOptions(render=True)

    ENV = EnvPreset.CARTPOLE_TS
    MODEL = ModelPreset.DENSE_MODEL

    EPISODES = 20000
    DISCOUNT_RATE = DiscountRatePreset.CONST_97
    EXPLORATION = ExplorationPreset.EPSILON_05

    RENDER = True

    run(trainer=TRAINER, model=MODEL, env=ENV, episodes=EPISODES, discount_rate=DISCOUNT_RATE,
        exploration=EXPLORATION, trainer_options=TRAINER_OPTIONS)
