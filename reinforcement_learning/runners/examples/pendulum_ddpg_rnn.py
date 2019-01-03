from functools import partial

from reinforcement_learning.models.rnn__critic_model import RNNCriticModel
from reinforcement_learning.models.simple_rnn_model import SimpleRNNModel
from reinforcement_learning.runners.presets.discount_rate_presets import DiscountRatePreset
from reinforcement_learning.runners.presets.env_presets import EnvPreset
from reinforcement_learning.runners.trainer_runner import run
from reinforcement_learning.trainers.ddpg_rnn_options import DDPGRNNTrainerOptions
from reinforcement_learning.trainers.ddpg_rnn_trainer import DDPGRNNTrainer
from reinforcement_learning.trainers.policies.ornstein_uhlenbeck import OrnsteinUhlenbeck

TRAINER = DDPGRNNTrainer
# rnn_steps = 1
# ENV = EnvPreset.PENDULUM
rnn_steps = 2
ENV = EnvPreset.PENDULUM_EXCLUDE_VEL

TRAINER_OPTIONS = DDPGRNNTrainerOptions(rnn_steps=rnn_steps, render=True)


MODEL = partial(SimpleRNNModel, rnn_steps=rnn_steps, learning_rate=1e-3,
                inner_activation='relu', output_activation='linear')
CRITIC_MODEL = partial(RNNCriticModel, rnn_steps=rnn_steps)

EPISODES = 20000
DISCOUNT_RATE = DiscountRatePreset.INCREASING_20000
EXPLORATION = OrnsteinUhlenbeck(theta=0.15, sigma=0.3)


run(
    trainer=TRAINER, model=MODEL, critic_model=CRITIC_MODEL,
    env=ENV, episodes=EPISODES, discount_rate=DISCOUNT_RATE, exploration=EXPLORATION,
    trainer_options=TRAINER_OPTIONS
)
