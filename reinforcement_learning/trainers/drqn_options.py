from dataclasses import dataclass

from reinforcement_learning.trainers.dqn_options import DQNTrainerOptions


@dataclass
class DRQNTrainerOptions(DQNTrainerOptions):
    rnn_steps: int = 5
