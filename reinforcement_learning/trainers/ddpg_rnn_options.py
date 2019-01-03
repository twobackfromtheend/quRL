from dataclasses import dataclass

from reinforcement_learning.trainers.ddpg_options import DDPGTrainerOptions


@dataclass
class DDPGRNNTrainerOptions(DDPGTrainerOptions):
    rnn_steps: int = 3
