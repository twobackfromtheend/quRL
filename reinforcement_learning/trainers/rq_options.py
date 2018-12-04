from dataclasses import dataclass

from reinforcement_learning.trainers.base_classes.base_options import BaseTrainerOptions


@dataclass
class RQTrainerOptions(BaseTrainerOptions):
    rnn_steps: int = 5
