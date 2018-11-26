from dataclasses import dataclass

from reinforcement_learning.trainers.base_classes.base_options import BaseTrainerOptions


@dataclass
class DQNTrainerOptions(BaseTrainerOptions):
    update_target_every: int = 1
    update_target_soft: bool = True
    update_target_tau: float = 0.01
