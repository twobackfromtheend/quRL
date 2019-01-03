from dataclasses import dataclass
from tensorflow.python.keras.optimizers import Adam, Optimizer

from reinforcement_learning.trainers.base_classes.base_options import BaseTrainerOptions


@dataclass
class DDPGTrainerOptions(BaseTrainerOptions):
    update_target_every: int = 1
    update_target_soft: bool = True
    update_target_tau: float = 0.01
    actor_optimizer: Optimizer = Adam(1e-3)
