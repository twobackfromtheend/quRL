from reinforcement_learning.models.base_model import BaseModel
from reinforcement_learning.trainers.base_trainer import BaseTrainer


class BaseAgent:
    def __init__(self, model: BaseModel, trainer: BaseTrainer):
        self.model = model
        self.trainer = trainer

