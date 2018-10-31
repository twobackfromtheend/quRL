from reinforcement_learning.models.base_model import BaseModel


class BaseTrainer:

    def __init__(self, model: BaseModel, env):
        self.model = model
        self.env = env

    def train(self):
        raise NotImplementedError
