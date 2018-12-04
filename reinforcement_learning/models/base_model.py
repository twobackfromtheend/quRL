import numpy as np


class BaseModel:
    def __init__(self, inputs: int, outputs: int):
        self.inputs = inputs
        self.outputs = outputs

    def build_model(self):
        raise NotImplementedError

    def save_model(self, filename: str):
        raise NotImplementedError

    def create_copy(self):
        return self.__class__(**self.__dict__)

    def set_learning_rate(self, learning_rate: float):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def train_on_batch(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError
