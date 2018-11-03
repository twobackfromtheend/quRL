import time
from typing import Union

from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.models import save_model


class BaseModel:

    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> Union[Sequential, Model]:
        raise NotImplementedError

    def save_model(self, name: str, use_timestamp: bool = True):
        filename = f"model_{name}"
        if use_timestamp:
            filename += f"{time.strftime('%Y%m%d-%H%M%S')}"
        save_model(self.model, filename)
