from typing import Union

from tensorflow.python.keras import Sequential, Model


class BaseModel:

    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> Union[Sequential, Model]:
        raise NotImplementedError
