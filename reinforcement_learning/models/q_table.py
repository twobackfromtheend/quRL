from typing import Callable

import numpy as np

from reinforcement_learning.models.base_model import BaseModel


class QTableData(dict):

    def __init__(self, outputs: Callable[[], np.ndarray]) -> None:
        super().__init__()
        self.outputs = outputs
        self.initialiser = self.generate_initialiser()

    def __setitem__(self, k, v) -> None:
        if isinstance(k, np.ndarray):
            k = self.get_hashable_from_array(k)
        elif isinstance(k, tuple):
            pass
        else:
            raise TypeError(f"Key has to be np.ndarray, not {type(k)}.")
        super().__setitem__(k, v)

    def __getitem__(self, k):
        assert isinstance(k, np.ndarray), f"Key has to be np.ndarray, not {type(k)}."
        k = self.get_hashable_from_array(k)
        if k not in self:
            self[k] = self.initialiser()
        return super().__getitem__(k)

    def __str__(self):
        rows = []
        for k, v in sorted(self.items()):
            row_string = '    '
            row_string += ' '.join(format(_k, '5.2f') for _k in k) + ': '
            row_string += ' '.join(format(_v, '6.3f') for _v in v)
            rows.append(row_string)

        return f"QTable ({len(self)} rows):\n" + '\n'.join(rows)

    @staticmethod
    def get_hashable_from_array(array: np.ndarray):
        """
        Turns a NumPy array (e.g. representing a state) into a tuple to use as key.
        """
        return tuple(array)

    def generate_initialiser(self) -> Callable[[], np.ndarray]:
        def initialiser():
            # return np.random.random((self.outputs,))
            return np.ones(self.outputs)

        return initialiser


class QTable(BaseModel):
    def __init__(self, inputs: int, outputs: int, learning_rate: float, **kwargs):
        super().__init__(inputs, outputs)
        self.learning_rate = learning_rate
        self.model: QTableData = self.build_model()

    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate

    def build_model(self) -> QTableData:
        return QTableData(outputs=self.outputs)

    def predict(self, x: np.ndarray) -> np.ndarray:
        y = [self.model[_x] for _x in x]
        return np.array(y)

    def add_state_to_table(self, x: np.ndarray):
        """
        Adds state to table, returns initialised value
        :param x:
        :return:
        """
        if x in self.model:
            raise ValueError(f"State {x} already exists in table and should not be re-added.")
        zero_output = self.get_zero_output()
        self.model[x] = zero_output
        return zero_output

    def get_zero_output(self):
        return np.zeros((1, self.outputs))

    def train_on_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Performs Q table update, returns loss as [mse, mae]
        :param x:
        :param y:
        :return:
        """
        loss = []

        for _x, _y in zip(x, y):
            # Q(s, a) <- Q(s, a) + lr * [ R(s, a) + gamma * maxQ(s+1, all a) - Q(s, a) ]
            # _x = s, _y = (target per action)
            error = _y - self.model[_x]
            self.model[_x] = self.model[_x] + self.learning_rate * error

            loss.append(((error ** 2).mean(), np.absolute(error).mean()))

        return np.array(loss).flatten()
