import numpy as np


class EpisodicData:
    def __init__(self):
        self.states: list = None
        self.targets: list = None

    def reset(self):
        """
        Initialises the states and targets attributes
        """
        self.states = []
        self.targets = []

    def record_step_data(self, state, target):
        self.states.append(state)
        self.targets.append(target)

    def get_data_to_train(self):
        """
        Gets the saved data as NumPy arrays to be trained on.
        :return:
        """
        return np.array(self.states), np.array(self.targets)
