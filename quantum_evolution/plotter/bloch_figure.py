import logging
from typing import Sequence, List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from qutip import *

logger = logging.getLogger(__name__)

SpinArray = np.array


class BlochFigure:
    def __init__(self, static_states: Sequence[Qobj] = None):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig, azim=-40)
        self.ax.set_aspect('equal')
        self.sphere = Bloch(axes=self.ax)

        self.static_states = static_states

        self.previous_points = [[], [], []]

    def update(self, states: Sequence[Qobj]) -> None:
        self.sphere.clear()
        if len(self.previous_points[0]) != 0:
            self.sphere.add_points(self.previous_points)
        plotted_states = max(len(states) // 3, 1)
        for i in range(len(states)):
            if i % plotted_states == 0:
                state_vector = self.get_expected_value_for_state(states[i])
                for _i in range(3):
                    self.previous_points[_i].append(state_vector[_i])
        self.sphere.add_vectors(self.get_expected_value_for_state(states[-1]))

        if self.static_states is not None:
            self.sphere.add_states(self.static_states)
        self.sphere.make_sphere()
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        self.fig.show()

    def reset(self):
        self.previous_points = [[], [], []]

    @staticmethod
    def get_expected_value_for_state(state) -> List[float]:
        return [
            expect(sigmax(), state),
            expect(sigmay(), state),
            expect(sigmaz(), state)
        ]
