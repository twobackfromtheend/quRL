import logging
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from qutip import *
from qutip.solver import Result

from logger_utils.logger_utils import log_process

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

    def update(self, state: Qobj) -> None:
        self.sphere.clear()
        if len(self.previous_points[0]) != 0:
            self.sphere.add_points(self.previous_points)

        current_state = [
            expect(sigmax(), state),
            expect(sigmay(), state),
            expect(sigmaz(), state)
        ]
        self.sphere.add_vectors(current_state)

        for i in range(3):
            self.previous_points[i].append(current_state[i])
        if self.static_states is not None:
            self.sphere.add_states(self.static_states)
        self.sphere.make_sphere()
        self.fig.canvas.flush_events()
        self.fig.show()
