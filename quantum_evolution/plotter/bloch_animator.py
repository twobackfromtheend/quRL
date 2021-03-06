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


class BlochAnimator:
    def __init__(self, results: Sequence[Result], plot_expect: bool = None, static_states: Sequence[Qobj] = None):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig, azim=-40)
        self.ax.set_aspect('equal')
        self.sphere = Bloch(axes=self.ax)
        self.anim: Optional[FuncAnimation] = None
        self.results = results

        self.plot_expect = plot_expect if plot_expect is not None else len(results[0].expect) == 3
        self.static_states = static_states

    @staticmethod
    def animate(i: int, self: 'BlochAnimator') -> None:
        self.sphere.clear()
        for result in self.results:
            self.sphere.add_states(result.states[i])
            if self.plot_expect:
                self.sphere.add_points([_expect[:i + 1] for _expect in result.expect])

        if self.static_states is not None:
            self.sphere.add_states(self.static_states)
        self.sphere.make_sphere()

    def init_func(self) -> None:
        pass

    @log_process(logger, 'generating animation')
    def generate_animation(self):
        self.anim = FuncAnimation(
            self.fig,
            self.animate,
            fargs=(self,),
            frames=len(self.results[0].states),
            interval=1 / 60,
            init_func=self.init_func,
            repeat=False,
        )

    @log_process(logger, 'saving animation')
    def save(self, filename: str = 'bloch_sphere.mp4', fps: int = 60):
        if not self.anim:
            raise AnimationNotGeneratedError()
        self.anim.save(filename, writer="ffmpeg", fps=fps)

    @log_process(logger, 'showing animation')
    def show(self):
        if not self.anim:
            raise AnimationNotGeneratedError()
        plt.show()


class AnimationNotGeneratedError(Exception):
    message = 'Animation not generated. Run self.generate_animation to set self.anim.'
