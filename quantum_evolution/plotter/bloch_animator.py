import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from qutip import *
from qutip.solver import Result

logger = logging.getLogger(__name__)

SpinArray = np.array


class BlochAnimator:
    def __init__(self, result: Result):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig, azim=-40)
        self.ax.set_aspect('equal')
        self.sphere = Bloch(axes=self.ax)
        self.anim: Optional[FuncAnimation] = None
        self.result = result

    @staticmethod
    def animate(i: int, self: 'BlochAnimator') -> None:
        self.sphere.clear()
        self.sphere.add_states(self.result.states[i])
        self.sphere.make_sphere()

    def init_func(self) -> None:
        self.sphere.vector_color = ['r']

    def generate_animation(self):
        logger.info('Generating animation')
        self.anim = FuncAnimation(
            self.fig,
            self.animate,
            fargs=(self,),
            frames=len(self.result.states),
            interval=1 / 60,
            init_func=self.init_func,
            repeat=False,
        )
        logger.info('Finished generating animation.')

    def save(self, filename: str = 'bloch_sphere.mp4', fps: int = 30):
        if not self.anim:
            raise AnimationNotGeneratedError()
        logger.info('Saving animation.')
        self.anim.save(filename, writer="ffmpeg", fps=fps)
        logger.info('Finished saving animation.')

    def show(self):
        if not self.anim:
            raise AnimationNotGeneratedError()
        logger.info('Showing animation.')
        plt.show()
        logger.info('Finished showing animation.')


class AnimationNotGeneratedError(Exception):
    message = 'Animation not generated. Run self.generate_animation to set self.anim.'
