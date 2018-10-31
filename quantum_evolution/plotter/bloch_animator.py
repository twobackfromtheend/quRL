from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from qutip import *
from qutip.solver import Result

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
        self.anim = FuncAnimation(
            self.fig,
            self.animate,
            fargs=(self,),
            frames=len(self.result.states),
            interval=1/60,
            init_func=self.init_func,
            repeat=False,
        )

    def save(self, filename: str = 'bloch_sphere.mp4', fps: int = 30):
        if not self.anim:
            raise AnimationNotGeneratedError()
        self.anim.save(filename, writer="ffmpeg", fps=fps)

    def show(self):
        if not self.anim:
            raise AnimationNotGeneratedError()
        plt.show()


class AnimationNotGeneratedError(Exception):
    message = 'Animation not generated. Run self.generate_animation to set self.anim.'
