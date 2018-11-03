import random
from collections import OrderedDict
from typing import List, Sequence

import numpy as np
from gym import spaces
from qutip import Qobj, fidelity, sigmax, sigmay, sigmaz

from quantum_evolution.envs.base_pseudo_env import BasePseudoEnv
from quantum_evolution.plotter.bloch_animator import BlochAnimator
from quantum_evolution.simulations.base_simulation import HamiltonianData
from quantum_evolution.simulations.env_simulation import EnvSimulation


class QEnv2(BasePseudoEnv):
    """
    state / observation: S = (t, h_x(t))
    action: A = 0, 1; corresponding to stay (dh = 0), switch sign (dh = +-8).
    """

    def __init__(self,
                 hamiltonian: Sequence[HamiltonianData],
                 N: int,
                 t: float,
                 initial_state: Qobj = None,
                 target_state: Qobj = None):
        super().__init__(initial_state, target_state)

        self.N = N
        self.t = t
        self.dt = t / N
        self.current_step: int = 0
        self.current_h_x = -4

        self.bloch_sphere = None

    def step(self, action: int):
        assert action == 0 or action == 1, 'Action has to be 0 or 1.'
        self.current_h_x = self.current_h_x if action is 0 else -self.current_h_x
        # self.current_h_x *= -1 ** action  # equivalent to above
        self.simulation.solve_with_coefficient(self.current_h_x)
        self.current_state = self.result.states[-1]

        reward = self.get_reward()

        done = self.current_step == self.N - 1
        self.current_step += 1
        return self.get_state(), reward, done, {}

    def render(self, mode='human'):
        bloch_animation = BlochAnimator([self.result], static_states=[self.target_state])
        bloch_animation.generate_animation()
        bloch_animation.show()

    def get_reward(self) -> float:
        if self.current_step != self.N:
            return 0

        _fidelity = fidelity(self.current_state, self.target_state)
        return _fidelity - 0.5

    def reset(self) -> np.ndarray:
        self.current_step = 0
        super().reset()

        # Override simulation
        t_list = np.linspace(0, self.dt, 50)
        self.simulation = EnvSimulation(self.hamiltonian, psi0=self.current_state, t_list=t_list,
                                        e_ops=[sigmax(), sigmay(), sigmaz()])
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return np.array((self.current_step * self.dt, self.current_h_x))
