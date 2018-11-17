import logging
import random
from typing import Sequence

import numpy as np
from qutip import Qobj, fidelity, sigmax, sigmay, sigmaz

from quantum_evolution.envs.base_pseudo_env import BasePseudoEnv
from quantum_evolution.plotter.bloch_figure import BlochFigure
from quantum_evolution.simulations.base_simulation import HamiltonianData
from quantum_evolution.simulations.env_simulation import EnvSimulation

logger = logging.getLogger(__name__)


class QEnv2(BasePseudoEnv):
    """
    state / observation: S = (t, h_x(t))
    action: A = 0, 1; corresponding to stay (dh = 0), switch sign (dh = +-8).
    """

    def __init__(self,
                 hamiltonian: Sequence[HamiltonianData],
                 t: float,
                 N: int,
                 initial_state: Qobj = None,
                 target_state: Qobj = None,
                 initial_h_x=-4):
        super().__init__(initial_state, target_state)
        self.hamiltonian = hamiltonian
        self.initial_h_x = initial_h_x
        self.N = N
        self.t = t / 2  # Normalisation with units.
        self.dt = self.t / N
        self.current_step: int = 0
        self.current_h_x = None
        self.bloch_figure = None
        logger.info(f"Created QEnv2 with N: {N}, t: {t}")

    def step(self, action: int):
        assert action == 0 or action == 1, 'Action has to be 0 or 1.'
        self.current_h_x = self.current_h_x if action is 0 else -self.current_h_x
        # self.current_h_x *= -1 ** action  # equivalent to above
        t_list = np.linspace(0, self.dt, 50)
        self.simulation = EnvSimulation(self.hamiltonian, psi0=self.current_state, t_list=t_list,
                                        e_ops=[sigmax(), sigmay(), sigmaz()])
        self.simulation.solve_with_coefficient(self.current_h_x)
        self.current_state = self.simulation.result.states[-1]

        reward = self.get_reward()

        done = self.current_step == self.N - 1
        self.current_step += 1
        return self.get_state(), reward, done, {}

    def render(self, mode='human'):
        if self.bloch_figure is None:
            self.bloch_figure = BlochFigure(static_states=(self.target_state,))
        self.bloch_figure.update(self.simulation.result.states)

    def get_reward(self) -> float:
        if self.current_step != self.N - 1:
            return 0

        _fidelity = fidelity(self.current_state, self.target_state)
        return _fidelity ** 2  # As per paper's definition of fidelity.

    def reset(self) -> np.ndarray:
        if self.bloch_figure is not None:
            self.bloch_figure.reset()
        self.current_step = 0
        self.current_h_x = self.initial_h_x
        # Override simulation
        t_list = np.linspace(0, self.dt, 50)
        self.simulation = EnvSimulation(self.hamiltonian, psi0=self.initial_state, t_list=t_list,
                                        e_ops=[sigmax(), sigmay(), sigmaz()])
        return super().reset()

    def get_state(self) -> np.ndarray:
        return np.array((self.current_step * self.dt, self.current_h_x))

    def get_random_action(self):
        return random.getrandbits(1)
