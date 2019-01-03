import logging
import random
from typing import Sequence, Union

import numpy as np
from qutip import Qobj, fidelity, sigmax, sigmay, sigmaz

from quantum_evolution.envs.base_q_env import BaseQEnv
from quantum_evolution.plotter.bloch_figure import BlochFigure
from quantum_evolution.simulations.base_simulation import HamiltonianData
from quantum_evolution.simulations.env_simulation import EnvSimulation

logger = logging.getLogger(__name__)


class SingleQEnv(BaseQEnv):
    """
    state / observation: S = (h_x(t))
    action:
        If discrete, A = 0, 1; corresponding to stay (dh = 0), switch sign (dh = +-8).
        else A = h_x
    """

    def __init__(self,
                 hamiltonian: Sequence[HamiltonianData],
                 t: float,
                 N: int,
                 initial_state: Qobj = None,
                 target_state: Qobj = None,
                 initial_h_x: float = -4,
                 min_h_x: float = -4,
                 max_h_x: float = 4,
                 state_contains_step_number: bool = False,
                 discrete: bool = True,
                 action_coefficient: float = None):
        super().__init__(initial_state, target_state)
        self.state_contains_step_number = state_contains_step_number

        self.hamiltonian = hamiltonian
        self.initial_h_x = initial_h_x
        assert initial_h_x == min_h_x or initial_h_x == max_h_x, \
            f"initial_h_x has to be either min_h_x ({min_h_x}) or max_h_x ({max_h_x}), not {initial_h_x}"

        self.min_h_x = min_h_x
        self.max_h_x = max_h_x

        self.discrete = discrete
        self.action_coefficient = action_coefficient

        self.N = N
        self.t = t / 2  # Normalisation with units.
        self.dt = self.t / N

        self.current_step: int = 0
        self.current_h_x = None
        self.bloch_figure = None
        logger.info(f"Created SingleQEnv with N: {N}, t: {t}")

    def step(self, action: Union[float, Sequence]):
        if self.discrete:
            assert action == 0 or action == 1, 'Action has to be 0 or 1.'
            if action == 1:
                # Swap between min and max
                self.current_h_x = self.max_h_x if self.current_h_x == self.min_h_x else self.min_h_x
        else:
            if self.action_coefficient is not None:
                action *= self.action_coefficient
            clipped_action = np.clip(action, self.min_h_x, self.max_h_x)
            assert len(clipped_action) == 1, "action has to be a list of 1 float."
            self.current_h_x = clipped_action[0]

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
        if self.state_contains_step_number:
            return np.array((self.current_step * self.dt, self.current_h_x))
        else:
            return np.array([self.current_h_x])

    def get_random_action(self):
        if self.discrete:
            return random.getrandbits(1)
        else:
            return np.array([random.uniform(self.min_h_x, self.max_h_x)])
