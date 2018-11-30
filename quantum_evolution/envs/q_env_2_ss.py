import logging
import random
from typing import Sequence

import numpy as np
from qutip import Qobj, fidelity, sigmax, sigmay, sigmaz

from quantum_evolution.envs.base_q_env import BaseQEnv
from quantum_evolution.simulations.base_simulation import HamiltonianData
from quantum_evolution.simulations.env_simulation import EnvSimulation

logger = logging.getLogger(__name__)


class QEnv2SingleSolve(BaseQEnv):
    """
    state / observation: S = (t, h_x(t))
    action: A = 0, 1; corresponding to stay (dh = 0), switch sign (dh = +-8).
    Solves once per episode
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

        self.current_protocol = []

        logger.info(f"Created QEnv2SingleSolve with N: {N}, t: {t}")

    def step(self, action: int):
        assert action == 0 or action == 1, 'Action has to be 0 or 1.'
        self.current_h_x = self.current_h_x if action is 0 else -self.current_h_x
        self.current_protocol.append(action)

        reward = self.get_reward()

        done = self.current_step == self.N - 1
        self.current_step += 1
        return self.get_state(), reward, done, {}

    def render(self, mode='human'):
        raise NotImplementedError

    def get_reward(self) -> float:
        if self.current_step != self.N - 1:
            return 0

        simulation = EnvSimulation(self.hamiltonian, psi0=self.initial_state,
                                   t_list=np.linspace(0, self.t, self.N * 10),
                                   e_ops=[sigmax(), sigmay(), sigmaz()])
        simulation.solve_with_actions(self.current_protocol, self.N)
        final_state = simulation.result.states[-1]

        _fidelity = fidelity(final_state, self.target_state) ** 2
        return _fidelity

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.current_h_x = self.initial_h_x
        self.current_protocol = []

        self.simulation = 1  # Placeholder value
        return super().reset()

    def get_state(self) -> np.ndarray:
        return np.array((self.current_step * self.dt, self.current_h_x))

    def get_random_action(self):
        return random.getrandbits(1)
