from typing import List, Tuple

import numpy as np
from qutip import rand_dm, Qobj, sigmaz, sigmay, sigmax
from qutip.solver import Result

from quantum_evolution.simulations.base_simulation import HamiltonianData
from quantum_evolution.simulations.env_simulation import EnvSimulation


class BasePseudoEnv:
    """
    Temporary construct to avoid having to create full-fledged env.
    See https://stackoverflow.com/a/47132897
    """

    def __init__(self,
                 hamiltonian: List[HamiltonianData],
                 initial_state: Qobj,
                 target_state: Qobj,
                 t_list: np.ndarray):
        self.initial_state = initial_state
        self.given_target_state = target_state
        self.hamiltonian = hamiltonian
        self.t_list = t_list

        self.current_state: Qobj = None
        self.target_state: Qobj = None
        self.simulation: EnvSimulation = None
        self.result: Result = None  # Should be set in the step() method of subclasses.
        self.reset()

    def step(self, action) -> Tuple[np.ndarray, float, bool, object]:
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def seed(self, seed=None):
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        self.current_state: Qobj = self.initial_state if self.initial_state is not None else self.get_random_state()
        self.target_state: Qobj = self.given_target_state \
            if self.given_target_state is not None else self.get_random_state()

        self.simulation = EnvSimulation(self.hamiltonian, psi0=self.current_state, t_list=self.t_list,
                                        e_ops=[sigmax(), sigmay(), sigmaz()])
        self.result = None
        return self.get_state_as_observation(self.current_state)

    @staticmethod
    def get_random_state():
        return rand_dm(2).unit()

    def get_reward(self) -> float:
        raise NotImplementedError

    @staticmethod
    def get_state_as_observation(state: Qobj) -> np.ndarray:
        return state.data.toarray()
