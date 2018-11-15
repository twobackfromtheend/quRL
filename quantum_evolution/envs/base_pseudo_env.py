from typing import Tuple

import numpy as np
from qutip import rand_dm, Qobj
from qutip.solver import Result

from quantum_evolution.simulations.env_simulation import EnvSimulation


class BasePseudoEnv:
    """
    Temporary construct to avoid having to create full-fledged env.
    See https://stackoverflow.com/a/47132897
    """

    def __init__(self,
                 initial_state: Qobj,
                 target_state: Qobj):
        self.initial_state = initial_state
        self.given_target_state = target_state

        self.current_state: Qobj = self.initial_state if self.initial_state is not None else self.get_random_state()
        self.target_state: Qobj = self.given_target_state \
            if self.given_target_state is not None else self.get_random_state()
        self.simulation: EnvSimulation = None

    def step(self, action) -> Tuple[np.ndarray, float, bool, object]:
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def seed(self, seed=None):
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        if self.simulation is None:
            raise NotImplementedError("Subclasses of BasePseudoEnv must reset .simulation before calling super.reset")

        self.current_state: Qobj = self.initial_state if self.initial_state is not None else self.get_random_state()
        self.target_state: Qobj = self.given_target_state \
            if self.given_target_state is not None else self.get_random_state()
        return self.get_state()

    def get_state(self) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def get_random_state():
        return rand_dm(2).unit()

    def get_reward(self) -> float:
        raise NotImplementedError
