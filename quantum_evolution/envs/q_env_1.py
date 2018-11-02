import random
from collections import OrderedDict
from typing import List, Union

import numpy as np
from gym import spaces
from qutip import Qobj, fidelity
from qutip.solver import Result

from quantum_evolution.envs.base_pseudo_env import BasePseudoEnv
from quantum_evolution.simulations.base_simulation import HamiltonianData

observation_space = OrderedDict()
vector_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.float64)
observation_space['target'] = vector_space
observation_space['observation'] = vector_space
observation_space = spaces.Dict(observation_space)


class QEnv1(BasePseudoEnv):
    action_space = spaces.Discrete(2)
    observation_space = observation_space

    def __init__(self,
                 hamiltonian: List[HamiltonianData],
                 t_list: np.ndarray,
                 N: int,
                 initial_state: Qobj = None,
                 target_state: Qobj = None):
        super().__init__(hamiltonian, initial_state, target_state, t_list)
        self.N = N

    def step(self, action):
        actions = self.convert_int_to_bit_list(action, self.N)
        self.simulation.solve_with_actions(actions, self.N)
        result: Result = self.simulation.result

        self.current_state = result.states[-1]
        observation = self.get_state_as_observation(self.current_state)

        reward = self.get_reward()
        done = True
        return observation, reward, done, {}

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        return seed

    def get_reward(self) -> float:
        print(self.current_state)
        print(self.target_state)
        print(fidelity(self.current_state, self.target_state))
        return fidelity(self.current_state, self.target_state)

    def get_random_action(self) -> int:
        return random.getrandbits(self.N)

    @staticmethod
    def convert_int_to_bit_list(action: int, N: int) -> List[int]:
        """
        :param actions: Integer that should be intepreted as N bits.
        :param N:
        :return:
        """
        assert action.bit_length() <= N, f"Integer ({action}) cannot be represented with N ({N}) bits"
        return [action >> i & 1 for i in range(N)][::-1]
