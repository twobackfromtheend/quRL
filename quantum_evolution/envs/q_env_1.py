import random
from collections import OrderedDict
from typing import List, Sequence

import numpy as np
from gym import spaces
from qutip import Qobj, fidelity

from quantum_evolution.envs.base_pseudo_env import BasePseudoEnv
from quantum_evolution.plotter.bloch_animator import BlochAnimator
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
                 hamiltonian: Sequence[HamiltonianData],
                 t_list: np.ndarray,
                 N: int,
                 initial_state: Qobj = None,
                 target_state: Qobj = None):
        super().__init__(hamiltonian, initial_state, target_state, t_list)
        self.N = N

    def step(self, action):
        actions = self.convert_int_to_bit_list(action, self.N)
        self.simulation.solve_with_actions(actions, self.N)
        self.result = self.simulation.result

        self.current_state = self.result.states[-1]
        observation = self.get_state_as_observation(self.current_state)

        reward = self.get_reward()
        done = True
        return observation, reward, done, {}

    def render(self, mode='human'):
        bloch_animation = BlochAnimator([self.result], static_states=[self.target_state])
        bloch_animation.generate_animation()
        bloch_animation.show()

    def seed(self, seed=None):
        return seed

    def get_reward(self) -> float:
        _fidelity = fidelity(self.current_state, self.target_state)
        return _fidelity - 0.5

    def randomise_action(self, action: int, epsilon: float) -> int:
        bits_to_randomise = np.random.choice((0, 1), size=self.N, p=(1 - epsilon, epsilon))
        bit_mask = self.convert_bit_list_to_int(bits_to_randomise)
        random_bits = random.getrandbits(self.N)

        randomised_action = action ^ (random_bits & bit_mask)
        # print(self.convert_int_to_bit_list(action, self.N), '\n',
        #       self.convert_int_to_bit_list(randomised_action, self.N), epsilon)
        return randomised_action

    @staticmethod
    def convert_int_to_bit_list(action: int, N: int) -> List[int]:
        """
        :param action: Integer that should be interpreted as N bits.
        :param N:
        :return:
        """
        assert action.bit_length() <= N, f"Integer ({action}) cannot be represented with N ({N}) bits"
        return [action >> i & 1 for i in range(N)][::-1]

    @staticmethod
    def convert_bit_list_to_int(bit_list: Sequence[int]) -> int:
        _int = 0
        for bit in bit_list:
            _int = (_int << 1) | int(bit)
        return _int
