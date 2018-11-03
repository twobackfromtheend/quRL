import random
from collections import OrderedDict
from typing import List, Sequence

import numpy as np
from gym import spaces
from qutip import Qobj, fidelity, sigmax, sigmay, sigmaz, expect

from quantum_evolution.envs.base_pseudo_env import BasePseudoEnv
from quantum_evolution.plotter.bloch_animator import BlochAnimator
from quantum_evolution.simulations.base_simulation import HamiltonianData
from quantum_evolution.simulations.env_simulation import EnvSimulation

observation_space = OrderedDict()
vector_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.float64)
observation_space['target'] = vector_space
observation_space['observation'] = vector_space
observation_space = spaces.Dict(observation_space)


class QEnv1(BasePseudoEnv):
    """
    Might be fundamentally misinterpreting the paper.

    Given the starting state, this asks for 2^N outputs (Q value for each of the possible set of actions).
    Every episode being a single step.
    """
    action_space = spaces.Discrete(2)
    observation_space = observation_space  # Not used.

    def __init__(self,
                 hamiltonian: Sequence[HamiltonianData],
                 t_list: np.ndarray,
                 N: int,
                 initial_state: Qobj = None,
                 target_state: Qobj = None):
        super().__init__(initial_state, target_state)
        self.t_list = t_list
        self.N = N
        self.hamiltonian = hamiltonian

    def step(self, action):
        actions = self.convert_int_to_bit_list(action, self.N)
        self.simulation.solve_with_actions(actions, self.N)
        self.result = self.simulation.result

        self.current_state = self.result.states[-1]

        observation = self.get_state()
        reward = self.get_reward()
        done = True
        return observation, reward, done, {}

    def render(self, mode='human'):
        bloch_animation = BlochAnimator([self.result], static_states=[self.target_state])
        bloch_animation.generate_animation()
        bloch_animation.show()

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

    def get_state(self) -> np.ndarray:
        return self.get_state_as_observation(self.current_state)

    @staticmethod
    def get_state_as_observation(state: Qobj) -> np.ndarray:
        expect_operators = [sigmax(), sigmay(), sigmaz()]
        return np.array([expect(operator, state) for operator in expect_operators])

    def reset(self) -> np.ndarray:
        self.simulation = EnvSimulation(self.hamiltonian, psi0=self.current_state, t_list=self.t_list,
                                        e_ops=[sigmax(), sigmay(), sigmaz()])
        return super().reset()
