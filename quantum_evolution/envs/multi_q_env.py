import random

import numpy as np
from qutip import *

from quantum_evolution.envs.base_q_env import BaseQEnv
from quantum_evolution.simulations.base_simulation import HamiltonianData
from quantum_evolution.simulations.env_simulation import EnvSimulation


class MultiQEnv(BaseQEnv):
    """
    Multi-Qubit environment, with L qubits and coupling factor g
    """

    def __init__(self,
                 t: float,
                 N: int,
                 L: int = 6,
                 g: float = 1,
                 initial_state: Qobj = None,
                 target_state: Qobj = None,
                 initial_h_x: float = -4,
                 min_h_x: float = -4,
                 max_h_x: float = 4,
                 discrete: bool = True):
        super().__init__(initial_state, target_state)
        self.L = L
        self.g = g
        self.N = N
        self.t = t
        self.dt = self.t / N
        self.current_h_x = initial_h_x
        self.initial_h_x = initial_h_x

        self.min_h_x = min_h_x
        self.max_h_x = max_h_x

        self.initial_state = self.hamiltonian_generator(-2).groundstate()[1]
        self.given_target_state = self.hamiltonian_generator(2).groundstate()[1]

        self.current_step: int = 0
        self.discrete = discrete

    def hamiltonian_generator(self, h: float):
        """
        Generates the Hamiltonian for a system of L qubits, Sx coefficient h, coupling g
        For L qubits, the Hamiltonian must be a tensor of dimensions 2^L by 2^L
        See http://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/spin-chain.ipynb
        for example
        :param h:
        :return: H
        """
        si = qeye(2)
        sx_list = []
        sy_list = []
        sz_list = []

        for n in range(self.L):
            op_list = []
            for m in range(self.L):
                op_list.append(si)

            op_list[n] = sigmax() / 2  # Replaces the nth element of identity matrix list with the Sx matrix
            sx_list.append(tensor(op_list))
            # Resulting tensor operates on the nth qubit only --> sigmax() operates on the nth qubit,
            # depending on where sigmax() was appended
            # sx_list contains the n sigmax() that operate on the n qubits, with each index operating on a certain qubit

            op_list[n] = sigmay() / 2
            sy_list.append(tensor(op_list))

            op_list[n] = sigmaz() / 2
            sz_list.append(tensor(op_list))

        exp_list = [sx_list, sy_list, sz_list]

        H = 0
        # For following Hamiltonian, see main paper multi-qubit section
        for n in range(self.L):
            H += -self.g * sz_list[n]  # centre term, with g as the coupling? Just set to 1 as paper did
            H += -h * sx_list[n]  # h is the controllable coefficient again

            if n < self.L - 1:
                H += -sz_list[n + 1] * sz_list[n]  # Cross-term

            elif self.L == 1:  # Cross-term should disappear if L set to 1, reduces to single-qubit
                H += 0
            else:
                H += -sz_list[0] * sz_list[n]  # Closed-chain, so last qubit must be linked to first

        return H

    def step(self, action: float):
        if self.discrete:
            assert action == 0 or action == 1, 'Action has to be 0 or 1.'
            self.current_h_x = self.current_h_x if action is 0 else -self.current_h_x
        else:
            clipped_action = min(max(action, self.min_h_x), self.max_h_x)
            self.current_h_x = clipped_action

        t_list = np.linspace(0, self.dt, 50)

        hamiltonian = [HamiltonianData(self.hamiltonian_generator(self.current_h_x))]
        self.simulation = EnvSimulation(hamiltonian, self.current_state, t_list)
        self.simulation.solve()
        self.current_state = self.simulation.result.states[-1]
        reward = self.get_reward()

        done = self.current_step == self.N - 1
        self.current_step += 1
        return self.get_state(), reward, done, {}

    def get_reward(self) -> float:
        if self.current_step != self.N - 1:
            return 0

        _fidelity = fidelity(self.current_state, self.target_state)
        return _fidelity ** 2  # As per paper's definition of fidelity.

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.current_h_x = self.initial_h_x
        self.current_state = self.initial_state  # Reset to initial state,
        #  may seem redundant but super.reset chooses to give a random state
        #  on first instance of reset for some reason

        # Override simulation
        self.simulation = 1  # Placeholder
        return super().reset()

    def get_state(self) -> np.ndarray:
        return np.array((self.current_step * self.dt, self.current_h_x))

    def get_random_action(self):
        if self.discrete:
            return random.getrandbits(1)
        else:
            return np.array([random.uniform(self.min_h_x, self.max_h_x)])
