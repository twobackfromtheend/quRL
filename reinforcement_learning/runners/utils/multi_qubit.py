from qutip import *
import numpy as np
from quantum_evolution.simulations.base_simulation import HamiltonianData

from quantum_evolution.envs.q_env_2 import QEnv2
from quantum_evolution.simulations.env_simulation import EnvSimulation


class MultiQEnv(QEnv2):

    def __init__(self, hamiltonian, t, N, L: int = 3, g: float = 1):
        super().__init__(hamiltonian, t, N)
        self.L = L
        self.g = g
        self.N = N
        self.t = t / 2
        self.hamiltonian = hamiltonian
        self.hmax = self.hamiltonian_generator(-4)
        self.hmin = self.hamiltonian_generator(4)
        self.current_h = self.hmax
        self.initial_state = self.hamiltonian_generator(-2).groundstate()[1]
        self.given_target_state = self.hamiltonian_generator(h=2).groundstate()[1]
        self.dt = self.t / N
        self.current_step: int = 0
        # self.current_state = self.initial_state

    def hamiltonian_generator(self, h: int):
        "Generates the Hamiltonian for a system of L qubits, field coeff h, coupling g"
        "For L qubits, the Hamiltonian must be a tensor of dimensions 2^L by 2^L"
        "See QuTip for tutorial"

        si = qeye(2)
        sx_list = []
        sy_list = []
        sz_list = []

        for n in range(self.L):
            op_list = []
            for m in range(self.L):
                op_list.append(si)

            op_list[n] = sigmax()  # Replaces the nth element of identity matrix list with the Sx matrix
            sx_list.append(tensor(op_list))
            # Resulting tensor operates on the nth qubit only --> sigmax() operates on the nth qubit,
            # depending on where sigmax() was appended
            # sx_list contains the n sigmax() that operate on the n qubits, with each index operating on a certain qubit

            op_list[n] = sigmay()
            sy_list.append(tensor(op_list))

            op_list[n] = sigmaz()
            sz_list.append(tensor(op_list))

        exp_list = [sx_list, sy_list, sz_list]

        H = 0

        "For following Hamiltonian, see main paper multi-qubit section"

        for n in range(self.L):
            H += -self.g * sz_list[n]  # centre term, with g as the coupling? Just set to 1 as paper did
            H += -h * sx_list[n]  # h is the controllable coefficient again

            if n < self.L - 1:
                H += -sz_list[n + 1] * sz_list[n]  # Cross-term, according to paper

            elif self.L == 1:  # Cross-term should disappear if L set to 1, reduces to single-qubit
                H += 0
            else:
                H += -sz_list[0] * sz_list[n]  # Closed-chain, so last qubit must be linked to first

        return H

    def step(self, action: int):
        assert action == 0 or action == 1, 'Action has to be 0 or 1.'
        if action is 0:
            self.current_h = self.current_h

        elif action is 1:
            if self.current_h == self.hmax:
                self.current_h = self.hmin

            else:
                self.current_h = self.hmax

        # self.current_h_x *= -1 ** action  # equivalent to above
        t_list = np.linspace(0, self.dt, 50)

        result = mesolve(self.current_h, self.current_state, t_list, [], [])
        # self.simulation.solve()
        # self.simulation.solve_with_coefficient(self.current_h_x)
        self.current_state = result.states[-1]
        reward = self.get_reward()

        done = self.current_step == self.N - 1
        self.current_step += 1
        return self.get_state(), reward, done, {}
