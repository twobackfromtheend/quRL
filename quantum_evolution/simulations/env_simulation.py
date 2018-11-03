import math
from typing import List, Union, Sequence

import numpy as np
from qutip import Qobj, basis

from quantum_evolution.simulations.base_simulation import BaseSimulation, HamiltonianData


class EnvSimulation(BaseSimulation):

    def __init__(self,
                 hamiltonian: Sequence[HamiltonianData],
                 psi0: Qobj = basis(2, 0),
                 t_list: Union[list, np.array] = np.linspace(0, 2 * np.pi, 100), c_ops: List[Qobj] = None,
                 e_ops: List[Qobj] = None):
        super().__init__(hamiltonian, psi0, t_list, c_ops, e_ops)

    def solve_with_actions(self, actions: List[int], N: int):
        hamiltonian_data = self.hamiltonian[1]
        hamiltonian_data.callback = self.get_H1_coeff(actions, N)
        super().solve()

    def get_H1_coeff(self, actions: List[int], N: int):
        max_t = np.max(self.t_list)

        def H1_coeff(t, args):
            action_index = min(N - 1, int(math.floor(t / max_t * N)))
            action_to_use = actions[action_index]
            action = -4 * (action_to_use - 0.5) * 2
            # print(action)
            return action

        return H1_coeff
