from typing import List, Union, Sequence

import numpy as np
from qutip import Qobj, basis

from quantum_evolution.simulations.base_simulation import BaseSimulation, HamiltonianData
from quantum_evolution.utils.protocol_utils import get_h_list, get_H1_coeff


class EnvSimulation(BaseSimulation):

    def __init__(self,
                 hamiltonian: Sequence[HamiltonianData],
                 psi0: Qobj = basis(2, 0),
                 t_list: Union[list, np.array] = np.linspace(0, 2 * np.pi, 100), c_ops: List[Qobj] = None,
                 e_ops: List[Qobj] = None):
        super().__init__(hamiltonian, psi0, t_list, c_ops, e_ops)

    def solve_with_actions(self, actions: List[int], N: int):
        h_list = get_h_list(actions)
        H1_coeff = get_H1_coeff(self.t_list.max(), N, h_list)
        self.hamiltonian[1].callback = H1_coeff
        super().solve()

    def solve_with_coefficient(self, coefficient: float):
        def H1_coeff(t, args):
            return coefficient
        self.hamiltonian[1].callback = H1_coeff
        super().solve()
