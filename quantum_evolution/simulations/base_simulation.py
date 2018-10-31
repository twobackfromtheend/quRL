import logging
from typing import List, Union, Callable

import numpy as np
from qutip import *

from quantum_evolution.logger_utils.logger_utils import log_process

logger = logging.getLogger(__name__)


class CoefficientArgs:
    def __init__(self):
        pass


class HamiltonianData:
    def __init__(self, operator: Qobj, callback: Union[Callable[[float, CoefficientArgs], float], str]):
        self.operator = operator
        self.callback = callback

    def format_for_solver(self):
        return [self.operator, self.callback]


class BaseSimulation:

    def __init__(self, hamiltonian: List[HamiltonianData], psi0: Qobj = basis(2, 0)):
        self.hamiltonian = hamiltonian
        self.psi0 = psi0
        self.result = None

    @staticmethod
    def H1_coeff(t: float, args: CoefficientArgs):
        raise NotImplementedError()

    @log_process(logger, 'solving')
    def solve(self, tlist: Union[list, np.array]):
        self.result = mesolve(
            [hamiltonian_data.format_for_solver() for hamiltonian_data in self.hamiltonian],
            self.psi0,
            tlist
        )
