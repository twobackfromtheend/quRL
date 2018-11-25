from qutip import *

from quantum_evolution.simulations.base_simulation import HamiltonianData


def get_quantum_variables(t: float):
    N = int(t / 0.05)

    initial_state = (-sigmaz() + 2 * sigmax()).groundstate()[1]
    target_state = (-sigmaz() - 2 * sigmax()).groundstate()[1]

    def placeholder_callback(t, args):
        raise RuntimeError

    hamiltonian_datas = [
        HamiltonianData(-sigmaz()),
        HamiltonianData(-sigmax(), placeholder_callback)
    ]

    return initial_state, target_state, hamiltonian_datas, N
