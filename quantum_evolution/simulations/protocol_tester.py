from typing import Sequence, Callable

import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, mesolve, sigmax, sigmay, sigmaz, Options, fidelity

from quantum_evolution.plotter.bloch_animator import BlochAnimator
from quantum_evolution.simulations.base_simulation import HamiltonianData


def test_protocol(
        protocol: Sequence[int],
        hamiltonian: Sequence[HamiltonianData],
        t: float,
        N: int,
        initial_state: Qobj,
        target_state: Qobj
):

    e_ops = [sigmax(), sigmay(), sigmaz()]

    h_list = get_h_list(protocol)

    H1_coeff = get_H1_coeff(t, N, h_list)

    hamiltonian[1].callback = H1_coeff

    result = mesolve(
        [hamiltonian_data.format_for_solver() for hamiltonian_data in hamiltonian],
        initial_state,
        tlist=np.linspace(0, t, N * 10),
        e_ops=e_ops,
        options=Options(store_states=True)
    )

    fidelity_ = fidelity(result.states[-1], target_state)
    print(f"fidelity: {fidelity_}")

    bloch_animation = BlochAnimator([result], static_states=[initial_state, target_state])
    bloch_animation.generate_animation()
    bloch_animation.show()


def get_h_list(protocol: Sequence[int]) -> Sequence[int]:
    h_list = []
    current_h_x = 4

    for i in range(len(protocol)):
        if protocol[i] == 1:
            current_h_x *= -1
        h_list.append(current_h_x)

    return h_list


def get_H1_coeff(t: float, N: int, h_list: Sequence[int]) -> Callable:
    t_list = np.linspace(0, t, N)

    def H1_coeff(t, args):
        if t < 0:
            return 0
        if t > t_list[-1]:
            return 0

        index = int(np.argmax(t_list > t) - 1)
        return -h_list[index]

    return H1_coeff


def plot_h(t: float, protocol: Sequence[int]):
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()

    h_list = get_h_list(protocol)
    t_list = np.linspace(0, t, len(protocol))
    plt.plot(t_list, h_list, 'o')

    H1_coeff = get_H1_coeff(t, len(protocol), h_list)
    t_list = np.linspace(0, t, len(protocol) * 10)
    plt.plot(t_list, [-H1_coeff(t, None) for t in t_list], '-')

    ax.set_xlim([0, 3.1])
    ax.set_ylim([-4.2, 4.2])
    ax.yaxis.set_ticks(np.arange(-4, 4, 2))
    plt.show()


if __name__ == '__main__':
    initial_state = (-sigmaz() + 2 * sigmax()).groundstate()[1]
    target_state = (-sigmaz() - 2 * sigmax()).groundstate()[1]


    def placeholder_callback(t, args):
        raise RuntimeError


    hamiltonian = [
        HamiltonianData(-sigmaz()),
        HamiltonianData(sigmax(), placeholder_callback)
    ]

    protocol = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
                0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1]
    t = 3
    N = 60
    plot_h(t, protocol)

    test_protocol(
        protocol,
        hamiltonian,
        t=t,
        N=N,
        initial_state=initial_state,
        target_state=target_state
    )
