from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, mesolve, sigmax, sigmay, sigmaz, Options, fidelity

from quantum_evolution.plotter.bloch_animator import BlochAnimator
from quantum_evolution.simulations.base_simulation import HamiltonianData
from quantum_evolution.utils.protocol_utils import get_h_list, get_H1_coeff


def evaluate_protocol(
        protocol: Sequence[int],
        hamiltonian: Sequence[HamiltonianData],
        t: float,
        N: int,
        initial_state: Qobj,
        target_state: Qobj
):
    # Normalise units
    t = t / 2

    e_ops = [sigmax(), sigmay(), sigmaz()]

    h_list = get_h_list(protocol)
    print(f"h_list: {h_list}")

    H1_coeff = get_H1_coeff(t, N, h_list)

    hamiltonian[1].callback = H1_coeff

    result = mesolve(
        [hamiltonian_data.format_for_solver() for hamiltonian_data in hamiltonian],
        initial_state,
        tlist=np.linspace(0, t, N * 20),
        e_ops=e_ops,
        options=Options(store_states=True)
    )

    fidelity_ = fidelity(result.states[-1], target_state) ** 2
    print(f"fidelity: {fidelity_}")

    bloch_animation = BlochAnimator([result], static_states=[initial_state, target_state])
    bloch_animation.generate_animation()
    bloch_animation.save(f"fidelity_{fidelity_:.3f}.mp4")
    bloch_animation.show()


def plot_h(t: float, protocol: Sequence[int]):
    """
    Plots h_x as a function of time (as per paper)
    :param t:
    :param protocol:
    :return:
    """
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()

    h_list = get_h_list(protocol)
    t_list = np.linspace(0, t, len(protocol) + 1)[:-1]
    plt.plot(t_list, h_list, 'o')

    H1_coeff = get_H1_coeff(t, len(protocol), h_list)
    t_list = np.linspace(0, t, len(protocol) * 20)
    plt.plot(t_list, [H1_coeff(t, None) for t in t_list], '-')

    ax.set_xlim([0, t])
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
        HamiltonianData(-sigmax(), placeholder_callback)
    ]

    # Test with protocol with length 3.0 - expected fidelity 1.000
    protocol_30 = [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                   0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1]
    t = 3
    N = 60
    plot_h(t, protocol_30)

    evaluate_protocol(
        protocol_30,
        hamiltonian,
        t=t,
        N=N,
        initial_state=initial_state,
        target_state=target_state
    )

    # Test with protocol with length 1.0 - expected fidelity 0.576
    protocol_10 = [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    t = 1
    N = 20
    plot_h(t, protocol_10)

    evaluate_protocol(
        protocol_10,
        hamiltonian,
        t=t,
        N=N,
        initial_state=initial_state,
        target_state=target_state
    )

    # Test with protocol with length 0.5 - expected fidelity 0.331
    protocol_05 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    t = 0.5
    N = 10
    plot_h(t, protocol_05)
    evaluate_protocol(
        protocol_05,
        hamiltonian,
        t=t,
        N=N,
        initial_state=initial_state,
        target_state=target_state
    )
