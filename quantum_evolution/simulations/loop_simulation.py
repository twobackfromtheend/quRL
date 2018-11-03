"""
Tests whether looping a mesolve over segments is equivalent to running mesolve across the full duration.
"""

import numpy as np
from qutip import sigmay, sigmaz, rand_ket, mesolve, Options

from quantum_evolution.plotter.bloch_animator import BlochAnimator
from quantum_evolution.simulations.base_simulation import HamiltonianData, sigmax, BaseSimulation

pis = 3

def H1_coeff(t, args):
    coeff = 1 if t // np.pi % 2 == 0 else -1
    return coeff


DEFAULT_HAMILTONIAN_DATA = HamiltonianData(sigmax(), H1_coeff)

psi0 = rand_ket(2)
t_list = np.linspace(0, pis * np.pi, 50 * pis)
e_ops = [sigmax(), sigmay(), sigmaz()]


result_1 = mesolve(
    [DEFAULT_HAMILTONIAN_DATA.format_for_solver()],
    psi0,
    t_list,
    e_ops=e_ops,
    options=Options(store_states=True)
)

# bloch_animation = BlochAnimator([result_1])
# bloch_animation.generate_animation()
# bloch_animation.show()


coeffs = [1 if t % 2 == 0 else -1 for t in range(pis)]
states = []
expect = None
result_2 = None
for i in range(pis):
    coeff = coeffs[i]
    _hamiltonian_data = DEFAULT_HAMILTONIAN_DATA
    _hamiltonian_data.callback = lambda t, args: coeff
    result = mesolve(
        [_hamiltonian_data.format_for_solver()],
        psi0,
        np.linspace(0, np.pi, 50),
        e_ops=e_ops,
        options=Options(store_states=True)
    )

    states += result.states
    if expect is None:
        expect = result.expect
    else:
        expect = np.concatenate((expect, result.expect), axis=1)
    result_2 = result
result_2.states = states
result_2.expect = expect

# bloch_animation = BlochAnimator([result_2])
# bloch_animation.generate_animation()
# bloch_animation.show()
results_difference = np.array(result_1.expect) - np.array(result_2.expect)
difference_magnitude = np.sqrt(np.sum(results_difference ** 2, axis=0))
print(difference_magnitude)

bloch_animation = BlochAnimator([result_1, result_2])
bloch_animation.generate_animation()
bloch_animation.show()

