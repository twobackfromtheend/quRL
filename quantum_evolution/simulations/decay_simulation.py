import logging

import numpy as np
from qutip import *

from quantum_evolution.plotter.bloch_animator import BlochAnimator
from quantum_evolution.simulations.base_simulation import HamiltonianData, BaseSimulation

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# TODO: Improve name, give proper description, implement proper scenario.

def H1_coeff(t, args):
    return -0.1 * t


DEFAULT_HAMILTONIAN_DATA = HamiltonianData(sigmax(), H1_coeff)

psi0 = rand_ket(2)
t_list = np.linspace(0, 2 * np.pi, 200)
e_ops = [sigmax(), sigmay(), sigmaz()]
decay_simulation = BaseSimulation([DEFAULT_HAMILTONIAN_DATA], psi0, t_list=t_list, e_ops=e_ops)

decay_simulation.solve()

bloch_animation = BlochAnimator([decay_simulation.result])
bloch_animation.generate_animation()
bloch_animation.show()

