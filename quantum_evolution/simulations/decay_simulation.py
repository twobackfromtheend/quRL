import logging

import numpy as np
from qutip import *

from quantum_evolution.plotter.bloch_animator import BlochAnimator
from quantum_evolution.simulations.base_simulation import HamiltonianData, BaseSimulation

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def H1_coeff(t, args):
    return -0.1 * t


DEFAULT_HAMILTONIAN_DATA = HamiltonianData(sigmax(), H1_coeff)

decay_simulation = BaseSimulation([DEFAULT_HAMILTONIAN_DATA])

tlist = np.linspace(0, 2 * np.pi, 100)
decay_simulation.solve(tlist, e_ops=[sigmax(), sigmay(), sigmaz()])

bloch_animation = BlochAnimator(decay_simulation.result)
bloch_animation.generate_animation()
bloch_animation.show()
# bloch_animation.save()
