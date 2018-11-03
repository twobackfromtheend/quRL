import unittest

import matplotlib.pyplot as plt
import numpy as np
from qutip import sigmax, basis, mesolve, Options, sigmay, sigmaz

from quantum_evolution.plotter.bloch_figure import BlochFigure


class BlochFigureTest(unittest.TestCase):

    def setUp(self):
        self.bloch_figure = BlochFigure()
        self.result = mesolve(
            H=sigmax(),
            rho0=basis(2, 0),
            tlist=np.linspace(0, np.pi, 50),
            e_ops=[sigmax(), sigmay(), sigmaz()],
            options=Options(store_states=True)
        )

    def test_bloch_figure(self):
        for i in range(len(self.result.states)):
            self.bloch_figure.update(self.result.states[i])
            plt.pause(1e-3)


if __name__ == '__main__':
    unittest.main()
