import os
import unittest
import numpy as np
from qutip import sigmax, basis, mesolve, Options, sigmay, sigmaz

from quantum_evolution.plotter.bloch_animator import BlochAnimator, AnimationNotGeneratedError


class BlochAnimatorTest(unittest.TestCase):
    def setUp(self):
        self.result = mesolve(
            H=sigmax(),
            rho0=basis(2, 0),
            tlist=np.linspace(0, np.pi, 50),
            e_ops=[sigmax(), sigmay(), sigmaz()],
            options=Options(store_states=True)
        )
        self.bloch_animation = BlochAnimator([self.result])

    def test_bloch_animator_show(self):
        with self.assertRaises(AnimationNotGeneratedError):
            self.bloch_animation.show()

        self.bloch_animation.generate_animation()
        self.bloch_animation.show()

    def test_bloch_animator_save(self):
        with self.assertRaises(AnimationNotGeneratedError):
            self.bloch_animation.save()

        test_file_name = 'test_bloch_animation.mp4'
        if os.path.isfile(test_file_name):
            os.remove(test_file_name)

        self.bloch_animation.generate_animation()
        self.bloch_animation.save(test_file_name)

        file_directory = os.path.dirname(os.path.realpath(__file__))
        self.assertIn(test_file_name, os.listdir(file_directory))
        os.remove(test_file_name)


if __name__ == '__main__':
    unittest.main()
