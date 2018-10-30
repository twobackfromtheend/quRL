# See https://github.com/qutip/qutip-notebooks/blob/master/examples/control-grape-single-qubit-rotation.ipynb

import matplotlib.pyplot as plt
import numpy as np

from qutip import *
from qutip.control import *

T = 1
times = np.linspace(0, T, 100)

theta, phi = np.random.rand(2)

U = rz(phi) * rx(theta)
print(f"Target: \n{U}\n")

R = 150

H_ops = [sigmax(), sigmay(), sigmaz()]
H_labels = [
    r'$u_{x}$',
    r'$u_{y}$',
    r'$u_{z}$',
]

H0 = 0 * np.pi * sigmaz()

# GRAPE

from qutip.control.grape import plot_grape_control_fields, _overlap
# from qutip.control.cy_grape import cy_overlap
from qutip.control.grape import cy_grape_unitary, grape_unitary_adaptive

from scipy.interpolate import interp1d
from qutip.ui.progressbar import TextProgressBar

u0 = np.array([np.random.rand(len(times)) * 2 * np.pi * 0.005 for _ in range(len(H_ops))])

u0 = [np.convolve(np.ones(10) / 10, u0[idx, :], mode='same') for idx in range(len(H_ops))]
result = cy_grape_unitary(U, H0, H_ops, R, times, u_start=u0, eps=2 * np.pi / T, phase_sensitive=False,
                          progress_bar=TextProgressBar())
plot_grape_control_fields(times, result.u[:, :, :] / (2 * np.pi), H_labels, uniform_axes=True)
plt.show()

print(f"Target: \n{U}\n")
print(f"Result: \n{result.U_f}\n")

print("Overlap:", _overlap(U, result.U_f).real, abs(_overlap(U, result.U_f)) ** 2)

c_ops = []
U_f_numerical = propagator(result.H_t, times[-1], c_ops, args={})
print(f"Result (numerical): {U_f_numerical}")
print("Overlap:", _overlap(U, U_f_numerical))


# Bloch sphere dynamics

psi0 = basis(2, 0)
e_ops = [sigmax(), sigmay(), sigmaz()]


me_result = mesolve(result.H_t, psi0, times, c_ops, e_ops)

b = Bloch()

b.add_points(me_result.expect)

b.add_states(psi0)
b.add_states(U * psi0)
b.render()

plt.show()
