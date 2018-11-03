from typing import Sequence
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip import Bloch, sigmax, sigmaz


def plot_to_bloch(states: Sequence = None, vectors: Sequence = None, points: Sequence = None):
    fig = plt.figure()
    ax = Axes3D(fig, azim=-40)
    ax.set_aspect('equal')
    sphere = Bloch(axes=ax)
    if vectors:
        sphere.add_vectors(vectors)
    if points:
        sphere.add_points(points)
    if states:
        sphere.add_states(states)
    sphere.make_sphere()
    sphere.show()


if __name__ == '__main__':
    # Show paper's initial and target states
    psi_i = -sigmaz() + 2 * sigmax()
    psi_t = -sigmaz() - 2 * sigmax()
    plot_to_bloch(states=[state.groundstate()[1] for state in (psi_i, psi_t)])
