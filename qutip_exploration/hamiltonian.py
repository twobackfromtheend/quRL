from qutip import *
import numpy as np

sx = sigmax()
sy = sigmay()
sz = sigmaz()

expectations = [sx, sy, sz]

def get_variables():
    H1 = sigmax()

    def H1_coeff(t, args):
        return -0.1 * t

    H = [[H1, H1_coeff]]

    psi0 = basis(2, 0)
    t = np.linspace(0, 2 * np.pi, 100)

    return H, psi0, t


def main_method():
    H, psi0, t = get_variables()

    results = mesolve(H, psi0, t, [], expectations)
    print(results.states)
    print(results.expect)


def loop_method():
    H, psi0, t = get_variables()
    results = []
    prev_t = 0
    for _t in t:
        result = mesolve(H, psi0, [0, _t - prev_t], [], [])
        results.append(result)
        psi0 = result.states[-1]
        prev_t = _t
    print([result.states for result in results])


main_method()
