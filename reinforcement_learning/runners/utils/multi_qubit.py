from qutip import *
import numpy as np


def hamiltonian_generator(L: int, h: float, g: float = 1):
    "Generates the Hamiltonian for a system of L qubits"
    "For L qubits, the Hamiltonian must be a tensor of dimensions 2^L by 2^L"
    "See QuTip for tutorial"

    si = qeye(2)
    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(L):
        op_list = []
        for m in range(L):
            op_list.append(si)

        op_list[n] = sigmax()  # Replaces the nth element of identity matrix list with the Sx matrix
        sx_list.append(tensor(op_list))
        # Resulting tensor operates on the nth qubit only --> sigmax() operates on the nth qubit,
        # depending on where sigmax() was appended
        # sx_list contains the n sigmax() that operate on the n qubits, with each index operating on a certain qubit

        op_list[n] = sigmay()
        sy_list.append(tensor(op_list))

        op_list[n] = sigmaz()
        sz_list.append(tensor(op_list))

    H = 0

    "For following Hamiltonian, see main paper multi-qubit section"

    for n in range(L):
        H += -g * sz_list[n]  # centre term, with g as the coupling? Just set to 1 as paper did
        H += -h * sx_list[n]  # h is the controllable coefficient again

        if n < L - 1:
            H += -sz_list[n + 1] * sz_list[n]  # Cross-term, according to paper

        elif L == 1:  # Cross-term should disappear if L set to 1, reduces to single-qubit
            H += 0
        else:
            H += -sz_list[n] * sz_list[0]  # Closed-chain, so last qubit must be linked to first

    return H


