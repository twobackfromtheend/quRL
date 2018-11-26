from typing import Sequence, Callable, Any, List

import numpy as np


def get_h_list(protocol: Sequence[int]) -> Sequence[int]:
    """
    Gets a list of h_x's for the given protocol.
    :param protocol:
    :return:
    """
    h_list = []
    current_h_x = -4

    for i in range(len(protocol)):
        if protocol[i] == 1:
            current_h_x *= -1
        h_list.append(current_h_x)

    return h_list


def get_H1_coeff(t: float, N: int, h_list: Sequence[int]) -> Callable[[float, Any], float]:
    """
    Gets the H1_coeff function that is required for the hamiltonian.
    :param t: Total duration
    :param N: Number of steps
    :param h_list: List of h_x's (given from get_h_list())
    :return: H1_coeff - a function that returns the coefficient of the time-dependent part of the hamiltonian.
    """
    t_list = np.linspace(0, t, N + 1)

    def H1_coeff(t: float, args: Any) -> float:
        if t < 0:
            return 0
        if t > t_list[-1]:
            return 0

        index = int(np.argmax(t_list > t) - 1)
        return h_list[index]

    return H1_coeff


def convert_int_to_bit_list(action: int, N: int) -> List[int]:
    """
    :param action: Integer that should be interpreted as N bits.
    :param N:
    :return:
    """
    assert action.bit_length() <= N, f"Integer ({action}) cannot be represented with N ({N}) bits"
    return [action >> i & 1 for i in range(N)][::-1]


def convert_bit_list_to_int(bit_list: Sequence[int]) -> int:
    _int = 0
    for bit in bit_list:
        _int = (_int << 1) | int(bit)
    return _int
