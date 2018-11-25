import math
from enum import Enum


def get_cyclical_learning_rate(scale: int = 100, upper_limit: float = 3e-3, lower_limit: float = 1e-5):
    # https://www.wolframalpha.com/input/?i=y+%3D+((cos(x+%2F+100)+%2B+1.000)+%2F+2+*+6+*+10%5E-3)+*+exp(-(x+%2F+100+%2F+10))+%2B+3+*+10%5E-5+for+x+from+0+to+3000

    def get_learning_rate(i: int) -> float:
        return ((math.cos(i / scale) + 1.000) / 2 * upper_limit) * math.e ** -(i / scale / 10) + lower_limit

    return get_learning_rate


class LearningRatePreset(Enum):
    CONST_1_3 = 1e-3
    CONST_1_5 = 1e-5
    CYCLICAL_100_2_5 = get_cyclical_learning_rate(100, 1e-2, 1e-5)
    CYCLICAL_100_3_5 = get_cyclical_learning_rate(100, 1e-3, 1e-5)
    CYCLICAL_100_4_5 = get_cyclical_learning_rate(100, 1e-4, 1e-5)
