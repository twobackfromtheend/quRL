from enum import Enum


def get_increasing_discount_rate(steps: int, starting_value: float = 0.97, ending_value: float = 0.999):
    def discount_rate(i: int) -> float:
        return min(ending_value, starting_value + (1 - starting_value) * i / steps)

    return discount_rate


class DiscountRatePreset(Enum):
    CONST_95 = 0.95
    CONST_97 = 0.97
    CONST_99 = 0.99
    INCREASING_5000 = get_increasing_discount_rate(5000)
    INCREASING_10000 = get_increasing_discount_rate(10000)
    INCREASING_20000 = get_increasing_discount_rate(20000)
