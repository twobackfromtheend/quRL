from typing import Callable, Optional

from dataclasses import dataclass


@dataclass
class BaseTrainerOptions:
    render: bool = False
    save_every: Optional[int] = 100000
    evaluate_every: Optional[int] = 50
    with_tensorboard: bool = True
    learning_rate_override: Optional[Callable[[int], float]] = None

