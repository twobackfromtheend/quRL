from collections import deque
from typing import Sequence

import numpy as np


class BaseReplayHandler:

    def __init__(self):
        raise NotImplementedError

    def record_protocol(self, protocol: Sequence[int], reward: float):
        raise NotImplementedError

    def generator(self):
        raise NotImplementedError


