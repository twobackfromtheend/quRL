from collections import deque
from typing import Sequence

import numpy as np


class EpochReplayHandler:
    """
    Stores the last `count` replays in memory,
    each replay provides the best among the first `i` replays in memory.
    ie:
    i = 0: replays first protocol
    i = 1: replays first or second protocol, whichever is better
    i = 2: replays first or second or third protocol, whichever is best
    etc.
    """
    def __init__(self, count: int = 40):
        self.count = count
        self.protocols = deque(maxlen=count)
        self.protocol_rewards = deque(maxlen=count)

    def record_protocol(self, protocol: Sequence[int], reward: float):
        self.protocols.append(protocol)
        self.protocol_rewards.append(reward)

    def generator(self):
        for i in range(self.count):
            # Find "best-encountered protocol up to given episode"
            best_protocol_index = int(np.argmax([self.protocol_rewards[_i] for _i in range(i + 1)]))
            best_protocol = self.protocols[best_protocol_index]
            yield best_protocol


