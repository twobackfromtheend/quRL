import logging
from typing import Sequence

logger = logging.getLogger(__name__)


class ExclusiveBestReplayHandler:

    def __init__(self, count: int = 40):
        self.count = count
        self.max_reward: float = None
        self.best_protocol: Sequence[int] = None

    def record_protocol(self, protocol: Sequence[int], reward: float):
        if self.max_reward is None or reward >= self.max_reward:
            logger.info(f"Found new best protocol with reward {reward} - {protocol}")
            self.best_protocol = protocol
            self.max_reward = reward

    def generator(self):
        for _ in range(self.count):
            yield self.best_protocol
