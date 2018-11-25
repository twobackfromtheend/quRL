from typing import NamedTuple, Type, Union

from quantum_evolution.envs.base_q_env import BaseQEnv
from reinforcement_learning.time_sensitive_envs.base_time_sensitive_env import BaseTimeSensitiveEnv


class EnvData(NamedTuple):
    env_class: Union[Type[BaseQEnv], Type[BaseTimeSensitiveEnv]]
    inputs: int
    outputs: int
