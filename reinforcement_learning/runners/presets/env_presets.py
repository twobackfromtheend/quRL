from enum import Enum

from quantum_evolution.envs.presolved_q_env_2 import PresolvedQEnv2
from quantum_evolution.envs.q_env_2 import QEnv2
from quantum_evolution.envs.q_env_3 import QEnv3
from reinforcement_learning.runners.utils.env_data import EnvData
from reinforcement_learning.time_sensitive_envs.acrobot_env import AcrobotTSEnv
from reinforcement_learning.time_sensitive_envs.cartpole_env import CartPoleTSEnv


class EnvPreset(Enum):
    QENV2 = EnvData(QEnv2, 2, 2)
    QENV3 = EnvData(QEnv3, 3, 2)
    PRESOLVED_QENV2 = EnvData(PresolvedQEnv2, 2, 2)
    CARTPOLE_TS = EnvData(CartPoleTSEnv, 5, 2)
    ACROBOT_TS = EnvData(AcrobotTSEnv, 7, 2)
    CARTPOLE = EnvData(lambda: CartPoleTSEnv(time_sensitive=False), 4, 2)
    ACROBOT = EnvData(lambda: AcrobotTSEnv(time_sensitive=False), 6, 2)
