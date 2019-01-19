from enum import Enum

from quantum_evolution.envs.presolved_q_env_2 import PresolvedQEnv2
from quantum_evolution.envs.q_env_2 import QEnv2
from quantum_evolution.envs.q_env_2_ss import QEnv2SingleSolve
from quantum_evolution.envs.q_env_3 import QEnv3
from quantum_evolution.envs.single_q_env import SingleQEnv
from reinforcement_learning.runners.utils.env_data import EnvData
from reinforcement_learning.time_sensitive_envs.acrobot_env import AcrobotTSEnv
from reinforcement_learning.time_sensitive_envs.cartpole_env import CartPoleTSEnv
from reinforcement_learning.time_sensitive_envs.pendulum_env import PendulumTSEnv


class EnvPreset(Enum):
    QENV2 = EnvData(QEnv2, 2, 2)
    SINGLEQENV = EnvData(SingleQEnv, 1, 2)
    SINGLEQENV_CONTINUOUS = EnvData(lambda **kwargs: SingleQEnv(discrete=False, **kwargs), 1, 1)
    SINGLEQENV_CONTINUOUS_CONTAINS_STEP = EnvData(lambda **kwargs: SingleQEnv(discrete=False, state_contains_step_number=True, **kwargs), 2, 1)
    MULTIQENV_CONTINUOUS_CONTAINS_STEP = EnvData(lambda **kwargs: SingleQEnv(discrete=False, state_contains_step_number=True, **kwargs), 2, 1)
    QENV3 = EnvData(QEnv3, 3, 2)
    PRESOLVED_QENV2 = EnvData(PresolvedQEnv2, 2, 2)
    QENV2_SS = EnvData(QEnv2SingleSolve, 2, 2)
    CARTPOLE_TS = EnvData(CartPoleTSEnv, 5, 2)
    ACROBOT_TS = EnvData(AcrobotTSEnv, 7, 2)
    CARTPOLE = EnvData(lambda: CartPoleTSEnv(time_sensitive=False), 4, 2)
    ACROBOT = EnvData(lambda: AcrobotTSEnv(time_sensitive=False), 6, 2)
    PENDULUM = EnvData(lambda: PendulumTSEnv(time_sensitive=False), 3, 1)
    PENDULUM_EXCLUDE_VEL = EnvData(lambda: PendulumTSEnv(time_sensitive=False, exclude_velocity=True), 2, 1)
