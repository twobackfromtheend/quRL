import numpy as np

from reinforcement_learning.envs.base_env import BaseTimeSensitiveEnv
import gym


class AcrobotTSEnv(BaseTimeSensitiveEnv):
    def __init__(self, max_episode_steps: int = 250, sparse: bool = True):
        super().__init__(max_episode_steps)
        gym.envs.register(
            id='AcrobotFixedTime-v0',
            entry_point='gym.envs.classic_control:AcrobotEnv',
            max_episode_steps=max_episode_steps,
        )
        self.env = gym.make('AcrobotFixedTime-v0')
        self.sparse = sparse

    def step(self, action):
        new_state, reward, done, info = self.env.step(action)
        done = self.increment_step_number()

        if self.sparse:
            if done:
                reward = self._get_height()
            else:
                reward = 0
        else:
            if done:
                reward = self._get_height() * 1010
            else:
                reward = self._get_height()
        new_state = self.get_observation(new_state)
        return new_state, reward, done, info

    def _get_height(self):
        unwrapped_state = self.env.unwrapped.state
        return -np.cos(unwrapped_state[0]) - np.cos(unwrapped_state[1] + unwrapped_state[0]) + 2
