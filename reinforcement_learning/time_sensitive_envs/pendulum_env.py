import gym

from reinforcement_learning.time_sensitive_envs.base_time_sensitive_env import BaseTimeSensitiveEnv


class PendulumTSEnv(BaseTimeSensitiveEnv):
    # discrete_actions = [-2, 2]
    discrete_actions = [-1, 1]

    def __init__(self, max_episode_steps: int = 200, time_sensitive: bool = True, discrete: bool = False,
                 exclude_velocity: bool = False):
        super().__init__(max_episode_steps, time_sensitive)
        self.discrete = discrete
        self.exclude_velocity = exclude_velocity

        gym.envs.register(
            id='PendulumTimeSensitive-v0',
            entry_point='gym.envs.classic_control:PendulumEnv',
            max_episode_steps=max_episode_steps
        )
        self.env = gym.make('PendulumTimeSensitive-v0')

    def get_observation(self, state):
        if self.exclude_velocity:
            state = state[:-1]
        return super().get_observation(state)

    def step(self, action):
        if self.discrete:
            action = self.discrete_actions[action]
        new_state, reward, done, info = self.env.step(action)
        _ = self.increment_step_number()

        return self.get_observation(new_state), reward, done, info
