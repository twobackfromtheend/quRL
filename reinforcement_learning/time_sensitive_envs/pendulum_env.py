import gym

from reinforcement_learning.time_sensitive_envs.base_time_sensitive_env import BaseTimeSensitiveEnv


class PendulumTSEnv(BaseTimeSensitiveEnv):
    discrete_actions = [-2, 2]

    def __init__(self, max_episode_steps: int = 200, time_sensitive: bool = True, discrete: bool = False):
        super().__init__(max_episode_steps, time_sensitive)
        self.discrete = discrete

        gym.envs.register(
            id='PendulumTimeSensitive-v0',
            entry_point='gym.envs.classic_control:PendulumEnv',
            max_episode_steps=max_episode_steps
        )
        self.env = gym.make('PendulumTimeSensitive-v0')

    def step(self, action):
        if self.discrete:
            action = self.discrete_actions[action]
        new_state, reward, done, info = self.env.step(action)
        _ = self.increment_step_number()  # Ignore done-ness returned - it only returns True if on last stage.

        if done:
            reward = 0
        return self.get_observation(new_state), reward, done, info

    def vanilla_step(self, action):
        """
        Non-time-sensitive.
        """
        if self.discrete:
            action = [self.discrete_actions[action]]
        new_state, reward, done, info = self.env.step(action)
        if done:
            reward = 0
        return new_state, reward, done, info
