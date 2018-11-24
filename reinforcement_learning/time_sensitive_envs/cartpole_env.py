import gym

from reinforcement_learning.time_sensitive_envs.base_time_sensitive_env import BaseTimeSensitiveEnv


class CartPoleTSEnv(BaseTimeSensitiveEnv):

    def __init__(self, max_episode_steps: int = 200, time_sensitive: bool = True):
        super().__init__(max_episode_steps)

        gym.envs.register(
            id='CartPoleTimeSensitive-v0',
            entry_point='gym.time_sensitive_envs.classic_control:CartPoleEnv',
            max_episode_steps=max_episode_steps
        )
        self.env = gym.make('CartPoleTimeSensitive-v0')

        if not time_sensitive:
            self.step = self.vanilla_step
            self.reset = self.vanilla_reset

    def step(self, action):
        new_state, reward, done, info = self.env.step(action)
        _ = self.increment_step_number()  # Ignore done-ness returned - it only returns True if on last stage.

        if done:
            reward = 0
        return self.get_observation(new_state), reward, done, info

    def vanilla_step(self, action):
        """
        Non-time-sensitive.
        """
        new_state, reward, done, info = self.env.step(action)
        if done:
            reward = 0
        return new_state, reward, done, info

    def vanilla_reset(self, *args, **kwargs):
        """
        Non-time-sensitive.
        """
        return self.env.reset(*args, **kwargs)