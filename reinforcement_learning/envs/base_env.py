import numpy as np


class BaseTimeSensitiveEnv:
    env = None

    def __init__(self, max_episode_steps):
        self.step_number = 0
        self.max_episode_steps = max_episode_steps

    def step(self, *args, **kwargs):
        new_state, reward, done, info = self.env.step(*args, **kwargs)
        return self.get_observation(new_state), reward, done, info

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        self.step_number = 0
        return self.get_observation(state)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_observation(self, state):
        """
        Appends the current step number to the state.
        :param state:
        :return:
        """
        return np.append(state, self.step_number)

    def increment_step_number(self) -> bool:
        """
        Increments step number or sets it to 0 if on last step
        :return: done - whether it is on the last step.
        """
        if self.step_number >= self.max_episode_steps - 1:
            done = True
        else:
            done = False
        self.step_number += 1
        return done
