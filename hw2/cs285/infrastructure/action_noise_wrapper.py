import gymnasium as gym
import numpy as np

class ActionNoiseWrapper(gym.ActionWrapper):
    def __init__(self, env, seed, std):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.std = std

    def action(self, act):
        act = act + self.rng.normal(0, self.std, act.shape)
        return act
