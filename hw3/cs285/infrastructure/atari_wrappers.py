import numpy as np
import gym
from gym import spaces
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


def wrap_deepmind(env: gym.Env):
    """Configure environment for DeepMind-style Atari."""
    # Record the statistics of the _underlying_ environment, before frame-skip/reward-clipping/etc.
    env = RecordEpisodeStatistics(env)
    # Standard Atari preprocessing
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
    )
    env = FrameStack(env, num_stack=4)
    return env
