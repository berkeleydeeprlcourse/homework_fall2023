import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
from gym.spaces import Box
import mujoco


class Reacher7DOFEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)
        # placeholder
        self.hand_sid = -2
        self.target_sid = -1

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(
            self,
            curr_dir + "/assets/sawyer.xml",
            2,
            observation_space=observation_space,
            **kwargs
        )
        utils.EzPickle.__init__(self)
        self.observation_dim = 20
        self.action_dim = 7

        self.hand_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "finger"
        )
        self.target_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target"
        )
        self.skip = self.frame_skip
        if self.render_mode == "rgb_array":
            super().render()
            self.renderer.render_step()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat,  # [7]
                self.data.qvel.flatten() / 10.0,  # [7]
                self.data.site_xpos[self.hand_sid],  # [3]
                self.model.site_pos[self.target_sid],  # [3]
            ]
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward, done = self.get_reward(ob, a)

        score = self.get_score(ob)

        # finalize step
        env_info = {"ob": ob, "rewards": self.reward_dict, "score": score}

        return ob, reward, done, env_info

    def get_score(self, obs):
        hand_pos = obs[-6:-3]
        target_pos = obs[-3:]
        score = -1 * np.abs(hand_pos - target_pos)
        return score

    def get_reward(self, observations, actions):
        """get reward/s of given (observations, actions) datapoint or datapoints

        Args:
            observations: (batchsize, obs_dim) or (obs_dim,)
            actions: (batchsize, ac_dim) or (ac_dim,)

        Return:
            r_total: reward of this (o,a) pair, dimension is (batchsize,1) or (1,)
            done: True if env reaches terminal state, dimension is (batchsize,1) or (1,)
        """

        # initialize and reshape as needed, for batch mode
        self.reward_dict = {}
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, axis=0)
            actions = np.expand_dims(actions, axis=0)
            batch_mode = False
        else:
            batch_mode = True

        # get vars
        hand_pos = observations[:, -6:-3]
        target_pos = observations[:, -3:]

        # calc rew
        dist = np.linalg.norm(hand_pos - target_pos, axis=1)
        self.reward_dict["r_total"] = -10 * dist

        # done is always false for this env
        dones = np.zeros((observations.shape[0],))

        # return
        if not batch_mode:
            return self.reward_dict["r_total"][0], dones[0]
        return self.reward_dict["r_total"], dones

    def reset(self, **kwargs):
        _ = self.reset_model()

        self.model.site_pos[self.target_sid] = [0.1, 0.1, 0.1]

        observation, _reward, done, _info = self.step(np.zeros(7))
        ob = self._get_obs()

        return ob

    def reset_model(self, seed=None):
        if seed is not None:
            self.seed(seed)

        self.reset_pose = self.init_qpos.copy()
        self.reset_vel = self.init_qvel.copy()

        self.reset_goal = np.zeros(3)
        self.reset_goal[0] = self.np_random.uniform(low=-0.3, high=0.3)
        self.reset_goal[1] = self.np_random.uniform(low=-0.2, high=0.2)
        self.reset_goal[2] = self.np_random.uniform(low=-0.25, high=0.25)

        return self.do_reset(self.reset_pose, self.reset_vel, self.reset_goal)

    def do_reset(self, reset_pose, reset_vel, reset_goal):
        self.set_state(reset_pose, reset_vel)

        # reset target
        self.reset_goal = reset_goal.copy()
        self.model.site_pos[self.target_sid] = self.reset_goal

        # return
        return self._get_obs()

    def render(
        self,
    ):
        ren = super().render()
        self.renderer.render_step()
        return [ren[0]]
