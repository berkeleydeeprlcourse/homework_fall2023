import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64)

        mujoco_env.MujocoEnv.__init__(
            self, "half_cheetah.xml", 1, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)

        self.skip = self.frame_skip

        self.action_dim = self.ac_dim = self.action_space.shape[0]
        self.observation_dim = self.obs_dim = self.observation_space.shape[0]

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
        xvel = observations[:, 9].copy()
        body_angle = observations[:, 2].copy()
        front_leg = observations[:, 6].copy()
        front_shin = observations[:, 7].copy()
        front_foot = observations[:, 8].copy()
        zeros = np.zeros((observations.shape[0],)).copy()

        # ranges
        leg_range = 0.2
        shin_range = 0
        foot_range = 0
        penalty_factor = 10

        # calc rew
        self.reward_dict["run"] = xvel

        front_leg_rew = zeros.copy()
        front_leg_rew[front_leg > leg_range] = -penalty_factor
        self.reward_dict["leg"] = front_leg_rew

        front_shin_rew = zeros.copy()
        front_shin_rew[front_shin > shin_range] = -penalty_factor
        self.reward_dict["shin"] = front_shin_rew

        front_foot_rew = zeros.copy()
        front_foot_rew[front_foot > foot_range] = -penalty_factor
        self.reward_dict["foot"] = front_foot_rew

        # total reward
        self.reward_dict["r_total"] = (
            self.reward_dict["run"]
            + self.reward_dict["leg"]
            + self.reward_dict["shin"]
            + self.reward_dict["foot"]
        )

        # return
        dones = zeros.copy()
        if not batch_mode:
            return self.reward_dict["r_total"][0], dones[0]
        return self.reward_dict["r_total"], dones

    def get_score(self, obs):
        xposafter = obs[0]
        return xposafter

    ##############################################

    def step(self, action):
        # step
        self.do_simulation(action, self.frame_skip)

        # obs/reward/done/score
        ob = self._get_obs()
        rew, done = self.get_reward(ob, action)
        score = self.get_score(ob)

        # return
        env_info = {
            "obs_dict": self.obs_dict,
            "rewards": self.reward_dict,
            "score": score,
        }
        return ob, rew, done, env_info

    def _get_obs(self):
        self.obs_dict = {}
        self.obs_dict["joints_pos"] = self.data.qpos.flat.copy()
        self.obs_dict["joints_vel"] = self.data.qvel.flat.copy()
        self.obs_dict["com_torso"] = self.get_body_com("torso").flat.copy()

        return np.concatenate(
            [
                self.obs_dict["joints_pos"],  # 9
                self.obs_dict["joints_vel"],  # 9
                self.obs_dict["com_torso"],  # 3
            ]
        )

    ##############################################

    def reset_model(self, seed=None):
        # set reset pose/vel
        self.reset_pose = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        self.reset_vel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        # reset the env to that pose/vel
        return self.do_reset(self.reset_pose.copy(), self.reset_vel.copy())

    def do_reset(self, reset_pose, reset_vel, reset_goal=None):
        # reset
        self.set_state(reset_pose, reset_vel)

        # return
        return self._get_obs()

    def render(
        self,
    ):
        ren = super().render()
        self.renderer.render_step()
        return ren
