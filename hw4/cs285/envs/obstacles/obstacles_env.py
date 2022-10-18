import gym
import numpy as np
from gym import spaces

class Obstacles(gym.Env):
    def __init__(self, start=[-0.5, 0.75], end=[0.7, -0.8], random_starts=True, **kwargs):

        import matplotlib.pyplot as plt #inside, so doesnt get imported when not using this env
        self.plt = plt

        self.action_dim = self.ac_dim = 2
        self.observation_dim = self.obs_dim = 4
        self.boundary_min = -0.99
        self.boundary_max = 0.99

        low = self.boundary_min*np.ones((self.action_dim,))
        high = self.boundary_max*np.ones((self.action_dim,))
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.env_name = 'obstacles'
        self.is_gym = True

        self.start = np.array(start)
        self.end = np.array(end)
        self.current = np.array(start)
        self.random_starts = random_starts

        #obstacles are rectangles, specified by [x of top left, y of topleft, width x, height y]
        self.obstacles = []
        self.obstacles.append([-0.4, 0.8, 0.4, 0.3])
        self.obstacles.append([-0.9, 0.3, 0.2, 0.6])
        self.obstacles.append([0.6, -0.1, 0.12, 0.4])
        self.obstacles.append([-0.1, 0.2, 0.15, 0.4])
        self.obstacles.append([0.1, -0.7, 0.3, 0.15])

        self.eps = 0.1
        self.fig = self.plt.figure()

    def seed(self, seed):
        np.random.seed(seed)

    #########################################

    def pick_start_pos(self):
        if self.random_starts:
            temp = np.random.uniform([self.boundary_min, self.boundary_min+1.25], [self.boundary_max-0.4, self.boundary_max], (self.action_dim,))
            if not self.is_valid(temp[None, :]):
                temp = self.pick_start_pos()
        else:
            temp = self.start
        return temp

    #########################################

    def reset(self, seed=None):
        if seed:
            self.seed(seed)

        self.reset_pose = self.pick_start_pos()
        self.reset_vel = self.end

        return self.do_reset(self.reset_pose, self.reset_vel)

    def do_reset(self, reset_pose, reset_vel, reset_goal=None):

        self.current = reset_pose.copy()
        self.end = reset_vel.copy()

        #clear
        self.counter = 0
        self.plt.clf()

        #return
        return self._get_obs()

    #########################################

    def _get_obs(self):
        return np.concatenate([self.current,self.end])

    def get_score(self, obs):
        curr_pos = obs[:2]
        end_pos = obs[-2:]
        score = -1*np.abs(curr_pos-end_pos)
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

        #initialize and reshape as needed, for batch mode
        self.reward_dict = {}
        if(len(observations.shape)==1):
            observations = np.expand_dims(observations, axis = 0)
            actions = np.expand_dims(actions, axis = 0)
            batch_mode = False
        else:
            batch_mode = True

        #get vars
        curr_pos = observations[:, :2]
        end_pos = observations[:, -2:]

        #calc rew
        dist = np.linalg.norm(curr_pos - end_pos, axis=1)
        self.reward_dict['dist'] = -dist
        self.reward_dict['r_total'] = self.reward_dict['dist']

        #done
        dones = np.zeros((observations.shape[0],))
        dones[dist<self.eps] = 1

        #check oob
        oob = np.zeros((observations.shape[0],))
        oob[curr_pos[:,0]<self.boundary_min] = 1
        oob[curr_pos[:,1]<self.boundary_min] = 1
        oob[curr_pos[:,0]>self.boundary_max] = 1
        oob[curr_pos[:,1]>self.boundary_max] = 1
        dones[oob==1] = 1

        #return
        if(not batch_mode):
            return self.reward_dict['r_total'][0], dones[0]
        return self.reward_dict['r_total'], dones

    def step(self, action):
        self.counter += 1
        action = np.clip(action, -1, 1) #clip (-1, 1)
        action = action / 10. #scale (-1,1) to (-0.1, 0.1)

        # move, only if its a valid move (else, keep it there because it cant move)
        temp = self.current + action
        if self.is_valid(temp[None, :]):
            self.current = temp

        ob = self._get_obs()
        reward, done = self.get_reward(ob, action)
        score = self.get_score(ob)
        env_info = {'ob': ob,
                    'rewards': self.reward_dict,
                    'score': score}

        return ob, reward, done, env_info

    ########################################
    # utility functions
    ########################################

    def render(self, mode=None):
        # boundaries
        self.plt.plot([self.boundary_min, self.boundary_min],
                      [self.boundary_min, self.boundary_max], 'k')
        self.plt.plot([self.boundary_max, self.boundary_max],
                      [self.boundary_min, self.boundary_max], 'k')
        self.plt.plot([self.boundary_min, self.boundary_max],
                      [self.boundary_min, self.boundary_min], 'k')
        self.plt.plot([self.boundary_min, self.boundary_max],
                      [self.boundary_max, self.boundary_max], 'k')
        # obstacles
        for obstacle in self.obstacles:
            tl_x = obstacle[0]
            tl_y = obstacle[1]
            tr_x = tl_x + obstacle[2]
            tr_y = tl_y
            bl_x = tl_x
            bl_y = tl_y - obstacle[3]
            br_x = tr_x
            br_y = bl_y
            self.plt.plot([bl_x, br_x], [bl_y, br_y], 'r')
            self.plt.plot([tl_x, tr_x], [tl_y, tr_y], 'r')
            self.plt.plot([bl_x, bl_x], [bl_y, tl_y], 'r')
            self.plt.plot([br_x, br_x], [br_y, tr_y], 'r')
        # current and end
        self.plt.plot(self.end[0], self.end[1], 'go')
        self.plt.plot(self.current[0], self.current[1], 'ko')
        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return [img]

    def is_valid(self, dat):

        oob_mask = np.any(self.oob(dat), axis=1)

        # old way
        self.a = self.boundary_min + (self.boundary_max-self.boundary_min)/3.0
        self.b = self.boundary_min + 2*(self.boundary_max-self.boundary_min)/3.0
        data_mask = (dat[:, 0] < self.a) | (dat[:, 0] > self.b) | \
                    (dat[:, 1] < self.a) | (dat[:, 1] > self.b)

        #
        in_obstacle = False
        for obstacle in self.obstacles:
            tl_x = obstacle[0]
            tl_y = obstacle[1]
            tr_x = tl_x + obstacle[2]
            tr_y = tl_y
            bl_x = tl_x
            bl_y = tl_y - obstacle[3]
            br_x = tr_x
            br_y = bl_y

            if dat[:, 0]>tl_x and dat[:, 0]<tr_x and dat[:, 1]>bl_y and dat[:, 1]<tl_y:
                in_obstacle = True
                return False

        # not in obstacle, so return whether or not its in bounds
        return (not oob_mask)


    def oob(self, x):
        return (x <= self.boundary_min) | (x >= self.boundary_max)


