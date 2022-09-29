"""This file includes a collection of utility functions that are useful for
implementing DQN."""
import random
from collections import namedtuple

import gym
import numpy as np
from torch import nn
import torch.optim as optim

from cs285.infrastructure.atari_wrappers import wrap_deepmind
from gym.envs.registration import register

import torch


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

OptimizerSpec = namedtuple(
    "OptimizerSpec",
    ["constructor", "optim_kwargs", "learning_rate_schedule"],
)


def register_custom_envs():
    from gym.envs.registration import registry
    if 'LunarLander-v3' not in registry.env_specs:
        register(
            id='LunarLander-v3',
            entry_point='cs285.envs.box2d.lunar_lander:LunarLander',
            max_episode_steps=1000,
            reward_threshold=200,
        )


def get_env_kwargs(env_name):
    if env_name in ['MsPacman-v0', 'PongNoFrameskip-v4']:
        kwargs = {
            'learning_starts': 50000,
            'target_update_freq': 10000,
            'replay_buffer_size': int(1e6),
            'num_timesteps': int(2e8),
            'q_func': create_atari_q_network,
            'learning_freq': 4,
            'grad_norm_clipping': 10,
            'input_shape': (84, 84, 4),
            'env_wrappers': wrap_deepmind,
            'frame_history_len': 4,
            'gamma': 0.99,
        }
        kwargs['optimizer_spec'] = atari_optimizer(kwargs['num_timesteps'])
        kwargs['exploration_schedule'] = atari_exploration_schedule(kwargs['num_timesteps'])

    elif env_name == 'LunarLander-v3':
        def lunar_empty_wrapper(env):
            return env
        kwargs = {
            'optimizer_spec': lander_optimizer(),
            'q_func': create_lander_q_network,
            'replay_buffer_size': 50000,
            'batch_size': 32,
            'gamma': 1.00,
            'learning_starts': 1000,
            'learning_freq': 1,
            'frame_history_len': 1,
            'target_update_freq': 3000,
            'grad_norm_clipping': 10,
            'lander': True,
            'num_timesteps': 500000,
            'env_wrappers': lunar_empty_wrapper
        }
        kwargs['exploration_schedule'] = lander_exploration_schedule(kwargs['num_timesteps'])

    else:
        raise NotImplementedError

    return kwargs


def create_lander_q_network(ob_dim, num_actions):
    return nn.Sequential(
        nn.Linear(ob_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, num_actions),
    )

class Ipdb(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        import ipdb; ipdb.set_trace()
        return x


class PreprocessAtari(nn.Module):
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        return x / 255.


def create_atari_q_network(ob_dim, num_actions):
    return nn.Sequential(
        PreprocessAtari(),
        nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
        nn.ReLU(),
        nn.Linear(512, num_actions),
    )

def atari_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_timesteps / 8, 0.01),
        ], outside_value=0.01
    )


def atari_ram_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 0.2),
            (1e6, 0.1),
            (num_timesteps / 8, 0.01),
        ], outside_value=0.01
    )


def atari_optimizer(num_timesteps):
    lr_schedule = PiecewiseSchedule(
        [
            (0, 1e-1),
            (num_timesteps / 40, 1e-1),
            (num_timesteps / 8, 5e-2),
        ],
        outside_value=5e-2,
    )

    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1e-3,
            eps=1e-4
        ),
        learning_rate_schedule=lambda t: lr_schedule.value(t),
    )


def lander_optimizer():
    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1,
        ),
        learning_rate_schedule=lambda epoch: 1e-3,  # keep init learning rate
    )


def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def compute_exponential_averages(variables, decay):
    """Given a list of tensorflow scalar variables
    create ops corresponding to their exponential
    averages
    Parameters
    ----------
    variables: [tf.Tensor]
        List of scalar tensors.
    Returns
    -------
    averages: [tf.Tensor]
        List of scalar tensors corresponding to averages
        of al the `variables` (in order)
    apply_op: tf.runnable
        Op to be run to update the averages with current value
        of variables.
    """
    averager = tf.train.ExponentialMovingAverage(decay=decay)
    apply_op = averager.apply(variables)
    return [averager.average(v) for v in variables], apply_op

def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)

def initialize_interdependent_variables(session, vars_list, feed_dict):
    """Initialize a list of variables one at a time, which is useful if
    initialization of some variables depends on initialization of the others.
    """
    vars_left = vars_list
    while len(vars_left) > 0:
        new_vars_left = []
        for v in vars_left:
            try:
                session.run(tf.variables_initializer([v]), feed_dict)
            except tf.errors.FailedPreconditionError:
                new_vars_left.append(v)
        if len(new_vars_left) >= len(vars_left):
            # This can happen if the variables all depend on each other, or more likely if there's
            # another variable outside of the list, that still needs to be initialized. This could be
            # detected here, but life's finite.
            raise Exception("Cycle in variable dependencies, or extenrnal precondition unsatisfied.")
        else:
            vars_left = new_vars_left

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)

class MemoryOptimizedReplayBuffer(object):
    def __init__(self, size, frame_history_len, lander=False):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.lander = lander

        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.float32 if self.lander else np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

