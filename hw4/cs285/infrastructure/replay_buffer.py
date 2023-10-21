from cs285.infrastructure.utils import *


class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None

    def sample(self, batch_size):
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observations": self.observations[rand_indices],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.next_observations[rand_indices],
            "dones": self.dones[rand_indices],
        }

    def __len__(self):
        return self.size

    def insert(
        self,
        /,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(action, int):
            action = np.array(action, dtype=np.int64)

        if self.observations is None:
            self.observations = np.empty(
                (self.max_size, *observation.shape), dtype=observation.dtype
            )
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.empty(
                (self.max_size, *next_observation.shape), dtype=next_observation.dtype
            )
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)

        assert observation.shape == self.observations.shape[1:]
        assert action.shape == self.actions.shape[1:]
        assert reward.shape == ()
        assert next_observation.shape == self.next_observations.shape[1:]
        assert done.shape == ()

        self.observations[self.size % self.max_size] = observation
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.next_observations[self.size % self.max_size] = next_observation
        self.dones[self.size % self.max_size] = done

        self.size += 1

    def batched_insert(
        self,
        /,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
            ):
        """
        Insert a batch of transitions into the replay buffer.
        """
        if self.observations is None:
            self.observations = np.empty(
                (self.max_size, *observations.shape[1:]), dtype=observations.dtype
            )
            self.actions = np.empty(
                (self.max_size, *actions.shape[1:]), dtype=actions.dtype
            )
            self.rewards = np.empty(
                (self.max_size, *rewards.shape[1:]), dtype=rewards.dtype
            )
            self.next_observations = np.empty(
                (self.max_size, *next_observations.shape[1:]),
                dtype=next_observations.dtype,
            )
            self.dones = np.empty((self.max_size, *dones.shape[1:]), dtype=dones.dtype)

        assert observations.shape[1:] == self.observations.shape[1:]
        assert actions.shape[1:] == self.actions.shape[1:]
        assert rewards.shape[1:] == self.rewards.shape[1:]
        assert next_observations.shape[1:] == self.next_observations.shape[1:]
        assert dones.shape[1:] == self.dones.shape[1:]

        indices = np.arange(self.size, self.size + observations.shape[0]) % self.max_size
        self.observations[indices] = observations
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_observations[indices] = next_observations
        self.dones[indices] = dones

        self.size += observations.shape[0]
