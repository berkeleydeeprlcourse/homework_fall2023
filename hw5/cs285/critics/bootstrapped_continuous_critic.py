from .base_critic import BaseCritic
from torch import nn
from torch import optim
import pdb

from cs285.infrastructure import pytorch_util as ptu


class BootstrappedContinuousCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.critic_network = ptu.build_mlp(
            self.ob_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            self.learning_rate,
        )

    def forward(self, obs):
        return self.critic_network(obs).squeeze(1)

    def forward_np(self, obs):
        obs = ptu.from_numpy(obs)
        predictions = self(obs)
        return ptu.to_numpy(predictions)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                nothing
        """
        raise NotImplementedError
        # Not needed for this homework

    ####################################
    ####################################
