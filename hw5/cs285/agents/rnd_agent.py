import torch
from torch import nn
import numpy as np

from typing import Callable, List, Tuple

from cs285.agents.dqn_agent import DQNAgent
import cs285.infrastructure.pytorch_util as ptu

def init_network(model):
    if isinstance(model, nn.Linear):
        model.weight.data.normal_()
        model.bias.data.normal_()

class RNDAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        num_actions: int,
        make_rnd_network: Callable[[Tuple[int, ...]], nn.Module],
        make_rnd_network_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        make_target_rnd_network: Callable[[Tuple[int, ...]], nn.Module],
        rnd_weight: float,
        **kwargs
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.rnd_weight = rnd_weight

        self.rnd_net = make_rnd_network(observation_shape)
        self.rnd_target_net = make_target_rnd_network(observation_shape)

        self.rnd_target_net.apply(init_network)

        # Freeze target network
        for p in self.rnd_target_net.parameters():
            p.requires_grad_(False)

        self.rnd_optimizer = make_rnd_network_optimizer(
            self.rnd_net.parameters()
        )

    def update_rnd(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Update the RND network using the observations.
        """
        # TODO(student): update the RND network
        loss = ...

        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()

        return loss.item()

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        with torch.no_grad():
            # TODO(student): Compute RND bonus for batch and modify rewards
            rnd_error = ...
            assert rnd_error.shape == rewards.shape
            rewards = ...

        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the RND network.
        rnd_loss = self.update_rnd(observations)
        metrics["rnd_loss"] = rnd_loss

        return metrics

    def num_aux_plots(self) -> int:
        return 1
    
    def plot_aux(
        self,
        axes: List,
    ) -> dict:
        """
        Plot the RND prediction error for the observations.
        """
        import matplotlib.pyplot as plt
        assert len(axes) == 1
        ax: plt.Axes = axes[0]

        with torch.no_grad():
            # Assume a state space of [0, 1] x [0, 1]
            x = torch.linspace(0, 1, 100)
            y = torch.linspace(0, 1, 100)
            xx, yy = torch.meshgrid(x, y)

            inputs = ptu.from_numpy(np.stack([xx.flatten(), yy.flatten()], axis=1))
            targets = self.rnd_target_net(inputs)
            predictions = self.rnd_net(inputs)

            errors = torch.norm(predictions - targets, dim=-1)
            errors = torch.reshape(errors, xx.shape)

            # Log scale, aligned with normal axes
            from matplotlib import cm
            ax.imshow(ptu.to_numpy(errors).T, extent=[0, 1, 0, 1], origin="lower", cmap="hot")
            plt.colorbar(ax.images[0], ax=ax)
