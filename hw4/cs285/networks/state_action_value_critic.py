import torch
from torch import nn

import cs285.infrastructure.pytorch_util as ptu

class StateActionCritic(nn.Module):
    def __init__(self, ob_dim, ac_dim, n_layers, size):
        super().__init__()
        self.net = ptu.build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        ).to(ptu.device)
    
    def forward(self, obs, acs):
        return self.net(torch.cat([obs, acs], dim=-1)).squeeze(-1)
