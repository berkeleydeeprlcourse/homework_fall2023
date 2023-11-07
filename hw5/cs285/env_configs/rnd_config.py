import torch
from cs285.env_configs.dqn_config import basic_dqn_config
from cs285.infrastructure import pytorch_util as ptu


def rnd_config(
    rnd_weight: float,
    rnd_dim: int = 5,
    rnd_network_hidden_size: int = 400,
    rnd_network_num_layers: int = 2,
    rnd_network_learning_rate: float = 1e-3,
    total_steps: int = 50000,
    discount: float = 0.95,
    **kwargs,
):
    config = basic_dqn_config(total_steps=total_steps, discount=discount, **kwargs)
    config["agent_kwargs"]["rnd_weight"] = rnd_weight
    config["log_name"] = "{env_name}_rnd{rnd_weight}".format(
        env_name=config["env_name"], rnd_weight=rnd_weight
    )
    config["agent"] = "rnd"

    config["agent_kwargs"]["make_rnd_network"] = lambda obs_shape: ptu.build_mlp(
        input_size=obs_shape[0],
        output_size=rnd_dim,
        n_layers=rnd_network_num_layers,
        size=rnd_network_hidden_size,
    )
    config["agent_kwargs"]["make_target_rnd_network"] = lambda obs_shape: ptu.build_mlp(
        input_size=obs_shape[0],
        output_size=rnd_dim,
        n_layers=rnd_network_num_layers,
        size=rnd_network_hidden_size,
    )
    config["agent_kwargs"][
        "make_rnd_network_optimizer"
    ] = lambda params: torch.optim.Adam(params, lr=rnd_network_learning_rate)

    return config
