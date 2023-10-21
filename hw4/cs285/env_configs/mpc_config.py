import torch.nn as nn
from cs285.infrastructure import pytorch_util as ptu
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
import gym
import torch
from typing import Optional


def mpc_config(
    env_name: str,
    exp_name: str,
    hidden_size: int = 128,
    num_layers: int = 3,
    learning_rate: float = 1e-3,
    ensemble_size: int = 3,
    mpc_horizon: int = 10,
    mpc_strategy: str = "random",
    mpc_num_action_sequences: int = 1000,
    cem_num_iters: Optional[int] = None,
    cem_num_elites: Optional[int] = None,
    cem_alpha: Optional[float] = None,
    initial_batch_size: int = 20000,  # number of transitions to collect with random policy at the start
    batch_size: int = 8000,  # number of transitions to collect per per iteration thereafter
    train_batch_size: int = 512,  # number of transitions to train each dynamics model per iteration
    num_iters: int = 20,
    replay_buffer_capacity: int = 1000000,
    num_agent_train_steps_per_iter: int = 20,
    num_eval_trajectories: int = 10,
):
    # hardcoded for this assignment
    if env_name == "reacher-cs285-v0":
        ep_len = 200
    if env_name == "cheetah-cs285-v0":
        ep_len = 500
    if env_name == "obstacles-cs285-v0":
        ep_len = 100

    def make_dynamics_model(ob_dim: int, ac_dim: int) -> nn.Module:
        return ptu.build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=ob_dim,
            n_layers=num_layers,
            size=hidden_size,
        )

    def make_optimizer(params: nn.ParameterList):
        return torch.optim.Adam(params, lr=learning_rate)

    def make_env(render: bool = False):
        return RecordEpisodeStatistics(
            gym.make(env_name, render_mode="single_rgb_array" if render else None),
        )

    log_string = f"{env_name}_{exp_name}_l{num_layers}_h{hidden_size}_mpc{mpc_strategy}_horizon{mpc_horizon}_actionseq{mpc_num_action_sequences}"
    if mpc_strategy == "cem":
        log_string += f"_cem_iters{cem_num_iters}"

    return {
        "agent_kwargs": {
            "make_dynamics_model": make_dynamics_model,
            "make_optimizer": make_optimizer,
            "ensemble_size": ensemble_size,
            "mpc_horizon": mpc_horizon,
            "mpc_strategy": mpc_strategy,
            "mpc_num_action_sequences": mpc_num_action_sequences,
            "cem_num_iters": cem_num_iters,
            "cem_num_elites": cem_num_elites,
            "cem_alpha": cem_alpha,
        },
        "make_env": make_env,
        "replay_buffer_capacity": replay_buffer_capacity,
        "log_name": log_string,
        "num_iters": num_iters,
        "ep_len": ep_len,
        "batch_size": batch_size,
        "initial_batch_size": initial_batch_size,
        "train_batch_size": train_batch_size,
        "num_agent_train_steps_per_iter": num_agent_train_steps_per_iter,
        "num_eval_trajectories": num_eval_trajectories,
    }
