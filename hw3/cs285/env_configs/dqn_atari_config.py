from typing import Optional, Tuple

import gym
from gym.wrappers.frame_stack import FrameStack

import numpy as np
import torch
import torch.nn as nn

from cs285.env_configs.schedule import (
    LinearSchedule,
    PiecewiseSchedule,
    ConstantSchedule,
)
from cs285.infrastructure.atari_wrappers import wrap_deepmind
import cs285.infrastructure.pytorch_util as ptu


class PreprocessAtari(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim in [3, 4], f"Bad observation shape: {x.shape}"
        assert x.shape[-3:] == (4, 84, 84), f"Bad observation shape: {x.shape}"
        assert x.dtype == torch.uint8

        return x / 255.0


def atari_dqn_config(
    env_name: str,
    exp_name: Optional[str] = None,
    learning_rate: float = 1e-4,
    adam_eps: float = 1e-4,
    total_steps: int = 1000000,
    discount: float = 0.99,
    target_update_period: int = 2000,
    clip_grad_norm: Optional[float] = 10.0,
    use_double_q: bool = False,
    learning_starts: int = 20000,
    batch_size: int = 32,
    **kwargs,
):
    def make_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
        assert observation_shape == (
            4,
            84,
            84,
        ), f"Observation shape: {observation_shape}"

        return nn.Sequential(
            PreprocessAtari(),
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, num_actions),
        ).to(ptu.device)

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate, eps=adam_eps)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            PiecewiseSchedule(
                [
                    (0, 1),
                    (20000, 1),
                    (total_steps / 2, 5e-1),
                ],
                outside_value=5e-1,
            ).value,
        )

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (20000, 1),
            (total_steps / 2, 0.01),
        ],
        outside_value=0.01,
    )

    def make_env(render: bool = False):
        return wrap_deepmind(
            gym.make(env_name, render_mode="rgb_array" if render else None)
        )

    log_string = "{}_{}_d{}_tu{}_lr{}".format(
        exp_name or "dqn",
        env_name,
        discount,
        target_update_period,
        learning_rate,
    )

    if use_double_q:
        log_string += "_doubleq"

    if clip_grad_norm is not None:
        log_string += f"_clip{clip_grad_norm}"

    return {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "target_update_period": target_update_period,
            "clip_grad_norm": clip_grad_norm,
            "use_double_q": use_double_q,
        },
        "log_name": log_string,
        "exploration_schedule": exploration_schedule,
        "make_env": make_env,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        **kwargs,
    }
