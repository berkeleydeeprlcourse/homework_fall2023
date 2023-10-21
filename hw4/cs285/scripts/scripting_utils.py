import yaml
import os
import time

import cs285.env_configs
from cs285.infrastructure.logger import Logger


def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    base_config_name = config_kwargs.pop("base_config")
    return cs285.env_configs.configs[base_config_name](**config_kwargs)


def make_logger(config: dict) -> Logger:
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = config["log_name"] + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    return Logger(logdir)
