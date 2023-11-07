from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter
import os
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import numpy as np

def extract_tensorboard_scalars(log_file, scalar_keys):
    # Initialize an EventAccumulator with the path to the log directory
    event_acc = EventAccumulator(log_file)
    event_acc.Reload()  # Load the events from disk

    if isinstance(scalar_keys, str):
        scalar_keys = [scalar_keys]

    # Extract the scalar summaries
    scalars = {}
    for tag in scalar_keys:
        scalars_for_tag = event_acc.Scalars(tag)
        scalars[tag] = {
            'step': [s.step for s in scalars_for_tag],
            'wall_time': [s.wall_time for s in scalars_for_tag],
            'value': [s.value for s in scalars_for_tag],
        }

    return scalars

def compute_mean_std(scalars: List[Dict[str, Any]],
                     data_key: str,
                     ninterp=100):
    min_step = min([s for slog in scalars for s in slog[data_key]['step']])
    max_step = max([s for slog in scalars for s in slog[data_key]['step']])
    steps = np.linspace(min_step, max_step, ninterp)
    scalars_interp = np.stack([
        np.interp(steps, slog[data_key]['step'], slog[data_key]['value'], left=float('nan'), right=float('nan'))
        for slog in scalars
    ], axis=1)

    mean = np.mean(scalars_interp, axis=1)
    std = np.std(scalars_interp, axis=1)

    return steps, mean, std


def plot_mean_std(ax: plt.Axes,
                  steps: np.ndarray,
                  mean: np.ndarray,
                  std: np.ndarray,
                  name: str,
                  color: str):
    ax.fill_between(steps, mean-std, mean+std, color=color, alpha=0.3)
    ax.plot(steps, mean, color=color, label=name)

def plot_scalars(ax: plt.Axes,
                 scalars: Dict[str, Any],
                 data_key: str,
                 name: str,
                 color: str):
    ax.plot(scalars[data_key]['step'], scalars[data_key]['value'], color=color, label=name)

if __name__ == '__main__':
    import argparse

    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_log_files", "-i", nargs='+', required=True)
    parser.add_argument("--human_readable_names", "-n", nargs='+', default=None, required=False)
    parser.add_argument("--colors", "-c", nargs='+', default=None, required=False)
    parser.add_argument("--data_key", "-d", type=str, required=True)
    parser.add_argument("--plot_mean_std", "-std", action="store_true")
    args = parser.parse_args()

    has_names = True

    if args.plot_mean_std:
        if args.colors is None:
            args.colors = [None]

        if args.human_readable_names is None:
            has_names = False
            args.human_readable_names = [None]

        assert len(args.human_readable_names) == 1
        assert len(args.colors) == 1

        all_scalars = [extract_tensorboard_scalars(log, args.data_key) for log in args.input_log_files]
        xs, mean, std = compute_mean_std(all_scalars, args.data_key)
        plot_mean_std(plt.gca(), xs, mean, std, args.human_readable_names[0], args.colors[0])
    else:
        if args.colors is None:
            args.colors = [None] * len(args.input_log_files)

        if args.human_readable_names is None:
            has_names = False
            args.human_readable_names = [None] * len(args.input_log_files)

        assert len(args.human_readable_names) == len(args.input_log_files)
        assert len(args.colors) == len(args.input_log_files)

        for log, name, color in zip(args.input_log_files, args.human_readable_names, args.colors):
            scalars = extract_tensorboard_scalars(log, args.data_key)
            if not args.plot_mean_std:
                plot_scalars(plt.gca(), scalars, args.data_key, name, color)

    if has_names:
        plt.legend()

    plt.show()
