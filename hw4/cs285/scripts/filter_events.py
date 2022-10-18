# -*- coding: utf-8 -*-
"""
Usage:

Run the command
```
python filter_events.py --events SOME_DIRECTORY
```

and it will generate a directory named `SOME_DIRECTORY_filtered` with the video 
events removed.
"""
from __future__ import print_function
import os
import sys
import argparse
import tqdm

# Adapted from
# https://gist.github.com/serycjon/c9ad58ecc3176d87c49b69b598f4d6c6

import tensorflow as tf


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--event', help='event file', required=True)

    return parser.parse_args()


def main(args):
    out_path = os.path.dirname(args.event) + '_filtered'
    writer = tf.summary.FileWriter(out_path)

    total = None
    for event in tqdm.tqdm(tf.train.summary_iterator(args.event), total=total):
        event_type = event.WhichOneof('what')
        if event_type != 'summary':
            writer.add_event(event)
        else:
            wall_time = event.wall_time
            step = event.step
            filtered_values = [value for value in event.summary.value if
                               'rollouts' not in value.tag]
            summary = tf.Summary(value=filtered_values)

            filtered_event = tf.summary.Event(summary=summary,
                                              wall_time=wall_time,
                                              step=step)
            writer.add_event(filtered_event)
    writer.close()
    return 0


if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(main(args))
