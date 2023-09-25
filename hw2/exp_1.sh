#!/bin/bash

# Experiment 1
declare -a args_strings_1=(
    "--env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole"
    "--env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg"
    "--env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na"
    "--env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na"
    "--env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb"
    "--env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg"
    "--env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na"
    "--env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na"
)

for args in "${args_strings_1[@]}"
do
    python cs285/scripts/run_hw2.py $args &
done

# Wait for all background processes to finish
wait
