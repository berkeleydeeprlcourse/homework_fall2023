#!/bin/bash

# Experiment 1
declare -a args_strings_2=(
    "--env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah"
    "--env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline"
)

for args in "${args_strings_2[@]}"
do
    python cs285/scripts/run_hw2.py $args &
done

# Wait for all background processes to finish
wait
