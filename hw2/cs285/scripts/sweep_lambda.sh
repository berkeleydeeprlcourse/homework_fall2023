#!/bin/bash

# Lambda values to iterate over
lambda_values=(0 0.95 0.98 0.99 1)

# Function to run the command for a specific lambda value
run_command() {
    lambda=$1
    python cs285/scripts/run_hw2.py \
        --env_name LunarLander-v2 --ep_len 1000 \
        --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
        --use_reward_to_go --use_baseline --gae_lambda $lambda \
        --exp_name lunar_lander_lambda$lambda
}

# Export the function to run in parallel
export -f run_command

# Run the command in parallel for each lambda value
parallel run_command ::: "${lambda_values[@]}"
