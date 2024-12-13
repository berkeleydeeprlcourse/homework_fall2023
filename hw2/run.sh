python ./cs285/scripts/run_hw2.py \
--env_name Humanoid-v5 --ep_len 1000 \
--discount 0.99 -n 200 -l 3 -s 256 -b 50000 -lr 0.001 \
--baseline_gradient_steps 50 \
-na --use_reward_to_go --use_baseline --gae_lambda 0.97 \
--exp_name humanoid --video_log_freq 5

# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v5 \
# -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
# --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline -na --video_log_freq 20