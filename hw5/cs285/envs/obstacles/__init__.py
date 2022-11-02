from gym.envs.registration import register

register(
    id='obstacles-cs285-v0',
    entry_point='cs285.envs.obstacles:Obstacles',
    max_episode_steps=500,
)
from cs285.envs.obstacles.obstacles_env import Obstacles
