from gym.envs.registration import register

register(
    id='ant-cs285-v0',
    entry_point='cs285.envs.ant:AntEnv',
    max_episode_steps=1000,
)
from cs285.envs.ant.ant import AntEnv
