from gym.envs.registration import register

def register_envs():
    register(
        id='cheetah-cs285-v0',
        entry_point='cs285.envs.cheetah:HalfCheetahEnv',
        max_episode_steps=1000,
    )
    register(
        id='obstacles-cs285-v0',
        entry_point='cs285.envs.obstacles:Obstacles',
        max_episode_steps=500,
    )
    register(
        id='reacher-cs285-v0',
        entry_point='cs285.envs.reacher:Reacher7DOFEnv',
        max_episode_steps=500,
    )
