import gym

from .pointmass import Pointmass

gym.register(
    id="PointmassEasy-v0",
    entry_point=Pointmass,
    kwargs={"difficulty": 0},
)

gym.register(
    id="PointmassMedium-v0",
    entry_point=Pointmass,
    kwargs={"difficulty": 1},
)

gym.register(
    id="PointmassHard-v0",
    entry_point=Pointmass,
    kwargs={"difficulty": 2},
)
