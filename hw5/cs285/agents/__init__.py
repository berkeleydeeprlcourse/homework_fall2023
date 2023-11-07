from .awac_agent import AWACAgent
from .cql_agent import CQLAgent
from .dqn_agent import DQNAgent
from .random_agent import RandomAgent
from .iql_agent import IQLAgent
from .rnd_agent import RNDAgent

agents = {
    "random": RandomAgent,
    "dqn": DQNAgent,
    "cql": CQLAgent,
    "awac": AWACAgent,
    "rnd": RNDAgent,
    "iql": IQLAgent,
}
