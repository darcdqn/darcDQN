
from darcDQN.agents.random_agent import RandomAgent as random_agent
from darcDQN.agents.registration import registry, register, make, spec

register(
        id='RandomAgent-v0',
        entry_point='darcDQN.agents.random_agent:RandomAgent',
)

register(
        id='DQNAgent-v0',
        entry_point='darcDQN.agents.dqn_agent:DQNAgent',
)

