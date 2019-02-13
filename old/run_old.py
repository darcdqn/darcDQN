
import sys

from collections import defaultdict
from importlib import import_module

from .envs.registration import registry
from .error import UnregisteredAgentModule
from .utils.cli_util.py import common_arg_parser  # , parse_unknown_args


_game_envs = defaultdict(set)
for env in registry.all():
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)


def get_env_type(env_id):
    # Re-parse the environment registry, since there could be new envs since
    # last time.
    for env in registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, (
               'env_id {} is not recognized in env types'
               .format(env_id, _game_envs.keys()))

    return env_type, env_id


def get_agent_module(agent, submodule=None):
    submodule = submodule or agent
    try:
        agent_module = import_module('.'.join(['agents', agent, submodule]))
    except ImportError:
        e = ('Module {} is not recognized for agent {}'
             .format(agent_module, agent))
        raise UnregisteredAgentModule(e)
    return agent_module


def get_learn_function(agent):
    return get_agent_module(agent).learn


def get_learn_function_defaults(agent, env_type):
    try:
        agent_defaults = get_agent_module(agent, 'defaults')
        kwargs = getattr(agent_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def train(args):
    env_type, env_id = get_env_type(args.env)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.agent)
    agent_kwargs = get_learn_function_defaults(args.agent, env_type)

    env = build_env(args)


def main(args):
    arg_parser = common_arg_parser()
    args, _ = arg_parser.parse_known_args(args)

    model, env = train(args)


if __name__ == "__main__":
    main(sys.argv)
