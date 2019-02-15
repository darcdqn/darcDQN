#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from darcDQN.utils.cmd_util import common_arg_parser
from darcDQN import agents
from darcDQN import utils
from gym import error, logger, envs


def train(args):
    env = envs.make(args.env)

    n_inputs = 0
    for i in range(len(env.observation_space.shape)):
        n_inputs += env.observation_space.shape[i]

    n_outputs = env.action_space.n

    kwargs = {'n_inputs': n_inputs,
              'n_outputs': n_outputs}

    agent = agents.make(args.agent, **kwargs)

    logger.info('Training agent {} in env {}'.format(args.agent, args.env))

    mode = args.mode

    epoch = 0
    if args.num_epochs == 0:
        num_epochs = -1
    else:
        num_epochs = int(args.num_epochs)
    num_timesteps = int(args.num_timesteps)

    epoch_rewards = []
    while epoch != num_epochs:
        epoch_rewards.append(0)
        obs = env.reset()
        #  TODO: figure the render thing out #
        env.render(mode)
        logger.info('Epoch {} of {}'.format(epoch, num_epochs))
        for t in range(num_timesteps):
            done = False

            action = agent.get_action(obs)

            new_obs, reward, done, info_dict = env.step(action)
            epoch_rewards[-1] += reward

            agent.observe(obs, new_obs, action, reward, done)
            obs = new_obs

            env.render(mode)

            if done:
                break

        logger.info(('Epoch {} finished after {} timesteps with a total score '
                     + 'of {}').format(epoch, t, epoch_rewards[-1]))
        epoch += 1
    return agent, env


def play(agent, env):
    logger.info('Playing agent {} in env {}'.format(agent, env))

    obs = env.reset()
    while True:
        action = agent.get_action(obs)
        obs, reward, done, info_dict = env.step(action)
        env.render()

        if done:
            obs = env.reset()


def main(args):
    parser = common_arg_parser()
    args, unknown_args = parser.parse_known_args(args)

    """
    if unknown_args is not None:
        m = 'Unknown argument'
        if len(unknown_args) > 1:
            m += 's'
        m += ': {}'.format(', '.join(unknown_args))
        raise utils.error.UnknownArgument(m)
    """

    logger.set_level(logger.INFO)

    """ Train """
    if args.num_epochs is not None and args.num_timesteps is not None:
        trained_agent, env = train(args)
    else:
        logger.info('num_epochs or num_timesteps not provided, not training')

    """ Play """
    if args.play:
        play(trained_agent, env)


if __name__ == "__main__":
    main(sys.argv)
