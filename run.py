#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

from darcDQN.utils.cmd_util import common_arg_parser
from darcDQN import agents
from darcDQN import utils
from gym import logger, envs
from gym.envs.registration import registry


def run_episode(env, agent, mode, num_timesteps):
    obs = env.reset()
    episode_reward = 0.0
    #  TODO: figure the render thing out #
    if mode is not None:
        env.render(mode)
    for timestep in range(num_timesteps):
        done = False

        action = agent.get_action(obs)

        new_obs, reward, done, info_dict = env.step(action)
        episode_reward += reward

        agent.observe(obs, new_obs, action, reward, done)
        obs = new_obs

        if mode is not None:
            env.render(mode)

        if done:
            break
    return episode_reward, timestep


def train(args):
    env = envs.make(args.env)

    load_path = args.load_path
    save_path = args.save_path

    n_inputs = 0
    for i in range(len(env.observation_space.shape)):
        n_inputs += env.observation_space.shape[i]

    n_outputs = env.action_space.n

    kwargs = {'n_inputs': n_inputs,
              'n_outputs': n_outputs}

    agent = agents.make(args.agent, **kwargs)

    if load_path is not None:
        agent.load_agent(load_path)

    logger.info('Training agent {} in env {}'.format(args.agent, args.env))

    if args.mode.lower() == 'none':
        mode = None
    else:
        mode = args.mode

    spec = registry.env_specs[args.env]
    trials = spec.trials
    reward_threshold = spec.reward_threshold
    num_timesteps = args.num_timesteps or spec.max_episode_steps

    episode = 1
    num_episodes = int(args.num_episodes)

    trial_rewards = np.zeros(trials)
    max_average_reward = 0.0
    total_timesteps = 0
    is_solved = False

    while not is_solved and episode != num_episodes:
        episode_index = (episode - 1) % trials
        trial_rewards[episode_index] = 0.0

        episode_reward, episode_steps = run_episode(env,
                                                    agent,
                                                    mode,
                                                    num_timesteps)

        trial_rewards[episode_index] += episode_reward
        total_timesteps += episode_steps

        trial_average_reward = np.mean(trial_rewards)
        if episode > trials + 1 and trial_average_reward > reward_threshold:
            is_solved = True

        if episode % trials == 0:
            if trial_average_reward > max_average_reward:
                logger.info('Maximum average reward: {:.1f} -> {:.1f}'
                            .format(max_average_reward, trial_average_reward))
                max_average_reward = trial_average_reward
                if save_path is not None:
                    agent.save_agent(save_path)
                    logger.info('Agent saved')

            logger.info(('Episode {}'
                         + ' : {} timesteps'
                         + ' : {}-trial average reward {:.1f}')
                        .format(episode,
                                total_timesteps,
                                trials,
                                trial_average_reward))

        episode += 1
        if is_solved:
            if save_path is not None:
                agent.save_agent(save_path)
            logger.info(('Agent finished with average score {:.1f}'
                        .format(trial_average_reward)))
            break
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
    trained_agent, env = train(args)

    """ Play """
    if args.play:
        play(trained_agent, env)


if __name__ == "__main__":
    main(sys.argv)
