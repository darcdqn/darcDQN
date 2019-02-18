import time

from darcDQN.agents.dqn_agent import DQNAgent

import gym
from gym import wrappers, logger


def run_epoch(env, agent, render):
    ob = env.reset()
    tot_reward = 0
    while True:
        action = agent.get_action(ob) #act(ob, reward, done)
        n_ob, reward, done, _ = env.step(action)
        agent.observe(ob, n_ob, action, reward, done)
        ob = n_ob
        if render:
            env.render()
        tot_reward += reward
        if done:
            break
    return tot_reward

if __name__ == '__main__':
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make('CartPole-v1')

    env.seed(0)
    print('obs_space =', env.observation_space.shape[0],
          ', action_space.n =', env.action_space.n)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    version = 0.3
    path = './darcDQN/agents/saved_agents/'
    load_path = path + 'cartpole_{:.2f}/agent.ckpt'.format(version)
    save_path = path + 'cartpole_{:.2f}/agent.ckpt'.format(version+0.1)

    agent.load_agent(load_path)

    episode_count = 10000
    trials = 100

    reward = 0
    max_reward = 0
    done = False
    start = time.perf_counter()
    for i in range(1, episode_count):
        #reward += run_epoch(env, agent, i % trials == 0)
        reward += run_epoch(env, agent, False)
        if i % trials == 0:
            print('EPOCH {} over after {} sec, avg reward={:.1f}'.format(
                i, int(time.perf_counter() - start), reward/trials))
            start = time.perf_counter()
            if reward > max_reward:
                print('Average reward over {} epochs increased from {} to {}!'\
                      .format(trials, max_reward/trials, reward/trials))
                agent.save_agent(save_path)
                max_reward = reward
            reward = 0

    agent.save_agent(save_path)
    for i in range(episode_count):
        tot_reward = run_epoch(env, agent, True)
        print('EPOCH {} OVER, reward='.format(i), tot_reward)

    # Close the env and write monitor result info to disk
env.close()

