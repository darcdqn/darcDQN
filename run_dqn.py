import time

from darcDQN.agents.dqn_agent import DQNAgent

import gym
from gym import wrappers, logger


def run_epoch(env, agent, render):
    ob = env.reset()
    tot_reward = 0
    done = False
    while not done:
        action = agent.get_action(ob) #act(ob, reward, done)
        n_ob, reward, done, _ = env.step(action)
        agent.observe(ob, n_ob, action, reward, done)
        ob = n_ob
        if render:
            env.render()
        tot_reward += reward
    return tot_reward

if __name__ == '__main__':
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    env.seed(0)
    print('obs_space =', env.observation_space.shape[0],
          ', action_space.n =', env.action_space.n)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    version = 0.19
    path = './darcDQN/agents/saved_agents/' + env_name
    load_path = path + '_{:.2f}/agent.ckpt'.format(version)
    save_path = path + '_{:.2f}'.format(version+0.01)

    agent.load_agent(load_path)

    with open(save_path + '_conf.txt', 'w') as fp:
        fp.write('Activation    = ' + agent.activation + '\n')
        fp.write('Batch size    = {}\n'.format(agent.batch_size))
        fp.write('Copy-rate     = {}\n'.format(agent.copy_steps))
        fp.write('Learning rate = {}\n'.format(agent.learning_rate))
        fp.write('Hidden layers = {}\n'.format(agent.n_hidden))
        #fp.write('Hidden layers = {}\n'.format(agent.hidden_layers))
        fp.write('Discount rate = {}\n'.format(agent.discount_rate))
        fp.write('Optimizer     = {}\n'.format(agent.optimizer_name))
        fp.write('Epsilon between {} and {}\n'.format(
            agent.eps_max, agent.eps_min))
        fp.write('Eps decay steps     = {}\n'.format(agent.eps_decay_steps))
        fp.write('exp replay mem size = {}\n'.format(agent.replay_memory_size))
        fp.write('Training interval   = {}\n'.format(agent.training_interval))
        #fp.write(' = ' + agent + '\n')

    save_path = save_path + '/agent.ckpt'


    episode_count = 5000
    trials = 100

    reward = run_epoch(env, agent, False)
    max_reward = 100*reward
    start = time.perf_counter()
    for i in range(1, episode_count):
        #reward += run_epoch(env, agent, i % trials == 0)
        reward += run_epoch(env, agent, False)
        if i % trials == 0:
            t = int(time.perf_counter() - start)
            print('{} sec, {} runs, steps={}, eps={:.2f}, reward={:.1f}'.format(
                t, i, agent.step, agent.get_eps(), reward/trials))
            start = time.perf_counter()
            if reward > max_reward:
                print('Average reward over {} epochs increased from {} to {}!'\
                      .format(trials, max_reward/trials, reward/trials))
                agent.save_agent(save_path)
                max_reward = reward
            reward = 0

    for i in range(episode_count):
        reward = run_epoch(env, agent, True)
        print('EPOCH {} OVER, reward='.format(i), reward)

    # Close the env and write monitor result info to disk
env.close()

