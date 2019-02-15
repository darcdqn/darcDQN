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
        # Note there's no env.render() here. But the environment still can
        # open window and render if asked by env.monitor: it calls
        # env.render('rgb_array') to record video. Video is not recorded
        # every episode, see capped_cubic_video_schedule for details.
    return tot_reward

if __name__ == '__main__':
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make('CartPole-v1')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    #outdir = 'tmp/random-agent-results'
    #env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    print("obs_space =", env.observation_space.shape[0],
          ", action_space.n =", env.action_space.n)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)


    episode_count = 10000
    reward = 0
    done = False

    for i in range(episode_count):
        tot_reward = run_epoch(env, agent, False)
        if i % 100 == 0:
            print("EPOCH {} OVER, reward=".format(i), tot_reward)

    for i in range(episode_count):
        tot_reward = run_epoch(env, agent, True)
        print("EPOCH {} OVER, reward=".format(i), tot_reward)
    
    # Close the env and write monitor result info to disk
env.close()

