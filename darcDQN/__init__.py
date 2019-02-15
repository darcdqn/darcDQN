from gym.envs.registration import register

register(
        id='Berry-v0',
        entry_point='darcDQN.envs.berry:BerryEnv',
        max_episode_steps=200,
        reward_threshold=190.0,
)
