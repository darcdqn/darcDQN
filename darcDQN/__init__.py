from gym.envs.registration import register

register(
        id='Berry-v0',
        entry_point='darcDQN.envs.berry:BerryEnv',
        trials=10,
        reward_threshold=190.0,
        max_episode_steps=200,
)
