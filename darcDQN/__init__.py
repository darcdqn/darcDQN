from gym.envs.registration import register

register(
        id='berry-v0',
        entry_point='envs.berry:BerryEnv',
        max_episode_steps=200,
        reward_threshold=190.0,
)
