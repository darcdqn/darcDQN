
import gym
import numpy as np


from gym import spaces
from gym.utils import seeding
from .berry_helper import Berries


class BerryEnv(gym.Env):

    """
    Description:
        An agent is presented with a berry and the option to eat or not
        to eat the berry. The goal is to differentiate between berries
        that should be eaten and berries that should not be eaten.

    Observation:
        Type: Box(3)
        Num     Observation     Min     Max
        0       Sweet           0.0     1.0
        1       Sour            0.0     1.0
        2       Bitter          0.0     1.0

    Actions:
        Type: Discrete(2)
        Num     Action
        0       Do not eat berry
        1       Eat berry

    Reward:
        Reward is 1 if the agent makes a correct decision to eat or not
        to eat a berry and -1 if it makes an incorrect decision.

    Starting State:
        A random berry with some taste and information if it should be
        eaten or not.

    Episode Termination:
        Episode length is longer than 200
        Solved Requirements
        Consider solved when the average reward is 190.0 over 10
        consecutive trials.
    """

    def __init__(self):
        low = 0.0
        high = 1.0
        shape = (3,)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=low, high=high, shape=shape)

        self.seed()
        self.state = None

        self.berries = Berries()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), \
                "{0!r} ({0!s}) invalid".format(action, type(action))
        state = self.state
        berry, do_eat = state
        self.state = self._random_berry()

        reward = self._get_reward(action, do_eat)
        done = False

        return np.array(self.state), reward, done, {}

    def _random_berry(self):
        n_berries = self.berries.get_n_berries()
        index = self.np_random.randint(n_berries)
        berry, do_eat = self.berries.get_berry(index)
        return berry, do_eat

    def _get_reward(self, action, do_eat):
        if action is do_eat:
            reward = 1.0
        else:
            reward = -1.0
        return reward

    def reset(self):
        berry, do_eat = self._random_berry()
        self.state = (berry, do_eat)
        return np.array(self.state)

    #  TODO: What to do when closing? #
    def close(self):
        return

    #  TODO: Should we even render something in this env? #
    def render(self, mode='human'):
        return
        super(BerryEnv, self).render(mode=mode)  # will raise an exception
