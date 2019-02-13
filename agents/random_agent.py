from random import randrange

from .abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):

    """
    Description:
        A dummy implementation of an agent, which does not regard the
        current state, and simply takes random actions all the time.
    """

    def __init__(self, n_inputs, n_outputs):
        super().__init__(n_inputs, n_outputs)

    def get_action(self, state):
        return randrange(self.n_outputs)


